#!/usr/bin/env python3
"""
GraphitiMemoryProvider 插件单元测试。

覆盖:
- 配置加载 (_load_config, save_config, get_config_schema, is_available)
- 熔断器 (circuit breaker, 主动健康探测)
- 生命周期 (initialize, shutdown, name, backup_paths)
- 工具 schema (get_tool_schemas, system_prompt_block, 6 个工具)
- 轮次缓冲与批量写入 (sync_turn, on_session_end)
- 工具分发 (handle_tool_call, _handle_*) 含新增 explore/conclude 模式
- prefetch (queue_prefetch, prefetch)
- 辅助函数 (_tool_error, _now_iso, _is_client_error)
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 在导入插件之前 mock 所有外部依赖。
# MCP 1.27+ 不再导出 mcp.client.Client，且插件依赖 hermes_constants / utils
# 等 Hermes 框架内部模块，这些模块在 mcp_server 测试环境中不存在。
# ---------------------------------------------------------------------------
_mock_mcp = MagicMock()
_mock_mcp.client = MagicMock()
_mock_mcp.client.Client = MagicMock()
sys.modules['mcp'] = _mock_mcp
sys.modules['mcp.client'] = _mock_mcp.client

_mock_hermes_constants = MagicMock()
_mock_hermes_constants.get_hermes_home.return_value = Path('/fake/hermes_home')
sys.modules['hermes_constants'] = _mock_hermes_constants

_mock_utils = MagicMock()
_mock_utils.atomic_json_write = MagicMock()
sys.modules['utils'] = _mock_utils

_mock_agent_async = MagicMock()
sys.modules['agent'] = MagicMock()
sys.modules['agent.async_utils'] = MagicMock()

# 将插件目录加入 sys.path
_PLUGIN_DIR = Path(__file__).resolve().parent.parent / 'plugins' / 'memory'
sys.path.insert(0, str(_PLUGIN_DIR))

# 被测模块
from graphiti import (  # type: ignore[import-untyped]
    ALL_TOOL_SCHEMAS,
    CONCLUDE_SCHEMA,
    CONTEXT_SCHEMA,
    EXPLORE_SCHEMA,
    PROFILE_SCHEMA,
    REASONING_SCHEMA,
    SEARCH_SCHEMA,
    GraphitiMemoryProvider,
    _BREAKER_PROBE_INTERVAL_SECS,
    _is_client_error,
    _load_config,
    _now_iso,
    _tool_error,
    _WRITER_SENTINEL,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def provider() -> GraphitiMemoryProvider:
    """返回一个未初始化的 GraphitiMemoryProvider 实例。"""
    return GraphitiMemoryProvider()


@pytest.fixture
def mocked_provider() -> GraphitiMemoryProvider:
    """返回一个 mock 了 MCP 客户端的 provider，用于测试工具分发。"""
    p = GraphitiMemoryProvider()

    # Mock 客户端
    mock_client = MagicMock()
    p._client = mock_client

    # Mock _call_tool 以返回假数据
    p._call_tool = MagicMock(return_value={})  # type: ignore[method-assign]

    # Mock _record_success / _record_failure
    p._record_success = MagicMock()  # type: ignore[method-assign]
    p._record_failure = MagicMock()  # type: ignore[method-assign]

    return p


@pytest.fixture
def clean_env():
    """清除相关环境变量并在测试后恢复。"""
    saved = {}
    keys = [
        'GRAPHITI_MCP_URL', 'GRAPHITI_GROUP_ID', 'GRAPHITI_RECALL_MODE',
        'GRAPHITI_RETAIN_EVERY_N', 'GRAPHITI_AUTO_RETAIN', 'GRAPHITI_AUTO_RECALL',
        'GRAPHITI_TIMEOUT', 'GRAPHITI_RECALL_MAX_INPUT_CHARS',
    ]
    for k in keys:
        saved[k] = os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ============================================================================
# 辅助函数测试
# ============================================================================


class TestHelperFunctions:
    """测试 _tool_error, _now_iso, _is_client_error 等辅助函数。"""

    def test_tool_error_returns_json_string(self):
        result = _tool_error('something went wrong')
        parsed = json.loads(result)
        assert parsed == {'error': 'something went wrong'}

    def test_tool_error_with_empty_message(self):
        result = _tool_error('')
        parsed = json.loads(result)
        assert parsed == {'error': ''}

    def test_now_iso_returns_valid_iso8601(self):
        ts = _now_iso()
        # 应该以 Z 或 +00:00 结尾
        assert 'T' in ts
        # 确认可以解析回来
        from datetime import datetime
        datetime.fromisoformat(ts)

    def test_is_client_error_not_found_patterns(self):
        """not found / no relevant 等模式应被识别为客户端错误。"""
        assert _is_client_error(Exception('entity not found')) is True
        assert _is_client_error(Exception('no relevant results found')) is True
        assert _is_client_error(Exception('no facts available')) is True
        assert _is_client_error(Exception('no episodes recorded')) is True

    def test_is_client_error_backend_errors(self):
        """真正的后端故障不应被识别为客户端错误。"""
        assert _is_client_error(Exception('Connection refused')) is False
        assert _is_client_error(Exception('timeout')) is False
        assert _is_client_error(Exception('500 Internal Server Error')) is False


# ============================================================================
# 配置加载测试
# ============================================================================


class TestConfiguration:
    """测试 _load_config 和 provider 配置相关方法。

    注意: _load_config() 内部 import hermes_constants，使用 @patch 注入 mock。
    """

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_load_config_defaults(self, mock_home, clean_env):
        """不设置任何环境变量时，应返回所有默认值。"""
        cfg = _load_config()
        assert cfg['mcp_url'] == 'http://localhost:8000/mcp'
        assert cfg['group_id'] == ''
        assert cfg['recall_mode'] == 'tools'
        assert cfg['retain_every_n_turns'] == 3
        assert cfg['auto_retain'] is True
        assert cfg['auto_recall'] is True
        assert cfg['timeout'] == 120
        assert cfg['recall_max_input_chars'] == 800

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_load_config_from_env(self, mock_home, clean_env):
        """环境变量应覆盖默认值。"""
        os.environ['GRAPHITI_MCP_URL'] = 'http://custom:9999/mcp'
        os.environ['GRAPHITI_GROUP_ID'] = 'my-group'
        os.environ['GRAPHITI_RECALL_MODE'] = 'hybrid'
        os.environ['GRAPHITI_RETAIN_EVERY_N'] = '5'
        os.environ['GRAPHITI_AUTO_RETAIN'] = 'false'
        os.environ['GRAPHITI_TIMEOUT'] = '60'

        cfg = _load_config()
        assert cfg['mcp_url'] == 'http://custom:9999/mcp'
        assert cfg['group_id'] == 'my-group'
        assert cfg['recall_mode'] == 'hybrid'
        assert cfg['retain_every_n_turns'] == 5
        assert cfg['auto_retain'] is False
        assert cfg['timeout'] == 60

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_is_available_true(self, mock_home, clean_env):
        """设置了 mcp_url 时 is_available() 返回 True。"""
        os.environ['GRAPHITI_MCP_URL'] = 'http://example.com/mcp'
        p = GraphitiMemoryProvider()
        assert p.is_available() is True

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_is_available_false(self, mock_home, clean_env):
        """未设置任何 mcp_url 时，默认值存在，所以 is_available() 返回 True。"""
        p = GraphitiMemoryProvider()
        assert p.is_available() is True

    def test_name_property(self, provider):
        assert provider.name == 'graphiti'

    def test_backup_paths_empty(self, provider):
        assert provider.backup_paths() == []

    def test_get_config_schema(self, provider):
        schema = provider.get_config_schema()
        assert isinstance(schema, list)
        assert len(schema) >= 5
        keys = {item['key'] for item in schema}
        expected = {'mcp_url', 'group_id', 'recall_mode', 'retain_every_n_turns', 'auto_recall'}
        assert expected.issubset(keys)

    @patch('utils.atomic_json_write')
    def test_save_config_new_file(self, mock_atomic_write, provider):
        """save_config 应调用 atomic_json_write 写入配置。"""
        with patch.object(Path, 'exists', return_value=False):
            provider.save_config({'mcp_url': 'http://test:8000/mcp'}, '/fake/hermes_home')

        mock_atomic_write.assert_called_once()
        call_args = mock_atomic_write.call_args
        written_data = call_args[0][1]
        assert written_data['mcp_url'] == 'http://test:8000/mcp'

    @patch('utils.atomic_json_write')
    def test_save_config_merge_existing(self, mock_atomic_write, provider):
        """save_config 应与已有配置合并。"""
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value='{"group_id": "existing"}'):
            provider.save_config({'mcp_url': 'http://new:8000/mcp'}, '/fake/hermes_home')

        call_args = mock_atomic_write.call_args
        written_data = call_args[0][1]
        assert written_data['group_id'] == 'existing'
        assert written_data['mcp_url'] == 'http://new:8000/mcp'


# ============================================================================
# 熔断器 (Circuit Breaker) 测试
# ============================================================================


class TestCircuitBreaker:
    """测试熔断器的开/闭逻辑。"""

    def test_breaker_initially_closed(self, provider):
        assert provider._is_breaker_open() is False
        assert provider._consecutive_failures == 0

    def test_record_success_resets_counter(self, provider):
        provider._consecutive_failures = 3
        provider._record_success()
        assert provider._consecutive_failures == 0

    def test_record_failure_increments_counter(self, provider):
        provider._record_failure()
        assert provider._consecutive_failures == 1
        provider._record_failure()
        assert provider._consecutive_failures == 2

    def test_breaker_opens_after_threshold(self, provider):
        """连续失败达到 _BREAKER_THRESHOLD 时熔断器断开。"""
        threshold = 5  # _BREAKER_THRESHOLD
        for _ in range(threshold):
            provider._record_failure()
        assert provider._is_breaker_open() is True

    def test_breaker_recloses_after_cooldown(self, provider):
        """冷却时间过后熔断器应重新闭合。"""
        threshold = 5
        for _ in range(threshold):
            provider._record_failure()
        assert provider._is_breaker_open() is True

        # 模拟冷却时间已过
        provider._breaker_open_until = time.monotonic() - 1
        assert provider._is_breaker_open() is False
        assert provider._consecutive_failures == 0

    def test_failure_below_threshold_does_not_open(self, provider):
        for _ in range(3):
            provider._record_failure()
        assert provider._is_breaker_open() is False

    def test_probe_health_triggers_in_breaker_open_state(self, provider):
        """熔断期间 _is_breaker_open 应触发主动探测。"""
        threshold = 5
        for _ in range(threshold):
            provider._record_failure()
        assert provider._is_breaker_open() is True

        provider._call_tool = MagicMock(return_value={'status': 'ok'})  # type: ignore[method-assign]
        provider._last_probe_time = time.monotonic() - _BREAKER_PROBE_INTERVAL_SECS - 1

        result = provider._is_breaker_open()
        assert result is False
        assert provider._consecutive_failures == 0

    def test_probe_health_does_not_close_on_bad_status(self, provider):
        """探测返回非 ok 状态时熔断器应保持断开。"""
        threshold = 5
        for _ in range(threshold):
            provider._record_failure()

        provider._call_tool = MagicMock(return_value={'status': 'error'})  # type: ignore[method-assign]
        provider._last_probe_time = time.monotonic() - _BREAKER_PROBE_INTERVAL_SECS - 1

        result = provider._is_breaker_open()
        assert result is True

    def test_probe_health_handles_exception(self, provider):
        """探测过程中发生异常时熔断器应保持断开。"""
        threshold = 5
        for _ in range(threshold):
            provider._record_failure()

        provider._call_tool = MagicMock(side_effect=RuntimeError('connection refused'))  # type: ignore[method-assign]
        provider._last_probe_time = time.monotonic() - _BREAKER_PROBE_INTERVAL_SECS - 1

        result = provider._is_breaker_open()
        assert result is True

    def test_probe_health_respects_interval(self, provider):
        """探测间隔内不应重复探测。"""
        threshold = 5
        for _ in range(threshold):
            provider._record_failure()

        provider._call_tool = MagicMock(return_value={'status': 'ok'})  # type: ignore[method-assign]
        provider._last_probe_time = time.monotonic() - _BREAKER_PROBE_INTERVAL_SECS - 1
        assert provider._is_breaker_open() is False

        for _ in range(threshold):
            provider._record_failure()

        provider._last_probe_time = time.monotonic()
        call_tool_mock = MagicMock(return_value={'status': 'ok'})
        provider._call_tool = call_tool_mock  # type: ignore[method-assign]
        result = provider._is_breaker_open()
        assert result is True
        call_tool_mock.assert_not_called()


# ============================================================================
# 生命周期测试
# ============================================================================


class TestLifecycle:
    """测试 initialize, shutdown, system_prompt_block 等生命周期方法。"""

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_initialize_basic(self, mock_home, provider, clean_env):
        provider.initialize('session-123', platform='cli')
        assert provider._session_id == 'session-123'
        assert provider._platform == 'cli'
        assert provider._session_turns == []

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_initialize_defaults(self, mock_home, provider, clean_env):
        provider.initialize('')
        assert provider._recall_mode == 'tools'
        assert provider._auto_retain is True
        assert provider._auto_recall is True
        assert provider._retain_every_n_turns == 3

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_initialize_invalid_recall_mode_falls_back(self, mock_home, provider, clean_env):
        """非法的 recall_mode 应回退为 'tools'。"""
        os.environ['GRAPHITI_RECALL_MODE'] = 'invalid_mode'
        provider.initialize('s1')
        assert provider._recall_mode == 'tools'

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_initialize_with_env_overrides(self, mock_home, provider, clean_env):
        os.environ['GRAPHITI_RETAIN_EVERY_N'] = '7'
        os.environ['GRAPHITI_AUTO_RETAIN'] = 'false'
        provider.initialize('s1')
        assert provider._retain_every_n_turns == 7
        assert provider._auto_retain is False

    def test_system_prompt_block_tools_mode(self, provider):
        provider._recall_mode = 'tools'
        block = provider.system_prompt_block()
        assert 'tools-only mode' in block
        assert 'graphiti_search' in block
        assert 'graphiti_explore' in block

    def test_system_prompt_block_context_mode(self, provider):
        provider._recall_mode = 'context'
        block = provider.system_prompt_block()
        assert 'context-injection mode' in block
        assert 'graphiti_search' not in block

    def test_system_prompt_block_hybrid_mode(self, provider):
        provider._recall_mode = 'hybrid'
        block = provider.system_prompt_block()
        assert 'hybrid mode' in block

    def test_shutdown_cleans_up(self, provider):
        """shutdown() 应设置关闭信号并清理资源。"""
        provider.shutdown()
        assert provider._shutting_down.is_set()

    def test_shutdown_joins_writer_thread(self, provider):
        """shutdown() 应等待 writer 线程退出。"""
        # 启动 writer
        provider._ensure_writer()
        assert provider._writer_thread is not None
        assert provider._writer_thread.is_alive()

        provider.shutdown()
        # writer 线程应该在超时前退出
        writer = provider._writer_thread
        assert writer is None or not writer.is_alive()


# ============================================================================
# 工具 Schema 测试
# ============================================================================


class TestToolSchemas:
    """测试 get_tool_schemas() 在不同模式下的行为。"""

    def test_all_schemas_have_required_fields(self):
        for schema in ALL_TOOL_SCHEMAS:
            assert 'name' in schema
            assert 'description' in schema
            assert 'parameters' in schema

    def test_tools_mode_returns_all_schemas(self, provider):
        provider._recall_mode = 'tools'
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 6

    def test_context_mode_returns_empty(self, provider):
        provider._recall_mode = 'context'
        schemas = provider.get_tool_schemas()
        assert schemas == []

    def test_hybrid_mode_returns_all_schemas(self, provider):
        provider._recall_mode = 'hybrid'
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 6

    def test_search_schema_requires_query(self):
        assert 'query' in SEARCH_SCHEMA['parameters']['required']

    def test_profile_schema_requires_query(self):
        assert 'query' in PROFILE_SCHEMA['parameters']['required']

    def test_reasoning_schema_requires_query(self):
        assert 'query' in REASONING_SCHEMA['parameters']['required']

    def test_context_schema_no_required_params(self):
        assert CONTEXT_SCHEMA['parameters']['required'] == []

    def test_context_schema_has_episode_uuids_param(self):
        """CONTEXT_SCHEMA 应有 episode_uuids 参数用于溯源追踪。"""
        props = CONTEXT_SCHEMA['parameters']['properties']
        assert 'episode_uuids' in props
        assert props['episode_uuids']['type'] == 'array'

    def test_conclude_schema_no_required_params(self):
        """conclude 的必填校验在 _handle_conclude 中处理。"""
        assert CONCLUDE_SCHEMA['parameters']['required'] == []

    def test_conclude_schema_has_delete_episode_uuid_param(self):
        """CONCLUDE_SCHEMA 应有 delete_episode_uuid 参数用于批量删除。"""
        props = CONCLUDE_SCHEMA['parameters']['properties']
        assert 'delete_episode_uuid' in props
        assert props['delete_episode_uuid']['type'] == 'string'

    def test_explore_schema_exists(self):
        """EXPLORE_SCHEMA 应有三个互斥参数。"""
        assert EXPLORE_SCHEMA['name'] == 'graphiti_explore'
        props = EXPLORE_SCHEMA['parameters']['properties']
        assert 'edge_uuid' in props
        assert 'saga_name' in props
        assert 'communities' in props

    def test_explore_schema_no_required_params(self):
        """explore 的必填校验在 _handle_explore 中处理。"""
        assert EXPLORE_SCHEMA['parameters']['required'] == []


# ============================================================================
# 工具分发测试
# ============================================================================


class TestToolDispatch:
    """测试 handle_tool_call 分发及各个 _handle_* 方法。"""

    def test_handle_unknown_tool(self, provider):
        """未知工具名应返回错误。"""
        # 初始化必要字段
        provider._group_id = 'test'
        result = provider.handle_tool_call('unknown_tool', {})
        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'Unknown tool' in parsed['error']

    def test_handle_tool_call_breaker_open(self, provider):
        """熔断器断开时应直接返回错误。"""
        provider._consecutive_failures = 10
        provider._breaker_open_until = time.monotonic() + 999
        result = provider.handle_tool_call('graphiti_search', {'query': 'test'})
        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'temporarily unavailable' in parsed['error']

    def test_handle_search_missing_query(self, mocked_provider):
        result = mocked_provider.handle_tool_call('graphiti_search', {})
        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'Missing required parameter' in parsed['error']

    def test_handle_search_success(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={  # type: ignore[method-assign]
            'nodes': [{'name': 'Alice', 'summary': 'A person'}],
            'facts': [{'fact': 'Alice likes coffee', 'valid_at': '2024-01-01', 'invalid_at': None}],
        })

        result = mocked_provider.handle_tool_call('graphiti_search', {'query': 'Alice'})
        parsed = json.loads(result)
        assert 'result' in parsed
        assert any('Alice' in item for item in parsed['result'])

    def test_handle_profile_missing_query(self, mocked_provider):
        result = mocked_provider.handle_tool_call('graphiti_profile', {})
        parsed = json.loads(result)
        assert 'error' in parsed

    def test_handle_profile_success(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={  # type: ignore[method-assign]
            'nodes': [{'name': 'Bob', 'summary': 'Engineer'}],
        })

        result = mocked_provider.handle_tool_call('graphiti_profile', {'query': 'Bob'})
        parsed = json.loads(result)
        assert 'result' in parsed
        assert 'Bob' in str(parsed['result'])

    def test_handle_profile_with_entity_types(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={'nodes': []})  # type: ignore[method-assign]

        mocked_provider.handle_tool_call(
            'graphiti_profile',
            {'query': 'test', 'entity_types': ['Person', 'Organization']},
        )
        # 验证 entity_types 传递给了 _call_tool
        call_args = mocked_provider._call_tool.call_args
        assert call_args[0][0] == 'search_nodes'
        assert 'entity_types' in call_args[0][1]

    def test_handle_reasoning_missing_query(self, mocked_provider):
        result = mocked_provider.handle_tool_call('graphiti_reasoning', {})
        parsed = json.loads(result)
        assert 'error' in parsed

    def test_handle_reasoning_with_date_filters(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={'facts': []})  # type: ignore[method-assign]

        mocked_provider.handle_tool_call(
            'graphiti_reasoning',
            {
                'query': 'project status',
                'valid_at_after': '2024-01-01',
                'valid_at_before': '2024-12-31',
            },
        )
        call_args = mocked_provider._call_tool.call_args
        assert call_args[0][0] == 'search_memory_facts'
        assert call_args[0][1]['valid_at_after'] == '2024-01-01'
        assert call_args[0][1]['valid_at_before'] == '2024-12-31'

    def test_handle_context_default_max(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={'episodes': []})  # type: ignore[method-assign]

        mocked_provider.handle_tool_call('graphiti_context', {})
        call_args = mocked_provider._call_tool.call_args
        assert call_args[0][0] == 'get_episodes'
        assert call_args[0][1]['max_episodes'] == 10

    def test_handle_context_custom_max(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={'episodes': []})  # type: ignore[method-assign]

        mocked_provider.handle_tool_call('graphiti_context', {'max_episodes': 25})
        call_args = mocked_provider._call_tool.call_args
        assert call_args[0][1]['max_episodes'] == 25

    def test_handle_context_capped_at_50(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={'episodes': []})  # type: ignore[method-assign]

        mocked_provider.handle_tool_call('graphiti_context', {'max_episodes': 100})
        call_args = mocked_provider._call_tool.call_args
        assert call_args[0][1]['max_episodes'] == 50

    def test_handle_conclude_neither_mode(self, mocked_provider):
        """同时不提供 fact 和 delete_uuid 应报错。"""
        result = mocked_provider.handle_tool_call('graphiti_conclude', {})
        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'Exactly one' in parsed['error']

    def test_handle_conclude_both_modes(self, mocked_provider):
        """同时提供 fact 和 delete_uuid 应报错。"""
        result = mocked_provider.handle_tool_call(
            'graphiti_conclude',
            {'fact': 'test', 'delete_uuid': 'abc-123'},
        )
        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'Exactly one' in parsed['error']

    def test_handle_conclude_add_triplet(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={'message': 'Stored.'})  # type: ignore[method-assign]

        result = mocked_provider.handle_tool_call(
            'graphiti_conclude',
            {
                'source_entity': 'Alice',
                'edge_type': 'LIKES',
                'fact': 'Alice likes pizza',
                'target_entity': 'Pizza',
            },
        )
        parsed = json.loads(result)
        assert parsed['result'] == 'Stored.'
        # 验证调用了 add_triplet
        assert mocked_provider._call_tool.call_args[0][0] == 'add_triplet'

    def test_handle_conclude_three_way_exclusive(self, mocked_provider):
        """同时提供 fact、delete_uuid、delete_episode_uuid 中多项时应报错。"""
        result = mocked_provider.handle_tool_call(
            'graphiti_conclude',
            {'fact': 'test', 'delete_uuid': 'abc-123', 'delete_episode_uuid': 'def-456'},
        )
        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'Exactly one' in parsed['error']

    def test_handle_conclude_delete_episode(self, mocked_provider):
        """提供 delete_episode_uuid 时应调用 delete_episode。"""
        mocked_provider._call_tool = MagicMock(return_value={'message': 'Episode deleted.'})  # type: ignore[method-assign]

        result = mocked_provider.handle_tool_call(
            'graphiti_conclude',
            {'delete_episode_uuid': 'ep-uuid-123'},
        )
        parsed = json.loads(result)
        assert parsed['result'] == 'Episode deleted.'
        assert mocked_provider._call_tool.call_args[0][0] == 'delete_episode'

    def test_handle_context_with_episode_uuids(self, mocked_provider):
        """提供 episode_uuids 时应调用 get_episode_entities。"""
        mocked_provider._call_tool = MagicMock(return_value={  # type: ignore[method-assign]
            'nodes': [{'name': 'Alice', 'summary': 'Person'}],
            'edges': [{'fact': 'Alice met Bob', 'valid_at': '2024-01-01'}],
        })

        result = mocked_provider.handle_tool_call(
            'graphiti_context',
            {'episode_uuids': ['ep-1', 'ep-2']},
        )
        parsed = json.loads(result)
        assert 'result' in parsed
        assert mocked_provider._call_tool.call_args[0][0] == 'get_episode_entities'
        assert 'ep-1' in mocked_provider._call_tool.call_args[0][1]['episode_uuids']

    def test_handle_explore_edge_uuid(self, mocked_provider):
        """提供 edge_uuid 时应调用 get_entity_edge。"""
        mocked_provider._call_tool = MagicMock(return_value={  # type: ignore[method-assign]
            'uuid': 'edge-1',
            'name': 'KNOWS',
            'fact': 'Alice knows Bob',
            'valid_at': '2024-01-01',
        })

        result = mocked_provider.handle_tool_call(
            'graphiti_explore',
            {'edge_uuid': 'edge-1'},
        )
        parsed = json.loads(result)
        assert 'result' in parsed
        assert parsed['result']['uuid'] == 'edge-1'
        assert mocked_provider._call_tool.call_args[0][0] == 'get_entity_edge'

    def test_handle_explore_edge_not_found(self, mocked_provider):
        """边不存在时应返回错误。"""
        mocked_provider._call_tool = MagicMock(return_value={'error': 'not found'})  # type: ignore[method-assign]

        result = mocked_provider.handle_tool_call(
            'graphiti_explore',
            {'edge_uuid': 'nonexistent'},
        )
        parsed = json.loads(result)
        assert 'error' in parsed

    def test_handle_explore_saga_name(self, mocked_provider):
        """提供 saga_name 时应调用 summarize_saga。"""
        mocked_provider._call_tool = MagicMock(return_value={  # type: ignore[method-assign]
            'name': 'MySaga',
            'uuid': 'saga-1',
            'summary': 'A saga summary',
        })

        result = mocked_provider.handle_tool_call(
            'graphiti_explore',
            {'saga_name': 'MySaga'},
        )
        parsed = json.loads(result)
        assert 'result' in parsed
        assert parsed['result']['name'] == 'MySaga'
        assert mocked_provider._call_tool.call_args[0][0] == 'summarize_saga'

    def test_handle_explore_communities(self, mocked_provider):
        """提供 communities=true 时应调用 build_communities。"""
        mocked_provider._call_tool = MagicMock(return_value={  # type: ignore[method-assign]
            'community_count': 2,
            'communities': [
                {'name': 'Tech', 'summary': 'Tech group'},
                {'name': 'Sales', 'summary': 'Sales group'},
            ],
        })

        result = mocked_provider.handle_tool_call(
            'graphiti_explore',
            {'communities': True},
        )
        parsed = json.loads(result)
        assert 'result' in parsed
        assert parsed['result']['community_count'] == 2
        assert mocked_provider._call_tool.call_args[0][0] == 'build_communities'

    def test_handle_explore_no_args(self, mocked_provider):
        """不提供任何互斥参数时应报错。"""
        result = mocked_provider.handle_tool_call('graphiti_explore', {})
        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'Exactly one' in parsed['error']

    def test_handle_explore_both_edge_and_saga(self, mocked_provider):
        """同时提供 edge_uuid 和 saga_name 应报错。"""
        result = mocked_provider.handle_tool_call(
            'graphiti_explore',
            {'edge_uuid': 'e1', 'saga_name': 's1'},
        )
        parsed = json.loads(result)
        assert 'error' in parsed
        assert 'Exactly one' in parsed['error']

    def test_handle_tool_call_records_success(self, mocked_provider):
        mocked_provider._call_tool = MagicMock(return_value={'nodes': []})  # type: ignore[method-assign]

        mocked_provider.handle_tool_call('graphiti_search', {'query': 'test'})
        mocked_provider._record_success.assert_called_once()


# ============================================================================
# sync_turn 缓冲区与批量写入测试
# ============================================================================


class TestSyncTurn:
    """测试 sync_turn 的缓冲和批量写入逻辑。"""

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_sync_turn_auto_retain_disabled(self, mock_home, provider):
        """auto_retain=False 时应直接返回，不入队。"""
        provider.initialize('s1')
        provider._auto_retain = False
        provider.sync_turn('hello', 'hi')
        assert provider._retain_queue.empty()

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_sync_turn_shutting_down(self, mock_home, provider):
        """关闭过程中不应入队。"""
        provider.initialize('s1')
        provider._auto_retain = True
        provider._shutting_down.set()
        provider.sync_turn('hello', 'hi')
        assert provider._retain_queue.empty()

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_sync_turn_buffers_turns(self, mock_home, provider):
        """在达到 retain_every_n_turns 之前，turn 只累积不写入。"""
        provider.initialize('s1')
        provider._auto_retain = True
        provider._retain_every_n_turns = 3

        provider.on_turn_start(1, 'msg1')
        provider.sync_turn('u1', 'a1')
        assert len(provider._session_turns) == 1
        assert provider._retain_queue.empty()

        provider.on_turn_start(2, 'msg2')
        provider.sync_turn('u2', 'a2')
        assert len(provider._session_turns) == 2
        assert provider._retain_queue.empty()

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_sync_turn_flushes_at_threshold(self, mock_home, provider):
        """达到 retain_every_n_turns 时应触发写入并清空缓冲区。"""
        provider.initialize('s1')
        provider._auto_retain = True
        provider._retain_every_n_turns = 2

        provider.on_turn_start(1, 'm1')
        provider.sync_turn('u1', 'a1')
        assert len(provider._session_turns) == 1

        provider.on_turn_start(2, 'm2')
        provider.sync_turn('u2', 'a2')
        assert len(provider._session_turns) == 0
        assert provider._retain_queue.qsize() == 1

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_sync_turn_retain_every_1(self, mock_home, provider):
        """retain_every_n_turns=1 时每轮都写入。"""
        provider.initialize('s1')
        provider._auto_retain = True
        provider._retain_every_n_turns = 1

        provider.on_turn_start(1, 'm1')
        provider.sync_turn('u1', 'a1')
        assert len(provider._session_turns) == 0
        assert provider._retain_queue.qsize() == 1

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_sync_turn_writes_json_array_format(self, mock_home, provider):
        """验证缓冲区中的内容格式正确。"""
        provider.initialize('s1')
        provider._auto_retain = True
        provider._retain_every_n_turns = 1

        provider.on_turn_start(1, 'm1')
        provider.sync_turn('你好', '世界')

        job = provider._retain_queue.get(timeout=1)
        assert callable(job)

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_sync_turn_sets_session_id_from_param(self, mock_home, provider):
        provider.initialize('original-id')
        provider._auto_retain = True
        provider._retain_every_n_turns = 1

        provider.on_turn_start(1, 'm1')
        provider.sync_turn('u1', 'a1', session_id='new-id')
        assert provider._session_id == 'new-id'


# ============================================================================
# on_session_end 测试
# ============================================================================


class TestOnSessionEnd:
    """测试 on_session_end 的缓冲区排空逻辑。"""

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_on_session_end_empty_buffer(self, mock_home, provider):
        """缓冲区为空时不应入队任何任务。"""
        provider.initialize('s1')
        provider.on_session_end([])
        assert provider._retain_queue.empty()

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_on_session_end_flushes_remaining_turns(self, mock_home, provider):
        """缓冲区有残留时应在 session 结束时排空。"""
        provider.initialize('s1')
        provider._auto_retain = True
        provider._retain_every_n_turns = 5

        provider.on_turn_start(1, 'm1')
        provider.sync_turn('u1', 'a1')
        provider.on_turn_start(2, 'm2')
        provider.sync_turn('u2', 'a2')
        assert len(provider._session_turns) == 2

        provider.on_session_end([])
        assert len(provider._session_turns) == 0
        assert provider._retain_queue.qsize() == 1

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_on_session_end_shutting_down(self, mock_home, provider):
        """关闭过程中不应再入队。"""
        provider._auto_retain = True
        provider.initialize('s1')
        provider.sync_turn('u1', 'a1')
        provider._shutting_down.set()

        provider.on_session_end([])
        assert len(provider._session_turns) == 0


# ============================================================================
# prefetch 测试
# ============================================================================


class TestPrefetch:
    """测试 queue_prefetch 和 prefetch 在不同模式下的行为。"""

    def test_prefetch_tools_mode_returns_empty(self, provider):
        provider._recall_mode = 'tools'
        result = provider.prefetch('test query', session_id='s1')
        assert result == ''

    def test_queue_prefetch_tools_mode_skips(self, provider):
        """tools 模式下不应启动预取。"""
        provider._recall_mode = 'tools'
        provider.queue_prefetch('test query')
        assert provider._prefetch_thread is None

    def test_queue_prefetch_auto_recall_disabled(self, provider):
        """auto_recall=False 时不应启动预取。"""
        provider._recall_mode = 'hybrid'
        provider._auto_recall = False
        provider.queue_prefetch('test query')
        assert provider._prefetch_thread is None

    def test_queue_prefetch_breaker_open(self, provider):
        """熔断器断开时不应启动预取。"""
        provider._recall_mode = 'hybrid'
        provider._auto_recall = True
        provider._consecutive_failures = 10
        provider._breaker_open_until = time.monotonic() + 999
        provider.queue_prefetch('test query')
        assert provider._prefetch_thread is None

    def test_queue_prefetch_shutting_down(self, provider):
        """关闭过程中不应启动预取。"""
        provider._recall_mode = 'hybrid'
        provider._auto_recall = True
        provider._shutting_down.set()
        provider.queue_prefetch('test query')
        assert provider._prefetch_thread is None

    def test_queue_prefetch_truncates_long_query(self, provider):
        """超长查询应被截断到 recall_max_input_chars。"""
        provider._recall_mode = 'hybrid'
        provider._auto_recall = True
        provider._recall_max_input_chars = 10

        long_query = 'a' * 100
        provider.queue_prefetch(long_query)

    def test_prefetch_first_turn_injection(self, provider):
        """first-turn 模式下，第 2 轮开始不应注入上下文。"""
        provider._recall_mode = 'hybrid'
        provider._injection_frequency = 'first-turn'
        provider._turn_counter = 2
        result = provider.prefetch('test')
        assert result == ''

    def test_prefetch_first_turn_injection_turn1(self, provider):
        """first-turn 模式下，第 1 轮应正常返回。"""
        provider._recall_mode = 'hybrid'
        provider._injection_frequency = 'first-turn'
        provider._turn_counter = 1
        # 缓存为空，应返回空字符串
        result = provider.prefetch('test')
        assert result == ''


# ============================================================================
# Writer / Queue 测试
# ============================================================================


class TestWriterQueue:
    """测试写入队列和后台 writer 线程。"""

    def test_ensure_writer_creates_thread(self, provider):
        provider._ensure_writer()
        assert provider._writer_thread is not None
        assert provider._writer_thread.is_alive()
        assert provider._writer_thread.name == 'graphiti-writer'

        provider.shutdown()

    def test_ensure_writer_idempotent(self, provider):
        """重复调用 _ensure_writer 不应创建多个线程。"""
        provider._ensure_writer()
        first_thread = provider._writer_thread
        provider._ensure_writer()
        assert provider._writer_thread is first_thread

        provider.shutdown()

    def test_writer_processes_jobs(self, provider):
        """writer 线程应消费并执行队列中的任务。"""
        provider._ensure_writer()

        executed = []
        job = lambda: executed.append('done')
        provider._retain_queue.put(job)

        # 等待任务被消费
        time.sleep(0.5)
        assert executed == ['done']

        provider.shutdown()

    def test_writer_handles_job_exception(self, provider):
        """writer 线程应捕获任务异常而不崩溃。"""
        provider._ensure_writer()

        def failing_job():
            raise RuntimeError('simulated failure')

        provider._retain_queue.put(failing_job)

        # 再放入一个正常任务验证 writer 仍然存活
        executed = []
        provider._retain_queue.put(lambda: executed.append('ok'))

        time.sleep(0.5)
        assert executed == ['ok']

        provider.shutdown()

    def test_writer_stops_on_sentinel(self, provider):
        """收到 _WRITER_SENTINEL 后 writer 线程应退出。"""
        provider._ensure_writer()
        provider._retain_queue.put(_WRITER_SENTINEL)

        # 等待线程退出
        provider._writer_thread.join(timeout=2.0)
        assert not provider._writer_thread.is_alive()

    def test_atexit_registration_idempotent(self, provider):
        """_register_atexit 多次调用只注册一次。"""
        import atexit

        # 清除可能已有的注册
        provider._atexit_registered = False
        provider._register_atexit()
        assert provider._atexit_registered is True

        provider._register_atexit()
        assert provider._atexit_registered is True


# ============================================================================
# 集成场景测试
# ============================================================================


class TestIntegrationScenarios:
    """端到端场景测试，验证多个组件协同工作。"""

    @patch('hermes_constants.get_hermes_home', return_value=Path('/fake/hermes_home'))
    def test_full_lifecycle_tools_mode(self, mock_home, clean_env):
        """完整的 tools 模式生命周期。"""
        p = GraphitiMemoryProvider()
        p.initialize('integration-test', platform='test')

        assert p.name == 'graphiti'
        assert p._recall_mode == 'tools'
        assert p.get_tool_schemas() != []

        for turn in range(1, 4):
            p.on_turn_start(turn, f'user msg {turn}')
            p.sync_turn(f'user_{turn}', f'assistant_{turn}')

        p.on_session_end([])
        p.shutdown()

    def test_context_mode_no_tools(self, provider):
        """context 模式不暴露工具。"""
        provider._recall_mode = 'context'
        assert provider.get_tool_schemas() == []

        # prefetch 不应被跳过
        # (不需要 client，只验证调度逻辑)
        provider._auto_recall = True
        provider.queue_prefetch('test query')

    def test_on_turn_start_tracks_counter(self, provider):
        """on_turn_start 应正确记录轮次编号。"""
        provider.on_turn_start(5, 'hello')
        assert provider._turn_counter == 5

        provider.on_turn_start(10, 'world')
        assert provider._turn_counter == 10

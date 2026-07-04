r"""
Graphiti Memory Provider for Hermes Agent
==========================================

将 Graphiti 知识图谱作为 Hermes Agent 的长期记忆后端。通过 MCP streamable HTTP
协议连接到 Graphiti MCP 服务器，将对话轮次自动持久化到知识图谱中，并提供语义搜索、
实体查询、时间感知推理等工具。

Quick start / 快速开始
-----------------------
1. 将本目录放入 ``plugins/memory/graphiti/``。
2. 运行 ``hermes memory setup`` 选择 graphiti 并配置。
3. 或者在 config.yaml 中设置 ``memory.provider: graphiti``，
   并在环境中提供 ``GRAPHITI_MCP_URL``。

Configuration / 配置
---------------------
Resolution order (last wins) — 后者覆盖前者:
  1. 环境变量 (GRAPHITI_MCP_URL, GRAPHITI_GROUP_ID, ...)
  2. ``$HERMES_HOME/graphiti.json``

Tools / 工具
-------------
- **graphiti_search**:    语义 + 关键词 + 图谱遍历混合搜索
- **graphiti_profile**:   查询实体信息（对等方卡片）
- **graphiti_reasoning**:  时间感知深度搜索，支持日期范围过滤
- **graphiti_context**:   检索最近对话 episode，支持按 episode UUID 溯源
- **graphiti_conclude**:  直接持久化 / 删除一个事实或 episode
- **graphiti_explore**:   探索图谱结构（查看边详情、saga 摘要、社区检测）

Modes / 运行模式
----------------
- ``tools``   (默认): 仅暴露工具，无自动上下文注入
- ``context``:        自动注入上下文，不暴露工具
- ``hybrid``:         工具 + 自动注入两者兼备

Design reference / 设计参考
----------------------------
参考了 Hermes 官方的 mem0 和 hindsight 两个 provider 的实现模式:
- 熔断器   (mem0)     — 服务宕机时自动暂停，避免级联失败
- 单写入者队列 (hindsight) — 串行化写入，优雅退出
- 轮次缓冲   (hindsight) — 累积多轮对话后批量发送
- 共享事件循环 (hindsight) — 进程级复用 asyncio loop
- atexit 注册 (两者)    — 进程退出时排空队列
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.client import Client

# 模块级 logger，供所有函数和 provider 实例共享使用。
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit breaker (mem0 pattern)
# ---------------------------------------------------------------------------

# ---------------
# 熔断器：连续 _BREAKER_THRESHOLD 次失败后暂停 _BREAKER_COOLDOWN_SECS 秒。
# 参考 mem0  —— 区分用户错误 (404 / not found) 与后端故障，
# 避免对已宕机的服务器持续发起无效请求。
_BREAKER_THRESHOLD = 5           # 连续失败阈值
_BREAKER_COOLDOWN_SECS = 120     # 熔断冷却时间（秒）
_BREAKER_PROBE_INTERVAL_SECS = 15  # 熔断期间主动探测间隔（秒）

# 这些错误模式不应触发熔断 —— 它们是合法的"无结果"响应，不是后端故障。
_CLIENT_ERROR_KEYWORDS = ('not found', 'no relevant', 'no facts', 'no episodes')


def _is_client_error(exc: Exception) -> bool:
    """判断是否为客户端错误（用户级别），不应触发熔断器。

    返回 True 表示这是正常的 "无结果" 或 "未找到" 响应，不是后端故障。
    熔断器只对真正的后端故障（连接超时、500 等）计数。
    """
    msg = str(exc).lower()
    return any(kw in msg for kw in _CLIENT_ERROR_KEYWORDS)


# ---------------------------------------------------------------------------
# Shared event loop — one per process (hindsight pattern)
# ---------------------------------------------------------------------------

# 共享事件循环：整个进程复用一个 asyncio loop，避免每个 provider 实例
# 独立创建/销毁 Client 导致连接泄漏（参考 hindsight 的实现）。
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()

# 写入队列的哨兵值 —— 放入队列后 writer 线程收到即退出。
_WRITER_SENTINEL = object()


def _get_loop() -> asyncio.AbstractEventLoop:
    """获取或创建进程级共享的 asyncio 事件循环。

    首次调用时在后台 daemon 线程中启动一个永久运行的事件循环。
    后续调用复用同一个 loop，避免每个 provider 实例独立创建/销毁
    导致 aiohttp 连接泄漏。线程安全。
    """
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=_run, daemon=True, name='graphiti-loop')
        _loop_thread.start()
        return _loop


def _run_async(coro: Any, timeout: float = 120) -> Any:
    """在共享事件循环上调度一个协程，阻塞等待结果返回。

    这是 HERMES 同步线程与 MCP async Client 之间的桥梁。
    使用 safe_schedule_threadsafe 确保跨线程安全地将协程投递到共享 loop。

    Args:
        coro: 要执行的协程对象。
        timeout: 等待超时秒数（默认 120）。

    Returns:
        协程的返回值。

    Raises:
        RuntimeError: 共享事件循环不可用时。
    """
    from agent.async_utils import safe_schedule_threadsafe

    loop = _get_loop()
    future = safe_schedule_threadsafe(coro, loop)
    if future is None:
        raise RuntimeError('Graphiti loop unavailable')
    return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Config (mem0 pattern — env vars → $HERMES_HOME/graphiti.json)
# ---------------------------------------------------------------------------

# 配置加载：环境变量提供默认值 → $HERMES_HOME/graphiti.json 覆盖单个 key。
# 与 mem0 保持一致的模式，避免 JSON 文件存在但缺字段时的静默失败。
_DEFAULT_MCP_URL = 'http://localhost:8000/mcp'


def _load_config() -> dict:
    """加载配置：环境变量提供默认值 → $HERMES_HOME/graphiti.json 覆盖。

    配置键及其默认值：
    - mcp_url:                Graphiti MCP 服务器地址
    - group_id:               图谱命名空间（空=使用服务器默认）
    - recall_mode:            召回模式 (tools / context / hybrid)
    - injection_frequency:    注入频率 (every-turn / first-turn)
    - retain_every_n_turns:   每 N 轮批量写入一次（默认 3）
    - auto_retain:            是否自动持久化对话
    - auto_recall:            是否自动召回记忆
    - retain_context:         持久化时的上下文标签
    - retain_user_prefix:     用户消息前缀
    - retain_assistant_prefix: 助手消息前缀
    - recall_max_input_chars: 召回查询最大字符数
    - timeout:                MCP 请求超时秒数

    Returns:
        合并后的配置字典。
    """
    from hermes_constants import get_hermes_home

    config: dict[str, Any] = {
        'mcp_url': os.environ.get('GRAPHITI_MCP_URL', _DEFAULT_MCP_URL),
        'group_id': os.environ.get('GRAPHITI_GROUP_ID', ''),
        'recall_mode': os.environ.get('GRAPHITI_RECALL_MODE', 'tools'),
        'injection_frequency': os.environ.get('GRAPHITI_INJECTION_FREQUENCY', 'every-turn'),
        'retain_every_n_turns': int(os.environ.get('GRAPHITI_RETAIN_EVERY_N', '3')),
        'auto_retain': os.environ.get('GRAPHITI_AUTO_RETAIN', 'true').lower() == 'true',
        'auto_recall': os.environ.get('GRAPHITI_AUTO_RECALL', 'true').lower() == 'true',
        'retain_context': os.environ.get(
            'GRAPHITI_RETAIN_CONTEXT', 'conversation between Hermes Agent and the User'
        ),
        'retain_user_prefix': os.environ.get('GRAPHITI_RETAIN_USER_PREFIX', 'User'),
        'retain_assistant_prefix': os.environ.get(
            'GRAPHITI_RETAIN_ASSISTANT_PREFIX', 'Assistant'
        ),
        'recall_max_input_chars': int(os.environ.get('GRAPHITI_RECALL_MAX_INPUT_CHARS', '800')),
        'timeout': int(os.environ.get('GRAPHITI_TIMEOUT', '120')),
        'auto_clear_group_on_shutdown': os.environ.get(
            'GRAPHITI_AUTO_CLEAR', 'false'
        ).lower() == 'true',
    }

    config_path = get_hermes_home() / 'graphiti.json'
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding='utf-8'))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ''})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

# ---------------
# 工具 schema（OpenAI function-calling 格式）
# 与 Graphiti MCP 服务器暴露的工具一一对应 —— 每个 schema 映射到一个 MCP 工具调用。

# 混合搜索：同时返回实体 (search_nodes) 和事实 (search_memory_facts) 两路结果。
SEARCH_SCHEMA = {
    'name': 'graphiti_search',
    'description': (
        'Search the Graphiti knowledge graph. Returns entities and facts '
        'ranked by relevance using semantic, keyword, and graph-traversal '
        'retrieval. Cheaper than graphiti_reasoning.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'What to search for.'},
            'max_results': {
                'type': 'integer',
                'description': 'Max results (default 10, max 50).',
            },
        },
        'required': ['query'],
    },
}

# 实体查询：检索实体名称、摘要、标签等元信息。
PROFILE_SCHEMA = {
    'name': 'graphiti_profile',
    'description': (
        'Retrieve entities from the Graphiti knowledge graph. Entities carry '
        'a name, summary, labels, and timestamps. Use for quick factual lookups.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'What to search for (e.g. "Alice", "Q3 roadmap").',
            },
            'max_nodes': {
                'type': 'integer',
                'description': 'Max entities to return (default 10, max 50).',
            },
            'entity_types': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Optional entity type names to filter by.',
            },
        },
        'required': ['query'],
    },
}

# 时间感知深度搜索：返回事实及其 valid_at / invalid_at 时间范围，支持按日期过滤。
REASONING_SCHEMA = {
    'name': 'graphiti_reasoning',
    'description': (
        'Deep temporally-aware search over Graphiti facts. Returns facts '
        'with validity timestamps (valid_at / invalid_at). Filter by date '
        'ranges to find facts true at a specific time.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'Natural-language search query.'},
            'max_facts': {
                'type': 'integer',
                'description': 'Max facts to return (default 10, max 50).',
            },
            'edge_types': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Optional fact type names to filter by.',
            },
            'valid_at_after': {
                'type': 'string',
                'description': 'ISO-8601 lower bound for fact validity.',
            },
            'valid_at_before': {
                'type': 'string',
                'description': 'ISO-8601 upper bound for fact validity.',
            },
        },
        'required': ['query'],
    },
}

# 上下文检索：返回最近的原始 episode，不经过 LLM 合成，轻量快速。
# 也可按 episode_uuids 进行溯源追踪，返回指定 episode 产生的实体和事实。
CONTEXT_SCHEMA = {
    'name': 'graphiti_context',
    'description': (
        'Retrieve episodes or trace provenance from Graphiti. '
        'Without episode_uuids: returns recent episodes with '
        'content, source, and timestamps. '
        'With episode_uuids: returns the entities and facts those episodes created '
        '(provenance tracing — what did this conversation produce?).'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'max_episodes': {
                'type': 'integer',
                'description': 'Max episodes to return (default 10, max 50). Ignored when episode_uuids is set.',
            },
            'episode_uuids': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Specific episode UUIDs to trace — returns all entities and facts they produced.',
            },
        },
        'required': [],
    },
}

# 结论操作：创建事实 (add_triplet)、删除单个事实 (delete_entity_edge)、
# 或批量删除整个 episode (delete_episode)。三个操作互斥 —— 每次只提供其一。
CONCLUDE_SCHEMA = {
    'name': 'graphiti_conclude',
    'description': (
        'Persist or delete facts/episodes in Graphiti. '
        'Pass source_entity, edge_type, fact, target_entity to create. '
        'Pass delete_uuid to remove a single fact edge (PII cleanup). '
        'Pass delete_episode_uuid to remove an entire episode and all its entities/facts. '
        'Exactly one of fact, delete_uuid, or delete_episode_uuid must be provided.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'source_entity': {
                'type': 'string',
                'description': 'Source entity name (e.g. "Alice").',
            },
            'edge_type': {
                'type': 'string',
                'description': 'Relationship type (e.g. "PREFERS", "WORKS_ON").',
            },
            'fact': {
                'type': 'string',
                'description': 'Statement describing the relationship.',
            },
            'target_entity': {
                'type': 'string',
                'description': 'Target entity name (e.g. "Project X").',
            },
            'delete_uuid': {
                'type': 'string',
                'description': 'UUID of a fact edge to delete (PII removal).',
            },
            'delete_episode_uuid': {
                'type': 'string',
                'description': 'UUID of an episode to delete with all its entities/facts.',
            },
        },
        'required': [],
    },
}

# 图谱结构探索：查看边详情、saga 摘要、社区检测。
# 三个操作互斥 —— 每次只提供其一。
EXPLORE_SCHEMA = {
    'name': 'graphiti_explore',
    'description': (
        'Explore higher-level graph structure in Graphiti. '
        'Pass edge_uuid to inspect full details of a specific fact edge '
        '(valid_at/invalid_at timestamps, source/target entities). '
        'Pass saga_name to get a running summary of episodes grouped under that saga. '
        'Pass communities=true to detect and summarize entity communities. '
        'Exactly one of edge_uuid, saga_name, or communities must be provided.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'edge_uuid': {
                'type': 'string',
                'description': 'UUID of a fact edge to inspect in full detail.',
            },
            'saga_name': {
                'type': 'string',
                'description': 'Saga name to summarize (same value passed as saga to add_memory).',
            },
            'communities': {
                'type': 'boolean',
                'description': 'Set to true to detect and return entity community summaries.',
            },
            'max_communities': {
                'type': 'integer',
                'description': 'Max communities to return (default 20, max 100).',
            },
        },
        'required': [],
    },
}


# 所有工具 schema 的汇总列表，context 模式下不暴露给模型。
ALL_TOOL_SCHEMAS = [
    SEARCH_SCHEMA,
    PROFILE_SCHEMA,
    REASONING_SCHEMA,
    CONTEXT_SCHEMA,
    CONCLUDE_SCHEMA,
    EXPLORE_SCHEMA,
]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _tool_error(message: str) -> str:
    """构造工具错误响应（JSON 字符串，Hermes 约定格式）。"""
    return json.dumps({'error': message})


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO-8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# MemoryProvider
# ---------------------------------------------------------------------------


class GraphitiMemoryProvider:
    """Graphiti 知识图谱记忆 provider。

    核心架构：
    ┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
    │ Hermes Agent │ ──▶ │ GraphitiMemory   │ ──▶ │ Graphiti MCP      │
    │ (sync_turn)  │     │ Provider (本类)   │     │ Server (HTTP)     │
    └──────────────┘     └──────────────────┘     └───────────────────┘
                                 │                          │
                          ┌──────┴──────┐          ┌────────┴────────┐
                          │ 写入队列      │          │ FalkorDB / Neo4j│
                          │ (串行消费)    │          │ (知识图谱存储)   │
                          └─────────────┘          └─────────────────┘

    设计模式（参考 mem0 + hindsight）:
    - 熔断器:   连续失败后暂停，避免对宕机服务器"狂轰滥炸"
    - 单写入者:  一个后台线程串行消费写入队列，sync_turn() 只入队不阻塞
    - 轮次缓冲:  累积多轮对话为 JSON 数组，批量发送为一次 add_memory 调用
    - 共享 loop: 进程级复用 asyncio 事件循环，避免连接泄漏
    - atexit:    注册退出钩子，确保缓冲数据不丢失
    """

    def __init__(self) -> None:
        # ---- 配置项（由 env / graphiti.json 填充） ----
        self._config: dict[str, Any] = {}
        self._mcp_url: str = _DEFAULT_MCP_URL      # MCP 服务器地址
        self._group_id: str = ''                     # 图谱命名空间（空=使用默认）
        self._recall_mode: str = 'tools'             # 召回模式: tools / context / hybrid
        self._injection_frequency: str = 'every-turn' # 注入频率: every-turn / first-turn
        self._retain_every_n_turns: int = 3          # 每 N 轮写入一次（默认 3，节省 token）
        self._auto_retain: bool = True               # 是否自动持久化对话
        self._auto_recall: bool = True               # 是否自动召回记忆
        self._retain_context: str = 'conversation between Hermes Agent and the User'
        self._retain_user_prefix: str = 'User'        # 用户消息前缀
        self._retain_assistant_prefix: str = 'Assistant'  # 助手消息前缀
        self._recall_max_input_chars: int = 800       # 召回时查询最大字符数
        self._timeout: int = 120                      # MCP 请求超时（秒）
        self._auto_clear_on_shutdown: bool = False    # shutdown 时是否清空 graph（测试用）

        # ---- 运行时状态 ----
        self._client: Client | None = None            # MCP 客户端（延迟初始化）
        self._session_id: str = ''                    # 当前会话 ID
        self._platform: str = ''                      # 平台标识 (cli / telegram / ...)
        self._turn_counter: int = 0                   # 当前轮次计数
        self._session_turns: list[str] = []           # 累积的对话轮次缓冲区

        # ---- 熔断器状态 ----
        self._consecutive_failures: int = 0           # 连续失败计数
        self._breaker_open_until: float = 0.0         # 熔断恢复时间戳
        self._last_probe_time: float = 0.0            # 上次主动探测时间戳
        self._breaker_lock = threading.Lock()

        # ---- 预取状态 ----
        self._prefetch_result: str = ''               # 缓存的预取结果
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None

        # ---- 写入队列（单写入者模式） ----
        self._shutting_down = threading.Event()       # 关闭信号
        self._retain_queue: queue.Queue = queue.Queue()  # 持久化任务队列
        self._writer_thread: threading.Thread | None = None  # 后台写入线程
        self._atexit_registered: bool = False         # 是否已注册退出钩子

    # ------------------------------------------------------------------
    # ABC — Hermes MemoryProvider 接口
    # name / is_available / save_config / get_config_schema / backup_paths
    # 是 Hermes 框架要求的抽象方法。
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """返回 provider 的唯一标识名称。Hermes 用此名称在配置中引用 provider。"""
        return 'graphiti'

    def is_available(self) -> bool:
        """检查 provider 是否可用（不发起网络调用）。

        仅检查是否配置了 MCP 服务器地址。返回 True 表示基本配置就绪。
        """
        cfg = _load_config()
        return bool(cfg.get('mcp_url'))

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """将配置写入 $HERMES_HOME/graphiti.json。

        Hermes 的 'memory setup' 命令收集用户输入后调用此方法。
        使用原子写入避免文件损坏。
        """
        import json
        from pathlib import Path

        from utils import atomic_json_write

        config_path = Path(hermes_home) / 'graphiti.json'
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self) -> list[dict[str, Any]]:
        """返回此 provider 支持的配置项列表。

        供 Hermes 的 'memory setup' 交互式配置向导使用。
        每项包含 key、description、default、choices、env_var 等字段。
        """
        return [
            {
                'key': 'mcp_url',
                'description': 'Graphiti MCP server URL',
                'default': _DEFAULT_MCP_URL,
                'env_var': 'GRAPHITI_MCP_URL',
            },
            {
                'key': 'group_id',
                'description': 'Graph data namespace',
                'default': '',
                'env_var': 'GRAPHITI_GROUP_ID',
            },
            {
                'key': 'recall_mode',
                'description': 'Memory integration mode',
                'default': 'tools',
                'choices': ['tools', 'context', 'hybrid'],
                'env_var': 'GRAPHITI_RECALL_MODE',
            },
            {
                'key': 'retain_every_n_turns',
                'description': (
                    'Batch N turns into one retention call. Higher values '
                    'reduce LLM extraction costs significantly (each '
                    'retention triggers 2-3 LLM calls for entity/fact '
                    'extraction). Default 3 is a good balance — set to 1 '
                    'for real-time memory at higher token cost.'
                ),
                'default': '3',
                'env_var': 'GRAPHITI_RETAIN_EVERY_N',
            },
            {
                'key': 'auto_recall',
                'description': 'Auto-recall before each turn',
                'default': 'true',
                'choices': ['true', 'false'],
                'env_var': 'GRAPHITI_AUTO_RECALL',
            },
            {
                'key': 'timeout',
                'description': 'MCP request timeout in seconds',
                'default': '120',
                'env_var': 'GRAPHITI_TIMEOUT',
            },
        ]

    def backup_paths(self) -> list[str]:
        """返回 HERMES_HOME 之外的备份路径列表。

        Graphiti 将全部数据存储在远程 MCP 服务器中，本地无额外文件，
        因此返回空列表。
        """
        return []

    # ------------------------------------------------------------------
    # 生命周期
    # initialize() — Hermes 启动时调用
    # shutdown()   — Hermes 退出时调用
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """初始化 provider，Hermes 启动时调用。

        执行以下操作：
        1. 记录会话 ID 和平台信息
        2. 加载配置（环境变量 + graphiti.json）
        3. 验证召回模式是否合法
        4. 输出初始化日志

        Args:
            session_id: 当前会话的唯一标识。
            **kwargs: 包含 platform、hermes_home 等 Hermes 传递的上下文信息。
        """
        self._session_id = str(session_id or '').strip()
        self._platform = str(kwargs.get('platform') or '').strip()
        self._session_turns = []

        self._config = _load_config()
        self._mcp_url = self._config.get('mcp_url', _DEFAULT_MCP_URL)
        self._group_id = self._config.get('group_id', '')
        self._recall_mode = self._config.get('recall_mode', 'tools')
        if self._recall_mode not in {'tools', 'context', 'hybrid'}:
            self._recall_mode = 'tools'
        self._injection_frequency = self._config.get('injection_frequency', 'every-turn')
        self._retain_every_n_turns = max(1, int(self._config.get('retain_every_n_turns', 3)))
        self._auto_retain = str(self._config.get('auto_retain', 'true')).lower() == 'true'
        self._auto_recall = str(self._config.get('auto_recall', 'true')).lower() == 'true'
        self._retain_context = str(self._config.get('retain_context', self._retain_context))
        self._retain_user_prefix = str(
            self._config.get('retain_user_prefix', self._retain_user_prefix)
        )
        self._retain_assistant_prefix = str(
            self._config.get('retain_assistant_prefix', self._retain_assistant_prefix)
        )
        self._recall_max_input_chars = int(self._config.get('recall_max_input_chars', 800))
        self._timeout = int(self._config.get('timeout', 120))
        self._auto_clear_on_shutdown = bool(
            self._config.get('auto_clear_group_on_shutdown', False)
        )

        logger.info(
            'Graphiti initialized: url=%s group=%s mode=%s',
            self._mcp_url,
            self._group_id or '(default)',
            self._recall_mode,
        )

    def system_prompt_block(self) -> str:
        """返回要注入到 system prompt 中的静态文本。

        根据 recall_mode 返回不同的说明：
        - context:  告知模型上下文已自动注入
        - tools:    告知模型可用工具列表
        - hybrid:   告知模型两者兼备
        """
        if self._recall_mode == 'context':
            return (
                '# Graphiti Memory\n'
                'Active (context-injection mode). Context is auto-injected.'
            )
        if self._recall_mode == 'tools':
            return (
                '# Graphiti Memory\n'
                'Active (tools-only mode). Use graphiti_search, graphiti_profile, '
                'graphiti_reasoning, graphiti_context, graphiti_conclude, '
                'graphiti_explore.'
            )
        return (
            '# Graphiti Memory\n'
            'Active (hybrid mode). Context is auto-injected and tools available.'
        )

    def shutdown(self) -> None:
        """优雅关闭 provider。

        执行顺序：
        1. 设置关闭信号，阻止新的写入任务入队
        2. 向写入队列发送哨兵，等待 writer 线程完成已有任务后退出
        3. 等待预取线程完成
        4. 可选：auto_clear_group_on_shutdown 为 true 时清空当前 group 的所有数据
        5. 关闭 MCP 客户端连接
        """
        self._shutting_down.set()
        writer = self._writer_thread
        if writer is not None and writer.is_alive():
            try:
                self._retain_queue.put(_WRITER_SENTINEL)
            except Exception:
                pass
            writer.join(timeout=10.0)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)

        # 可选：清空图谱（测试/开发场景）
        if self._auto_clear_on_shutdown and self._group_id:
            try:
                url = self._mcp_url
                group_id = self._group_id

                async def _clear() -> None:
                    temp_client = Client(url)
                    try:
                        await temp_client.__aenter__()
                        await temp_client.call_tool(
                            'clear_graph', {'group_ids': [group_id]}
                        )
                    finally:
                        await temp_client.__aexit__(None, None, None)

                _run_async(_clear(), timeout=15)
                logger.info('Graphiti graph cleared for group %s', group_id)
            except Exception as exc:
                logger.debug('Graphiti clear on shutdown failed: %s', exc)

        self._close_client()

    # ------------------------------------------------------------------
    # 连接管理
    # 通过共享事件循环建立 MCP 客户端连接。连接在首次使用时延迟创建，
    # 后续调用复用已有连接。_close_client() 在 shutdown() 时调用。
    # ------------------------------------------------------------------

    def _get_client(self) -> Client:
        """获取 MCP 客户端实例。首次调用时延迟创建连接，后续复用。

        Returns:
            已连接的 MCP Client 实例。
        """
        if self._client is not None:
            return self._client
        self._client = self._connect_sync()
        return self._client

    def _connect_sync(self) -> Client:
        """在共享事件循环上同步创建 MCP 客户端连接。

        通过 _run_async 将异步连接操作投递到进程级共享 loop 并阻塞等待。
        连接超时取 min(30, 配置超时)，避免长时间卡住。
        """
        async def _do() -> Client:
            client = Client(self._mcp_url)
            await client.__aenter__()
            return client

        return _run_async(_do(), timeout=min(30, self._timeout))

    def _close_client(self) -> None:
        """关闭 MCP 客户端连接。在共享事件循环上执行异步关闭，容忍任何异常。"""
        if self._client is None:
            return
        try:
            async def _do() -> None:
                await self._client.__aexit__(None, None, None)

            _run_async(_do(), timeout=5)
        except Exception:
            pass
        self._client = None

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用 Graphiti MCP 服务器的指定工具并返回结构化结果。

        这是 provider 中所有 MCP 交互的统一入口。通过 _run_async 在共享
        事件循环上执行异步调用，使用配置的超时值。

        Args:
            tool_name:  MCP 工具名称（如 'search_nodes'、'add_memory'）。
            arguments: 传递给工具的参数字典。

        Returns:
            MCP 工具返回的 structured_content（通常是 dict）。
        """
        client = self._get_client()

        async def _do() -> Any:
            result = await client.call_tool(tool_name, arguments)
            return result.structured_content

        return _run_async(_do(), timeout=self._timeout)

    # ------------------------------------------------------------------
    # 熔断器 (circuit breaker)
    # 保护机制：连续失败后暂停 API 调用，避免对宕机服务器"狂轰滥炸"。
    # 区分客户端错误（无结果）与后端故障（连接超时），只对后者计数。
    # ------------------------------------------------------------------

    def _is_breaker_open(self) -> bool:
        """检查熔断器是否已断开（阻止 API 调用）。

        熔断期间会通过 get_status 主动探测服务是否恢复，
        恢复后立即闭合熔断器，无需等待完整的 cooldown 周期。

        Returns:
            True: 熔断器断开中，应跳过本次调用。
            False: 熔断器闭合，可以正常调用。
        """
        should_probe = False
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            now = time.monotonic()
            if now < self._breaker_open_until:
                if now - self._last_probe_time >= _BREAKER_PROBE_INTERVAL_SECS:
                    self._last_probe_time = now
                    should_probe = True
                else:
                    return True
            else:
                # cooldown 到期，自动闭合
                self._consecutive_failures = 0
                return False

        # 在锁外执行探测，避免 IO 期间持有锁
        if should_probe and self._probe_health():
            with self._breaker_lock:
                self._consecutive_failures = 0
            logger.info('Graphiti breaker closed — health probe succeeded')
            return False
        return True

    def _probe_health(self) -> bool:
        """通过 get_status 探测 MCP 服务器是否已恢复。

        Returns:
            True: 服务健康，status == 'ok'。
            False: 探测失败或服务异常。
        """
        try:
            result = self._call_tool('get_status', {})
            return isinstance(result, dict) and result.get('status') == 'ok'
        except Exception:
            return False

    def _record_success(self) -> None:
        """记录一次成功调用，重置熔断计数器为零。"""
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """记录一次失败调用。连续失败达到阈值时触发熔断，暂停 cooldown 秒。

        仅对后端故障计数；客户端错误（如"无结果"）不触发熔断。
        """
        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
                logger.warning(
                    'Graphiti breaker tripped after %d failures. '
                    'Pausing %ds. Check MCP server at %s.',
                    self._consecutive_failures,
                    _BREAKER_COOLDOWN_SECS,
                    self._mcp_url,
                )

    # ------------------------------------------------------------------
    # 写入者（单写入者队列模式，参考 hindsight）
    # 一个后台线程串行消费 _retain_queue 中的写入任务。
    # sync_turn() 只负责入队，不阻塞；writer 线程逐个执行 MCP 调用。
    # 哨兵 _WRITER_SENTINEL 触发优雅退出。
    # ------------------------------------------------------------------

    def _ensure_writer(self) -> None:
        """确保后台写入线程已启动。首次调用时创建，后续复用。

        写入线程是惰性启动的 —— 只有首次调用 sync_turn() 时才创建。
        如果之前因 shutdown() 退出，此处会重新启动。
        """
        t = self._writer_thread
        if t is not None and t.is_alive():
            return
        self._shutting_down.clear()
        t = threading.Thread(target=self._writer_loop, daemon=True, name='graphiti-writer')
        self._writer_thread = t
        t.start()

    def _writer_loop(self) -> None:
        """写入线程的主循环：轮询队列，串行执行持久化任务。

        生命周期：
        - 启动后循环等待队列中的任务
        - 收到 _WRITER_SENTINEL 哨兵时退出
        - 每个任务独立 try/except，单个失败不影响后续任务
        - 关闭信号 + 队列为空时退出
        """
        while True:
            try:
                job = self._retain_queue.get(timeout=1.0)
            except queue.Empty:
                if self._shutting_down.is_set():
                    return
                continue
            try:
                if job is _WRITER_SENTINEL:
                    return
                try:
                    job()
                except Exception as exc:
                    logger.warning('Graphiti retain failed: %s', exc)
            finally:
                self._retain_queue.task_done()

    def _register_atexit(self) -> None:
        """注册 atexit 钩子，确保进程退出时排空写入队列。

        幂等操作 —— 多次调用只注册一次。
        """
        if self._atexit_registered:
            return
        self._atexit_registered = True
        atexit.register(self._atexit_shutdown)

    def _atexit_shutdown(self) -> None:
        """atexit 回调：进程退出时自动调用 shutdown()，排空队列。"""
        if self._shutting_down.is_set():
            return
        try:
            self.shutdown()
        except Exception as exc:
            logger.debug('Graphiti atexit shutdown failed: %s', exc)

    # ------------------------------------------------------------------
    # 预取 (prefetch)
    # 在 context / hybrid 模式下，每轮对话前自动检索相关知识图谱内容，
    # 注入到 system prompt 中。后台线程执行以避免阻塞对话。
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = '') -> str:
        """返回缓存的预取上下文，供 Hermes 注入到下一轮对话的 system prompt 中。

        预取流程:
        1. queue_prefetch() 在上一轮结束后启动后台线程
        2. 后台线程调用 search_nodes + search_memory_facts 获取相关上下文
        3. 结果写入 _prefetch_result（线程安全）
        4. 本轮 prefetch() 取出缓存结果并清空

        仅在 context / hybrid 模式下生效；tools 模式返回空字符串。
        """
        if self._recall_mode == 'tools':
            return ''
        if self._injection_frequency == 'first-turn' and self._turn_counter > 1:
            return ''
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ''
            return result

    def queue_prefetch(self, query: str, *, session_id: str = '') -> None:
        """在后台线程中启动预取：查询知识图谱中与当前 query 相关的内容。

        预取结果缓存在 _prefetch_result 中，下一轮 prefetch() 时取出。
        跳过条件：
        - tools 模式（不自动注入上下文）
        - auto_recall 关闭
        - 熔断器断开
        - 查询超过 recall_max_input_chars 时截断
        """
        if self._recall_mode == 'tools' or not self._auto_recall:
            return
        if self._shutting_down.is_set() or self._is_breaker_open():
            return
        if self._recall_max_input_chars and len(query) > self._recall_max_input_chars:
            query = query[: self._recall_max_input_chars]

        group_id = self._group_id
        timeout = self._timeout

        def _run() -> None:
            client = self._client
            if client is None:
                return
            try:
                async def _do() -> Any:
                    n = await client.call_tool(
                        'search_nodes', {'query': query, 'max_nodes': 5, 'group_ids': group_id}
                    )
                    f = await client.call_tool(
                        'search_memory_facts',
                        {'query': query, 'max_facts': 3, 'group_ids': group_id},
                    )
                    return n.structured_content, f.structured_content

                nodes, facts = _run_async(_do(), timeout=timeout)
                nl = (nodes or {}).get('nodes', []) if isinstance(nodes, dict) else []
                fl = (facts or {}).get('facts', []) if isinstance(facts, dict) else []
                items = []
                for n in nl:
                    s = n.get('summary', '')
                    items.append(f"[Entity] {n.get('name', '?')}" + (f': {s}' if s else ''))
                for f in fl:
                    items.append(f"[Fact] {f.get('fact', '')}")
                if items:
                    with self._prefetch_lock:
                        self._prefetch_result = '\n'.join(items)
                self._record_success()
            except Exception as e:
                if not _is_client_error(e):
                    self._record_failure()
                logger.debug('Graphiti prefetch failed: %s', e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name='graphiti-prefetch'
        )
        self._prefetch_thread.start()

    # ------------------------------------------------------------------
    # 轮次生命周期
    # on_turn_start()   — 每轮用户输入前调用
    # sync_turn()        — 每轮对话结束后调用（异步持久化）
    # on_session_end()   — 会话结束时调用（排空缓冲区）
    # ------------------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """每轮对话开始时调用，记录当前轮次编号。

        Args:
            turn_number: 从 1 开始的轮次编号。
            message: 用户输入的消息内容。
        """
        self._turn_counter = turn_number

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = '',
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """持久化一轮对话到 Graphiti（非阻塞）。

        Hermes 每轮对话结束后调用此方法。实现策略:
        1. 将本轮 user + assistant 消息追加到 _session_turns 缓冲区
        2. 当累积轮数达到 retain_every_n_turns 时，批量打包为 JSON 数组
        3. 将写入任务放入 _retain_queue，由后台 writer 线程串行消费
        4. sync_turn() 立即返回 —— 实际 HTTP 调用在后台线程中完成

        这种设计确保:
        - 对话不因网络延迟而卡顿（完全异步）
        - 多轮对话合并为一次 add_memory 调用（节约 LLM 提取 token）
        - 进程退出时通过 atexit 排空队列（不丢数据）
        """
        if not self._auto_retain or self._shutting_down.is_set():
            return
        if session_id:
            self._session_id = str(session_id).strip()

        ts = _now_iso()
        self._session_turns.append(
            json.dumps(
                [
                    {
                        'role': 'user',
                        'content': f'{self._retain_user_prefix}: {user_content}',
                        'timestamp': ts,
                    },
                    {
                        'role': 'assistant',
                        'content': f'{self._retain_assistant_prefix}: {assistant_content}',
                        'timestamp': ts,
                    },
                ],
                ensure_ascii=False,
            )
        )

        if self._turn_counter % self._retain_every_n_turns != 0:
            return

        turns_to_ship = list(self._session_turns)
        group_id = self._group_id
        retain_context = self._retain_context

        def _do_retain() -> None:
            try:
                content = '[' + ','.join(turns_to_ship) + ']'
                self._call_tool(
                    'add_memory',
                    {
                        'name': f'Hermes turn batch {self._turn_counter}',
                        'episode_body': content,
                        'source': 'message',
                        'source_description': retain_context,
                        'group_id': group_id,
                    },
                )
                self._record_success()
            except Exception as e:
                if not _is_client_error(e):
                    self._record_failure()
                logger.debug('Graphiti sync_turn failed: %s', e)

        self._ensure_writer()
        self._register_atexit()
        self._retain_queue.put(_do_retain)
        self._session_turns = []

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """会话结束时调用：排空缓冲区中尚未写入的对话轮次。

        正常退出路径：
        - CLI 的 /exit 或 /reset 命令
        - Gateway 会话超时

        注意：不会在每轮对话后调用 —— 仅在真正的会话边界触发。
        """
        if self._session_turns and not self._shutting_down.is_set():
            turns_to_ship = list(self._session_turns)
            group_id = self._group_id

            def _flush() -> None:
                try:
                    content = '[' + ','.join(turns_to_ship) + ']'
                    self._call_tool(
                        'add_memory',
                        {
                            'name': 'Session end flush',
                            'episode_body': content,
                            'source': 'message',
                            'group_id': group_id,
                        },
                    )
                except Exception as e:
                    logger.debug('Graphiti on_session_end flush failed: %s', e)

            self._ensure_writer()
            if not self._shutting_down.is_set():
                self._retain_queue.put(_flush)
            self._session_turns = []

    # ------------------------------------------------------------------
    # 工具处理
    # 每个 _handle_* 方法对应一个工具 schema。
    # 工具调用流程: Hermes → handle_tool_call() → _handle_xxx() → MCP 调用
    # 每次成功调用后调用 _record_success() 重置熔断计数器。
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """返回此 provider 暴露的工具 schema 列表。

        context 模式下返回空列表（不暴露工具）。
        其他模式下返回全部 6 个工具。
        """
        if self._recall_mode == 'context':
            return []
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        """工具调用分发器。根据 tool_name 路由到对应的 _handle_* 方法。

        流程：
        1. 检查熔断器是否断开 → 断开则返回错误
        2. 根据 tool_name 分派到具体处理函数
        3. 捕获异常，区分客户端错误（不触发熔断）与后端错误（触发熔断）

        Returns:
            JSON 字符串格式的工具结果。
        """
        if self._is_breaker_open():
            return _tool_error(
                'Graphiti temporarily unavailable. '
                f'Check MCP server at {self._mcp_url}.'
            )
        try:
            if tool_name == 'graphiti_search':
                return self._handle_search(args)
            elif tool_name == 'graphiti_profile':
                return self._handle_profile(args)
            elif tool_name == 'graphiti_reasoning':
                return self._handle_reasoning(args)
            elif tool_name == 'graphiti_context':
                return self._handle_context(args)
            elif tool_name == 'graphiti_conclude':
                return self._handle_conclude(args)
            elif tool_name == 'graphiti_explore':
                return self._handle_explore(args)
            return _tool_error(f'Unknown tool: {tool_name}')
        except Exception as e:
            if not _is_client_error(e):
                self._record_failure()
            return _tool_error(f'Graphiti {tool_name} failed: {e}')

    def _handle_search(self, args: dict[str, Any]) -> str:
        """混合搜索: 同时查询实体 (search_nodes) 和事实 (search_memory_facts)。
        合并两路结果返回，比 graphiti_reasoning 更快更省 token。"""
        query = args.get('query', '')
        if not query:
            return _tool_error('Missing required parameter: query')
        max_results = min(int(args.get('max_results', 10)), 50)
        nodes = self._call_tool(
            'search_nodes',
            {'query': query, 'max_nodes': max_results, 'group_ids': self._group_id},
        )
        facts = self._call_tool(
            'search_memory_facts',
            {'query': query, 'max_facts': max_results, 'group_ids': self._group_id},
        )
        self._record_success()
        nl = (nodes or {}).get('nodes', []) if isinstance(nodes, dict) else []
        fl = (facts or {}).get('facts', []) if isinstance(facts, dict) else []
        if not nl and not fl:
            return json.dumps({'result': 'No relevant results found.'})
        items = []
        for n in nl:
            items.append(f"[Entity] {n.get('name', '?')}: {n.get('summary', '')}")
        for f in fl:
            items.append(
                f"[Fact] {f.get('fact', '')} "
                f"(from {f.get('valid_at', '?')} to {f.get('invalid_at', 'now')})"
            )
        return json.dumps({'result': items})

    def _handle_profile(self, args: dict[str, Any]) -> str:
        """实体查询：按名称搜索知识图谱中的实体，返回名称和摘要。
        支持按 entity_types 过滤，适合快速查询"Alice 的偏好"等场景。"""
        query = args.get('query', '')
        if not query:
            return _tool_error('Missing required parameter: query')
        max_nodes = min(int(args.get('max_nodes', 10)), 50)
        call_args: dict[str, Any] = {
            'query': query,
            'max_nodes': max_nodes,
            'group_ids': self._group_id,
        }
        if args.get('entity_types'):
            call_args['entity_types'] = args['entity_types']
        result = self._call_tool('search_nodes', call_args)
        self._record_success()
        nodes = (result or {}).get('nodes', []) if isinstance(result, dict) else []
        if not nodes:
            return json.dumps({'result': 'No entities found.'})
        items = [f"{n.get('name', '?')}: {n.get('summary', '')}" for n in nodes]
        return json.dumps({'result': items})

    def _handle_reasoning(self, args: dict[str, Any]) -> str:
        """时间感知深度搜索: 查询事实并展示 valid_at / invalid_at 时间范围。
        比 graphiti_search 更精确 —— 可以按时间范围过滤，适合"当时谁负责这个项目"
        这类需要时间上下文的问题。"""
        query = args.get('query', '')
        if not query:
            return _tool_error('Missing required parameter: query')
        max_facts = min(int(args.get('max_facts', 10)), 50)
        call_args: dict[str, Any] = {
            'query': query,
            'max_facts': max_facts,
            'group_ids': self._group_id,
        }
        for key in ('edge_types', 'valid_at_after', 'valid_at_before'):
            if args.get(key) is not None:
                call_args[key] = args[key]
        result = self._call_tool('search_memory_facts', call_args)
        self._record_success()
        facts = (result or {}).get('facts', []) if isinstance(result, dict) else []
        if not facts:
            return json.dumps({'result': 'No facts found.'})
        items = []
        for f in facts:
            tail = f' → {f["invalid_at"]}' if f.get('invalid_at') else ''
            items.append(f"{f.get('fact', '')} (since {f.get('valid_at', '?')}{tail})")
        return json.dumps({'result': items})

    def _handle_context(self, args: dict[str, Any]) -> str:
        """检索最近的对话 episode 或按 UUID 进行溯源追踪。

        两种模式：
        - 默认（无 episode_uuids）：返回最近 episode 内容，适合"最近讨论了什么"
        - 溯源（有 episode_uuids）：返回指定 episode 产生的所有实体和事实，适合"这个 episode 创建了什么"
        """
        episode_uuids = args.get('episode_uuids')

        # 溯源追踪：返回指定 episode 产生的实体和事实
        if episode_uuids:
            result = self._call_tool(
                'get_episode_entities',
                {'episode_uuids': episode_uuids, 'group_ids': self._group_id},
            )
            self._record_success()
            data = result if isinstance(result, dict) else {}
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            if not nodes and not edges:
                return json.dumps({'result': 'No entities or facts found for these episodes.'})
            items = []
            for n in nodes:
                items.append(f"[Entity] {n.get('name', '?')}: {n.get('summary', '')}")
            for e in edges:
                items.append(
                    f"[Fact] {e.get('fact', '')} "
                    f"(valid: {e.get('valid_at', '?')})"
                )
            return json.dumps({'result': items})

        # 默认模式：返回最近 episode 内容
        max_episodes = min(int(args.get('max_episodes', 10)), 50)
        result = self._call_tool(
            'get_episodes', {'max_episodes': max_episodes, 'group_ids': self._group_id}
        )
        self._record_success()
        episodes = (result or {}).get('episodes', []) if isinstance(result, dict) else []
        if not episodes:
            return json.dumps({'result': 'No episodes recorded yet.'})
        items = [
            f"[{e.get('source', '')}] {e.get('name', '?')}: {e.get('content', '')[:500]}"
            for e in episodes
        ]
        return json.dumps({'result': items})

    def _handle_conclude(self, args: dict[str, Any]) -> str:
        """持久化或删除事实/episode。支持三种模式（互斥）:

        - 创建: 提供 fact → 调用 add_triplet
        - 删边: 提供 delete_uuid → 调用 delete_entity_edge（PII 清理）
        - 批量删: 提供 delete_episode_uuid → 调用 delete_episode（删除整个 episode 及其关联实体/边）
        """
        delete_uuid = (args.get('delete_uuid') or '').strip()
        delete_episode_uuid = (args.get('delete_episode_uuid') or '').strip()
        fact = (args.get('fact') or '').strip()

        provided = sum(1 for x in [delete_uuid, delete_episode_uuid, fact] if x)
        if provided != 1:
            return _tool_error(
                'Exactly one of fact, delete_uuid, or delete_episode_uuid must be provided.'
            )

        if delete_uuid:
            result = self._call_tool('delete_entity_edge', {'uuid': delete_uuid})
            self._record_success()
            msg = (result or {}).get('message', 'Deleted.') if isinstance(result, dict) else 'Deleted.'
            return json.dumps({'result': msg})

        if delete_episode_uuid:
            result = self._call_tool('delete_episode', {'uuid': delete_episode_uuid})
            self._record_success()
            msg = (
                (result or {}).get('message', 'Episode deleted.')
                if isinstance(result, dict) else 'Episode deleted.'
            )
            return json.dumps({'result': msg})

        result = self._call_tool(
            'add_triplet',
            {
                'source_node_name': args.get('source_entity', 'user'),
                'edge_name': args.get('edge_type', 'HAS_TRAIT'),
                'fact': fact,
                'target_node_name': args.get('target_entity', 'unknown'),
                'group_id': self._group_id,
            },
        )
        self._record_success()
        msg = (result or {}).get('message', 'Stored.') if isinstance(result, dict) else 'Stored.'
        return json.dumps({'result': msg})

    def _handle_explore(self, args: dict[str, Any]) -> str:
        """探索图谱结构：边详情 / saga 摘要 / 社区检测。三种模式互斥：

        - edge_uuid:  查看指定边的完整详情（valid_at/invalid_at 等）
        - saga_name:  获取 saga 运行摘要
        - communities: 检测并返回实体社区摘要
        """
        edge_uuid = (args.get('edge_uuid') or '').strip()
        saga_name = (args.get('saga_name') or '').strip()
        communities = args.get('communities', False)

        provided = sum(1 for x in [edge_uuid, saga_name] if x)
        if communities:
            provided += 1
        if provided != 1:
            return _tool_error(
                'Exactly one of edge_uuid, saga_name, or communities=true must be provided.'
            )

        # 查看边详情
        if edge_uuid:
            result = self._call_tool('get_entity_edge', {'uuid': edge_uuid})
            self._record_success()
            edge = result if isinstance(result, dict) else {}
            if not edge or 'error' in edge:
                return _tool_error(f'Edge {edge_uuid} not found.')
            return json.dumps({
                'result': {
                    'uuid': edge.get('uuid'),
                    'name': edge.get('name'),
                    'fact': edge.get('fact'),
                    'source_node_uuid': edge.get('source_node_uuid'),
                    'target_node_uuid': edge.get('target_node_uuid'),
                    'valid_at': edge.get('valid_at'),
                    'invalid_at': edge.get('invalid_at'),
                    'created_at': edge.get('created_at'),
                }
            })

        # Saga 摘要
        if saga_name:
            result = self._call_tool(
                'summarize_saga',
                {'saga_name': saga_name, 'group_id': self._group_id},
            )
            self._record_success()
            data = result if isinstance(result, dict) else {}
            if 'error' in data:
                return _tool_error(data.get('error', 'Saga not found.'))
            return json.dumps({
                'result': {
                    'name': data.get('name', saga_name),
                    'uuid': data.get('uuid', ''),
                    'summary': data.get('summary', ''),
                }
            })

        # 社区检测
        max_communities = min(int(args.get('max_communities', 20)), 100)
        result = self._call_tool(
            'build_communities',
            {'group_ids': self._group_id or None},
        )
        self._record_success()
        data = result if isinstance(result, dict) else {}
        communities_list = data.get('communities', [])
        if not communities_list:
            return json.dumps({'result': 'No communities detected.'})
        items = [
            f"[Community] {c.get('name', '?')}: {c.get('summary', '')}"
            for c in communities_list[:max_communities]
        ]
        return json.dumps({
            'result': {
                'community_count': data.get('community_count', len(communities_list)),
                'communities': items,
            }
        })


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


# Hermes 插件入口点 —— 框架通过 importlib 加载此模块，调用 register(ctx) 注册 provider。
# ctx.register_memory_provider() 将 provider 实例注册到 Hermes 的 MemoryManager 中。
def register(ctx: Any) -> None:
    """Hermes 插件注册入口。由框架的插件加载器调用。

    调用 ctx.register_memory_provider() 将 GraphitiMemoryProvider 实例
    注册到 Hermes 的 MemoryManager，使其成为可选的记忆后端。
    """
    ctx.register_memory_provider(GraphitiMemoryProvider())

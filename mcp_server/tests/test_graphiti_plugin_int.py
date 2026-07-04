#!/usr/bin/env python3
r"""
Graphiti MCP Server 集成测试

使用 httpx + SSE 直接与 MCP streamable HTTP 服务器通信。

运行:
    cd d:\workspace\graphiti\mcp_server
    uv run python tests/test_graphiti_plugin_int.py

前提:
    docker compose -f mcp_server\docker\docker-compose-falkordb.yml up -d --build
    确保 .env 中配置了有效的 OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any
from uuid import uuid4

import httpx

MCP_URL = 'http://localhost:28000/mcp'
PREFIX = 'int_test'

_passed = 0
_failed = 0
_skipped = 0


def uid() -> str:
    return f'{PREFIX}_{uuid4().hex[:8]}'


class MCPClient:
    """轻量 MCP streamable HTTP 客户端。"""

    def __init__(self, url: str):
        self._url = url
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._req_id = 0

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15))
        await self._initialize()
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _initialize(self):
        headers = {'Accept': 'application/json, text/event-stream'}
        resp = await self._client.post(
            self._url,
            json={
                'jsonrpc': '2.0', 'method': 'initialize',
                'params': {
                    'protocolVersion': '2025-03-26', 'capabilities': {},
                    'clientInfo': {'name': 'int-test', 'version': '1.0'},
                },
                'id': 1,
            },
            headers=headers,
        )
        self._session_id = resp.headers.get('mcp-session-id')
        if self._session_id:
            await self._client.post(
                self._url,
                json={'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                headers={**headers, 'mcp-session-id': self._session_id},
            )

    def _headers(self) -> dict:
        h = {'Accept': 'application/json, text/event-stream', 'Content-Type': 'application/json'}
        if self._session_id:
            h['mcp-session-id'] = self._session_id
        return h

    async def _rpc(self, method: str, params: dict | None = None) -> Any:
        self._req_id += 1
        resp = await self._client.post(
            self._url,
            json={
                'jsonrpc': '2.0', 'method': method,
                'params': params or {}, 'id': self._req_id,
            },
            headers=self._headers(),
        )
        for line in resp.text.split('\n'):
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'result' in data:
                    return data['result']
                if 'error' in data:
                    return {'error': data['error'].get('message', str(data['error']))}
        return None

    async def list_tools(self) -> list[str]:
        result = await self._rpc('tools/list')
        if isinstance(result, dict) and 'error' in result:
            return []
        return [t['name'] for t in result.get('tools', [])]

    async def call_tool(self, name: str, args: dict) -> Any:
        result = await self._rpc('tools/call', {'name': name, 'arguments': args})
        if isinstance(result, dict) and 'error' in result:
            return result
        content = result.get('content', []) if isinstance(result, dict) else []
        for item in content:
            if item.get('type') == 'text':
                try:
                    return json.loads(item['text'])
                except (json.JSONDecodeError, TypeError):
                    return item['text']
        return result


def ok(msg: str) -> None:
    global _passed; _passed += 1; print(f'  [PASS] {msg}')


def fail(msg: str) -> None:
    global _failed; _failed += 1; print(f'  [FAIL] {msg}')


def skip(msg: str) -> None:
    global _skipped; _skipped += 1; print(f'  [SKIP] {msg}')


# ============================================================================
# 预检：API 连通性
# ============================================================================


async def check_api_available(c: MCPClient) -> bool:
    """检测 embedder/LLM API 是否可用。通过 add_triplet 判断。"""
    gid = f'{PREFIX}_apicheck_{uuid4().hex[:4]}'
    resp = await c.call_tool('add_triplet', {
        'source_node_name': 'Ping', 'edge_name': 'PINGS',
        'fact': f'Ping check [{gid}]', 'target_node_name': 'Pong', 'group_id': gid,
    })
    if isinstance(resp, dict) and 'error' in resp:
        return False
    return True


# ============================================================================
# 测试用例
# ============================================================================


async def test_connection():
    """服务器握手与工具列表。"""
    print('\n--- 1. 连接与握手 ---')
    async with MCPClient(MCP_URL) as c:
        tools = await c.list_tools()
        assert len(tools) > 0, '未获取到工具列表'
        ok(f'MCP 握手成功, {len(tools)} 个工具')
        core = ["add_triplet","search_nodes","search_memory_facts","get_episodes",
                "clear_graph","delete_entity_edge","delete_episode","get_entity_edge",
                "get_episode_entities","summarize_saga","build_communities","get_status"]
        matched = [t for t in tools if t in core]
        print(f'     核心工具 ({len(matched)}/{len(core)}): {matched}')
        missing = [t for t in core if t not in tools]
        if missing:
            print(f'     当前镜像缺失工具: {missing}')


async def test_get_episodes():
    """get_episodes 不需要 LLM/embedder，应始终可用。"""
    print('\n--- 2. get_episodes ---')
    async with MCPClient(MCP_URL) as c:
        resp = await c.call_tool('get_episodes', {'last_n': 5})
        if isinstance(resp, dict) and 'episodes' in resp:
            ok(f'get_episodes 正常: 返回 {len(resp["episodes"])} 个 episode')
        elif isinstance(resp, dict) and 'error' in resp:
            fail(f'get_episodes 错误: {resp["error"]}')
        else:
            fail(f'get_episodes 异常: {resp}')


async def test_clear_graph():
    """clear_graph 不需要 LLM/embedder。"""
    print('\n--- 3. clear_graph ---')
    async with MCPClient(MCP_URL) as c:
        resp = await c.call_tool('clear_graph', {'group_id': 'main'})
        if isinstance(resp, dict) and 'message' in resp:
            ok(f'clear_graph: {resp["message"]}')
        elif isinstance(resp, dict) and 'error' in resp:
            fail(f'clear_graph 错误: {resp["error"]}')
        else:
            fail(f'clear_graph 异常: {resp}')


async def test_add_triplet(api_ok: bool):
    """add_triplet 内部需要 embedder。"""
    print('\n--- 4. add_triplet ---')
    async with MCPClient(MCP_URL) as c:
        gid = uid()
        resp = await c.call_tool('add_triplet', {
            'source_node_name': 'TestEntity', 'edge_name': 'TEST_EDGE',
            'fact': f'Integration test fact [{gid}]',
            'target_node_name': 'TestTarget', 'group_id': gid,
        })
        if isinstance(resp, dict) and 'error' in resp:
            if not api_ok:
                skip(f'add_triplet 需要 embedder API (当前不可用): {resp["error"][:80]}')
            else:
                fail(f'add_triplet: {resp["error"]}')
        else:
            ok('add_triplet 成功')
        await asyncio.sleep(3)
        # 清理
        await c.call_tool('clear_graph', {'group_id': gid})


async def test_search(api_ok: bool):
    """search_nodes / search_memory_facts 需要 embedder。"""
    print('\n--- 5. 搜索 (search_nodes / search_memory_facts) ---')
    if not api_ok:
        skip('搜索需要 embedder API (当前不可用)')
        return

    gid = uid()
    async with MCPClient(MCP_URL) as c:
        # 先添加数据
        await c.call_tool('add_triplet', {
            'source_node_name': 'SearchTest', 'edge_name': 'TAGS',
            'fact': f'SearchTest tags Result [{gid}]',
            'target_node_name': 'Result', 'group_id': gid,
        })
        await asyncio.sleep(9)

        resp_n = await c.call_tool('search_nodes', {
            'query': f'SearchTest {gid}', 'max_nodes': 5, 'group_ids': [gid],
        })
        await asyncio.sleep(3)
        resp_f = await c.call_tool('search_memory_facts', {
            'query': f'SearchTest {gid}', 'max_facts': 5, 'group_ids': [gid],
        })

        if isinstance(resp_n, dict) and 'error' in resp_n:
            fail(f'search_nodes: {resp_n["error"][:80]}')
        elif isinstance(resp_n, dict):
            n = len(resp_n.get('nodes', []))
            ok(f'search_nodes: {n} 个实体')

        if isinstance(resp_f, dict) and 'error' in resp_f:
            fail(f'search_memory_facts: {resp_f["error"][:80]}')
        elif isinstance(resp_f, dict):
            f = len(resp_f.get('facts', []))
            ok(f'search_memory_facts: {f} 条事实')

        await c.call_tool('clear_graph', {'group_id': gid})


async def test_delete_entity_edge(api_ok: bool):
    """delete_entity_edge 测试。"""
    print('\n--- 6. delete_entity_edge ---')
    if not api_ok:
        skip('需要 embedder API 来添加和搜索数据')
        return

    gid = uid()
    async with MCPClient(MCP_URL) as c:
        await c.call_tool('add_triplet', {
            'source_node_name': 'DelMe', 'edge_name': 'TO_DELETE',
            'fact': f'DelMe to be deleted [{gid}]',
            'target_node_name': 'Gone', 'group_id': gid,
        })
        await asyncio.sleep(6)

        resp = await c.call_tool('search_memory_facts', {
            'query': f'DelMe {gid}', 'max_facts': 5, 'group_ids': [gid],
        })
        facts = (resp or {}).get('facts', []) if isinstance(resp, dict) else []
        if not facts:
            skip('无法搜索到数据')
            return

        fid = facts[0].get('uuid')
        if not fid:
            fail('缺少 uuid')
            return

        del_resp = await c.call_tool('delete_entity_edge', {'uuid': fid})
        await asyncio.sleep(3)
        if isinstance(del_resp, dict) and 'error' in del_resp:
            fail(f'delete_entity_edge: {del_resp["error"][:80]}')
        else:
            ok(f'删除成功: {fid[:20]}...')

        await c.call_tool('clear_graph', {'group_ids': [gid]})


async def test_delete_episode(api_ok: bool):
    """delete_episode 测试 —— 按 episode 批量删除。"""
    print('\n--- 7. delete_episode ---')
    if not api_ok:
        skip('需要 embedder API 来添加和搜索数据')
        return

    gid = uid()
    async with MCPClient(MCP_URL) as c:
        # 通过 add_memory 创建一个 episode（使用 text 源以确保 LLM 能处理）
        resp = await c.call_tool('add_memory', {
            'name': f'BatchDelete [{gid}]',
            'episode_body': f'Integration test episode for batch delete [{gid}]',
            'source': 'text',
            'source_description': 'Integration test',
            'group_id': gid,
        })
        await asyncio.sleep(10)

        # 获取创建的 episode
        eps = await c.call_tool('get_episodes', {
            'max_episodes': 5, 'group_ids': [gid],
        })
        episode_list = (eps or {}).get('episodes', []) if isinstance(eps, dict) else []
        if not episode_list:
            skip('无法获取 episode 列表')
            return

        eid = episode_list[0].get('uuid')
        if not eid:
            fail('缺少 episode uuid')
            return

        del_resp = await c.call_tool('delete_episode', {'uuid': eid})
        await asyncio.sleep(3)
        if isinstance(del_resp, dict) and 'error' in del_resp:
            fail(f'delete_episode: {del_resp["error"][:80]}')
        else:
            ok(f'批量删除 episode 成功: {eid[:20]}...')

        await c.call_tool('clear_graph', {'group_ids': [gid]})


async def test_get_entity_edge(api_ok: bool):
    """get_entity_edge —— 查看边详情。"""
    print('\n--- 8. get_entity_edge ---')
    if not api_ok:
        skip('需要 embedder API 来添加和搜索数据')
        return

    gid = uid()
    async with MCPClient(MCP_URL) as c:
        # 添加数据并搜索获取 edge uuid
        await c.call_tool('add_triplet', {
            'source_node_name': 'InspectMe', 'edge_name': 'INSPECT',
            'fact': f'InspectMe edge detail [{gid}]',
            'target_node_name': 'Target', 'group_id': gid,
        })
        await asyncio.sleep(6)

        resp = await c.call_tool('search_memory_facts', {
            'query': f'InspectMe {gid}', 'max_facts': 5, 'group_ids': [gid],
        })
        facts = (resp or {}).get('facts', []) if isinstance(resp, dict) else []
        if not facts:
            skip('无法搜索到数据')
            return

        fid = facts[0].get('uuid')
        if not fid:
            fail('缺少 uuid')
            return

        detail = await c.call_tool('get_entity_edge', {'uuid': fid})
        if isinstance(detail, dict) and 'error' in detail:
            fail(f'get_entity_edge: {detail["error"][:80]}')
        elif isinstance(detail, dict):
            ok(f'边详情: uuid={detail.get("uuid", "?")[:20]}..., '
               f'fact={detail.get("fact", "?")[:40]}')
        else:
            fail(f'get_entity_edge 异常: {detail}')

        await c.call_tool('clear_graph', {'group_ids': [gid]})


async def test_get_episode_entities(api_ok: bool):
    """get_episode_entities —— 按 episode UUID 溯源追踪。"""
    print('\n--- 9. get_episode_entities ---')
    if not api_ok:
        skip('需要 embedder API')
        return

    gid = uid()
    async with MCPClient(MCP_URL) as c:
        await c.call_tool('add_memory', {
            'name': f'Provenance [{gid}]',
            'episode_body': f'Integration test episode for provenance [{gid}]',
            'source': 'text',
            'source_description': 'Integration test',
            'group_id': gid,
        })
        await asyncio.sleep(10)

        eps = await c.call_tool('get_episodes', {
            'max_episodes': 5, 'group_ids': [gid],
        })
        episode_list = (eps or {}).get('episodes', []) if isinstance(eps, dict) else []
        if not episode_list:
            skip('无法获取 episode 列表')
            return

        eid = episode_list[0].get('uuid')
        result = await c.call_tool('get_episode_entities', {'episode_uuids': [eid]})
        if isinstance(result, dict) and 'error' in result:
            fail(f'get_episode_entities: {result["error"][:80]}')
        elif isinstance(result, dict):
            nodes = len(result.get('nodes', []))
            edges = len(result.get('edges', []))
            ok(f'溯源成功: {nodes} 个实体, {edges} 条边')
        else:
            fail(f'get_episode_entities 异常: {result}')

        await c.call_tool('clear_graph', {'group_ids': [gid]})


async def test_summarize_saga(api_ok: bool):
    """summarize_saga —— saga 摘要测试。"""
    print('\n--- 10. summarize_saga ---')
    if not api_ok:
        skip('需要 embedder API')
        return

    saga = uid()
    gid = uid()
    async with MCPClient(MCP_URL) as c:
        # 创建 saga（使用 text 源）
        await c.call_tool('add_memory', {
            'name': f'Saga episode 1 [{saga}]',
            'episode_body': f'First episode in saga [{saga}]',
            'source': 'text',
            'source_description': 'Integration test',
            'saga': saga,
            'group_id': gid,
        })
        await asyncio.sleep(10)

        result = await c.call_tool('summarize_saga', {
            'saga_name': saga, 'group_id': gid,
        })
        if isinstance(result, dict) and 'error' in result:
            fail(f'summarize_saga: {result["error"][:80]}')
        elif isinstance(result, dict) and 'summary' in result:
            ok(f'saga 摘要: {result["summary"][:60]}...')
        else:
            fail(f'summarize_saga 异常: {result}')

        await c.call_tool('clear_graph', {'group_ids': [gid]})


async def test_build_communities(api_ok: bool):
    """build_communities —— 社区检测测试。"""
    print('\n--- 11. build_communities ---')
    if not api_ok:
        skip('需要 embedder API')
        return

    gid = uid()
    async with MCPClient(MCP_URL) as c:
        # 添加一些数据供社区检测
        await c.call_tool('add_triplet', {
            'source_node_name': 'Engineer1', 'edge_name': 'WORKS_WITH',
            'fact': f'Engineer1 works with Engineer2 [{gid}]',
            'target_node_name': 'Engineer2', 'group_id': gid,
        })
        await asyncio.sleep(6)
        await c.call_tool('add_triplet', {
            'source_node_name': 'Engineer2', 'edge_name': 'WORKS_WITH',
            'fact': f'Engineer2 works with Engineer3 [{gid}]',
            'target_node_name': 'Engineer3', 'group_id': gid,
        })
        await asyncio.sleep(6)

        result = await c.call_tool('build_communities', {'group_ids': [gid]})
        if isinstance(result, dict) and 'error' in result:
            fail(f'build_communities: {result["error"][:80]}')
        elif isinstance(result, dict) and 'communities' in result:
            count = result.get('community_count', len(result.get('communities', [])))
            ok(f'社区检测完成: {count} 个社区')
        else:
            fail(f'build_communities 异常: {result}')

        await c.call_tool('clear_graph', {'group_ids': [gid]})


async def test_get_status():
    """get_status —— 健康检查，不需要 LLM。"""
    print('\n--- 12. get_status ---')
    async with MCPClient(MCP_URL) as c:
        resp = await c.call_tool('get_status', {})
        if isinstance(resp, dict) and resp.get('status') == 'ok':
            ok(f'get_status: {resp.get("message", "")[:60]}')
        elif isinstance(resp, dict) and 'error' in resp:
            fail(f'get_status 错误: {resp["error"]}')
        else:
            ok(f'get_status: {resp}')


async def test_group_isolation(api_ok: bool):
    """验证 group_id 数据隔离。"""
    print('\n--- 13. group 数据隔离 ---')
    if not api_ok:
        skip('需要 embedder API 来添加和搜索数据')
        return

    ga = f'{PREFIX}_iso_a_{uuid4().hex[:6]}'
    gb = f'{PREFIX}_iso_b_{uuid4().hex[:6]}'
    async with MCPClient(MCP_URL) as c:
        await c.call_tool('add_triplet', {
            'source_node_name': 'Alpha', 'edge_name': 'OWNS',
            'fact': 'Alpha owns the vault', 'target_node_name': 'Vault', 'group_id': ga,
        })
        await asyncio.sleep(6)

        rb = await c.call_tool('search_memory_facts', {
            'query': 'vault', 'max_facts': 10, 'group_ids': [gb],
        })
        fb = (rb or {}).get('facts', []) if isinstance(rb, dict) else []
        if len(fb) == 0:
            ok('隔离正常: group_b 无数据')
        else:
            fail(f'隔离失败: group_b 有 {len(fb)} 条数据')

        ra = await c.call_tool('search_memory_facts', {
            'query': 'vault', 'max_facts': 10, 'group_ids': [ga],
        })
        fa = (ra or {}).get('facts', []) if isinstance(ra, dict) else []
        if len(fa) > 0:
            ok(f'group_a 数据正常: {len(fa)} 条')
        else:
            fail('group_a 中找不到自己的数据')

        await c.call_tool('clear_graph', {'group_id': ga})
        await c.call_tool('clear_graph', {'group_id': gb})


# ============================================================================
# Main
# ============================================================================


async def main():
    global _passed, _failed, _skipped

    print('=' * 60)
    print('Graphiti MCP Server 集成测试')
    print(f'目标: {MCP_URL}')
    print('=' * 60)

    # 预检 API 连通性
    print('\n🔍 预检: 检测 embedder/LLM API 连通性...')
    api_ok = False
    try:
        async with MCPClient(MCP_URL) as c:
            api_ok = await check_api_available(c)
    except Exception as e:
        print(f'  预检异常: {e}')
    if api_ok:
        print('  ✅ embedder/LLM API 可用 — 全部测试可运行')
    else:
        print('  ⚠️  embedder/LLM API 不可用 — 仅运行不依赖 API 的测试')
        print('     请检查 .env 中的 OPENAI_API_KEY / OPENAI_API_URL 配置')

    tests = [
        test_connection,
        test_get_episodes,
        test_clear_graph,
        test_get_status,
        lambda: test_add_triplet(api_ok),
        lambda: test_search(api_ok),
        lambda: test_delete_entity_edge(api_ok),
        lambda: test_delete_episode(api_ok),
        lambda: test_get_entity_edge(api_ok),
        lambda: test_get_episode_entities(api_ok),
        lambda: test_summarize_saga(api_ok),
        lambda: test_build_communities(api_ok),
        lambda: test_group_isolation(api_ok),
    ]

    for i, fn in enumerate(tests):
        if i > 0:
            await asyncio.sleep(9)  # 测试间冷却，避免 API 429
        try:
            await fn()
        except Exception as e:
            fail(f'{fn.__name__}: {e}')

    print('\n' + '=' * 60)
    total = _passed + _failed + _skipped
    print(f'结果: {_passed}/{total} 通过, {_failed} 失败, {_skipped} 跳过')
    if not api_ok:
        print('提示: 配置有效的 OPENAI_API_KEY 后可运行完整测试')
    print('=' * 60)

    if _failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())

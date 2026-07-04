# Graphiti Memory Provider — Hermes 接口实现说明

本文档对照 [Hermes Memory Provider Plugin 规范](https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin)，
说明 `plugins/memory/graphiti/__init__.py` 如何满足各项要求，并记录设计决策与跨 provider 对比。

---

## 目录

- [目录结构](#目录结构)
- [plugin.yaml](#pluginyaml)
- [工具设计](#工具设计)
  - [6 个工具 Schema](#6-个工具-schema)
  - [MCP 端点覆盖情况](#mcp-端点覆盖情况)
  - [与其他 Provider 的工具对比](#与其他-provider-的工具对比)
  - [设计哲学：为什么是 6 个工具而非 13 个](#设计哲学为什么是-6-个工具而非-13-个)
- [核心生命周期（Required）](#核心生命周期required)
- [配置管理（Required）](#配置管理required)
  - [完整配置项列表](#完整配置项列表)
  - [推荐配置方案](#推荐配置方案)
  - [配置项调优指南](#配置项调优指南)
- [可选 Hook](#可选-hook)
- [线程契约（Threading Contract）](#线程契约threading-contract)
- [熔断器（Circuit Breaker）](#熔断器circuit-breaker)
- [预取机制（Prefetch）](#预取机制prefetch)
- [共享事件循环（Shared Event Loop）](#共享事件循环shared-event-loop)
- [优雅退出（Graceful Shutdown）](#优雅退出graceful-shutdown)
- [配置加载](#配置加载)
- [设计参考与跨 Provider 对比](#设计参考与跨-provider-对比)
- [运行模式](#运行模式)
- [插件入口](#插件入口)

---

## 目录结构

```
plugins/memory/graphiti/
├── __init__.py           # MemoryProvider 实现 + register() 入口
├── plugin.yaml           # 元数据（name, description, hooks）
├── graphiti.json.example # 配置样例（含注释）
└── README.md             # 本文档
```

✅ 符合 Hermes 规范：`plugins/memory/<name>/__init__.py` + `plugin.yaml`

---

## plugin.yaml

```yaml
name: graphiti
version: 1.0.0
description: "Graphiti — temporally-aware knowledge graph ..."
pip_dependencies:
  - "mcp>=1.27.2"
requires_env: []
hooks:
  - on_session_end
```

| 字段 | Hermes 要求 | 实现 |
|---|---|---|
| `name` | 唯一标识 | `graphiti` |
| `version` | 版本号 | `1.0.0` |
| `description` | 简要描述 | ✅ |
| `pip_dependencies` | 额外 pip 依赖 | `mcp>=1.27.2` |
| `hooks` | 声明实现的 hook | `on_session_end` |

---

## 工具设计

### 6 个工具 Schema

共 6 个工具（OpenAI function-calling 格式），每个都映射到 Graphiti MCP 服务器的对应端点：

| # | 工具名 | 意图（LLM 视角） | 底层 MCP 调用 | 说明 |
|---|--------|-----------------|-------------|------|
| 1 | `graphiti_search` | "搜索知识图谱" | `search_nodes` + `search_memory_facts` | 混合搜索：同时返回实体和事实 |
| 2 | `graphiti_profile` | "找人/找实体" | `search_nodes` | 按名称搜索实体，支持 `entity_types` 过滤 |
| 3 | `graphiti_reasoning` | "时间推理" | `search_memory_facts` | 时间感知深度搜索，支持 `valid_at_after/before` 过滤 |
| 4 | `graphiti_context` | "回顾对话" | `get_episodes` 或 `get_episode_entities` | 默认返回最近 episode；传 `episode_uuids` 时溯源追踪 |
| 5 | `graphiti_conclude` | "下定论" | `add_triplet` / `delete_entity_edge` / `delete_episode` | 三选一：创建事实、删单边、批量删 episode |
| 6 | `graphiti_explore` | "探索结构" | `get_entity_edge` / `summarize_saga` / `build_communities` | 三选一：边详情、saga 摘要、社区检测 |

### MCP 端点覆盖情况

Graphiti MCP 服务器共暴露 **13 个工具端点**，本 provider 实现 **100% 覆盖**：

```
MCP 端点                  Provider 中的承载方式              暴露给 Agent?
──────────────────────────────────────────────────────────────────────
add_memory              内部 sync_turn / on_session_end          ❌ 透明
search_nodes            graphiti_search / graphiti_profile        ✅
search_memory_facts     graphiti_search / graphiti_reasoning      ✅
add_triplet             graphiti_conclude (fact 分支)             ✅
delete_entity_edge      graphiti_conclude (delete_uuid 分支)      ✅
delete_episode          graphiti_conclude (delete_episode_uuid)   ✅ ✨
get_episodes            graphiti_context (默认模式)               ✅
get_episode_entities    graphiti_context (episode_uuids 模式)     ✅ ✨
get_entity_edge         graphiti_explore (edge_uuid 模式)         ✅ ✨
summarize_saga          graphiti_explore (saga_name 模式)         ✅ ✨
build_communities       graphiti_explore (communities 模式)       ✅ ✨
get_status              内部熔断器主动探测 (_probe_health)         ❌ 透明 ✨
clear_graph             内部 shutdown 清理 (_auto_clear)           ❌ 透明 ✨
──────────────────────────────────────────────────────────────────────
Agent 可见工具: 6 个    内部集成: 2 个    覆盖率: 13/13 = 100%
```

> ✨ = 相对于原有实现的新增覆盖

### 与其他 Provider 的工具对比

| # | Provider | 工具数 | 工具命名 | 设计风格 |
|---|----------|:----:|----------|---------|
| 1 | **ByteRover** | 3 | `brv_query`, `brv_curate`, `brv_status` | CLI 子进程 |
| 2 | **Hindsight** | 3 | `hindsight_retain`, `hindsight_recall`, `hindsight_reflect` | 认知语义 |
| 3 | **Holographic** | 2 | `fact_store`（action 枚举 9 种）, `fact_feedback` | 单工具 + action |
| 4 | **Honcho** | 5 | `honcho_search/profile/reasoning/context/conclude` | 认知语义 |
| 5 | **Mem0** | 5 | `mem0_list/search/add/update/delete` | CRUD 直译 |
| 6 | **OpenViking** | 6 | `viking_search/read/browse/remember/forget/add_resource` | 认知语义 + 领域扩展 |
| 7 | **RetainDB** | 10 | `retaindb_profile/search/context/remember/forget` + 5 文件工具 | CRUD + 文件 |
| 8 | **Supermemory** | 4 | `supermemory_store/search/forget/profile` | CRUD 直译 |
| 9 | **Graphiti** | **6** | `graphiti_search/profile/reasoning/context/conclude/explore` | 认知语义 |

**命名规律**：所有 9 个 provider 采用 `{provider前缀}_{操作动词}` 模式，确保 LLM 能清晰区分不同 provider 的工具。

**架构模式对比**：

| 模式 | 采用者 |
|------|--------|
| 熔断器 (Circuit Breaker) | Mem0, Graphiti |
| 单写入者队列 (Single Writer) | Hindsight, RetainDB, Graphiti |
| 共享事件循环 (Shared Event Loop) | Hindsight, Graphiti |
| atexit 安全网 | OpenViking, Graphiti |
| 后台预取 (Background Prefetch) | Mem0, Honcho, RetainDB, Supermemory |
| 轮次缓冲 (Turn Buffer) | Hindsight, Graphiti, Supermemory |
| MCP 协议通信 | **Graphiti（唯一）** |

### 设计哲学：为什么是 6 个工具而非 13 个

Graphiti MCP 服务器暴露 13 个工具端点，但 provider 选择将它们**聚合为 6 个面向 LLM 的高层抽象**，而非 1:1 透传。

#### 核心权衡：语义聚合 > 技术透传

| 维度 | 1:1 透传（13 工具） | 语义聚合（6 工具） |
|------|:---:|:---:|
| Schema 注入 token | ~2500 (+200%) | ~1000 (基线 +25%) |
| LLM 选择准确率 | ⬇️ 13 选 1，相似工具多 | ⬆️ 6 选 1，语义边界清晰 |
| 功能完整度 | 100% | 100% |
| 参数复杂度 | 每个工具参数少 | 部分工具有互斥参数 |

#### 聚合逻辑

每个工具对应 LLM 容易理解的**单一意图**，参数只是意图的自然变体：

```
"找人/找实体"          → graphiti_profile
"找事实/搜索"          → graphiti_search
"时间推理/深度搜索"     → graphiti_reasoning
"回顾对话/溯源"        → graphiti_context  （默认=最近列表, episode_uuids=溯源）
"创建/删除"            → graphiti_conclude （fact=创建, delete_uuid=删边, delete_episode_uuid=批量删）
"探索图谱结构"         → graphiti_explore  （edge_uuid=边详情, saga_name=摘要, communities=社区检测）
```

#### 为什么不是 5 个工具（与 Honcho 完全对齐）

最初考虑严格对齐 Honcho 的 5 工具模式，将 `get_entity_edge`、`summarize_saga`、`build_communities` 拆分塞入现有工具中。但语义分析表明：

| 操作 | 放入哪个工具？ | 语义适配度 |
|------|-------------|:---:|
| `get_entity_edge`（边详情） | `graphiti_profile` | ❌ "Profile 实体" vs "查看边" — 认知冲突 |
| `summarize_saga`（saga 摘要） | `graphiti_context` | ❌ "回顾原始记录" vs "获取总结" — 认知冲突 |
| `build_communities`（社区检测） | 任何现有工具 | ❌ 完全不匹配 |

**工具数不是负担，语义歧义才是负担。** Holographic 只有 2 个工具但 `fact_store` 的 `action` 枚举有 9 个值——LLM 更容易混淆。RetainDB 有 10 个工具但文件工具和 memory 工具边界明确——LLM 不容易混淆。

6 个工具在 Hermes 生态中处于中间地带（Honcho 5、OpenViking 6、RetainDB 10），是可接受的增量。

---

## 核心生命周期（Required）

### 1. `name` (property)

```python
@property
def name(self) -> str:
    return 'graphiti'
```

✅ Hermes 用此名称在配置中引用 provider（`memory.provider: graphiti`）。

### 2. `is_available()` — 必须无网络调用

```python
def is_available(self) -> bool:
    cfg = _load_config()
    return bool(cfg.get('mcp_url'))
```

✅ 仅检查是否配置了 MCP 服务器地址，**不发起网络请求**。

### 3. `initialize(session_id, **kwargs)` — Agent 启动时调用

```python
def initialize(self, session_id: str, **kwargs: Any) -> None:
    self._session_id = str(session_id or '').strip()
    self._platform = str(kwargs.get('platform') or '').strip()
    self._session_turns = []
    self._config = _load_config()
    # ... 加载所有配置项，包括 auto_clear_group_on_shutdown ...
```

✅ 从 `kwargs` 中提取 `platform`、`hermes_home` 等 Hermes 传递的上下文。

### 4. `get_tool_schemas()` — 返回工具 schema 列表

```python
def get_tool_schemas(self) -> list[dict[str, Any]]:
    if self._recall_mode == 'context':
        return []
    return list(ALL_TOOL_SCHEMAS)
```

✅ 返回 **6 个**工具 schema。`context` 模式下返回空列表（不暴露工具）。

### 5. `handle_tool_call(tool_name, args, **kwargs)` — 工具分发

```python
def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
    if self._is_breaker_open():
        return _tool_error('Graphiti temporarily unavailable...')
    try:
        if tool_name == 'graphiti_search':    return self._handle_search(args)
        elif tool_name == 'graphiti_profile':  return self._handle_profile(args)
        elif tool_name == 'graphiti_reasoning': return self._handle_reasoning(args)
        elif tool_name == 'graphiti_context':  return self._handle_context(args)
        elif tool_name == 'graphiti_conclude': return self._handle_conclude(args)
        elif tool_name == 'graphiti_explore':  return self._handle_explore(args)
        return _tool_error(f'Unknown tool: {tool_name}')
    except Exception as e:
        if not _is_client_error(e):
            self._record_failure()
        return _tool_error(f'Graphiti {tool_name} failed: {e}')
```

✅ 根据 `tool_name` 分派到对应的 `_handle_*` 方法（6 个），返回 JSON 字符串。熔断器断开时直接返回错误。

---

## 配置管理（Required）

### 6. `get_config_schema()` — 声明可配置字段

```python
def get_config_schema(self) -> list[dict[str, Any]]:
    return [
        {
            'key': 'mcp_url',
            'description': 'Graphiti MCP server URL',
            'default': 'http://localhost:8000/mcp',
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
        # ... retain_every_n_turns, auto_recall, timeout ...
    ]
```

✅ 每个字段包含 `key`、`description`、`default`、`env_var`。
`recall_mode` 提供 `choices` 限制选项。遵循"最小 schema"原则 — 仅暴露核心配置项。

### 7. `save_config(values, hermes_home)` — 持久化配置

```python
def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
    from utils import atomic_json_write
    config_path = Path(hermes_home) / 'graphiti.json'
    existing = {}
    if config_path.exists():
        existing = json.loads(config_path.read_text())
    existing.update(values)
    atomic_json_write(config_path, existing, mode=0o600)
```

✅ 写入 `$HERMES_HOME/graphiti.json`。使用 `atomic_json_write` 保证原子性。
已有配置合并追加（不覆盖未提供的字段）。符合 Profile Isolation 要求 — 使用传入的 `hermes_home` 而非硬编码路径。

### 完整配置项列表

| 配置键 | 环境变量 | 默认值 | 说明 |
|--------|---------|--------|------|
| `mcp_url` | `GRAPHITI_MCP_URL` | `http://localhost:8000/mcp` | MCP 服务器地址 |
| `group_id` | `GRAPHITI_GROUP_ID` | `""` | 图谱命名空间（空=使用服务器默认） |
| `recall_mode` | `GRAPHITI_RECALL_MODE` | `tools` | 召回模式：`tools` / `context` / `hybrid` |
| `injection_frequency` | `GRAPHITI_INJECTION_FREQUENCY` | `every-turn` | 注入频率：`every-turn` / `first-turn` |
| `retain_every_n_turns` | `GRAPHITI_RETAIN_EVERY_N` | `3` | 每 N 轮批量写入一次 |
| `auto_retain` | `GRAPHITI_AUTO_RETAIN` | `true` | 是否自动持久化对话 |
| `auto_recall` | `GRAPHITI_AUTO_RECALL` | `true` | 是否自动召回记忆 |
| `retain_context` | `GRAPHITI_RETAIN_CONTEXT` | 对话上下文描述 | 持久化时的上下文标签 |
| `retain_user_prefix` | `GRAPHITI_RETAIN_USER_PREFIX` | `User` | 用户消息前缀 |
| `retain_assistant_prefix` | `GRAPHITI_RETAIN_ASSISTANT_PREFIX` | `Assistant` | 助手消息前缀 |
| `recall_max_input_chars` | `GRAPHITI_RECALL_MAX_INPUT_CHARS` | `800` | 召回查询最大字符数 |
| `timeout` | `GRAPHITI_TIMEOUT` | `120` | MCP 请求超时秒数 |
| `auto_clear_group_on_shutdown` | `GRAPHITI_AUTO_CLEAR` | `false` | shutdown 时是否清空当前 group ✨ |

> ✨ = 新增配置项

### 推荐配置方案

不同场景下的推荐配置，附原因说明。

#### 场景 A：本地开发 / 测试

```json
{
    "mcp_url": "http://localhost:8000/mcp",
    "group_id": "dev",
    "recall_mode": "tools",
    "retain_every_n_turns": 1,
    "auto_clear_group_on_shutdown": true,
    "timeout": 60
}
```

| 配置 | 值 | 原因 |
|------|:---:|------|
| `recall_mode` | `tools` | 开发时需要显式调用工具调试，不让自动注入干扰 |
| `retain_every_n_turns` | `1` | 每轮立即写入，方便即时验证记忆效果 |
| `auto_clear_group_on_shutdown` | `true` | 每次重启清空测试数据，避免历史数据污染 |
| `timeout` | `60` | 本地网络低延迟，不需要过长超时 |

#### 场景 B：个人日常使用（CLI Agent）

```json
{
    "mcp_url": "http://localhost:8000/mcp",
    "group_id": "",
    "recall_mode": "hybrid",
    "injection_frequency": "every-turn",
    "retain_every_n_turns": 3,
    "auto_retain": true,
    "auto_recall": true,
    "timeout": 120
}
```

| 配置 | 值 | 原因 |
|------|:---:|------|
| `recall_mode` | `hybrid` | 工具 + 自动注入兼备 —— Agent 可主动搜索，也能自动获得相关上下文 |
| `retain_every_n_turns` | `3` | 每 3 轮批量写入，在记忆即时性和 LLM 成本之间取平衡。设为 1 会显著增加 token 消耗 |
| `group_id` | `""` | 使用服务器默认 group，单用户场景无需隔离 |
| `timeout` | `120` | 本地部署默认值 |

#### 场景 C：多用户 / 多租户生产环境

```json
{
    "mcp_url": "https://graphiti.example.com/mcp",
    "group_id": "user-{{user_id}}",
    "recall_mode": "hybrid",
    "injection_frequency": "first-turn",
    "retain_every_n_turns": 5,
    "auto_retain": true,
    "auto_recall": true,
    "recall_max_input_chars": 500,
    "timeout": 180
}
```

| 配置 | 值 | 原因 |
|------|:---:|------|
| `group_id` | `user-{id}` | 每个用户独立命名空间，数据严格隔离 |
| `injection_frequency` | `first-turn` | 仅首轮注入上下文，节省每轮的 token 开销。后续轮次 Agent 可通过工具按需搜索 |
| `retain_every_n_turns` | `5` | 较高批量值，减少 LLM 提取调用频率，控制成本 |
| `recall_max_input_chars` | `500` | 限制召回查询长度，避免长对话产生过大的上下文注入 |
| `timeout` | `180` | 远程部署可能网络延迟较高 |
| `mcp_url` | HTTPS | 生产环境必须加密传输 |

#### 场景 D：低成本 / 弱 LLM 部署

```json
{
    "mcp_url": "http://localhost:8000/mcp",
    "recall_mode": "context",
    "injection_frequency": "first-turn",
    "retain_every_n_turns": 10,
    "auto_retain": true,
    "auto_recall": true
}
```

| 配置 | 值 | 原因 |
|------|:---:|------|
| `recall_mode` | `context` | 不暴露工具，减少 function-calling 的 token 和推理开销。弱模型在 tool choice 上容易出错 |
| `retain_every_n_turns` | `10` | 极高的批量值，最小化 LLM 提取调用次数。记忆延迟高但成本极低 |
| `injection_frequency` | `first-turn` | 仅首轮一次上下文注入 |

#### 场景 E：纯工具模式（其他 provider 风格）

```json
{
    "mcp_url": "http://localhost:8000/mcp",
    "recall_mode": "tools",
    "auto_recall": false,
    "retain_every_n_turns": 3
}
```

| 配置 | 值 | 原因 |
|------|:---:|------|
| `recall_mode` | `tools` | 与其他 provider（mem0、hindsight）一致的体验：Agent 必须主动调用工具才能获取记忆 |
| `auto_recall` | `false` | 关闭自动召回，所有记忆检索由 Agent 通过工具调用完成 |

---

### 配置项调优指南

#### `retain_every_n_turns` — 成本与即时性的权衡

```
值        每次写入 LLM 调用次数    记忆延迟    月 token 成本（估算）
─────────────────────────────────────────────────────────────
1         每轮 2-3 次             实时        高（~800 次/月）
3（默认）  每 3 轮 2-3 次          ~3 轮      中（~270 次/月）
5         每 5 轮 2-3 次          ~5 轮      中低（~160 次/月）
10        每 10 轮 2-3 次         ~10 轮     低（~80 次/月）
```

> 每次 `add_memory` 调用触发 2-3 次 LLM 调用（实体提取 + 去重 + 摘要）。以每天 100 轮对话为例。

#### `recall_mode` — 模式选择决策树

```
是否信任 Agent 的工具选择能力？
├─ 是（强模型：GPT-4.1, Claude 4.5, Gemini 2.5）
│   └─ tools 或 hybrid
│       ├─ 需要自动上下文？ → hybrid
│       └─ 不需要自动上下文 → tools
└─ 否（弱模型 / 低成本模型）
    └─ context（纯注入，不给工具）
```

#### `group_id` — 隔离策略

| 策略 | 配置 | 适用场景 |
|------|------|---------|
| 无隔离 | `""` | 单用户、个人使用 |
| 持久隔离 | `"user-alice"` | 多用户，每个用户永久独立 graph |
| 会话隔离 | `"session-{uuid}"` | 每次对话独立 graph，关闭即丢弃 |
| 平台隔离 | `"cli"` / `"telegram"` | 同一用户不同平台间隔离 |

### 8. `backup_paths()` — 返回额外备份路径

```python
def backup_paths(self) -> list[str]:
    return []
```

✅ Graphiti 数据完全存储在远程 MCP 服务器中，本地无额外文件。

---

## 可选 Hook

| Hook | 实现 | 说明 |
|---|---|---|
| `system_prompt_block()` | ✅ | 根据 `recall_mode` 返回不同提示文本（提及 6 个工具） |
| `prefetch(query)` | ✅ | 返回缓存的预取上下文 |
| `queue_prefetch(query)` | ✅ | 后台线程预取，不阻塞对话 |
| `sync_turn(...)` | ✅ | 非阻塞：入队后立即返回（见线程契约） |
| `on_session_end(messages)` | ✅ | 排空缓冲区中未写入的对话轮次 |
| `shutdown()` | ✅ | 设置关闭信号 + 排空队列 + 可选清空图谱 + 关闭 MCP 连接 |

---

## 线程契约（Threading Contract）

> `sync_turn()` MUST be non-blocking. — Hermes 规范

### 实现方案：单写入者队列 + 轮次缓冲

```
sync_turn()                          后台 writer 线程
    │                                     │
    ├─ 追加 turn 到 _session_turns        │
    ├─ 达到 retain_every_n_turns?         │
    │   ├─ 是: 打包为 _do_retain() 闭包   │
    │   │       放入 _retain_queue ──────▶ 串行消费队列
    │   │       清空 _session_turns        │   └─ 调用 MCP add_memory
    │   └─ 否: 继续累积                   │
    └─ 立即返回（不阻塞）                 │
```

关键代码：

```python
def sync_turn(self, user_content, assistant_content, *, session_id='', messages=None):
    # ... 累积到 _session_turns 缓冲区 ...
    if self._turn_counter % self._retain_every_n_turns != 0:
        return  # 未达阈值，不写入

    def _do_retain():
        # 在后台线程中执行 MCP 调用
        self._call_tool('add_memory', {...})

    self._ensure_writer()
    self._retain_queue.put(_do_retain)  # 入队后立即返回
```

✅ `sync_turn()` 只负责**入队**，实际 HTTP 调用在后台 daemon 线程中完成。对话不会因网络延迟而卡顿。

---

## 熔断器（Circuit Breaker）

> 参考 mem0 provider 的实现模式 —— 服务宕机时自动暂停。

### 基础机制

```python
_BREAKER_THRESHOLD = 5       # 连续失败阈值
_BREAKER_COOLDOWN_SECS = 120 # 冷却时间（兜底值）
```

- 连续 5 次后端故障后触发熔断
- 区分客户端错误（`not found` 等"无结果"）与后端故障（连接超时、500 等）
- 只对后端故障计数

### 主动健康探测（增强特性）✨

传统熔断器在触发后**被动等待**固定的 120 秒冷却时间。本实现通过 `get_status` MCP 端点实现了**主动探测**：

```python
_BREAKER_PROBE_INTERVAL_SECS = 15  # 熔断期间每 15 秒主动探测一次

def _is_breaker_open(self) -> bool:
    # ...
    if now < self._breaker_open_until:
        if now - self._last_probe_time >= _BREAKER_PROBE_INTERVAL_SECS:
            self._last_probe_time = now
            # 在锁外执行 IO 探测，避免阻塞
            should_probe = True
    # 在锁外探测
    if should_probe and self._probe_health():
        # 服务恢复，立即闭合熔断器
        self._consecutive_failures = 0
        return False
    return True

def _probe_health(self) -> bool:
    """通过 get_status 探测 MCP 服务器是否已恢复。"""
    try:
        result = self._call_tool('get_status', {})
        return isinstance(result, dict) and result.get('status') == 'ok'
    except Exception:
        return False
```

**收益**：服务恢复后**最多 15 秒**即可重新可用，而非固定等待 120 秒。`get_status` 不暴露给 Agent，完全透明。

**并发安全**：探测（IO 操作）在 `_breaker_lock` 锁外执行，避免 IO 期间持有锁导致死锁或阻塞其他线程。

---

## 预取机制（Prefetch）

> 参考 hindsight provider 的实现模式。

```
上一轮结束                      本轮开始
    │                              │
    ├─ queue_prefetch(query)       ├─ prefetch(query)
    │   └─ 后台线程:                │   └─ 返回缓存的 _prefetch_result
    │       search_nodes +          │       并清空缓存
    │       search_memory_facts     │
    │       → _prefetch_result      │
```

✅ 后台线程执行（不阻塞），结果缓存供下一轮使用。`tools` 模式跳过。

---

## 共享事件循环（Shared Event Loop）

> 参考 hindsight provider 的实现模式。

```python
_loop: asyncio.AbstractEventLoop | None = None  # 进程级共享

def _get_loop() -> asyncio.AbstractEventLoop:
    # 首次调用时在后台 daemon 线程启动永久事件循环
    # 后续复用，避免每个 provider 实例独立创建/销毁连接
```

✅ 进程级复用 asyncio loop，通过 `_run_async()` 将协程投递到共享 loop 并阻塞等待结果。

---

## 优雅退出（Graceful Shutdown）

```python
def shutdown(self) -> None:
    self._shutting_down.set()       # 1. 阻止新任务入队
    self._retain_queue.put(_WRITER_SENTINEL)  # 2. 通知 writer 退出
    if writer and writer.is_alive():
        writer.join(timeout=10.0)   # 3. 等待 writer 完成已有任务

    # 4. 可选：清空当前 group 的所有数据（测试/开发场景）
    if self._auto_clear_on_shutdown and self._group_id:
        # 使用独立临时客户端，避免影响主连接关闭
        _run_async(_clear(), timeout=15)

    self._close_client()            # 5. 关闭 MCP 连接

def _register_atexit(self):
    atexit.register(self._atexit_shutdown)  # 进程退出时自动调用
```

✅ `atexit` 注册确保进程异常退出时也不会丢数据。
✅ `auto_clear_group_on_shutdown` 支持测试/开发场景下一键重置知识图谱（默认 `false`，生产安全）。

---

## 配置加载

> 参考 mem0 provider 的模式：环境变量 → JSON 文件覆盖。

```python
def _load_config() -> dict:
    config = {
        'mcp_url': os.environ.get('GRAPHITI_MCP_URL', _DEFAULT_MCP_URL),
        'group_id': os.environ.get('GRAPHITI_GROUP_ID', ''),
        # ... 所有配置项从环境变量读取默认值 ...
    }
    # $HERMES_HOME/graphiti.json 覆盖
    config_path = get_hermes_home() / 'graphiti.json'
    if config_path.exists():
        file_cfg = json.loads(config_path.read_text())
        config.update({k: v for k, v in file_cfg.items() if v})
    return config
```

✅ 环境变量提供默认值，JSON 文件覆盖。符合 Profile Isolation（使用 `get_hermes_home()`）。

---

## 设计参考与跨 Provider 对比

### 模式参考

本实现综合参考了 Hermes 官方的 **mem0** 和 **hindsight** 两个 provider，并与 **Honcho** 的命名风格对齐：

| 特性 | 来源 | 说明 |
|------|------|------|
| 熔断器 | mem0 | 连续失败后暂停，区分客户端/后端错误 |
| 单写入者队列 | hindsight | 串行化写入，`sync_turn()` 非阻塞 |
| 轮次缓冲 | hindsight | 累积多轮后批量发送，节省 LLM token |
| 共享事件循环 | hindsight | 进程级复用 asyncio loop |
| atexit 注册 | 两者 | 进程退出时排空队列 |
| 配置加载模式 | mem0 | 环境变量 → JSON 覆盖 |
| 工具命名 | Honcho | `{prefix}_{search/profile/reasoning/context/conclude}` |
| 主动健康探测 | 自研 ✨ | 熔断期间 `get_status` 探测，缩短恢复时间 |
| MCP 协议 | 自研 | Graphiti 是唯一通过 MCP streamable HTTP 通信的 provider |

### 工具设计对比

```
高度聚合 ──────────────────────────────────────▶ 高度细分

Holographic  Hindsight  Mem0    Graphiti  OpenViking  RetainDB
  (2 工具)    (3 工具)  (5 工具)  (6 工具)   (6 工具)   (10 工具)
  单工具+action           CRUD直译  认知语义  认知语义    CRUD+文件
```

| 设计风格 | 代表 | 特点 |
|----------|------|------|
| **单工具 + action 枚举** | Holographic | 所有操作通过一个 `fact_store` tool + `action` 参数承载 |
| **CRUD 直译** | Mem0, Supermemory | 后端 API 的直接映射：add/search/update/delete |
| **认知语义映射** | Honcho, Graphiti, Hindsight, OpenViking | 把操作映射为人类认知行为：search/profile/reason/conclude |
| **领域扩展** | RetainDB, OpenViking | 在 memory CRUD 之外附加文件/资源管理 |

Graphiti 选择了 **认知语义映射** 路线，与 Honcho 高度一致，确保 LLM 能直观理解每个工具的用途。

---

## 运行模式

| 模式 | `get_tool_schemas()` | 上下文注入 | 说明 |
|---|---|---|---|
| `tools` | 返回全部 6 个 | 无 | 默认模式 |
| `context` | 返回空列表 | 有（`prefetch`） | 纯上下文注入 |
| `hybrid` | 返回全部 6 个 | 有（`prefetch`） | 两者兼备 |

---

## 插件入口

```python
def register(ctx: Any) -> None:
    ctx.register_memory_provider(GraphitiMemoryProvider())
```

✅ Hermes 通过 `importlib` 加载模块，调用 `register(ctx)` 注册 provider。

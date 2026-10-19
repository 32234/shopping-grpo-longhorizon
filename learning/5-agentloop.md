# 05 · AgentLoop：veRL 骨架上的四个钩子

这篇只讲一个文件：`src/shopping_grpo/training/grpo/adapter/agent_loop.py`。
它是 ShopSimulator 和 veRL 之间唯一的轨迹驱动器。

目标：看完能说清"它相对原生 `ToolAgentLoop` 改了什么、为什么必须改、替换时哪些契约不能破"。

GRPO 全景见 04 篇。

---

## 1. 它不做什么

`ShoppingToolAgentLoop` **没有重写多轮循环**。下面这些全是 veRL 父类的：

| 环节 | 谁做 |
|---|---|
| prompt 编码 | `ToolAgentLoop` |
| 调 vLLM 生成 | `ToolAgentLoop` |
| `qwen3_coder` 解析 tool call | `ToolAgentLoop` |
| 把 tool 响应拼回 token 序列 | `ToolAgentLoop` |
| `response_ids` / `response_mask` / `response_logprobs` | `ToolAgentLoop` |
| turn 计数、`max_assistant_turns` | `ToolAgentLoop` |
| tokenizer | `ToolAgentLoop` |

所以这个类只有 **336 行**，其中 `__init__` 占 79 行（近四分之一），
四个 override 合计约 190 行。

---

## 2. 它做什么：四个 override

```python
_handle_generating_state()        # 生成前：上下文预算 + 每轮 max_tokens
_call_tool()                      # 工具后：observation 投影
_handle_processing_tools_state()  # 处理工具前：拒绝并行调用
run()                             # 整条轨迹：租约 + 奖励 + 诊断
```

### 2.1 `_handle_generating_state` —— 上下文预算

默认配置（`context_compaction_enable: false`）走这条：

```python
if current_input_tokens > (
    self.context_window_tokens
    - self.context_generation_reserve_tokens
    - self.context_safety_margin_tokens
):
    state["terminate"] = True
    state["termination_reason"] = "context_hard_limit_exceeded"
    state["infrastructure_invalid"] = True
    return AgentState.TERMINATED
```

即 **24576 − 512 − 512 = 23552** 是硬上限。超了不是截断、不是压缩，是**判这条轨迹无效**。

开启压缩（`context_compaction_enable: true`）时改走 `compact_token_trajectory`：
只删**完整的** assistant+tool token 组，保留固定 prompt 和最近 `preserve_recent_groups` 组。
删不动仍超预算 → `context_budget_exhausted` + 判无效。

还有一个边缘情况：

```python
if stats is not None and stats.removed_tokens:
    if agent_data.routed_experts is not None:
        # routed experts 的额外状态无法随 token 一起裁剪
        terminate("context_compaction_unsupported_routed_experts")
        infrastructure_invalid = True
```

**每轮输出上限**：

```python
if "max_tokens" in bounded_sampling_params:
    bounded_sampling_params["max_tokens"] = min(
        int(bounded_sampling_params["max_tokens"]),
        self.context_generation_reserve_tokens,      # 512
    )
```

防止模型连续输出纯文本耗尽窗口。

### 2.2 `_call_tool` —— observation 投影的插入点

```python
response, reward, step = await super()._call_tool(tool_call, tools_kwargs, agent_data)

state = current_runtime_state.get()
if state is None:
    return response, reward, step

raw_observation = state.pop("_pending_raw_observation", None)
if raw_observation is None:
    return response, reward, step

try:
    parameters = json.loads(tool_call.arguments or "{}")
    visible_observation, projection = project_observation(
        tool_name=tool_call.name,
        observation=raw_observation,
        parameters=parameters,
        count_tokens=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False)),
        token_budget=self.observation_token_budget,                    # 1536
        detail_token_budget=self.observation_detail_token_budget,      # 4096
        generic_token_budget=self.observation_generic_token_budget,    # 768
        search_top_k=self.observation_search_top_k,                    # 20
    )
    if len(visible_observation) > self.max_tool_response_length:       # 16384 字符兜底
        raise ObservationProjectionError(
            "projected observation exceeds veRL character fallback limit"
        )
except Exception as exc:
    state["terminate"] = True
    state["termination_reason"] = "observation_projection_failed"
    state["infrastructure_invalid"] = True
    state["observation_footer_failures"] += 1
    response.text = "Error: observation projection failed; trajectory is invalid."
    return response, 0.0, step

# 成功：覆盖父类将要 tokenize 的文本
state["latest_observation_raw"] = raw_observation
state["latest_observation"] = visible_observation
record_observation_projection(state, projection.to_dict())
if isinstance(step, dict):
    step["projection"] = projection.to_dict()
response.text = visible_observation
return response, reward, step
```

**时序是这个方法的全部意义**：父类会在 `_call_tool` 返回后把 `response.text`
编成 token。所以在这里改写 `response.text`，模型实际看到的就是压缩后的文本。

失败时**不降级**——直接把轨迹标记为无效，而不是把超长原始 observation 塞进去。

### 2.3 `_handle_processing_tools_state` —— 串行化

```python
if runtime_state is not None and len(agent_data.tool_calls) > 1:
    terminate("parallel_tool_calls")
    infrastructure_invalid = True
    return AgentState.TERMINATED

next_state = await super()._handle_processing_tools_state(agent_data)

# 让工具内部设置的 terminate 生效（守卫连拒 3 次、max_steps）
if runtime_state.get("terminate"):
    return AgentState.TERMINATED
return next_state
```

**为什么要拒并行**：一回合点两个目标，等价于在旧 observation 上批量操作，
第二个动作的合法性无从判断。SFT 采集那边是"截断保留第一个"，RL 这边更严格。

### 2.4 `run` —— 生命周期 + 奖励

```python
async def run(self, sampling_params, **kwargs):
    task_id = task_id_from_kwargs(kwargs)                  # 从 extra_info 取

    session = ShopSimulatorSession(
        base_url=self.base_url,
        timeout=self.timeout,
        max_steps=self.max_steps,
        required_environment_version=self.required_environment_version,
        env_factory=self.env_factory,
    )
    state = await session.start(task_id)                   # 租槽位 + 绑 ContextVar

    try:
        output = await super().run(sampling_params, **kwargs)

        if not state["done"] and not state["error"]:
            state["error"] = "assistant_finished_without_environment_done"
            state["termination_reason"] = state["error"]
            state["terminate"] = True

        breakdown = apply_reward_length_shaping(
            reward_breakdown(state), state,
            enabled=self.reward_length_shaping_enable,      # 默认 false
            soft_threshold=self.reward_length_soft_threshold,
            penalty_per_step=self.reward_length_penalty_per_step,
            max_penalty=self.reward_length_max_penalty,
        )
        output.reward_score = (
            float(breakdown["total"])
            if self.reward_mode == "constraint_aware"
            else terminal_reward(state, mode=self.reward_mode)
        )
        output.extra_fields["shopping"] = { ... }           # 见 §7
        if session.trace_target is not None:
            output.extra_fields["trace_target"] = session.trace_target
        return output
    finally:
        await session.close()                               # 释放槽位 + 解绑 ContextVar
```

---

## 3. Session：租约 + ContextVar

`adapter/session.py`。一条轨迹一个。

```python
async def start(self, task_id):
    self.env = self.env_factory(base_url=self.base_url, timeout=self.timeout)
    self.env.include_trace_target = True                 # ① 申请 gold（给 TRACE）

    initial = await asyncio.to_thread(self.env.reset, int(task_id))   # ② 阻塞调用丢线程
    # POST /api/shop_agent {"action":"reset","idx":65}
    # ← 服务端建 user_sessions["slot-3-65"]，返回 env_idx

    self.state = make_runtime_state(task_id=task_id, max_steps=self.max_steps)

    if initial.get("_trace_target") is not None:          # ③ 解析 gold 目标
        self.trace_target = canonical_purchase_target(
            private_target.get("asin"), private_target.get("options"))

    if required_environment_version != actual:            # ④ 版本闸
        await release(); raise RuntimeError(...)

    self.state["latest_observation"] = render_structured_observation(
        initial["observation_state"])

    self._environment_token = current_environment.set(self.env)     # ⑤ 发布
    self._state_token = current_runtime_state.set(self.state)
    return self.state
```

`close()` 反向：`release_one` → `ContextVar.reset(token)` → 清引用。
**`reset` 用 token 而不是 `set(None)`**，才能精确恢复到进入前的值。

**`asyncio.to_thread` 是必须的**：`ShopAgentEnv` 用阻塞的 `urllib`，直接 await 会卡住事件循环。

---

## 4. runtime state：训练端的账本

`make_runtime_state(task_id, max_steps)` 返回 dict，不是对象。

| 类别 | 字段 |
|---|---|
| 轨迹结构 | `steps[]`, `done`, `terminate`, `termination_reason`, `task_id`, `max_steps` |
| 终局 | `terminal_result`, `final_reward`, `reward_version`, `reward_type`, `reward_valid`, `reward_unverifiable`, `reward_detail` |
| 错误 | `infrastructure_invalid`, `error` |
| 动作统计 | `action_attempt_count`, `repeat_action_count`, `recent_action_signatures` |
| 守卫 | `guard_rejection_count`, `guard_rejection_reason_counts`, `consecutive_guard_rejections` |
| 上下文 | `context_compactions`, `context_tokens_removed`, `context_max_input_tokens` |
| observation | `latest_observation`, `latest_observation_raw`, `observation_*` 各项统计 |

**谁读写**：

```
ShopSimulatorTool.execute()   读 latest_observation（守卫）
                              写 steps / done / final_reward / reward_detail / guard_*
agent_loop._call_tool()       写 latest_observation（投影后）+ observation_* 统计
agent_loop.run()              读全部 → reward_breakdown → output.reward_score
```

**关键**：这份 state **从不持有 gold**。`trace_target` 单独挂在 `session` 上，
最后放进 `output.extra_fields["trace_target"]`。

### 4.1 两套 state 不要混淆

| | 服务端 `user_sessions["slot-3-65"]` | 训练端 `runtime_state` |
|---|---|---|
| 谁建 | `SimServer.receive` | `make_runtime_state` |
| 存什么 | goal、keywords、page、asin、options、selected_price、progress_tracker | steps、reward、guard 计数、projection 统计 |
| 谁读 | `evaluate_purchase` / `record_progress` | `reward_breakdown` / 诊断聚合 |
| 生命周期 | reset → release | `session.start` → `session.close` |

---

## 5. ContextVar：为什么不用实例属性

因为 Tool 是 veRL 工具注册表按 `configs/tools.json` 构造的，**拿不到 loop 实例**：

```python
class ShopSimulatorTool(BaseTool):
    def __init__(self, config, tool_schema):     # 只有配置
        ...

    async def execute(self, instance_id, parameters, **kwargs):
        env = current_environment.get()          # ← 唯一获取途径
        state = current_runtime_state.get()
        if env is None or state is None:
            raise RuntimeError("ShopSimulator tool executed without a trajectory-local interaction state")
```

### 5.1 并发隔离

`ContextVar` 是 **Task 级**隔离：每个 `asyncio.Task` 创建时复制父上下文。

```
Task A: set(env_A)  → 只有 A 的 get() 看到
Task B: set(env_B)  → 只有 B 的 get() 看到
```

三层隔离保证并发安全：

| 层 | 机制 |
|---|---|
| 协程 | `ContextVar`（Task 级） |
| 客户端 | 每条轨迹一个 `ShopAgentEnv` 实例，`env_idx` 存在实例上 |
| 服务端 | session key = `slot-{i}-{task_id}`，不同槽位不同条目 |

即使两条轨迹跑同一个 `task_id`，也会租到不同槽位（`slot-3-65` vs `slot-5-65`）。

### 5.2 线程里也对

`asyncio.to_thread` 会复制当前上下文进线程（Python 3.9+），所以线程里的 `get()` 也正确。
不过 Tool 通常在协程里先 `get()` 再丢线程，这条是额外保险。

### 5.3 两个必须配对

1. `set()` 返回 token，`reset(token)` 恢复——不要 `set(None)`
2. `close()` 必须在 `finally` 里，否则同一 Task 内连续跑轨迹会读到上一条的残留

---

## 6. Tool 侧：为什么恒返回 0.0

`adapter/tools.py` 的 `execute()` **所有返回路径的 reward 都是 `0.0`**：

```python
return ToolResponse(text="Error: environment is already terminal; ..."), 0.0, {}
return ToolResponse(text="Error: maximum executed tool steps reached."), 0.0, {...}
return ToolResponse(text="Reasoning recorded. ..."), 0.0, step
return ToolResponse(text=f"Error: action guard rejected ..."), 0.0, {...}
return ToolResponse(text=f"Error: ShopSimulator tool execution failed: {exc}"), 0.0, {...}
return ToolResponse(text="Environment terminated."), 0.0, step       # ← 终局也是 0
return ToolResponse(text=observation), 0.0, step
```

**奖励只在 `AgentLoop.run()` 里注入。** 这是"为什么不能用原生 `tool_agent`"的第一层原因。

`release()` 也是空实现：

```python
async def release(self, instance_id, **kwargs):
    # veRL 0.8 会在每次 tool call 后执行 release；真正的环境租约由 AgentLoop 释放。
    del instance_id, kwargs
```

第二层原因：原生 loop 不会调 `session.close()`，8 个槽位漏完就永久卡死。

---

## 7. `extra_fields["shopping"]`：动态采样的输入

`agent_loop.py:282`。这是训练诊断的唯一出口，也是动态采样的输入。

| 分组 | 字段 |
|---|---|
| 基本 | `task_id`, `steps`, `actions[]`, `done`, `termination_reason`, `error`, `infrastructure_invalid` |
| 奖励 | `reward_mode`, `reward_version`, `reward_type`, `reward_valid`, `reward_unverifiable`, `reward`(完整 breakdown), `overlong` |
| 动作 | `action_attempts`, `repeat_actions`, `guard_rejections`, `guard_rejection_reasons`, `guard_rejections_after_truncation`, `action_attempts_after_truncation` |
| 上下文 | `context_compactions`, `context_tokens_removed`, `context_max_input_tokens` |
| observation | `observation_projection_count`, `observation_truncated_count`, `observation_raw_tokens`, `observation_visible_tokens`, `observation_max_*`, `observation_visible_asin_count`, `observation_visible_button_count`, `observation_any_truncated`, `observation_footer_failures` |

### 7.1 动态采样必须读到的三个字段

`dynamic_sampling.py:186 extract_shopping_group_signals`：

```python
info["reward"]["terminal_utility"]      # 有限数
info["reward"]["purchase_success"]      # bool/int/float
info["infrastructure_invalid"]          # 必须存在
```

缺任一 → `ValueError`，训练停。`reward_unverifiable` 和 `overlong` 用于构造 `sampling_invalid`。

---

## 8. 失败分类：哪些判无效，哪些只是结束

| 情况 | termination_reason | infrastructure_invalid | reward |
|---|---|---|---|
| 上下文硬上限 | `context_hard_limit_exceeded` | ✅ | 0 |
| 压缩装不下 | `context_budget_exhausted` | ✅ | 0 |
| 压缩与 routed_experts 冲突 | `context_compaction_unsupported_routed_experts` | ✅ | 0 |
| 投影失败 / 超 16384 字符 | `observation_projection_failed` | ✅ | 0 |
| 一回合多个 tool call | `parallel_tool_calls` | ✅ | 0 |
| 工具抛异常 | `tool_error:{类型}:{消息}` | ✅ | 0 |
| 终局 `over` 不是 True | `invalid_terminal_result` | ✅ | 0 |
| `reward_detail` schema 不符 | `invalid_terminal_reward_detail:{原因}` | ✅ | 0 |
| 守卫连拒 3 次 | `too_many_guard_rejections` | ❌ | 0 |
| 满 35 步 | `max_steps` | ❌ | 0 |
| 父类结束但环境没 done | `assistant_finished_without_environment_done` | ❌ | 0 |

**判无效的轨迹会毒化整个 group**（动态采样整组丢弃）；只是结束的轨迹会按实际 reward 参与。

---

## 9. 构造函数校验

`__init__` 里逐项拒绝非法配置：

```python
if not 0 < context_input_budget_tokens <= window - reserve - margin:  raise
if context_preserve_recent_groups < 1:                                raise
if min(三个 observation 预算) < 64:                                    raise
if observation_search_top_k < 1:                                      raise
if reward_mode not in {"native", "constraint_aware"}:                  raise
if reward_length_shaping_enable and reward_mode != "constraint_aware": raise
if not 0 < reward_length_soft_threshold < max_steps:                  raise
if min(penalty_per_step, max_penalty) < 0:                            raise
```

---

## 10. 写一个自定义 AgentLoop

### 10.1 最小可跑版本（约 25 行）

```python
from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop

class MyLoop(ShoppingToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        shopping = output.extra_fields["shopping"]
        new_total = my_formula(shopping["reward"], shopping)

        output.reward_score = new_total
        shopping["reward"]["terminal_utility"] = new_total     # 必须同步
        shopping["reward"]["total"] = new_total                # 必须同步
        return output
```

配置两处：

```yaml
# configs/agent_loop.yaml
- name: my_loop
  _target_: mypkg.MyLoop
  ...（其余参数照抄 shopping_tool_agent）

# configs/grpo.yaml
default_agent_loop: my_loop
```

**`name` 和 `default_agent_loop` 必须一致**，否则 veRL 注册表找不到。

### 10.2 从零写一个：必须自己做的四件事

如果不继承 `ShoppingToolAgentLoop` 而是直接继承 `ToolAgentLoop`：

| 必须做 | 不做会怎样 |
|---|---|
| 创建 `ShopSimulatorSession` 并 `start(task_id)` | Tool 拿不到 env，第一次调用就 `RuntimeError` |
| `finally: await session.close()` | 槽位泄漏，8 条后卡死 |
| `output.reward_score = ...` | 奖励恒为 0，动态采样丢掉所有组 |
| 填 `extra_fields["shopping"]["reward"]["terminal_utility"]` / `purchase_success` / `infrastructure_invalid` | 动态采样 `ValueError` |

还需要（否则长轨迹必炸）：上下文检查、observation 投影、并行调用拒绝。

### 10.3 不能做的事

| 想法 | 为什么不行 |
|---|---|
| 用 `default_agent_loop=tool_agent` | Tool 拿不到 ContextVar；租约无人释放；奖励恒 0 |
| 用 `reward_model.custom_reward_function` | 只能看到文本 + `extra_info`（只有 task_id），拿不到 asin/options/价格 |
| 改环境 Reward v3 常量 | 三重锁：`DEFAULT_REWARDS` / `environment.json` / manifest runtime hash |

---

## 11. `configs/agent_loop.yaml` 全表

```yaml
- name: shopping_tool_agent
  _target_: shopping_grpo.training.grpo.adapter.agent_loop.ShoppingToolAgentLoop
  base_url: ${oc.env:SHOPSIM_BASE_URL,http://127.0.0.1:5700}
  timeout: 60
  max_steps: 35
  required_environment_version: shopsimulator-environment-v2.1
  reward_mode: constraint_aware

  # 长度惩罚（默认关，且要求 constraint_aware）
  reward_length_shaping_enable: ${oc.env:SHOPPING_LENGTH_SHAPING_ENABLE,false}
  reward_length_soft_threshold: ${oc.env:SHOPPING_SOFT_LENGTH_THRESHOLD,20}
  reward_length_penalty_per_step: ${oc.env:SHOPPING_LENGTH_PENALTY_PER_STEP,0.01}
  reward_length_max_penalty: ${oc.env:SHOPPING_MAX_LENGTH_PENALTY,0.15}

  # 上下文
  context_window_tokens: 24576
  context_generation_reserve_tokens: 512
  context_safety_margin_tokens: 512
  context_input_budget_tokens: 16384      # 只在开压缩时用
  context_preserve_recent_groups: 1
  context_compaction_enable: false

  # observation 投影
  observation_token_budget: 1536          # search_results
  observation_detail_token_budget: 4096   # product_detail
  observation_generic_token_budget: 768   # 其他
  observation_search_top_k: 20
```

`max_tool_response_length: 16384` 不在这里，来自 `grpo.yaml` 的
`multi_turn.max_tool_response_length`，父类持有，子类读取。

---

## 12. 排查清单

| 症状 | 先查 |
|---|---|
| 训练奖励全是 0 | `default_agent_loop` 是否指向 `ShoppingToolAgentLoop`；`reward_mode` |
| 动态采样报 `ValueError` | `extra_fields["shopping"]` 三个必填字段 |
| 跑到第 9 条轨迹卡住 | 槽位泄漏——`session.close()` 是否在 `finally` |
| `RuntimeError: ... without a trajectory-local interaction state` | Session 没启动 / ContextVar 没设 |
| 大量 `context_hard_limit_exceeded` | observation 预算太松，或投影没生效 |
| 大量 `observation_projection_failed` | `max_tool_response_length` 太小，或页面 footer 缺失 |
| `parallel_tool_calls` | 模型输出格式问题（`qwen3_coder` parser 或 prompt） |
| `too_many_guard_rejections` | 模型没按当前页操作（守卫规则见 02 篇） |

---

## 13. 一句话

> **`ShoppingToolAgentLoop` 不重写循环，只在四个位置插入约束：生成前查上下文、工具后压 observation、处理前拒并行、结束时算奖励并释放租约。**
> 它之所以不可替换，是因为 **环境租约、轨迹状态、奖励注入三件事都只存在于 `run()` 里**——
> 原生 `ToolAgentLoop` 既没有租约也没有奖励，`custom_reward_function` 既拿不到状态也没有 asin/options。

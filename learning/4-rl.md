# 04 · GRPO：把 ShopSimulator 接进 veRL

这篇讲训练侧：**SFT 之后的模型怎么在真实环境里采样、怎么把 Reward v3 变成梯度**。
目标是看完能自己拼出命令、知道每个开关在哪、知道哪些能换哪些不能。

前置：01 讲 SFT 数据，02 讲环境，03 讲 SFT 训练。

---

## 1. 一句话边界

```
verl==0.8.0（pip 安装，不 vendor）
  + 1 个 SHA-256 校验的补丁（只改 verl/trainer/ppo/ray_trainer.py）
  + 项目侧 4 个适配模块
```

**veRL 本身没改。** 补丁存在的唯一理由：veRL 0.8 的 `ray_trainer.py` 没有多轮 AgentLoop 的
有界重采样，而且 `global_steps` 会在真正更新前就自增。

---

## 2. 训练数据从哪来

**GRPO 和 SFT 用的是两套完全不相交的 task。**

| | SFT | GRPO |
|---|---|---|
| 文件 | `data/sft_pure_v4/all.jsonl` | `data/grpo/train.parquet` / `validation.parquet` |
| task 数 | 1,192（1,073 train / 119 dev） | **1,000 train / 50 validation** |
| 行内容 | 完整对话 messages | `prompt`（chat messages）+ `extra_info.task_id` |
| 用途 | 监督示范 | 在线采样 |

### 2.1 怎么选出来的

`data/grpo/metadata.json`：

```json
"provenance": {
  "excluded": "新 SFT 全部任务 + 冻结评测 200 题",
  "reason": "保持 SFT 任务与 GRPO 任务池不相交（沿用原始实验设计）",
  "rebuilt": true,
  "seed": 20260808,
  "note": "未做难度探测，缺少 probe_steps / length_bucket 字段"
}
```

算术：

```
ShopSimulator goal 池                    23,421   (task_id 0..23420)
  − 旧 SFT 的 1,000 个 task_id（data/sft 的 800+200）
  − Final-200 的 200 个
  ────────────────────────────────────────
  候选                                    22,221
  → seed 20260808 抽 1,000 train + 50 val
```

**没有做难度或长度分层**——metadata 自己写明"未做难度探测"。

### 2.2 parquet 里有什么

从 parquet footer 的 Thrift 元数据能读到的列名：

```
prompt          chat messages（system + user instruction）
extra_info      {"task_id": 53}
data_source
index
```

`data/grpo/train.jsonl`（一行 `{"task_id": 53}`）只是人看的索引，**不是训练输入**。

**关键差异**：GRPO 的 prompt 是**建 parquet 时烤进去的**，不是运行时从
`env.reset` 拼的。SFT 采集相反——它用 `env.reset` 返回的 `instruction` 现场拼 messages。
所以如果建 parquet 时的 system prompt 和 `evaluation/rollout.py:34` 不一致，
GRPO 训练和评测会在 prompt 层漂移。**首次运行前应抽一行 `prompt[0]["content"]` 核对。**

### 2.3 实测重叠

```
旧 data/sft/ (1,000) ∩ grpo_train = 0
v4         (1,192)   ∩ grpo_train = 6     ← 7682, 8993, 11298, 12380, 14953, 17219
eval       (200)     ∩ grpo_train = 0
eval       (200)     ∩ grpo_val   = 0
```

**为什么 v4 漏 6 个**：GRPO 池在 `2923a34` 重建，当时排除的是旧 `data/sft/` 的 1,000 个；
后来 `sft_pure_v4` 合并时从 merged 包引入了 **192 个全新 task**，这些从未参与 exclusion。

这 6 个在难度标注里都是 `hard`，只在 stage C（最后一轮、LR 5e-5）过一遍。
影响面是 6/1,000；要严格满足 task-disjoint 契约，需要重新抽池并重算 parquet 哈希。

---

## 3. 命令：只有四个 Hydra 参数

`scripts/grpo.sh` → `scripts/train_grpo.py`，拼出来是：

```bash
cd <repo>
.venv/bin/python -m verl.trainer.main_ppo \
  --config-path=<repo>/configs \
  --config-name=grpo \
  trainer.logger=[console] \
  trainer.experiment_name=shopping-agent-grpo
```

`--` 之后的内容原样追加成 Hydra override：

```bash
bash scripts/grpo.sh -- trainer.total_training_steps=20 trainer.save_freq=10
```

### 2.1 环境变量（全部由 `train_grpo.py:84` 注入）

```bash
PYTHONPATH=<repo>/src
SHOPPING_GRPO_ROOT=<repo>
SHOPPING_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1
SHOPPING_ENV_MANIFEST=<repo>/data/environment.json

GRPO_MODEL_PATH=<repo>/outputs/models/sft-merged
GRPO_TRAIN_FILE=<repo>/data/grpo/train.parquet
GRPO_VAL_FILE=<repo>/data/grpo/validation.parquet
GRPO_OUTPUT_DIR=<repo>/outputs/models/grpo
SHOPPING_GRPO_DIAGNOSTICS_PATH=<repo>/outputs/models/grpo/training_diagnostics.jsonl

SHOPSIM_BASE_URL=http://127.0.0.1:5700
SHOPPING_AGENT_LOOP_CONFIG=<repo>/configs/agent_loop.yaml
SHOPPING_TOOL_CONFIG=<repo>/configs/tools.json
GRPO_CONFIG_NAME=grpo
```

被 `configs/grpo.yaml` 用 `${oc.env:...}` 读走：

```yaml
data:
  train_files: ${oc.env:GRPO_TRAIN_FILE}
  val_files: ${oc.env:GRPO_VAL_FILE}
actor_rollout_ref:
  model:
    path: ${oc.env:GRPO_MODEL_PATH}
  rollout:
    agent:
      agent_loop_config_path: ${oc.env:SHOPPING_GRPO_ROOT}/configs/agent_loop.yaml
    multi_turn:
      tool_config_path: ${oc.env:SHOPPING_GRPO_ROOT}/configs/tools.json
trainer:
  default_local_dir: ${oc.env:GRPO_OUTPUT_DIR}
```

> `SHOPPING_AGENT_LOOP_CONFIG` 是**死变量**——全仓没有读者，路径实际靠
> `SHOPPING_GRPO_ROOT` 拼。`SHOPPING_TOOL_CONFIG` 只在 preflight 里被读。

### 2.2 `--model` 默认值是旧的

```python
DEFAULT_MODEL = ROOT / "outputs/models/sft-merged"
```

当前配方应该显式传：

```bash
bash scripts/grpo.sh --model outputs/models/sft-curriculum/stage-c/merged
```

启动前 `build_command` 会校验：`config.json` 存在、有权重文件
（`model.safetensors` / index / `pytorch_model.bin` / index 之一）、输出目录为空。

---

## 4. AgentLoop 在哪指定（三层）

**不在 `train_grpo.py` 里。**

**① `configs/grpo.yaml:70-73` —— 名字 + 文件**

```yaml
actor_rollout_ref:
  rollout:
    agent:
      num_workers: 8
      default_agent_loop: shopping_tool_agent
      agent_loop_config_path: ${oc.env:SHOPPING_GRPO_ROOT}/configs/agent_loop.yaml
```

**② `configs/agent_loop.yaml:1-2` —— 类**

```yaml
- name: shopping_tool_agent
  _target_: shopping_grpo.training.grpo.adapter.agent_loop.ShoppingToolAgentLoop
  base_url: ${oc.env:SHOPSIM_BASE_URL,http://127.0.0.1:5700}
  ...
```

**③ `train_grpo.py:88` —— 提供 `${oc.env:SHOPPING_GRPO_ROOT}`**

```
name 必须和 default_agent_loop 一致（两处都要改）
_target_ 是 Hydra 实例化路径
下面的参数全部作为构造参数传入
```

---

## 5. 四个适配模块

| 模块 | 继承/形式 | 职责 |
|---|---|---|
| `adapter/agent_loop.py` | `ToolAgentLoop` | 轨迹生命周期：上下文检查、投影、串行工具、奖励注入、释放租约 |
| `adapter/tools.py` | `BaseTool` | 守卫 → 动作翻译 → `env.step`（丢线程）→ 校验 Reward v3 |
| `adapter/session.py` | 普通类 | 环境租约 + `ContextVar` 绑定 + `finally` 释放 |
| `adapter/runtime.py` | 纯函数 | Reward v3 → 训练信号；严格 schema 校验 |

### 4.1 `ShoppingToolAgentLoop` 的四个插点

```python
_handle_generating_state()      # 每轮生成前
_call_tool()                    # 每次工具返回后
_handle_processing_tools_state()# 每回合处理工具调用前
run()                           # 整条轨迹的入口/出口
```

**失败一律判无效，不降级：**

| 情况 | termination_reason | infrastructure_invalid |
|---|---|---|
| 上下文超过 `24576-512-512` | `context_hard_limit_exceeded` | True |
| 压缩后无法安全裁剪（routed_experts 不为空） | `context_compaction_unsupported_routed_experts` | True |
| 同回合 >1 个 tool call | `parallel_tool_calls` | True |
| 投影失败 / 超过 16384 字符 | `observation_projection_failed` | True |
| 父类结束但环境未 done | `assistant_finished_without_environment_done` | —（reward 0） |

**上下文超限时 GRPO 直接判无效，不压缩**（`context_compaction_enable: false`）。

### 4.2 `ShopSimulatorTool` 恒返回 0.0

这是最容易踩的一点。**所有返回路径的 reward 都是 `0.0`**，包括终局：

```python
if step["done"]:
    state["done"] = True
    state["final_reward"] = step["reward"]
    ...
    return ToolResponse(text="Environment terminated."), 0.0, step   # ← 0.0
```

真正的奖励由 `AgentLoop.run()` 写进 `output.reward_score`。

### 4.3 `validate_reward` 是严格 schema 闸

终局时校验环境回传的 `reward_detail`：

```python
reward_version == "shopsimulator-reward-v3"
reward_type ∈ 9 种合法值
termination_reason == reward_type
reward_valid / purchase_success / sampling_invalid 都是 bool
(reward_type == "reward_unverifiable") == (not reward_valid)
sampling_invalid == (not reward_valid)
terminal_utility == step["reward"]                     # ← 交叉验证
hard_gates[*].status ∈ {pass, fail, unverifiable}
hard_gates[*].passed == (status == "pass")
hard_gates[*].verifiable == (status != "unverifiable")
weighted_score / evidence_coverage ∈ [0, 1]
dimension_scores 四项都在 [0, 1]
```

任一不符 → `infrastructure_invalid`。

### 4.4 Session 用 `ContextVar` 而非全局

```python
current_environment: ContextVar
current_runtime_state: ContextVar
```

并发轨迹不会串状态。`include_trace_target=True`（只有 GRPO 会打开），
reset 时拿到 `_trace_target = {"asin": goal["asin"], "options": goal["goal_options"]}`。

---

## 6. 奖励从哪来

### 5.1 veRL 配置里只有一个开关

```yaml
# configs/grpo.yaml:123
reward_model:
  enable: false
```

配置里**没有** `custom_reward_function`，也没有 `reward_manager`。veRL 不会自己算奖励。

### 5.2 真正来源

```python
# agent_loop.py:277
output.reward_score = (
    float(breakdown["total"])
    if self.reward_mode == "constraint_aware"
    else terminal_reward(state, mode=self.reward_mode)
)
```

`reward_mode` 在 `configs/agent_loop.yaml:11`：

```yaml
reward_mode: constraint_aware
```

只有两档（`agent_loop.py:101` 强制）：

| 值 | 行为 |
|---|---|
| `constraint_aware` | `reward_breakdown(state)["total"]`，带完整诊断 |
| `native` | `state["final_reward"]`，infra 无效时 0 |

### 5.3 完整链路

```
configs/agent_loop.yaml: reward_mode
        ↓
AgentLoop.run() → output.reward_score
        ↓
runtime.reward_breakdown(state)
        total = terminal_utility = state["final_reward"]
        ↓
ShopSimulatorTool.execute() → state["final_reward"] = step["reward"]
        ↓
POST /api/shop_agent {"action":"interact", ...}
  返回 {"done": true, "reward": 1.0, "reward_detail": {...}}
        ↓
SimServer.done() → evaluate_purchase(...)
  用 configs/environment.json 的 "reward" 常量表
```

**奖励数值本身在 `environments/ShopSimulator/shop_env/configs/environment.json`**，
`engine/config.py:71` 启动时逐值比对代码常量。

### 5.4 训练只用一个标量

`reward_breakdown` 产出一大堆字段（`r_type` / `r_price` / `r_att` / `r_option` /
`match_score` / `purchase_success` / `repeat_action_rate` …），但**只有 `total`
进 optimizer**，其余是诊断。

---

## 7. 动态采样

```yaml
shopping_dynamic_sampling:
  enable: true
  metric: seq_reward
  max_num_gen_batches: 3
  max_consecutive_skipped_updates: 10
  reward_tolerance: 1.0e-8
```

### 6.1 丢弃规则

| 情况 | 处理 |
|---|---|
| 组内 `terminal_utility` 全相同（差 ≤ tolerance） | 丢整组 |
| 组内任一成员 `sampling_invalid` | **丢整组** |

第二条很关键：**一条轨迹的环境故障会毒化整个 group**。

`sampling_invalid` 的来源：

```python
info["infrastructure_invalid"]      # 上下文超限、投影失败、并行调用、环境异常
info["reward_unverifiable"]         # 价格算不出来
info["overlong"]                    # 开启长度惩罚时
reward["sampling_invalid"]
```

### 6.2 边界

```
最多重采 3 批 → 还凑不够 → 记 skipped_update，不推进
连续跳 10 次 → 抛错中止
```

### 6.3 补丁为什么必须

veRL 0.8 的 `ray_trainer.py` 缺两块：

1. 多轮 AgentLoop 的有界重采样
2. `global_steps` 只在真正 optimizer 更新后自增（原版会提前自增并提交）

补丁文件 `patches/verl-0.8.0-shopping-dynamic-sampling.patch`，只改一个文件。

`scripts/apply_verl_dynamic_sampling_patch.py` 用**三重哈希**锁死：

```
EXPECTED_ORIGINAL_SHA256   = de58d295…
EXPECTED_PATCHED_SHA256    = 684b491e…
SUPERSEDED_PATCHED_SHA256  = fc3564cc…   （可升级的旧版本）
PATCH_MARKER               = SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V3
```

校验条件：`verl.__version__ == 0.8.0`、`verl.__file__` 必须在项目 `.venv` 内、
目标文件哈希匹配、打完后重新校验哈希 + marker + `py_compile`，失败自动回滚。

### 6.4 诊断文件

`SHOPPING_GRPO_DIAGNOSTICS_PATH` → `training_diagnostics.jsonl`，三种事件：

| event | 内容 |
|---|---|
| `generation_batch` | 每条 rollout 的 tool 序列、终局、reward breakdown、守卫拒绝原因、组 keep/drop 决策 |
| `optimizer_step` | veRL 标量指标：entropy / PPO KL / clip fraction / response length / 有效组率 |
| `skipped_update` | 零信号尝试，不推进 optimizer step |

---

## 8. 配置：非默认值一览

```yaml
# configs/grpo.yaml
data:
  train_batch_size: 2
  val_batch_size: 2
  max_prompt_length: 4096
  max_response_length: 20480
  return_raw_chat: true
  truncation: error

actor_rollout_ref:
  model:
    lora_rank: 16
    lora_alpha: 32
    target_modules: all-linear          # ← 和 SFT 的显式列表不同
    lora: {merge: true}
    override_config: {attn_implementation: sdpa}
  actor:
    ppo_mini_batch_size: 2
    ppo_micro_batch_size_per_gpu: 1
    ppo_max_token_len_per_gpu: 24576
    use_dynamic_bsz: false
    clip_ratio_low: 0.20
    clip_ratio_high: 0.20               # 对称，不是 DAPO 的 0.20/0.28
    entropy_coeff: 0.0
    calculate_entropy: true             # 只测不用
    use_kl_loss: false
    optim: {lr: 1.0e-6, lr_warmup_steps_ratio: 0.03}
  rollout:
    name: vllm
    mode: async
    n: 4
    temperature: 0.7
    top_p: 0.9
    gpu_memory_utilization: 0.45        # 和 FSDP actor 共享一张卡
    max_model_len: 24576
    max_num_batched_tokens: 24576
    free_cache_engine: true
    agent: {num_workers: 8, default_agent_loop: shopping_tool_agent}
    multi_turn:
      format: qwen3_coder
      max_user_turns: 40
      max_assistant_turns: 40
      max_parallel_calls: 1
      max_tool_response_length: 16384
    val_kwargs: {n: 1, temperature: 0.0, top_p: 1.0, do_sample: false}

algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
  norm_adv_by_std_in_grpo: false        # ← GRPO 默认会除 std，这里关掉
  rollout_correction:
    bypass_mode: true                   # 动态采样要求
    loss_type: ppo_clip

trainer:
  total_epochs: 50
  total_training_steps: 500
  save_freq: 50
  test_freq: 50
  val_before_train: true
  n_gpus_per_node: 1
  nnodes: 1
```

**`norm_adv_by_std_in_grpo: false` 配合动态采样是自洽的**：组方差已由重采样保证非零，
不再额外除 std。

### 7.1 AgentLoop 侧参数（`configs/agent_loop.yaml`）

```yaml
max_steps: 35
reward_mode: constraint_aware
required_environment_version: shopsimulator-environment-v2.1
context_window_tokens: 24576
context_generation_reserve_tokens: 512   # 每轮 max_tokens 被 cap 到这里
context_safety_margin_tokens: 512
context_input_budget_tokens: 16384       # 只在开压缩时用
context_compaction_enable: false
observation_token_budget: 1536           # 搜索结果
observation_detail_token_budget: 4096    # 商品详情
observation_generic_token_budget: 768
observation_search_top_k: 20
reward_length_shaping_enable: ${oc.env:SHOPPING_LENGTH_SHAPING_ENABLE,false}
reward_length_soft_threshold: 20
reward_length_penalty_per_step: 0.01
reward_length_max_penalty: 0.15
```

长度惩罚默认关闭，且 `reward_mode` 必须是 `constraint_aware` 才能开（`agent_loop.py:103`）。

---

## 9. Preflight：加载模型前的硬闸

`scripts/train_grpo.py` 在真正训练前先跑：

```bash
.venv/bin/python scripts/check_grpo_runtime.py <同样的 overrides>
```

| 检查 | 内容 |
|---|---|
| Python | 必须 3.12 |
| 依赖版本 | verl 0.8.0 / vllm 0.25.1 / torch 2.11.0 / transformers 5.15.0.dev0 / ray 2.56.1 / tensordict 0.10.0 / numpy 2.2.6 / swanlab 0.9.1 |
| transformers revision | 固定 git commit `7ea2320c…`（读 `direct_url.json`） |
| 环境 manifest | `data/environment.json` + **5 个 runtime 文件重新哈希** |
| 内存预算 | `max_prompt_length + max_response_length ≤ 24576`；五个 `*_max_token_len_per_gpu` 必须都等于 24576 |
| micro batch | `use_dynamic_bsz=false`、`log_prob_micro_batch_size_per_gpu=1` |
| 动态采样 | 补丁 marker + 两条配置 + 4 样本 sanity check |
| TRACE | LoRA 存在、单卡、horizon/ε 合法 |
| CUDA | `torch.cuda.is_available()` |
| veRL API | `ShoppingToolAgentLoop` 是 `ToolAgentLoop` 子类、`qwen3_coder` parser 存在 |

**任一不过就不加载模型。**

---

## 10. 能换什么、不能换什么

| 目标 | 能否 | 做法 |
|---|---|---|
| 换 AgentLoop 实现 | ✅ | `agent_loop.yaml` 的 `_target_` + 两处 `name` |
| 换 reward 公式 | ✅（唯一可行方式） | 子类化 `ShoppingToolAgentLoop`，覆盖 `run()` 最后一段 |
| 换 `reward_mode` | ⚠️ 只有两档 | `native` / `constraint_aware` |
| 加长度惩罚 | ✅ | 4 个环境变量 |
| 关动态采样 | ✅ | `shopping_dynamic_sampling.enable=false`（但会失去有界重采样） |
| 开 TRACE | ✅ | `shopping_trace.enable=true` |
| 用 `custom_reward_function` | ❌ | 见下 |
| 改环境 Reward v3 常量 | ❌ | 三重锁：`DEFAULT_REWARDS` / `environment.json` / manifest runtime hash |
| 改用原生 `default_agent_loop=tool_agent` | ❌ | 见下 |

### 9.1 为什么 `custom_reward_function` 走不通

1. **工具恒返回 0.0**：真正的 reward 只在 `run()` 里注入
2. **拿不到状态**：`compute_score` 只有响应文本 + `extra_info`（只有 `task_id`），
   没有 asin / options / 价格 / goal
3. **键层级也不对**：veRL 0.8 的块是 `reward_model.custom_reward_function.*`

### 9.2 为什么不能换回原生 `tool_agent`

`ShopSimulatorTool.execute()` 开头：

```python
env = current_environment.get()
state = current_runtime_state.get()
if env is None or state is None:
    raise RuntimeError("ShopSimulator tool executed without a trajectory-local interaction state")
```

这两个 `ContextVar` 由 `ShopSimulatorSession.start()` 设置，而它**只在
`ShoppingToolAgentLoop.run()` 里被调用**。

即使改成工具自建环境，**租约没人释放**——`ShopSimulatorTool.release()` 是空实现
（注释明说"真正的环境租约由 AgentLoop 释放"）。8 个槽位，漏 8 条后全部卡死。

### 9.3 换 reward 的最小写法

```python
# mypkg/reward_only_loop.py
from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop

class MyLoop(ShoppingToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        shopping = output.extra_fields["shopping"]
        new_total = my_formula(shopping["reward"], shopping)
        output.reward_score = new_total
        shopping["reward"]["terminal_utility"] = new_total   # ← 必须同步
        shopping["reward"]["total"] = new_total              # ← 必须同步
        return output
```

**必须保留的字段**（`dynamic_sampling.py:186` 会读，缺了 `ValueError`）：

```python
info["reward"]["terminal_utility"]
info["reward"]["purchase_success"]
info["infrastructure_invalid"]
```

---

## 11. TRACE（已实现，默认关闭）

```yaml
shopping_trace:
  enable: false
  epsilon: 0.1
  horizon: 3
  discount: 0.8
  terminal_weight: 2.0
  outcome_weight: 1.0
  turn_weight: 0.2
  max_sequence_length: 24576
```

实现方式：`compat.py:4` monkey-patch `RayPPOTrainer._update_actor`，
在原有 GRPO 更新前：

```
冻结 LoRA base 对私有 gold target 打分
  → 每个工具边界的 mean log-prob
  → remaining-gap → log-ratio delta
  → K-step backup + terminal fill
  → 混合 advantage（α_out × outcome + α_turn × turn）
```

开启条件由 preflight 强制：LoRA rank > 0（要冻结 base 当 reference）、单卡、
`adv_estimator == grpo`、`max_sequence_length == 24576`。

关闭时 `update_actor_with_trace` 会主动 `pop("trace_target")`，避免残留。

**研究结论见 `docs/research/2026-09-02-trace-grpo-credit-assignment.md`**——
论文只在短答案搜索任务验证过，购物是结构化多解目标，建议先做 Phase 0 离线 proxy 验证。

---

## 12. 产物与导出

```
outputs/models/grpo/
  global_step_100/actor/                  veRL FSDP checkpoint
  training_diagnostics.jsonl
```

导出成可服务模型：

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

内部是：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir <actor> --target_dir <merged> --trust-remote-code
```

**checkpoint 用验证集指标选，不看最后一个 step**（训练 500 步，报告用 100）。

---

## 13. RL 采样 vs SFT 采样：四个行为差异

环境契约完全共用（`actions.py` / `tools.py` / `projection.py` / `observation.py`），
但循环是**两套独立实现**。

| 维度 | SFT 采集 | RL rollout |
|---|---|---|
| 谁生成 | 外部教师 `deepseek-v4-flash` | **当前策略**（vLLM in-process） |
| 循环代码 | `rollout.py:collect_for_task` | veRL `ToolAgentLoop` |
| 并发 | `ThreadPoolExecutor` + 阻塞 HTTP | `asyncio` + `asyncio.to_thread` |
| 每股几条 | `attempts_per_task`（默认 1） | `n=4` |
| 采样参数 | temp 0.0 / top_p 1.0 | temp 0.7 / top_p 0.9 |
| gold 可见 | `include_trace_target=False` | **True** |

### 四个行为差异

| 情况 | SFT | RL |
|---|---|---|
| 并行 tool call | 保留第一个，其余记 `tool_call_truncations`，继续 | `terminate("parallel_tool_calls")` + 判无效 |
| 守卫拒绝的消息 | 带"可打开的商品 ASIN / 可点击的按钮"完整提示 | 只给原因 |
| 守卫连拒 3 次 | `invalid_action_limit`（status） | `too_many_guard_rejections`（termination_reason） |
| 上下文超限 | `ContextBudgetError` → 该条 `status=error` | `context_hard_limit_exceeded` + **判无效** |
| 投影失败 | `ToolExecutionError` → `status=error` | `observation_projection_failed` + **判无效** |

### reward 用法差异

| | SFT | RL |
|---|---|---|
| 何时看 | 采集**全部结束后**批量 | **每条轨迹结束时**当场 |
| 校验强度 | 5 个顶层字段 | 30+ 条 schema + `terminal_utility == step.reward` |
| 用途 | 二值过滤 | 连续训练信号 |
| 能学哪些结局 | **只有 `gold_purchase`** | 全部 9 种（含 −0.85） |
| 无效时 | 丢掉该条 | reward 0 + **毒化整个 group** |

### 为什么必须两套

SFT 要的是**可训练的完整对话序列 + 事后验收**（落 JSONL）；
RL 要的是**同 prompt 的多条 token 序列 + 当场 advantage**（`DataProto`）。

**代价**：同一个交互循环实现两遍。改守卫规则时**两个路径都要验**——
`collection/sft.py:_tool_step_reject_reason` 的"重放校验"就是为防这个漂移设的。

---

## 14. 复现清单

```bash
# 0. 依赖 + 环境
bash scripts/setup.sh                 # uv sync --extra sft --extra grpo
                                      # + 解压商品 + 建索引 + 打 veRL 补丁
bash scripts/start_environment.sh     # 127.0.0.1:5700，8 槽位

# 1. 先看命令，不启动 CUDA/Ray
bash scripts/grpo.sh --dry-run

# 2. 正式训练（会自动跑 preflight）
bash scripts/grpo.sh --model outputs/models/sft-curriculum/stage-c/merged

# 3. 短跑验证
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10

# 4. 验证补丁状态
bash scripts/apply_verl_dynamic_sampling_patch.py --check

# 5. 导出
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

### 跑完核对

```bash
# 诊断文件应该有三种 event
python - <<'PY'
import json, collections
p = "outputs/models/grpo/training_diagnostics.jsonl"
c = collections.Counter()
steps = set()
for line in open(p, encoding="utf-8"):
    r = json.loads(line)
    c[r["event"]] += 1
    if r["event"] == "optimizer_step":
        steps.add(r["global_step"])
print(dict(c))
print("optimizer steps:", len(steps), "| max:", max(steps) if steps else None)
PY
# 期望：generation_batch 数量 ≈ optimizer_step × 1~3
#       skipped_update 应当远少于 optimizer_step

# 组有效率（在 optimizer_step 记录里，字段名来自补丁）
python - <<'PY'
import json
for line in open("outputs/models/grpo/training_diagnostics.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["event"] == "optimizer_step":
        print("step", r["global_step"],
              "generated", r.get("group/generated"),
              "trained", r.get("group/trained"),
              "effective_ratio", round(r.get("group/effective_ratio", 0), 3),
              "sampling_invalid", r.get("group/sampling_invalid"))
        break
PY
```

补丁实际写出的 `group/*` 指标：

```
group/generated                 本次生成的组数
group/trained                   实际进入训练的组数
group/effective_ratio           有效组 / 生成组
group/all_equal_ratio           奖励全相同的组占比
group/all_zero_utility_ratio    全零效用的组占比
group/no_purchase_success_ratio 无购买成功的组占比
group/all_purchase_success_ratio 全部购买成功的组占比
group/sampling_invalid          含无效成员的组数
group/resample_batches          本次重采批次数
```

---

## 15. 一张图

```
scripts/grpo.sh
  └── train_grpo.py
        ├─ 注入 env（GRPO_MODEL_PATH / GRPO_OUTPUT_DIR / SHOPPING_GRPO_ROOT …）
        ├─ subprocess: check_grpo_runtime.py   ← 依赖/哈希/内存/manifest 全过才继续
        └─ subprocess: verl.trainer.main_ppo
              ├─ configs/grpo.yaml
              │     └─ default_agent_loop + agent_loop_config_path
              │           └─ configs/agent_loop.yaml → _target_ = ShoppingToolAgentLoop
              │                 ├─ Session  → POST reset / release_one
              │                 ├─ Tool     → 守卫 → POST interact
              │                 └─ run()    → output.reward_score
              │                       └─ runtime.reward_breakdown
              │                             └─ 环境 evaluate_purchase（Reward v3）
              ├─ 动态采样（补丁）→ 丢常量组 / 丢 sampling_invalid 组
              ├─ GRPO advantage（norm_adv_by_std = false）
              └─ optimizer step（lr 1e-6, clip 0.2/0.2, 无 KL）
```

**一句话**：**veRL 提供多轮骨架，项目只负责"环境生命周期 + 状态 + 奖励注入"三件事；
奖励是环境同步算的 `terminal_utility`，不是事后给文本打分。**

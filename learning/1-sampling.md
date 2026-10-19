# 01 · SFT 数据采样：从任务列表到三个分桶

这篇文档只覆盖一件事：**`data/grpo/train.jsonl` 里的 task_id 列表，怎样变成
`data/sft_curriculum/manifest.json` 里 A/B/C 三个阶段的分桶。** 也就是"喂给 SFT 之前"
的全部工序。训练本身（tokenize、label mask、LoRA）见 02 篇。

读完之后你应该能回答：一条轨迹为什么被丢、1,192 这个数字怎么来的、为什么同一个
task 只能进一个桶、以及怎样从零复现这四个文件。

---

## 0. 全景

```
data/grpo/train.jsonl (1,000 task_id)
        │  ① 采集（教师模型 × 环境，逐轮工具调用）
        ▼
outputs/sft-collection/raw.jsonl          ← 唯一真相源，append-only，可断点续跑
        │  ② 验收 + 清洗 + 按 task 切分
        ▼
outputs/sft-collection/{accepted,rejected,sft,train,validation}.jsonl
        │  （冻结提交为 data/sft/{train,validation}.jsonl，即 current_v4）
        │
        │  ③ 两个批次合并：内容哈希去重 + 同题择优
        ▼
data/sft_pure_v4/all.jsonl                (1,192 条，全 gold、0 守卫拒绝)
        │  ④ 打难度标签（另一次 deepseek-v4-flash 调用）
        ▼
data/sft_pure_v4/difficulty_labels.jsonl  (1,192 条)
        │  ⑤ 分桶 + 分层切分 + 累积阶段
        ▼
data/sft_curriculum/manifest.json         (A=256 / B=799 / C=1,073)
```

四道工序对应四个脚本，四个产物文件互相用哈希锁死：

| 工序 | 脚本 | 产物 | 哈希位置 |
|---|---|---|---|
| ① 采集 | `scripts/collect_sft_data.py` | `outputs/sft-collection/raw.jsonl` | 不提交 |
| ② 验收清洗 | 同上（`collection/sft.py`） | `data/sft/{train,validation}.jsonl` | `data/sft/metadata.json` |
| ③ 合并去重 | `scripts/merge_sft_pure_v4.py` | `data/sft_pure_v4/all.jsonl` | `data/sft_pure_v4/metadata.json` |
| ④ 难度打标 | `scripts/label_sft_difficulty.py` | `data/sft_pure_v4/difficulty_labels.jsonl` | 同上 |
| ⑤ 课程冻结 | `scripts/prepare_sft_curriculum.py` | `data/sft_curriculum/manifest.json` | manifest 内 `source.sha256` |

> 关键认知：**① 和 ② 是同一个脚本的两半。** `collect_sft_data.py` 先跑 ① 写
> `raw.jsonl`，再调 `build_collection_artifacts` 跑 ② 重建所有派生文件。所以
> `--build-only` 可以完全不碰网络和模型、只把 ② 重跑一遍。

---

## 1. 工序 ①：采集

### 1.1 命令

```bash
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY=...
bash scripts/start_environment.sh          # ShopSimulator 监听 127.0.0.1:5700

python scripts/collect_sft_data.py \
  --tasks data/grpo/train.jsonl \
  --output-dir outputs/sft-collection \
  --model deepseek-v4-flash \
  --target-accepted 1000 \
  --workers 4
```

默认参数里几个重要的（`collect_sft_data.py:46`）：`--held-out-tasks
data/evaluation/tasks.jsonl`、`--max-steps 35`、`--temperature 0.0`、`--top-p 1.0`、
`--max-tokens 512`、`--validation-ratio 0.1`、`--seed 42`、
`--observation-token-budget 0`（默认**关闭**投影）、`--context-window 0`（默认关闭）。

### 1.2 先剔掉评测任务，再做并发控制

`main()`（:265）在读 task 之后立刻过滤：

```python
tasks = [t for t in load_tasks(args.tasks) if int(t["task_id"]) not in held_out_ids]
```

`held_out_ids` 就是 `data/evaluation/tasks.jsonl` 的 200 个 ID。这是**第一道防泄漏**，
`build_collection_artifacts` 里还会再查一次（第二道）。

`--target-accepted` 模式下走 `collect_until_target`（:94），并发策略是：

```python
remaining = target_accepted - accepted
max_pending = min(workers, max(remaining, 0))     # 还差 3 条就只再开 3 个
```

**它不会超采。** 每次只用当前剩余名额开新任务，所以最终 accepted 数不会明显越过
target。

### 1.3 断点续跑

`raw.jsonl` 是 append-only 的真相源。`completed_task_attempts`（`rollout.py:256`）
读出所有已完成的 `(task_id, attempt_index)`，重跑时跳过。`_accepted_count`（:164）
重放一遍验收规则数出已到手多少条，用来算 `remaining`。所以中断后原样重跑即可。

### 1.4 单条轨迹的交互循环

核心是 `collect_for_task`（`rollout.py:274`）。**采集用的 rollout 代码和评测用的是
同一套**（同一个 `OpenAIClient`、同一个 `SYSTEM_PROMPT`、同一个守卫），只有模型不同
——这是"教师看到的环境和最终 Actor 看到的环境逐字段一致"的保证。

**(a) reset 建立租约**

`env.reset(task_id)` → `POST /api/shop_agent`：

```json
请求: {"action": "reset", "idx": 65}
响应: {"result": {"env_idx": 3, "idx": 65,
                 "environment_version": "shopsimulator-environment-v2.1",
                 "instruction": "我想要买一款适合幼儿园小朋友绘画使用的水彩笔……",
                 "observation_state": {...}}}
```

注意 `include_trace_target` 保持 **False**（只有 GRPO 的 `session.py:44` 会打开），
所以教师**拿不到 gold**。

**(b) 初始 messages**

`_initial_messages`（:560）：`system = SYSTEM_PROMPT`（7 条规则，见 `rollout.py:34`），
`user = "Instruction: <instruction>"`。如果 task 行自带 `prompt` 字段则用它当主体。

**(c) 循环：生成 → 守卫 → 执行 → 投影**

`while len(trajectory["steps"]) < 35:`（:322）

1. **生成** — `client.complete(messages, SHOP_TOOL_SCHEMAS)`，带 13 个工具的
   Schema，`tool_choice=auto`、`max_tokens=512`、`temperature=0.0`。
   对 `deepseek-v4*` 会显式加 `"thinking": {"type": "disabled"}`（:185），因为
   OpenCode Go 默认把 V4 设成思考开启，不写等于开着。

2. **串行化** — `_enforce_serial_tool_call`（:550）只保留第一个 tool_call，其余记进
   `tool_call_truncations`。**一个 assistant 回合只能动一次环境。**

3. **没有 tool call** → `status = "assistant_final"`，break。这是失败分支，
   `done` 仍是 `False`。

4. **动作守卫** — `action_reject_reason(name, arguments, latest_observation)`
   （`environment/actions.py:31`），严格按顺序查四层：

   | 顺序 | 检查 | 拒绝原因 |
   |---|---|---|
   | ① | schemas 外字段 | `schema_extra_arguments:query` |
   | ② | `think` | 放行（不碰环境） |
   | ② | `finish_without_purchase` 的 reason 非 `no_suitable_product` | `invalid_finish_reason` |
   | ② | `search_products` 但当前页 `搜索功能是否可用: False` | `search_not_available_on_current_page` |
   | ③ | `open_product` 的 asin 不在最新页 `^\d+\|(asin)\|` 行里 | `click_not_in_previous_observation` |
   | ④ | `select_option` 点了导航按钮 | `select_option_is_navigation_button` |
   | ④ | click 目标不在最新页 footer 的 `可点击的按钮: [...]` | `click_not_in_previous_observation` |

   守卫从 `latest_observation` 里**正则抠**目标（`actions.py:113/125`），
   和投影器共用同一套解析函数。

5. **被拒** → 写进 `blocked_tool_calls`，追加一条带 `runtime_action_guard: true`
   标记的 tool 错误消息（`action_guard_tool_message`，`actions.py:80`），里面会列出
   "当前可打开的商品 ASIN / 可点击的按钮 / 下一步只能…"。**被拒不消耗步数**，
   但连续 3 次 → `status = "invalid_action_limit"`，break。

6. **放行** → `tool_call_to_action`（`environment/tools.py:21`）翻译，再 `env.step`：

   | tool call | 环境动作 |
   |---|---|
   | `search_products {"query":"水彩笔"}` | `search[水彩笔]` |
   | `open_product {"asin":"879…"}` | `click[879…]` |
   | `view_features {}` | `click[Features]` |
   | `select_option {"value":"24色三角粗杆…"}` | `click[24色三角粗杆…]` |
   | `buy_now {}` | `click[Buy Now]` |
   | `finish_without_purchase {"reason":"no_suitable_product"}` | `finish[no_suitable_product]` |

   请求体：`{"action":"interact","env_idx":3,"response":"search[水彩笔 24色 三角杆 儿童]"}`

7. **投影压缩** — `project_observation`（`environment/projection.py:56`）。按页型选
   token 预算：搜索结果 **1536** / 商品详情 **4096** / 其他 **768**。做法是**二分找
   能塞进预算的最长标题**，然后反向自检：footer 的 `搜索功能是否可用:` 和
   `可点击的按钮:` 必须完整、该页所有 ASIN 必须仍可操作、导航按钮集合不能变。任何
   一条不满足就**抛错而不是静默丢数据**。

   实际效果（搜索结果行）：

   ```
   原始: 1|796075776215|16.0 to 29.0|官方国货甄选|文具电教/文化用品/…|幼儿园宝宝,…不脏手|晨光水彩笔儿…笔套装
   投影: 1|796075776215|16.0 t…9.0|官方国货甄选|文具电教/文…水彩笔|幼儿园宝宝,…不脏手|晨光水彩笔儿…笔套装
   ```

   `rank` 和 `asin` **原样保留**，标题按 `头 2/3 + … + 尾 1/3` 压。

   > 1,192 条最终数据里有 **922 条**至少触发过一次投影压缩。

8. **追加 tool message**，`latest_observation` 更新，回到 1。

**(d) 终止：`done` 只可能来自环境**

这是最容易误解的一点。**模型表达"结束"的方式是调用终止工具，而不是停止输出
tool call。** 唯一把 `done` 置真的地方是 `rollout.py:407`：

```python
if step["done"]:
    trajectory["status"] = "done"
    trajectory["terminal_result"] = step["result"]
    trajectory["final_reward"] = step["reward"]
    trajectory["done"] = True
    return trajectory
```

而 `step["done"] = bool(result.get("done", False))`（:526），`result` 来自
`env.step(action)`。环境侧有三条终止路径：

| 触发 | 环境代码 | `reward_type` |
|---|---|---|
| `click[Buy Now]` | `web_agent_text_env.py:909` → `done()` → `evaluate_purchase` | `gold_purchase` / `valid_alternative_purchase` / `partial_alternative_purchase` / `wrong_purchase` |
| `finish[no_suitable_product]` | `:786` → `finish_without_purchase()` → `evaluate_abstain` | `graceful_stop` / `early_abstain` |
| 进度跟踪器强制终止 | `:200-215` → `terminate_session()` | `repeat_loop` / `max_steps` |

第三条容易被忽略：`EvidenceProgressTracker`（`engine/termination.py:168`）在
`consecutive_repeats >= 2` 或 `no_progress_steps >= 4` 或 `steps >= 35` 时**环境自己**
把 `done=True` 塞回来——模型根本没机会决定。

`status` 的完整取值，**只有第一种能产出可用数据**：

| status | 谁决定 | `done` | 进 SFT？ |
|---|---|---|---|
| `done` | 环境返回 `done=True` | True | ✅（且还要过 Reward v3 那关） |
| `assistant_final` | 模型没吐 tool call | False | ❌ |
| `invalid_action_limit` | 本地守卫连拒 3 次 | False | ❌ |
| `max_steps` | 本地循环跑满 35 步 | False | ❌ |
| `error` / `environment_release_failed` | 异常 | False | ❌ |

**(e) finally release**

`POST {"action":"release_one","env_idx":3}`。释放失败会把 `status` 改成
`environment_release_failed`，并触发 `CollectionInfrastructureError` **中断整批采集**
（避免污染后续任务）。`reset` 阶段失败也会先 release 再抛。

> 服务端**不会**在 `over` 时自动释放租约——`pack_api.py` 明确注释这是为了不和
> 客户端的 `finally` release 竞争。租约归调用方所有，直到显式 `release_one`。

### 1.5 一个真实例子（task_id=65）

用户需求：24 色三角粗杆、可水洗、无毒、送 4 件绘画工具的幼儿园水彩笔，**20 元以内**。
原始数据 `data/sft_pure_v4/all.jsonl` 第一条，共 11 步、24 条 message。

| # | 动作 | 结果 |
|---|---|---|
| 0 | `search_products {"query":"水彩笔 24色 三角杆 儿童"}` | Page 1 of 8，150 条命中，本页 20 个候选 |
| 1 | `open_product 899532603012` | 得力，10 个规格可选 |
| 2 | `select_option "24色【加量粗杆可水洗】送绘画4件礼"` | variant 价 **21 元** |
| 3 | `prev_page {}` | 回搜索页（21 > 20，放弃） |
| 4 | `open_product 744447021982` | 西瓜太郎，赠品是勾线笔不是"4 件绘画工具" |
| 5 | `prev_page {}` | 回搜索页 |
| 6 | `open_product 886076453056` | 得力另一款 |
| 7 | `select_option "24色三角粗杆【加量-可水洗】送绘画4件"` | variant 价 **17 元** ✓ |
| 8 | `view_features {}` | 子页内容为空 |
| 9 | `prev_page {}` | 回商品页 |
| 10 | `buy_now {}` | **`gold_purchase`，reward = 1.0** |

终端 observation 长这样（`observation_state` 渲染后）：

```
[SHOPPING_OBSERVATION_V2]
page_type: terminal

搜索功能是否可用: False
可点击的按钮: []
```

---

## 2. 工序 ②：验收 + 清洗 + 切分

`build_collection_artifacts`（`collection/sft.py:104`）逐条读 `raw.jsonl`。

### 2.1 硬验收 `acceptance_reasons`（:32）

全部必须满足：

- `trajectory["error"]` 为空、`release_error` 为空
- `status == "done"` 且 `trajectory["done"] is True`
- `terminal_result` 里 `done is True` 且 `over is True`
- 存在 `buy_now` step（或 `env_action == "click[Buy Now]"`）
- `reward_version == "shopsimulator-reward-v3"`
- `reward_type == "gold_purchase"`
- `reward_valid is True`
- `purchase_success is True`
- `termination_reason == "gold_purchase"`
- 没有任何 assistant 消息带 >1 个 tool_call
- **逐 step 重放守卫** `_tool_step_reject_reason`（:253）：重新解析 arguments、
  重跑 `action_reject_reason`，并要求 `tool_call_to_action(name, args)` 的结果
  **等于**当时记录的 `env_action`。这一步是为了防止"采集时是旧守卫、后来收紧了"
  导致脏数据入库。

任一不过 → 写进 `rejected.jsonl`，原因累计进 `reject_reasons`。

**实测审计**（`data/sft/metadata.json` → `provenance`）：

| 项 | 值 |
|---|---:|
| 原始轨迹 | 2,498 |
| 通过验收 | 1,026 |
| 接受率 | **41.07%** |
| 冻结使用 | 1,000 |
| 未用的通过轨迹 | 26 |

未通过的原因分布（`drop_counts`）：

| 原因 | 条数 |
|---|---:|
| `guard_violation` | 716 |
| `not_gold:partial_alternative_purchase` | 223 |
| `not_gold:repeat_loop` | 214 |
| `not_gold:unknown` | 100 |
| `not_gold:wrong_purchase` | 97 |
| `not_gold:max_steps` | 85 |
| `not_gold:reward_unverifiable` | 32 |
| 其余（early_abstain / graceful_stop / valid_alternative） | 5 |

> 注意 `reward_type_counts` 里 **`gold_purchase` 有 1,742 条**，但只接受了 1,026。
> 差额主要来自 `guard_violation`：轨迹最终买对了，但过程中有被守卫拒过的动作。

### 2.1.1 ⚠️ 1,026 → 1,000 这一步不在本仓库的代码里

上面这些审计数字来自 `data/sft/metadata.json`，但**本仓库的采集器产不出这个文件**。
`grep -rn "unused_accepted|split_mode|exact_rows|provenance|drop_counts|acceptance_rate"
--include=*.py .` 是 **0 匹配**。两边的 metadata 形状完全不同：

| | `data/sft/metadata.json` | `build_collection_artifacts` 实际写的 |
|---|---|---|
| schema | `shopping-sft-dataset-v1` | `shopping-sft-collection-v1` |
| 审计字段 | `provenance{raw_trajectories, accepted, acceptance_rate, unused_accepted, split_mode, drop_counts, source_files, teacher_models, seed, val_ratio}` | `{total, accepted, rejected, held_out_excluded, duplicate_tasks_excluded, train, validation, reject_reasons}` + `{environment, reward, validation_ratio, split_seed, collection_config, files, raw}` |
| 切分维度 | `split_mode: "exact_rows"`（按行） | `split_rows_by_task`（按 task） |
| 验证比例 | 800/200 = **20%** | `validation_ratio=0.1` = **10%** |

三处都对不上。**更关键的是 `unused_accepted: 26` 本身就是反证**：本仓库的
`collect_until_target`（`collect_sft_data.py:125`）是配额驱动的——

```python
remaining   = int(target_accepted) - accepted
max_pending = min(workers, max(remaining, 0))
```

每次只开 `remaining` 个并发，**永远不可能超采**。用 `--target-accepted 1000` 跑，
结果正好是 1,000 accepted / 0 unused。所以这个数据集不是用当前脚本、
按文档里那条命令产出的；最可能是当时不带 `--target-accepted` 跑完全部（`collect_tasks`），
拿到 1,026 条，然后在**仓库外**用另一个脚本砍到 1,000。

`docs/research/sft_grpo_diagnosis.md:45` 已记录这个断档：metadata 声称由
`scripts/build_data_new.py` 生成，但该脚本不在工作区，且
`outputs/teacher_collection/raw_trajectories.jsonl` 也不存在。经查
`build_data_new.py` 在**整个 git 历史里从未出现过**。

**能确定的**：

- 不是"前 1,000 条"——冻结的 task_id 在文件里不是升序（首行 1535 / 16063 / 5911 / …），
  排序后 65–23405，中位数 12821，只有 24 个 ≤ 1200。是某种稳定顺序 + 截断。
- 800/200 是硬编码的 20%。
- `source_files` 只有一个，`attempts_per_task` 看起来是 1，所以 2,498 ≈ 2,498 个 task。
- 算术自洽：SFT 1,000 + 评测 200 + GRPO 1,000 = 2,200 ≤ 2,498，余约 298 被丢弃。

**不能确定的**：具体用哪个哈希/顺序截的。产出它的脚本没进仓库，
**这一步不可复现**。工序 ③–⑤ 完全可复现；工序 ①② 的"从 raw 到 1,026"可复现，
"从 1,026 到 1,000"不可复现。

**附带结构发现**：`data/sft/` ∩ `data/grpo/train.jsonl` = **0**。若各自从同一
~2,500 池随机抽 1,000，重叠期望 ~400。零重叠说明这是刻意分区：SFT 先冻结 1,000，
GRPO 再从不含 SFT 任务与评测 200 题的剩余池里抽 1,000
（`grpo/metadata.json` 的 `provenance.excluded` 正是这么写的）。

### 2.2 两次去泄漏 + 每 task 只留一条

```python
if accepted and task_id in held_out:      reasons = ["held_out_task"]
elif accepted and task_id in accepted_task_ids: reasons = ["duplicate_task"]
```

- `held_out` = Final-200 的 200 个 ID（`data/evaluation/tasks.jsonl`）→ **第二道防泄漏**
- `accepted_task_ids` 保证**每个 task 最多一条轨迹**（先到先得）

### 2.3 行级清洗 `build_sft_row`（:83）

这一步才真正把"采集日志"变成"训练样本"：

1. `_terminal_tool_call_id`（:303）找出终局那一步的 tool_call id；
2. `blocked_tool_calls` 里的 id 收集成 `blocked_call_ids`，
   `_is_blocked_training_message`（:321）把**这些守卫被拒的 assistant 回合和对应的
   tool 回复整对删掉**——被拒的尝试根本不出现在训练数据里；
3. `_sanitize_message`（:334）用白名单
   `ALLOWED_MESSAGE_KEYS = {role, content, tool_calls, tool_call_id, name}`
   只留 5 个字段，`tool_calls` 再收敛到 `{id, type, function{name, arguments}}`；
4. **终局那条 tool message 的 content 改成 `购买已完成。`**，抹掉环境返回的
   `reward_detail`（里面带着 `weighted_score`、`hard_gates`、gold ASIN）；
5. 每行附带 `"tools": deepcopy(SHOP_TOOL_SCHEMAS)`——13 个工具的完整 Schema
   跟着数据走，不靠训练脚本注入。

> 这一步把采集日志里的 `runtime_action_guard`、`projection`、`raw_observation`、
> `latest_observation` 等**全部剥掉**。

### 2.4 按 task 切分

`split_rows_by_task(rows, validation_ratio=0.1, seed=42)`（`training/sft/dataset.py:203`）：

```python
def stable_key(task_id):
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
ordered_ids = sorted(task_ids, key=stable_key)      # 按哈希排，不是按 ID 排
validation_count = max(1, round(len(ordered_ids) * ratio))
validation_ids = set(ordered_ids[:validation_count])
```

**按 task 分而不是按行分**，保证同一题不跨集。

### 2.5 产物与记账

```
outputs/sft-collection/
  raw.jsonl          ← 真相源，绝不重写
  accepted.jsonl     通过验收的完整轨迹
  rejected.jsonl     task_id + 拒绝原因
  reject_stats.json  聚合审计
  sft.jsonl          清洗后、切分前
  train.jsonl        按 task 切分
  validation.jsonl
  metadata.json      rows + sha256 + collection_config
```

`metadata.json` 给每个派生文件记 rows 和 sha256，连 `raw.jsonl` 的哈希一起。

### 2.6 冻结提交

这 1,000 条冻结提交成 `data/sft/{train,validation}.jsonl`（800 / 200），
元数据在 `data/sft/metadata.json`，**并且作为 `current_v4` 参与工序 ③**。

> **关于终局 content 的一处历史不一致**：`data/sft/` 的 1,000 条最后一条 tool
> message 全部是环境原始渲染（`page_type: terminal`），**没有**被替换成
> `购买已完成。`——说明这份提交早于 `_sanitize_message` 的替换逻辑。后续工序 ③ 从
> merged 包带进来 261 条 `购买已完成。`，于是最终数据里两种形式并存。详见 §4.3。

---

## 3. 工序 ③：两批合并去重

### 3.1 两个来源是什么

| 来源 | 条数 | 出处 |
|---|---:|---|
| `current_v4` | 1,000 | 工序 ①② 的产物，`data/sft/{train,validation}.jsonl` |
| `merged_v4` | 878 | **外部送进来的另一个 DeepSeek-v4-flash 采集批次**（`merged` 包，含 train/validation） |

脚本：`scripts/merge_sft_pure_v4.py`。

```bash
python scripts/merge_sft_pure_v4.py \
  --current-train      data/sft/train.jsonl \
  --current-validation data/sft/validation.jsonl \
  --merged-train       <merged>/train.jsonl \
  --merged-validation  <merged>/validation.jsonl \
  --legacy-ref HEAD^ \
  --output-dir data/sft_pure_v4
```

### 3.2 去重是两步，解决的问题不同

**第一步：内容哈希过滤（−428）**

`content_hash`（:35）**只哈希 `messages` + `tools`，不含 `task_id` / `trajectory_id`**：

```python
def content_hash(row):
    payload = {"messages": row.get("messages"), "tools": row.get("tools")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
```

`sort_keys=True` + 固定分隔符保证同内容必得同哈希。哈希是**逐行**的，不是整文件。

`legacy_hashes` 来自 `read_git_jsonl(repo, "HEAD^", "data/sft/train.jsonl")`
（:206-207，连 validation 一起读）——**从 git 历史里读旧数据**，而不是读工作区文件。
然后处理 merged 来源时：

```python
if source.startswith("merged") and content_hash(row) in legacy_hashes:
    continue
```

**merged 包基本上把 `current_v4` 又带了一份**：878 条里 428 条和旧数据字节级相同，
直接丢，只剩 450。

**第二步：同题择优（−258）**

剩下的按 `task_id` 分组，每组用 `_candidate_key`（:81）选一个赢家：

```python
(
    guard_rejections == 0,   # 1. 没被守卫拒过优先
    gold_purchase,           # 2. 真的 gold 购买优先
    -steps,                  # 3. 步数少优先
    _tie_rank(seed, task_id, row_hash),   # 4. sha256(f"{seed}:{task_id}:{row_hash}") 抽签
)
```

`quality_features`（:52）算的就是前三个：`guard_rejections / gold_purchase /
steps / searches / opens / unique_candidates`。

### 3.3 合并算术

```
merged 包                          878
  − 内容哈希命中旧 data/sft/        428   ← legacy_rows_removed_from_merged
  = 存活 merged                     450
current_v4                       1,000
  ──────────────────────────────────────
  候选合计                         1,450   ← candidate_rows_after_legacy_removal
  − 同题重复组里被淘汰               258
  = 最终                           1,192
```

验证：`1,000 + 450 − 258 = 1,192` ✓（不是 `1,000 + 878 − 258 = 1,620`）。

`source_rows` 实测：`current_v4_train 800 / current_v4_validation 200 /
merged_v4_train 405 / merged_v4_validation 45`。

258 个重复组的胜负分布：

| 赢家 | 输家 | 组数 |
|---|---|---:|
| current | merged | **189** |
| merged | current | **69** |
| 合计 | | **258** |

（等价说法：merged 被淘汰 189 条、current 被淘汰 69 条。）

### 3.4 一个必须知道的副作用：94 组是纯抽签

258 个**跨源**重复组里，**94 组四项质量指标完全相同**（例：task 781，两边都是
0 守卫拒绝 / gold / 9 步 / 1 搜索 / 1 打开 / 1 唯一候选），结果**完全由种子
20260810 的抽签决定**。

按来源拆最终 1,192 条的终端 message 形式，正好对上：

| 来源 | data/sft/ 里有同内容行？ | 条数 |
|---|---|---:|
| `current_v4_train` | 是 | 745 |
| `current_v4_validation` | 是 | 186 |
| `merged_v4_train` | 否 | 234 |
| `merged_v4_validation` | 否 | 27 |

`（current 931 条哈希 == 旧数据）/（merged 261 条哈希 != 旧数据）`。

**最终数据里 merged 来源占 261 / 1,192 ≈ 21.9%。** 而
`docs/research/sft_data_mixing_diagnosis.md` 测出：正是这部分的行在 200 题 held-out
上把成绩从 142/200 拉到 112/200（merged 来源的 assistant 平均 264.9 字符、
非空 assistant 回合 3.98，而 original 是 1,218.8 字符、8.37 回合）。

也就是说：**这 94 次抽签的实际影响可能远大于它看起来的样子**，因为质量特征里
没有"assistant 自然语言长度"这一项。

### 3.5 血缘核实（可复现的检查）

```python
import hashlib, json

def ch(row):
    p = {"messages": row.get("messages"), "tools": row.get("tools")}
    e = json.dumps(p, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(e.encode()).hexdigest()

legacy = {}
for p in ["data/sft/train.jsonl", "data/sft/validation.jsonl"]:
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); legacy[r["task_id"]] = ch(r)

v4 = [json.loads(l) for l in open("data/sft_pure_v4/all.jsonl", encoding="utf-8")]
same = sum(1 for r in v4 if legacy.get(r["task_id"]) == ch(r))
diff = sum(1 for r in v4 if r["task_id"] in legacy and legacy[r["task_id"]] != ch(r))
new  = sum(1 for r in v4 if r["task_id"] not in legacy)
print(same, diff, new)     # → 931 69 192
```

---

## 4. 工序 ④：难度打标

### 4.1 命令

```bash
export OPENAI_BASE_URL=... OPENAI_API_KEY=...
python scripts/label_sft_difficulty.py \
  --input  data/sft_pure_v4/all.jsonl \
  --output data/sft_pure_v4/difficulty_labels.jsonl \
  --model deepseek-v4-flash \
  --workers 3 --batch-size 5 --max-batch-chars 60000
```

批处理策略：默认 5 行/批、单批 JSON ≤60,000 字符；tool 内容超长时按
`头 3/4 + 截断标记 + 尾 1/4` 压缩（`--max-tool-content-chars`，默认不限）。
已完成的行按 task_id 跳过，可续跑。

### 4.2 提示词的核心要求

`SYSTEM_PROMPT`（`label_sft_difficulty.py:14`）**特别强调把"任务内在难度"和"教师轨迹
观察到复杂度"分开**，因为教师绕路不代表题难：

> Judge intrinsic TASK difficulty separately from the observed teacher TRAJECTORY
> complexity. The number of written constraints alone is weak evidence. …
> A long or repetitive teacher trajectory may be a teacher failure mode; do not
> automatically call the task hard for that reason.

每条 label 的字段：

| 字段 | 取值 | 用途 |
|---|---|---|
| `difficulty` | simple / medium / hard | **决定进哪个桶** |
| `trajectory_complexity` | simple / medium / hard | 只做观察记录 |
| `needs_query_rewrite` | bool | **参与分桶** |
| `needs_candidate_comparison` | bool | **参与分桶** |
| `domain` | 中文品类名 | 分析 |
| `price_band` | low/medium/high/unknown | 分析 |
| `variant_complexity` / `budget_risk` / `loop_risk` | low/medium/high | 分析 |
| `confidence` | 0–1 | 质检 |
| `evidence` | ≤80 中文字 | 人工复核 |

`validate_labels`（:90）要求 task_id 集合完全匹配、枚举值合法、confidence ∈ [0,1]，
不合格的**整批丢弃**并记录（可重试）。

### 4.3 实测分布

| difficulty | 条数 | 占比 |
|---|---:|---:|
| simple | 284 | 23.8% |
| medium | 798 | 66.9% |
| hard | 110 | 9.2% |

平均 confidence 0.894；低于 0.8 的只有 4 条。1,186 条用完整轨迹标注，6 条
（task_id `983, 10574, 10646, 14803, 18595, 20915`）因超长用了压缩 tool 内容。

`metadata.json` 里还留了一条**没有执行的决定**：请求过的 30/50/20 配比
`status: "not_materialized"`，理由是"hard 只有 110 条，无重复采样的话最多凑 550 行，
会丢掉 642 条合法数据"。所以**保持了自然分布**。

---

## 5. 工序 ⑤：分桶 + 累积课程

### 5.1 命令

```bash
python scripts/prepare_sft_curriculum.py \
  --source     data/sft_pure_v4/all.jsonl \
  --labels     data/sft_pure_v4/difficulty_labels.jsonl \
  --evaluation data/evaluation/tasks.jsonl \
  --output     data/sft_curriculum/manifest.json \
  --seed 20260814 --validation-ratio 0.1
```

### 5.2 逐行复审 `_validate_row`（:53）

比采集时的验收更严，因为它假设拿到的是**已清洗**的行：

- tool 内容里**不许**出现 `guard rejected` / `action guard` / `动作守卫拒绝` /
  `guard_violation` → 出现就 `raise`（清洗漏网当场炸）
- 每个 assistant 回合 ≤1 个 tool call
- `arguments` 必须是合法 JSON object，工具名必须在该行 `tools` 里
- **恰好一个 `buy_now`，且必须是最后一个 tool call**
- 最后一条 message 必须是 `role=tool`，且其 `tool_call_id` 等于那个 buy_now 的 id
- 终局 content 必须是 `购买已完成。` **或**含 `page_type: terminal`

同时 `build_manifest`（:130）检查：SFT 行与 difficulty label 的 task_id 集合完全
相同、无重复、**与评测集零重叠**（有重叠直接 `ValueError`）。

> 终局 content 那条校验是**故意放宽的**：严格只认 `购买已完成。` 会毙掉 931 条
> 合规数据（§3.4 的两派来源），所以两种都收。

### 5.3 分桶 `_bucket`（:100）

```python
if difficulty == "simple":
    return "foundation"
if difficulty == "medium" and not needs_query_rewrite and not needs_candidate_comparison:
    return "constraints"
return "strategy"
```

| 桶 | 规则 | 行数 |
|---|---|---:|
| `foundation` | simple | 284 |
| `constraints` | medium 且不需要改写查询、不需要比较候选 | 603 |
| `strategy` | 其余（medium 需改写/比较 + 全部 hard） | 305 |
| | **合计** | **1,192** |

### 5.4 桶内切分 `_split`（:112）

```python
ordered = sorted(ids, key=lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest())
validation_count = int(len(ordered) * ratio + 0.5)
validation = ordered[:validation_count]
train      = ordered[validation_count:]
```

**每个桶内部**独立分层（seed `20260814`，与工序 ② 的 42 不同）：

| 桶 | train | validation | 合计 |
|---|---:|---:|---:|
| foundation | 256 | 28 | 284 |
| constraints | 543 | 60 | 603 |
| strategy | 274 | 31 | 305 |
| **合计** | **1,073** | **119** | **1,192** |

### 5.5 累积阶段 `STAGE_CONFIG`（:14）

| stage | buckets | train | dev | LR | epochs |
|---|---|---:|---:|---:|---:|
| A | foundation | 256 | 28 | 1e-4 | 1 |
| B | foundation + constraints | **799** | 88 | 7e-5 | 1 |
| C | foundation + constraints + strategy | **1,073** | 119 | 5e-5 | 1 |

**关键设计：manifest 里存的是 task_id 列表，不是行本身。** 数据文件永远只有
`all.jsonl` 那一份，阶段差异靠"选哪些 task_id"表达。`train_lora_sft.py:434` 会校验
`matched != len(train_task_ids)` 就硬失败——保证清单和数据严格对齐。

### 5.6 review_flags（只标记，不删数据）

`build_manifest` 顺带算出四类可疑样本的 task_id：

| flag | 含义 | 条数 |
|---|---|---:|
| `candidate_comparison_under_evidenced` | 标了需要比较候选，但实际只打开了 <2 个商品 | 168 |
| `trajectory_long` | 步数超过该难度的上限（simple 10 / medium 16 / hard 25） | 27 |
| `search_heavy` | 搜索次数超过上限（2 / 4 / 6） | 5 |
| `consecutive_exact_action` | 存在连续两次完全相同的动作 | 2 |

设计上**刻意保留**这些"可疑但合法"的难例，只给人工复核用。

### 5.7 manifest 结构

```jsonc
{
  "schema_version": "shopping-sft-curriculum-v1",
  "split_seed": 20260814,
  "validation_ratio": 0.1,
  "counts": { "rows": 1192, "train": 1073, "validation": 119, "evaluation_overlap": 0 },
  "buckets": {
    "foundation":  { "rows": 284, "train_task_ids": [...], "validation_task_ids": [...] },
    "constraints": { "rows": 603, "train_task_ids": [...], "validation_task_ids": [...] },
    "strategy":    { "rows": 305, "train_task_ids": [...], "validation_task_ids": [...] }
  },
  "stages": {
    "a": { "buckets": ["foundation"], "train_rows": 256, "validation_rows": 28, "epochs": 1.0, "learning_rate": 1e-4 },
    "b": { "buckets": ["foundation","constraints"], "train_rows": 799, "validation_rows": 88, "epochs": 1.0, "learning_rate": 7e-5 },
    "c": { "buckets": ["foundation","constraints","strategy"], "train_rows": 1073, "validation_rows": 119, "epochs": 1.0, "learning_rate": 5e-5 }
  },
  "review_flags": { "candidate_comparison_under_evidenced": [...], ... },
  "source":   { "path": "data/sft_pure_v4/all.jsonl", "sha256": "7b70db71…" },
  "labels":   { "path": "data/sft_pure_v4/difficulty_labels.jsonl", "sha256": "50da1b90…" },
  "evaluation": { "path": "data/evaluation/tasks.jsonl", "sha256": "d99112a2…" }
}
```

### 5.8 阶段之间的隐含重加权

因为阶段是累积的，每个桶被"过"的次数不同：

| 桶 | 出现在 | 训练次数 | 名义 LR 曝光 |
|---|---|---:|---|
| foundation | A, B, C | 3 | 1e-4 + 7e-5 + 5e-5 = 2.2e-4 |
| constraints | B, C | 2 | 7e-5 + 5e-5 = 1.2e-4 |
| strategy | C | 1 | 5e-5 |

总计 256×3 + 543×2 + 274×1 = **2,128 次曝光**（1,073 行数据）。

按**难度**看偏移（更有意义，因为桶和难度不是一一对应）：

| difficulty | train 行数 | 行占比 | 曝光次数 | 曝光占比 |
|---|---:|---:|---:|---:|
| simple | 256 | 23.9% | 768 | **36.1%** |
| medium | 717 | 66.8% | 1,260 | **59.2%** |
| hard | 100 | 9.3% | 100 | **4.7%** |

（按桶看则是 foundation 768 / constraints 1,086 / strategy 274，即 36.1% / 51.0% /
12.9%——`strategy` 桶里混着 hard 和"需改写/需比较的 medium"，所以桶占比和难度占比
不同。）

`data/sft_curriculum/README.md` 明确把"偏向简单任务"列为课程收益不明显的**可能
原因之一**，并标注这是假设、不是因果结论。hard 只有 100 行且只被过一遍，
曝光占比被压到 4.7%。

---

## 6. 复现检查清单

> **能复现到哪一步**：工序 ③④⑤ 完全可复现（纯本地 + git）；
> 工序 ①② 的"从 `raw.jsonl` 到 1,026 条"可复现；
> **"从 1,026 到 1,000"这一步不可复现**——产出它的脚本不在仓库里，见 §2.1.1。
> 所以真正的端到端复现是：自己跑采集（拿到自己的 `raw.jsonl` 和 N 条合格数据），
> 自己决定切多少，然后从工序 ③ 接上。

跑完之后逐项核对：

```bash
# 1. 行数与哈希
python -c "
import json,hashlib
for p in ['data/sft_pure_v4/all.jsonl','data/sft_pure_v4/difficulty_labels.jsonl',
          'data/sft_curriculum/manifest.json']:
    b=open(p,'rb').read()
    print(p, b.count(b'\n'), hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()[:16])
"
# 期望: all.jsonl 1192 / 7b70db7174daf766 ; labels 1192 / 50da1b9014e87684

# 2. 防泄漏（必须是 0）
python -c "
import json
ev={json.loads(l)['task_id'] for l in open('data/evaluation/tasks.jsonl',encoding='utf-8')}
v4={json.loads(l)['task_id'] for l in open('data/sft_pure_v4/all.jsonl',encoding='utf-8')}
print('v4 ∩ Final-200 =', len(v4&ev))          # → 0
"

# 3. 课程清单计数
python -c "
import json; m=json.load(open('data/sft_curriculum/manifest.json',encoding='utf-8'))
print(m['counts'])
for s in 'abc': print(s, m['stages'][s]['train_rows'], m['stages'][s]['validation_rows'])
"
# 期望: {'rows':1192,'train':1073,'validation':119,'evaluation_overlap':0}
#       a 256 28 / b 799 88 / c 1073 119

# 4. 数据与清单一致（train_lora_sft.py 会自己做，也可单独验）
python -c "
import json
m=json.load(open('data/sft_curriculum/manifest.json',encoding='utf-8'))
rows={json.loads(l)['task_id'] for l in open('data/sft_pure_v4/all.jsonl',encoding='utf-8')}
ids={t for s in 'abc' for b in m['stages'][s]['buckets'] for t in m['buckets'][b]['train_task_ids']}
print('清单 train 数 =', len(ids), '| 全部存在于数据 =', ids <= rows)
"
```

### 已知的环境坑（Windows 工作区）

- 仓库用 `core.autocrlf=true`，工作区 JSONL 是 **CRLF**，所以直接 `sha256sum` 算出的
  哈希**不等于**文档里记的值。用 `git show HEAD:<path>` 拿 LF 版本，或如上先把
  `\r\n` 归一成 `\n` 再算。
- `data/sft/metadata.json` 里的 sha256（`8c3a6ff0…` / `9525cc2f…`）是**采集时的
  LF 版本**哈希。

---

## 7. 一句话总结

> 采集产出的是**审计日志**（`raw.jsonl`，含守卫、投影、原始 observation）；
> `build_sft_row` 把它削成**只含合法动作的白名单行**；
> `merge_sft_pure_v4` 用**内容哈希**去掉跨批次的重复、用**质量优先级 + 抽签**在
> 同题轨迹里选一条；
> `label_sft_difficulty` 用另一次 LLM 调用把"任务难度"和"轨迹复杂度"分开标注；
> `prepare_sft_curriculum` 把难度映射成三个桶、桶内分层切分、拼成三个**累积阶段**。
>
> 全程每一步都留哈希和拒绝原因；任何一条数据被扔掉，都能追到是哪条规则扔的。

---

## 附：脚本 → 模块对照

| 脚本 | 主要依赖 | 核心函数 |
|---|---|---|
| `scripts/collect_sft_data.py` | `evaluation/rollout.py`, `collection/sft.py` | `collect_for_task`, `collect_until_target`, `build_collection_artifacts` |
| `scripts/merge_sft_pure_v4.py` | 无（纯本地 + git） | `content_hash`, `_candidate_key`, `merge_sources` |
| `scripts/label_sft_difficulty.py` | `evaluation/model_client.py` | `make_batches`, `label_batch`, `validate_labels` |
| `scripts/prepare_sft_curriculum.py` | 无 | `_validate_row`, `_bucket`, `_split`, `build_manifest` |
| `scripts/run_sft_curriculum.py` | `scripts/train_lora_sft.py`, `scripts/merge_lora_adapter.py` | `build_stage_commands`（⑥ 之后的训练入口） |

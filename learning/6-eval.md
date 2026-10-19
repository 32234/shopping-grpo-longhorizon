# 06 · 评估：四把尺子量一条轨迹

一条轨迹跑完，问四个**互不替代**的问题：

```
买对了吗？             → Reward v3（代码算）
用户要的每条满足了吗？   → Rubric（代码出题 + Flash 选题 + Pro 判卷）
过程像个会买东西的人吗？ → 五维（Pro 打分）
动作合法吗？           → 确定性指标（代码算）
```

**不合成总分。** 因为会互相掩盖：买对了但只看了 1 个候选、过程漂亮但选错规格、Reward 满分但用户要的"礼盒"没被标注——这三件事一个总分都表达不了。

---

## 0. 先看清一件事：能跑的和文档写的不一样

`bash scripts/evaluate.sh NAME` 实际只做三件事：

```
① evaluate_shop_benchmark.py   → 200 条轨迹 + summary.json
② build_eval_report.py         → report.html
```

**Rubric 冻结、Pro Judge、四面板拼装、配对比较，在 `src/shopping_grpo/evaluation/`
里是完整的库代码，但没有任何脚本调用它们**（没有 `__main__`，`scripts/` 里也没有引用）。

所以：

| 面板 | 能直接跑吗 |
|---|---|
| ① Reward 与终局 | ✅ `summary.json` |
| ② Query Rubric | ❌ 只有库函数 |
| ③ 五维 | ❌ 只有库函数 |
| ④ 确定性 | ⚠️ 部分（步数、守卫数、上下文投影） |

要跑完整四面板，得自己写一个 ~100 行的 runner（§7 给了骨架）。

---

## 1. Rollout：200 题怎么跑

```bash
bash scripts/start_environment.sh          # ShopSimulator :5700
bash scripts/serve_model.sh <model>        # vLLM :8000，服务名 shopping-agent
bash scripts/evaluate.sh grpo
```

`evaluate_shop_benchmark.py` 的评测口径（和训练不同）：

| 项 | 值 | 为什么 |
|---|---|---|
| 每题 rollout | **1 次** | 可复现 |
| temperature / top_p | **0.0 / 1.0** | 贪心，不能靠采样碰运气 |
| max_steps | 35 | 和训练一致 |
| max_tokens | 512 | 每回合上限 |
| context_window | 24576 | |
| observation 预算 | 1536 / 4096 / 768 | 和训练一致 |
| context compaction | 关闭 | |

复用 `rollout.py:collect_for_task`——和 SFT 采集**同一套代码**。唯一的区别是模型不同、参数不同。

产物：

```
outputs/evaluation/grpo/
  trajectories.jsonl    200 行，每行一条完整轨迹
  summary.json          单模型汇总
  report.html           可视化
```

---

## 2. 一个例子：task 8187 走完四个面板

Query：求一对卡通-永结同心款的高档酒红色木梳，希望配备礼盒，能作为新娘结婚的陪嫁物品，价格在20元左右。

### 面板 ①：Reward（代码，不重算）

环境在 `buy_now` 那一刻已经算好了，评估只是**读**：

```jsonc
"reward_detail": {
  "reward_type": "gold_purchase", "reward_valid": true,
  "terminal_utility": 1.0, "weighted_score": 1.0,
  "hard_gates": {"category": "pass", "budget": "pass"},
  "dimension_scores": {"brand": 0.0, "model": 0.0,
                       "core_functions": 1.0, "key_options": 1.0}
}
```

> **评估不会重新跑 `evaluate_purchase`。** 它信任环境返回值，只校验 `reward_version == v3`。
> 换环境版本，历史结果就不能直接比。

### 面板 ②：Rubric（三方分工）

**第一步：代码穷举候选（7 条）**

| 候选 | 从哪来 |
|---|---|
| 品类：新娘配件 | gold 商品的 category |
| 高档、结婚、陪嫁、卡通-永结同心 | instruction 的人工标注 attributes |
| 【卡通-永结同心】2个装 | instruction_options，到 gold 的 customization_options 里定位轴 |
| 20 元左右 | **正则扫 Query**（"左右"→ soft） |

品牌和型号候选是 0 条——别名表里没有"漫步喜"，token 交集为空。

**第二步：Flash 选题（7 → 5）**

它只能选、不能造，而且必须贴 Query 原文：

| rubric | hardness | 要求 |
|---|---|---|
| r0001 | hard | 品类应为新娘配件 |
| r0002 | hard | 支持高档 |
| r0003 | hard | 支持卡通-永结同心 |
| r0004 | hard | 选择【卡通-永结同心】2个装 |
| r0005 | **soft** | 价格倾向 20 元左右 |

**"结婚/陪嫁"两条被丢掉**——品类"新娘配件"已经覆盖了这两个语义，重复计分没意义。

代码当场验：`candidate_id` 存在、`query_quote` 逐字出现在 Query 里、
`field_path / operator / expected_value` 从候选回抄（Flash 改了不生效）。

**第三步：Pro 逐条判**

```
r0001 satisfied  证据 e0002, e0010
r0002 satisfied  证据 e0002, e0005
r0003 satisfied  证据 e0002
r0004 satisfied  证据 e0009
r0005 satisfied  证据 e0009
```

四个状态：`satisfied / violated / unknown / not_applicable`。

> `unknown` 不是"没满足"，是"轨迹里看不出来"。这个区分很重要——它会单独统计。

**Rubric 没有分数。** 只有状态表。

### 面板 ③：五维（Pro 打分）

它拿到的输入：

```jsonc
{
  "query": "...",
  "rubric": [r0001…r0005],
  "dimension_spec": {"search_strategy": {"allowed_scores": [0,1,2]}, …},
  "frozen_error_taxonomy": [22 项],
  "actor_visible_trajectory": {"events": [e0001 … e0010]},
  "terminal_state": {"done": true, "over": true},      // 只有两个布尔
  "judge_visible_metrics": {actions_and_efficiency, repetition, legality, context}
}
```

**不给**：Reward、gold ASIN、原始 observation、购买记录、其他模型的结果。
连 `termination_reason` 都剥了——`gold_purchase` 这个词本身就是答案。

输出：

| 维度 | 分 | 理由 |
|---|---:|---|
| search_strategy | 2 | 搜索覆盖卡通、永结同心、木梳、礼盒 |
| candidate_utilization | **1** | 只打开一个候选，比较不足 |
| evidence_verification | **1** | 规格和价格已确认，详情证据有限 |
| decision_quality | 2 | 硬约束、规格、预算均满足 |
| termination_efficiency | 2 | 10 步内完成，无循环 |

每个分都必须带 `reason` 和 `evidence_event_ids`；**引用的 event_id 必须真实存在**，伪造直接拒。
**禁止输出总分**——代码见到 `total_score` / `overall_score` / `weighted_total` 就报错。

### 面板 ④：确定性（纯代码计数）

```jsonc
{
  "actions_and_efficiency": {"executed_tool_steps": 10, "action_attempts": 10,
                             "opened_candidate_count": 1, "visible_search_candidate_count": 20},
  "repetition": {"duplicate_canonical_action_count": 0, "duplicate_search_query_count": 0},
  "legality":   {"guard_rejection_count": 0, "malformed_tool_call_count": 0},
  "context":    {"projection_count": 9, "truncated_observation_count": 2,
                 "critical_footer_failures": 0, "max_input_tokens": 9874},
  "validity":   {"infrastructure_invalid": false, "judge_eligible": true}
}
```

**没有一项经过 LLM。**

### 拼起来

```jsonc
{
  "task_id": 8187,
  "reward_and_terminal":   { … },
  "requirement_rubric":    {rubrics, assessments, reward_rubric_disagreement: false},
  "trajectory_quality":    {judge_status: "valid", dimension_scores, errors, overall_diagnosis},
  "deterministic":         { … }
}
```

四块各占一个字段，**不合成**。

---

## 3. 200 题跑完，你手里是什么

### `summary.json`（真实数据，GRPO step 100）

```json
{
  "expected_tasks": 200, "completed_tasks": 200,
  "done_tasks": 193, "done_rate": 0.965,

  "strict_successes": 124, "strict_success_rate": 0.62,
  "purchase_successes": 125, "purchase_success_rate": 0.625,
  "reward_valid_tasks": 192, "reward_valid_rate": 0.96,

  "total_final_reward": 103.1684, "mean_final_reward": 0.5158,
  "mean_weighted_score": 0.7496,

  "average_steps": 11.85, "guard_rejections": 38,

  "reward_type_counts": {
    "gold_purchase": 124, "valid_alternative_purchase": 1,
    "partial_alternative_purchase": 34, "wrong_purchase": 5,
    "repeat_loop": 25, "max_steps": 3,
    "reward_unverifiable": 1, "unknown": 7
  }
}
```

**17 行，就是 200 道题的全部。**

### 三个 headline 的关系

```
strict_success_rate ≤ purchase_success_rate ≤ done_rate
0.620 ≤ 0.625 ≤ 0.965   ✓
```

| 指标 | 回答 |
|---|---|
| `done_rate` | 模型会不会走完流程（协议能力） |
| `purchase_success_rate` | 买的东西满足需求吗（含替代品） |
| `strict_success_rate` | 买的是不是 gold（严格） |

SFT：`0.605 / 0.605 / 0.965`
Baseline：`0.0 / 0.0 / 0.18` ← **会走完的 18% 全是 repeat_loop**

### 三个模型的对照

| | Baseline | SFT | GRPO |
|---|---:|---:|---:|
| 严格成功率 | 0.0% | 60.5% | **62.0%** |
| 完成率 | **18.0%** | 96.5% | 96.5% |
| 平均步数 | 5.9 | 12.3 | **11.9** |
| 守卫拒绝 | **752** | 52 | **38** |
| 买错 | 0 | 7 | **5** |
| 循环耗尽 | 34 | 27 | 25 |
| 未终局 | **164** | 7 | 7 |

三个故事一眼能看出：

1. **Baseline 不会用工具**——82% 跑不完，752 次非法动作。
2. **SFT 学会了流程**——完成率 18% → 96.5%，代价是步数翻倍。
3. **GRPO 在收尾**——步数压回来、守卫降、买错降，成功率只涨 1.5 个点。

### 四面板版本会多什么

`evaluation_summary.json` 额外三段（需要自己补 runner）：

```jsonc
"requirement_rubric": {
  "status_counts_by_hardness": {"hard": {"satisfied": 690, "violated": 31, "unknown": 22, "not_applicable": 5},
                                "soft": {"satisfied": 171, "violated": 13, "unknown": 8, "not_applicable": 4}},
  "reward_rubric_disagreement_tasks": 7
},
"trajectory_quality": {
  "judge_coverage_rate": 0.975,
  "dimensions": {"search_strategy": {"score_counts": {"0":3,"1":41,"2":151},
                                     "mean_score_among_valid_judges": 1.759}, …},
  "primary_error_counts": {"repeat_loop": 11, "premature_abstain": 8, …}
},
"deterministic": {
  "average_executed_steps_fixed_denominator": 11.85,
  "infrastructure_invalid_tasks": 5
}
```

---

## 4. 四个面板的统计单位不一样

这是最容易混的地方：

| 面板 | 单位 | 分母 |
|---|---|---|
| ① Reward | **题** | 200 |
| ② Rubric | **条**（rubric 项） | 全部条数 ≈ 1000 |
| ③ 五维 | **题**（只算有效判卷） | `valid` 数 ≈ 195 |
| ④ 确定性 | 题 / 次 | 200 |

所以：

```
gold_purchase_rate = 124/200        ← 题
satisfied = 861                     ← 条要求，不是 861 道题
mean_score_among_valid_judges       ← 195 题
total_guard_rejections = 38         ← 次数总和
```

**不要拿 rubric 的条数去除 200。**

---

## 5. 冲突时怎么办

代码只在四种情况下标 `reward_rubric_disagreement = true`：

```
Reward 成功，但有 rubric violated
Reward 成功，但有 hard rubric violated
wrong_purchase，但所有 hard rubric 都 satisfied
partial，但所有 rubric 都 satisfied
```

**两个面板各自保留，谁也不覆盖谁。**

最典型的是第一种：买对了，但用户正文里某条要求（比如"礼盒"）没被 Reward 标注——
Reward 给 1.0，Rubric 判 violated。

---

## 6. 配对比较：GRPO 到底改了什么

`compare_evaluation_runs` 按 `task_id` 对齐：

```jsonc
"sft_to_grpo": {
  "paired_tasks": 200,
  "strict_success_transitions": {
    "failure_to_failure": 71, "failure_to_success": 12,
    "success_to_failure": 9,  "success_to_success": 109
  },
  "reward_type_transitions": {"partial_alternative_purchase -> gold_purchase": 6, …},
  "hard_violation_delta": {"mean_delta": -0.18, "improved_tasks": 17, "worsened_tasks": 9},
  "executed_tool_steps": {"mean_delta": -0.49},
  "guard_rejections":   {"mean_delta": -0.07}
}
```

**这张表才是结论。** 不是看 `62.0 − 60.5`，而是看：

```
12 题从失败变成功，9 题从成功变失败，净 +3
McNemar p ≈ 0.664 → 统计上不可区分
```

`docs/research/sft_grpo_diagnosis.md` 就是基于这个算出来的。

---

## 7. 复现

### 能直接跑的

```bash
bash scripts/setup.sh
bash scripts/start_environment.sh
bash scripts/serve_model.sh outputs/models/grpo-merged
bash scripts/evaluate.sh grpo          # → trajectories.jsonl + summary.json + report.html
bash scripts/report.sh grpo            # 补生成 HTML
```

### 补四面板 runner 需要的东西

```python
# 1) TaskFacts + Rubric（每题一次，三模型共用）
from web_agent_site.engine.engine import load_products
from web_agent_site.engine.goal import get_goals
all_products, product_item_dict, prices, _ = load_products("data/items_eval_train.json")
goals = get_goals(all_products, prices)

facts = task_facts_from_environment(task_ids, goals, product_item_dict)
cands = [extract_rubric_candidates(f) for f in facts]
# 调 V4 Flash → materialize_rubric_bundle

# 2) 每个模型：规范化 → 硬检查 → Judge-safe → V4 Pro → 拼装
norm  = normalize_trajectory(raw)
met   = compute_deterministic_metrics(norm)
msgs  = build_trajectory_judge_messages(normalized=norm, rubric_bundle=b,
                                        deterministic_metrics=met)
judge = validate_judge_result(client.complete_json(msgs)["result"], …)
rec   = assemble_task_evaluation(…)

# 3) 汇总 + 配对
summarize_evaluations(expected_task_ids=range(200), evaluations=[...])
compare_evaluation_runs(expected_task_ids=..., runs={"sft": …, "grpo": …})
```

**一个障碍**：`task_facts_from_environment` 需要环境的 `goals` 和 `product_item_dict`，
也就是要能 import `web_agent_site.engine`（那个隔离的 Python 3.10 venv）。
要么在那个 venv 里跑，要么先把 goals 导出成 JSON。

---

## 8. 三个口径陷阱

**① 均值的分母在三处不一样**

| 来源 | 平均 Reward 的分母 |
|---|---|
| `evaluation/summary.py`（`evaluate.sh` 用的） | **完成数** |
| `evaluation/results.py`（四面板） | **200**，字段名带 `_fixed_denominator` |
| README 表格 | 未说明 |

**② 五维均值的分母是"有效判卷数"**

两个模型都 1.8，但一个覆盖 99%、一个覆盖 80%——不能直接比。
必须和 `judge_coverage_rate` 一起看。

**③ `experiments/*/summary.json` 是旧 schema**

它的 `schema_version` 是 `shopping-experiment-summary-v1`，**仓库里没有脚本生成这个**。
只有 17 个键，没有上下文投影那块。当前 `evaluate.sh` 产的是 26 键版本。

---

## 9. 一句话

> **评估 = 200 条轨迹 + 四把尺子。**
> **Reward 说结果，Rubric 说每条需求，五维说过程，确定性说合法性——单位不同、分母不同、互不相加。**
> **配对比较看的是 12 升 9 降，不是两个百分点的差；而且 Rubric/五维这两把尺子，仓库里只有库代码，没有入口。**

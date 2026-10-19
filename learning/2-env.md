# 02 · ShopSimulator 模拟环境：一局完整交互

这篇讲**环境侧**：一个 task_id 从 reset 到终局，环境内部到底发生了什么、改了哪些状态、
最后怎么算出一个 reward。全篇以 **task 65** 的真实数据为例。

读完之后你应该能：自己写一个 HTTP 客户端跑通一局、看懂 `reward_detail` 每个字段的来源、
知道哪些字段是 gold 泄露点必须剥离。

前置：01 篇讲的是"怎么采集数据"，这篇讲"环境本身是什么"。

---

## 1. 环境是怎么起来的

`bash scripts/start_environment.sh` → `environments/ShopSimulator/shop_env/start.sh`：

```bash
export SHOP_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1
export SHOP_ENV_CONFIG=configs/environment.json
export SHOP_SEARCH_INDEX=search_engine/products.sqlite3
export SHOP_MAX_STEPS=35
export SHOPSIM_ENV_SLOTS=8      # 并发槽位数
export SHOPSIM_PORT=5700
exec python pack_api.py
```

`pack_api.py:127 initialize_environments()` 的关键结构：

```python
shared_server = None
for i in range(env_max_num):            # 8 个槽位
    env = WebAgentTextEnv(observation_mode='text', split="train",
                          num_products=DEBUG_PROD_SIZE,   # None → 全量语料
                          server=shared_server,           # 第 1 个为 None，之后复用
                          session_prefix=f"slot-{i}")
    if shared_server is None:
        shared_server = env.server       # 只加载一次
```

**一个 `SimServer` + 8 个轻量 `WebAgentTextEnv` 槽位。** 商品语料（23,421 条，140 MB JSON）
和 BM25 索引只加载一次，这是启动慢的原因。

---

## 2. 数据源：一份 JSON，两种产物

唯一的数据文件是 `shop_env/data/items_eval_train.json`
（由 `fine_items_eval_train_all.json.gz` 解压，sha256 `57b10950…559f`，23,421 条）。

一条原始记录（task 65，数组下标 65）：

```jsonc
{
  // ── 外层：商品元数据 ──
  "asin": "886076453056",
  "tag": "eval",
  "title": "得力水彩笔幼儿园专用可水洗24色儿童彩笔无毒36色小学生专用涂色填色绘画笔不脏手水溶性易可洗美术绘画套装",
  "shop_name": "得力文具专卖店",
  "category": "文具电教/文化用品/商务用品›画具/画材用品›水彩笔",
  "attribute": ["幼儿园专用", "不脏手", "无毒", "水溶性", "绘画"],
  "pricing": [16.0, 17.0],
  "customization_options": {
    "颜色分类": [
      {"value": "24色三角粗杆【加量-可水洗】送绘画4件", "price": 17.0, "price_string": "17"},
      {"value": "36色【加量粗杆可水洗】", "price": 21.0, ...}, ...
    ]
  },
  "full_description": "...", "images": [...], "sub_title": "", "user_persona": {...},

  // ── 内层：这一条商品对应的"任务"标注 ──
  "instructions": [{
      "asin": "886076453056",
      "instruction": "我想要买一款适合幼儿园小朋友绘画使用的水彩笔，24色，三角粗杆设计，具有水溶性可以水洗不脏手，安全无毒，还附赠4件绘画工具，价格在20元以内。",
      "instruction_simple": "我想要买一款适合幼儿园小朋友和小学生绘画使用的水彩笔，价格在20元以内。",
      "attributes": ["幼儿园专用", "不脏手", "无毒", "水溶性", "绘画"],   // ← 与外层 attribute 完全相同
      "instruction_options": ["24色三角粗杆【加量-可水洗】送绘画4件"],
      "options": [...],           // 与 instruction_options 相同
      "worker_id": 1
  }]
}
```

**注意 `instructions[0].attributes` 和外层 `attribute` 内容相等**（逐值验证过）。
看起来冗余，但**用途完全不同**，见 §4。

这份 JSON 被**分叉处理两次**，产出两份独立的东西：

```
items_eval_train.json[N]
   │
   ├─ ① load_products ──→ product_item_dict[asin] ──→ 页面显示什么
   │
   └─ ② get_existed_goals ──→ goals[N] ──→ 打多少分
              （用到 instructions[0]）
```

`task_id` **就是数组下标**：`goals[65]` 来自第 65 条记录，`goals[8187]` 来自第 8187 条。
（验证：`goals[8187].asin == "831919364430"`，正是 `docs/evaluation.md` 里那把木梳。）

---

## 3. 产物 ①：商品字典（决定页面）

`engine.py:239 load_products` 对每条记录：**删 14 个字段**（`clean_product_keys`，:219）
—— 包括 **`brand` 和 `brand_url`** —— 再加一批派生字段（:267-332）：

| 新字段 | 值 | 来源 |
|---|---|---|
| `Title` | 同 `title` | 复制 |
| `Description` | `full_description` | 复制 |
| `BulletPoints` | `[small_description]` | 复制 |
| `Reviews` | `[]` | **写死空** |
| `Rating` | `'N.A.'` | **写死** |
| `Attributes` | `attribute` | 复制 |
| `Price` | `"16.0 to 17.0"` | 由 `pricing` 拼串 |
| `options` | `{"颜色分类": ["24色三角粗杆【加量-可水洗】送绘画4件", ...]}` | 从 `customization_options` **拆出，值 `.lower()`，`/`→`\|`** |
| `option_to_price` | `{"24色三角粗杆【加量-可水洗】送绘画4件": 17.0, ...}` | 同上 |

三个后果：

1. **商品没有品牌字段**。页面上的 `brand:` 走 `product.get("brand") or product.get("shop_name")`，
   所以显示的是**店名**。Reward 的品牌比对也用 `brand|shop_name`。
2. **`Reviews` 永远是空的**。这个工具存在、按钮存在，但没有数据。
3. **规格值被小写化**。`click[...]` 的参数必须是这个低配版，否则守卫匹配不上。

---

## 4. 产物 ②：goal（决定得分）

`goal.py:63 get_existed_goals`，每条记录生成一个 goal：

```python
goal = {
    'asin':               item['asin'],                      # ← 商品
    'category':           item['category'],                  # ← 商品
    'name':               item['title'],                     # ← 商品
    'instruction_text':   product['instruction'],            # ← 标注
    'instruction_simple': product['instruction_simple'],
    'attributes':         product['attributes'],             # ← 标注
    'price_upper':        explicit_budget_from_instruction(instruction),   # ← 正则
    'goal_options':       product['instruction_options'],    # ← 标注
    'user_persona': ..., 'reason_key': ...,
}
goal.update(compile_reward_features(product, item))          # ← 叠一层
```

`compile_reward_features`（`reward_features.py:184`）加进来的才是**评分真正用的字段**：

| 字段 | task 65 的值 | 算法 |
|---|---|---|
| `category` | `"文具电教/…›水彩笔"` | = 商品 category |
| `expected_brand` | `[]` | 别名表里的词要**在 instruction 和 title+shop_name 两边都出现** |
| `expected_model` | `["24"]` | instruction 与 title/desc 里**都出现**的「含数字 token」交集 |
| `expected_core_functions` | `["幼儿园专用","不脏手","无毒","水溶性","绘画"]` | = `instructions[0].attributes` |
| `required_options_by_key` | `{"color": {"value":"24色三角粗杆【加量-可水洗】送绘画4件","source_axis":"颜色分类"}}` | 拿 `instruction_options` 的值去商品的 `customization_options` 里定位规格轴 |
| `unresolved_option_requirements` | `[]` | 定位不到唯一轴的 |

`price_upper` 用 `constraints.py:63 explicit_budget_from_instruction` 的正则，命中
`"价格在20元以内"` → `20.0`。（`左右` 会 ×1.1；`4k+` 这种开区间显式返回 `None`；
解不出就是 `None`，此时预算 gate 走"未声明"分支直接 PASS，**不是**不可验证。）

**`goal` 是"商品 + 需求"的合成体**：带 gold 的 asin/category/name，也带用户要求的
attributes/options/预算。整局只读，永不修改。

### 为什么内层 `attributes` 和外层 `attribute` 重复

| 字段 | 角色 |
|---|---|
| **外层** `attribute` | **被检方**：在候选商品上找这些词 |
| **内层** `attributes` → `goal.expected_core_functions` | **检方**：用户要求什么 |

gold 商品自己比自己当然满分——但 Reward 要处理"模型买了**别的**商品"的情况，
那时被检方换成那个商品的 `attribute`，检方仍是 gold 的标注。

---

## 5. reset：开局

```jsonc
请求 {"action": "reset", "idx": 65}
```

`shop_agent.py:22 _handle_reset_action` → `env.reset(idx=65)`：

```python
self.session = f"{session_prefix}-{idx}"      # "slot-3-65"：槽位 + 任务
self.history = self.history_init[:]           # 重置对话历史
self.instruction_text = goals[65]['instruction_text']
```

`SimServer.receive` 走 `not kwargs` 分支，建会话并**清零全部状态机**：

```python
user_sessions["slot-3-65"] = {
    'goal': goals[65],              # 只读
    'asin': None, 'options': {}, 'selected_price': None,
    'page': None, 'subpage': None,
    'asins': set(),                 # 打开过的商品
    'distinct_normalized_queries': set(),
    'search_result_asins': [], 'current_page_asins': [],
    'candidate_eligibility': {}, 'known_valid_asins': set(),
    'progress_tracker': EvidenceProgressTracker(max_steps=35, ...),   # 计数器全 0
    'actions': defaultdict(int),
}
```

**响应**（`pack_api` 包装成 `{"result": {...}}`）：

```jsonc
{
  "env_idx": 3,                      // ← 槽位，后续所有请求都要带
  "idx": 65,
  "instruction": "我想要买一款适合幼儿园小朋友绘画使用的水彩笔……",
  "instruction_simple": "...",
  "goal_options": ["24色三角粗杆【加量-可水洗】送绘画4件"],   // ⚠️ gold 规格
  "environment_version": "shopsimulator-environment-v2.1",
  "observation_state": {"page_type": "search_home", "actions": [], "search_available": true}
}
```

⚠️ **`goal_options` 是 gold**。项目侧只取 `instruction` 和 `observation_state`
（`rollout.py:307`），这个字段不读。

---

## 6. 交互循环：每个动作环境做什么

```jsonc
请求 {"action": "interact", "env_idx": 3, "response": "click[886076453056]"}
```

`WebAgentTextEnv.step(action)` 先 `parse_action` 拆成 `("click", "886076453056")`（参数转小写），
然后四路分发：

| 动作 | 路径 |
|---|---|
| `finish[...]` | `server.finish_without_purchase()` → 结算放弃 |
| `search[...]` | `server.search_results()` → BM25 |
| `click[X]` **且 X 在当前页可点击列表里** | 按 X 分流（见下） |
| 其他一切 | **静默失败**：`reward=0, done=False`，页面原样返回 |

第四路是"静默失败"，环境不报错。**所以采样侧必须有动作守卫**，否则模型以为操作成功了。

`click[X]` 的 8 种分流：

```
Buy Now         → done()：结算购买（终局）
Back to Search  → index()：回搜索首页，清空 asin/options/page
Next >          → 搜索结果页且非末页：翻页
< Prev          → 搜索结果页：上一页 / 详情页：回搜索结果 / 子页：回详情页
Description/Features/Reviews/Attributes → item_sub_page()
其他            → item_page()：打开商品详情
```

最后一路里，环境靠 **HTML 元素类型**区分"开商品"和"选规格"：

```python
if clickable.class == 'product-link':     session["asin"] = 名字.upper()
elif clickable.name is not None:          session["options"][轴] = 名字
# 然后无论哪种，都重算价格：
price_resolution = resolve_variant_price(product, session["options"])
session["selected_price"] = price_resolution["price"] if pass else None
```

### 每步之后的记账

```python
ob = self.observation
if not status["done"]:
    progress = self.server.record_progress(session, 动作名, 参数, 当前页ASIN, ...)
    if progress["termination_reason"]:
        status = self.server.terminate_session(session, progress["termination_reason"])
```

`EvidenceProgressTracker`（`termination.py`）数"这一步有没有带来新东西"
（新结果集 / 新商品 / 新子页 / 新规格 / 新约束校验）。**连续 4 步没新东西，或同一动作连续 2 次，
环境自己把 `done=True`、`reward_type=repeat_loop` 塞回来** ——模型没机会决定。

### 两套证据账本

tracker 维护两份集合，这是最容易看错的地方：

| 集合 | 作用 |
|---|---|
| `runtime_*` | 决定**是否还在前进**（liveness） |
| `*_evidence`（有 budget 上限） | 决定**放弃资格**（abstention） |

```python
product_open_progress_budget = 5    # 最多 5 次商品打开算"有效证据"
subpage_progress_budget = 6
result_set_progress_budget = 3
min_new_asins_per_result_set = 3    # 一次搜索要有 ≥3 个新 ASIN 才算新结果集
```

有 budget 的那个集合才是 `effective_result_sets`。**模型不能靠反复打开同一批商品
刷"有效探索"来换取 graceful_stop 资格。**

### 三个独立的终止上限

| 上限 | 位置 | 值 |
|---|---|---|
| 步数 | tracker `max_steps` | 35 |
| 连续重复 / 无进展 | tracker | 2 / 4 |
| **对话轮数** | `shop_agent.py:161 "over"` | `len(env.history) > 42` |

**第三个独立于前两个**——`over` 是每轮 assistant 消息 +1，和工具步数不是一回事。

---

## 7. 完整例子：task 65 的 11 步

需求：24 色三角粗杆可水洗水彩笔，送 4 件绘画工具，20 元以内。

| # | 动作 | 环境做了什么 | session 变化 | 页面 |
|---|---|---|---|---|
| 0 | `reset` | 建会话、清零、渲染首页 | — | `search_home` |
| 1 | `search_products{"水彩笔 24色 三角杆 儿童"}` | BM25 top-150 存缓存，切前 20 | `search_result_asins`(150), `current_page_asins`(20), `page=1` | `search_results` |
| 2 | `open_product{"899532603012"}` | 元素是 product-link | `asin=899…`, `asins.add()`, `selected_price=None` | `product_detail` |
| 3 | `select_option{"24色【加量粗杆可水洗】送绘画4件礼"}` | 元素是 radio | `options={颜色分类:…}`, **`selected_price=21.0`** | `product_detail` |
| 4 | `prev_page{}` | 详情页 → 回搜索结果（**复用缓存，不重新检索**） | `page` 不变 | `search_results` |
| 5 | `open_product{"744447021982"}` | 第二个候选（赠品不对） | `asin=744…` | `product_detail` |
| 6 | `prev_page{}` | 回搜索结果 | — | `search_results` |
| 7 | `open_product{"886076453056"}` | 第三个候选 | `asin=886…` | `product_detail` |
| 8 | `select_option{"24色三角粗杆【加量-可水洗】送绘画4件"}` | 选规格 | `options={…}`, **`selected_price=17.0`** | `product_detail` |
| 9 | `view_features{}` | 进子页，正文取 `BulletPoints`（空） | `subpage="Features"` | `information_subpage` |
| 10 | `prev_page{}` | 子页 → 回详情页 | `subpage=None` | `product_detail` |
| 11 | `buy_now{}` | **结算购买** | `done=True`, `reward=1.0` | `terminal` |

### 每一步的页面内容（模型实际看到的）

**① 首页**

```
[SHOPPING_OBSERVATION_V2]
page_type: search_home
使用 search_products 提交简短、具有区分度的商品查询。

搜索功能是否可用: True
可点击的按钮: []
```

⚠️ `搜索功能是否可用: True` **只在首页出现**。搜索后变 False，想换关键词必须先 `back_to_search`。

**② 搜索结果**

```
page_type: search_results
query: 水彩笔 24色 三角杆 儿童
Page 1 of 8 (Total results: 150; ranks 1-20)
格式: rank|asin|price|brand|category|key_attributes|title
1|796075776215|16.0 to 29.0|官方国货甄选|文具电教/...|幼儿园宝宝,...|晨光水彩笔...
12|899532603012|14.0 to 37.0|得力创优闪购专卖店|文具电教/...|24色,水彩...|得力24色水...
15|886076453056|16.0 to 37.0|得力文具专卖店|文具电教/...|幼儿园...|...
products_shown: 20

搜索功能是否可用: False
可点击的按钮: ["back to search", "next >", "796075776215", ..., "886076453056"]
```

**③ 商品详情（未选规格）**

```
page_type: product_detail
asin: 899532603012
title: 得力24色水彩笔可水洗幼儿园小学生专用无毒涂色填色绘画笔...
brand: 得力创优闪购专卖店          ← 实际是 shop_name
category: 文具电教/文化用品/商务用品›画具/画材用品›水彩笔
price: 14.0 to 37.0                ← ⚠️ 区间：规格没选，价格不确定
key_attributes: 24色, 水彩笔, 不透, 可洗, 粗杆, 加量, 可水洗
selected_options: {}
available_options: {"颜色分类": ["18色【加量粗杆可水洗】", "24色...", ...]}

搜索功能是否可用: False
可点击的按钮: ["back to search", "< prev", "description", "features", "reviews", "attributes", "buy now", "18色【加量粗杆可水洗】", ...]
```

**④ 选规格后（同一个 asin）**

```
price: 21.0                        ← 变单值了
selected_options: {"颜色分类": "24色【加量粗杆可水洗】送绘画4件礼"}
```

**这是全流程里价格唯一变单值的时机。**

**⑤ 信息子页**

```
page_type: information_subpage
asin: 886076453056
...
subpage: Features
content:                            ← BulletPoints 为空

搜索功能是否可用: False
可点击的按钮: ["back to search", "< prev"]     ← 只剩返回按钮
```

**子页是死胡同**：规格、Buy Now、其他子页全都不在列表里。

### 每一步的 progress 记账

| # | 动作 | tracker 效果 |
|---|---|---|
| 1 | search | `result_set_evidence` +1（20 个全新 ASIN） |
| 2 | open 899… | `product_evidence` +1 |
| 3 | select | `option_evidence` +1 |
| 4 | prev_page | 结果集指纹已见 → **`no_progress_steps += 1`** |
| 5 | open 744… | `product_evidence` +1 → 计数器归零 |
| 6 | prev_page | **`no_progress_steps += 1`** |
| 7 | open 886… | `product_evidence` +1 → 归零 |
| 8 | select | `option_evidence` +1 |
| 9 | view_features | `subpage_evidence` +1 |
| 10 | prev_page | **`no_progress_steps += 1`** |
| 11 | buy_now | — |

第 4/6/10 步是 `repeat_loop` 的主要制造机：`prev_page` 从详情页回搜索结果**复用缓存**，
页面与第 1 步完全相同，因此算无进展。本例每次都有别的进展把计数器清零，所以没触发。

---

## 8. 终局：一个 reward 怎么算出来

### 触发

只有两个动作能产生 `done=True`（外加环境自己判死）：

```python
click[Buy Now]                → SimServer.done()      → evaluate_purchase()
finish[no_suitable_product]   → finish_without_purchase() → evaluate_abstain()
（tracker 判定 repeat_loop / max_steps → terminate_session()）
```

### `buy_now` 内部

```python
session["actions"]["purchase"] += 1
price = session["selected_price"]                 # 17.0
result = evaluate_purchase(
    purchased_product = product_item_dict["886076453056"],
    goal              = session["goal"],
    selected_options  = session["options"],
    price_resolution  = session["price_resolution"],
)
session["done"] = True;  session["reward"] = result.reward
```

### `evaluate_purchase` 三段

```python
# ── 第一段：两个 hard gate ──
hard_gates = {
  "category": compare_category(goal["category"], product),
  "budget":   _price_gate(price_resolution, goal),
}
```

**category**（`comparators.py:162`）：两边按 `›` 切链，要求**叶子相等**且**祖先有交集**。

```
goal : [..., '水彩笔']      买到: [..., '水彩笔']    → 叶子相等 ✓ 祖先有交集 ✓ → PASS
```

**budget**：`resolve_variant_price` 算出的实际变体价 ≤ `price_upper`。

```
17.0 ≤ 20.0 → PASS
```

任一 `fail` → `wrong_purchase` **-0.85**；任一 `unverifiable` → `reward_unverifiable`
**0.0** 且 `reward_valid=false`。

```python
# ── 第二段：四个 soft 维度，只算"激活"的 ──
```

| 维度 | 权重 | 期望（来自 goal） | 被检 | task 65 结果 |
|---|---:|---|---|---|
| brand | 0.35 | `[]` | 商品的 `brand`→回退 `shop_name` | **不激活** |
| model | 0.25 | `["24"]` | `model`→回退 `title` | `title` 含 `24色`，边界匹配 → 1.0 |
| core_functions | 0.25 | `["幼儿园专用","不脏手","无毒","水溶性","绘画"]` | 商品的 `attribute` | 5 个全中 → 1.0 |
| key_options | 0.15 | `{"color":"24色三角粗杆【加量-可水洗】送绘画4件"}` | `session["options"]` | 完全相同 → 1.0 |

```python
active_weight = 0.25 + 0.25 + 0.15 = 0.65        # brand 不激活，不进分母
match_score      = (0.25·1 + 0.25·1 + 0.15·1) / 0.65 = 1.0
evidence_coverage = 1.0
all_satisfied = (match_score == 1.0) and (coverage == 1.0) = True
```

```python
# ── 第三段：终局判定（顺序即短路优先级）──
if hard_unverifiable:            → reward_unverifiable        0.0   (invalid)
elif hard_failed:                → wrong_purchase            -0.85
elif all_satisfied:
    asin_match = (product["asin"] == goal["asin"])
    → gold_purchase              1.00    if asin_match
    → valid_alternative_purchase 0.55    if not
else:                            → partial_alternative_purchase  min(0.25, -0.30+0.55·S)
```

task 65：两个 gate 都 PASS，`all_satisfied=True`，`asin_match=True` → **`gold_purchase`，1.0**。

### 终局响应

```jsonc
{
  "done": true, "reward": 1.0, "over": true,
  "reward_detail": {
    "reward_version": "shopsimulator-reward-v3",
    "reward_type": "gold_purchase", "reward_valid": true,
    "termination_reason": "gold_purchase",
    "target_asin_match": true,
    "hard_gates": {"category": {"status":"pass",...}, "budget": {"status":"pass",...}},
    "weighted_score": 1.0, "evidence_coverage": 1.0,
    "dimension_scores": {"brand":0.0, "model":1.0, "core_functions":1.0, "key_options":1.0},
    "terminal_utility": 1.0, "purchase_success": true, "sampling_invalid": false
  },
  "purchase": {"asin":"886076453056","price":17.0,"options":{...},"attributes":[...]},
  "goal": { ... 完整 goal，含 gold asin ... },        // ⚠️ gold
  "observation_state": {"page_type":"terminal","actions":[],"search_available":false}
}
```

### 三个必须知道的缺陷

**① `goal` 字段在终局响应里把答案吐出来了。** 项目侧在 `trajectory.py:90 _terminal_view`
只取 `done/over/reward/termination_reason/reward_detail/purchase/progress`，**不含 `goal`**；
SFT 那层再用消息白名单把终局 tool 内容整个换掉。两道闸都是为这个字段设的。

**② `dimension_scores` 里 brand 是 0.0，但它不参与评分。** 因为 `active=False`。
看 `dimension_scores` 不能直接判断"品牌不满足"。

**③ `weighted_score` 只反映激活维度。** 一个只有 `key_options` 激活的任务，
`weighted_score=1.0` 只说明"规格选对了"，不代表"需求全满足"。

---

## 9. reward 与 instruction 的两处缺口

这是整套设计里最重要的限制，也是为什么评估还要额外两层 LLM。

**缺口一：只算被标注的字段。**

`expected_core_functions` 来自 `instructions[0].attributes`（人工标注）。
task 65 的正文说了"附赠4件绘画工具"，但 `attributes` 里**没标**——于是这条要求
**完全不参与打分**。

`Final-200 Clean` 剔除 81 道题的理由之一就是这个：
"Query 与 gold 冲突或 gold 偷加约束"、"Reward v3 无法严格校验规格或价格"。

所以评估流水线额外加了 **V4 Flash Rubric 层**，从 Query 正文重新抽约束，**不信这套标注**。

**缺口二：`gold_purchase` 不衡量过程。**

`all_satisfied` 只看买了什么，不看怎么走到的。"第一个商品就瞎点买对了"和
"搜索比较十步后买对"都是 1.0。SFT 验收**两者都收**。

所以评估有 **V4 Pro 轨迹 Judge 的五维打分**。

**数字对照**：2,498 条原始轨迹里 **1,742 条是 `gold_purchase`**，但只接受了 **1,026**。
差额 716 条全部是 `guard_violation`——"买对 asin"其实很宽松（70%），
**真正卡人的是"过程合法"**。

---

## 10. 复现检查清单

```bash
# 启动（需要 Linux + 已建好的索引）
bash scripts/setup.sh                       # 解压商品数据 + 建 BM25 索引 + 打 veRL 补丁
bash scripts/start_environment.sh           # 监听 127.0.0.1:5700

# 手工跑一条
curl -s localhost:5700/api/shop_agent \
  -H 'Content-Type: application/json' \
  -d '{"action":"reset","idx":65}' | python -m json.tool
# → 记下 env_idx，后续 interact 都要带上

curl -s localhost:5700/api/shop_agent \
  -H 'Content-Type: application/json' \
  -d '{"action":"interact","env_idx":3,"response":"search[水彩笔 24色 三角杆 儿童]"}' \
  | python -m json.tool

# 用完必须显式释放
curl -s localhost:5700/api/shop_agent \
  -H 'Content-Type: application/json' \
  -d '{"action":"release_one","env_idx":3}'
```

**租约语义**：服务端**不会**在 `over` 时自动释放。`pack_api.py:111` 的注释解释了原因——
自动释放会和客户端的 `finally` release 竞争，另一个 worker 可能在两次 release 之间租到这个槽位。

拉不到槽位时（`acquire` 失败重试 5 次 × 5 秒）返回字面串：

```
Unable to get available environment resource, please try again later
```

`rollout.py:501` 就是匹配这个字符串来决定中断整批采集。

### Windows 工作区坑

- 仓库 `core.autocrlf=true`，JSONL 工作区是 CRLF，直接 `sha256sum` 与文档记录不符。
  用 `git show HEAD:<path>` 拿 LF 版本。
- 商品数据必须解压到 `shop_env/data/items_eval_train.json`（不是 `.gz` 的名字），
  哈希 `57b10950…559f`。`load_products` 里有个 `num_products is not None` 的守卫
  会拒绝非全量加载，但 `DEBUG_PROD_SIZE` 恒为 `None`，所以实际永不触发。

---

## 11. 一张图总结

```
                    items_eval_train.json[65]
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
         商品字典                           goal
   ───────────────────                ──────────────────
   title/price/options               category ← 商品
   attributes                        price_upper ← instruction 正则
   customization_options             expected_* ← instruction 标注
              │                               │
              │ 页面显示                       │ 打分期望
              ▼                               ▼
     模型看到 → 决定动作                和"实际买的"比
              │                               │
              └──────── session ──────────────┘
                 asin / options / selected_price
                 （唯一被写的状态）

   buy_now 时：商品字典 + session  →  对上 goal 的期望
              ├─ hard: category, budget
              └─ soft: brand, model, core_functions, key_options
                        ↓
                all_satisfied ?  ── 是 ──→ asin 相同? 1.0 : 0.55
                                 └ 否 ──→ partial ≤ 0.25
```

**一句话**：**sqlite 只负责搜出候选；商品字典决定页面显示；goal 决定打多少分；
session 是唯一被写的账本。购买身份 = asin + options。**

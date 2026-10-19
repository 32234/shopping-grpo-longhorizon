# 03 · SFT 训练：数据怎么变成梯度

这篇讲 `data/sft_pure_v4/all.jsonl` → LoRA adapter → merged checkpoint 这一段。
数据怎么来的见 01 篇；训练完交给 GRPO 见 04 篇（待写）。

训练框架本身很朴素：**HF `Trainer` + `peft` + 单卡**，没有 packing、没有 FlashAttention、
没有 FSDP/DeepSpeed。复杂度都在**四个"不处理就静默出错"的点**上。

---

## 1. 入口与三层结构

```
bash scripts/sft_curriculum.sh
  └── scripts/run_sft_curriculum.py      # 编排 A→B→C
        ├── scripts/train_lora_sft.py    # 训练单个阶段（×3）
        └── scripts/merge_lora_adapter.py# 合并 adapter（×3）
```

`scripts/sft.sh` 是**旧入口**（直接读 `data/sft/{train,validation}.jsonl`、3 epoch），
当前配方不走它。

单阶段产物：

```
outputs/models/sft-curriculum/stage-{a,b,c}/
  adapter/          # LoRA 权重 + tokenizer/processor + train_summary.json
  merged/           # 合并后的 BF16 全量权重 + merge_manifest.json
```

**GRPO 的起点是 `stage-c/merged`，不是 adapter。**

---

## 2. 数据 → token → label

核心是 `training/sft/dataset.py:56 build_supervised_example`。

### 2.1 全序列先 mask，再逐个 assistant 回合打开

```python
full_text = template.apply_chat_template(rendered_messages, tools=tools,
                                         tokenize=False, add_generation_prompt=False)
input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
labels = [IGNORE_INDEX] * len(input_ids)      # IGNORE_INDEX = -100

for index in assistant_indices:
    prefix_text  = apply_chat_template(msgs[:index],   add_generation_prompt=True)
    through_text = apply_chat_template(msgs[:index+1], add_generation_prompt=False)
    start = common_prefix_length(prefix_ids, through_ids)
    end   = len(through_assistant_ids)
    labels[start:end] = input_ids[start:end]
```

**为什么用"公共前缀"而不是硬编码长度**：注释里写了一个真实坑——

> 部分 chat template 的 generation prompt 与实际 assistant 起始 token 会有极小差异
> （例如额外换行）。以公共前缀定位，避免把可用样本误判为损坏。

这样**完全不需要知道具体模板的 token 格式**，换模型不用改代码。

同时还有两道一致性校验：

```python
if start >= end or end > len(input_ids):       return None
if input_ids[:end] != through_assistant_ids:   return None
```

### 2.2 静默失败点 ①：超长**丢弃**，不截断

```python
if len(input_ids) > int(max_length):
    return None
```

`max_length = 24576`。理由：截断可能砍掉终局决策或支撑它的证据。
丢了多少条会记在 `stats["dropped"]`，训练日志里能看到。

### 2.3 静默失败点 ②：`arguments` 必须从 JSON 字符串转成 dict

`dataset.py:32 normalize_messages_for_chat_template`：

```python
raw = function["arguments"]          # OpenAI 标准：JSON 字符串
parsed = json.loads(raw)
function["arguments"] = parsed       # Qwen3.5 模板会遍历 arguments 的键值对
```

**不转的话模板渲染会错，但不会报错。** 解析失败（非 JSON / 非 object）→ 整行丢弃。

原始 JSONL 保持 OpenAI 格式，转换只发生在训练期（`deepcopy` 后改）。

### 2.4 静默失败点 ③：Qwen3.5 的模板在 **processor** 上

`train_lora_sft.py:283`：

```python
is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
if is_multimodal:
    processor = AutoProcessor.from_pretrained(...)
    return processor.tokenizer, processor, True   # 模板用 processor，token id 用 tokenizer
```

模型类也跟着切：

```python
model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM
```

`merge_lora_adapter.py:11 choose_model_class` 有同样的分叉。**漏了会加载失败或模板取错。**

### 2.5 loader 的行为

`load_supervised_examples`（:129）：

| 行为 | 说明 |
|---|---|
| **先按 task_id 过滤，再 tokenize** | 课程阶段只处理本阶段的 task，省大量无用 tokenize |
| 单行异常不中断 | `KeyError/TypeError/JSONDecodeError` → 该行记为 dropped |
| 返回统计 | `{total, kept, dropped}`，课程模式额外 `{matched, filtered_out}` |
| 进度条 | 有 `tqdm` 就用，没有就退回原迭代器 |

`select_training_examples`（:178）是**确定性抽样**，按
`sha256(f"{seed}:{task_id}:{trajectory_id}")` 排序取前 N——只给消融实验用
（`--train-count` / `--train-ratio`），主配方不传。

---

## 3. 模型与 LoRA

### 3.1 target modules 必须覆盖 Gated DeltaNet

`train_lora_sft.py:17`：

```python
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # Qwen3.5 的大多数文本层是 Gated DeltaNet，不能遗漏其线性注意力投影。
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)
```

普通 Qwen 只需要前 7 个。后 5 个是线性注意力层的投影——**不加的话大部分文本层根本没在训练，
但不会报任何错**。这是静默失败点 ④。

### 3.2 LoRA 与量化

```python
LoraConfig(task_type=CAUSAL_LM, r=16, lora_alpha=32,
           lora_dropout=0.05, bias="none", target_modules=[...])
```

QLoRA 路径的顺序有讲究（PEFT 要求）：

```python
if qlora: model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=...)
if gradient_checkpointing:
    model.config.use_cache = False
    if not qlora and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(...))
```

### 3.3 精度解析

`--dtype auto` 在 CUDA 上优先 bf16，其次 fp16，CPU 用 fp32。`--bf16` 是兼容旧命令的别名，
不能和别的 `--dtype` 同时用。

训练里调用的是 `--dtype bf16` + `--attention-implementation sdpa`。

---

## 4. 训练参数

`TrainingArguments` 的实际取值（课程脚本传的）：

| 参数 | 值 | 来源 |
|---|---|---|
| `num_train_epochs` | **1** | manifest 每阶段的 `epochs` |
| `per_device_train_batch_size` | 1 | 默认 |
| `per_device_eval_batch_size` | 1 | 默认 |
| `gradient_accumulation_steps` | 8 | 默认 → 有效 batch 8 |
| `learning_rate` | 1e-4 → 7e-5 → 5e-5 | manifest 每阶段 |
| `warmup_ratio` | 0.03 | 默认 |
| `bf16` | True | `--dtype bf16` |
| `gradient_checkpointing` | True | 显式开 |
| `save_strategy` / `save_total_limit` | `"epoch"` / 3 | |
| `eval_strategy` | `"epoch"` | 有 validation 时 |
| `logging_steps` | 5 | 默认 |
| `remove_unused_columns` | False | 必须 |
| `seed` | 42 | |
| `report_to` | `"none"`（默认）/ `"swanlab"` | 显式 `--swanlab` 才开 |

**`save_strategy="epoch"` + 每阶段 1 epoch ⇒ 每阶段只会留 1 个 checkpoint。**
所以"在阶段内挑最好的 checkpoint"这个选项不存在；`--resume-from-checkpoint` 只用于中断恢复。

### 4.1 一个显存技巧：loss-only eval 跳过 logits

`train_lora_sft.py:249`：

```python
if enable_skip_logits and prediction_loss_only and inputs.get("labels") is not None:
    inputs["skip_logits"] = True
```

只在 `liger_kernel and is_multimodal` 时启用。原因写在注释里：

> Qwen3.5 的 Liger forward 默认只在 `model.training` 时启用融合 LM-head + cross-entropy；
> Trainer 验证会先 `model.eval()`，即使最终只需要 `eval_loss`，也会物化
> `[batch, sequence, vocab]` logits。20K 上下文和 248K 词表会因此产生约 **20 GiB** 的瞬时 FP32 张量。

### 4.2 collate

右侧 padding，原生长度外全部 `-100`：

```python
input_ids    : pad_token_id
attention_mask: 0
labels       : -100
```

`pad_token_id` 缺失时退回 `eos_token`；`padding_side = "right"`。

---

## 5. 课程执行：三段串行

`scripts/run_sft_curriculum.py`。

```
Stage A: 基础 base        → LoRA(A) → merged(A)
Stage B: merged(A)        → LoRA(B) → merged(B)
Stage C: merged(B)        → LoRA(C) → merged(C)
```

关键点：

```python
model = base_model if stage == "a" else f"{output_root}/stage-{prev}/merged"
```

**每阶段从上一阶段的 merged 权重重新 `get_peft_model`**——不是让同一个 adapter 连续训练。
每段都是全新的 LoRA，只有基座权重继承。

| stage | buckets | train | dev | LR | epoch |
|---|---|---:|---:|---:|---:|
| A | foundation | 256 | 28 | 1e-4 | 1 |
| B | + constraints | 799 | 88 | 7e-5 | 1 |
| C | + strategy | 1,073 | 119 | 5e-5 | 1 |

### 5.1 两道硬闸

**① 数据哈希必须与 manifest 一致**

```python
source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
if source_sha != manifest.get("source", {}).get("sha256"):
    raise SystemExit("Pure V4 数据与课程清单 SHA256 不一致；请先重新生成并审查清单")
```

**② 拒绝覆盖已完成的阶段**

```python
merged = output_root / f"stage-{stage}" / "merged"
if merged.exists() and any(merged.iterdir()):
    raise SystemExit(f"拒绝覆盖已完成阶段：{merged}")
```

所以重跑必须先手工清理，不能静默覆盖。

### 5.2 train 和 validation 是同一个文件

```python
"--train", str(source), "--validation", str(source)     # 都是 all.jsonl
```

靠 `--curriculum-manifest` + `--curriculum-stage` 里的 task_id 列表过滤：

```python
train_task_ids      = _curriculum_task_ids(manifest, stage, "train")
validation_task_ids = _curriculum_task_ids(manifest, stage, "validation")
```

并且**硬校验**：

```python
if train_stats["matched"] != len(train_task_ids):
    raise SystemExit("课程清单中的训练 task_id 未全部出现在 --train 数据中")
```

清单和数据对不上直接退出，不是警告。

### 5.3 训练前的防泄漏闸

`train_lora_sft.py:356`：

```python
guard_blind_final([args.train, *([args.validation] if args.validation else [])], allowed=False)
```

两道检查同时做：

1. 文件 sha256 是否等于 Final-200 本体
2. 逐行 `task_id` 是否落在 blind 集合里

任一命中 → `SystemExit`。

---

## 6. 合并 adapter

`scripts/merge_lora_adapter.py`：

```python
base   = model_class.from_pretrained(base_model, torch_dtype=bf16)
merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
AutoProcessor.from_pretrained(base_model).save_pretrained(output)
```

- 输出目录**必须为空或不存在**，否则拒绝
- 写 `merge_manifest.json`：

```json
{
  "operation": "peft_merge_and_unload",
  "source": {"base_model": "...", "adapter": "...", "model_type": "qwen3_5"},
  "output": "...",
  "next_step": "load this standalone checkpoint as GRPO base and attach a new LoRA adapter"
}
```

**为什么要 merged 而不是直接用 adapter：** GRPO 的 veRL 启动器只接一个完整模型目录；
而且 GRPO 会**再挂一个新的 LoRA**（`lora_rank: 16`），不能和 SFT 的 adapter 叠加。
这条边界在 `docs/sft.md` 里叫 "output contract"。

---

## 7. 没有做的事

| 没做 | 说明 |
|---|---|
| 序列 packing | 一条轨迹一个样本，右侧 padding |
| FlashAttention | `attn_implementation="sdpa"`，不要求编译 FA2 |
| FSDP / DeepSpeed | 纯 HF Trainer，单卡 |
| 动态 batch | 固定 `per_device=1` + grad accum 8 |
| 截断 | 超长直接丢 |
| 训练期去重 | 只在数据准备阶段按 task_id 去过一次 |
| rationale trimming | `sft_data_mixing_diagnosis.md` 建议过，**没有实现** |

---

## 8. 复现清单

```bash
# 0. 依赖
bash scripts/setup.sh          # uv sync --extra sft --extra grpo

# 1. 先看命令，不加载模型
bash scripts/sft_curriculum.sh --dry-run

# 2. 正式训练（A→B→C，每段 train + merge）
bash scripts/sft_curriculum.sh --swanlab

# 3. 从中断处继续（跳过已完成的 A）
bash scripts/sft_curriculum.sh --start-stage b

# 4. 只跑到 B
bash scripts/sft_curriculum.sh --stop-after-stage b

# 5. 阶段内恢复
bash scripts/sft_curriculum.sh \
  --start-stage b \
  --resume-from-checkpoint outputs/models/sft-curriculum/stage-b/adapter/checkpoint-100
```

训练完校验：

```bash
python - <<'PY'
import json, pathlib
for s in "abc":
    d = pathlib.Path(f"outputs/models/sft-curriculum/stage-{s}")
    summary = json.loads((d/"adapter/train_summary.json").read_text())
    manifest = json.loads((d/"merged/merge_manifest.json").read_text())
    print(s,
          "train=", summary["train_examples"],
          "val=", summary["validation_examples"],
          "loss=", round(summary["train_loss"], 4),
          "peak_gpu=", summary["peak_gpu_memory_gib"],
          "merge=", manifest["operation"])
PY
# 期望：A 256/28，B 799/88，C 1073/119
```

序列长度分布（确认 `max_length=24576` 砍掉多少）：

```python
# train_summary.json 里有 log_history，但 dropped 数只在训练日志的
# "train_data= {...}" 行里，形如 {'total':1192,'kept':...,'dropped':...}
```

---

## 9. 产物与下游

```
outputs/models/sft-curriculum/
  stage-a/{adapter,merged}
  stage-b/{adapter,merged}
  stage-c/{adapter,merged}     ← GRPO 起点
```

`adapter/` 里除权重外还有：

- tokenizer / processor（`chat_template.save_pretrained`）
- `train_summary.json`：`train_examples`、`validation_examples`、`train_loss`、
  `metrics`、`log_history`、`peak_gpu_memory_gib`、`total_time_minutes`、
  `monitoring`、`acceleration`、`arguments`

`train_summary.json` 里的 `log_history` 是排障的主要依据——`ProgressCallback`
每个 logging step 会补充 `step_time_s` 和 `gpu_peak_memory_gib`。

---

## 10. 一句话

> **SFT 训练 = 一个 60 行的 label-mask 函数 + 三段串行的 LoRA 训练 + 一次 adapter 合并。**
> 真正的技术点只有四个：模板 diff 定位 label、超长丢弃、`arguments` 字符串转 dict、
> LoRA 覆盖 Gated DeltaNet；加上两道硬闸（blind guard、manifest 哈希）和一处显存技巧
> （loss-only eval 跳过 logits）。

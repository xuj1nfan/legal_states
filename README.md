# Legal State Pilot

比较逐步 CoT、Generic State 与 Legal State 在中文长程法律推理中的表现。当前 benchmark 为
[MSLR-Bench](https://github.com/yuwenhan07/MSLR-Bench) Task 2：模型读取真实内幕交易案件材料，沿
“内幕信息形成 → 信息知悉 → 交易行为 → 违法所得 → 法律适用与判决类型 → 处罚结果”生成完整分析和裁判结论。

三种方法共享模型、原始案情、调用上限和累计输出预算，不提供额外法条、检索或训练。参考答案与结构化字段仅在评分时读取，不进入模型提示词。

## 安装

项目要求 Python 3.12+，所有命令从仓库根目录执行：

```bash
uv sync --locked
PYTHONPATH=src uv run python -m legal_state.experiment --help
```

## 1. 获取并准备 MSLR

```bash
git clone --depth 1 \
  https://github.com/yuwenhan07/MSLR-Bench.git third_party/MSLR-Bench

PYTHONPATH=src uv run python -m legal_state.experiment prepare \
  --source third_party/MSLR-Bench/data/input_data_anonymized.json \
  --references third_party/MSLR-Bench/data/processed_anonymized \
  --out data/processed/mslr
```

`prepare` 校验输入与逐案结构化参考是否一一对应，拒绝覆盖非空目录，并记录上游 commit、文件哈希和官方 IRAC 评分脚本哈希。输出为：

上游输入数组按 `entry_*.json` 文件名的字典序生成，而“序号”只是数组内的连续编号。适配器按同一字典序逐条核对案情，并以结构化文件号作为稳定 ID；这样也修正了上游预测脚本和评分脚本按不同编号配对时可能出现的参考错位。

| 文件 | 内容 |
|---|---|
| `cases.jsonl` | 仅含模型可见的匿名化案情、固定任务说明和稳定 ID |
| `targets.jsonl` | 严格字段、语义字段与参考裁判；不会由运行器读取 |
| `review.csv` | MSLR 已审核样本清单，默认全部合格 |
| `validation.json` | 数据版本、有效数量、错误记录、长度范围和上游哈希 |

上游目前包含 1,389 个 2005–2024 年内幕交易样本。数据、上游代码与运行结果由 `.gitignore` 排除，不随本项目提交。使用数据时遵守
[MSLR-Bench Apache-2.0 许可及免责声明](https://github.com/yuwenhan07/MSLR-Bench/blob/master/LICENSE)。

## 2. 固定开发集和正式集

MSLR 未提供独立测试划分。下面固定 139 个开发样本和 1,250 个正式样本，并从正式集抽取 100 个过程复核样本：

```bash
PYTHONPATH=src uv run python -m legal_state.experiment split \
  --data data/processed/mslr \
  --review data/processed/mslr/review.csv \
  --count 1389 --dev-count 139 --process-count 100 --seed 20260930
```

划分写入 `splits.json`，且不能覆盖。MSLR 的六阶段过程复核项会自动写入 `annotations/issues.jsonl`；它们只用于输出侧人工评估。

若完整实验成本过高，可在新的数据输出目录使用较小 `--count`。不同样本规模的结果不能直接混为同一实验。

## 3. 配置模型与预算

通用配置是 [configs/mslr.yaml](configs/mslr.yaml)。本地 Qwen3.5-9B 可使用
[configs/mslr_qwen35_9b.yaml](configs/mslr_qwen35_9b.yaml)，其中通过 `chat_template_kwargs.enable_thinking=false` 关闭隐藏推理。

```bash
export LSP_MODEL='实际模型名称'
export LSP_ENDPOINT='完整的 HTTP(S) Chat Completions 地址'
read -rsp 'API key: ' LSP_API_KEY
export LSP_API_KEY
```

通用配置中的 `hidden_reasoning_disabled` 默认为 `false`。按服务商实际参数关闭隐藏推理后，将它改为 `true`。密钥不得写入 YAML。

MSLR 的默认预算比原选择题实验更高：32 次推理调用、单次最多 1,024 输出 token、累计最多 24,576 输出 token，最终答案转写最多 2,048 token。三组使用同一上限，实际消耗仍可能不同；32 步也为 Legal State 的六阶段“建争点、绑定材料、提交结论、解决争点”留出正常停止空间。

案情长度中位数约 1,400 字符，最长超过 11,000 字符，且状态轨迹会继续增长；按默认累计输出预算运行时，模型服务的上下文窗口应至少配置为 64K token。

```bash
PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/mslr.yaml
```

## 4. 开发、冻结和运行

```bash
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/mslr.yaml --split dev --run-id mslr_dev_01

PYTHONPATH=src uv run python -m legal_state.experiment score \
  --config configs/mslr.yaml --run-id mslr_dev_01

PYTHONPATH=src uv run python -m legal_state.experiment freeze \
  --config configs/mslr.yaml \
  --dev-run-id mslr_dev_01 --run-id mslr_test_01

PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/mslr.yaml --split test --run-id mslr_test_01
```

开发集三种方法都必须至少有一条正常 `STOP` 且生成非空开放式结论，才能冻结正式运行。中断后可加 `--resume`；已完成或已失败的组合不会被选择性重跑，状态和响应日志均有哈希保护。

无需凭据即可查看调用与输出预算上界：

```bash
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/mslr.yaml --split test --run-id preview --dry-run
```

## 5. MSLR 评分

普通 `score` 不加载大型嵌入模型，报告可复现的字面字段覆盖率，适合开发期快速诊断：

```bash
PYTHONPATH=src uv run python -m legal_state.experiment score \
  --config configs/mslr.yaml --run-id mslr_test_01
```

正式结果使用上游 IRAC Recall：严格字段按原文包含匹配，语义字段由 ChatLaw-Text2Vec 以 0.6 阈值匹配。先在当前 Python 环境安装上游 `sentence-transformers` 依赖，并按
[MSLR 官方说明](https://github.com/yuwenhan07/MSLR-Bench#-how-to-evaluate-models)下载 ChatLaw-Text2Vec，然后运行：

```bash
PYTHONPATH=src uv run python -m legal_state.experiment score \
  --config configs/mslr.yaml --run-id mslr_test_01 \
  --official-check --embedding-model /path/to/ChatLaw-Text2Vec
```

`--official-check` 会先核对上游 commit 与评分脚本哈希，再直接调用上游 `evaluate_prediction`。失败、空输出和格式错误均以零分进入分母。结果包括：

- `summary.csv`：IRAC Recall、严格字段召回、失败率、token、调用数和耗时。
- `per_case_scores.csv`：逐案例字段匹配与分数。
- `scores.json`：评分模式、官方明细和 10,000 次案例级配对 bootstrap 区间。
- `mslr/<method>/output_<序号>.json`：上游兼容预测文件。
- `mslr/sample_mapping.json`：原始序号与稳定案例 ID 的映射。

官方还提供基于裁判模型的 A/B/C LLM Score。它依赖独立裁判模型及凭据，当前受控实验不自动调用，以免把待测模型差异与裁判模型变动混在一起。

## 6. 人工过程评估与报告

```bash
PYTHONPATH=src uv run python -m legal_state.experiment human-review-export \
  --config configs/mslr.yaml --run-id mslr_test_01

PYTHONPATH=src uv run python -m legal_state.experiment human-review-import \
  --config configs/mslr.yaml --run-id mslr_test_01 \
  --ratings runs/mslr_test_01/human_review/materials/ratings.csv

PYTHONPATH=src uv run python -m legal_state.experiment report \
  --config configs/mslr.yaml --run-id mslr_test_01
```

人工材料隐藏方法标签和自动分数，围绕六阶段推理链记录识别、分析、矛盾与错误类型。`report.md` 汇总最终答案、过程质量、成本和配对区间；区间跨零时不把方向性差异写成确定收益。

## 兼容性

原 LawBench 3-6 配置与历史文档仍保留，便于复现已有运行；实验 CLI 的默认配置和主文档已切换为 MSLR。MSLR 是开放式长程推理任务，因此最终答案不再限制为 A/B/C/D，基于选项的 `verified`、`scoped_options`、question frame 与 option comparison 工作流不适用于 MSLR。

# Legal State Pilot

比较逐步 CoT、Generic State 与 Legal State 在中文法律案例分析中的表现。
首轮使用 LawBench 3-6，三组共享模型、原题、调用上限和输出预算，暂不提供额外法条、检索或训练。

实验代码已接通；真实模型调用仍需配置模型与密钥。所有命令从仓库根目录运行，使用 Python 3.12+ 和 uv。

## 安装与验证

```bash
uv sync --locked
PYTHONPATH=src uv run pytest
PYTHONPATH=src uv run python -m legal_state.experiment --help
PYTHONPATH=src uv run python examples/demo.py
```

核心代码位于 `src/legal_state/`，实验入口位于 `src/legal_state/experiment/`，测试位于 `tests/`。
`legal-state-pilot` 控制台入口仍是项目初始化的问候示例；实验使用上面的模块命令。

## 1. 获取并检查 LawBench 数据

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/open-compass/LawBench.git third_party/LawBench
git -C third_party/LawBench sparse-checkout set data/zero_shot evaluation

PYTHONPATH=src uv run python -m legal_state.experiment prepare \
  --source third_party/LawBench/data/zero_shot/3-6.json \
  --out data/processed/lawbench_3_6
```

如果 `third_party/LawBench` 已存在，跳过 clone；如果输出目录已有 `validation.json`，跳过 prepare。
prepare 拒绝覆盖已有数据。更换数据版本时使用新输出目录，并修改配置中的 `data_dir`。

输出包括：

| 文件 | 内容 |
|---|---|
| `cases.jsonl` | 原题、题干、选项和稳定 ID；不包含答案 |
| `targets.jsonl` | 评分用参考答案；不进入模型提示词 |
| `review.csv` | 人工资格筛选表，包含题目和待填字段 |
| `validation.json` | 来源 SHA、上游 commit、重复/错误记录和评分源码哈希 |

本次接入检查到 500 道题，全部合法，没有重复题；题目长度为 54–858 字符。
初次获取记录版本；复现时应使用 `validation.json` 中的上游 commit。

官方来源：
[LawBench](https://github.com/open-compass/LawBench)、
[3-6 数据](https://github.com/open-compass/LawBench/blob/main/data/zero_shot/3-6.json)、
[JEC-QA](https://jecqa.thunlp.org/)。
LawBench 要求遵循原始数据来源的使用许可；上游仓库、数据和运行结果默认忽略，不随代码提交。

## 2. 人工筛选与固定样本

编辑 `data/processed/lawbench_3_6/review.csv`，逐题填写：

- `eligible=yes`：有具体场景，需要规则应用或案件判断。
- `eligible=no`：没有具体场景、主要考定义/背诵、题目损坏或参考答案有明显歧义；填写 `reason`。
- 不修改 `case_id`。所有题目必须审核完成，不能根据模型成绩决定保留哪些题目。

历史题按数据集原答案及对应规则背景处理；不要混用当前规则后再根据结果调整参考答案。
本轮涵盖多个法律领域，题干中的主张和候选选项不会自动视为已证实事实。

```bash
PYTHONPATH=src uv run python -m legal_state.experiment split \
  --data data/processed/lawbench_3_6 \
  --review data/processed/lawbench_3_6/review.csv \
  --count 100 --dev-count 15 --process-count 30 --seed 20260930
```

结果写入 `splits.json`：15 道开发题、85 道正式题、从正式题提前抽取的 30 道人工分析题。
不足 100 道合格题时停止，不自动混入知识题。划分不允许覆盖。

填写 `annotations/issues.jsonl` 中每一道题的必要争点，并复核：

```json
{
  "case_id": "lawbench-3-6-000127",
  "required_issues": [
    {
      "issue_id": "G1",
      "description": "解答本题必须判断的法律问题",
      "coverage_criterion": "哪些具体推理内容可以算作实际分析了这个问题"
    }
  ],
  "review_notes": "标注依据、规则背景和复核说明",
  "reviewed": true
}
```

每题至少一个必要判断点，不能机械地把四个选项标为四个争点。标注不进入模型输入。
开发运行不要求完成这些标注；正式冻结要求全部完成并复核。

## 3. 配置模型与预算

编辑 `configs/lawbench_3_6.yaml`。模型名称、完整 Chat Completions 地址和密钥从环境读取：

```bash
export LSP_MODEL='实际模型名称'
export LSP_ENDPOINT='完整的 HTTP(S) 模型调用地址'
read -rsp 'API key: ' LSP_API_KEY
export LSP_API_KEY
```

不要将密钥写入 YAML。`.env` 被忽略，但程序不会自动加载它。

选择普通非推理模型，或按服务商的参数关闭隐藏推理。将参数写入 `model.provider_options`，确认后设置
`model.hidden_reasoning_disabled: true`。默认值为 false，因此未确认时不会调用模型。
例如支持该字段的服务可设置 `provider_options: {thinking: {type: disabled}}`；其他服务请使用其实际参数。
不发送字段的默认 `{}` 适用于本身没有隐藏推理的服务。

该确认是人工声明。服务若报告非零 `reasoning_tokens`，程序拒绝该响应；若不报告，则在预检查记录中注明无法通过 usage 独立核验。

默认控制：

| 项目 | 三组一致的设置 |
|---|---:|
| temperature | 0 |
| 推理调用上限 | 24 |
| 单次推理输出上限 | 512 token |
| 累计推理输出上限 | 8192 token |
| 最终答案调用 | 1 次，最多 128 token |
| 并发 / 自动重试 / 重复次数 | 1 / 0 / 1 |

剩余额度不足 512 时使用剩余额度作为该次上限。预算或步数耗尽后，仍对已有材料生成一次最终答案。
正常停止、受限终止、截断、模型错误和协议错误分别记录。不得将累计上限误认为实际消耗相同。

```bash
PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/lawbench_3_6.yaml
```

preflight 使用独立检查题，验证接口、usage、结束原因和输出契约，不使用正式题。
实际模型名称必须在后续运行中保持一致。代码或配置改变后需要重新 preflight。

## 4. 开发、冻结、正式运行

```bash
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6.yaml --split dev \
  --run-id lawbench_3_6_dev_01

PYTHONPATH=src uv run python -m legal_state.experiment score \
  --run-id lawbench_3_6_dev_01 --official-check
```

开发集有 45 个“题目 × 方法”组合。检查轨迹、预算、结束标记、JSON 错误、截断和最终转写是否合理。
仅在开发集修改代码和提示词；修改后采用新运行 ID 并重新 preflight。

三组都应至少完成一次正常 STOP 推理并给出有效选项；Legal State 还应形成结论并解决争点。
所有开发组合必须有终止记录。完成 30 题标注及复核后冻结：

```bash
PYTHONPATH=src uv run python -m legal_state.experiment freeze \
  --config configs/lawbench_3_6.yaml \
  --dev-run-id lawbench_3_6_dev_01 --run-id lawbench_3_6_test_01

PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6.yaml --split test \
  --run-id lawbench_3_6_test_01
```

冻结保存代码与数据哈希、配置、模型身份、Python 与依赖版本及开发记录指纹。
正式运行有 255 个组合，程序轮换各方法的执行顺序。

无需模型配置即可预览输出预算上界：

```bash
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --split test --run-id preview --dry-run
```

85 题最多 6375 次调用，输出 token 上界为 2,121,600；输入成本与金额需依据开发结果和实际服务价格估计。

### 中断后继续

```bash
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --split test --run-id lawbench_3_6_test_01 --resume
```

- 已有成功和失败记录都跳过，不选择性重跑失败案例。
- 未完成组合复用日志中已有响应，继续缺失步骤。
- 已发请求但没有响应记录时，结果未知，记为失败，不自动再次收费调用。
- 损坏日志保留并记录失败，不静默丢弃。已完成批次的记录有文件哈希保护。
- 配置、代码、模型、筛选表、划分或正式参考标注改变时，旧运行不能续跑。

## 5. 自动评分与官方复核

```bash
PYTHONPATH=src uv run python -m legal_state.experiment score \
  --run-id lawbench_3_6_test_01 --official-check
```

最终答案生成器只转写上游推理，输出 `{"answer":"A"}`，允许 A/B/C/D/UNKNOWN。
UNKNOWN、截断或执行失败在主准确率中按未答对处理，不从分母剔除。
推理全文保存但不送入答案评分器，避免其中出现多个选项字母导致误判。

`--official-check` 校验上游版本和评分源码哈希，直接执行上游 `multi_choice_judge` 和 `compute_jec_ac`
的原始函数定义。只提取两个纯函数，不加载其他任务的 ROUGE/NLTK 导入，无需额外评分依赖。
上游来源：[jec_ac.py](https://github.com/open-compass/LawBench/blob/main/evaluation/evaluation_functions/jec_ac.py)、
[function_utils.py](https://github.com/open-compass/LawBench/blob/main/evaluation/utils/function_utils.py)。

结果写入：

- `summary.csv`：准确率、失败率、受限终止率、token、调用数和耗时。
- `per_case_scores.csv`：逐案例评分。
- `scores.json`：官方复核及配对差异、10000 次案例级 bootstrap 的 95% 区间。
- `lawbench/<method>/3-6.json`：官方兼容格式，连续编号。
- `lawbench/sample_mapping.json`：导出编号到原始案例的映射。

主比较为 Legal State − Generic；辅助比较为 Legal State − CoT。失败调用中未取得的 usage 不估填为已知费用。

## 6. 人工过程评估

```bash
PYTHONPATH=src uv run python -m legal_state.experiment human-review-export \
  --run-id lawbench_3_6_test_01
```

生成 `human_review/materials/`：90 份随机排列的推理材料及 `ratings.csv`。
向评估者提供该目录；不要提供相邻的 `private_mapping.json`。材料隐藏方法标签与自动正确性，但格式仍可能暴露方法。

每个必要争点填写：

| 字段 | 填写方式 |
|---|---|
| identified | yes / no，是否识别必要问题 |
| analyzed | yes / no，是否实际分析；yes 时 identified 必须 yes |
| contradiction | yes / no，案例级语义矛盾；同一材料各行一致 |
| error_categories | 可为空，多个标签用逗号分隔 |
| notes | 推理证据、判定理由 |

失败组合的主过程指标为零覆盖，模板预填 identified/analyzed=no。不要把结构拒绝自动算作法律语义矛盾。
错误标签：issue_omission、false_issue、fact_binding_error、knowledge_error、state_contradiction、
premature_resolution、unsupported_conclusion、reasoning_error、format_error。

```bash
PYTHONPATH=src uv run python -m legal_state.experiment human-review-import \
  --run-id lawbench_3_6_test_01 \
  --ratings runs/lawbench_3_6_test_01/human_review/materials/ratings.csv

PYTHONPATH=src uv run python -m legal_state.experiment report \
  --run-id lawbench_3_6_test_01
```

人工评分输出 `process_scores.json` 和 `process_summary.csv`：案例平均覆盖率、汇总覆盖率、遗漏率和语义矛盾率。
参考标注只有必要争点，因此不报告完整 Issue Precision/F1。

## 7. 分析与解释

`report.md` 汇总自动和人工结果。`case_analysis.json` 选取最多 20 题，覆盖 Legal 成功、Legal 失败、共同失败和高成本成功；填写原因与轨迹证据。
重复生成报告不会覆盖已经填写的案例分析模板。人工评分尚未导入时，报告明确标记过程结论待完成。

首轮是冻结的案例子集实验，不能直接与全 500 题的官方排行榜成绩比较。
知识不足与推理错误都可能影响闭卷结果；字段、状态操作和校验机制共同构成方法。
区间跨零时不能将小幅差异写成确定收益。扩大数据、加入规则包或做消融实验应建立新实验版本。

## 本地 Qwen3.5-9B 开发测试

本机服务 `http://127.0.0.1:8000/v1/models` 返回模型名 `Qwen3.5-9B`。
专用配置为 `configs/lawbench_3_6_qwen35_9b_local.yaml`，使用
`chat_template_kwargs: {enable_thinking: false}` 关闭隐藏思考；独立预检查已确认
`reasoning_tokens=0`。该参数见 [Qwen 官方说明](https://huggingface.co/Qwen/Qwen3.5-9B#instruct-or-non-thinking-mode)。

在本机无认证服务上，`EMPTY` 仅用于满足客户端非空 key 的要求；开启认证时请使用真实密钥。

```bash
export LSP_MODEL='Qwen3.5-9B'
export LSP_ENDPOINT='http://127.0.0.1:8000/v1/chat/completions'
export LSP_API_KEY='EMPTY'

PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/lawbench_3_6_qwen35_9b_local.yaml

PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6_qwen35_9b_local.yaml \
  --split dev --run-id lawbench_3_6_qwen35_9b_dev_01

PYTHONPATH=src uv run python -m legal_state.experiment score \
  --config configs/lawbench_3_6_qwen35_9b_local.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_01 --official-check

PYTHONPATH=src uv run python -m legal_state.experiment report \
  --config configs/lawbench_3_6_qwen35_9b_local.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_01
```

已有运行继续时，在 run 命令末尾加 `--resume`；不要同时启动同一运行 ID。
本次服务上下文上限为 4096 token，45 个开发组合的初始提示词均可容纳；后续历史和状态增长仍可能超限。
服务快照及初始长度检查保存在 `runs/setup_qwen35_9b_20261001/`。
若开发发现上下文错误，调整服务后使用新开发运行 ID，保留旧记录。
正式测试仍需完成 `data/processed/lawbench_3_6/annotations/issues.jsonl` 中的 30 题人工争点标注并冻结。

## 规则条件映射与选项复核

`configs/lawbench_3_6_qwen35_9b_verified.yaml` 启用四选项独立评估、规则条件与原文依据映射、
可修订的竞争选项复核及 JSON Schema 约束。正常 Legal State 路径为六次推理调用，
完整轨迹保留，最终答案只转写复核决策。旧流程默认设置继续可用。

失败诊断、运行命令和准确率验收见 [verified_workflow.md](docs/verified_workflow.md)。
`examples/check_accuracy.py` 将“主准确率严格高于 CoT”写成可执行检查，包含所有失败与 UNKNOWN。
开发集通过仍需独立的正式集冻结验证。

`configs/lawbench_3_6_qwen35_9b_hybrid.yaml` 在条件核查前增加一次自然 CoT 草稿，
草稿保存为待验证的模型分析，与事实和给定知识区分，并计入相同的推理配额。
正常路径为七次推理调用。实验属于 CoT 与 Legal State 的混合流程，
配置、来源处理与完整对照命令见 [hybrid_workflow.md](docs/hybrid_workflow.md)。

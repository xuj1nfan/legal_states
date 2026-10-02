# Legal State：规则条件映射与选项复核

## 已有失败证据

读取 dev_02、dev_03_scoped 和 dev_04_sequential 的报告、逐题评分及完整轨迹。
`case_analysis.json` 的人工原因字段仍为空，不能把模板视为完成的法律复核。
以下是开发轨迹诊断；不将模型输出当作核实后的法律依据。

逐争点流程虽然 15/15 正常闭合，准确率仍为 9/15，低于 CoT 的 12/15。
因此继续加强格式检查不足以解决主要差距。

| 开发题 ID 后缀 | 轨迹中的失分现象 | 对应改动 |
|---|---|---|
| 000348、000355 | `C1.content` 仅为 `D`，未记录适用规则或条件；CoT 有规则与选项比较 | 禁止只有字母的评估，要求规则、条件、原文引用及例外 |
| 000394 | 评估到 C 后停止，未分析 D；已注册争点闭合不代表选项覆盖 | A–D 覆盖由执行器控制，覆盖不足不能复核或 STOP |
| 000011 | 只创建登记经营者争点，将一个主体的判断直接扩展为其他主体无关 | 四项独立评估，综合复核复合命题及竞争选项 |
| 000353、000276 | 单个初判后停止；与 CoT 共同失分，存在规则记忆和适用条件风险 | 显式区分记忆规则与给定知识；复核时间、阶段和例外 |

作用域单独实验还出现重复动作、非法引用及上下文超限，Legal State 降至 5/15。
新流程保留原文作用域，同时只向模型提供当前阶段所需的材料与紧凑评估。

## 新流程

设置 `legal_state_workflow: verified` 和 `legal_state_materials: scoped_options`。
本机配置还启用 `legal_state_constrained_json: true`，要求服务支持 JSON Schema。
预检查会实际发送约束请求；不支持时明确失败，不静默降级。
旧 `free`、`sequential` 配置继续可用，新字段为空时不进入旧状态序列化。

1. A–D 各执行一次 `ASSESS_OPTION`，输出规则来源、最多三项条件、原文依据、条件对应、例外和初判。
   单个条件缺少题面依据时可以保留空引用并说明缺失；整项评估至少有一条有效材料引用，不虚构证据。
2. 初判提示词不提供此前选项的判断，减少锚定。引用必须属于题干或当前选项。
3. `AUDIT_OPTIONS` 重新检查四项并可更正初判；写明唯一选择及最强竞争选项的排除理由。
4. 完成复核后才允许 `STOP`。正常路径为六次推理调用；步数或 token 不足仍如实记为受限终止。
5. 每个动作返回重新校验的快照。原始初判、复核结果及全部状态转移保留；最终答案生成器只转写复核决策。

启用约束时，每次请求使用当前动作的 Pydantic JSON Schema，包括字段和枚举。
Schema 与请求一同保存并参与重放指纹。它约束生成格式，不能核实规则、引用或结论；
引用有效性、作用域和四项覆盖仍通过本地校验；判断的语义一致性需复核，截断和其他错误仍计为失败。
接口方式参见 [vLLM 结构化输出文档](https://docs.vllm.ai/en/stable/features/structured_outputs/)。

结构校验检查覆盖和引用，不验证法律语义。闭卷规则仍可能记忆错误。
`meets_question` 指符合原题问法，例如问“不违法”时不等于判断选项违法。

## 验证与验收

固定原有 15 道开发题，三组共享 Qwen3.5-9B、temperature=0、关闭隐藏思考、
24 步、512 token/次、8192 推理输出 token 上限，最终答案仍为一次 128 token 调用。
CoT / Generic 提示词和评分保持不变；不添加外部法条或读取参考答案作为模型输入。

以下示例 ID 应换成尚不存在的新 ID：

```bash
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/lawbench_3_6_qwen35_9b_verified.yaml
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6_qwen35_9b_verified.yaml \
  --split dev --run-id lawbench_3_6_qwen35_9b_dev_next_verified
PYTHONPATH=src uv run python -m legal_state.experiment score \
  --config configs/lawbench_3_6_qwen35_9b_verified.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_next_verified --official-check
PYTHONPATH=src uv run python examples/check_accuracy.py \
  --config configs/lawbench_3_6_qwen35_9b_verified.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_next_verified --official-check
```

验收比较相同样本的主准确率，所有失败和 UNKNOWN 均计入分母。
必须严格高于 CoT 才满足本轮开发目标，同时报告配对区间与实际 token 成本。
开发集已用于诊断和改进，即使达标，也不能作为未见题或显著泛化优势的证据。
正式集仍须按既有标注、冻结流程独立验证，禁止根据正式成绩回调提示词。
`check_accuracy.py` 输出 `accuracy_acceptance.json`，未严格超过 CoT 时退出码为 1，
并单独标注开发集通过与正式集配对区间大于零的区别。

## 开发迭代记录

初版 `dev_05_verified` 已完整运行并封存 45 项，官方评分一致。
Legal State 为 8/15，CoT 为 12/15，未满足目标。
Legal 的五项失败分别为两个条件空引用、一次缺少 JSON 结束括号、两次非法 verdict 枚举。
其余两个失分为有效但错误的判断，暴露出时间条件遗漏和程序混同。
没有修复旧响应或单独重跑被拒绝案例；原始结果保留。

第二版允许明确记录条件缺失依据，增加日期、期限、程序及原文限定词检查，
并启用服务端 Schema 约束。三组预算及 CoT / Generic 提示词继续不变，
以全新的 `dev_06_verified` 重跑全部开发组合。结果为 9/15，仍低于 CoT 的 12/15。
四次失败来自初判、复核标签或竞争选项字段之间的矛盾；另外两题为有效但错误的判断。
第三版简化复核字段，并在无草稿的初判阶段仅展示题干与当前选项，避免先看到其他选项判断。
`dev_07_verified` 复用了逐调用精确重放验证的 30 项开发基线，Legal State 全部重新调用；
模型服务在第五题中断，最终仅 2/15 正确、11 项执行失败。该轮保留封存，不能据此判断方法效果。

| 运行 | Legal State | CoT | Legal 执行失败 | 说明 |
|---|---:|---:|---:|---|
| dev_04_sequential | 9/15 | 12/15 | 0 | 状态闭合不保证法律正确 |
| dev_05_verified | 8/15 | 12/15 | 5 | 格式和引用拒绝 |
| dev_06_verified | 9/15 | 12/15 | 4 | 约束生成后仍存在语义字段矛盾 |
| dev_07_verified | 2/15 | 12/15 | 11 | 服务中断，基线复用有来源记录 |

服务恢复后另建 `dev_08_hybrid`，重新运行三组，不在旧记录中修补失败。
可选自然分析草稿及其预算、来源处理见 [hybrid_workflow.md](hybrid_workflow.md)。

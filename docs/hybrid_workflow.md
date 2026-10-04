# 自然分析草稿与 Legal State 复核

## 动机与方法边界

原流程在单个争点得出字母后可能提前停止；四选项状态流程可以保证覆盖，
但规则记忆、条件适用和结构字段之间的矛盾仍会失分。
开发轨迹还显示，强制结构化会改变自然分析的判断路径。
本实验在结构化前加入一次自然分析，检验保留推理草稿后进行条件核查是否有效。
这是 **CoT 草稿 + Legal State 复核的混合流程**，不能将其收益归因于纯状态表示。

## 状态与预算

配置 `configs/lawbench_3_6_qwen35_9b_hybrid.yaml` 启用
`legal_state_initial_analysis: cot`，仅适用于 `verified` 流程。

1. 使用与 CoT 基线首步完全相同的提示词，进行一次自由文本分析。
   这是初步草稿，不保证完成完整的 CoT 轨迹。
2. 将草稿保存为 `draft_reasoning`；它不是 Fact 或给定 Knowledge。
   原始调用、输出和来源记录在 `draft_step` 与调用日志中。
3. 依次核查 A–D 的规则、条件、原文依据和例外，再综合复核。
   当前版本初判可看到草稿，作为待验证假说；综合复核比较草稿与条件分析的实质理由。
   其他选项的假设不得作为当前选项的事实。
4. 完成复核后 STOP，最终答案只转写复核决策。初判和草稿保留以便追查。

草稿消耗共享的 24 步和 8192 推理输出 token 配额，每次最多 512 token。
正常路径共七次推理调用，加一次最多 128 token 的答案生成。
CoT、Generic 的提示词、预算和终止规则不变；实际消耗与延迟需另行比较。
中间动作使用服务端 JSON Schema；草稿及最终答案沿用原有输出协议。
JSON 约束只能保证格式，法律判断仍可能错误。
流程从当前可见材料提取日期并计算日历月份差，保留各端点的原文与材料 ID。
这是月份精度的算术辅助，不是精确经过天数、法条或权利有效期的判定。
综合复核保留原文材料、来源、作用域及月份计算，不只依赖初判摘要。
各项复核先写理由和 verdict，最后才写答案；本地校验答案与唯一符合项一致。
新约束请求必须返回四项非空 verdict，旧的无 verdict 状态记录仍可读取。
一致性检查不能核验法律规则；不一致响应作为失败保存，不修复或重试。

## 可复现实验

设置模型环境变量后执行；运行 ID 应使用尚不存在的新名称：

```bash
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/lawbench_3_6_qwen35_9b_hybrid.yaml
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6_qwen35_9b_hybrid.yaml \
  --split dev --run-id lawbench_3_6_qwen35_9b_dev_next_hybrid
PYTHONPATH=src uv run python examples/check_accuracy.py \
  --config configs/lawbench_3_6_qwen35_9b_hybrid.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_next_hybrid --official-check
PYTHONPATH=src uv run python -m legal_state.experiment report \
  --config configs/lawbench_3_6_qwen35_9b_hybrid.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_next_hybrid
```

`dev_08_hybrid` 在服务恢复后重新生成全部 45 项，未复用旧基线。
每题三组使用相同模型、题面和闭卷条件，不将参考答案或外部法条输入模型。
所有执行失败与 UNKNOWN 计入准确率分母；严格超过 CoT 才通过开发验收。
完整结果封存后记录在 `accuracy_acceptance.json`、`scores.json` 和 `report.md`。
开发集已用于调试，成绩不能替代正式集或显著性证据。
正式集仍须完成已有人工争点标注、冻结及独立评估流程。

## 历史结果与后续迭代

首版 `dev_08_hybrid` 全部 45 项已封存，官方评分校验一致：
Legal State 为 11/15（73.3%），CoT 为 12/15（80%），目标未达成。
Legal State 无执行失败，但包含一次 UNKNOWN；不能将正常执行等同于法律正确。
配对准确率差为 −6.7 个百分点，95% bootstrap 区间为 [−20.0, 0.0] 个百分点。

第二版 `dev_09_hybrid` 已验证并复用该轮 30 项 CoT / Generic 基线，
模型身份、预算、服务指纹以及每次请求和响应均精确重放一致；Legal 全部新生成。
用户要求暂停时已完成两项 Legal State，第三题调用被中断，尚未形成完整成绩。
暂停记录及后续步骤见 [development_checkpoint.md](development_checkpoint.md)。

后续恢复运行、失败保留和复核一致性改动见
[accuracy_iterations.md](accuracy_iterations.md)。第二版完整运行 `dev_10_hybrid`
为 9/15，低于 CoT 的 12/15，不能作为准确率改进采用。

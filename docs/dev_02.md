# 第二轮开发验证

## 改动与目标

- 三组共同要求简短单步分析、保留必要依据、检查已有结论并及时停止。
- CoT 每步只推进一个未完成判断，完成时输出 `[STOP]`。
- Generic State 的 `plan` 只保留未完成任务；完成时清空并设置 `stop=true`，示例展示完成状态。
- Legal State 提供来自当前状态的字段引用白名单，说明 `COMMIT → RESOLVE → STOP` 路径，避免重复绑定和结论。

本轮修改提示词，不引入自动 JSON 修复、非法引用替换、重复状态强制停止或失败重试。
模型、温度、预算、评分和固定的 15 道开发题沿用第一轮设置。

## 运行

先恢复本机模型服务并确认返回 `Qwen3.5-9B`：

```bash
curl --fail http://127.0.0.1:8000/v1/models
export LSP_MODEL='Qwen3.5-9B'
export LSP_ENDPOINT='http://127.0.0.1:8000/v1/chat/completions'
export LSP_API_KEY='EMPTY'
```

`EMPTY` 仅用于已验证无需认证的本机接口。以下命令从仓库根目录执行：

```bash
PYTHONPATH=src uv run pytest

PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/lawbench_3_6_qwen35_9b_local.yaml

PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6_qwen35_9b_local.yaml \
  --split dev --run-id lawbench_3_6_qwen35_9b_dev_02

PYTHONPATH=src uv run python -m legal_state.experiment score \
  --config configs/lawbench_3_6_qwen35_9b_local.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_02 --official-check

PYTHONPATH=src uv run python -m legal_state.experiment report \
  --config configs/lawbench_3_6_qwen35_9b_local.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_02
```

中断后在同一 run 命令加 `--resume`。保留第一轮 `dev_01`，代码变更后不要续跑第一轮。

## 检查结果

比较两轮的失败率、截断率、正常停止轨迹、准确率及平均总 token。
失败仍计入分母；单独检查非法引用、重复状态、重复 COMMIT 和提前停止。
Legal State 的正常完成样例应有实质结论、已解决争点和 STOP。

协议改善不能独立证明法律推理收益。可人工分析开发题的必要争点与遗漏，记录逐题证据。
若协议改善后仍没有准确率或过程收益，且成本持续增加，停止当前实现与任务组合的继续投入。
正式题的 30 题标注与冻结在决定进入正式验证后完成。

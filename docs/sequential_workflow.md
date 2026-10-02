# Legal State 逐争点推进实验

## 规则

原流程同时允许重复绑定、重复提交、新建争点和停止。
此变体把第一个未解决争点作为当前目标，根据状态只提供下一阶段操作：

| 当前状态 | 允许操作 |
|---|---|
| 没有争点 | `EXPAND_ISSUE` |
| 当前争点为 open | `BIND_FACT` |
| 当前争点为 reasoning，尚无结论 | `COMMIT` |
| 当前争点已有结论 | `RESOLVE` |
| 所有已注册争点解决且有结论 | `EXPAND_ISSUE` 或 `STOP` |

解析后同时检查操作和目标 ID。违反阶段规则仍记为失败，保存轨迹，不自动修复或重试。
完整状态保留；不把达到步数上限强制改成成功停止。

## 边界

每个争点只执行一次绑定和一次结论提交，因此模型应一次绑定当前判断所需材料。
这是对固定输入的简化流程，不适合需要反复补充证据、修订结论或先展开子争点再解决父争点的任务。
停止条件仅检查已注册争点的结构闭合，不保证必要争点覆盖或结论正确。

`legal_state_workflow` 默认 `free`，不改变原流程。
配置 `configs/lawbench_3_6_qwen35_9b_sequential.yaml` 启用 `sequential`，保留原有题干 `F1`。
该实验与选项作用域实验分开，沿用 dev_02 的 15 道开发题、模型、三组设置和预算。

## 执行

使用已配置的本机模型环境，从仓库根目录运行：

```bash
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/lawbench_3_6_qwen35_9b_sequential.yaml
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6_qwen35_9b_sequential.yaml \
  --split dev --run-id lawbench_3_6_qwen35_9b_dev_04_sequential
PYTHONPATH=src uv run python -m legal_state.experiment score \
  --config configs/lawbench_3_6_qwen35_9b_sequential.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_04_sequential --official-check
PYTHONPATH=src uv run python -m legal_state.experiment report \
  --config configs/lawbench_3_6_qwen35_9b_sequential.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_04_sequential
```

中断后，在代码和配置未变化时加 `--resume`。不要续跑旧轮次。
同时比较准确率、非法动作、正常闭合停止、重复动作和 token 成本。
本开发消融不替代正式集验证；正式测试题未参与改造。

## 本轮结果

`lawbench_3_6_qwen35_9b_dev_04_sequential` 已封存 45 项，官方评分一致，服务指纹与 dev_02 相同。

| 指标 | dev_02 原流程 | 逐争点流程 |
|---|---:|---:|
| Legal State 正确数 | 9/15 | 9/15 |
| 执行失败 | 0 | 0 |
| 正常 STOP 且全部已注册争点闭合 | 12/15 | 15/15 |
| 步数上限终止 | 1/15 | 0/15 |
| 平均总 token | 10,516.8 | 8,338.5 |

平均总 token 下降约 20.7%，没有完全相同的结论重复提交或相同材料组合重复绑定。
逐题有两题由错变对、两题由对变错，准确率总体未提升。
CoT 仍为 12/15，Generic 仍为 10/15；Legal State 总 token 约为 CoT 的 6 倍。
新旧 Legal State 配对准确率差的 95% bootstrap 区间为 [-26.7, +26.7] 个百分点。

本变体改善了流程与成本，仍没有证明法律推理收益，暂不进入正式规模实验。
后续优先分析有效但错误的结论及必要争点遗漏，设计规则条件与材料依据的显式映射。
完整比较见 `runs/lawbench_3_6_qwen35_9b_dev_04_sequential/dev_02_vs_sequential.md`、
`dev_02_vs_sequential.csv` 和 `offline_diagnostics.md`。

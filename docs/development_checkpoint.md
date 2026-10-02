# Legal State 开发暂停记录

## 当前结论

2026-10-03（北京时间）按用户要求暂停模型实验并提交代码。
“Legal State 主准确率严格高于 CoT”仍是未完成目标，不能宣布达标。
最新完整运行 `lawbench_3_6_qwen35_9b_dev_08_hybrid`：

| 方法 | 正确数 | 准确率 | 执行失败 |
|---|---:|---:|---:|
| CoT | 12/15 | 80.0% | 0 |
| Legal State + CoT 草稿（首版） | 11/15 | 73.3% | 0 |

评分含所有失败和 UNKNOWN，LawBench 官方评分一致。
该轮在服务恢复后完整新生成三组 45 项；记录、封存哈希、报告和验收文件在
`runs/lawbench_3_6_qwen35_9b_dev_08_hybrid/`。
`runs/` 按现有 `.gitignore` 留在本机，不进入提交。

## 已实现的第二版

- 四选项初判仅看到题干及当前选项；草稿保留至综合复核，减少错误草稿锚定。
- 从当前可见原文计算日期的日历月份差，保留来源；不加入外部法条或参考答案。
- 复核要求选择与理由一致，禁止凭“更全面”等泛泛排序排除实质正确的选项。
- 状态快照、引用作用域、共享预算、原始响应与重放日志仍严格保留。

当前配置为 `configs/lawbench_3_6_qwen35_9b_hybrid.yaml`，实验版本 `hybrid_v2`。
全量测试：`PYTHONPATH=src uv run pytest -q`，**280 passed**；`git diff --check` 通过。

## 暂停位置与恢复

`lawbench_3_6_qwen35_9b_dev_09_hybrid` 复用 dev_08 的 30 项基线，
复用前已精确重放校验每个请求、预算、响应、终止结果和用量，来源记在 manifest。
已完成 Legal State 的 `000255`（A）及 `000223`（B）。
`000348` 的调用索引 4 已发送但未记录响应；SIGINT 中断客户端，运行未封存。
保留这一未知结果，不补写响应或自动重发；`--resume` 会将该调用记为中断失败。

后续验证第二版建议使用新的运行 ID，例如 `lawbench_3_6_qwen35_9b_dev_10_hybrid`，
完整执行 15 道 Legal State。可按混合流程文档重跑三组；若复用基线，须先 preflight，
再运行 `examples/reuse_baselines.py`，来源使用完整封存的 dev_08，不能使用未完成的 dev_09。
然后评分、`--official-check`、`examples/check_accuracy.py` 验收，并更新文档。

执行源快照分别保存在 `runs/setup_qwen35_9b_dev_08_hybrid/execution_source/`
和 `runs/setup_qwen35_9b_dev_09_hybrid/execution_source/`。
正式集仍有 30 题人工争点标注待完成；开发成绩不能证明正式集优势。

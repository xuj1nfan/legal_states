# Legal State 选项材料作用域实验

## 改动与假设

dev_02 的部分题干只有提问，独立案情或待验证法律命题都在选项中。
原实现只有题干 `F1`，无法在状态中明确引用选项材料。
本变体检验：给材料增加可追溯引用和独立作用域，是否改善判断及其依据。

- `F1` 保留完整题干；`F2`–`F5` 分别保留 A–D 的完整原文，不做语义抽取。
- `Fact.material` 标记 `quoted_material`、原始 `question` 的字符范围 `[start,end)` 和选项 `scope`。字符索引遵循 Python 字符串，不是字节位置。
- 引文只是输入材料。选项中的法律命题仍需判断，不能视为已核实规则。
- `EXPAND_ISSUE.scope` 可为已有选项或 `null`；子争点继承父作用域。
- 选项争点只能绑定题干或本选项材料。`COMMIT.support` 同样隔离，并检查中间结论的递归依赖。
- 综合比较争点使用 `scope=null`，可以比较多个选项，引用来源仍保留。

这些检查约束声明的引用，不能检测无引用文字中的案情混用，也不证明法律结论正确。
尚未改动结论修订、规则条件映射、重复动作控制或争点覆盖检查。

## 兼容与对照

`legal_state_materials` 默认 `stem_only`，旧配置继续使用原实现。
空的新增字段不进入旧状态或行动的序列化结果，旧提示词格式保持一致。
新配置 `configs/lawbench_3_6_qwen35_9b_scoped.yaml` 显式启用 `scoped_options`，并写入运行清单。

沿用固定 15 道开发题、三组模型设置、24 步及 8192 输出 token 预算。
CoT / Generic 输入与提示词保持 dev_02 设置。变体增加状态输入长度，必须报告输入及总 token 成本。
不读取标准答案、不添加法律知识、不调用模型预处理、不改动正式题。
这是开发集消融验证，不能据此声称泛化收益。比较 dev_02 前核查模型服务指纹。

## 执行

在仓库根目录设置本机服务环境：

```bash
export LSP_MODEL=Qwen3.5-9B
export LSP_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export LSP_API_KEY=EMPTY
PYTHONPATH=src uv run pytest -q
```

```bash
PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/lawbench_3_6_qwen35_9b_scoped.yaml
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6_qwen35_9b_scoped.yaml \
  --split dev --run-id lawbench_3_6_qwen35_9b_dev_03_scoped
PYTHONPATH=src uv run python -m legal_state.experiment score \
  --config configs/lawbench_3_6_qwen35_9b_scoped.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_03_scoped --official-check
PYTHONPATH=src uv run python -m legal_state.experiment report \
  --config configs/lawbench_3_6_qwen35_9b_scoped.yaml \
  --run-id lawbench_3_6_qwen35_9b_dev_03_scoped
```

中断后只在相同代码和配置下加 `--resume`；代码变更必须另建运行。
同时检查准确率、失败、跨作用域拒绝、引用实际使用、遗漏选项、重复动作与成本。
若结构更明确但准确率和过程质量无收益，保留研究记录，避免继续扩大同一变体实验。

## 本轮结果

`lawbench_3_6_qwen35_9b_dev_03_scoped` 已完成并封存 45 项；与 dev_02 使用同一服务指纹，官方评分一致。
CoT 仍为 12/15，Generic 仍为 10/15。Legal State 从 9/15 降为 5/15，失败 5 题，
平均总 token 从 10,517 增至 33,823；达到步数上限 5 题。
引用隔离被触发，但重复绑定、重复创建争点和重复结论仍然存在。
其中一次模型调用发生上下文超限，无法获知该调用 token 用量；上述成本来自已报告用量。

该变体保留为研究开关，默认继续使用 `stem_only`。
逐题数据与原因见 `runs/lawbench_3_6_qwen35_9b_dev_03_scoped/dev_02_vs_scoped.md`、
`scoped_material_usage.csv` 和 `offline_diagnostics.md`。
下一项独立消融采用原有题干材料，只改变动作推进规则，避免混合归因。

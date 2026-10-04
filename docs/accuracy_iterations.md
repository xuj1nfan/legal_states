# 2026-10-03 准确率迭代

目标是完整配对运行中 Legal State 主准确率严格高于 CoT。执行失败和
UNKNOWN 均计入分母，使用 LawBench 官方评分核对，不以完成率代替准确率。
开发集已反复用于调试，不能据此宣称正式集优势。

## 提交与实验诊断

- `cfd5e2e` 引入实验引擎，记录请求、响应、用量及评分。
- `239b8cc` 固定本机 Qwen3.5-9B 设置。
- `cbc76f8` 引入带来源的选项材料、逐项条件评估、竞争复核与可选 CoT 草稿。
  首版混合流程 `dev_08_hybrid` 为 11/15，CoT 为 12/15，目标没有达成。
- 提交中的第二版隐藏初判阶段的草稿，并加入月份计算和复核提示。
  本次新建 `dev_10_hybrid` 完成了全部 15 项 Legal State；30 项基线来自完整封存的
  `dev_08_hybrid`，逐项请求、预算、响应与用量均通过离线重放校验。
  第二版为 9/15，CoT 为 12/15，官方评分一致，不能作为改进采用。

逐题轨迹显示：模型可能把选项中的合法条件命题当成题干必须已经发生的事实；
可能遗漏决定性的身份、时间或行为分支；也可能在综合复核时推翻正确初判。
`dev_08_hybrid` 的 `000171` 甚至选择 D，却在复核理由中否定 D，最终答案为 UNKNOWN。
结构引用有效不代表规则记忆或法律结论正确。

## 当前实现

混合配置保持相同模型、温度、24 步、单次 512 token、总推理输出 8192 token，
最终答案统一生成且最多 128 token。CoT 与 Generic 的提示词和预算保持原样。

1. 初判可看到暂定草稿，但草稿不是 Fact、给定 Knowledge 或可引用 evidence。
2. 明确区分题干事实与选项提出的条件命题。
3. 综合复核回看带作用域的原文材料与月份计算，避免只依赖模型自己的摘要。
4. 先生成各项理由、判断、最终理由与竞争理由，最后才生成答案字母。
5. 新复核包含每项 verdict。只有唯一一项 meets_question 时才能选择它；
   零项或多项未解决时为 UNKNOWN。该检查只保证声明的一致性，不证明法律正确。
6. 旧的无 verdict 决策仍可读取；新约束 JSON 请求必须返回四项非空 verdict。
   不一致输出保留为失败，不自动修复或重试。

仅有提示词修改的第三版由 `dev_11_hybrid` 开始完整重跑三组。
服务在 `000353` 调用过程中断开，后续记录连续失败；该轮已完整保留并封存，
计入所有失败后 Legal State 为 5/15、CoT 为 6/15，官方评分一致。
服务恢复后新建 `dev_12_hybrid` 重新运行，未覆盖或补写原失败记录。
第三版完整重跑三组共 45 项后为 Legal State 10/15、CoT 10/15，目标未达成。
CoT 有两项因推理历史增长超过服务 4096 token 上下文而失败，仍计入分母。
第四版 `dev_13_hybrid` 对照重放校验过的 `dev_08_hybrid` 基线，
为 Legal State 11/15、CoT 12/15；无 Legal State 执行失败，官方评分一致。
第四版修复了第三版 `000348` 由正确初判被复核改错的问题，尚未解决规则记忆错误。

## 可选争点检查计划

`configs/lawbench_3_6_qwen35_9b_framed.yaml` 在混合流程上启用
`legal_state_question_frame: true`，仅适用于带作用域材料的 verified 流程。
在四项初判前增加一次 `FRAME_QUESTION`：

- 只读取原题、带引用的原文与日期算术，不读取草稿，不预先选择答案。
- 将可能决定规则分支的具体疑问保存为 `question_frame.checks`。
- 检查项只引用题干或其自身选项；引用未知对象、草稿或其他选项均被拒绝。
- 每个初判只看到相关检查项；复核看到完整计划，以核查遗漏的疑问。
- 新计划必须包含原文 `trigger_quote`。生成语法将引用限制为原文短句白名单，
  同时限制作用域、材料 ID 及引文所属材料，执行后仍重新校验快照。
  白名单只做原文字符切分，不引入法条、题目答案或法律判断。
- 计划不写入 Fact 或给定 Knowledge，不作为条件的可引用依据。

正常路径为八次推理调用及一次最终答案调用，仍消耗共享的 24 步和 8192 token。
计划只允许在初判前设置一次，保留完整快照和重放日志。
关闭该开关时，已完成的第四版请求、结果和用量通过精确离线重放校验。
开发对照均使用新的运行 ID，不覆盖既有运行：

| 运行 | Legal State | CoT | 发现 |
|---|---:|---:|---|
| dev_14_framed | 8/15 | 12/15 | 四项跨选项引用失败，保留并计入分母 |
| dev_15_framed | 5/15 | 12/15 | 自由生成引文不逐字匹配原文，失败未修复重试 |
| dev_16_framed | 10/15 | 12/15 | 原文白名单消除上述引用失败，仍有规则错误及一次截断 |
| dev_17_framed | 10/15 | 12/15 | 精简提示、条件先生成；无失败但没有准确率收益 |
| dev_18_framed | 11/15 | 12/15 | 去掉 CoT 草稿后 171、353、230 正确，另有一项漏引用失败 |
| dev_19_framed | 12/15 | 12/15 | 独立评估加引用语法约束；一次复核空白循环截断计为失败 |
| dev_20_framed | 9/15 | 10/15 | 全新生成 45 项；全程紧凑语法消除空白循环，但规则判断回退 |
| dev_21_framed | 8/15 | 12/15 | 仅复核使用紧凑语法并要求选择前缀；整体回退，不采用该提示 |
| dev_22_framed | 10/15 | 12/15 | 先回忆条件规则；仍将竞合误判为吸收，另有一次选择声明冲突 |

`dev_16_framed` 的 `000230` 原始复核已写出 D，但 JSON 没有闭合且
服务报告 length，仍作为执行失败计分，不能从部分响应提取 D 补成正确答案。
其他错误包括 `000171` 的被告条件分支、`000353` 的再审程序和
`000276` 的权利期限。定位了事实并不保证能正确回忆法律规则。

`dev_17_framed` 改为先生成 conditions，再生成 rule、例外及 verdict。
`dev_18_framed` 在相同条件先生成流程中去掉草稿，表明草稿锚定可能造成回退；
但正确答案不代表理由均符合法律，仍需独立过程评估。

`dev_19_framed` 保留条件先生成与无草稿，恢复详细复核提示；
生成语法限定当前选项、可见引用 ID，并要求第一项条件包含原文引用。
后续缺失条件仍允许 evidence=[]，避免强迫引用不存在的依据。
预检以独立合成题实际请求 FRAME_QUESTION 和 ASSESS_OPTION，记录语法与原始响应，
防止到正式题目才发现语法不受服务支持。预算、CoT/Generic 提示和封存基线保持原样。

该轮 `000230` 复核写出了 D，但未闭合最外层括号，随后反复生成空白直至
finish_reason=length，原样计为失败。新配置开启 `legal_state_compact_json: true`，
仅对带 JSON schema 的请求编译明确的 EBNF 并发送 `structured_outputs.grammar`。
语法禁止字段间空白，保留字符串内的空格与转义，固定字段顺序，并保留
原有枚举、引文所属材料、当前选项、首项引用和数组长度限制。
不支持的 schema 约束直接拒绝，避免忽略约束后生成不合要求的数据。
CoT、Generic、自然草稿及最终转写的 HTTP 请求保持原样，原始输出仍不修复。
请求级 disable_any_whitespace 在本机预检中未生效，未将其作为有效修复采用。
预检另实际覆盖完整嵌套 AUDIT_OPTIONS 语法，检查字段间空白是否泄漏；
保存每步 schema、实际语法与原始响应。测试用独立 Lark 解析器验证允许及拒绝的动作。
API 参考 [vLLM 结构化输出](https://docs.vllm.ai/en/latest/features/structured_outputs/)。
开发验证使用新运行 `dev_20_framed`，三组共 45 项全部重新生成，不复用旧基线。

`dev_20_framed` 仍未达标。CoT 两项因上下文长度失败，全部计分；
Legal State 的 `000124` 虽选 A，但把法律正确的 B、C、D 标成 meets_question，
与选择声明冲突，执行器原样拒绝，未从响应补出正确答案。
全程更换语法也改变了题面定位和初判生成，不能以格式通过率推断准确率改善。
`dev_21_framed` 使用 `legal_state_compact_json_scope: audit`，
FRAME_QUESTION 和 ASSESS_OPTION 保持原 JSON Schema 请求，只对最终复核使用紧凑语法。
复核明确要求 reason 说明应选、不应选或无法确定，避免将表述正确性与选择标签混同。
本轮对照仍使用重放校验过的历史 CoT 12/15，而非较低的新基线。

`dev_22_framed` 恢复第 19 轮的复核提示，保留仅复核使用紧凑语法。
新配置 `configs/lawbench_3_6_qwen35_9b_framed_rules.yaml` 设置
`legal_state_initial_analysis: rules`：先闭卷回忆法律关系的规则、条件和例外，
要求不选择选项。这段模型回忆保存在 draft_reasoning 和原始 draft_step 中，
标记 kind=model_rule_recall，不变成 Fact 或给定 Knowledge，不能作为引用依据。
它消耗同一个步数和输出配额，正常路径九次调用（含最终转写），不提供外部法条或答案。
FRAME_QUESTION 仍不读取这段回忆；初判与复核将其视为待验证假设。

第 22 轮未超过 CoT。`000348` 的规则回忆已写出想象竞合择一重处断，
但复核又改成吸收关系并选 B。下一轮不靠增加格式约束解决该法律分歧：
`legal_state_compare_options: true` 让初判看到全部选项文字以辨别待比较的主张；
可引用材料仍只有题干和当前选项，其他选项不成为该选项的事实或引用依据。
日期运算也按题干加单个选项分别生成，避免把不同备选案例的日期拼接成同一时间线。
以上是待验证假设，尚无准确率收益证据；使用第 23 轮独立快照与新预检验证。

离线核对已保存的四项初判与最终结果，第 22 轮复核把 `000348` 的 C、
`000353` 的 D 从正确改成错误，同时纠正 `000230`、`000372`、`000394`。
因此既不能假定复核必然提升准确率，也不能直接取消复核。
诊断文件为 `runs/accuracy_audit_regressions_20261003.json`，仅用于离线分析；
没有保存完整初判的失败记录不计入这项阶段比较，仍计入最终准确率分母。

第 23 轮快照已经保存，全量测试 318 项通过。但新预检因
`127.0.0.1:8000` 返回 Connection refused 失败，检查确认无监听端口；
没有开始第 23 轮题目运行，也没有新准确率成绩。
需要恢复原模型服务再重新预检；服务问题只阻碍下一轮验证，
不解释此前已完成运行中的准确率回退。

代码快照保存在对应的 `runs/setup_qwen35_9b_dev_<轮次>_<流程>/execution_source/`。
运行中的进程使用其启动时的代码；新版本使用新的预检、运行 ID 和代码快照。
pytest 只从 `tests/` 收集，避免实验快照中的同名测试混入根目录测试。

## 复现与解释范围

运行前设置本机模型环境变量，执行：

```bash
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run python -m legal_state.experiment preflight \
  --config configs/lawbench_3_6_qwen35_9b_hybrid.yaml
PYTHONPATH=src uv run python -m legal_state.experiment run \
  --config configs/lawbench_3_6_qwen35_9b_hybrid.yaml \
  --split dev --run-id <新的运行ID>
PYTHONPATH=src uv run python examples/check_accuracy.py \
  --config configs/lawbench_3_6_qwen35_9b_hybrid.yaml \
  --run-id <新的运行ID> --official-check
PYTHONPATH=src uv run python -m legal_state.experiment report \
  --config configs/lawbench_3_6_qwen35_9b_hybrid.yaml --run-id <新的运行ID>
```

这是 CoT 草稿与 Legal State 复核的混合方法，收益不能归因于纯状态表示。
正式集仍需完成并复核 30 道必要争点标注，冻结后独立验证。

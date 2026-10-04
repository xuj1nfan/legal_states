"""Research tables and a fixed, balanced case-analysis selection."""

import csv
from pathlib import Path

from legal_state.experiment.config import ExperimentConfig
from legal_state.experiment.evaluation import load_run, score
from legal_state.experiment.io import read_json, write_json, write_text


def report(config: ExperimentConfig, root: Path, run_id: str) -> dict:
    path, manifest, records = load_run(config, root, run_id)
    scores = score(config, root, run_id)
    rows = scores["summary"]
    is_mslr = scores.get("benchmark") == "mslr"
    process_path = path / "process_scores.json"
    process = read_json(process_path) if process_path.exists() else None
    if process is not None:
        if process.get("run_identity_sha256") != manifest["identity_sha256"]:
            raise ValueError("Process scores belong to a different frozen run")
        snapshot = (
            path / "human_review" / "imports" / f"{process['ratings_sha256']}.json"
        )
        if read_json(snapshot) != process:
            raise ValueError("Process scores changed after import")
    lines = [
        f"# {'MSLR 多步法律推理' if is_mslr else 'LawBench 3-6 案例子集'}实验：{run_id}",
        "",
        f"划分：{scores['split']}；样本数：{len(manifest['identity']['case_ids'])}。",
        "三组共享模型、原题、调用上限和输出预算；无额外法律知识。",
        "",
        "## 最终答案与运行质量",
        "",
        (
            "| 方法 | IRAC Recall | 空输出率 | 执行失败率 | 格式失败率 | 预算/步数受限率 |"
            if is_mslr
            else "| 方法 | Accuracy | 无效答案率 | 执行失败率 | 格式失败率 | 预算/步数受限率 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        primary = row["irac_recall"] if is_mslr else row["accuracy"]
        empty = row["empty_prediction_rate"] if is_mslr else row["abstention_rate"]
        lines.append(
            f"| {row['method']} | {primary:.2%} | {empty:.2%} | {row['failure_rate']:.2%} | {row['format_failure_rate']:.2%} | {row['limited_termination_rate']:.2%} |"
        )
    if is_mslr and scores["metric"] == "literal_field_recall":
        lines += [
            "",
            "当前为无需嵌入模型的字面字段覆盖率，仅用于快速诊断；正式结果应使用 --official-check 和 ChatLaw-Text2Vec。",
        ]
    lines += [
        "",
        "## 成本",
        "",
        "| 方法 | 平均输入 token | 平均输出 token | 平均总 token | 中位总 token | 平均调用数 | 平均调用耗时秒 | 未知用量调用数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['avg_input_tokens']:.1f} | {row['avg_output_tokens']:.1f} | {row['avg_total_tokens']:.1f} | {row['median_total_tokens']:.1f} | {row['avg_calls']:.2f} | {row['avg_latency_seconds']:.2f} | {row['unknown_usage_calls']} |"
        )
    lines += [
        "",
        "已知用量是报告值；失败或中断调用可能产生未报告费用。耗时为各次调用等待时间之和，中断期间停机时间不计入。",
        "",
        "## 配对比较",
        "",
    ]
    for name, comparison in scores["comparisons"].items():
        lo, hi = comparison["ci95"]
        lines.append(
            f"- {name}：{'IRAC Recall' if is_mslr else '准确率'}差 {comparison['difference'] * 100:+.2f} 个百分点，95% 配对 bootstrap 区间 [{lo * 100:+.2f}, {hi * 100:+.2f}]。"
        )
    lines += ["", "## 人工过程评估", ""]
    if process is None:
        lines.append("尚未导入完整人工评分。不能据此判断争点覆盖或语义矛盾是否改善。")
    else:
        lines += [
            "| 方法 | 案例数 | 必要争点覆盖率（案例平均） | 争点遗漏率 | 语义矛盾率 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in process["summary"]:
            lines.append(
                f"| {row['method']} | {row['count']} | {row['issue_coverage_rate']:.2%} | {row['omission_rate']:.2%} | {row['semantic_contradiction_rate']:.2%} |"
            )
    lines += [
        "",
        "## 解释范围",
        "",
        "本结果覆盖冻结的案例子集；零外部知识设置同时包含知识不足与推理失误。",
        "状态字段、操作约束和校验共同构成方法，不能将差异全部归因于字段结构。",
        "相同预算上限不意味着实际输入成本相同。人工材料隐去方法名与正确性，但格式可能暴露方法。",
        "置信区间跨零时，将方向性差异视为证据不足；本轮是冻结子集上的受控比较。",
        "",
        "## 复现",
        "",
        f"数据与代码哈希、模型身份、解码设置及环境记录见 manifest.json；样本映射见 {'mslr' if is_mslr else 'lawbench'}/sample_mapping.json。",
    ]
    write_text(path / "report.md", "\n".join(lines) + "\n")
    per_case = {}
    for row in records:
        per_case.setdefault(row["case_id"], {})[row["method"]] = row
    if is_mslr:
        with (path / "per_case_scores.csv").open(encoding="utf-8", newline="") as handle:
            scored = list(csv.DictReader(handle))
        values_by_case = {
            cid: {
                row["method"]: float(row["irac_recall"])
                for row in scored
                if row["case_id"] == cid
            }
            for cid in manifest["identity"]["case_ids"]
        }
    else:
        exported = {
            method: read_json(path / "lawbench" / method / "3-6.json")
            for method in config.methods
        }
    categories = {
        name: []
        for name in ("legal_win", "legal_loss", "all_wrong", "all_correct", "mixed")
    }
    for index, cid in enumerate(manifest["identity"]["case_ids"]):
        values = (
            values_by_case[cid]
            if is_mslr
            else {
                method: item[str(index)]["prediction"] == item[str(index)]["refr"][5]
                for method, item in exported.items()
            }
        )
        if is_mslr:
            legal = values["legal_state"]
            baselines = (values["generic"], values["cot"])
            if legal > max(baselines):
                category = "legal_win"
            elif legal < max(baselines):
                category = "legal_loss"
            elif all(value == 0 for value in values.values()):
                category = "all_wrong"
            elif all(value == 1 for value in values.values()):
                category = "all_correct"
            else:
                category = "mixed"
            categories[category].append(cid)
            continue
        if values["legal_state"] and not values["generic"] and not values["cot"]:
            category = "legal_win"
        elif not values["legal_state"] and (values["generic"] or values["cot"]):
            category = "legal_loss"
        elif not any(values.values()):
            category = "all_wrong"
        elif all(values.values()):
            category = "all_correct"
        else:
            category = "mixed"
        categories[category].append(cid)
    for category, ids in categories.items():
        ids.sort(
            key=(
                lambda cid: (
                    -sum(r["usage"]["total_tokens"] for r in per_case[cid].values()),
                    cid,
                )
            )
            if category == "all_correct"
            else None
        )
    chosen = [
        (category, cid)
        for category, ids in categories.items()
        if category != "mixed"
        for cid in ids[:5]
    ]
    selected_ids = {cid for _, cid in chosen}
    leftovers = [
        (category, cid)
        for category, ids in categories.items()
        for cid in ids
        if cid not in selected_ids
    ]
    chosen += leftovers[: max(0, min(20, len(per_case)) - len(chosen))]
    template = path / "case_analysis.json"
    if not template.exists():
        write_json(
            template,
            [
                {
                    "case_id": cid,
                    "category": category,
                    "failure_types": [],
                    "description": "",
                    "possible_cause": "",
                    "evidence": "",
                    "possible_fix": "",
                }
                for category, cid in chosen
            ],
        )
    return {
        "report": str(path / "report.md"),
        "process_review_complete": process is not None,
        "case_analysis": str(template),
    }

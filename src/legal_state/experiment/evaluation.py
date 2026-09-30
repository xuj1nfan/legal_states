"""Final-answer scoring, upstream parity and paired case-level statistics."""

import ast
import csv
import io
import math
import random
from pathlib import Path
from statistics import mean, median

from legal_state.experiment.config import ExperimentConfig
from legal_state.experiment.data import (
    dataset_fingerprint,
    load_cases,
    upstream_revision,
)
from legal_state.experiment.io import (
    file_digest,
    read_json,
    read_jsonl,
    source_fingerprint,
    write_json,
    write_text,
)
from legal_state.experiment.storage import completed_records, run_path


def choice_judge(prediction: str, reference: str) -> dict:
    """LawBench multi_choice_judge semantics (Apache-2.0 upstream code).

    Source: open-compass/LawBench/evaluation/utils/function_utils.py.
    Count each A-D at most once. Abstention only means no A-D appeared.
    """
    if reference not in ("A", "B", "C", "D"):
        raise ValueError("Reference must be a single A-D letter")
    present = {letter for letter in "ABCD" if letter in prediction}
    return {"score": int(present == {reference}), "abstention": int(not present)}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buffer.getvalue())


def load_run(
    config: ExperimentConfig, root: Path, run_id: str
) -> tuple[Path, dict, list[dict]]:
    path = run_path(config, run_id, root)
    manifest = read_json(path / "manifest.json")
    data = root / manifest["identity"]["config"]["data_dir"]
    actual = dataset_fingerprint(
        data, annotations=manifest["identity"]["split"] == "test"
    )
    if actual != manifest["identity"]["dataset"]:
        raise ValueError("Dataset or reference annotations changed after freezing")
    return path, manifest, completed_records(path, manifest)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_comparison(
    first: list[int], second: list[int], seed: int, samples: int = 10000
) -> dict:
    if not first or len(first) != len(second):
        raise ValueError("Paired comparison requires equally sized nonempty lists")
    differences = [a - b for a, b in zip(first, second, strict=True)]
    rng = random.Random(seed)
    draws = [
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    ]
    return {
        "difference": mean(differences),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "first_only_correct": sum(
            a == 1 and b == 0 for a, b in zip(first, second, strict=True)
        ),
        "second_only_correct": sum(
            a == 0 and b == 1 for a, b in zip(first, second, strict=True)
        ),
        "both_correct": sum(a == b == 1 for a, b in zip(first, second, strict=True)),
        "both_wrong": sum(a == b == 0 for a, b in zip(first, second, strict=True)),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def check_official(
    path: Path,
    upstream: Path,
    expected_commit: str | None,
    expected_hashes: dict | None = None,
) -> dict:
    if (
        expected_commit is not None
        and upstream_revision(upstream / "data/zero_shot/3-6.json") != expected_commit
    ):
        raise ValueError("Upstream revision differs from prepared dataset")
    module_path = upstream / "evaluation/evaluation_functions/jec_ac.py"
    for name, expected in (expected_hashes or {}).items():
        if file_digest(upstream / "evaluation" / name) != expected:
            raise ValueError("Upstream scoring source changed after data preparation")
    if not module_path.exists():
        raise ValueError("Missing upstream jec_ac.py; obtain LawBench evaluation files")
    compute = load_official_scorer(upstream)
    reports = {}
    for method in ("cot", "generic", "legal_state"):
        exported = read_json(path / "lawbench" / method / "3-6.json")
        reports[method] = compute([exported[str(i)] for i in range(len(exported))])
    return reports


def load_official_scorer(upstream: Path):
    """Execute the two unmodified pure upstream functions, without unrelated imports.

    function_utils.py imports ROUGE/NLTK for other tasks. Extracting these exact
    AST definitions preserves the official 3-6 implementation and needs neither.
    The checkout is trusted code obtained explicitly by the operator.
    """
    namespace = {"__name__": "lawbench_official"}
    for relative, name in (
        ("utils/function_utils.py", "multi_choice_judge"),
        ("evaluation_functions/jec_ac.py", "compute_jec_ac"),
    ):
        path = upstream / "evaluation" / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(functions) != 1:
            raise ValueError(f"Expected exactly one upstream function {name}")
        module = ast.Module(body=functions, type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 -- exact functions from verified upstream checkout
    return namespace["compute_jec_ac"]


def score(
    config: ExperimentConfig, root: Path, run_id: str, official_check: bool = False
) -> dict:
    path, manifest, records = load_run(config, root, run_id)
    data = root / manifest["identity"]["config"]["data_dir"]
    targets = {
        row["case_id"]: row["gold_answer"] for row in read_jsonl(data / "targets.jsonl")
    }
    cases = load_cases(data)
    selected = manifest["identity"]["case_ids"]
    rows_by_key = {(row["case_id"], row["method"]): row for row in records}
    summaries, scores_by_method, per_case = [], {}, []
    for method in manifest["identity"]["config"]["methods"]:
        exported, evaluations = {}, []
        for index, cid in enumerate(selected):
            row = rows_by_key[cid, method]
            answer = row.get("final_answer")
            prediction = (
                answer
                if row["status"] == "completed" and answer in ("A", "B", "C", "D")
                else ""
            )
            evaluation = choice_judge(prediction, targets[cid])
            evaluations.append(evaluation)
            exported[str(index)] = {
                "origin_prompt": [{"role": "HUMAN", "prompt": cases[cid].question}],
                "prediction": prediction,
                "refr": f"正确答案:{targets[cid]}。",
            }
            per_case.append(
                {
                    "case_id": cid,
                    "method": method,
                    "prediction": prediction,
                    "reference": targets[cid],
                    "correct": evaluation["score"],
                    "status": row["status"],
                }
            )
        scores_by_method[method] = [evaluation["score"] for evaluation in evaluations]
        subset = [rows_by_key[cid, method] for cid in selected]
        completed = [
            evaluation["score"]
            for cid, evaluation in zip(selected, evaluations, strict=True)
            if rows_by_key[cid, method]["status"] == "completed"
        ]
        stages = [((row.get("failure") or {}).get("stage")) for row in subset]
        counts = [row["usage"] for row in subset]
        summaries.append(
            {
                "method": method,
                "count": len(selected),
                "accuracy": mean(scores_by_method[method]),
                "abstention_rate": mean(e["abstention"] for e in evaluations),
                "failure_rate": mean(row["status"] == "failed" for row in subset),
                "model_call_failure_rate": mean(
                    stage == "model_call" for stage in stages
                ),
                "truncation_rate": mean(row["truncated_calls"] > 0 for row in counts),
                "format_failure_rate": mean(
                    stage in {"parse_action", "parse_reasoning", "parse_final_answer"}
                    for stage in stages
                ),
                "limited_termination_rate": mean(
                    row["termination_reason"] in {"max_steps", "token_budget"}
                    for row in subset
                ),
                "completed_accuracy": mean(completed) if completed else None,
                "avg_input_tokens": mean(row["input_tokens"] for row in counts),
                "avg_output_tokens": mean(row["output_tokens"] for row in counts),
                "avg_total_tokens": mean(row["total_tokens"] for row in counts),
                "median_total_tokens": median(row["total_tokens"] for row in counts),
                "avg_calls": mean(row["call_count"] for row in counts),
                "median_calls": median(row["call_count"] for row in counts),
                "avg_latency_seconds": mean(row["latency_seconds"] for row in subset),
                "median_latency_seconds": median(
                    row["latency_seconds"] for row in subset
                ),
                "unknown_usage_calls": sum(
                    row["unknown_usage_calls"] for row in counts
                ),
            }
        )
        write_json(path / "lawbench" / method / "3-6.json", exported)
    comparisons = {
        f"legal_state_minus_{other}": paired_comparison(
            scores_by_method["legal_state"],
            scores_by_method[other],
            manifest["identity"]["config"]["seed"] + 1,
        )
        for other in ("generic", "cot")
    }
    result = {
        "run_id": run_id,
        "split": manifest["identity"]["split"],
        "summary": summaries,
        "comparisons": comparisons,
        "scoring_source_sha256": source_fingerprint(root),
        "original_execution_source_sha256": manifest["identity"]["source_sha256"],
    }
    if official_check:
        validation = read_json(data / "validation.json")
        upstream = root / manifest["identity"]["config"]["upstream_dir"]
        result["official"] = check_official(
            path,
            upstream,
            validation.get("upstream_commit"),
            validation.get("upstream_scorer_hashes"),
        )
        for summary in summaries:
            actual = result["official"][summary["method"]]
            if not math.isclose(
                actual["score"], summary["accuracy"]
            ) or not math.isclose(
                actual["abstention_rate"], summary["abstention_rate"]
            ):
                raise ValueError("Official scoring parity check failed")
    elif (path / "scores.json").exists():
        previous = read_json(path / "scores.json")
        if (
            previous.get("scoring_source_sha256") == result["scoring_source_sha256"]
            and previous.get("summary") == summaries
            and "official" in previous
        ):
            result["official"] = previous["official"]
    write_json(path / "scores.json", result)
    write_csv(path / "summary.csv", summaries)
    write_csv(path / "per_case_scores.csv", per_case)
    write_json(
        path / "lawbench" / "sample_mapping.json",
        {str(i): cid for i, cid in enumerate(selected)},
    )
    return result

"""Anonymous process-review materials and validated human ratings."""

import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from legal_state.experiment.config import ExperimentConfig
from legal_state.experiment.data import load_cases, validate_annotations
from legal_state.experiment.evaluation import load_run, write_csv
from legal_state.experiment.io import file_digest, read_json, write_json, write_text

ERROR_CATEGORIES = {
    "issue_omission",
    "false_issue",
    "fact_binding_error",
    "knowledge_error",
    "state_contradiction",
    "premature_resolution",
    "unsupported_conclusion",
    "reasoning_error",
    "format_error",
}


def export_review(config: ExperimentConfig, root: Path, run_id: str) -> dict:
    path, manifest, records = load_run(config, root, run_id)
    if manifest["identity"]["split"] != "test":
        raise ValueError("Human process review is defined on the held-out test split")
    destination = path / "human_review"
    if destination.exists():
        raise ValueError(
            "Human review already exists; preserve the existing materials and ratings"
        )
    data = root / manifest["identity"]["config"]["data_dir"]
    cases = load_cases(data)
    annotations = {row["case_id"]: row for row in validate_annotations(data)}
    selected = [row for row in records if row["case_id"] in annotations]
    random.Random(manifest["identity"]["config"]["seed"] + 2).shuffle(selected)
    mapping, ratings, material_hashes = {}, [], {}
    for index, row in enumerate(selected):
        review_id = f"R{index + 1:03d}"
        cid = row["case_id"]
        mapping[review_id] = {
            "case_id": cid,
            "method": row["method"],
            "failed": row["status"] == "failed",
        }
        # No prompts, method labels, automatic correctness or reference answer.
        history = []
        for step in row["steps"]:
            history.append(
                {
                    "raw_output": step.get(
                        "raw_model_output", step.get("call", {}).get("raw_text", "")
                    ),
                    "after": step.get("after_state", step.get("after")),
                }
            )
        material = {
            "review_id": review_id,
            "case_id": cid,
            "question": cases[cid].question,
            "required_issues": annotations[cid]["required_issues"],
            "incomplete_run": row["status"] == "failed",
            "reasoning_artifact": row["reasoning_artifact"],
            "history": history,
        }
        filename = destination / "materials" / f"{review_id}.json"
        write_json(filename, material)
        material_hashes[review_id] = file_digest(filename)
        for issue in annotations[cid]["required_issues"]:
            ratings.append(
                {
                    "review_id": review_id,
                    "issue_id": issue["issue_id"],
                    "description": issue["description"],
                    "coverage_criterion": issue["coverage_criterion"],
                    "identified": "no" if row["status"] == "failed" else "",
                    "analyzed": "no" if row["status"] == "failed" else "",
                    "contradiction": "",
                    "error_categories": "",
                    "notes": "",
                }
            )
    write_json(
        destination / "private_mapping.json",
        {
            "run_identity_sha256": manifest["identity_sha256"],
            "mapping": mapping,
            "material_hashes": material_hashes,
        },
    )
    write_json(
        destination / "export_manifest.json",
        {"private_mapping_sha256": file_digest(destination / "private_mapping.json")},
    )
    write_csv(destination / "materials" / "ratings.csv", ratings)
    return {
        "materials": len(mapping),
        "issue_rows": len(ratings),
        "review_dir": str(destination / "materials"),
    }


def import_review(
    config: ExperimentConfig, root: Path, run_id: str, ratings_file: Path
) -> dict:
    path, manifest, _ = load_run(config, root, run_id)
    review = path / "human_review"
    private = read_json(review / "private_mapping.json")
    if (
        file_digest(review / "private_mapping.json")
        != read_json(review / "export_manifest.json")["private_mapping_sha256"]
    ):
        raise ValueError("Private review mapping changed after export")
    if private["run_identity_sha256"] != manifest["identity_sha256"]:
        raise ValueError("Review belongs to a different frozen experiment")
    data = root / manifest["identity"]["config"]["data_dir"]
    annotations = {row["case_id"]: row for row in validate_annotations(data)}
    expected = set()
    for review_id, item in private["mapping"].items():
        if (
            file_digest(review / "materials" / f"{review_id}.json")
            != private["material_hashes"][review_id]
        ):
            raise ValueError("Review material changed after export")
        expected.update(
            (review_id, issue["issue_id"])
            for issue in annotations[item["case_id"]]["required_issues"]
        )
    seen, grouped = set(), defaultdict(list)
    with ratings_file.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["review_id"], row["issue_id"]
            if key not in expected or key in seen:
                raise ValueError(f"Unknown/duplicate rating: {key}")
            seen.add(key)
            for name in ("identified", "analyzed", "contradiction"):
                row[name] = row[name].strip().lower()
                if row[name] not in {"yes", "no"}:
                    raise ValueError(f"{key}: {name} must be yes or no")
            if row["analyzed"] == "yes" and row["identified"] != "yes":
                raise ValueError("An analyzed issue must also be identified")
            item = private["mapping"][row["review_id"]]
            if item["failed"] and (
                row["identified"] == "yes" or row["analyzed"] == "yes"
            ):
                raise ValueError(
                    "Failed runs must have zero issue coverage in the primary process metric"
                )
            categories = {
                value.strip()
                for value in row["error_categories"].split(",")
                if value.strip()
            }
            if not categories.issubset(ERROR_CATEGORIES):
                raise ValueError(
                    f"Unsupported error categories: {categories - ERROR_CATEGORIES}"
                )
            row["error_categories"] = sorted(categories)
            grouped[row["review_id"]].append(row)
    if seen != expected:
        raise ValueError("Every exported issue row must be rated")
    case_scores = []
    for review_id, rows in grouped.items():
        if len({row["contradiction"] for row in rows}) != 1:
            raise ValueError(
                "Case-level contradiction rating must be consistent across its issues"
            )
        item = private["mapping"][review_id]
        case_scores.append(
            {
                "review_id": review_id,
                "case_id": item["case_id"],
                "method": item["method"],
                "required_issues": len(rows),
                "identified": sum(row["identified"] == "yes" for row in rows),
                "analyzed": sum(row["analyzed"] == "yes" for row in rows),
                "contradiction": rows[0]["contradiction"] == "yes",
                "error_categories": sorted(
                    {category for row in rows for category in row["error_categories"]}
                ),
            }
        )
    summaries = []
    for method in manifest["identity"]["config"]["methods"]:
        rows = [row for row in case_scores if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "count": len(rows),
                "issue_identification_rate": mean(
                    row["identified"] / row["required_issues"] for row in rows
                ),
                "issue_coverage_rate": mean(
                    row["analyzed"] / row["required_issues"] for row in rows
                ),
                "pooled_issue_coverage": sum(row["analyzed"] for row in rows)
                / sum(row["required_issues"] for row in rows),
                "omission_rate": mean(
                    row["analyzed"] < row["required_issues"] for row in rows
                ),
                "semantic_contradiction_rate": mean(
                    row["contradiction"] for row in rows
                ),
            }
        )
    result = {
        "run_identity_sha256": manifest["identity_sha256"],
        "summary": summaries,
        "case_scores": case_scores,
        "ratings_sha256": file_digest(ratings_file),
        "ratings": list(grouped.values()),
    }
    write_text(
        review / "imports" / f"{result['ratings_sha256']}.csv",
        ratings_file.read_bytes().decode("utf-8"),
    )
    write_json(review / "imports" / f"{result['ratings_sha256']}.json", result)
    write_json(path / "process_scores.json", result)
    write_csv(path / "process_summary.csv", summaries)
    return result

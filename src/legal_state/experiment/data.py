"""LawBench adaptation, human eligibility review and frozen sample splits."""

import csv
import io
import random
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_state.experiment.io import (
    digest,
    file_digest,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_text,
)


class CaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    case_id: str = Field(pattern=r"^lawbench-3-6-[0-9]{6}$")
    source_index: int = Field(ge=0)
    question: str = Field(min_length=1)
    stem: str = Field(min_length=1)
    options: dict[str, str]

    @model_validator(mode="after")
    def validate_material(self):
        if set(self.options) != set("ABCD") or any(
            not value.strip() for value in self.options.values()
        ):
            raise ValueError("Require nonempty A/B/C/D options")
        if self.case_id != f"lawbench-3-6-{self.source_index:06d}":
            raise ValueError("Case ID does not match source index")
        if (
            self.stem + "".join(f"{letter}:{self.options[letter]}" for letter in "ABCD")
            != self.question
        ):
            raise ValueError("Stem/options must preserve the original question exactly")
        return self


def upstream_revision(source: Path) -> str | None:
    # Only identify a LawBench checkout, not the enclosing pilot repository.
    repository = source.parent.parent.parent
    if not (repository / "evaluation").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def prepare(source: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; use a new dataset version")
    rows = read_json(source)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source must be a nonempty JSON list")
    inputs, targets, errors, duplicates = [], [], [], []
    seen: set[str] = set()
    option_pattern = re.compile(r"^(.*?)A:(.*?)B:(.*?)C:(.*?)D:(.*)$", re.DOTALL)
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("instruction", "question", "answer")
            ):
                raise ValueError("Missing/non-string/empty source field")
            match = option_pattern.fullmatch(row["question"])
            if match is None or any(not part.strip() for part in match.groups()):
                raise ValueError("Missing stem or A/B/C/D options")
            gold = re.fullmatch(r"正确答案[:：]\s*([A-D])[。.]?\s*", row["answer"])
            if gold is None:
                raise ValueError("Reference must contain exactly one A-D answer")
            normalized = re.sub(r"\s+", "", row["question"])
            if normalized in seen:
                duplicates.append(index)
                continue
            seen.add(normalized)
            case = CaseInput(
                case_id=f"lawbench-3-6-{index:06d}",
                source_index=index,
                question=row["question"],
                stem=match[1],
                options=dict(zip("ABCD", match.groups()[1:], strict=True)),
            )
            inputs.append(case.model_dump())
            targets.append(
                {
                    "case_id": case.case_id,
                    "gold_answer": gold[1],
                    "source_answer": row["answer"],
                }
            )
        except ValueError as error:
            errors.append({"source_index": index, "error": str(error)})
    if not inputs:
        raise ValueError("No valid cases")
    report = {
        "source": str(source.resolve()),
        "source_sha256": file_digest(source),
        "upstream_commit": upstream_revision(source),
        "source_count": len(rows),
        "valid_count": len(inputs),
        "duplicates": duplicates,
        "errors": errors,
        "question_chars": {
            "min": min(len(c["question"]) for c in inputs),
            "max": max(len(c["question"]) for c in inputs),
        },
    }
    evaluation_dir = source.parent.parent.parent / "evaluation"
    report["upstream_scorer_hashes"] = {
        name: file_digest(evaluation_dir / name)
        for name in ("evaluation_functions/jec_ac.py", "utils/function_utils.py")
        if (evaluation_dir / name).is_file()
    }
    write_jsonl(out / "cases.jsonl", inputs)
    write_jsonl(out / "targets.jsonl", targets)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=["case_id", "eligible", "reason", "question"]
    )
    writer.writeheader()
    writer.writerows(
        {
            "case_id": c["case_id"],
            "eligible": "",
            "reason": "",
            "question": c["question"],
        }
        for c in inputs
    )
    write_text(out / "review.csv", buffer.getvalue())
    write_json(out / "validation.json", report)
    return report


def split(
    data: Path, review: Path, count: int, dev_count: int, process_count: int, seed: int
) -> dict:
    if not (0 < dev_count < count and 0 < process_count <= count - dev_count):
        raise ValueError(
            "Require 0 < dev_count < count and process_count <= test count"
        )
    if (data / "splits.json").exists():
        raise ValueError("Splits already exist; use a new dataset version")
    cases = {
        row["case_id"]: CaseInput.model_validate(row)
        for row in read_jsonl(data / "cases.jsonl")
    }
    eligible, reviewed = [], set()
    with review.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = row["case_id"]
            if cid not in cases or cid in reviewed:
                raise ValueError(f"Unknown/duplicate review ID: {cid}")
            reviewed.add(cid)
            status = row["eligible"].strip().lower()
            if status not in {"yes", "no"}:
                raise ValueError(f"Review {cid}: eligible must be yes or no")
            if status == "no" and not row["reason"].strip():
                raise ValueError(f"Excluded case {cid} needs a reason")
            if status == "yes":
                eligible.append(cid)
    if reviewed != set(cases):
        raise ValueError("Every valid case must be reviewed before sampling")
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible cases, need {count}")
    rng = random.Random(seed)
    selected = rng.sample(sorted(eligible), count)
    dev, test = selected[:dev_count], selected[dev_count:]
    process = rng.sample(test, process_count)
    canonical_review = data / "review.csv"
    if canonical_review.resolve() != review.resolve():
        write_text(canonical_review, review.read_bytes().decode("utf-8"))
    result = {
        "seed": seed,
        "dev": dev,
        "test": test,
        "process": process,
        "review_sha256": file_digest(canonical_review),
    }
    write_json(data / "splits.json", result)
    write_jsonl(
        data / "annotations" / "issues.jsonl",
        [
            {
                "case_id": cid,
                "required_issues": [],
                "review_notes": "",
                "reviewed": False,
            }
            for cid in process
        ],
    )
    return result


def load_cases(data: Path) -> dict[str, CaseInput]:
    rows = read_jsonl(data / "cases.jsonl")
    cases = {row["case_id"]: CaseInput.model_validate(row) for row in rows}
    if len(cases) != len(rows):
        raise ValueError("Duplicate case IDs")
    return cases


def validate_annotations(data: Path) -> list[dict]:
    splits = read_json(data / "splits.json")
    rows = read_jsonl(data / "annotations" / "issues.jsonl")
    if len(rows) != len(splits["process"]) or {row["case_id"] for row in rows} != set(
        splits["process"]
    ):
        raise ValueError("Annotations must cover exactly the frozen process subset")
    for row in rows:
        issues = row.get("required_issues")
        if (
            row.get("reviewed") is not True
            or not isinstance(issues, list)
            or not issues
        ):
            raise ValueError(
                f"Complete and review required issues for {row['case_id']}"
            )
        seen = set()
        for issue in issues:
            if any(
                not isinstance(issue.get(key), str) or not issue[key].strip()
                for key in ("issue_id", "description", "coverage_criterion")
            ):
                raise ValueError(
                    "Each issue needs issue_id, description, coverage_criterion"
                )
            if issue["issue_id"] in seen:
                raise ValueError("Duplicate annotated issue ID")
            seen.add(issue["issue_id"])
    return rows


def dataset_fingerprint(data: Path, *, annotations: bool = True) -> dict:
    cases = load_cases(data)
    targets = read_jsonl(data / "targets.jsonl")
    if len(targets) != len(cases) or {row["case_id"] for row in targets} != set(cases):
        raise ValueError("References must cover exactly the model input cases")
    if any(row.get("gold_answer") not in ("A", "B", "C", "D") for row in targets):
        raise ValueError("Every reference must be a single A-D option")
    names = ["cases.jsonl", "targets.jsonl", "validation.json", "splits.json"]
    splits = read_json(data / "splits.json")
    review = data / "review.csv"
    if file_digest(review) != splits["review_sha256"]:
        raise ValueError("Eligibility review changed after splitting")
    names.append("review.csv")
    if annotations:
        validate_annotations(data)
        names.append("annotations/issues.jsonl")
    hashes = {name: file_digest(data / name) for name in names}
    return {"files": hashes, "sha256": digest(hashes)}

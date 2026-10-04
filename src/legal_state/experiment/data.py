"""Benchmark adaptation, human eligibility review and frozen sample splits."""

import csv
import io
import random
import re
import subprocess
from pathlib import Path
from typing import Literal

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


MSLR_QUESTION = (
    "请分析该内幕交易案件，依次说明内幕信息形成、信息知悉、交易行为、"
    "违法所得、法律适用与判决类型、处罚结果，并给出完整裁判结论。"
)

MSLR_STRICT_FIELDS = (
    "内幕交易信息的认定.内幕信息形成时间",
    "内幕交易信息的认定.内幕交易的股票名称",
    "内幕交易信息的认定.内幕信息公开时间",
    "当事人信息.当事人基础信息.姓名",
    "当事人信息.当事人基础信息.性别",
    "当事人信息.当事人基础信息.出生年份",
    "当事人信息.当事人基础信息.职务",
    "当事人信息.当事人的内幕交易认定.当事人知悉内幕交易时间",
    "当事人信息.当事人的内幕交易认定.买入/卖出",
    "当事人信息.当事人的内幕交易认定.买入时间",
    "当事人信息.当事人的内幕交易认定.买入金额（元）（最早买入时间均价）",
    "当事人信息.当事人的内幕交易认定.最早买入时间",
    "当事人信息.当事人的内幕交易认定.最晚买入时间",
    "当事人信息.当事人的内幕交易认定.基准日金额（元）",
    "当事人信息.当事人的内幕交易认定.违法所得（元）",
    "当事人信息.当事人处罚结果.没收违法所得金额（元）",
    "当事人信息.当事人处罚结果.罚款倍数",
    "当事人信息.当事人处罚结果.罚款数额（元）",
)

MSLR_SEMANTIC_FIELDS = (
    "内幕交易信息的认定.内幕信息内容",
    "内幕交易信息的认定.内幕交易信息认定条款",
    "内幕交易信息的认定.内幕信息所属类型",
    "内幕交易信息的认定.内幕交易形成时间发生事项",
    "内幕交易信息的认定.内幕信息公开时间发生事项",
    "当事人信息.当事人的内幕交易认定.当事人角色",
    "当事人信息.当事人的内幕交易认定.当事人所属类型",
    "当事人信息.当事人的内幕交易认定.当事人知悉内幕信息的方式（原文）",
    "当事人信息.当事人的内幕交易认定.知悉方式类型",
    "当事人信息.当事人的内幕交易认定.当事人内幕交易所属类型",
    "当事人信息.当事人处罚结果.处罚依据",
)


class CaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    benchmark: Literal["lawbench_3_6", "mslr"] = "lawbench_3_6"
    case_id: str = Field(pattern=r"^(lawbench-3-6|mslr)-[0-9]{6}$")
    source_index: int = Field(ge=0)
    question: str = Field(min_length=1)
    stem: str = Field(min_length=1)
    options: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_material(self):
        if self.benchmark == "mslr":
            if self.options:
                raise ValueError("MSLR cases do not have answer options")
            if self.source_index < 1 or self.case_id != f"mslr-{self.source_index:06d}":
                raise ValueError("MSLR case ID does not match source index")
            if self.question != MSLR_QUESTION:
                raise ValueError("MSLR question must use the frozen task instruction")
        else:
            if set(self.options) != set("ABCD") or any(
                not value.strip() for value in self.options.values()
            ):
                raise ValueError("Require nonempty A/B/C/D options")
            if self.case_id != f"lawbench-3-6-{self.source_index:06d}":
                raise ValueError("Case ID does not match source index")
            if (
                self.stem
                + "".join(f"{letter}:{self.options[letter]}" for letter in "ABCD")
                != self.question
            ):
                raise ValueError(
                    "Stem/options must preserve the original question exactly"
                )
        return self


def upstream_revision(source: Path) -> str | None:
    repository = source if source.is_dir() else source.parent
    while repository != repository.parent and not (repository / ".git").exists():
        repository = repository.parent
    if not (repository / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _extract_mslr_values(entry: dict) -> tuple[dict[str, str], dict[str, str]]:
    strict_values: dict[str, str] = {}
    semantic_values: dict[str, str] = {}
    ignored = {
        "序号",
        "案件信息",
        "当事人信息.当事人处罚结果申辩",
        "法律文书原文",
        "案件描述",
        "案件分析",
        "最终判决",
    }

    def walk(value, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}" if path else key
                if any(child.startswith(name) for name in ignored):
                    continue
                walk(item, child)
        elif isinstance(value, list):
            for item in value:
                walk(item, path)
        elif value is not None:
            normalized = str(value).strip()
            if normalized and normalized != "-":
                if path in MSLR_STRICT_FIELDS:
                    strict_values[path] = normalized
                elif path in MSLR_SEMANTIC_FIELDS:
                    semantic_values[path] = normalized

    walk(entry)
    return strict_values, semantic_values


def _prepare_mslr(source: Path, references: Path, out: Path, rows: list) -> dict:
    inputs, targets, errors = [], [], []
    seen: set[int] = set()
    reference_paths = sorted(references.glob("entry_*.json"))
    if len(reference_paths) != len(rows):
        raise ValueError("MSLR input and reference counts differ")
    for position, (row, reference_path) in enumerate(
        zip(rows, reference_paths, strict=True)
    ):
        try:
            if not isinstance(row, dict) or type(row.get("序号")) is not int:
                raise ValueError("Missing integer 序号")
            input_index = row["序号"]
            description = row.get("案件描述")
            if input_index != position + 1:
                raise ValueError("Input 序号 does not match its source position")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Missing positive 序号 or nonempty 案件描述")
            match = re.fullmatch(r"entry_([0-9]+)\.json", reference_path.name)
            if match is None:
                raise ValueError("Malformed MSLR reference filename")
            source_id = int(match[1])
            if source_id in seen:
                raise ValueError("Duplicate reference index")
            seen.add(source_id)
            reference = read_json(reference_path)
            if reference.get("案件描述") != description:
                raise ValueError("Input description differs from structured reference")
            strict_values, semantic_values = _extract_mslr_values(reference)
            if not strict_values and not semantic_values:
                raise ValueError("Reference has no scorable MSLR fields")
            case = CaseInput(
                benchmark="mslr",
                case_id=f"mslr-{source_id:06d}",
                source_index=source_id,
                question=MSLR_QUESTION,
                stem=description,
            )
            inputs.append(case.model_dump())
            targets.append(
                {
                    "case_id": case.case_id,
                    "source_index": source_id,
                    "input_index": input_index,
                    "strict_values": strict_values,
                    "semantic_values": semantic_values,
                    "reference_answer": "\n".join(
                        part.strip()
                        for part in (
                            reference.get("案件分析", ""),
                            reference.get("最终判决", ""),
                        )
                        if isinstance(part, str) and part.strip()
                    ),
                }
            )
        except (OSError, ValueError, TypeError) as error:
            errors.append({"source_position": position, "error": str(error)})
    if not inputs:
        raise ValueError("No valid MSLR cases")
    repository = source.parent.parent
    scorer = repository / "script/evaluate_IRACScore_task2.py"
    report = {
        "benchmark": "mslr",
        "source": str(source.resolve()),
        "references": str(references.resolve()),
        "source_sha256": file_digest(source),
        "upstream_commit": upstream_revision(source),
        "source_count": len(rows),
        "valid_count": len(inputs),
        "errors": errors,
        "question_chars": {
            "min": min(len(c["stem"]) for c in inputs),
            "max": max(len(c["stem"]) for c in inputs),
        },
        "upstream_scorer_hashes": {
            "script/evaluate_IRACScore_task2.py": file_digest(scorer)
        }
        if scorer.is_file()
        else {},
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
            "case_id": case["case_id"],
            "eligible": "yes",
            "reason": "",
            "question": case["stem"],
        }
        for case in inputs
    )
    write_text(out / "review.csv", buffer.getvalue())
    write_json(out / "validation.json", report)
    return report


def prepare(source: Path, out: Path, references: Path | None = None) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; use a new dataset version")
    rows = read_json(source)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source must be a nonempty JSON list")
    if isinstance(rows[0], dict) and {"序号", "案件描述"}.issubset(rows[0]):
        reference_dir = references or source.parent / "processed_anonymized"
        if not reference_dir.is_dir():
            raise ValueError("MSLR preparation requires the processed reference directory")
        return _prepare_mslr(source, reference_dir, out, rows)
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
        "benchmark": "lawbench_3_6",
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
    is_mslr = read_json(data / "validation.json").get("benchmark") == "mslr"
    mslr_issues = [
        ("I1", "内幕信息形成", "识别内幕信息内容、形成时间及公开时间"),
        ("I2", "信息知悉", "分析当事人身份、知悉时间与知悉方式"),
        ("I3", "交易行为", "分析交易方向、期间、账户和异常交易事实"),
        ("I4", "违法所得", "说明违法所得或相关金额；材料不足时明确说明"),
        ("I5", "法律适用与判决类型", "结合事实说明适用规则、行为定性及判决类型"),
        ("I6", "处罚结果", "给出没收、罚款或其他最终处理结论"),
    ]
    write_jsonl(
        data / "annotations" / "issues.jsonl",
        [
            {
                "case_id": cid,
                "required_issues": [
                    {
                        "issue_id": issue_id,
                        "description": description,
                        "coverage_criterion": criterion,
                    }
                    for issue_id, description, criterion in mslr_issues
                ]
                if is_mslr
                else [],
                "review_notes": (
                    "由 MSLR 官方六阶段推理链生成；仅用于输出侧过程评估。"
                    if is_mslr
                    else ""
                ),
                "reviewed": is_mslr,
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
    validation = read_json(data / "validation.json")
    if validation.get("benchmark", "lawbench_3_6") == "mslr":
        if any(
            not isinstance(row.get("strict_values"), dict)
            or not isinstance(row.get("semantic_values"), dict)
            or not row.get("strict_values") and not row.get("semantic_values")
            for row in targets
        ):
            raise ValueError("Every MSLR reference needs scorable structured fields")
    elif any(row.get("gold_answer") not in ("A", "B", "C", "D") for row in targets):
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

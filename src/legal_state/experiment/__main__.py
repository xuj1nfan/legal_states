"""Run with PYTHONPATH=src uv run python -m legal_state.experiment."""

import argparse
import json
import sys
from pathlib import Path

from legal_state.experiment.config import load_config
from legal_state.experiment.data import prepare, split
from legal_state.experiment.evaluation import score
from legal_state.experiment.io import read_json
from legal_state.experiment.report import report
from legal_state.experiment.review import export_review, import_review
from legal_state.experiment.storage import freeze, preflight, run_batch


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description="MSLR multi-step legal reasoning experiments")
    sub = top.add_subparsers(dest="command", required=True)
    command = sub.add_parser(
        "prepare", help="Validate source data and create eligibility review"
    )
    command.add_argument("--source", required=True, type=Path)
    command.add_argument(
        "--references",
        type=Path,
        help="MSLR processed_anonymized directory (inferred beside the source by default)",
    )
    command.add_argument("--out", required=True, type=Path)
    command = sub.add_parser(
        "split", help="Create immutable development/test/process subsets"
    )
    command.add_argument("--data", required=True, type=Path)
    command.add_argument("--review", required=True, type=Path)
    command.add_argument("--count", type=int, default=100)
    command.add_argument("--dev-count", type=int, default=15)
    command.add_argument("--process-count", type=int, default=30)
    command.add_argument("--seed", type=int, default=20260930)
    for name in (
        "preflight",
        "freeze",
        "run",
        "score",
        "human-review-export",
        "human-review-import",
        "report",
    ):
        command = sub.add_parser(name)
        command.add_argument(
            "--config", type=Path, default=Path("configs/mslr.yaml")
        )
        if name != "preflight":
            command.add_argument("--run-id", required=True)
        if name == "freeze":
            command.add_argument("--dev-run-id", default="mslr_dev_01")
        if name == "run":
            command.add_argument("--split", choices=("dev", "test"), required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument(
                "--dry-run",
                action="store_true",
                help="Show bounds without loading credentials or calling a model",
            )
        if name == "score":
            command.add_argument("--official-check", action="store_true")
            command.add_argument(
                "--embedding-model",
                type=Path,
                help="Local ChatLaw-Text2Vec path required by official MSLR IRAC scoring",
            )
        if name == "human-review-import":
            command.add_argument("--ratings", required=True, type=Path)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path.cwd()
    try:
        if args.command == "prepare":
            result = prepare(args.source, args.out, args.references)
        elif args.command == "split":
            result = split(
                args.data,
                args.review,
                args.count,
                args.dev_count,
                args.process_count,
                args.seed,
            )
        else:
            config = load_config(args.config)
            if args.command == "preflight":
                result = preflight(config, root)
            elif args.command == "freeze":
                result = freeze(config, root, args.run_id, args.dev_run_id)
            elif args.command == "run":
                if args.dry_run:
                    ids = read_json(root / config.data_dir / "splits.json")[args.split]
                    combinations = len(ids) * len(config.methods)
                    result = {
                        "cases": len(ids),
                        "combinations": combinations,
                        "max_calls": combinations * (config.reasoning.max_steps + 1),
                        "max_output_tokens": combinations
                        * (
                            config.reasoning.max_output_tokens_total
                            + config.final_answer.max_output_tokens
                        ),
                        "note": "Input tokens, fees and unknown usage are not bounded by this output estimate; no model called.",
                    }
                else:
                    result = run_batch(
                        config, root, args.run_id, args.split, args.resume
                    )
            elif args.command == "score":
                result = score(
                    config,
                    root,
                    args.run_id,
                    args.official_check,
                    args.embedding_model,
                )
            elif args.command == "human-review-export":
                result = export_review(config, root, args.run_id)
            elif args.command == "human-review-import":
                result = import_review(config, root, args.run_id, args.ratings)
            else:
                result = report(config, root, args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

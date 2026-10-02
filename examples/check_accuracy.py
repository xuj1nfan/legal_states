"""Check the strict Legal State > CoT target on a complete, sealed paired run."""

import argparse
import json
from pathlib import Path

from legal_state.experiment.config import load_config
from legal_state.experiment.evaluation import score
from legal_state.experiment.io import write_json
from legal_state.experiment.storage import run_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--official-check", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_config(args.config)
    result = score(config, root, args.run_id, args.official_check)
    summaries = {row["method"]: row for row in result["summary"]}
    paired = result["comparisons"]["legal_state_minus_cot"]
    acceptance = {
        "run_id": args.run_id,
        "split": result["split"],
        "cases": summaries["legal_state"]["count"],
        "criterion": "legal_state accuracy strictly greater than cot on all paired cases",
        "passed": paired["difference"] > 0,
        "legal_state_accuracy": summaries["legal_state"]["accuracy"],
        "cot_accuracy": summaries["cot"]["accuracy"],
        "paired_comparison": paired,
        "failures_and_unknown_in_denominator": True,
        "held_out_positive_ci": result["split"] == "test" and paired["ci95"][0] > 0,
        "note": "A development pass is not evidence of held-out or statistically established superiority.",
    }
    write_json(run_path(config, args.run_id, root) / "accuracy_acceptance.json", acceptance)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

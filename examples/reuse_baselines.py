"""Seed a development ablation with fully replay-verified, sealed baselines."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from legal_state.experiment.config import load_config
from legal_state.experiment.data import load_cases
from legal_state.experiment.engine import JournalClient, run_case
from legal_state.experiment.evaluation import load_run
from legal_state.experiment.io import file_digest, write_json
from legal_state.experiment.storage import (
    checked_preflight,
    run_path,
    signature,
    write_manifest,
)


class ReplayOnlyClient:
    def generate(self, *args, **kwargs):
        raise RuntimeError("Baseline replay attempted a new model call")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    config = load_config(args.config)
    destination = run_path(config, args.run_id, root)
    if destination.exists():
        raise ValueError("Destination exists; choose a new run ID")
    source, old, records = load_run(config, root, args.baseline_run_id)
    new_identity = signature(config, root, "dev")
    if old["identity"]["split"] != "dev":
        raise ValueError("Baseline reuse is limited to development experiments")
    for field in ("model", "case_ids", "dataset"):
        if old["identity"][field] != new_identity[field]:
            raise ValueError(f"Baseline {field} differs")
    for field in ("model", "reasoning", "final_answer", "execution", "knowledge_mode"):
        if old["identity"]["config"][field] != config.model_dump()[field]:
            raise ValueError(f"Baseline settings differ: {field}")
    ready = checked_preflight(config, root)
    fingerprints = [
        json.loads(report["raw_response"]).get("system_fingerprint")
        for report in (old["preflight"], ready)
    ]
    if not fingerprints[0] or fingerprints[0] != fingerprints[1]:
        raise ValueError("Reported service fingerprints differ or are unavailable")
    cases = load_cases(root / config.data_dir)
    verified = []
    with tempfile.TemporaryDirectory(prefix="legal-baseline-replay-") as temporary:
        for original in records:
            cid, method = original["case_id"], original["method"]
            if method not in {"cot", "generic"}:
                continue
            journal = source / "journals" / cid / f"{method}.jsonl"
            copied = Path(temporary) / cid / f"{method}.jsonl"
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(journal, copied)
            client = JournalClient(ReplayOnlyClient(), config, copied, old["returned_model"])
            replay = run_case(cases[cid], method, config, client)
            if client.cursor != len(client.requests) or client.usage()["call_count"] != original["usage"]["call_count"]:
                raise ValueError("Baseline replay did not consume exactly the original calls")
            if file_digest(copied) != file_digest(journal):
                raise ValueError("Baseline replay appended a new request")
            for field in ("status", "final_answer", "termination_reason", "reasoning_artifact", "usage"):
                if replay[field] != original[field]:
                    raise ValueError(f"Baseline replay changed {cid}/{method}: {field}")
            verified.append((original, journal))
    manifest = write_manifest(config, root, args.run_id, "dev")
    manifest["baseline_reuse"] = {
        "run_id": args.baseline_run_id,
        "records_manifest_sha256": file_digest(source / "records_manifest.json"),
        "source_sha256": old["identity"]["source_sha256"],
        "reported_service_fingerprint": fingerprints[0],
        "all_prompts_and_budgets_replay_verified": True,
        "count": len(verified),
    }
    write_json(destination / "manifest.json", manifest)
    for original, journal in verified:
        cid, method = original["case_id"], original["method"]
        copied = destination / "journals" / cid / f"{method}.jsonl"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(journal, copied)
        record = {**original, "run_id": args.run_id, "reused_from": args.baseline_run_id}
        write_json(destination / "records" / cid / f"{method}.json", record)
    print(json.dumps(manifest["baseline_reuse"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

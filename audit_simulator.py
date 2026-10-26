"""Independent repeatable stress accounting audit; does not train/select models.

Exactly reproduces the 40-seed, 3-policy, 168-hour checks originally performed
read-only during review. Writes a source-hashed receipt without changing the
simulator, its corpus, fine-tuning configuration, or any model checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time

import numpy as np

import simulator


ROOT = Path(__file__).resolve().parent
PARAMETER_SEED = 61723
PATH_SEEDS = tuple(range(40))
POLICIES = ("flat", "reactive", "anticipatory")
LENGTH = 168
SCALES = {"median_fare_usd": 30., "median_positive_completed_count": 25.}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def run_audit(output):
    started = time.perf_counter()
    simulator_path = Path(simulator.__file__).resolve()
    initial_hash = sha256(simulator_path)
    receipt = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": sha256(__file__),
        "simulator_source": str(simulator_path),
        "simulator_sha256": initial_hash,
        "runtime": {name: metadata.version(name) for name in ("numpy", "pandas")},
        "parameter_seed": PARAMETER_SEED,
        "path_seeds": list(PATH_SEEDS),
        "policies": list(POLICIES),
        "hours_per_trajectory": LENGTH,
        "illustrative_scale_inputs": SCALES,
        "training_data_used": False,
        "real_world_parameter_validation": False,
        "checks": [
            "Every hour: starting queue + requests = completions + abandonments + ending queue",
            "Whole trajectory: requests = completions + abandonments + final pending queue",
            "Whole trajectory: 0 <= completions <= potential arrivals",
            "Every hour: completions <= service budget",
            "Every hour: posted rider fare >= reference anchor (tolerance 1e-10)",
            "Every hour: posted rider fare <= reference anchor * cap (tolerance 1e-10)",
        ],
        "passed_trajectories": 0,
        "passed_hourly_transitions": 0,
    }
    try:
        rng = np.random.default_rng(PARAMETER_SEED)
        for seed in PATH_SEEDS:
            parameters = simulator.sample_parameters(rng, SCALES)
            exogenous = simulator.make_exogenous(parameters, seed, length=LENGTH)
            for policy in POLICIES:
                path = simulator.simulate_path(exogenous, parameters, policy)
                label = f"seed={seed}, policy={policy}"
                check(np.all(path["queue_start"] + path["requested"] ==
                             path["count"] + path["abandoned"] + path["queue_end"]),
                      f"Hourly conservation failed: {label}")
                check(path["requested"].sum() == path["count"].sum() +
                      path["abandoned"].sum() + path["queue_end"][-1],
                      f"Global conservation failed: {label}")
                check(0 <= path["count"].sum() <= path["potential"].sum(),
                      f"All-arrival conversion bounds failed: {label}")
                check(np.all(path["count"] <= path["service_budget"]),
                      f"Capacity constraint failed: {label}")
                check(np.all(path["posted_fare"] >= parameters["base_anchor"] - 1e-10),
                      f"Lower rider-fare bound failed: {label}")
                check(np.all(path["posted_fare"] <= parameters["base_anchor"] * parameters["cap"] + 1e-10),
                      f"Upper rider-fare bound failed: {label}")
                receipt["passed_trajectories"] += 1
                receipt["passed_hourly_transitions"] += LENGTH
        check(sha256(simulator_path) == initial_hash, "Simulator source changed during audit")
        receipt["status"] = "passed"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["elapsed_seconds"] = time.perf_counter() - started
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/synthetic/independent_audit.json")
    args = parser.parse_args()
    result = run_audit(args.output)
    print(json.dumps({key: result[key] for key in (
        "status", "passed_trajectories", "passed_hourly_transitions", "simulator_sha256")}, indent=2))

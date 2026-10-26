"""Independent saved-artifact checks; never fit or choose a model here."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEYS = ["origin", "series_id", "horizon", "target_timestamp"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = ROOT / "outputs"
    manifest = json.loads((out / "run_manifest.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    data = json.loads((ROOT / "data/metadata.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    check("protocol_hash_matches", sha(ROOT / "protocol.json") == manifest["protocol_sha256"])
    check("panel_hash_matches_run_and_data_receipt", sha(ROOT / "data/panel.parquet") == manifest["panel_sha256"] == data["panel_sha256"])
    for source in ("benchmark.py", "baselines.py", "prepare_data.py"):
        check(f"executed_source_unchanged_{source}", sha(ROOT / source) == manifest["source_hashes"][source])
    panel = pd.read_parquet(ROOT / "data/panel.parquet")
    check("no_duplicate_panel_hours", not panel.duplicated(["series_id", "timestamp"]).any())
    check("low_count_targets_not_fabricated", panel.loc[panel["count"] < 10, "fare"].isna().all())
    for month, audit in data["monthly_disjoint_exclusion_audits"].items():
        check(f"exclusions_reconcile_{month}", audit["rows_read"] == audit["rows_retained"] + sum(v for k, v in audit.items() if k.startswith("exclude_")))
    frames = {}
    for split in ("validation", "test"):
        frame = pd.read_parquet(out / f"{split}_predictions.parquet")
        frames[split] = frame
        check(f"{split}_no_duplicate_forecasts", not frame.duplicated(KEYS + ["model"]).any())
        check(f"{split}_finite_predictions_and_truth", np.isfinite(frame[["actual", "prediction", "q10", "q90"]]).all().all())
        check(f"{split}_same_support_all_models", frame.groupby(KEYS).model.nunique().eq(frame.model.nunique()).all())
        check(f"{split}_same_truth_all_models", frame.groupby(KEYS).actual.nunique().eq(1).all())
        check(f"{split}_horizons_correct", (frame.target_timestamp == frame.origin + pd.to_timedelta(frame.horizon - 1, unit="h")).all())
        check(f"{split}_noncrossing_interval_endpoints", (frame.q10 <= frame.q90).all())
        check(f"{split}_observed_target_counts", frame.target_count.ge(10).all())
        for row in metrics[split]:
            subset = frame[frame.model.eq(row["model"])]
            if row["horizon"] != "all":
                subset = subset[subset.horizon.eq(int(row["horizon"]))]
            check(f'{split}_{row["model"]}_{row["horizon"]}_mae_recomputed', np.isclose((subset.actual - subset.prediction).abs().mean(), row["mae"], rtol=0, atol=1e-10))
    check("validation_outcomes_available_before_test", (frames["validation"].target_timestamp < pd.Timestamp(protocol["validation_end"]) - pd.Timedelta(hours=3)).all())
    check("test_starts_after_validation", (frames["test"].origin >= pd.Timestamp(protocol["validation_end"])).all())
    check("test_within_prespecified_window", (frames["test"].target_timestamp < pd.Timestamp(protocol["test_end"])).all())
    val_mae = frames["validation"].assign(e=lambda x: (x.actual-x.prediction).abs()).groupby("model").e.mean()
    comparison = metrics["comparison"]
    check("reference_baseline_selected_on_validation", comparison["baseline"] == val_mae[~val_mae.index.str.startswith("Chronos")].idxmin())
    check("chronos_variant_selected_on_validation", comparison["chronos"] == val_mae[val_mae.index.str.startswith("Chronos")].idxmin())
    if "synthetic_adaptation" in manifest:
        adaptation = manifest["synthetic_adaptation"]
        check("synthetic_checkpoint_selected_on_validation", adaptation["selected"]["steps"] == min(adaptation["candidates"], key=lambda c: (c["validation_mae"], c["steps"]))["steps"])
        check("zero_step_retained_as_adaptation_option", adaptation["validation_recommendation"]["steps"] == min(
            [adaptation["zero_step"], adaptation["selected"]], key=lambda c: (c["validation_mae"], c["steps"]))["steps"])
        training = json.loads((out / "synthetic/fine_tuning_manifest.json").read_text())
        check("synthetic_training_completed", training["status"] == "completed" and training["optimizer_steps_completed"] == 60)
        check("only_adapters_trained", training["original_base_weights_unchanged"] and training["zero_adapter_matches_original"])
        check("all_merged_checkpoints_verified", all(x["reload_finite"] and x["merged_matches_adapter"] for x in training["checkpoint_exports"]))
    receipt = {"status": "passed", "check_count": len(checks), "checks": checks,
        "test_models": sorted(frames["test"].model.unique().tolist()),
        "test_cases_per_model": frames["test"].groupby("model").size().to_dict(),
        "not_certified": ["Causal validity for actual Uber pricing", "Structural identification from completed-trip data", "Generalization outside selected routes/month", "Optimality of any simulated policy"]}
    (out / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Independent saved-artifact audit: {len(checks)} checks passed.")


if __name__ == "__main__":
    main()

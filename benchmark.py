"""Frozen local rider-fare benchmark. No live API, private Uber data, or uploads."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEYS = ["origin", "series_id", "horizon", "target_timestamp"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def calendar_features(times):
    local = pd.DatetimeIndex(times).tz_localize("UTC").tz_convert("America/New_York")
    return {
        "hour_sin": np.sin(2 * np.pi * local.hour / 24).astype(np.float32),
        "hour_cos": np.cos(2 * np.pi * local.hour / 24).astype(np.float32),
        "dow_sin": np.sin(2 * np.pi * local.dayofweek / 7).astype(np.float32),
        "dow_cos": np.cos(2 * np.pi * local.dayofweek / 7).astype(np.float32),
        "weekend": (local.dayofweek >= 5).astype(np.float32),
    }


def make_chronos_input(series, origin, protocol, covariates):
    cutoff = pd.Timestamp(origin) - pd.Timedelta(hours=protocol["embargo_hours"])
    # Last available pickup bin ends at cutoff; with <=3h trips all have ended by origin.
    times = pd.date_range(end=cutoff - pd.Timedelta(hours=1), periods=protocol["context_hours"], freq="h")
    history = series.reindex(times)
    target = history.fare.to_numpy(dtype=np.float32)[None, :]
    if np.isfinite(target).sum() < 24:
        raise ValueError("Insufficient context, do not silently impute a forecast")
    item = {"target": target}
    if covariates:
        future_times = pd.date_range(cutoff, periods=protocol["embargo_hours"] + max(protocol["horizons"]), freq="h")
        item["past_covariates"] = calendar_features(times)
        item["past_covariates"]["served_count"] = history["count"].to_numpy(dtype=np.float32)
        item["future_covariates"] = calendar_features(future_times)
    return item


def run_chronos(panel, origins, protocol, pipeline, mode, batch_size=32):
    import torch
    groups = {str(k): g.set_index("timestamp").sort_index() for k, g in panel.groupby("series_id")}
    cases, inputs = [], []
    for origin in origins:
        for sid, series in groups.items():
            cases.append((pd.Timestamp(origin), sid))
            inputs.append(make_chronos_input(series, origin, protocol, mode != "Chronos-2"))
    rows, elapsed = [], 0.0
    length = protocol["embargo_hours"] + max(protocol["horizons"])
    for start in range(0, len(inputs), batch_size):
        now = time.perf_counter()
        with torch.inference_mode():
            q, _ = pipeline.predict_quantiles(
                inputs[start:start + batch_size], prediction_length=length,
                quantile_levels=[.1, .5, .9], context_length=protocol["context_hours"],
                batch_size=batch_size, cross_learning=False,
            )
        elapsed += time.perf_counter() - now
        for (origin, sid), arr in zip(cases[start:start + batch_size], q):
            values = arr.detach().cpu().numpy()[0]
            if values.shape != (length, 3) or not np.isfinite(values).all():
                raise ValueError(f"Invalid quantile output {values.shape}")
            if np.any(np.diff(values, axis=1) < -1e-5):
                raise ValueError("Crossing model quantiles")
            for horizon in protocol["horizons"]:
                p = np.maximum(values[protocol["embargo_hours"] + horizon - 1], 0)
                rows.append({"origin": origin, "series_id": sid, "horizon": horizon,
                    "target_timestamp": origin + pd.Timedelta(hours=horizon - 1),
                    "model": mode, "q10": float(p[0]), "prediction": float(p[1]),
                    "q90": float(p[2]), "interval_method": "native model quantiles"})
        if start % (batch_size * 6) == 0:
            print(f"{mode}: {min(start + batch_size, len(inputs))}/{len(inputs)} cases", flush=True)
    return pd.DataFrame(rows), {"inference_seconds": elapsed, "route_origins": len(cases),
        "amortized_ms_per_route_origin": 1000 * elapsed / len(cases), "batch_size": batch_size,
        "note": "Batched CPU timing, not production end-to-end latency"}


def attach_actuals(pred, panel, thresholds):
    pred = pred.drop(columns=[c for c in ["actual", "y_true", "target_count"] if c in pred]).copy()
    pred["series_id"] = pred.series_id.astype(str)
    actual = panel[["series_id", "timestamp", "fare", "count"]].rename(columns={
        "timestamp": "target_timestamp", "fare": "actual", "count": "target_count"})
    pred = pred.merge(actual, on=["series_id", "target_timestamp"], how="left", validate="many_to_one")
    pred["high_fare_threshold"] = pred.series_id.map(thresholds)
    pred["high_fare"] = pred.actual > pred.high_fare_threshold
    return pred


def common_support(pred):
    if pred.duplicated(KEYS + ["model"]).any():
        raise ValueError("Duplicate forecast keys")
    n_models = pred.model.nunique()
    counts = pred.assign(valid=np.isfinite(pred.actual) & np.isfinite(pred.prediction)).groupby(KEYS).agg(
        models=("model", "nunique"), valid=("valid", "sum"))
    keep = counts[(counts.models == n_models) & (counts.valid == n_models)].reset_index()[KEYS]
    scored = pred.merge(keep, on=KEYS, validate="many_to_one")
    if not len(scored):
        raise ValueError("No common scoring support")
    return scored, {"candidate_route_origin_horizons": len(counts), "scored_route_origin_horizons": len(keep),
        "excluded_route_origin_horizons": len(counts) - len(keep), "model_count": int(n_models)}


def calibrate_baseline_intervals(val, test):
    val, test = val.copy(), test.copy()
    for (model, horizon), sub in val.groupby(["model", "horizon"]):
        if model.startswith("Chronos"):
            continue
        residual = (sub.actual - sub.prediction).dropna()
        lower, upper = np.quantile(residual, [.1, .9])
        for frame in (val, test):
            mask = frame.model.eq(model) & frame.horizon.eq(horizon)
            frame.loc[mask, "q10"] = np.maximum(0, frame.loc[mask, "prediction"] + lower)
            frame.loc[mask, "q90"] = np.maximum(0, frame.loc[mask, "prediction"] + upper)
            frame.loc[mask, "interval_method"] = "validation residual quantiles (not guaranteed coverage)"
    return val, test


def summarize(pred):
    out = []
    for (model, horizon), frame in pd.concat([pred.assign(scope="all"), pred.assign(scope=pred.horizon.astype(str))]).groupby(["model", "scope"]):
        e = frame.prediction - frame.actual
        high = frame.loc[frame.high_fare]
        err = frame.actual.to_numpy()[:, None] - frame[["q10", "prediction", "q90"]].to_numpy()
        q = np.array([.1, .5, .9])
        out.append({"model": model, "horizon": horizon, "n": len(frame), "mae": float(e.abs().mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "high_fare_mae": float((high.prediction - high.actual).abs().mean()) if len(high) else None,
            "high_fare_n": len(high), "coverage_80": float(((frame.actual >= frame.q10) & (frame.actual <= frame.q90)).mean()),
            "width_80": float((frame.q90 - frame.q10).mean()),
            "mean_pinball": float(np.maximum(q * err, (q - 1) * err).mean()),
            "interval_method": frame.interval_method.iloc[0]})
    return out


def paired_comparison(val, test):
    val_mae = val.assign(error=(val.prediction - val.actual).abs()).groupby("model").error.mean()
    baseline = val_mae[~val_mae.index.str.startswith("Chronos")].idxmin()
    chronos = val_mae[val_mae.index.str.startswith("Chronos")].idxmin()
    pivot = test[test.model.isin([baseline, chronos])].assign(error=lambda x: (x.prediction - x.actual).abs()).pivot(index=KEYS, columns="model", values="error").reset_index()
    pivot["day"] = pivot.origin.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.date
    daily = pivot.groupby("day")[[baseline, chronos]].agg(["sum", "count"])
    rng = np.random.default_rng(20261026)
    ix = rng.integers(0, len(daily), (2000, len(daily)))
    b = daily[(baseline, "sum")].to_numpy()[ix].sum(1) / daily[(baseline, "count")].to_numpy()[ix].sum(1)
    c = daily[(chronos, "sum")].to_numpy()[ix].sum(1) / daily[(chronos, "count")].to_numpy()[ix].sum(1)
    rel = 100 * (b - c) / b
    baseline_mae = float(pivot[baseline].mean())
    chronos_mae = float(pivot[chronos].mean())
    return {"selection": "Lowest validation MAE within each family, no test selection", "baseline": str(baseline),
        "chronos": str(chronos), "baseline_test_mae": baseline_mae, "chronos_test_mae": chronos_mae,
        "chronos_relative_mae_improvement_pct": 100 * (baseline_mae - chronos_mae) / baseline_mae,
        "paired_daily_block_bootstrap_95pct": np.quantile(rel, [.025, .975]).tolist(),
        "days": len(daily), "draws": 2000,
        "warning": "Exploratory one-city one-month inference. Days may remain serially dependent."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=ROOT / "data/panel.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic-checkpoints", type=Path,
                        help="Optional directory of merged step30/step60 models; validation selects one before test")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    p = json.loads((ROOT / "protocol.json").read_text())
    panel = pd.read_parquet(args.panel).sort_values(["series_id", "timestamp"])
    panel["series_id"] = panel.series_id.astype(str)
    if panel.duplicated(["series_id", "timestamp"]).any():
        raise ValueError("Duplicate panel hours")
    for _, g in panel.groupby("series_id"):
        if not g.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Panel must have continuous UTC hourly grid")
    threshold_cutoff = pd.Timestamp(p["train_end"]) - pd.Timedelta(hours=p["embargo_hours"])
    thresholds = panel[panel.timestamp < threshold_cutoff].groupby("series_id").fare.quantile(.9).to_dict()
    val_origins = pd.date_range(p["train_end"], p["validation_end"], freq=f'{p["origin_stride_hours"]}h', inclusive="left")
    test_origins = pd.date_range(p["validation_end"], p["test_end"], freq=f'{p["origin_stride_hours"]}h', inclusive="left")
    test_origins = test_origins[test_origins + pd.Timedelta(hours=max(p["horizons"])) <= pd.Timestamp(p["test_end"])]
    manifest = {"protocol": p, "protocol_sha256": sha256(ROOT / "protocol.json"), "panel_sha256": sha256(args.panel),
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "hardware": platform.platform(),
        "runtime": {k: metadata.version(k) for k in ["chronos-forecasting", "torch", "numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib"]},
        "source_hashes": {f.name: sha256(f) for f in ROOT.glob("*.py")}, "timings": {}, "train_high_fare_thresholds": thresholds}
    # Preserve the locked design before model fit/test outcomes are scored.
    save_json(args.output / "run_manifest.json", manifest)
    from baselines import BaselineSuite
    suite = BaselineSuite(panel, horizons=tuple(p["horizons"]), embargo_hours=p["embargo_hours"], context_hours=p["context_hours"])
    started = time.perf_counter()
    val_base, base_metadata = suite.fit(train_end=p["train_end"], val_start=p["train_end"], val_end=p["validation_end"], origin_stride=6, min_training_history_hours=168, refit=True)
    manifest["timings"]["baseline_fit_and_validation_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    test_base = suite.predict(test_origins)
    manifest["timings"]["baseline_test_prediction_seconds"] = time.perf_counter() - started
    manifest["timings"]["baseline_test_by_model_seconds"] = suite.last_predict_timings_
    route_origins = len(test_origins) * panel.series_id.nunique()
    manifest["model_latency_seconds"] = {name: {"seconds_per_case": elapsed / route_origins,
        "unit": "one route-origin producing all three scored horizons; batched CPU"}
        for name, elapsed in suite.last_predict_timings_.items()}
    manifest["baselines"] = base_metadata
    # Filter validation onto exact same fixed origins for all models.
    val_base = val_base[val_base.origin.isin(val_origins)]
    val_parts, test_parts = [val_base], [test_base]
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4)
    snapshot = Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots" / p["chronos_revision"]
    weight_hash = sha256(snapshot / "model.safetensors")
    if weight_hash != p["chronos_weights_sha256"]:
        raise ValueError("Pinned model weight hash mismatch")
    manifest["model_hash"] = weight_hash
    started = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(snapshot, local_files_only=True, device_map="cpu", dtype=torch.float32)
    manifest["timings"]["chronos_load_seconds"] = time.perf_counter() - started
    for mode in p["chronos_modes"]:
        for split, origins, parts in [("validation", val_origins, val_parts), ("test", test_origins, test_parts)]:
            pred, timing = run_chronos(panel, origins, p, pipeline, mode, args.batch_size)
            parts.append(pred)
            manifest["timings"][f"{mode}/{split}"] = timing
            if split == "test":
                manifest["model_latency_seconds"][mode] = {
                    "seconds_per_case": timing["inference_seconds"] / timing["route_origins"],
                    "unit": "one route-origin producing all three scored horizons; batched CPU"}
    if args.synthetic_checkpoints:
        # Predeclared small synthetic adaptation pilot. No test labels select a checkpoint.
        candidates = []
        validation_candidates = []
        val_cutoff = pd.Timestamp(p["validation_end"]) - pd.Timedelta(hours=p["embargo_hours"])
        for steps in (30, 60):
            checkpoint = args.synthetic_checkpoints / f"step{steps}"
            if not (checkpoint / "config.json").exists():
                raise FileNotFoundError(f"Required predeclared checkpoint missing: {checkpoint}")
            model_name = f"Chronos-2 + synthetic LoRA ({steps} steps)"
            adapted = Chronos2Pipeline.from_pretrained(checkpoint, local_files_only=True,
                                                       device_map="cpu", dtype=torch.float32)
            pred, timing = run_chronos(panel, val_origins, p, adapted, model_name, args.batch_size)
            evaluated = attach_actuals(pred, panel, thresholds)
            evaluated = evaluated[evaluated.target_timestamp < val_cutoff].dropna(subset=["actual"])
            mae = float((evaluated.prediction - evaluated.actual).abs().mean())
            candidates.append({"steps": steps, "model": model_name, "path": str(checkpoint),
                "validation_mae": mae, "weight_files": {f.name: sha256(f) for f in checkpoint.glob("*.safetensors")}})
            validation_candidates.append(evaluated)
            manifest["timings"][f"{model_name}/validation"] = timing
            print(f"Synthetic checkpoint {steps}: validation MAE ${mae:.4f}", flush=True)
        selected = min(candidates, key=lambda item: (item["validation_mae"], item["steps"]))
        original_cov = next(frame for frame in val_parts if len(frame) and
                            frame.model.iloc[0] == "Chronos-2 + calendar/count")
        original_eval = attach_actuals(original_cov, panel, thresholds)
        original_eval = original_eval[original_eval.target_timestamp < val_cutoff].dropna(subset=["actual"])
        zero_step = {"steps": 0, "model": "Chronos-2 + calendar/count", "path": str(snapshot),
            "validation_mae": float((original_eval.prediction - original_eval.actual).abs().mean())}
        recommendation = min([zero_step, selected], key=lambda item: (item["validation_mae"], item["steps"]))
        manifest["synthetic_adaptation"] = {"candidates": candidates, "selected": selected,
            "selected_is_diagnostic_synthetic_candidate_only": True,
            "zero_step": zero_step, "validation_recommendation": recommendation,
            "selection": "Best synthetic candidate selected on February MAE; zero-step is retained if better. Best synthetic is still evaluated and reported diagnostically to expose negative transfer. No test selection and no real validation/test gradient updates."}
        pd.concat(validation_candidates, ignore_index=True).to_parquet(args.output / "synthetic_checkpoint_validation.parquet", index=False)
        val_parts.append(next(frame for frame in validation_candidates if frame.model.iloc[0] == selected["model"]).drop(
            columns=["actual", "target_count", "high_fare_threshold", "high_fare"]))
        adapted = Chronos2Pipeline.from_pretrained(selected["path"], local_files_only=True,
                                                   device_map="cpu", dtype=torch.float32)
        pred, timing = run_chronos(panel, test_origins, p, adapted, selected["model"], args.batch_size)
        test_parts.append(pred)
        manifest["timings"][f'{selected["model"]}/test'] = timing
        manifest["model_latency_seconds"][selected["model"]] = {
            "seconds_per_case": timing["inference_seconds"] / timing["route_origins"],
            "unit": "one route-origin producing all three scored horizons; batched CPU"}
    val = attach_actuals(pd.concat(val_parts, ignore_index=True), panel, thresholds)
    val = val[val.target_timestamp < pd.Timestamp(p["validation_end"]) - pd.Timedelta(hours=p["embargo_hours"])]
    test = attach_actuals(pd.concat(test_parts, ignore_index=True), panel, thresholds)
    # All scored models receive exactly the same observed target rows.
    val, val_support = common_support(val)
    test, test_support = common_support(test)
    val, test = calibrate_baseline_intervals(val, test)
    val.to_parquet(args.output / "validation_predictions.parquet", index=False)
    test.to_parquet(args.output / "test_predictions.parquet", index=False)
    manifest["support"] = {"validation": val_support, "test": test_support}
    manifest["baseline_point_outside_calibrated_interval_rows"] = int((
        ~test.model.str.startswith("Chronos") & ((test.prediction < test.q10) | (test.prediction > test.q90))).sum())
    manifest["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    save_json(args.output / "run_manifest.json", manifest)
    metrics = {"validation": summarize(val), "test": summarize(test), "comparison": paired_comparison(val, test)}
    save_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Bounded zero-shot pressure forecasting plus an illustrative rider-price map.

This is a separate task from the NYC fare/LoRA experiment. It forecasts next-hour
potential arrivals and capacity on unseen synthetic worlds, not prices or causal
outcomes. All conditional responses come from the declared decision_layer model.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from decision_layer import ResponseConfig, choose_action, response_surface, solve_response

ROOT = Path(__file__).resolve().parent
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c"
TARGETS = ("potential", "capacity")
CALENDAR = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")
ORIGINS = (512, 640, 768, 896)
LAGS = (1, 2, 3, 6, 24, 48, 168)
PRICES = np.linspace(1., 2., 11)
SEED = 20261026


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def features(data, series, t):
    """Only completed history [:t] plus known calendar for target bin t."""
    row = []
    for target in TARGETS:
        values = data[target][series]
        row.extend(float(values[t-lag]) for lag in LAGS)
        row.extend(float(values[t-window:t].mean()) for window in (6, 24, 168))
        row.append(float(values[t-24:t].std()))
    row.extend(float(data[key][series, t]) for key in CALENDAR)
    return row


def chronos_inputs(data, cases):
    return [{"target": np.stack([data[k][s, t-256:t] for k in TARGETS]),
             "past_covariates": {k: data[k][s, t-256:t] for k in CALENDAR},
             "future_covariates": {k: data[k][s, t:t+1] for k in CALENDAR}}
            for s, t in cases]


def benchmark(output_dir):
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = (ROOT / "outputs/synthetic" / f for f in ("corpus.npz", "validation.npz"))
    # NPZ member access decompresses anew; cache the small corpus once rather
    # than repeating decompression for each lag feature.
    def load_arrays(path):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    train, heldout = (load_arrays(p) for p in (train_path, test_path))
    if train["potential"].shape[0] < 32 or heldout["potential"].shape[0] < 16:
        raise ValueError("Frozen selection requires 32 training and 16 heldout series")
    cases = [(s, t) for s in range(16) for t in ORIGINS]
    training_cases = [(s, t) for s in range(32) for t in range(256, train["potential"].shape[1], 2)]
    config = ResponseConfig()
    provenance = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "protocol_frozen_before_forecast_scoring",
        "targets": list(TARGETS), "horizon_hours": 1, "context_hours": 256,
        "heldout_series_indices": list(range(16)), "target_bin_indices": list(ORIGINS),
        "cases": len(cases), "example_case": {"series_index": 0, "target_bin_index": 512},
        "training_series_indices": list(range(32)), "training_origin_stride": 2,
        "training_origins_start": 256, "training_cases": len(training_cases),
        "training_corpus_sha256": sha256(train_path), "heldout_corpus_sha256": sha256(test_path),
        "pressure_benchmark_sha256": sha256(__file__), "decision_layer_sha256": sha256(ROOT / "decision_layer.py"),
        "model_snapshot": str(SNAPSHOT), "model_weight_sha256": sha256(SNAPSHOT / "model.safetensors"),
        "chronos_adaptation": "ZERO-SHOT. Original model; NO fare LoRA checkpoint loaded.",
        "chronos_device": "cpu", "dtype": "float32", "cpu_threads": 4,
        "cross_learning": False, "calendar_covariates": list(CALENDAR),
        "lags": list(LAGS), "rolling_means": [6, 24, 168], "rolling_std": [24],
        "ridge_alpha": 10., "hgb": {"max_iter": 120, "max_leaf_nodes": 31, "learning_rate": .07, "l2_regularization": 1., "early_stopping": False},
        "price_grid": PRICES.tolist(), "eta_constraint_minutes": 10.,
        "decision_model": asdict(config), "seed": SEED,
        "timing": "History [:t]; current queue = queue_end[t-1]; forecast state in next hourly bin t.",
        "no_selection_on_heldout": True,
        "limitations": [
            "Synthetic heldout worlds; not observed Uber arrivals/capacity.",
            "Classical models are trained on synthetic domain data; Chronos uses zero domain adaptation.",
            "Conditional responses use the same NEW illustrative structural calculator on predicted and realized inputs, not observed outcomes from the original simulator.",
            "Multi-target point forecasts do not supply an identified or calibrated joint predictive distribution.",
            "ETA is a quote proxy. Price is an action, not a forecasting target. No Uber One labels or causal policy claim.",
            "One-step best feasible grid action is not a globally optimal pricing policy.",
        ],
    }
    save_json(output_dir / "provenance.json", provenance)
    X = np.asarray([features(train, s, t) for s, t in training_cases], dtype=float)
    y = np.asarray([[train[k][s, t] for k in TARGETS] for s, t in training_cases], dtype=float)
    Xt = np.asarray([features(heldout, s, t) for s, t in cases], dtype=float)
    actual = np.asarray([[heldout[k][s, t] for k in TARGETS] for s, t in cases], dtype=float)
    if not all(np.isfinite(a).all() for a in (X, y, Xt, actual)):
        raise ValueError("Nonfinite pressure values or features")
    predictions, timings = {}, {}
    for name, lag in (("Persistence", 1), ("Seasonal 24h", 24)):
        tick = time.perf_counter()
        predictions[name] = np.asarray([[heldout[k][s, t-lag] for k in TARGETS] for s, t in cases])
        timings[name] = {"training_seconds": 0., "inference_seconds": time.perf_counter()-tick}
    models = {
        "Ridge (domain-trained)": make_pipeline(StandardScaler(), Ridge(alpha=10.)),
        "HGB (domain-trained)": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=31, learning_rate=.07, l2_regularization=1., early_stopping=False, random_state=SEED)),
    }
    with threadpool_limits(limits=4):
        for name, model in models.items():
            tick = time.perf_counter()
            model.fit(X, y)
            fit_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            predictions[name] = np.maximum(0., model.predict(Xt))
            timings[name] = {"training_seconds": fit_seconds, "inference_seconds": time.perf_counter()-tick}
            print(f"{name}: fitted {len(X)} training cases in {fit_seconds:.2f}s", flush=True)
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4)
    tick = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(str(SNAPSHOT), local_files_only=True, device_map="cpu", dtype=torch.float32)
    load_seconds = time.perf_counter()-tick
    inputs = chronos_inputs(heldout, cases)
    tick = time.perf_counter()
    with torch.inference_mode():
        quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=1,
            quantile_levels=[.1, .5, .9], context_length=256, batch_size=32, cross_learning=False)
    prediction = np.stack([a.detach().cpu().numpy()[:, 0, 1] for a in quantiles])
    if prediction.shape != (64, 2) or not np.isfinite(prediction).all():
        raise ValueError(f"Unexpected Chronos output {prediction.shape}")
    chronos_name = "Chronos-2 ZERO-SHOT"
    predictions[chronos_name] = np.maximum(0., prediction)
    timings[chronos_name] = {"training_seconds": 0., "model_load_seconds": load_seconds, "inference_seconds": time.perf_counter()-tick}
    del pipeline
    rows, surfaces, decisions, metrics = [], [], [], []
    example = {"series_id": str(heldout["series_id"][0]), "target_bin_index": 512,
               "target_timestamp": pd.Timestamp(int(heldout["timestamp_ns"][0,512]), tz="UTC").isoformat(),
               "queue": float(heldout["queue_end"][0,511]), "response_definition": "Realized-input NEW model response, not observed original-simulator outcomes", "models": {}}
    for i, (s, t) in enumerate(cases):
        a, cap = actual[i]
        queue = float(heldout["queue_end"][s, t-1])
        reference = response_surface(a, cap, queue, PRICES, config)
        oracle = choose_action(a, cap, queue, PRICES, config=config)
        if i == 0:
            example["realized_inputs"] = {"potential": float(a), "capacity": float(cap)}
            example["realized_input_model_response"] = reference
            example["full_information_grid_action"] = oracle["selected_action"]
        for name, pred in predictions.items():
            predicted_surface = response_surface(*pred[i], queue, PRICES, config)
            action = choose_action(*pred[i], queue, PRICES, config=config)
            selected = action["selected_action"]
            evaluated = solve_response(a, cap, queue, selected["price_multiplier"], config) if selected else None
            true_feasible = bool(evaluated is not None and not evaluated["zero_capacity"] and evaluated["quoted_eta_proxy_minutes"] <= 10.+1e-9)
            row = {"case_id": i, "series_index": s, "series_id": str(heldout["series_id"][s]), "target_index": t,
                   "target_timestamp": pd.Timestamp(int(heldout["timestamp_ns"][s,t]), tz="UTC"), "model": name, "queue": queue,
                   "actual_potential": float(a), "predicted_potential": float(pred[i,0]), "actual_capacity": float(cap), "predicted_capacity": float(pred[i,1])}
            rows.append(row)
            for p, ref in zip(predicted_surface, reference):
                surfaces.append({"case_id": i, "model": name, "price_multiplier": p["price_multiplier"],
                    **{f"predicted_{key}": p[key] for key in ("quoted_eta_proxy_minutes", "request_rate", "new_rider_conversion")},
                    **{f"realized_input_{key}": ref[key] for key in ("quoted_eta_proxy_minutes", "request_rate", "new_rider_conversion")}})
            oracle_action = oracle["selected_action"]
            feasible_both = oracle_action is not None and true_feasible
            regret = max(0., oracle_action["new_rider_completions"]-evaluated["new_rider_completions"]) if feasible_both else None
            decisions.append({"case_id": i, "model": name, "selected_price_multiplier": selected["price_multiplier"] if selected else None,
                "oracle_price_multiplier": oracle_action["price_multiplier"] if oracle_action else None,
                "forecast_feasible": bool(selected), "realized_input_feasible": true_feasible,
                "oracle_feasible": oracle_action is not None, "feasible_grid_regret_new_completions": regret,
                "evaluated_eta": evaluated["quoted_eta_proxy_minutes"] if evaluated else None})
            if i == 0:
                example["models"][name] = {"forecast_potential": float(pred[i,0]), "forecast_capacity": float(pred[i,1]),
                    "response_surface": predicted_surface, "selected_action": selected, "selected_action_evaluated_on_realized_inputs": evaluated}
    pred_frame, surface_frame, decision_frame = pd.DataFrame(rows), pd.DataFrame(surfaces), pd.DataFrame(decisions)
    common_ids = decision_frame.groupby("case_id").realized_input_feasible.all()
    common_ids = common_ids[common_ids].index.tolist()
    for name in predictions:
        p, s, d = (frame[frame.model == name] for frame in (pred_frame, surface_frame, decision_frame))
        ae = np.abs(p[["predicted_potential", "predicted_capacity"]].to_numpy()-p[["actual_potential", "actual_capacity"]].to_numpy())
        common_regret = d[d.case_id.isin(common_ids)].feasible_grid_regret_new_completions
        metrics.append({"model": name, "potential_mae": float(ae[:,0].mean()), "capacity_mae": float(ae[:,1].mean()), "joint_state_mae": float(ae.mean()),
            "eta_model_response_mae_minutes": float(np.abs(s.predicted_quoted_eta_proxy_minutes-s.realized_input_quoted_eta_proxy_minutes).mean()),
            "request_rate_model_response_mae_pp": float(100*np.abs(s.predicted_request_rate-s.realized_input_request_rate).mean()),
            "new_completion_conversion_model_response_mae_pp": float(100*np.abs(s.predicted_new_rider_conversion-s.realized_input_new_rider_conversion).mean()),
            "selected_action_count": int(d.forecast_feasible.sum()), "selected_action_feasible_under_realized_inputs_count": int(d.realized_input_feasible.sum()),
            "infeasible_selected_action_count": int((d.forecast_feasible & ~d.realized_input_feasible).sum()),
            "common_feasible_grid_regret_new_completions": float(common_regret.mean()) if len(common_regret) else None,
            "common_feasible_cases": len(common_ids), **timings[name]})
    payload = {"task": "Zero-shot next-hour arrival/capacity forecast -> conditional rider-price action map", "case_count": 64,
               "metrics": metrics, "heldout_selection": "First 16 synthetic heldout series, target indices 512/640/768/896; frozen before scoring",
               "oracle_feasible_cases": int(decision_frame[decision_frame.model == chronos_name].oracle_feasible.sum()),
               "common_feasible_case_ids": common_ids, "action_regret_note": "New-completion regret only on identical cases where every model's chosen action and the full-information toy-model grid action satisfy the modeled ETA limit. Feasibility failures reported separately; not global or empirical optimality.",
               "response_error_note": "Errors compare the same illustrative response calculator under forecast vs realized synthetic inputs, NOT observed ETA/conversion labels.",
               "total_seconds": time.perf_counter()-started}
    pred_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    surface_frame.to_parquet(output_dir / "surfaces.parquet", index=False)
    decision_frame.to_parquet(output_dir / "decisions.parquet", index=False)
    save_json(output_dir / "metrics.json", payload)
    save_json(output_dir / "example_surface.json", example)
    provenance.update(status="completed", total_seconds=time.perf_counter()-started, timings=timings,
                      python=platform.python_version(), packages={"torch": torch.__version__})
    save_json(output_dir / "provenance.json", provenance)
    render_report(output_dir, heldout)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def render_report(output_dir, heldout=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    if heldout is None:
        heldout = np.load(ROOT / "outputs/synthetic/validation.npz", allow_pickle=False)
    payload = json.loads((output_dir / "metrics.json").read_text())
    example = json.loads((output_dir / "example_surface.json").read_text())
    pred = pd.read_parquet(output_dir / "predictions.parquet")
    metrics = pd.DataFrame(payload["metrics"])
    colors = {"Persistence": "#929caf", "Seasonal 24h": "#a97c50", "Ridge (domain-trained)": "#8055a6", "HGB (domain-trained)": "#009e73", "Chronos-2 ZERO-SHOT": "#1764bd"}
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12,4.4))
    for ax, target, label in zip(axes, TARGETS, ("Potential rider arrivals / hour", "Service capacity / hour")):
        ax.plot(np.arange(-48, 1), heldout[target][0,464:513], color="#172842", label="Realized synthetic state", linewidth=1.8)
        for name, color in colors.items():
            p = pred[(pred.model==name)&(pred.case_id==0)].iloc[0]
            ax.scatter([0], [p[f"predicted_{target}"]], color=color, s=55, marker="x", zorder=5, label=name)
        ax.axvline(-.5, color="#6f7e91", linestyle="--", linewidth=.8)
        ax.set(xlabel="Hours relative to next-hour target", ylabel=label, title=label)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Prechosen first heldout world, origin 512: 48h context + one-step forecast", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_dir/"pressure_context.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13,4.2))
    fields = ("quoted_eta_proxy_minutes", "request_rate", "new_rider_conversion")
    labels = ("Modeled quoted ETA (minutes)", "Modeled request conversion (%)", "Modeled new completed conversion (%)")
    for ax, key, label in zip(axes, fields, labels):
        scale = 1 if key == fields[0] else 100
        ref = example["realized_input_model_response"]
        ax.plot([r["price_multiplier"] for r in ref], [scale*r[key] for r in ref], color="#172842", linewidth=2.6, label="Realized-input model response")
        for name in ("Chronos-2 ZERO-SHOT", "HGB (domain-trained)"):
            rows = example["models"][name]["response_surface"]
            ax.plot([r["price_multiplier"] for r in rows], [scale*r[key] for r in rows], color=colors[name], linestyle="--", linewidth=1.8, label=name)
        if key == fields[0]:
            ax.axhline(10., linestyle=":", color="#a65c28", label="10-minute modeled ETA constraint")
        ax.set(xlabel="Candidate rider-price multiplier (ACTION)", ylabel=label)
    axes[0].legend(fontsize=7)
    fig.suptitle("Same queue and same illustrative demand model; only forecast state changes", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_dir/"conditional_price_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    def img(name):
        return '<img alt="'+html.escape(name)+'" src="data:image/png;base64,'+base64.b64encode((figure_dir/name).read_bytes()).decode()+'">'
    def fmt(value, digits=2):
        return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"
    rows = ""
    for r in payload["metrics"]:
        rows += "<tr><th>"+html.escape(r["model"])+"</th>"+"".join("<td>"+fmt(r[k])+"</td>" for k in ("potential_mae","capacity_mae","eta_model_response_mae_minutes","request_rate_model_response_mae_pp","new_completion_conversion_model_response_mae_pp","common_feasible_grid_regret_new_completions"))+f'<td>{r["infeasible_selected_action_count"]}/{r["selected_action_count"]}</td></tr>'
    best = metrics.loc[metrics.joint_state_mae.idxmin()]
    ch = metrics[metrics.model=="Chronos-2 ZERO-SHOT"].iloc[0]
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Rider pricing: forecast pressure, then choose an action</title><style>
    body{{font:16px/1.55 system-ui,sans-serif;color:#172842;background:#edf2f8;margin:0}}main{{max-width:1120px;margin:auto;background:white;padding:44px}}h1{{font-size:32px;line-height:1.15}}h2{{font-size:22px;margin-top:32px}}.eyebrow{{text-transform:uppercase;letter-spacing:.14em;color:#1764bd;font-size:12px;font-weight:700}}.lead{{font-size:19px}}.note{{padding:16px 20px;background:#eef5fc;border-left:4px solid #1764bd}}.warn{{background:#fff6e9;border-left-color:#b8741c}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:9px 7px;border-bottom:1px solid #dce4ee;text-align:right}}th:first-child{{text-align:left}}thead{{background:#edf3fa}}a{{color:#1764bd}}small{{color:#536477}}@media print{{body{{background:white}}main{{padding:14px}}h2,figure{{break-inside:avoid}}}}</style></head><body><main>
    <div class="eyebrow">Local research demo · synthetic heldout worlds · zero-shot Chronos-2</div>
    <h1>Forecast pressure. Treat rider price as the decision.</h1>
    <p class="lead">A foundation model forecasts next-hour potential rider arrivals and service capacity jointly. A transparent economic model then maps candidate rider prices into quoted ETA and conversion. Price itself is not the forecast target.</p>
    <div class="note"><b>Observed result:</b> {html.escape(str(best['model']))} has the lowest average arrival/capacity MAE across the 64 fixed heldout cases ({best['joint_state_mae']:.2f} count units). Zero-shot Chronos-2 scores {ch['joint_state_mae']:.2f}. These are synthetic results, not proof of improvement at Uber.</div>
    <h2>1. Forecast the state with limited context</h2><p>First 16 unseen synthetic trajectories, four prespecified origins each, 256 hours of history, one-hour horizon. Chronos receives two target series plus known calendar features with <b>zero task-specific training</b>. Ridge and gradient boosting receive 12,032 labeled examples from the first 32 training trajectories; their hyperparameters are fixed, not tuned on these 64 cases.</p>
    {img('pressure_context.png')}
    <h2>2. Map each candidate price to outcomes</h2><p>All forecasters use the same known current queue and the same fixed heterogeneous price/time sensitivities. The navy curve uses realized synthetic arrival/capacity inputs in this <em>new illustrative calculator</em>; it is not an observed conversion or ETA label. Higher price can reduce requests and quoted congestion without increasing completed conversion when capacity is fixed.</p>
    {img('conditional_price_curves.png')}
    <h2>3. Compare forecasts, conditional responses, and feasible grid decisions</h2>
    <table><thead><tr><th>Model</th><th>Arrival MAE</th><th>Capacity MAE</th><th>ETA response MAE (min)</th><th>Request response MAE (pp)</th><th>Completed response MAE (pp)</th><th>Grid regret (rides)</th><th>ETA failures / actions</th></tr></thead><tbody>{rows}</tbody></table>
    <p><small>Response errors average all 11 price candidates per case. Grid rule: maximize new completed rides subject to modeled ETA ≤10 minutes; ties choose the lowest price. No feasible action means abstention. Regret uses only {len(payload['common_feasible_case_ids'])} identical cases where every model’s selected action is feasible under realized inputs; constraint violations are reported separately. The reference is the best feasible action on this one-step illustrative grid, not a globally optimal price.</small></p>
    <h2>What this demonstrates—and what it does not</h2>
    <div class="note warn">This is a scrappy architecture demonstration: forecast → explicit economic response → constrained action → validation. The demand/time coefficients are illustrative priors, capacity is exogenous, and neither treatment effects nor Uber One behavior are identified. Multi-target forecasts do not provide a calibrated joint uncertainty distribution. A production system would require internal quote/request/acceptance/pickup data, calibrated uncertainty, pricing experiments, guardrails, and engineering reliability.</div>
    <p>Science owns the assumptions, objective, counterfactual validation, and failure analysis. Engineering and science jointly own data contracts, latency budgets, safe serving, experiment rollout, and monitoring. The demo measures plug-in forecast error; it does not recommend deploying a new pricing policy.</p>
    <p><a href="../report.html">Secondary evidence: real NYC rider-fare forecasting benchmark</a>. That is a different target. Its synthetic-fare LoRA checkpoints are <b>not used here</b> and provide no evidence of transfer to arrival, ETA, or conversion prediction.</p>
    <p><small>Runtime {payload['total_seconds']:.1f}s, local CPU. Reproduce: <code>.venv/bin/python pressure_benchmark.py</code>. Frozen cases, model hash, structural priors, and timings: <a href="provenance.json">provenance.json</a>. Numerical results: <a href="metrics.json">metrics.json</a>.</small></p>
    </main></body></html>'''
    (output_dir / "report.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/decision_demo")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        render_report(args.output_dir)
    else:
        benchmark(args.output_dir)

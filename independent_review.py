"""Read-only numerical/heterogeneity audit; writes a separate dated receipt.

No forecast code is imported, no model fitted, and no protocol altered here.
The comparisons are descriptive diagnostics, never an extra selection stage.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
KEYS = ["origin", "series_id", "horizon", "target_timestamp"]
CHRONOS = "Chronos-2 + calendar/count"
SEASONAL = "Four-week seasonal"
HGB = "Gradient boosting"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(group):
    error = group.prediction - group.actual
    high = group.loc[group.high_fare]
    quantiles = np.array([.1, .5, .9])
    residual = group.actual.to_numpy()[:, None] - group[["q10", "prediction", "q90"]].to_numpy()
    return {
        "n": len(group), "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(error.mean()),
        "high_fare_n": len(high),
        "high_fare_mae": float((high.prediction-high.actual).abs().mean()) if len(high) else None,
        "high_fare_bias": float((high.prediction-high.actual).mean()) if len(high) else None,
        "coverage_80": float(((group.actual>=group.q10)&(group.actual<=group.q90)).mean()),
        "width_80": float((group.q90-group.q10).mean()),
        "mean_pinball": float(np.maximum(quantiles*residual,(quantiles-1)*residual).mean()),
    }


def main():
    out = ROOT / "outputs"
    protocol = json.loads((ROOT/"protocol.json").read_text())
    metrics = json.loads((out/"metrics.json").read_text())
    report = json.loads((out/"report_metrics.json").read_text())
    manifest = json.loads((out/"run_manifest.json").read_text())
    panel = pd.read_parquet(ROOT/"data/panel.parquet")
    frames = {s: pd.read_parquet(out/f"{s}_predictions.parquet") for s in ("validation", "test")}
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    cutoff = pd.Timestamp(protocol["train_end"])-pd.Timedelta(hours=protocol["embargo_hours"])
    thresholds = panel.loc[panel.timestamp<cutoff].groupby("series_id").fare.quantile(.9)
    recomputed = {}
    for split, frame in frames.items():
        recomputed[split] = {}
        check(f"{split}_same_truth", frame.groupby(KEYS).actual.nunique().eq(1).all())
        check(f"{split}_same_model_support", frame.groupby(KEYS).model.nunique().eq(frame.model.nunique()).all())
        check(f"{split}_no_duplicate_keys", not frame.duplicated(KEYS+["model"]).any())
        check(f"{split}_tail_uses_only_train_thresholds", np.allclose(frame.high_fare_threshold,frame.series_id.map(thresholds)))
        check(f"{split}_tail_indicator_correct", frame.high_fare.eq(frame.actual>frame.high_fare_threshold).all())
        for row in metrics[split]:
            group = frame.loc[frame.model.eq(row["model"])]
            if row["horizon"]!="all":
                group = group.loc[group.horizon.eq(int(row["horizon"]))]
            values = stats(group)
            for key in ("n","mae","rmse","bias","high_fare_n","high_fare_mae","coverage_80","width_80","mean_pinball"):
                check(f"{split}/{row['model']}/{row['horizon']}/{key}",
                      values[key] is None if row[key] is None else np.isclose(values[key],row[key],rtol=0,atol=1e-10))
            if row["horizon"]=="all":
                recomputed[split][row["model"]] = values
    for row in report["model_summary"]:
        for key in ("mae","rmse","high_fare_mae","coverage_80"):
            check(f"report/{row['model']}/{key}",np.isclose(row[key],recomputed["test"][row["model"]][key],rtol=0,atol=1e-10))

    val_scores = pd.Series({m:v["mae"] for m,v in recomputed["validation"].items()})
    baseline = val_scores.loc[~val_scores.index.str.startswith("Chronos")].idxmin()
    chronos = val_scores.loc[val_scores.index.str.startswith("Chronos")].idxmin()
    check("baseline_selected_on_validation",metrics["comparison"]["baseline"]==baseline==report["validation_selected_baseline"])
    check("chronos_selected_on_validation",metrics["comparison"]["chronos"]==chronos==report["validation_selected_chronos"])
    adaptation = manifest["synthetic_adaptation"]
    candidates = pd.read_parquet(out/"synthetic_checkpoint_validation.parquet")
    checkpoint_scores = candidates.assign(e=lambda f:(f.actual-f.prediction).abs()).groupby("model").e.mean()
    for row in adaptation["candidates"]:
        check(f"checkpoint/{row['steps']}/validation_score",np.isclose(checkpoint_scores[row["model"]],row["validation_mae"],atol=1e-10,rtol=0))
    check("diagnostic_adapter_choice_uses_only_validation",adaptation["selected"]["model"]==checkpoint_scores.idxmin())
    choices = [adaptation["zero_step"],adaptation["selected"]]
    check("zero_step_option_retained",adaptation["validation_recommendation"]==min(choices,key=lambda x:(x["validation_mae"],x["steps"])))

    test = frames["test"].copy()
    test["ae"] = (test.prediction-test.actual).abs()
    pair = test.loc[test.model.isin([baseline,chronos])].pivot(index=KEYS,columns="model",values="ae").reset_index()
    pair["day"] = pair.origin.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.date
    blocks = pair.groupby("day")[[baseline,chronos]].agg(["sum","count"])
    draws = np.random.default_rng(20261026).integers(0,len(blocks),(2000,len(blocks)))
    means = {m:blocks[(m,"sum")].to_numpy()[draws].sum(1)/blocks[(m,"count")].to_numpy()[draws].sum(1) for m in (baseline,chronos)}
    relative_ci = np.quantile(100*(means[baseline]-means[chronos])/means[baseline],[.025,.975])
    absolute_ci = np.quantile(means[chronos]-means[baseline],[.025,.975])
    check("relative_day_block_bootstrap_recomputed",np.allclose(relative_ci,metrics["comparison"]["paired_daily_block_bootstrap_95pct"],rtol=0,atol=1e-10))
    check("absolute_day_block_bootstrap_recomputed",np.allclose(absolute_ci,[report["paired_comparison"]["day_block_95pct_lower"],report["paired_comparison"]["day_block_95pct_upper"]],rtol=0,atol=1e-10))

    routes=[]
    for (sid,model),group in test.groupby(["series_id","model"]):
        routes.append({"series_id":sid,"model":model,**stats(group)})
    horizons=[]
    for (horizon,model),group in test.groupby(["horizon","model"]):
        horizons.append({"horizon":int(horizon),"model":model,**stats(group)})
    comparison_table = test.loc[test.model.isin([CHRONOS,SEASONAL,HGB])].groupby(["series_id","model"]).ae.mean().unstack()
    check("descriptive_chronos_beats_two_references_all_six_routes",comparison_table[CHRONOS].lt(comparison_table[[SEASONAL,HGB]].min(axis=1)).all())
    horizon_table = test.loc[test.model.isin([CHRONOS,SEASONAL,HGB])].groupby(["horizon","model"]).ae.mean().unstack()
    check("descriptive_chronos_beats_two_references_all_three_horizons",horizon_table[CHRONOS].lt(horizon_table[[SEASONAL,HGB]].min(axis=1)).all())
    selected = test.loc[test.model.eq(CHRONOS)]
    worst = selected.nlargest(10,"ae")[[*KEYS,"actual","prediction","target_count","ae"]].copy()
    for col in ("origin","target_timestamp"):
        worst[col] = worst[col].astype(str)
    raw_test = panel.loc[panel.timestamp.ge(pd.Timestamp(protocol["validation_end"])) & panel.timestamp.lt(pd.Timestamp(protocol["test_end"]))]
    coverage = [{"series_id":sid,"hours":len(g),"observed_target_hours":int(g.fare.notna().sum()),"observed_fraction":float(g.fare.notna().mean()),"median_miles":float(g.median_miles.median())} for sid,g in raw_test.groupby("series_id")]
    sc=recomputed["test"][CHRONOS]
    synth=recomputed["test"][adaptation["selected"]["model"]]
    receipt = {
        "status":"passed", "created_utc":datetime.now(timezone.utc).isoformat(),
        "check_count":len(checks),"checks":checks,
        "inputs_sha256":{name:sha(out/name) for name in ("metrics.json","test_predictions.parquet","validation_predictions.parquet","run_manifest.json","report_metrics.json")},
        "audit_script_sha256":sha(__file__),
        "overall":recomputed,"per_route":routes,"per_horizon":horizons,
        "test_target_coverage":coverage,"largest_chronos_calendar_errors":worst.to_dict("records"),
        "descriptive_effects":{
            "chronos_reduction_vs_validation_selected_seasonal_pct":100*(1-sc["mae"]/recomputed["test"][SEASONAL]["mae"]),
            "chronos_reduction_vs_hgb_pct":100*(1-sc["mae"]/recomputed["test"][HGB]["mae"]),
            "chronos_tail_mae_increase_vs_seasonal_pct":100*(sc["high_fare_mae"]/recomputed["test"][SEASONAL]["high_fare_mae"]-1),
            "synthetic_adapter_test_reduction_vs_original_covariates_pct":100*(1-synth["mae"]/sc["mae"]),
        },
        "interpretation":[
            "Synthetic adapter gain is negligible on test and validation selected step zero; no evidence here to recommend adaptation.",
            "Calendar/count Chronos improves aggregate MAE across all six routes and all three horizons against both named references, but not the high-fare tail against seasonal.",
            "Largest errors include several simultaneous March 23 07:00 NYC underpredictions; their external cause is not established.",
            "Different horizons have different target-clock samples under six-hour origins; lower h6 MAE does not show intrinsically improving long-horizon accuracy.",
            "80% native interval coverage is below nominal; no guarantee of calibrated risk control.",
            "Six short nonairport OD routes and one 28-calendar-day window are not representative proof for NYC, Uber surge, ETA, conversion, or policy improvement.",
            "Any already inspected real test outcomes are exploratory; no new model selection or tuning occurs in this audit.",
        ],
    }
    (out/"independent_review.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":receipt["status"],"checks":len(checks),"effects":receipt["descriptive_effects"]},indent=2))


if __name__=="__main__":
    main()

"""Build an offline interview report from completed, auditable benchmark outputs.

No synthetic results, live API calls, or model fitting occur in this report.
Example chart selection depends on the training-only route ranking and the first
test week, never on model performance. The HTML embeds its scientific figures.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEYS = ["origin", "target_timestamp", "series_id", "horizon"]
COLORS = ["#117b8b", "#1954a6", "#e3943b", "#7547a3", "#46667a", "#a65f65", "#5b8432", "#7087a5", "#b45a9d"]
SOURCE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
DICTIONARY = "https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf"
MODEL = "https://huggingface.co/amazon/chronos-2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#21324a", "text.color": "#21324a",
    "axes.edgecolor": "#b8c7d7", "xtick.color": "#41556d", "ytick.color": "#41556d",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def prepare_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "actual" not in frame and "y_true" in frame:
        frame = frame.rename(columns={"y_true": "actual"})
    required = {*KEYS, "model", "actual", "prediction"}
    if not required.issubset(frame):
        raise ValueError(f"{path.name}: missing columns {required - set(frame)}")
    frame["series_id"] = frame.series_id.astype(str)
    for col in ("origin", "target_timestamp"):
        frame[col] = pd.to_datetime(frame[col])
    if frame.duplicated(KEYS + ["model"]).any():
        raise ValueError("Duplicate model/case predictions must be resolved before reporting")
    # Labels can be missing on low-volume target hours. They are never fabricated.
    frame = frame.loc[np.isfinite(frame.actual)].copy()
    if frame.empty or not np.isfinite(frame.prediction).all():
        raise ValueError("Need nonempty, finite predictions on observed outcome cases")
    count = frame.groupby(KEYS, dropna=False).model.nunique()
    if not count.eq(frame.model.nunique()).all():
        raise ValueError("All models must be scored on exactly the same observed cases")
    if not frame.groupby(KEYS).actual.nunique().eq(1).all():
        raise ValueError("Model rows disagree about actual outcomes")
    frame["error"] = frame.prediction - frame.actual
    frame["absolute_error"] = frame.error.abs()
    frame["squared_error"] = frame.error ** 2
    if "high_fare" not in frame:
        frame["high_fare"] = False
    return frame


def model_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in frame.groupby("model", sort=True):
        tail = group.loc[group.high_fare.fillna(False).astype(bool)]
        row = {"model": name, "cases": len(group), "mae": float(group.absolute_error.mean()),
            "rmse": float(np.sqrt(group.squared_error.mean())),
            "high_fare_mae": float(tail.absolute_error.mean()) if len(tail) else None,
            "high_fare_cases": len(tail), "coverage_80": None, "interval_width": None}
        if {"q10", "q90"}.issubset(group):
            valid = np.isfinite(group.q10) & np.isfinite(group.q90)
            intervals = group.loc[valid]
            if len(intervals):
                if (intervals.q10 > intervals.q90).any():
                    raise ValueError(f"Crossed prediction intervals for {name}")
                row["coverage_80"] = float(((intervals.actual >= intervals.q10)
                    & (intervals.actual <= intervals.q90)).mean())
                row["interval_width"] = float((intervals.q90 - intervals.q10).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mae", "model"]).reset_index(drop=True)


def paired_comparison(frame: pd.DataFrame, model: str, baseline: str) -> dict:
    """Day-block bootstrap preserves within-day route/horizon dependence.

    This is descriptive uncertainty for the fixed test window, not a population
    guarantee. Adjacent days can remain dependent and the window is short.
    """
    pairs = frame.pivot(index=KEYS, columns="model", values="absolute_error")
    diff = (pairs[model] - pairs[baseline]).rename("difference").reset_index()
    diff["day"] = diff.origin.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.date
    blocks = diff.groupby("day").difference.agg(["sum", "count"])
    rng = np.random.default_rng(20261026)
    draw = rng.integers(0, len(blocks), size=(2000, len(blocks)))
    boot = blocks["sum"].to_numpy()[draw].sum(axis=1) / blocks["count"].to_numpy()[draw].sum(axis=1)
    lower, upper = np.quantile(boot, [.025, .975])
    return {"model": model, "baseline": baseline, "mae_difference_usd": float(diff.difference.mean()),
        "day_block_95pct_lower": float(lower), "day_block_95pct_upper": float(upper),
        "blocks": len(blocks), "bootstrap_draws": 2000,
        "interpretation": "Negative difference favors the named model. Descriptive short-window day-block bootstrap; not a generalization guarantee."}


def save_figure(fig, path: Path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def chart_errors(summary, frame, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, max(4.4, .49*len(summary)+1)), gridspec_kw={"width_ratios": [1.15, 1]})
    ordered = summary.sort_values("mae", ascending=False)
    colors = ["#1954a6" if "chronos" in m.lower() else "#94a7b9" for m in ordered.model]
    labels = ["\n".join(textwrap.wrap(m, 25)) for m in ordered.model]
    axes[0].barh(labels, ordered.mae, color=colors, height=.66)
    axes[0].tick_params(axis="y", labelsize=8.5)
    for i, value in enumerate(ordered.mae):
        axes[0].text(value + ordered.mae.max() * .018, i, f"${value:.2f}", va="center", fontsize=9)
    axes[0].set_xlim(0, ordered.mae.max() * 1.22)
    axes[0].set_xlabel("Mean absolute error (USD)")
    axes[0].set_title("Held-out fare forecast error", loc="left", weight="bold", pad=12)
    axes[0].grid(axis="x", alpha=.16)
    axes[0].set_axisbelow(True)
    for i, name in enumerate(summary.model):
        horizon = frame.loc[frame.model.eq(name)].groupby("horizon").absolute_error.mean()
        axes[1].plot(horizon.index, horizon.values, marker="o", markersize=4,
            color=COLORS[i % len(COLORS)], label="\n".join(textwrap.wrap(name, 28)), linewidth=1.7)
    axes[1].set_xlabel("Forecast horizon (hours)")
    axes[1].set_ylabel("Mean absolute error (USD)")
    axes[1].set_xticks(sorted(frame.horizon.unique()))
    axes[1].set_title("Error by horizon", loc="left", weight="bold", pad=12)
    axes[1].grid(alpha=.16)
    axes[1].legend(frameon=False, fontsize=7.5, loc="best")
    fig.tight_layout(w_pad=2)
    save_figure(fig, path)


def route_detail(metadata, sid):
    for route in metadata.get("routes", []):
        if str(route.get("series_id")) == sid:
            return f"{route.get('origin', sid)} → {route.get('destination', '')}"
    return sid


def chart_trace(frame, metadata, baseline, path):
    routes = metadata.get("routes", [])
    sid = str(routes[0]["series_id"]) if routes else sorted(frame.series_id.unique())[0]
    available = frame.loc[frame.series_id.eq(sid)]
    horizon = 3 if 3 in available.horizon.unique() else min(available.horizon)
    start = frame.origin.min()
    end = start + pd.Timedelta(days=7)
    selection = available.loc[available.horizon.eq(horizon) & available.origin.lt(end)].copy()
    names = [baseline, "Chronos-2", "Chronos-2 + calendar/count"]
    actual = selection.drop_duplicates("target_timestamp").sort_values("target_timestamp")
    fig, ax = plt.subplots(figsize=(11.8, 4.1))
    ax.plot(actual.target_timestamp, actual.actual, color="#182c42", label="Observed median rider fare", linewidth=1.6)
    for i, name in enumerate(names):
        group = selection.loc[selection.model.eq(name)].sort_values("target_timestamp")
        ax.plot(group.target_timestamp, group.prediction, color=COLORS[(i+1) % len(COLORS)],
                label=name.replace("Chronos-2", "Chronos-2 zero-shot"), linewidth=1.35, alpha=.9)
        if name == "Chronos-2 + calendar/count":
            ax.fill_between(group.target_timestamp, group.q10, group.q90, color=COLORS[(i+1) % len(COLORS)],
                            alpha=.12, label="Native 80% band (calendar/count)")
    ax.set_title(f"{route_detail(metadata, sid)} · {horizon}-hour forecast", loc="left", weight="bold", pad=12)
    ax.set_ylabel("Median rider fare (USD)")
    ax.set_xlabel("Target pickup hour (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=.16)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, path)
    return {"series_id": sid, "label": route_detail(metadata, sid), "horizon": int(horizon),
            "selection": "First route in January-only ranking, first seven test days, horizon three where available. Not selected for favorable model performance."}


def image_data(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def number(value, digits=2, prefix=""):
    return "—" if value is None or pd.isna(value) else f"{prefix}{value:,.{digits}f}"


def latency_by_model(manifest, frame):
    values = manifest.get("model_latency_seconds", {})
    values = dict(values) if isinstance(values, dict) else {}
    timings = manifest.get("timings", {})
    for model in frame.model.unique():
        if model in values:
            continue
        entry = timings.get(f"{model}/test", {})
        if "amortized_ms_per_route_origin" in entry:
            values[model] = entry["amortized_ms_per_route_origin"] / 1000
        elapsed = timings.get("baseline_test_by_model_seconds", {}).get(model)
        if elapsed is not None:
            n_origins = timings.get("Chronos-2/test", {}).get("route_origins")
            if n_origins is None:
                n_origins = frame.loc[frame.model.eq(model), ["origin", "series_id"]].drop_duplicates().shape[0]
            values[model] = elapsed / n_origins
    return values


def synthetic_section(output_dir, manifest, summary):
    directory = output_dir / "synthetic"
    simulation = read_json(directory / "manifest.json", {})
    if not simulation:
        return ""
    training = read_json(directory / "fine_tuning_manifest.json", {})
    recipe = read_json(directory / "fine_tuning_protocol.json", {})
    stress = read_json(directory / "stress_policy_comparison.json", {})
    illustrative = read_json(directory / "policy_comparison.json", {})
    adaptation = manifest.get("synthetic_adaptation", {})
    selection_text = html.escape(json.dumps(adaptation.get("validation_recommendation", "See saved selection manifest"), ensure_ascii=False))
    fine_rows = summary.loc[summary.model.str.contains("synthetic|simulation|economic|LoRA", case=False, regex=True)]
    comparison = ""
    if not fine_rows.empty and "Chronos-2 + calendar/count" in summary.model.values:
        frozen = summary.set_index("model").loc["Chronos-2 + calendar/count", "mae"]
        statements = []
        for _, row in fine_rows.iterrows():
            change = 100*(row.mae/frozen-1)
            statements.append(f"{html.escape(row.model)}: ${row.mae:.2f} test MAE, {abs(change):.1f}% {'higher' if change>0 else 'lower'} than frozen Chronos with the same covariates.")
        comparison = '<p class="callout">' + " ".join(statements) + " This diagnostic is shown even if validation recommends keeping the original model.</p>"
    # The stress file stores separately parameterized worlds. Do not pool them or
    # claim that across-world differences isolate a single parameter.
    worlds = stress.get("worlds", stress.get("rows", []))
    if not worlds and illustrative.get("rows"):
        worlds = [{"world": "Fixed illustration", "rows": illustrative["rows"]}]
    policy_rows = []
    for world in worlds:
        if not isinstance(world, dict) or "rows" not in world:
            continue
        for row in world["rows"]:
            cells = [html.escape(str(world.get("world", "Illustration"))), html.escape(row["policy"]),
                f"{100*row['requested_over_potential']:.1f}%", f"{100*row['completed_over_potential']:.1f}%",
                number(row.get("mean_quoted_eta_minutes_weighted_by_potential"), 1),
                number(row.get("gross_rider_fare_revenue_usd_before_costs"), 0, "$"),
                str(row.get("ending_queue", "—"))]
            policy_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    policy_table = ("<div class=\"tablewrap\"><table><thead><tr><th>Simulated world</th><th>Rule</th>"
        "<th>Requests / potential</th><th>Completed / potential</th><th>Quoted ETA (min)</th>"
        "<th>Gross rider fares</th><th>Ending queue</th></tr></thead><tbody>" + "".join(policy_rows)
        + "</tbody></table></div>") if policy_rows else ""
    elapsed = training.get("optimization_seconds")
    time_text = f"{elapsed:.1f} seconds" if isinstance(elapsed, (float,int)) else "Recorded in the training manifest"
    return f'''<section><h2>Economic simulation and synthetic-only adaptation</h2>
<p>The economic prior links common rider prices and quoted pickup ETA to heterogeneous riders’ willingness to request a trip. It is motivated by the price–time choice problem in <a href="https://doi.org/10.3982/ECTA18838">Buchholz, Doval, Kastl, Matějka and Salz (Econometrica, 2025)</a>, not a replication or use of their estimated parameters.</p>
<div class="cols"><div><h3>What the simulator assumes</h3><ul>
<li>Two-route, hourly capacity/queue episodes with positive, heterogeneous price and time sensitivities and an outside option.</li>
<li>FIFO service and explicit cancellations conserve the request queue. Pending requests retain their quoted fares.</li>
<li>Driver capacity is exogenous across pricing rules. Rider prices do not automatically increase driver pay or supply.</li>
<li>Parameters, shocks, scarcity and policy rules vary across episodes. NYC training data supply only coarse fare/count scales, not estimated elasticities.</li>
<li>This is not a fleet-conserving spatial equilibrium, an estimated demand system, or a production matching engine.</li></ul></div>
<div><h3>Adaptation contract</h3><ul>
<li>{simulation.get('episodes', '—')} simulated episodes × {simulation.get('routes_per_episode', '—')} routes × {simulation.get('hours', '—')} hours. Low-count fare outcomes remain missing.</li>
<li>{recipe.get('optimizer_steps', '—')} local LoRA steps; rank 8, learning rate 0.00001, context 256 hours, nine future steps. Five calendar covariates are known ahead; served count is historical only.</li>
<li>{number(training.get('trainable_parameters'), 0)} trainable parameters. Measured optimization time: {time_text}.</li>
<li>No real validation/test outcomes enter gradient updates. Step 0 remains a legitimate selection option.</li>
<li>Validation recommendation: <code>{selection_text}</code>. The adapted candidate remains visible as a diagnostic even when it should not be adopted.</li></ul></div></div>
{comparison}
<h3>Model-dependent policy scenarios</h3>
<p class="note">Compare rules only within the same world, where potential arrivals, capacity and choice uniforms are held fixed. Worlds differ in multiple parameters. The anticipatory rule uses calendar information and lagged potential arrivals; it does not use Chronos forecasts and requires arrival information absent from TLC. Quoted ETA is a proxy, not realized waiting time. Gross rider fares exclude driver costs and are not profit. These are synthetic outcomes, not treatment effects at Uber.</p>
{policy_table}
<p class="note">Source: saved simulation receipts, <code>synthetic_protocol.json</code>, <code>ECONOMIC_DESIGN.rst</code> and <code>outputs/synthetic/</code>. Lower synthetic loss or a favorable scenario cannot establish improved real-world forecasting or optimal pricing.</p>
</section>'''


def build_report(output_dir: Path):
    output_dir = output_dir.resolve()
    frame = prepare_predictions(output_dir / "test_predictions.parquet")
    validation = prepare_predictions(output_dir / "validation_predictions.parquet")
    metadata = read_json(ROOT / "data/metadata.json", {})
    manifest = read_json(output_dir / "run_manifest.json", {})
    summary = model_summary(frame)
    val_summary = model_summary(validation)
    baseline_rows = val_summary.loc[~val_summary.model.str.contains("chronos", case=False)]
    if baseline_rows.empty:
        raise ValueError("Report needs at least one non-Chronos baseline")
    baseline = baseline_rows.iloc[0].model
    chronos_names = [m for m in summary.model if "chronos" in m.lower()]
    if not chronos_names:
        raise ValueError("Report requires completed Chronos forecasts, not only baselines")
    # Select the Chronos configuration on validation as well, never on final test.
    val_chronos = val_summary.loc[val_summary.model.isin(chronos_names)].iloc[0].model
    comparison = paired_comparison(frame, val_chronos, baseline)
    best = summary.iloc[0]
    selected_chronos = summary.set_index("model").loc[val_chronos]
    baseline_test = summary.set_index("model").loc[baseline]
    relative = 100 * (selected_chronos.mae / baseline_test.mae - 1)
    difference_text = f"{abs(relative):.1f}% {'lower' if relative < 0 else 'higher'} MAE"
    head = (f"The validation-selected {html.escape(str(val_chronos))} has {difference_text} "
        f"than {html.escape(str(baseline))} on held-out NYC fares.")
    zero = summary.set_index("model").loc["Chronos-2 + calendar/count"]
    boosting = summary.set_index("model").loc["Gradient boosting"]
    boosting_gain = 100 * (1 - zero.mae / boosting.mae)
    caveat = (f"The unadapted, covariate-informed model also has {boosting_gain:.1f}% lower observed MAE than tuned gradient boosting "
        f"(${zero.mae:.2f} versus ${boosting.mae:.2f}). This is a descriptive test comparison. "
        f"Its high-fare MAE is ${zero.high_fare_mae:.2f}, versus ${baseline_test.high_fare_mae:.2f} for {html.escape(str(baseline))}; "
        f"nominal 80% intervals cover {100*zero.coverage_80:.1f}% of outcomes. Synthetic adaptation adds no material improvement in this bounded pilot; validation retains the original model.")
    figs = output_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    chart_errors(summary, frame, figs / "forecast_errors.png")
    example = chart_trace(frame, metadata, baseline, figs / "first_route_first_week.png")
    unique = frame.drop_duplicates(KEYS)
    routes = metadata.get("routes", [])
    route_text = "".join(f"<li>{html.escape(route_detail(metadata, str(r['series_id'])))} "
        f"<span>({int(r.get('train_trips', 0)):,} January rides)</span></li>" for r in routes)
    timing = latency_by_model(manifest, frame)
    headers = ["Model", "MAE ↓", "RMSE ↓", "High-fare MAE ↓", "80% coverage", "Interval width", "Inference / route-origin"]
    table_head = "".join(f"<th>{h}</th>" for h in headers)
    table_rows = []
    for row in summary.to_dict("records"):
        cls = "chronos" if "chronos" in row["model"].lower() else ""
        t = timing.get(row["model"])
        if isinstance(t, dict):
            t = t.get("seconds_per_case", t.get("per_case"))
        cells = [html.escape(row["model"]), number(row["mae"], prefix="$"), number(row["rmse"], prefix="$"),
            number(row["high_fare_mae"], prefix="$"),
            (number(100 * row["coverage_80"], 1) + "%") if row["coverage_80"] is not None else "—",
            number(row["interval_width"], prefix="$"), (number(1000*t, 1) + " ms") if t is not None else "—"]
        table_rows.append(f'<tr class="{cls}">' + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    created = datetime.now(timezone.utc).strftime("%d %B %Y %H:%M UTC")
    span = f"{unique.target_timestamp.min():%b %d}–{unique.target_timestamp.max():%b %d, %Y}"
    horizons = ", ".join(str(int(h)) for h in sorted(unique.horizon.unique()))
    high_note = ("High-fare hours exceed each route’s training-only 90th-percentile fare threshold."
        if unique.high_fare.fillna(False).astype(bool).any() else "High-fare subset is unavailable in this run.")
    band = f"[{comparison['day_block_95pct_lower']:+.2f}, {comparison['day_block_95pct_upper']:+.2f}]"
    simulation_html = synthetic_section(output_dir, manifest, summary)
    doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rider fare forecasting · Victor H. Aguiar</title>
<style>
:root{{--ink:#172c43;--muted:#53677c;--accent:#1954a6;--line:#dbe4ed;--soft:#f1f5fa;}}
*{{box-sizing:border-box}}body{{margin:0;background:#e9eef4;color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1160px;margin:30px auto;background:white;border-radius:8px;box-shadow:0 10px 36px #0f294012;overflow:hidden}}
header{{padding:40px 48px 32px;background:#102c49;color:white}}.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#aac4e0}}
h1{{font-size:34px;letter-spacing:-.025em;line-height:1.15;margin:10px 0 12px}}header p{{max-width:870px;color:#d9e5f2;margin:8px 0}}header .byline{{font-size:13px;color:#afc7e0}}
section{{padding:28px 48px;border-bottom:1px solid var(--line)}}h2{{font-size:21px;line-height:1.3;margin:0 0 14px}}h3{{font-size:16px;margin:0 0 8px}}p{{margin:8px 0 14px}}a{{color:var(--accent)}}
.result{{font-size:20px;line-height:1.45;max-width:970px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0 8px}}.card{{background:var(--soft);padding:16px;border-radius:5px}}.card .value{{display:block;font-size:24px;font-weight:650;line-height:1.2}}.card .label{{display:block;color:var(--muted);font-size:12px;margin-top:8px}}
.note{{font-size:13px;color:var(--muted)}}.callout{{border-left:3px solid #698eb6;padding:10px 16px;background:#f4f7fb;margin:16px 0}}
img{{width:100%;height:auto;display:block}}figure{{margin:20px 0 8px}}figcaption{{font-size:12px;color:var(--muted);margin:9px 0 0}}table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:18px}}th,td{{padding:10px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{font-weight:650;background:var(--soft)}}tr.chronos td:first-child{{color:var(--accent);font-weight:650}}.tablewrap{{overflow-x:auto}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:34px}}ul,ol{{padding-left:19px;margin:8px 0}}li{{margin:7px 0}}li span{{color:var(--muted);font-size:13px}}.routes{{font-size:13px}}.sources{{font-size:12px;overflow-wrap:anywhere}}footer{{padding:20px 48px;font-size:12px;color:var(--muted)}}code{{font-size:12px;background:#f1f4f7;padding:2px 4px;border-radius:3px}}details{{margin-top:14px}}summary{{cursor:pointer;color:var(--accent)}}pre{{font-size:11px;background:var(--soft);padding:16px;max-height:380px;overflow:auto;white-space:pre-wrap}}
@media(max-width:760px){{main{{margin:0;border-radius:0}}header,section,footer{{padding:24px}}h1{{font-size:28px}}.cards{{grid-template-columns:1fr 1fr}}.cols{{grid-template-columns:1fr}}}}
@media print{{body{{background:white;font-size:10pt}}main{{margin:0;box-shadow:none;max-width:none}}header{{background:white!important;color:#172c43;border-bottom:2px solid #172c43;padding:18px 24px}}header p,header .byline,.eyebrow{{color:#53677c}}h1{{font-size:25pt}}section{{padding:18px 24px}}figure,.cards,table,.cols{{break-inside:avoid}}details{{display:none}}footer{{padding:14px 24px}}a{{color:inherit}}}}
</style></head><body><main>
<header><div class="eyebrow">Local forecasting demonstration · NYC Uber completed trips</div>
<h1>Forecasting rider fare pressure</h1>
<p>A controlled comparison of time-series foundation forecasting and strong statistical and machine-learning baselines, with economic simulation kept distinct from empirical evidence.</p>
<div class="byline">Victor H. Aguiar · Interview discussion artifact · {created}</div></header>
<section><h2>What the held-out data show</h2><p class="result">{head}</p>
<p>{caveat}</p>
<div class="cards"><div class="card"><span class="value">{number(selected_chronos.mae,prefix='$')}</span><span class="label">Validation-selected Chronos test MAE</span></div>
<div class="card"><span class="value">{len(unique):,}</span><span class="label">Matched route × origin × horizon cases</span></div>
<div class="card"><span class="value">{unique.series_id.nunique()} routes</span><span class="label">Selected using January only</span></div>
<div class="card"><span class="value">{horizons} h</span><span class="label">Forecast horizons · {span}</span></div></div>
<p class="note">A horizon of h hours predicts the pickup-hour bin ending at origin + h, not an instantaneous price. Equal weight per observed route–origin–horizon case. Forecasts share histories and market conditions and are not independent trials. A short NYC panel does not establish universal superiority.</p>
<div class="callout">The outcome is the hourly median realized rider fare before tips, tolls, taxes and fees. It is neither an upfront price quote nor an identified surge premium.</div>
</section>
<section><h2>Benchmark results</h2><figure><img src="{image_data(figs/'forecast_errors.png')}" alt="Held-out mean absolute error by model and forecast horizon"><figcaption>Every model is evaluated on the same observed target hours. Lower prediction error is better.</figcaption></figure>
<div class="tablewrap"><table><thead><tr>{table_head}</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p class="note">{high_note} Interval coverage is empirical, with a nominal target of 80%. Baseline intervals use validation residuals; Chronos intervals are native model quantiles. Their width and calibration are reported rather than presumed equivalent. Latencies, where available, are batched per route–origin (all horizons together) and exclude model loading. Model timings are computational comparisons, not end-to-end service latencies.</p>
<p class="note">Origins are six hours apart. The 1-, 3- and 6-hour scores therefore cover different target-clock hours; a lower 6-hour error does not establish that longer forecasts are inherently easier.</p>
<p><strong>Validation-selected comparison:</strong> {html.escape(val_chronos)} versus {html.escape(baseline)}. Test MAE difference: {comparison['mae_difference_usd']:+.2f} USD; descriptive 95% day-block bootstrap interval: {band} USD ({comparison['blocks']} New York calendar-day blocks). Negative values favor Chronos. Adjacent-day dependence and the short test window limit this uncertainty estimate.</p>
</section>
<section><h2>A fixed example, not a selected success</h2><figure><img src="{image_data(figs/'first_route_first_week.png')}" alt="Observed fare and forecasts for the first route and first test week"><figcaption>{html.escape(example['selection'])} Lines join scored observations; missing low-volume hours are not fabricated.</figcaption></figure></section>
<section><h2>What makes this comparison credible</h2><div class="cols"><div><h3>Information available at the decision time</h3><ul>
<li>Route selection uses January only. Hyperparameter and model-variant selection use chronological validation. Selected learned baselines are refitted once on training plus available validation outcomes before testing.</li>
<li>The completed-trip history is embargoed until each pickup hour ends plus three hours. All methods forecast the same target bins.</li>
<li>Historical fares and served-trip activity are permitted. Future calendar features are known in advance. Future fares, realized trip duration and future trip counts are not predictors.</li>
<li>Missing target hours remain missing. UTC hourly grids handle daylight saving; rider behavior calendar features use New York local time.</li>
<li>No live quotes, private Uber data, driver compensation, or Uber One membership data are used.</li></ul></div><div><h3>Model and sample controls</h3><ul>
<li>The original Chronos-2 variants use frozen weights hash-identical to the October 30, 2025 release, preceding the 2026 outcomes. Any simulation-adapted variant is identified separately.</li>
<li>All forecasts run locally. No fine-tuning on the test window and no cross-learning across different forecast origins.</li>
<li>Statistical seasonality baselines, regularized lagged regression, and histogram gradient boosting provide substantive comparisons.</li>
<li>TLC publishes monthly files with a delay. This is an offline pseudo-online experiment, not a deployable public real-time feed.</li>
<li>Trip mix can change within a route, including unobserved service tiers. Fare predictability is not evidence of a causal pricing effect.</li></ul></div></div></section>
{simulation_html}
<section><h2>How I would take this into a science–engineering partnership</h2><div class="cols"><div><h3>From forecast to a usable system</h3><ol>
<li>Agree on the decision, the data contract and an explicit availability/latency budget with engineering.</li>
<li>Validate quotes and rider-fare decomposition. Monitor freshness, coverage, calibration and errors during high-pressure periods.</li>
<li>Run a shadow forecast with a robust seasonal fallback. Jointly own interfaces, tests, rollout gates and monitoring.</li>
<li>Use controlled evaluation to decide whether the forecast improves the actual product decision, not only offline error.</li></ol></div><div><h3>Economic questions that need internal evidence</h3><ul>
<li>Does anticipating high fares improve conversion, waiting choices, reliability or longer-run retention?</li>
<li>When does partial surge protection for subscribers create incremental value rather than subsidize rides that would happen anyway?</li>
<li>How do competitor prices, availability and alternative products change rider substitution?</li>
<li>Evaluate contribution margin and marketplace reliability alongside rider outcomes. Public completed-trip data alone cannot answer these questions.</li></ul></div></div></section>
<section><h2>Scope and reproducibility</h2><div class="cols"><div><h3>Routes</h3><ul class="routes">{route_text}</ul></div><div><h3>Sources and run files</h3><ul class="sources">
<li><a href="{SOURCE}">NYC TLC official trip-record downloads</a> and <a href="{DICTIONARY}">High Volume FHV data dictionary</a>. Provider filter: <code>HV0003</code>.</li>
<li><a href="{MODEL}">Amazon Chronos-2 model</a>; original weights <a href="https://huggingface.co/amazon/chronos-2/commit/95a9710e2596287d08352589f42634fa5abdf0a7">October 30, 2025 release</a>.</li>
<li><code>data/metadata.json</code>: source hashes, route selection and disjoint exclusion counts.</li>
<li><code>outputs/run_manifest.json</code>: protocol, runtime and pinned model. <code>outputs/test_predictions.parquet</code>: scored forecasts.</li>
<li><code>outputs/report_metrics.json</code>: report calculations and fixed example selection.</li></ul></div></div>
<details><summary>Inspect the saved run manifest</summary><pre>{html.escape(json.dumps(manifest,indent=2))}</pre></details></section>
<footer>Prepared for discussion, not an Uber product claim. No estimate of optimal surge pricing, demand elasticity, or Uber One’s causal effect is made. This report is self-contained and works offline.</footer>
</main></body></html>'''
    (output_dir / "report.html").write_text(doc)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "model_summary": summary.where(pd.notna(summary), None).to_dict("records"),
        "validation_selected_baseline": baseline, "validation_selected_chronos": val_chronos,
        "paired_comparison": comparison, "example": example,
        "unique_test_cases": len(unique), "test_winner_descriptive": str(best.model)}
    # Convert pandas' retained float NaNs to valid JSON nulls.
    def clean(value):
        if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    (output_dir / "report_metrics.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    print(f"Saved {output_dir/'report.html'}")
    print(f"Test winner: {best.model}, MAE ${best.mae:.3f}. {val_chronos}: {difference_text} versus {baseline}.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    build_report(args.output_dir)

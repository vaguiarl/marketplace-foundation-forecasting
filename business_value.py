"""Transparent back-of-envelope BUSINESS SCENARIOS, not an Uber ROI estimate.

No mapping from forecast error improvement to revenue is assumed. Conversion
lift is an unknown causal input against Uber's current production stack, net of
cannibalization and time-shifting. Gross Bookings are not Uber revenue or profit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "https://investor.uber.com/news-events/news/press-release-details/2026/Uber-Announces-Results-for-Second-Quarter-2026/default.aspx"


@dataclass(frozen=True)
class Assumptions:
    quarterly_mobility_gross_bookings_usd: float = 28_988_000_000.
    eligible_share: float = .20
    baseline_quote_to_completed_conversion: float = .60
    incremental_contribution_rate: float = .10
    annual_all_in_program_cost_usd: float = 1_000_000.
    average_fare_change_relative: float = 0.
    price_retention_factor: float = 1.
    extra_subsidy_usd: float = 0.


def validate(a):
    if not all(math.isfinite(v) for v in asdict(a).values()):
        raise ValueError("All assumptions must be finite")
    if a.quarterly_mobility_gross_bookings_usd<=0 or not 0<a.eligible_share<=1:
        raise ValueError("Positive scale and eligible share in (0,1] required")
    if not 0<a.baseline_quote_to_completed_conversion<=1 or not 0<a.incremental_contribution_rate<=1:
        raise ValueError("Conversion and contribution rate must lie in (0,1]")
    if a.annual_all_in_program_cost_usd<0 or a.average_fare_change_relative<=-1:
        raise ValueError("Cost must be nonnegative and average fare cannot fall by 100%")
    if not 0<=a.price_retention_factor<=1 or a.extra_subsidy_usd<0:
        raise ValueError("Price retention must lie in [0,1] and extra subsidy must be nonnegative")


def scenario(conversion_lift_pp, assumptions=Assumptions()):
    """Lift is ABSOLUTE percentage points: 0.05 means 60% becomes 60.05%."""
    validate(assumptions)
    if not math.isfinite(conversion_lift_pp):
        raise ValueError("Conversion lift must be finite")
    a=assumptions
    c=a.baseline_quote_to_completed_conversion
    dc=conversion_lift_pp/100.
    if not 0<=c+dc<=1:
        raise ValueError("Resulting conversion must remain between zero and one")
    annual_scale=4*a.quarterly_mobility_gross_bookings_usd
    eligible=annual_scale*a.eligible_share
    change=eligible*((1+dc/c)*(1+a.average_fare_change_relative)-1)
    relative_volume=dc/c
    volume_contribution=eligible*a.incremental_contribution_rate*relative_volume
    price_contribution=eligible*(1+relative_volume)*a.average_fare_change_relative*a.price_retention_factor
    contribution=volume_contribution+price_contribution-a.extra_subsidy_usd
    return {"conversion_lift_absolute_pp":float(conversion_lift_pp),
            "conversion_lift_fraction":dc,"conversion_after_fraction":c+dc,
            "additional_completions_per_10000_quotes":dc*10000,
            "average_fare_change_relative":a.average_fare_change_relative,
            "annualized_mobility_gross_bookings_scale_usd":annual_scale,
            "eligible_annualized_gross_bookings_scale_usd":eligible,
            "incremental_gross_bookings_usd":change,
            "incremental_volume_contribution_usd":volume_contribution,
            "gross_fare_change_contribution_usd":price_contribution,
            "extra_subsidy_usd":a.extra_subsidy_usd,
            "incremental_contribution_before_program_cost_usd":contribution,
            "annual_all_in_program_cost_usd":a.annual_all_in_program_cost_usd,
            "net_incremental_contribution_after_program_cost_usd":contribution-a.annual_all_in_program_cost_usd}


def break_even_delta_pp(assumptions=Assumptions()):
    validate(assumptions)
    a=assumptions
    eligible=4*a.quarterly_mobility_gross_bookings_usd*a.eligible_share
    denominator=eligible*(a.incremental_contribution_rate+a.average_fare_change_relative*a.price_retention_factor)
    if denominator<=0:
        raise ValueError("No increasing-conversion break-even: marginal contribution slope is nonpositive")
    relative_volume=(a.annual_all_in_program_cost_usd+a.extra_subsidy_usd-eligible*a.average_fare_change_relative*a.price_retention_factor)/denominator
    dc=a.baseline_quote_to_completed_conversion*relative_volume
    if not 0<=a.baseline_quote_to_completed_conversion+dc<=1:
        raise ValueError("Algebraic break-even lies outside feasible conversion range")
    return dc*100


def compute_illustration(seconds_per_forecast=.1276987327917202, units=10_000, interval_minutes=5., worker_price_per_hour=.20):
    if seconds_per_forecast<=0 or units<=0 or interval_minutes<=0 or worker_price_per_hour<0:
        raise ValueError("Compute inputs must be positive, price nonnegative")
    daily=units*24*60/interval_minutes
    yearly=daily*365
    hours=yearly*seconds_per_forecast/3600
    return {"forecast_units":units,"cadence_minutes":interval_minutes,
            "forecasts_per_day":daily,"forecasts_per_year":yearly,
            "measured_amortized_seconds_per_route_origin":seconds_per_forecast,
            "worker_threads":4,"worker_hours_per_year":hours,
            "average_equivalent_continuous_workers":hours/(24*365),
            "assumed_price_per_four_thread_worker_hour_usd":worker_price_per_hour,
            "illustrative_annual_model_compute_usd":hours*worker_price_per_hour,
            "interpretation":"Optimistic extrapolation of local batched M1 CPU timing, not a cloud quote, production capacity plan, latency guarantee, or separate addition to the all-in program budget.",
            "excluded":"Feature I/O, service overhead, peak-load headroom, tail latency, redundancy, online calibration and production integration. Target hardware must be benchmarked."}


def _money(x, units="m"):
    divisor=1e6 if units=="m" else 1e9
    return f"{'−' if x<0 else ''}${abs(x)/divisor:,.3f}{units}"


def _table(headers,rows):
    return "<table><thead><tr>"+"".join(f"<th>{escape(str(h))}</th>" for h in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{escape(str(v))}</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>"


def build(outputdir=ROOT/"outputs/business_value"):
    outputdir=Path(outputdir)
    outputdir.mkdir(parents=True,exist_ok=True)
    a=Assumptions()
    rows=[scenario(x,a) for x in [-.05,0,.01,.05,.10,.50]]
    fares=[scenario(.05,replace(a,average_fare_change_relative=x)) for x in [-.001,-.0001,0,.0001]]
    margins=[{"contribution_rate":m,"break_even_absolute_pp":break_even_delta_pp(replace(a,incremental_contribution_rate=m))} for m in [.05,.10,.15]]
    costs=[{"annual_all_in_program_cost_usd":cost,"break_even_absolute_pp":break_even_delta_pp(replace(a,annual_all_in_program_cost_usd=cost)),
            "net_at_plus_0_05pp_usd":scenario(.05,replace(a,annual_all_in_program_cost_usd=cost))["net_incremental_contribution_after_program_cost_usd"]} for cost in [500_000.,1_000_000.,2_000_000.]]
    compute=compute_illustration()
    model_evidence={"chronos_zero_shot_calendar_count_mae_usd":1.0667870679804452,
                    "gradient_boosting_mae_usd":1.1770075344681823,
                    "relative_mae_reduction":1-1.0667870679804452/1.1770075344681823,
                    "test_cases":1574,"nominal_80pct_interval_empirical_coverage":.7655654383735705,
                    "high_fare_mae_usd":2.550145253268155,
                    "four_week_seasonal_high_fare_mae_usd":2.476765734265734,
                    "source":"../report_metrics.json",
                    "interpretation":"Small NYC observational forecasting benchmark, not Uber production comparison. Chronos has worse high-fare MAE than four-week seasonal and undercovers its nominal80% interval. No revenue conversion follows from the MAE reduction."}
    assumptions={"created_utc":datetime.now(timezone.utc).isoformat(),"inputs":asdict(a),
       "verified_fact":{"quarter":"Q2 2026","metric":"Mobility Gross Bookings","usd":28_988_000_000,"source":SOURCE,"source_publication_date":"2026-08-05"},
       "annualization":"4 × one reported quarter is a scale reference, not a revenue estimate, forecast, guidance or estimate of annual seasonality.",
       "assumed_not_disclosed_by_uber":["Eligible share20%","Baseline quote-to-completed conversion60%","10% contribution rate on incremental VOLUME at baseline fare","Annual all-in program cost$1m","No average gross fare change or extra subsidy in the main scenario","Price retention factor1 in sensitivity only, holding driver payments and other costs fixed; actual taxes/processing may reduce it","Net causal conversion lifts listed in scenarios","$0.20 per four-thread worker-hour"],
       "causal_lift_definition":"Absolute percentage-point increase in completed rides per eligible quote, versus Uber's CURRENT production stack, AFTER netting cannibalization, time-shifting and displaced trips. Not observed in this demo.",
       "unit_economics":"m is assumed contribution on incremental VOLUME at baseline fare, not a revenue take rate, EBITDA/operating margin or profit rate. Price changes have a separate retention factor k. k=1 is illustrative fixed driver payments and other costs, not an Uber estimate; taxes/processing and other pass-through costs may reduce k.",
       "gross_fare_vs_coupon":"Uber defines Gross Bookings without adjusting for consumer discounts/refunds. fare_change here means gross fare per completed trip, NOT coupons. Extra platform-funded coupons/subsidies must be entered separately as extra_subsidy_usd and do not mechanically reduce reported Gross Bookings.",
       "cost_accounting":"All-in annual program cost includes science, engineering, serving/model compute, monitoring, rollout and already-budgeted incentives. Do not add the compute illustration again. extra_subsidy_usd counts only additional coupon/subsidy costs NOT already in program cost. Do not double-count any incentive.",
       "equations":{"eligible_G":"4 * quarterly_Mobility_Gross_Bookings * eligible_share","relative_volume_r":"delta_conversion_fraction/baseline_conversion","delta_GB":"eligible_G * ((1 + r) * (1 + gross_fare_change_p) - 1)","incremental_contribution":"G*m*r + G*(1+r)*p*k - extra_subsidy","net_contribution":"incremental_contribution - annual_all_in_program_cost","break_even_delta_conversion":"baseline_conversion * (cost+extra_subsidy-G*p*k)/(G*(m+p*k)); requires positive denominator and feasible resulting conversion"},
       "model_evidence":model_evidence,"compute_illustration":compute}
    coupon_example=scenario(.05,replace(a,extra_subsidy_usd=100_000.))
    scenarios={"default_scenarios":rows,"fare_sensitivity_at_0_05pp":fares,
               "margin_break_even":margins,"program_cost_sensitivity":costs,
               "extra_coupon_cost_example_100k":coupon_example,
               "default_break_even_absolute_pp":break_even_delta_pp(a)}
    (outputdir/"assumptions.json").write_text(json.dumps(assumptions,indent=2,allow_nan=False)+"\n")
    (outputdir/"scenarios.json").write_text(json.dumps(scenarios,indent=2,allow_nan=False)+"\n")
    headline=scenario(.05,a)
    be=break_even_delta_pp(a)
    lift_table=_table(["Net conversion lift","Extra completions / 10k quotes","Δ Gross Bookings / year","Contribution before cost","Net after $1m cost"],
      [[f"{r['conversion_lift_absolute_pp']:+.2f} pp",f"{r['additional_completions_per_10000_quotes']:+.0f}",_money(r['incremental_gross_bookings_usd']),_money(r['incremental_contribution_before_program_cost_usd']),_money(r['net_incremental_contribution_after_program_cost_usd'])] for r in rows])
    fare_table=_table(["Gross fare change (not coupons)","Δ Gross Bookings / year","Net after $1m cost"],
      [[f"{r['average_fare_change_relative']*100:+.2f}%",_money(r['incremental_gross_bookings_usd']),_money(r['net_incremental_contribution_after_program_cost_usd'])] for r in fares])
    cost_table=_table(["Annual all-in cost","Break-even absolute pp","Net at +0.05 pp"],
      [[_money(r['annual_all_in_program_cost_usd']),f"{r['break_even_absolute_pp']:.4f} pp",_money(r['net_at_plus_0_05pp_usd'])] for r in costs])
    margin_table=_table(["Assumed incremental contribution rate","Break-even absolute pp"],
      [[f"{r['contribution_rate']:.0%}",f"{r['break_even_absolute_pp']:.4f} pp"] for r in margins])
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Uber forecasting: business-value scenarios</title>
<style>:root{{--ink:#17242d;--muted:#4f606d;--line:#d9e0e5;--accent:#126b65;--light:#edf7f5}}*{{box-sizing:border-box}}body{{margin:0;background:#eef1f3;color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}main{{max-width:1100px;margin:auto;padding:24px}}.sheet{{background:white;padding:30px 36px;margin-bottom:22px;border:1px solid var(--line);border-radius:8px}}h1{{font-size:28px;line-height:1.15;margin:8px 0 12px}}h2{{font-size:19px;margin:24px 0 10px}}h3{{font-size:16px;margin:14px 0 8px}}p{{margin:8px 0}}a{{color:#075e9d}}nav{{font-size:13px;margin-bottom:18px;display:flex;gap:22px;flex-wrap:wrap}}.label{{text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:var(--muted)}}.callout{{border-left:4px solid var(--accent);background:var(--light);padding:12px 16px;margin:16px 0}}.warning{{background:#fff6e6;border-left:4px solid #bd820d;padding:10px 14px}}.small{{font-size:12px;color:var(--muted)}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}}th{{background:#edf1f4;text-align:left;font-weight:650}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);vertical-align:top}}td:not(:first-child){{font-variant-numeric:tabular-nums}}code{{font-size:12px;white-space:normal}}ul{{margin:8px 0;padding-left:19px}}li{{margin:4px 0}}footer{{font-size:11px;color:var(--muted);margin-top:14px;border-top:1px solid var(--line);padding-top:9px}}@media(max-width:760px){{main{{padding:8px}}.sheet{{padding:20px}}.columns{{grid-template-columns:1fr}}table{{font-size:11px}}th,td{{padding:6px 4px}}}}@media print{{@page{{size:A4 landscape;margin:10mm}}body{{background:white;font-size:10.5px}}main{{padding:0;max-width:none}}.sheet{{border:0;border-radius:0;padding:0;margin:0;break-after:page;page-break-after:always}}.sheet:last-child{{break-after:auto;page-break-after:auto}}h1{{font-size:21px}}h2{{font-size:15px;margin:13px 0 6px}}h3{{font-size:12px}}table{{font-size:10px}}th,td{{padding:5px 7px}}.small,footer{{font-size:9px}}nav{{font-size:10px;margin-bottom:8px}}.callout{{margin:9px 0;padding:8px 12px}}p{{margin:5px 0}}.columns{{gap:20px}}}}
</style><main><section class="sheet"><nav><a href="../report.html">NYC forecasting evidence</a><a href="../decision_demo/report.html">Price-action and ETA prototype</a><a href="assumptions.json">All assumptions</a><a href="scenarios.json">Exact scenario calculations</a></nav>
<div class="label">Interview discussion · October 26, 2026 · USD</div><h1>How small would a real business improvement need to be?</h1>
<p>This is a transparent investment screen. The causal business lift is unknown. The hurdle is improvement over <strong>Uber’s current production stack</strong>, not over a public-data baseline.</p>
<div class="callout"><strong>Illustrative case:</strong> a +0.05 percentage-point lift in quote-to-completed conversion—60.00% to 60.05%, or 5 additional completions per 10,000 eligible quotes—produces {_money(headline['incremental_contribution_before_program_cost_usd'])} annual incremental contribution before program costs and <strong>{_money(headline['net_incremental_contribution_after_program_cost_usd'])} after an assumed $1m all-in program cost</strong>. This assumes no average fare change.</div>
<div class="columns"><div><h3>One verified scale input</h3><p>Uber reported <strong>$28.988bn Mobility Gross Bookings</strong> in Q2 2026. Multiplying by four gives a $115.952bn annualized scale reference. It is not a forecast, annual guidance, Uber revenue, or profit.</p><p class="small"><a href="{SOURCE}">Source: Uber Q2 2026 results, August 5, 2026; Results by Offering and Segment.</a></p></div><div><h3>Analyst assumptions, not Uber disclosures</h3><p>20% eligible bookings share → $23.1904bn eligible scale; 60% baseline conversion; 10% incremental contribution on incremental bookings; $1m annual all-in cost; zero average fare change. Net causal lift is after cannibalization, displaced trips and time-shifting.</p></div></div>
<h2>Annual business scenarios at unchanged average fares</h2>{lift_table}
<p><strong>Break-even: +{be:.4f} absolute percentage points</strong>, or {be*100:.2f} additional completed rides per 10,000 eligible quotes, under the stated assumptions. Zero lift loses the full $1m program cost; negative lift is worse.</p>
<p class="small">Formula: G = 4 × quarterly Mobility Gross Bookings × eligible share; r = Δc/c. ΔGB = G × [(1+r)(1+p)−1]. Net contribution = Gmr + G(1+r)pk − extra subsidies − program cost. Here p=0, subsidies=0, m=10%; Δc is a fraction (+0.05 pp = +0.0005). Fixed quote opportunities and comparable eligible fare mix are simplifying assumptions.</p>
<h2>What the demo establishes—and what it does not</h2><p>On 1,574 NYC held-out forecasting cases, zero-shot Chronos-2 with calendar/count features had MAE <strong>$1.067 versus $1.177 for gradient boosting</strong>, a {model_evidence['relative_mae_reduction']:.2%} reduction. This does not imply any particular conversion, revenue or profit lift. High-fare MAE was $2.550 versus $2.477 for the four-week seasonal baseline; nominal 80% interval coverage was only 76.6%.</p>
<footer>Evidence is a small observational NYC forecasting benchmark. Its target is realized rider fare, not price elasticity or Uber production performance. The synthetic decision prototype supplies model-dependent response scenarios, not empirical causal effects.</footer></section>
<section class="sheet"><div class="label">Sensitivity and deployment economics</div><h1>Protect against a false-positive business case</h1>
<div class="columns"><div><h2>Price cuts can erase conversion value</h2><p>Hold net conversion lift at +0.05 pp. Apply gross fare changes to all eligible completions. Volume contributes at m=10%; fare changes use a separate illustrative retention k=100%, holding driver payments and other costs fixed.</p>{fare_table}<p class="small">k is not an Uber estimate; taxes, processing and payout responses require actual unit economics. Gross Bookings exclude adjustments for consumer discounts/refunds: coupons are not p. A separate $100k coupon cost reduces the unchanged-fare example’s net to {_money(coupon_example['net_incremental_contribution_after_program_cost_usd'])}, without mechanically changing Gross Bookings. Never count an incentive twice.</p></div><div><h2>Program-cost sensitivity</h2>{cost_table}<p class="small">At $2m all-in annual cost, +0.05 pp is slightly negative. Compute below is already within this program budget, not an extra subtraction.</p><h3>Volume-contribution sensitivity at $1m cost</h3>{margin_table}<p class="small">These 5–15% rates are assumptions about contribution on additional volume at baseline fare. They are not Uber’s revenue take rate, EBITDA margin, operating margin or profit rate.</p></div></div>
<h2>Separate compute-capacity illustration</h2><p>Measured local batched Chronos-2 calendar/count inference: <strong>{compute['measured_amortized_seconds_per_route_origin']:.4f} seconds per route-origin</strong> on M1 CPU with four threads, producing all three scored horizons. This is amortized throughput, not per-request tail latency.</p>
<div class="columns"><div>{_table(['Assumed workload / derived capacity','Value'],[['Forecast units / cadence','10,000 / every 5 minutes'],['Forecasts per day / year',f"{compute['forecasts_per_day']:,.0f} / {compute['forecasts_per_year']:,.0f}"],['Four-thread worker-hours per year',f"{compute['worker_hours_per_year']:,.0f}"],['Average continuous worker equivalents',f"{compute['average_equivalent_continuous_workers']:.2f}"]])}</div><div><div class="warning"><strong>Not a production budget or cloud quote.</strong> At an assumed $0.20 per four-thread worker-hour, arithmetic model compute is approximately <strong>${compute['illustrative_annual_model_compute_usd']:,.0f}/year</strong>. This optimistic extrapolation excludes feature I/O, service overhead, tail latency, redundancy and peak-load headroom. Benchmark the target hardware and batch shape before using it for capacity planning.</div></div></div>
<h2>The experiment that would justify the investment</h2><p>Compare a forecast-informed rider-pricing policy with the current production policy in a marketplace-aware randomized rollout. Estimate net completed rides per eligible quote and incremental contribution, while measuring realized waits, abandonment, cancellations, rider retention and supply-side spillovers. Include time-shifting and cannibalization. Do not choose the deployment decision from forecast MAE alone.</p>
<p><strong>Decision rule:</strong> establish an economically credible causal lift and guardrail performance, then compare uncertainty around incremental contribution with the full program cost. Neither zero-shot accuracy nor a synthetic policy result substitutes for that test.</p>
<footer>Sources: <a href="{SOURCE}">Uber Q2 2026 results</a>; local <a href="../report_metrics.json">forecast metrics</a> and <a href="../run_manifest.json">timing manifest</a>. All unverified business and capacity inputs are explicitly analyst assumptions. This report is an interview planning illustration, not an investment or earnings forecast.</footer></section></main></html>'''
    (outputdir/"report.html").write_text(html)
    return {"report":str(outputdir/"report.html"),"default_break_even_pp":be,
            "net_at_0_05pp":headline["net_incremental_contribution_after_program_cost_usd"],
            "compute_illustration_usd":compute["illustrative_annual_model_compute_usd"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/business_value")
    print(json.dumps(build(parser.parse_args().output),indent=2))

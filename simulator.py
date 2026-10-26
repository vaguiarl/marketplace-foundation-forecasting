"""Small economic simulator for synthetic-only Chronos adaptation.

Inspired by the economic importance of price/waiting-time heterogeneity, not a
replication of Buchholz et al. (2025). Unknown structural parameters are varied,
not estimated. Driver payments are not controlled; vehicle capacity is exogenous.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "synthetic_protocol.json"
CALIBRATION_CUTOFF = pd.Timestamp("2026-02-15 02:00:00")
CALENDAR_KEYS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")
POLICIES = ("flat", "reactive", "anticipatory")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def calibrate_scales(frame):
    """Only dimensional scales from strictly earlier training rows enter."""
    train = frame.loc[pd.to_datetime(frame.timestamp) < CALIBRATION_CUTOFF].copy()
    if train.empty:
        raise ValueError("No permitted training rows before calibration cutoff")
    fares = train.loc[np.isfinite(train.fare) & train.fare.gt(0), "fare"]
    counts = train.loc[np.isfinite(train["count"]) & train["count"].gt(0), "count"]
    if len(fares) < 100 or len(counts) < 100:
        raise ValueError("Insufficient training-only observations for dimensional scales")
    return {"median_fare_usd": float(fares.median()),
            "median_positive_completed_count": float(counts.median()),
            "calibration_cutoff_utc_exclusive": str(CALIBRATION_CUTOFF),
            "max_used_timestamp_utc": str(train.timestamp.max()),
            "rows_used": len(train), "observed_fare_rows": len(fares),
            "interpretation": "Dimensional anchors only; fare anchor is NOT an estimated no-surge fare, and count is NOT unconstrained demand."}


def choice_probability(price, quoted_eta, base_anchor, base_eta, alpha, beta, intercept):
    utility = np.asarray(intercept) - np.asarray(alpha) * (price / base_anchor - 1.0) - np.asarray(beta) * (quoted_eta - base_eta)
    return 1.0 / (1.0 + np.exp(-np.clip(utility, -40, 40)))


def sample_parameters(rng, scales):
    fare = scales["median_fare_usd"] * rng.uniform(.55, 1.8)
    count = scales["median_positive_completed_count"] * rng.uniform(.65, 1.6)
    return {
        "base_anchor": float(fare), "base_count": float(count),
        "base_potential": float(count * rng.uniform(1.5, 2.8)),
        "base_capacity": float(count * rng.uniform(1.05, 1.8)),
        "base_eta": float(rng.uniform(3, 8)),
        "alpha": ([.55, 1.3, 2.5] * np.array(rng.uniform(.5, 1.8))).tolist(),
        "beta": ([.04, .11, .22] * np.array(rng.uniform(.5, 1.7))).tolist(),
        "intercept": rng.uniform(1.1, 2.4, size=3).tolist(),
        "type_weights": rng.dirichlet([2., 2., 2.]).tolist(),
        "gain": float(rng.uniform(.15, 1.9)),
        "smoothing": float(rng.uniform(.12, .85)),
        "reaction_lag": int(rng.integers(1, 5)),
        "cap": float(rng.uniform(1.25, 3.5)),
        "rho": float(rng.uniform(.35, .95)),
        "shock_sd": float(rng.uniform(.04, .30)),
        "capacity_adjustment": float(rng.uniform(.08, .40)),
        "abandonment": float(rng.uniform(.35, .90)),
        "matching_efficiency": float(rng.uniform(.75, .98)),
        "eta_gain": float(rng.uniform(1, 8)),
        "morning_amplitude": float(rng.uniform(.15, .7)),
        "evening_amplitude": float(rng.uniform(.25, .9)),
        "night_amplitude": float(rng.uniform(0, .7)),
        "capacity_phase": float(rng.uniform(-2, 3)),
        "observation_noise": float(rng.uniform(.02, .15)),
    }


def make_exogenous(parameters, seed, length=1008, start="2025-01-06 05:00", common_shocks=None):
    """Draw a path once; all policies share it. Future shock draws are hidden."""
    if length < 2:
        raise ValueError("Need at least two simulated hours")
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=length, freq="h")
    local = idx.tz_localize("UTC").tz_convert("America/New_York")
    hour = local.hour.to_numpy()
    weekend = (local.dayofweek.to_numpy() >= 5).astype(float)
    periodic_peak = lambda h, center, width: np.exp(-np.minimum(abs(h-center), 24-abs(h-center))**2 / (2*width**2))
    seasonal = (.45 + parameters["morning_amplitude"] * periodic_peak(hour, 8, 2.5)
                + parameters["evening_amplitude"] * periodic_peak(hour, 18, 3)
                + parameters["night_amplitude"] * (1+weekend) * periodic_peak(hour, 23, 2.7))
    seasonal /= seasonal.mean()
    capacity_schedule = .75 + .4 * periodic_peak(hour, 17+parameters["capacity_phase"], 5)
    capacity_schedule /= capacity_schedule.mean()
    # Announced events are calendar information. Unannounced events remain a
    # potential-demand surprise and are never inserted in the price forecast.
    event = np.ones(length)
    announced = np.ones(length)
    for _ in range(max(1, length//120)):
        t = int(rng.integers(0, length))
        duration = int(rng.integers(parameters.get("event_duration_low", 2), parameters.get("event_duration_high", 13)))
        factor = float(rng.uniform(1.3, 3.3))
        event[t:min(length,t+duration)] *= factor
        if rng.random() < parameters.get("announced_probability", .5):
            announced[t:min(length,t+duration)] *= factor
    innovation = rng.standard_normal(length)
    if common_shocks is not None:
        if len(common_shocks) != length:
            raise ValueError("Common-shock path has wrong length")
        innovation = .6*np.asarray(common_shocks) + .8*innovation
    state = np.zeros(length)
    forecast_state = np.zeros(length)
    capacity = np.zeros(length)
    cstate = 0.
    previous_capacity = parameters["base_capacity"]
    for t in range(length):
        previous_state = state[t-1] if t else 0.
        forecast_state[t] = parameters["rho"] * previous_state
        state[t] = forecast_state[t] + parameters["shock_sd"] * innovation[t]
        cstate = .8*cstate + .08*rng.standard_normal()
        desired_capacity = parameters["base_capacity"]*capacity_schedule[t]*np.exp(cstate)
        previous_capacity += parameters["capacity_adjustment"]*(desired_capacity-previous_capacity)
        capacity[t] = max(1., previous_capacity)
    mean = parameters["base_potential"] * seasonal * event * np.exp(np.clip(state, -2, 2))
    potential = rng.poisson(np.minimum(mean, 1200)).astype(np.int32)
    # Forecast uses only the last observed potential-arrival count plus known
    # calendar/event information. It cannot inspect the latent AR state. In a
    # real platform this would require session/quote-arrival logs, absent in TLC.
    forecast_potential = np.zeros(length)
    for t in range(length):
        past_residual = (np.log((potential[t-1]+.5) /
            max(parameters["base_potential"]*seasonal[t-1]*announced[t-1], .1)) if t else 0.)
        forecast_potential[t] = parameters["base_potential"] * seasonal[t] * announced[t] * np.exp(
            np.clip(parameters["rho"]*past_residual, -2, 2))
    max_people = int(max(1, potential.max()))
    exo = {
        "timestamp_ns": idx.astype("datetime64[ns]").asi8,
        "potential": potential, "forecast_potential": forecast_potential,
        "capacity": capacity, "demand_innovation": innovation,
        "choice_uniform": rng.random((length, max_people), dtype=np.float32),
        "type_id": rng.choice(3, size=(length, max_people), p=parameters["type_weights"]).astype(np.int8),
        "abandon_uniform": rng.random(length), "service_uniform": rng.random(length),
        "fare_noise": rng.standard_normal(length),
        "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
        "dow_sin": np.sin(2*np.pi*local.dayofweek.to_numpy()/7),
        "dow_cos": np.cos(2*np.pi*local.dayofweek.to_numpy()/7), "weekend": weekend,
    }
    return exo


def _take_fifo(queue, amount):
    """Remove exactly amount from a queue of [count, locked_quote] cohorts."""
    taken = []
    remaining = int(amount)
    while remaining and queue:
        count, price = queue[0]
        n = min(remaining, count)
        taken.append((n, price))
        remaining -= n
        if n == count:
            queue.popleft()
        else:
            queue[0][0] -= n
    if remaining:
        raise AssertionError("Attempted to remove more riders than queue contains")
    return taken


def cohort_median(cohorts):
    if not cohorts:
        return np.nan
    ordered = sorted(cohorts, key=lambda x: x[1])
    n = sum(x[0] for x in ordered)
    cumulative = np.cumsum([x[0] for x in ordered])
    prices = np.array([x[1] for x in ordered])
    a = prices[np.searchsorted(cumulative, (n-1)//2 + 1)]
    b = prices[np.searchsorted(cumulative, n//2 + 1)]
    return float((a+b)/2)


def simulate_path(exogenous, parameters, policy="reactive", min_count=10):
    """Forward-only policy rollout with exact request/queue accounting."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy}")
    n = len(exogenous["potential"])
    names = ("fare", "count", "posted_fare", "quoted_eta", "potential", "requested", "abandoned",
             "queue_start", "queue_end", "gross_revenue", "capacity", "service_budget", "acceptance_probability")
    out = {k: np.zeros(n, dtype=np.float64) for k in names}
    out["fare"][:] = np.nan
    queue = deque()
    multiplier = 1.
    for t in range(n):
        q0 = sum(x[0] for x in queue)
        abandoned = min(q0, int(np.floor(q0*parameters["abandonment"] + exogenous["abandon_uniform"][t])))
        _take_fifo(queue, abandoned)
        q = q0-abandoned
        capacity = float(exogenous["capacity"][t])
        effective_capacity = capacity*parameters["matching_efficiency"]
        baseline_prob = float(np.dot(parameters["type_weights"], choice_probability(
            parameters["base_anchor"], parameters["base_eta"], parameters["base_anchor"],
            parameters["base_eta"], parameters["alpha"], parameters["beta"], parameters["intercept"])))
        predictable_requests = float(exogenous["forecast_potential"][t])*baseline_prob
        eta = float(np.clip(parameters["base_eta"] + 60*q/max(effective_capacity, 1.)
                            + parameters["eta_gain"]*max(predictable_requests/max(effective_capacity,1.)-1.,0), 1, 90))
        lag = int(parameters["reaction_lag"])
        if policy == "flat":
            multiplier = 1.
        else:
            if policy == "anticipatory":
                signal = predictable_requests
            else:
                signal = out["requested"][t-lag] if t >= lag else parameters["base_count"]
            pressure = (signal+q)/max(effective_capacity,1.)
            desired = float(np.clip(1+parameters["gain"]*max(pressure-.95,0), 1, parameters["cap"]))
            multiplier += parameters["smoothing"]*(desired-multiplier)
        price = parameters["base_anchor"]*multiplier
        probabilities = choice_probability(price, eta, parameters["base_anchor"], parameters["base_eta"],
                                            parameters["alpha"], parameters["beta"], parameters["intercept"])
        potential = int(exogenous["potential"][t])
        typ = exogenous["type_id"][t,:potential]
        requested = int(np.count_nonzero(exogenous["choice_uniform"][t,:potential] < probabilities[typ]))
        if requested:
            queue.append([requested, price])
        budget = max(0, int(np.floor(effective_capacity+exogenous["service_uniform"][t])))
        served = min(q+requested, budget)
        completed_cohorts = _take_fifo(queue, served)
        end_q = sum(x[0] for x in queue)
        if q0+requested != abandoned+served+end_q:
            raise AssertionError("Queue flow conservation failed")
        for k,v in {"count": served, "posted_fare": price, "quoted_eta": eta, "potential": potential,
                    "requested": requested, "abandoned": abandoned, "queue_start": q0, "queue_end": end_q,
                    "gross_revenue": sum(c*p for c,p in completed_cohorts), "capacity": capacity,
                    "service_budget": budget, "acceptance_probability": np.dot(parameters["type_weights"], probabilities)}.items():
            out[k][t] = v
        if served >= min_count:
            noise = parameters["observation_noise"]*exogenous["fare_noise"][t]/np.sqrt(served)
            out["fare"][t] = max(.01, cohort_median(completed_cohorts)*np.exp(noise))
    for key in ("timestamp_ns", *CALENDAR_KEYS):
        out[key] = np.array(exogenous[key], copy=True)
    return out


def policy_comparison(exogenous, parameters):
    paths, rows = {}, []
    for policy in POLICIES:
        path = simulate_path(exogenous, parameters, policy)
        paths[policy] = path
        potential = path["potential"].sum()
        requested = path["requested"].sum()
        completed = path["count"].sum()
        rows.append({"policy": policy, "potential_riders": int(potential), "requests": int(requested),
                     "completed": int(completed), "abandoned_after_request": int(path["abandoned"].sum()),
                     "ending_queue": int(path["queue_end"][-1]),
                     "requested_over_potential": float(requested/max(potential,1)),
                     "completed_over_potential": float(completed/max(potential,1)),
                     "abandoned_over_requests": float(path["abandoned"].sum()/max(requested,1)),
                     "mean_quoted_eta_minutes_weighted_by_potential": float(np.sum(path["quoted_eta"]*path["potential"])/potential) if potential else None,
                     "gross_rider_fare_revenue_usd_before_costs": float(path["gross_revenue"].sum()),
                     "percent_potential_not_completed_by_window_end": float(100*(1-completed/max(potential,1)))})
    return paths, rows


def _corpus(scales, seed, episodes, length):
    rng = np.random.default_rng(seed)
    samples, params_meta = [], []
    for episode in range(episodes):
        common = rng.standard_normal(length)
        start = datetime(2025,1,6,5) + timedelta(days=int(rng.integers(0,365)))
        for route in range(2):
            parameters = sample_parameters(rng, scales)
            path_seed = int(rng.integers(0, 2**31-1))
            policy = str(rng.choice(POLICIES, p=[.1,.55,.35]))
            exo = make_exogenous(parameters, path_seed, length, start, common)
            sample = simulate_path(exo, parameters, policy)
            samples.append(sample)
            params_meta.append({"episode": episode, "route": route, "path_seed": path_seed,
                               "policy": policy, "parameters": parameters})
    corpus = {k: np.stack([s[k] for s in samples]).astype(np.int64 if k=="timestamp_ns" else np.float32)
              for k in samples[0]}
    corpus["served_count"] = corpus["count"].copy()
    corpus["series_id"] = np.array([f"synthetic_{seed}_{i//2:03d}_{i%2}" for i in range(len(samples))])
    return corpus, params_meta


def stress_comparisons(scales, seed, outputdir):
    """Four predeclared, unselected worlds; each policy sees identical shocks."""
    names = ("low_price_sensitivity", "high_price_sensitivity", "short_unannounced_shocks", "persistent_unannounced_shocks")
    rows, arrays = [], {}
    for i, name in enumerate(names):
        world_seed = seed+3_000_101+i*1009
        parameters = sample_parameters(np.random.default_rng(world_seed), scales)
        if name == "low_price_sensitivity":
            parameters["alpha"] = [.3, .65, 1.25]
        elif name == "high_price_sensitivity":
            parameters["alpha"] = [1.1, 2.6, 5.0]
        elif name == "short_unannounced_shocks":
            parameters.update(event_duration_low=2,event_duration_high=5,announced_probability=0.,rho=.4)
        else:
            parameters.update(event_duration_low=8,event_duration_high=13,announced_probability=0.,rho=.95)
        exo = make_exogenous(parameters, world_seed+17, length=168)
        paths, summary = policy_comparison(exo, parameters)
        rows.append({"world":name,"seed":world_seed,"parameters":parameters,"rows":summary})
        for policy,path in paths.items():
            for key,value in path.items():
                arrays[f"{name}__{policy}__{key}"] = value
    np.savez_compressed(outputdir / "stress_policy_paths.npz", **arrays)
    (outputdir / "stress_policy_comparison.json").write_text(json.dumps({
        "label":"Four predeclared synthetic stress illustrations; never used to select a policy or train the forecasting model",
        "anticipatory_policy":"Calendar plus lagged observed potential-arrival heuristic, not a Chronos policy or an optimal policy. Requires internal arrival information absent from TLC.",
        "eta":"Model-generated quoted ETA proxy, not measured realized wait",
        "worlds":rows},indent=2,allow_nan=False)+"\n")


def generate_corpus(panel_path, outputdir, seed=20261026, episodes=96, length=1008):
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    if episodes < 1 or length < 24:
        raise ValueError("Need positive episodes and at least one day")
    scales = calibrate_scales(pd.read_parquet(panel_path))
    corpus, parameters = _corpus(scales, seed, episodes, length)
    corpus_path = outputdir / "corpus.npz"
    np.savez_compressed(corpus_path, **corpus)
    validation_seed = seed+1_000_003
    validation, validation_parameters = _corpus(scales, validation_seed, 16, length)
    validation_path = outputdir / "validation.npz"
    np.savez_compressed(validation_path, **validation)
    # Illustration uses an independent, deliberately unselected seed/path.
    prng = np.random.default_rng(seed+2_000_033)
    demo_parameters = sample_parameters(prng, scales)
    exo = make_exogenous(demo_parameters, seed+2_000_039, length=168)
    demo_paths, comparison = policy_comparison(exo, demo_parameters)
    np.savez_compressed(outputdir / "policy_paths.npz", **{
        f"{policy}__{key}": value for policy,path in demo_paths.items() for key,value in path.items()})
    (outputdir / "policy_comparison.json").write_text(json.dumps({
        "label": "Illustrative model-dependent simulation; not a real-world policy effect or optimization result",
        "parameters": demo_parameters, "rows": comparison,
        "same_exogenous_capacity_and_potential_paths": True,
        "same_choice_uniforms": True,
        "anticipatory_policy": "Calendar plus lagged observed potential-arrival heuristic, not Chronos or an optimized policy; requires internal arrival information absent from TLC.",
        "eta_is_model_generated_quote_not_realized_wait": True,
        "revenue_is_not_profit": True,
        "end_of_window_unserved_includes_pending_requests": True}, indent=2) + "\n")
    stress_comparisons(scales, seed, outputdir)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "episodes": episodes, "routes_per_episode": 2, "hours": length,
        "corpus_shape": list(corpus["fare"].shape), "calibration": scales,
        "finite_fare_fraction": float(np.isfinite(corpus["fare"]).mean()),
        "real_panel_path": str(Path(panel_path).resolve()), "real_panel_sha256": sha256(panel_path),
        "real_panel_hash_note": "Hash identifies source file, but only rows before explicit training cutoff influence scale calibration.",
        "protocol_path": str(PROTOCOL_PATH), "protocol_sha256": sha256(PROTOCOL_PATH),
        "simulator_sha256": sha256(__file__),
        "corpus_sha256": sha256(corpus_path), "validation_sha256": sha256(validation_path),
        "validation_seed": validation_seed, "validation_episodes": 16,
        "training_parameters": parameters, "synthetic_validation_parameters": validation_parameters,
        "model_parameters_estimated_from_data": False,
        "real_validation_test_outcomes_used": False,
        "calendar": "Synthetic timestamps are calendar coordinates, not claims of observed historical events. UTC storage, NYC-local calendar features.",
        "columns": {"fare": "Synthetic completed-trip median, NaN below10 completions",
                    "count": "Completed synthetic rides", "served_count": "Alias of count",
                    "posted_fare": "Current offered common rider price; not a training target substitute",
                    "timestamp_ns": "UTC epoch nanoseconds", "calendar": list(CALENDAR_KEYS)},
        "limitations": json.loads(PROTOCOL_PATH.read_text())["reporting_limits"]}
    manifest_path = outputdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"corpus_path": str(corpus_path), "validation_path": str(validation_path),
            "manifest_path": str(manifest_path), "shape": manifest["corpus_shape"],
            "finite_fare_fraction": manifest["finite_fare_fraction"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=ROOT / "data/panel.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/synthetic")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--length", type=int, default=1008)
    args = parser.parse_args()
    print(json.dumps(generate_corpus(args.panel, args.output, args.seed, args.episodes, args.length), indent=2), flush=True)


if __name__ == "__main__":
    main()

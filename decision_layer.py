"""Map a rider PRICE ACTION and a forecast state to modeled ETA/conversion.

Price is an input/action here, not a forecasting target. A forecaster supplies
potential rider arrivals and independently modeled service capacity. This is a
one-period illustrative queue/choice calculation, not an estimated Uber demand
system or an empirical causal forecast. Its parameters are deliberately explicit.

For positive capacity S, requests D(w,a) fall with quoted wait w and price
multiplier a. The quoted-wait map g(w)=base_eta+gain*max((Q+D(w,a))/S-.6,0)
is continuous and nonincreasing (and capped). Therefore w-g(w) is strictly
increasing and has a unique zero between base_eta and max_eta. Bisection solves
that zero without assuming that a policy or foundation model is superior.

All arrivals, service capacity, and queue counts use a common one-hour period.
ETA is a model-generated quote proxy in minutes, not measured realized wait.
Existing requests have FIFO service priority. No abandonment or capacity response
to rider price is assumed. Consequently, a higher price cannot magically create
vehicles or guarantee more completed rides in this fixed-capacity one-period map.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ResponseConfig:
    base_fare_anchor: float = 10.92
    base_eta_minutes: float = 5.0
    congestion_gain_minutes: float = 8.0
    load_threshold: float = 0.6
    max_eta_minutes: float = 120.0
    alpha: tuple[float, ...] = (.55, 1.3, 2.5)
    beta: tuple[float, ...] = (.04, .11, .22)
    intercept: tuple[float, ...] = (1.8, 1.8, 1.8)
    weights: tuple[float, ...] = (1/3, 1/3, 1/3)
    tolerance_minutes: float = 1e-9
    max_iterations: int = 100


WARNINGS = (
    "Illustrative structural priors, not estimates from Buchholz et al. or Uber.",
    "Quoted ETA is a model proxy, not a validated forecast of realized waiting time.",
    "Potential arrivals are not completed-trip counts; NYC TLC does not observe potential arrivals.",
    "Capacity is exogenous. No rider-price-to-driver-pay or supply response is assumed.",
    "An action selected under forecast inputs may violate the modeled ETA constraint when evaluated under realized inputs.",
    "This layer does not identify a real-world treatment effect, profit optimum, consumer welfare, or Uber One effect.",
)


def _config(config):
    cfg = ResponseConfig() if config is None else (ResponseConfig(**config) if isinstance(config, dict) else config)
    if not isinstance(cfg, ResponseConfig):
        raise TypeError("config must be ResponseConfig, a matching dict, or None")
    positive = (cfg.base_fare_anchor,cfg.base_eta_minutes,cfg.max_eta_minutes,cfg.tolerance_minutes)
    if not all(isfinite(x) and x>0 for x in positive):
        raise ValueError("Fare anchor, ETA bounds and tolerance must be finite and positive")
    if cfg.max_eta_minutes < cfg.base_eta_minutes:
        raise ValueError("Maximum ETA must be at least base ETA")
    if not isfinite(cfg.congestion_gain_minutes) or cfg.congestion_gain_minutes < 0:
        raise ValueError("Congestion gain must be finite and nonnegative")
    if not isfinite(cfg.load_threshold) or cfg.load_threshold < 0:
        raise ValueError("Load threshold must be finite and nonnegative")
    if cfg.max_iterations < 1:
        raise ValueError("Need positive iteration limit")
    alpha,beta,intercept,weights = map(lambda x:np.asarray(x,dtype=float),
                                      (cfg.alpha,cfg.beta,cfg.intercept,cfg.weights))
    if not (alpha.ndim==beta.ndim==intercept.ndim==weights.ndim==1
            and len(alpha)==len(beta)==len(intercept)==len(weights) and len(alpha)>0):
        raise ValueError("Preference vectors must be nonempty, one-dimensional and equally sized")
    if not all(np.isfinite(x).all() for x in (alpha,beta,intercept,weights)):
        raise ValueError("Preference vectors must be finite")
    if np.any(alpha<=0) or np.any(beta<=0) or np.any(weights<0) or weights.sum()<=0:
        raise ValueError("Price/time sensitivities positive; type weights nonnegative and nonzero")
    return cfg


def request_probability(quoted_eta_minutes, price_multiplier, config=None):
    """Heterogeneous logistic take-up at a supplied common price and ETA."""
    cfg = _config(config)
    eta,a = float(quoted_eta_minutes),float(price_multiplier)
    if not isfinite(eta) or eta<0 or not isfinite(a) or a<=0:
        raise ValueError("ETA must be finite/nonnegative and price multiplier finite/positive")
    utility = np.asarray(cfg.intercept) - np.asarray(cfg.alpha)*(a-1.) - np.asarray(cfg.beta)*(eta-cfg.base_eta_minutes)
    probabilities = 1/(1+np.exp(-np.clip(utility,-40,40)))
    weights = np.asarray(cfg.weights)
    return float(np.dot(weights/weights.sum(),probabilities))


def solve_response(potential, capacity, queue, price_multiplier, config=None):
    """Expected response to ONE action using ONLY the supplied forecast/state.

    Negative scalar forecasts are clamped to zero and explicitly flagged.
    NaN/infinite states are rejected. The same function evaluates a chosen action
    against realized inputs; it never retrains or reselects the action itself.
    """
    cfg = _config(config)
    raw = {"potential":float(potential),"capacity":float(capacity),"queue":float(queue)}
    if not all(isfinite(x) for x in raw.values()):
        raise ValueError("State inputs must be finite")
    A,S,Q = (max(0.,raw[k]) for k in ("potential","capacity","queue"))
    a = float(price_multiplier)
    if not isfinite(a) or a<=0:
        raise ValueError("Price multiplier must be finite and positive")

    def wait_map(wait):
        if S==0:
            return cfg.max_eta_minutes
        demand = A*request_probability(wait,a,cfg)
        return min(cfg.max_eta_minutes,cfg.base_eta_minutes+cfg.congestion_gain_minutes*max((Q+demand)/S-cfg.load_threshold,0.))

    low,high = cfg.base_eta_minutes,cfg.max_eta_minutes
    if S==0:
        wait,iterations = cfg.max_eta_minutes,0
    elif abs(low-wait_map(low))<=cfg.tolerance_minutes:
        wait,iterations = low,0
    elif abs(high-wait_map(high))<=cfg.tolerance_minutes:
        wait,iterations = high,0
    else:
        for iterations in range(1,cfg.max_iterations+1):
            wait = (low+high)/2
            gap = wait-wait_map(wait)
            if abs(gap)<=cfg.tolerance_minutes:
                break
            if gap>0:
                high=wait
            else:
                low=wait
        else:
            raise RuntimeError("ETA fixed-point bisection failed to converge")
    probability = request_probability(wait,a,cfg)
    D = A*probability
    served_existing = min(Q,S)
    served_new = min(D,max(S-Q,0.))
    served_total = served_existing+served_new
    queue_end = max(0.,Q+D-served_total)
    gap = wait-wait_map(wait)
    return {
        "price_multiplier":a,"price_usd":cfg.base_fare_anchor*a,
        "potential":A,"capacity":S,"queue_start":Q,
        "quoted_eta_proxy_minutes":float(wait),"expected_requests":float(D),
        "request_rate":float(D/A) if A>0 else 0.,
        "new_rider_completions":float(served_new),
        "new_rider_conversion":float(served_new/A) if A>0 else 0.,
        "existing_queue_completions":float(served_existing),
        "total_completions":float(served_total),"queue_end":float(queue_end),
        "new_unserved_requests":float(D-served_new),
        "model_acceptance_probability":probability,
        "inputs_clipped_to_zero":{k:raw[k] for k in raw if raw[k]<0},
        "zero_capacity":S==0,"fixed_point_residual_minutes":float(gap),
        "fixed_point_iterations":iterations,
        "eta_is_model_proxy":True,
    }


def response_surface(potential, capacity, queue, price_multipliers: Iterable[float] | None=None, config=None):
    """Map candidate rider prices into conditional ETA and conversion records."""
    grid = np.linspace(1.,2.,11) if price_multipliers is None else list(price_multipliers)
    if len(grid)==0:
        raise ValueError("At least one candidate price action is required")
    return [solve_response(potential,capacity,queue,a,config) for a in grid]


def choose_action(potential, capacity, queue, price_multipliers=None, eta_limit=10., config=None):
    """Maximize modeled NEW completions subject to ETA and price bounds.

    Candidate multipliers must be in [1,2]. Ties within 1e-9 completions choose
    the lowest price. No feasible candidate returns selected_action=None rather
    than silently relaxing the service constraint or pretending to be feasible.
    The objective excludes existing-queue completions because their old requests
    have priority and are not newly converted by this action.
    """
    if not isfinite(float(eta_limit)) or eta_limit<=0:
        raise ValueError("ETA limit must be finite and positive")
    rows = response_surface(potential,capacity,queue,price_multipliers,config)
    if any(row["price_multiplier"]<1 or row["price_multiplier"]>2 for row in rows):
        raise ValueError("Action-rule candidate price multipliers must be within [1,2]")
    for row in rows:
        row["meets_eta_constraint"] = bool(not row["zero_capacity"] and row["quoted_eta_proxy_minutes"]<=eta_limit+1e-9)
    feasible = [r for r in rows if r["meets_eta_constraint"]]
    selected = None
    if feasible:
        best = max(r["new_rider_completions"] for r in feasible)
        selected = min((r for r in feasible if r["new_rider_completions"]>=best-1e-9),key=lambda r:r["price_multiplier"])
    return {"feasible":bool(feasible),"selected_action":selected,"candidates":rows,
            "eta_limit_minutes":float(eta_limit),"price_bounds":[1.,2.],
            "objective":"Maximize expected NEW rider completions subject to modeled quoted ETA limit; tie chooses lowest price",
            "reason":"feasible_grid_action" if feasible else "No permitted candidate satisfies the modeled ETA constraint",
            "warnings":list(WARNINGS)}


select_action = choose_action

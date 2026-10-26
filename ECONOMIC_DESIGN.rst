Economic design: forecasting and rider-pricing scenarios
======================================================

Purpose
-------

This demonstration connects time-series forecasting to a price--waiting-time
problem. It is not an estimate of Uber's demand system, production surge policy,
conversion, welfare, or driver response. Forecast accuracy on public NYC records
and outcomes inside an illustrative simulator are separate empirical claims.

Research connection
-------------------

Buchholz, Doval, Kastl, Matějka and Salz (2025), *Personalized Pricing and the Value
of Time: Evidence From Auctioned Cab Rides*, Econometrica 93(3), 929--958, studies
Liftago in Prague. Riders choose among driver offers differing in price and pickup
wait, or reject all offers. The authors estimate heterogeneous price/time
preferences and a driver bidding model that recovers inclusive costs. Their
counterfactuals distinguish rider prices from driver payments and investigate
personalized and ETA-based pricing. This motivates modeling the joint price--ETA
choice, not an assumption that higher prices increase conversion. The data and
institutional setting differ from NYC Uber.

The complementary spatial-search reference is Buchholz (2022), *Spatial
Equilibrium, Search Frictions, and Dynamic Efficiency in the Taxi Industry*,
Review of Economic Studies 89(2), 556--591. Its New York yellow-taxi spatial
equilibrium is distinct from the 2025 Econometrica paper and from our small pilot.

Primary sources:

* Published article: https://doi.org/10.3982/ECTA18838
* Published PDF, especially Sections 2--3 and 6:
  https://economics.mit.edu/sites/default/files/inline-files/ECTA18838.pdf
* Spatial-search paper:
  https://nbuchholz.scholar.princeton.edu/sites/g/files/toruqf4066/files/nbuchholz/files/buchholz_taxi_spatial.pdf

Two separate experiments
------------------------

1. Public-data forecasting
   Predict hourly median realized rider fares for fixed NYC origin--destination
   zone pairs. Use only information available at the forecast origin under the
   declared internal-feed replay assumption. Evaluate against matched strong
   baselines. TLC's public releases are delayed and are not a live operating feed.

2. Synthetic adaptation and policy scenarios
   Generate varied hourly price, served-trip and calendar sequences from an
   illustrative capacity/queue model. Synthetic-only adaptation tests whether an
   economic data-generating prior transfers to real fare forecasting. A separate
   closed-loop simulation can evaluate pricing scenarios under its own assumed
   preferences, capacity and cancellation rules.

Neither better synthetic policy outcomes nor better real fare forecasts establish
an improvement in actual Uber conversion, ETA, welfare or contribution margin.
The simulator should not be selected after observing a favorable real test result.
Results from a real block already inspected during development are exploratory;
a subsequent untouched period is needed for a confirmatory transfer claim.

Current pilot scope
-------------------

The implementation is an hourly, two-route capacity/queue pilot.
It is not a replication of either cited paper. In particular, it is not a
fleet-conserving spatial equilibrium, a driver auction, an estimated mixed-logit
model, or a production matching engine. Capacity response, if included, is an
explicit reduced-form assumption and must not be described as an independently
validated supply equilibrium.

The current capacity paths are exogenous and identical across compared policies.
Routes can share shocks but do not exchange a conserved vehicle fleet. The
``anticipatory`` policy is a calendar-plus-lagged-arrival heuristic: it receives
known seasonality, announced events and previous potential-arrival counts, with
an illustrative persistence coefficient. It does not inspect the latent demand
state or current/future innovations. Potential-arrival logs would require an
internal platform feed and are absent from TLC. The heuristic does not consume
the trained Chronos-2 forecasts. Its results
therefore cannot establish the value of a Chronos-powered controller. The reported
``quoted_eta`` is a synthetic state-dependent proxy, not observed or simulated
realized pickup waiting time.

Only rider prices are policy choices. Driver compensation is outside the action
space. A model with fixed driver payments cannot justify a positive supply
response merely by asserting that a higher rider fare raises driver pay. Any
capacity response must instead have a separately declared mechanism, such as
lagged utilization-based repositioning, or be treated as an exogenous scenario.

Preference and conversion logic
-------------------------------

A transparent illustrative rider utility is::

    U_i = v_i - alpha_i * rider_price - beta_i * quoted_ETA + epsilon_i

Positive ``alpha_i`` and ``beta_i`` may vary across simulated riders or types.
An outside option makes requests endogenous. With a logistic utility shock,
``Pr(request) = logistic(v_i - alpha_i*p - beta_i*w)`` is one possible synthetic
choice rule. Its parameters are illustrative; the NYC completed-trip data do not
estimate them.

For a small change, the price--ETA trade-off is::

    delta U_i approximately equals -alpha_i * delta_price - beta_i * delta_ETA

Shorter ETA can offset a higher price for some riders, but not necessarily for
the market as a whole. Differences among retained riders can also reflect
selection. The result must be measured rather than imposed in the simulator's
objective or transition equations.

Keep these quantities distinct:

* Potential arrivals: people facing the opportunity to request a trip, before
  any fare-induced refusal.
* Quote acceptance: accepted quotes divided by potential arrivals.
* Service completion: completed trips divided by the same potential-arrival
  denominator, with queued cohorts attributed to their original arrival period.
* Conditional fulfillment: completed trips divided by accepted requests. This
  can improve while total service conversion falls.
* Cancellation: accepted requests abandoning before service, with explicit
  handling of residual queues at the simulation boundary.
* ETA: a quoted forecast of pickup wait, distinct from realized pickup wait and
  from in-vehicle travel time. Average wait among completed rides does not measure
  the experience of riders who declined or cancelled.

Queue and accounting requirements
---------------------------------

Each period should have an explicit event order: observe state, construct the
information set, select rider prices, draw potential arrivals and acceptance,
match/service within capacity, apply declared cancellation rules, then update
queues and capacity for the next period. A different event order is acceptable
only if documented consistently.

Queue conservation must hold::

    ending_queue = starting_queue + newly_accepted - served - cancelled

All terms must be nonnegative, and served plus cancelled cannot exceed the
available queue. Do not silently discard leftover riders at the horizon. Either
report the residual stock, add an identical runoff period for every policy, or
count it explicitly as unresolved. A physical driver model would additionally
need vehicle conservation, occupied-trip durations, destinations and relocation
delays; those features are beyond the current capacity/queue pilot.

Forecaster versus controller
---------------------------

Chronos-2 forecasts an outcome conditional on its input history and any declared
future covariates. A forecast of prices generated by an existing policy can
learn that policy's behavior; it does not by itself identify an optimal action.

Implemented price-action response map
-------------------------------------

``decision_layer.py`` adds a separate, one-period decision calculation. A
forecaster supplies potential arrivals ``A`` and service capacity ``S``; the
known starting queue is ``Q``. A candidate rider price multiplier ``a`` is an
action, not a future observed covariate or a target price forecast. With explicitly
illustrative positive price/time sensitivities, expected new requests are
``D(w,a) = A * Pr(request | quoted ETA w, price a)``. The model solves::

    w = min(max_ETA, base_ETA + gain * max((Q + D(w,a))/S - threshold, 0))

For positive capacity the right-hand side is nonincreasing in ``w``, so its
intersection with the increasing left-hand side is unique. Bisection finds that
fixed point. Existing requests receive service first; expected new completions
are ``min(D, max(S-Q, 0))``. New-rider conversion divides those completions by
``A``, not by all completions including the old queue. Unserved requests remain
explicitly pending. Zero capacity and infeasible ETA constraints are reported.

The declared action rule maximizes modeled new completions subject to a quoted
ETA bound and the candidate price grid, choosing the lowest price on ties. It
does not guarantee that a forecast-feasible action is feasible in the realized
state. With fixed capacity this one-period model does not claim that a higher
price improves total conversion; reducing demand can lower quoted ETA while
leaving completions unchanged or lower.

At an origin, queue input must come from the previous completed period, never
from the target period's ending queue. Response parameters are assumed known
illustrative priors, not learned behavioral estimates. Evaluating a selected
action again with realized synthetic arrivals/capacity is a *state-oracle
evaluation of the same response model*: it is not observed realized conversion
or waiting time from the original simulator, which uses a different response
rule. A separate oracle action may use realized state only as an explicitly
labeled non-deployable reference. Forecasting potential arrivals and capacity
with original Chronos weights is distinct from adapting Chronos to synthetic
completed-fare sequences; a fare-trained adapter is not automatically a valid
market-pressure model.

End-to-end controller requirements
----------------------------------

An end-to-end controller comparison requires:

* Identical policy objectives, action bounds, ramp-rate limits and state-update
  rules for reactive and predictive alternatives.
* A causal information set at each decision: no realized future demand, shock,
  ETA, fare or capacity unless explicitly labeled an oracle comparison.
* Identical latent shock paths or common random numbers across policies where
  feasible, and independent held-out seeds and parameter regimes.
* A forecast ablation that holds the controller fixed while replacing only its
  forecasting input. Otherwise gains cannot be attributed to Chronos-2.
* Explicit rider price caps, affordability and reliability guardrails, queue
  carryover, cancellation, capacity delays and uncertainty-sensitive fallback.
* Reporting of all-arrival service conversion, fare burden and waits alongside
  any revenue objective. More revenue or shorter waits alone is not success on
  all dimensions.

For synthetic training, randomize price/time sensitivities, shock persistence,
capacity scarcity and cancellation behavior rather than selecting one favorable
world. Document training sequences, random seeds, adapter settings and checkpoint
selection. Keep original public model weights unchanged and preserve zero-shot
and baseline comparisons.

What public NYC observations do not identify
--------------------------------------------

The panel contains completed-trip fares and counts, not rejected quote menus,
individual membership, potential arrivals, idle drivers or an isolated surge
component. It cannot identify price elasticity, willingness to pay for shorter
ETA, causal Uber One effects, structural supply response or a welfare-optimal
price. Route-specific historical fare levels are scale references, not proven
no-surge counterfactual prices. Synthetic parameter values must therefore be
labeled illustrative and assessed with sensitivity analysis, not presented as
estimates recovered from this panel or transported from Prague.

Technical appendix
==================

Victor H. Aguiar | Research discussion | 26 October 2026
1
Technical appendix
1. Public-data forecasting evidence
Question. Does an off-the-shelf time-series foundation model add predictive value over credible local
baselines on a small, reproducible rider-fare task? This does not test Uber's internal models or establish
a pricing treatment effect.
Target and sample. Hourly median base_passenger_fare for completed, nonshared Uber trips,
excluding taxes, tips, tolls and fees. Six short nonairport, cross-zone routes were selected using January
only. Roughly 237,000 retained rides form a 12,954-row hourly panel after processing 62.9 million public
rows. Hours with fewer than ten rides have missing labels, not zero fares.
Protocol. Training: January 1-February 14; validation: February 15-28; heldout test: March 1-28, 2026
(NYC calendar). Forecast origins every six hours; target hours ending 1, 3 and 6 hours later. A
three-hour completed-trip embargo and 672-hour context apply. Known calendar and past-only counts
are allowed. Hyperparameters are selected on validation; learned baselines refit on available training
plus validation before test.
March test: identical 1,574 cases per model
Model
MAE ($)
High-fare MAE ($)
80% coverage
Chronos-2 + synthetic LoRA (30-step candidate)
1.067
2.552
76.6%
Chronos-2 + calendar/count (zero-shot)
1.067
2.550
76.6%
Chronos-2
1.091
2.517
77.3%
Gradient boosting
1.177
2.622
84.9%
Ridge
1.194
3.398
64.8%
Four-week seasonal
1.311
2.477
78.9%
Daily seasonal
1.498
3.006
83.0%
Weekly seasonal
1.530
2.746
82.1%
Persistence
2.007
4.216
81.1%
The prespecified family comparison selected four-week seasonality on validation: the selected zero-shot Chronos variant
reduces its test MAE by 18.6%. The stronger test-performing boosting comparator is reported too: a 9.4% reduction. The
paired day-block interval versus the validation-selected seasonal baseline is 14.6%-22.4%, descriptive for one city/month;
adjacent-day dependence may remain.


Victor H. Aguiar | Research discussion | 26 October 2026
2
2. From forecasting to a price decision
Price is the action. A separate, wholly synthetic prototype forecasts potential rider arrivals and service
capacity jointly, then evaluates candidate price multipliers from 1.0 to 2.0 through a fixed heterogeneous
price/wait choice model. Every method receives the same current queue. Chronos-2 remains zero-shot;
the fare LoRA checkpoint is not loaded.
Evaluation fixes the first 16 heldout trajectories (eight two-route episodes) at four prespecified origins, for
64 cases. Each prediction uses 256 hours of past state and known future calendar. Ridge and boosting
use 12,032 labeled examples from 32 separate training trajectories. Thus the comparison is
domain-trained classical models versus an unadapted foundation model.
Illustration: the first prespecified heldout case. The solid curve is a NEW model response evaluated with realized synthetic
inputs, not observed ETA or conversion from the original simulator. Capacity is exogenous; higher rider prices do not create
drivers, guarantee more completions, or imply driver-pay pass-through.
Heldout pressure and conditional-response errors
Model
Joint state MAE
ETA MAE (min)
Completed-conversion
MAE (pp)
Persistence
6.972
1.516
6.818
Seasonal 24h
10.869
2.287
10.447
Ridge (domain-trained)
5.434
1.306
5.837
HGB (domain-trained)
5.377
1.299
5.907
Chronos-2 ZERO-SHOT
5.396
1.290
5.690
Decision rule and failure. Maximize new completions among grid prices satisfying a modeled ETA limit
of ten minutes; ties favor lower prices, and infeasibility triggers abstention. Chronos selected 47 actions:
13 violated the limit under realized inputs, with 17 abstentions. Boosting selected 48: 14 violations, 16
abstentions. Common-feasible regret is nearly identical (0.0837 versus 0.0834 rides on 27 cases). This
controller is unsafe for deployment.
Joint point forecasting does not provide a calibrated joint predictive distribution. Response errors compare the same
declared calculator under forecast and realized inputs; they are not empirical causal estimates. The oracle is the best
feasible action on this one-step grid, not global optimal pricing.


Victor H. Aguiar | Research discussion | 26 October 2026
3
3. Economic value: explicit assumptions
No conversion, revenue or profit lift is estimated by this demo. The business case asks how much net
causal improvement an internal experiment would need to justify an investment, holding the current
production stack as the comparator.
Scale and unit economics
Uber reported $28.988 billion in Q2 2026 Mobility Gross Bookings. Multiplying that one quarter by four
gives a $115.952 billion scale reference, not annual guidance or a forecast. At an assumed 20% eligible
share, G = $23.1904 billion. Assume baseline quote-to-completed conversion c = 60%, contribution m =
10% on incremental volume at baseline fare, and all-in annual program cost C = $1 million.
For unchanged average gross fare and no extra subsidy: incremental contribution = G x m x (d/c); net
contribution = G x m x (d/c) - C. Here d is the absolute conversion change as a fraction, so +0.05
percentage points means d = 0.0005, or five additional completed rides per 10,000 eligible quotes.
Net causal lift
Extra rides / 10k quotes
Contribution before cost
Net after $1m cost
-0.05 pp
-5
$-1.93m
$-2.93m
+0.00 pp
+0
$+0.00m
$-1.00m
+0.01 pp
+1
$+0.39m
$-0.61m
+0.05 pp
+5
$+1.93m
$+0.93m
+0.10 pp
+10
$+3.87m
$+2.87m
Interpretation. The +0.05-point scenario produces $1.93 million incremental contribution and $0.93
million net. Break-even is +0.0259 points under these assumptions. A $2 million program cost makes the
+0.05-point scenario slightly negative. These are accounting sensitivities, not estimated effects.
What must be measured, not assumed
Conversion lift must be incremental to today's production policy after cannibalization, time-shifting and
displaced trips. The 10% parameter is contribution on incremental volume, not Uber's disclosed take
rate, EBITDA margin or profit rate. Fare changes, driver payments, rider subsidies and subscription
benefits need separate accounting. Uber's Gross Bookings definition does not deduct consumer
discounts/refunds; do not treat coupons as a mechanical gross-fare reduction.
The all-in cost includes science, engineering, serving, monitoring and rollout. Local batched CPU inference is not a cloud
price quote or a tail-latency guarantee. Avoid double-counting compute or already-budgeted incentives. A lower fare can
erase conversion value; no forecast-MAE-to-revenue multiplier is defensible here.


Victor H. Aguiar | Research discussion | 26 October 2026
4
4. Limits, next experiment and sources
The failures that should shape the internal pilot
Tails and calibration. The selected zero-shot variant has high-fare MAE $2.550 versus $2.477 for
four-week seasonality (143 high-fare cases, thresholds fixed from training). Its nominal 80% intervals
cover 76.6%. The worst miss is Astoria to Steinway, March 23, one-hour target: $11.38 forecast versus
$29.13 realized median, an absolute error of $17.75. Route MAEs range from $0.947 to $1.299;
averages hide local risk.
Adaptation did not help materially. A stylized price/ETA/queue simulator generated synthetic-only
training episodes. Rank-8 LoRA trained for 60 steps with checkpoints at 30 and 60; real February
validation retained the unadapted model. The selected adapted diagnostic has test MAE $1.066502
versus $1.066787, only 0.027% lower. This is not evidence that the curriculum succeeds. Better
marketplace curricula remain a hypothesis.
Scope. Completed TLC trips omit rejected quotes and unmet demand. Product mix, rider composition
and trip conditions vary. No Uber One, competitor-price or rider-level causal data are observed. Public
files arrive monthly, so the embargoed replay is not a live public-data service. Neither the six short NYC
routes nor synthetic worlds establish transfer to another market.
A bounded internal path
1. Agree on a prediction target and data contract using quotes, nonrequests, availability, ETA and
outcomes. 2. Shadow the current stack with identical data freshness, calendar and external-signal rules;
audit tails, calibration, route shifts and latency. 3. Jointly design fallback, uncertainty-aware constraints
and monitoring with engineering. 4. Only then run a constrained experiment with marketplace spillovers,
revenue/cost accounting and rider reliability as guardrails.
Sources and reproducibility
NYC TLC: official trip data; HVFHV data dictionary. Chronos-2: technical report and model card. Cached weights are
hash-identical to the October 30, 2025 release, before the 2026 test data.
Buchholz, Doval, Kastl, Matejka and Salz (2025), "Personalized Pricing and the Value of Time: Evidence From Auctioned
Cab Rides," Econometrica 93(3), 929-958. doi:10.3982/ECTA18838. Inspiration for price/wait choice, not a replication or
transfer of their estimates.
Uber, Q2 2026 financial results, August 5, 2026: Mobility Gross Bookings scale only. Every other business-case parameter
is assumed.
https://github.com/vaguiarl/marketplace-foundation-forecasting
Frozen protocols, source/model hashes, cached predictions, audit receipts and runnable scripts support replication. Original
GARP-EFM research files were not changed. No private Uber data were used.

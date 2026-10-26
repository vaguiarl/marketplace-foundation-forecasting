# Technical appendix

## Three experiments and their claims

1. **Real NYC fare forecasting.** The outcome is hourly median `base_passenger_fare` among eligible completed Uber trips within a fixed origin-destination pair. It excludes tips, tolls, taxes and fees. It is not a binding upfront quote, a no-surge fare, an isolated surge premium, an ETA target, or a conversion target.
2. **Synthetic forecast-to-decision prototype.** Original Chronos-2 jointly forecasts next-hour potential arrivals and exogenous service capacity. An explicitly assumed economic response map converts candidate rider prices and those forecasts into a quoted-ETA proxy, request acceptance, and new-rider completion. Evaluation against realized synthetic states is a toy-model calculation, not an empirical counterfactual.
3. **Synthetic adaptation pilot.** A fresh LoRA adapter is trained on stylized fare trajectories only. This is a distinct target from the pressure prototype. Its negligible fare improvement does not establish that an economic curriculum helps.

## NYC data and timing

Official [NYC TLC downloads](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and [HVFHV dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf) provide January-March 2026 records. `HV0003` identifies Uber. The pipeline processes 62,874,417 records and retains 237,041 eligible trips on six selected routes. Raw monthly files are downloaded on demand and are not committed.

Filters remove non-Uber records, invalid/out-of-month times, trips lasting over three hours, nonpositive fares or fares above $500, distances outside 0.1-100 miles, shared/unknown-sharing trips, access-a-ride records, unknown zones, airport endpoints, and same-zone trips. Sequential exclusion counts are saved. Identical records are not automatically deduplicated because no unique trip identifier is supplied.

Route selection uses January only, ranking hourly coverage then volume under a declared density screen. Too few Manhattan routes qualified, so the predeclared NYC fallback selected Brooklyn/Queens routes. No route was selected for forecast performance. Hourly medians use original trip values, not medians of batch medians. Targets with fewer than ten eligible trips remain missing.

Source NYC clock times are localized explicitly; invalid DST times are dropped. The stored grid is continuous UTC, with NYC-local calendar covariates. Each origin uses only pickup bins strictly before origin minus three hours. Thus its newest bin starts at origin minus four hours; even the longest retained ride has ended by the origin. This mimics an internal completed-trip feed. Public TLC releases are delayed monthly files, not a live feed.

Training ends February 15 at 05:00 UTC; validation ends March 1 at 05:00 UTC. Available labels are additionally embargoed. The test spans March 1-28 NYC-local time. Origins are six hours apart, with targets in the first, third and sixth following pickup-hour bins. Context is 672 hours. Horizon scores cover different target-clock hours and should not be interpreted as a controlled comparison of horizon difficulty.

## Models and evaluation

The six baselines are persistence, daily seasonal, weekly seasonal, four-week seasonal, regularized lagged regression, and histogram gradient boosting. Learned features include causal fare/count lags, rolling summaries, route identifiers and known calendar information. Ridge evaluates three regularization values; boosting evaluates 15/31 leaves and squared/absolute-error loss. Hyperparameters use chronological validation. Selected learned models refit once on available training-plus-validation labels before testing.

The original `amazon/chronos-2` weights have SHA256 `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`. They are hash-identical to the [October 30, 2025 weight release](https://huggingface.co/amazon/chronos-2/commit/95a9710e2596287d08352589f42634fa5abdf0a7), preceding these 2026 outcomes. This matters because older taxi datasets appear in Chronos-2's pretraining inventory. The cached revision is `29ec3766d36d6f73f0696f85560a422f50e8498c`. Original and covariate-informed zero-shot variants use no domain gradient updates. Cross-learning across forecast origins is disabled.

All nine tested variants share 1,574 observed test cases. The validation-selected Chronos variant has MAE $1.066787 versus $1.310977 for the validation-selected seasonal reference: 18.6266% lower. Its MAE is 9.3645% lower than tuned boosting's $1.177008. The latter is a descriptive comparison, not the prespecified reference selection. A paired NYC-day bootstrap gives a descriptive 14.55%-22.45% interval for improvement over the seasonal reference; adjacent-day dependence and the short window limit its interpretation.

Important failures are preserved: on 143 high-fare cases, Chronos-calendar MAE is $2.550145 versus $2.476766 for the seasonal reference, 2.96% worse. Its nominal 80% intervals cover 76.56% of test outcomes. Large simultaneous cross-route misses occur; no cause is inferred without external evidence. Complete per-route, per-horizon and largest-error diagnostics are saved.

## Economic simulation and adaptation

[Buchholz, Doval, Kastl, Matejka and Salz (2025)](https://doi.org/10.3982/ECTA18838), *Personalized Pricing and the Value of Time: Evidence From Auctioned Cab Rides*, Econometrica 93(3):929-958, motivates heterogeneous price and waiting-time preferences. Its Liftago setting and estimated model are not replicated here. Our parameters are illustrative, not estimates from the paper or Uber.

The hourly simulator has heterogeneous rider acceptance, seasonal and stochastic arrivals, exogenous capacity, FIFO queues, abandonment and locked past quotes. Rider prices do not automatically change driver payments or create capacity. Calendar/lagged-arrival heuristics compare flat, reactive and anticipatory rules under common exogenous draws. Quoted ETA is a proxy. Separate worlds are robustness illustrations, not isolated parameter experiments.

Training-only observed scales anchor 192 synthetic series of 1,008 hours. A fresh rank-8, alpha-16 adapter changes 1,206,912 parameters for 60 CPU/float32 steps, at learning rate 1e-5, with 256-hour training context and nine-step prediction. Step 30 versus 60 is selected on real February validation only; zero adaptation remains an explicit option. Total training/check/export time was about 53 seconds. Base-weight immutability, zero-adapter equivalence, merged reloads and cross-runtime predictions were checked.

The selected diagnostic adapter has test MAE $1.066502, only 0.0267% lower than the original covariate model. Validation prefers the original. This bounded pilot does not demonstrate meaningful curriculum value; it supplies infrastructure for a better-founded future test.

## Price as an action

The separate pressure benchmark uses 64 predeclared cases from 16 held-out synthetic trajectories, not 16 necessarily independent worlds. Ridge and boosting train on the first 32 training trajectories; zero-shot Chronos uses no such training. Both forecast arrivals and service capacity from historical states and known calendar covariates. The queue supplied to the economic layer is from the preceding hour, not the future.

The response map solves a unique fixed point between congestion-based quoted ETA and heterogeneous price/time-sensitive acceptance. Existing queued riders receive priority. Candidate price multipliers lie between 1 and 2. A myopic grid rule maximizes modeled new completions subject to a ten-minute ETA constraint, preferring the lowest price on ties and abstaining when no candidate is feasible. This is not a globally optimal dynamic pricing policy. With capacity fixed, higher prices cannot be assumed to increase completions.

Joint state MAE is 5.3960 for zero-shot Chronos and 5.3771 for domain-trained boosting: near parity, with Chronos 0.35% higher. Modeled ETA-response MAE is 1.2896 versus 1.2988 minutes; modeled completion-response error is 5.6896 versus 5.9065 percentage points. These response values are generated by the new assumed calculator using realized inputs, not observed outcomes from the original simulator.

Safety failures remain explicit: Chronos abstains in 17/64 cases, and 13/47 selected actions violate the ETA constraint under realized-input evaluation. Boosting abstains in 16/64, with 14/48 violations. Common-feasible regret is essentially tied on 27 cases. Marginal quantiles do not constitute a calibrated joint distribution; point-forecast plug-in responses ignore state and parameter uncertainty. This is an interface demonstration, not evidence of safe or superior pricing.

## Business value sensitivity

Uber's [Q2 2026 release](https://investor.uber.com/news-events/news/press-release-details/2026/Uber-Announces-Results-for-Second-Quarter-2026/default.aspx) reports Mobility Gross Bookings of $28.988b. Four times this quarter is a $115.952b scale anchor, not a revenue forecast. The default scenario assumes 20% eligibility, 60% baseline quote-to-completion conversion, 10% contribution on incremental volume, $1m annual program cost, no average gross-fare change and no extra subsidy. These are not Uber operating estimates.

Under unchanged prices, contribution before program cost is `eligible_GB * (net_causal_conversion_lift / baseline_conversion) * incremental_volume_margin`. A +0.05 percentage-point net lift, five extra completions per 10,000 eligible quotes, yields approximately $1.933m contribution and $0.933m net. Zero lift loses the $1m program cost. Break-even is +0.02587 percentage points, or 2.59 additional completions per 10,000 eligible quotes. At $2m cost, the +0.05 pp scenario is slightly negative.

Gross fare changes and consumer coupon costs are separate channels because reported Gross Bookings precede consumer discounts. Sensitivities explicitly model price retention and extra subsidies. A forecast-error reduction is never converted mechanically into business lift. The needed causal improvement must be measured against Uber's current system, after cannibalization and time-shifting.

## Engineering path and audits

Refresh asynchronous forecasts into a cache rather than introducing a foundation-model call into every rider quote. Agree jointly on feature availability, target semantics, latency, calibration, fallbacks, logging and rollout gates. Internal quote/rejection, available-supply and ETA data are required to test decision value. Shadow evaluation should precede a marketplace-aware causal experiment.

The research package has 48 scientific unit tests after the business module was added, plus public-packaging path checks, 106 saved-artifact checks, 704 independent NYC metric/selection checks, 2,543 independent decision checks, and a repeatable 120-trajectory simulator stress audit. These checks verify implementation and arithmetic, not structural truth, external validity or deployment safety. See the saved audit JSON and reproduction guide.

# Foundation models for marketplace prediction

Victor H. Aguiar

## The five minute case

**Foundation models for structured data could strengthen a marketplace's prediction layer.** In this public-data pilot, zero-shot Chronos-2 reduced held-out fare error by **9.4% versus tuned gradient boosting**, without domain training. Chronos-2 is a time-series foundation model; my broader work on tabular foundation models and consumer behavior motivates testing a marketplace-specific curriculum next.

### Evidence

| NYC hourly median rider fare forecast | Held-out MAE |
| --- | ---: |
| Zero-shot Chronos-2 with calendar and past trip counts | $1.067 |
| Tuned gradient boosting | $1.177 |
| Four-week seasonal, selected on validation | $1.311 |

The comparison uses 1,574 matched cases on six short Brooklyn/Queens routes in March 2026. It does not compare against Uber's production models. A small synthetic LoRA pilot added no meaningful improvement, and validation retained the original model. Marketplace curriculum is a research hypothesis, not an established gain.

### Opportunity

Forecast arrivals and available capacity, then use economic structure to evaluate candidate rider prices through ETA and conversion. A separate synthetic prototype makes this decision interface explicit. Its zero-shot state forecasts roughly match a domain-trained tree model, but its action failures preclude deployment claims.

Small causal improvements could still matter at scale. Under explicit assumptions, **five additional completions per 10,000 eligible quotes** correspond to about **$1.9m annual incremental contribution**, or **$0.9m after a $1m program cost**. This is a sensitivity scenario, not a measured business effect. It assumes 20% eligibility, 60% baseline conversion, 10% incremental-volume contribution, unchanged gross fares, and four times Q2 2026 Mobility Gross Bookings as a scale anchor.

### How and what next

I reused my GARP-EFM infrastructure locally, processed 62.9m public TLC rows, selected routes using January only, tuned on February, and held out March. The pipeline includes six baselines, availability delays, pinned weights, and repeatable independent audits.

The next step is an internal replay against the current stack, followed by a cached shadow forecast with engineering-owned data contracts and shared rollout gates. Quote/rejection logs, available supply, and ETA observations are needed before testing pricing value. High-fare misses and under-calibrated intervals remain important limitations.

[One-page briefing](docs/one_page_brief.pdf) · [Technical appendix](docs/technical_appendix.md) · [PDF appendix](docs/technical_appendix.pdf) · [Run and modify](docs/reproduce.md) · [Notebook](demo.ipynb)

Independent demonstration. No affiliation, endorsement, proprietary Uber data, or production pricing claim is implied.

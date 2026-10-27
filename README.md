# Rider pricing and marketplace prediction

Victor H. Aguiar

## Context summary

Uber's rider-pricing problem is to allocate scarce, moving capacity without losing affordability or reliability. A price changes whether someone requests now, waits, chooses another product or leaves. Matching and driver acceptance determine pickup time and completion; each trip changes future availability elsewhere. Rider prices and driver earnings are distinct, so a higher rider price is not automatically an equivalent supply incentive. The relevant outcome is completed rides per original quote opportunity, alongside contribution and reliability, not conversion among a selected group of remaining requests.

Uber already forecasts market conditions and operates sophisticated ETA and matching systems. The opportunity is incremental: could a pretrained foundation model improve a state forecast, residual or uncertainty estimate that those systems use? Economics connects the forecast to candidate prices and heterogeneous price-waiting-time trade-offs. It also prevents the mistake of treating an accurate forecast of today's endogenous fare as an estimate of what a different price would cause.

I tested this entry point with public data and existing local infrastructure. Zero-shot Chronos-2 reduced held-out mean fare error by 9.4% versus tuned boosting on six NYC routes. A separate synthetic prototype connects joint arrival/capacity forecasts to price-dependent ETA and completion calculations. Neither establishes better pricing, and a small economic fine-tuning pilot added no material gain. My proposal is a bounded internal comparison against Uber's current prediction stack, followed by shadow evaluation and a constrained marketplace experiment only if the added signal improves decision-relevant outcomes. Marketplace-specific training would need its own test against an unadapted model.

## Read the argument and evidence

[Half-page summary](docs/one_page_brief.pdf) · [Narrative memo](docs/pricing_narrative.pdf) · [Readable narrative](docs/pricing_narrative.md) · [Technical appendix](docs/technical_appendix.md) · [PDF appendix](docs/technical_appendix.pdf)

The narrative explains the pricing context before the prediction opportunity, then covers the public-data experiment, economic simulation, failed adaptation result, business sensitivities and the proposed science-engineering test. It distinguishes official public information from our economic analysis and proposed design.

## Run and modify

[Reproduction guide](docs/reproduce.md) · [Notebook](demo.ipynb) · [Economic design](ECONOMIC_DESIGN.rst)

Cached HTML reports under `outputs/` work offline. The repository contains the aggregated public panel, synthetic corpora, strong baselines, pinned-model instructions and repeatable audits. It excludes raw trip files, model weights and private data. `narrative_content.py` is the editorial source; `build_narrative.py` rebuilds the current summary and memo without rerunning experiments.

Independent demonstration. No affiliation, endorsement, proprietary Uber data, or production pricing claim is implied.

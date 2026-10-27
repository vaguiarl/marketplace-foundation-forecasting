# Run and modify

## Read without installing anything

Clone this repository and open the self-contained HTML files locally:

- `outputs/report.html`: real NYC fare forecasts, including explicit zero-shot traces.
- `outputs/decision_demo/report.html`: synthetic joint-state forecasts and price-action responses.
- `outputs/business_value/report.html`: assumption-driven business-value sensitivity.
- `docs/one_page_brief.pdf`: the half-page context summary.
- `docs/pricing_narrative.pdf`: the dense narrative memo.

The reports embed their figures. No service, account, cloud job or private data is needed to read them. Alternatively run `python3 -m http.server 8773 --directory outputs` and visit `http://localhost:8773/report.html`.

## Reproduce models

Use Python 3.12 or later in an isolated environment. The recorded benchmark used Python 3.14.6 with `requirements.txt`; its pinned NumPy requires Python 3.12 or later. Exact historical package versions may have platform-specific availability.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_model.py
.venv/bin/python -W ignore::DeprecationWarning -m unittest discover -s tests -v
.venv/bin/python -W ignore::DeprecationWarning benchmark.py
.venv/bin/python report.py
.venv/bin/python pressure_benchmark.py
.venv/bin/python business_value.py
```

The small aggregated public NYC panel and synthetic corpora are included. `prepare_data.py` can rebuild the panel from approximately 1.5 GB of official monthly files. Original model weights are fetched from their pinned official Hugging Face revision, not committed. All inference runs locally. No training is needed for the zero-shot demonstrations.

`benchmark.py` without an adapter path reproduces the original-model and baseline comparison. To include newly generated adapters, pass `--synthetic-checkpoints outputs/synthetic/checkpoints`. The original run's cached predictions include its diagnostic adapter. Fresh runs replace output reports and receipts, so preserve the shipped results first or select a new `--output` directory.

## Reproduce synthetic adaptation

Use a separate environment with `requirements-training.txt`; do not mutate an existing research environment. The recorded training runtime used Python 3.11. Run `fine_tune.py --help`. It requires PEFT and fails instead of silently switching to full fine-tuning.

```bash
python3.11 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-training.txt
.venv-training/bin/python fine_tune.py --corpus outputs/synthetic/corpus.npz --output outputs/new_synthetic_training --run
.venv/bin/python benchmark.py --synthetic-checkpoints outputs/new_synthetic_training/checkpoints --output outputs/new_benchmark
```

The training script refuses to overwrite an already attempted optimization. The training data are synthetic, with coarse dimensional scales calibrated from the real training period only. Real validation chooses among the declared checkpoints; it is not used for gradient updates. A zero-step recommendation must remain available.

## Audit and extend

The current editorial source is `narrative_content.py`. Install `requirements-docs.txt` in a suitable isolated environment and run `build_narrative.py` to regenerate the half-page summary, narrative PDF and readable Markdown. `build_brief.py` now delegates to that builder. The historical four-page technical appendix remains supporting evidence; its original generator is retained as `build_legacy_package()` and is not run by default.

`audit_results.py`, `independent_review.py`, `audit_simulator.py` and `independent_pressure_review.py` repeat saved-artifact checks. Some shipped provenance paths were normalized for public distribution; the publication manifest records copied source/output hashes separately from original-run hashes. Recomputed raw-file checks require downloading the original TLC files. Recomputed adapter checks require recreating the uncommitted model files. Do not interpret missing large artifacts as an unchanged full audit passing.

Use the notebook to inspect cached results. For new hypotheses, freeze a new protocol and test set before examining outcomes. Useful extensions include conditional quote/ETA targets with suitable internal logs, better tail calibration, explicit uncertainty propagation, stronger economic curricula and tests across additional cities. The existing fare adapter is not a validated ETA/conversion model.

## Data and dependencies

TLC data and upstream model weights retain their respective terms. This repository does not include raw TLC trip files, model weights, private communications, credentials or proprietary Uber data. Public visibility does not change third-party rights. No separate broad license grant for the original demonstration code is specified here.

"""Build the concise discussion brief and a transparent technical appendix.

Read-only with respect to the experiments: reads cached JSON/figures only.
Run with the bundled ReportLab/pypdf runtime. No models, downloads or training.
"""
from pathlib import Path
import json
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/share"
REPO = "https://github.com/vaguiarl/marketplace-foundation-forecasting"
NAVY, BLUE, GREY = colors.HexColor("#142B46"), colors.HexColor("#1764A8"), colors.HexColor("#536475")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", fontName="Helvetica-Bold", fontSize=25, leading=28, textColor=NAVY, spaceAfter=10))
styles.add(ParagraphStyle(name="SubtitleCustom", fontName="Helvetica", fontSize=11, leading=14, textColor=GREY, spaceAfter=13))
styles.add(ParagraphStyle(name="BodyCustom", fontName="Helvetica", fontSize=11, leading=15, textColor=NAVY, spaceAfter=9))
styles.add(ParagraphStyle(name="BriefBody", fontName="Helvetica", fontSize=11.2, leading=15.2, textColor=NAVY, spaceAfter=10))
styles.add(ParagraphStyle(name="Punchline", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=BLUE, spaceAfter=4))
styles.add(ParagraphStyle(name="HeadCustom", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=BLUE, spaceBefore=9, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallCustom", fontName="Helvetica", fontSize=9.3, leading=12, textColor=GREY, spaceAfter=6))
styles.add(ParagraphStyle(name="CellCustom", fontName="Helvetica", fontSize=9.5, leading=12, textColor=NAVY))


def para(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def heading(text):
    return para(text, "HeadCustom")


def table(rows, widths):
    values = [[para(str(v), "CellCustom") for v in row] for row in rows]
    t = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF1F8")),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("LINEBELOW", (0,0), (-1,0), .7, BLUE),
        ("LINEBELOW", (0,1), (-1,-1), .3, colors.HexColor("#DAE2EA")),
        ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CFD9E4"))
    canvas.line(46, 40, letter[0]-46, 40)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(GREY)
    canvas.drawString(46, 26, "Victor H. Aguiar | Research discussion | 26 October 2026")
    canvas.drawRightString(letter[0]-46, 26, str(doc.page))
    canvas.restoreState()


BRIEF = [
    ("The opportunity", "Foundation forecasts could complement Uber's existing prediction stack; economic structure can connect predictions to decisions. Chronos-2 is a time-series foundation model, extending my broader work on foundation models for structured and tabular data."),
    ("What I built", "A reproducible local test, reusing my GARP-EFM infrastructure on an Apple M1: processed 62.9 million public NYC trip records and retained roughly 237,000 eligible rides across six short routes. Chronological training/validation preceded a March test of 1,574 common route-hour-horizon cases against six baselines."),
    ("The evidence", "Zero-shot Chronos-2 with calendar and past trip counts achieved <b>$1.067 MAE versus $1.177 for gradient boosting: 9.4% lower error</b>. This is realized rider-fare forecasting, not upfront quotes, identified surge, or a comparison with Uber's production models. Tail errors and interval calibration still need work. A 60-step synthetic LoRA pilot gave no material gain; validation retained zero-shot."),
    ("A research hypothesis", "A marketplace-specific synthetic curriculum could improve forecasting of arrivals, availability and disruption dynamics. In a separate toy prototype, joint zero-shot arrival/capacity forecasts were nearly tied with trained boosting. An economic calculator then maps candidate prices to ETA and conversion; its unsafe constraint failures show why shadow evaluation and guardrails come first."),
    ("Business case: a scenario, not measured lift", "At 20% eligibility, 60% baseline conversion, 10% incremental-volume contribution, $1 million annual program cost and unchanged average fare, a <b>net causal +0.05 percentage-point conversion lift</b> (5 extra completions per 10,000 quotes) implies $1.93 million contribution and $0.93 million net annually. Scale: $115.952 billion, four times Q2 2026 Mobility Gross Bookings, not a forecast. No lift is inferred from the MAE result."),
    ("Proposed next step", "With internal quotes, nonrequests, availability and ETA data, shadow a prediction service against the current stack. Establish calibration, tail performance and serving reliability jointly with engineering before a constrained experiment. The decision is whether this merits an internal pilot - not whether to replace pricing today."),
]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / "outputs/metrics.json").read_text())
    pm = json.loads((ROOT / "outputs/decision_demo/metrics.json").read_text())
    scenarios = json.loads((ROOT / "outputs/business_value/scenarios.json").read_text())
    all_rows = [r for r in m["test"] if r["horizon"] == "all"]
    story = [para("Foundation models for<br/>marketplace prediction", "TitleCustom"),
             para("Victor H. Aguiar | Five-minute research brief", "SubtitleCustom"),
             para("9.4% lower fare MAE than tuned boosting - zero domain training", "Punchline"),
             para("Six-route public NYC pilot; not a comparison with Uber's production stack.", "SubtitleCustom")]
    for title, text in BRIEF:
        story.append(para(f"<b>{title}.</b> {text}", "BriefBody"))
    story.append(para(f'<link href="{REPO}" color="#1764A8">Reproducible code, results and technical appendix</link>', "SmallCustom"))
    doc = SimpleDocTemplate(str(OUT / "one_page_brief.pdf"), pagesize=letter, rightMargin=46, leftMargin=46, topMargin=42, bottomMargin=50,
                            title="Foundation models for marketplace prediction", author="Victor H. Aguiar")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    brief_text = "Foundation models for marketplace prediction\nVictor H. Aguiar | Five-minute research brief\n\n9.4% lower fare MAE than tuned boosting - zero domain training\nSix-route public NYC pilot; not a comparison with Uber's production stack.\n\n"
    brief_text += "\n\n".join(title+". "+text.replace("<b>", "").replace("</b>", "") for title,text in BRIEF)+"\n\n"+REPO+"\n"
    (OUT / "ONE_PAGE_BRIEF.txt").write_text(brief_text)
    if len(PdfReader(OUT / "one_page_brief.pdf").pages) != 1:
        raise RuntimeError("Brief must be exactly one page; revise layout or text")
    if len(brief_text.split()) > 400:
        raise RuntimeError("Brief exceeds the 400-word limit")

    s = [para("Technical appendix", "TitleCustom"), para("1. Public-data forecasting evidence", "SubtitleCustom"),
         para("<b>Question.</b> Does an off-the-shelf time-series foundation model add predictive value over credible local baselines on a small, reproducible rider-fare task? This does not test Uber's internal models or establish a pricing treatment effect."),
         para("<b>Target and sample.</b> Hourly median base_passenger_fare for completed, nonshared Uber trips, excluding taxes, tips, tolls and fees. Six short nonairport, cross-zone routes were selected using January only. Roughly 237,000 retained rides form a 12,954-row hourly panel after processing 62.9 million public rows. Hours with fewer than ten rides have missing labels, not zero fares."),
         para("<b>Protocol.</b> Training: January 1-February 14; validation: February 15-28; heldout test: March 1-28, 2026 (NYC calendar). Forecast origins every six hours; target hours ending 1, 3 and 6 hours later. A three-hour completed-trip embargo and 672-hour context apply. Known calendar and past-only counts are allowed. Hyperparameters are selected on validation; learned baselines refit on available training plus validation before test."),
         heading("March test: identical 1,574 cases per model")]
    labels = {"Chronos-2 + calendar/count": "Chronos-2 + calendar/count (zero-shot)", "Chronos-2 + synthetic LoRA (30 steps)": "Chronos-2 + synthetic LoRA (30-step candidate)"}
    rows = [["Model", "MAE ($)", "High-fare MAE ($)", "80% coverage"]]
    for row in sorted(all_rows, key=lambda r:r["mae"]):
        rows.append([labels.get(row["model"],row["model"]), f'{row["mae"]:.3f}', f'{row["high_fare_mae"]:.3f}', f'{100*row["coverage_80"]:.1f}%'])
    s += [table(rows, [266,70,100,84]), Spacer(1,10),
          para("The prespecified family comparison selected four-week seasonality on validation: the selected zero-shot Chronos variant reduces its test MAE by 18.6%. The stronger test-performing boosting comparator is reported too: a 9.4% reduction. The paired day-block interval versus the validation-selected seasonal baseline is 14.6%-22.4%, descriptive for one city/month; adjacent-day dependence may remain.", "SmallCustom"), PageBreak(),
          para("2. From forecasting to a price decision", "TitleCustom"),
          para("Price is the action. A separate, wholly synthetic prototype forecasts potential rider arrivals and service capacity jointly, then evaluates candidate price multipliers from 1.0 to 2.0 through a fixed heterogeneous price/wait choice model. Every method receives the same current queue. Chronos-2 remains zero-shot; the fare LoRA checkpoint is not loaded."),
          para("Evaluation fixes the first 16 heldout trajectories (eight two-route episodes) at four prespecified origins, for 64 cases. Each prediction uses 256 hours of past state and known future calendar. Ridge and boosting use 12,032 labeled examples from 32 separate training trajectories. Thus the comparison is domain-trained classical models versus an unadapted foundation model."),
          Image(str(ROOT / "outputs/decision_demo/figures/conditional_price_curves.png"), width=520, height=113),
          para("Illustration: the first prespecified heldout case. The solid curve is a NEW model response evaluated with realized synthetic inputs, not observed ETA or conversion from the original simulator. Capacity is exogenous; higher rider prices do not create drivers, guarantee more completions, or imply driver-pay pass-through.", "SmallCustom"),
          heading("Heldout pressure and conditional-response errors")]
    rows = [["Model", "Joint state MAE", "ETA MAE (min)", "Completed-conversion MAE (pp)"]]
    for row in pm["metrics"]:
        rows.append([row["model"], f'{row["joint_state_mae"]:.3f}', f'{row["eta_model_response_mae_minutes"]:.3f}', f'{row["new_completion_conversion_model_response_mae_pp"]:.3f}'])
    s += [table(rows, [223,90,90,117]), Spacer(1,10),
          para("<b>Decision rule and failure.</b> Maximize new completions among grid prices satisfying a modeled ETA limit of ten minutes; ties favor lower prices, and infeasibility triggers abstention. Chronos selected 47 actions: 13 violated the limit under realized inputs, with 17 abstentions. Boosting selected 48: 14 violations, 16 abstentions. Common-feasible regret is nearly identical (0.0837 versus 0.0834 rides on 27 cases). This controller is unsafe for deployment."),
          para("Joint point forecasting does not provide a calibrated joint predictive distribution. Response errors compare the same declared calculator under forecast and realized inputs; they are not empirical causal estimates. The oracle is the best feasible action on this one-step grid, not global optimal pricing.", "SmallCustom"), PageBreak(),
          para("3. Economic value: explicit assumptions", "TitleCustom"),
          para("No conversion, revenue or profit lift is estimated by this demo. The business case asks how much net causal improvement an internal experiment would need to justify an investment, holding the current production stack as the comparator."),
          heading("Scale and unit economics"),
          para("Uber reported $28.988 billion in Q2 2026 Mobility Gross Bookings. Multiplying that one quarter by four gives a $115.952 billion scale reference, not annual guidance or a forecast. At an assumed 20% eligible share, G = $23.1904 billion. Assume baseline quote-to-completed conversion c = 60%, contribution m = 10% on incremental volume at baseline fare, and all-in annual program cost C = $1 million."),
          para("For unchanged average gross fare and no extra subsidy: incremental contribution = G x m x (d/c); net contribution = G x m x (d/c) - C. Here d is the absolute conversion change as a fraction, so +0.05 percentage points means d = 0.0005, or five additional completed rides per 10,000 eligible quotes."),
          table([["Net causal lift", "Extra rides / 10k quotes", "Contribution before cost", "Net after $1m cost"]]+[[f'{r["conversion_lift_absolute_pp"]:+.2f} pp',f'{r["additional_completions_per_10000_quotes"]:+.0f}',f'${r["incremental_contribution_before_program_cost_usd"]/1e6:+.2f}m',f'${r["net_incremental_contribution_after_program_cost_usd"]/1e6:+.2f}m'] for r in scenarios["default_scenarios"] if r["conversion_lift_absolute_pp"] <= .1], [104,130,143,143]),
          Spacer(1,10), para("<b>Interpretation.</b> The +0.05-point scenario produces $1.93 million incremental contribution and $0.93 million net. Break-even is +0.0259 points under these assumptions. A $2 million program cost makes the +0.05-point scenario slightly negative. These are accounting sensitivities, not estimated effects."),
          heading("What must be measured, not assumed"),
          para("Conversion lift must be incremental to today's production policy after cannibalization, time-shifting and displaced trips. The 10% parameter is contribution on incremental volume, not Uber's disclosed take rate, EBITDA margin or profit rate. Fare changes, driver payments, rider subsidies and subscription benefits need separate accounting. Uber's Gross Bookings definition does not deduct consumer discounts/refunds; do not treat coupons as a mechanical gross-fare reduction."),
          para("The all-in cost includes science, engineering, serving, monitoring and rollout. Local batched CPU inference is not a cloud price quote or a tail-latency guarantee. Avoid double-counting compute or already-budgeted incentives. A lower fare can erase conversion value; no forecast-MAE-to-revenue multiplier is defensible here.", "SmallCustom"), PageBreak(),
          para("4. Limits, next experiment and sources", "TitleCustom"),
          heading("The failures that should shape the internal pilot"),
          para("<b>Tails and calibration.</b> The selected zero-shot variant has high-fare MAE $2.550 versus $2.477 for four-week seasonality (143 high-fare cases, thresholds fixed from training). Its nominal 80% intervals cover 76.6%. The worst miss is Astoria to Steinway, March 23, one-hour target: $11.38 forecast versus $29.13 realized median, an absolute error of $17.75. Route MAEs range from $0.947 to $1.299; averages hide local risk."),
          para("<b>Adaptation did not help materially.</b> A stylized price/ETA/queue simulator generated synthetic-only training episodes. Rank-8 LoRA trained for 60 steps with checkpoints at 30 and 60; real February validation retained the unadapted model. The selected adapted diagnostic has test MAE $1.066502 versus $1.066787, only 0.027% lower. This is not evidence that the curriculum succeeds. Better marketplace curricula remain a hypothesis."),
          para("<b>Scope.</b> Completed TLC trips omit rejected quotes and unmet demand. Product mix, rider composition and trip conditions vary. No Uber One, competitor-price or rider-level causal data are observed. Public files arrive monthly, so the embargoed replay is not a live public-data service. Neither the six short NYC routes nor synthetic worlds establish transfer to another market."),
          heading("A bounded internal path"),
          para("1. Agree on a prediction target and data contract using quotes, nonrequests, availability, ETA and outcomes. 2. Shadow the current stack with identical data freshness, calendar and external-signal rules; audit tails, calibration, route shifts and latency. 3. Jointly design fallback, uncertainty-aware constraints and monitoring with engineering. 4. Only then run a constrained experiment with marketplace spillovers, revenue/cost accounting and rider reliability as guardrails."),
          heading("Sources and reproducibility"),
          para('NYC TLC: <link href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page" color="#1764A8">official trip data</link>; <link href="https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf" color="#1764A8">HVFHV data dictionary</link>. Chronos-2: <link href="https://arxiv.org/abs/2510.15821" color="#1764A8">technical report</link> and <link href="https://huggingface.co/amazon/chronos-2" color="#1764A8">model card</link>. Cached weights are hash-identical to the October 30, 2025 release, before the 2026 test data.', "SmallCustom"),
          para('Buchholz, Doval, Kastl, Matejka and Salz (2025), "Personalized Pricing and the Value of Time: Evidence From Auctioned Cab Rides," Econometrica 93(3), 929-958. <link href="https://doi.org/10.3982/ECTA18838" color="#1764A8">doi:10.3982/ECTA18838</link>. Inspiration for price/wait choice, not a replication or transfer of their estimates.', "SmallCustom"),
          para('Uber, <link href="https://investor.uber.com/news-events/news/press-release-details/2026/Uber-Announces-Results-for-Second-Quarter-2026/default.aspx" color="#1764A8">Q2 2026 financial results</link>, August 5, 2026: Mobility Gross Bookings scale only. Every other business-case parameter is assumed.', "SmallCustom"),
          para(f'<link href="{REPO}" color="#1764A8">{REPO}</link><br/>Frozen protocols, source/model hashes, cached predictions, audit receipts and runnable scripts support replication. Original GARP-EFM research files were not changed. No private Uber data were used.', "SmallCustom")]
    doc = SimpleDocTemplate(str(OUT / "technical_appendix.pdf"), pagesize=letter, rightMargin=46, leftMargin=46, topMargin=42, bottomMargin=51,
                            title="Marketplace foundation forecasting - technical appendix", author="Victor H. Aguiar")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    appendix_pages = len(PdfReader(OUT / "technical_appendix.pdf").pages)
    if appendix_pages != 4:
        raise RuntimeError(f"Appendix expected 4 deliberately composed pages, got {appendix_pages}")
    # Preserve readable content alongside the authoritative, visually checked PDF.
    appendix_text = "Technical appendix\n==================\n\n"+"\n\n".join(page.extract_text() for page in PdfReader(OUT / "technical_appendix.pdf").pages)
    (OUT / "TECHNICAL_APPENDIX.rst").write_text(appendix_text+"\n")
    print(json.dumps({"brief_pages": 1, "brief_words": len(brief_text.split()), "appendix_pages": appendix_pages,
                      "outputs": [str(OUT / "one_page_brief.pdf"), str(OUT / "technical_appendix.pdf")]}, indent=2))


if __name__ == "__main__":
    build()

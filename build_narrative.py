"""Render a half-page context summary and a dense narrative from editorial source.

This command reads cached evidence and does not train, score or alter experiments.
Use a Python environment with reportlab, pypdf and pdfplumber.
"""
from pathlib import Path
from html import escape
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, SimpleDocTemplate, Paragraph, Spacer, PageBreak
from pypdf import PdfReader
import pdfplumber

from narrative_content import TITLE, SUBTITLE, REPO, SUMMARY, SECTIONS, SOURCES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs"
INK = colors.HexColor("#172536")
MUTED = colors.HexColor("#52606e")
STYLES = {
    "title": ParagraphStyle("narrative_title", fontName="Times-Bold", fontSize=16, leading=18.5, spaceAfter=7, textColor=INK),
    "section": ParagraphStyle("narrative_section", fontName="Times-Bold", fontSize=13, leading=15.5, spaceBefore=9, spaceAfter=8, textColor=INK, keepWithNext=True),
    "body": ParagraphStyle("narrative_body", fontName="Times-Roman", fontSize=11.1, leading=13.7, spaceAfter=9, textColor=INK),
    "subtitle": ParagraphStyle("narrative_subtitle", fontName="Helvetica", fontSize=9, leading=11.5, spaceAfter=11, textColor=MUTED),
    "summary": ParagraphStyle("narrative_summary", fontName="Times-Roman", fontSize=11, leading=13.6, spaceAfter=8, textColor=INK),
    "source": ParagraphStyle("narrative_source", fontName="Times-Roman", fontSize=8.7, leading=10.2, spaceAfter=3, textColor=MUTED),
}


def p(text, style="body"):
    return Paragraph(escape(text), STYLES[style])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(54, 27, "Victor H. Aguiar | Rider pricing and marketplace prediction | 26 October 2026")
    canvas.drawRightString(558, 27, str(doc.page))
    canvas.restoreState()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((ROOT / "outputs/metrics.json").read_text())
    rows = {r["model"]: r for r in evidence["test"] if r["horizon"] == "all"}
    assert abs(rows["Chronos-2 + calendar/count"]["mae"] - 1.0667870679804452) < 1e-9
    assert abs(rows["Gradient boosting"]["mae"] - 1.1770075344681823) < 1e-9
    assert sum(len(x.split()) for x in SUMMARY) <= 270

    summary_path = OUT / "one_page_brief.pdf"
    doc = BaseDocTemplate(str(summary_path), pagesize=letter,
        title=TITLE + " - half-page summary", author="Victor H. Aguiar")
    # The entire title/body/link must fit in the top half of a Letter page.
    frame = Frame(54, 400, 504, 340, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0, id="half_page")
    doc.addPageTemplates([PageTemplate(id="half_page", frames=[frame])])
    story = [p(TITLE, "title"), p(SUBTITLE + " | Context summary", "subtitle")]
    story.extend(p(text, "summary") for text in SUMMARY)
    story.append(Paragraph(f'<link href="{REPO}" color="#185c91">Code, narrative memo and technical evidence</link>', STYLES["source"]))
    doc.build(story)
    if len(PdfReader(summary_path).pages) != 1:
        raise RuntimeError("Summary exceeded its half-page frame")
    with pdfplumber.open(summary_path) as pdf:
        bottom = max(char["bottom"] for char in pdf.pages[0].chars)
        if bottom > 396:
            raise RuntimeError(f"Summary extends beyond the page midpoint: {bottom:.1f}")

    narrative_path = OUT / "pricing_narrative.pdf"
    doc = SimpleDocTemplate(str(narrative_path), pagesize=letter, leftMargin=54,
        rightMargin=54, topMargin=47, bottomMargin=48,
        title=TITLE + " - narrative memorandum", author="Victor H. Aguiar")
    story = []
    for index, section in enumerate(SECTIONS):
        if not index:
            story.extend([p(TITLE, "title"), p(SUBTITLE + " | Narrative memorandum", "subtitle")])
        story.append(p(section["title"], "section"))
        story.extend(p(text) for text in section["paragraphs"])
        if index == len(SECTIONS) - 1:
            story.extend([Spacer(1, 5), p("Public sources and supporting evidence", "section")])
            for number, name, url, note in SOURCES:
                label = f'[{number}] <link href="{escape(url, quote=True)}" color="#185c91">{escape(name)}</link>. {escape(note)}'
                story.append(Paragraph(label, STYLES["source"]))
            story.append(Paragraph(f'Experiment protocols, predictions and independent audits: <link href="{REPO}" color="#185c91">{REPO}</link>.', STYLES["source"]))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(narrative_path).pages)
    if pages > 6:
        raise RuntimeError(f"Narrative exceeds the six-page ceiling: {pages}")

    summary_md = "# " + TITLE + "\n\n" + SUBTITLE + "\n\n" + "\n\n".join(SUMMARY) + "\n"
    (OUT / "pricing_summary.md").write_text(summary_md)
    (OUT / "ONE_PAGE_BRIEF.txt").write_text(TITLE + "\n" + SUBTITLE + "\n\n" + "\n\n".join(SUMMARY) + "\n\n" + REPO + "\n")
    memo_md = "# " + TITLE + "\n\n" + SUBTITLE + "\n\n"
    for section in SECTIONS:
        memo_md += "## " + section["title"] + "\n\n" + "\n\n".join(section["paragraphs"]) + "\n\n"
    memo_md += "## Public sources and supporting evidence\n\n"
    memo_md += "\n\n".join(f"[{number}] [{name}]({url}). {note}" for number, name, url, note in SOURCES) + "\n"
    (OUT / "pricing_narrative.md").write_text(memo_md)
    (OUT / "EDITORIAL_CHECKS.json").write_text(json.dumps({
        "summary_words": sum(len(x.split()) for x in SUMMARY),
        "summary_pages": 1, "summary_last_text_bottom_points": bottom,
        "summary_page_midpoint_points": 396,
        "narrative_pages": pages,
        "narrative_words": sum(len(p.split()) for s in SECTIONS for p in s["paragraphs"]),
        "experiments_changed": False,
    }, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), "summary_words": sum(len(x.split()) for x in SUMMARY),
        "summary_bottom_points": bottom, "narrative": str(narrative_path), "narrative_pages": pages}, indent=2))


if __name__ == "__main__":
    build()

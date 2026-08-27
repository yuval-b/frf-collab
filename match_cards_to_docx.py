"""
Convert match_cards.md to a printable Word document, 2 cards per page.

Requirements:
    pip install python-docx

Usage:
    python match_cards_to_docx.py
"""

import re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_BREAK

INPUT  = "output/match_cards.md"
OUTPUT = "output/match_cards.docx"

# ============================================================
# Parse the markdown into structured card data
# ============================================================

with open(INPUT) as f:
    content = f.read()

raw_cards = [c.strip() for c in content.split("---") if c.strip()]

def parse_card(text):
    lines = text.split("\n")
    card = {"name": "", "institution": "", "attending": [], "reach_out": []}
    section = None
    current_match = None

    def flush():
        nonlocal current_match
        if current_match and section:
            card[section].append(current_match)
        current_match = None

    for line in lines:
        line = line.rstrip()
        if line.startswith("# "):
            card["name"] = line[2:].strip()
        elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            card["institution"] = line.strip("*").strip()
        elif line.startswith("## Your top"):
            flush(); section = "attending"
        elif line.startswith("## Worth reaching out"):
            flush(); section = "reach_out"
        elif re.match(r"\*\*.+?\*\*", line):
            flush()
            m = re.match(r"\*\*(.+?)\*\*\s*[\u2014-]\s*\*(.+?)\*", line)
            if m:
                current_match = {
                    "name": m.group(1).strip(),
                    "institution": m.group(2).strip(),
                    "score": None,
                    "label": "",
                }
        elif line.startswith("Score:") and current_match:
            m = re.search(r"([\d.]+)", line)
            if m:
                current_match["score"] = float(m.group(1))
            # Also grab a label if it accidentally got glued onto the same line
            if ">" in line:
                current_match["label"] = line.split(">", 1)[1].strip()
        elif line.startswith(">") and current_match:
            current_match["label"] = line[1:].strip()

    flush()
    return card

cards = [parse_card(c) for c in raw_cards]
print(f"Parsed {len(cards)} cards.")

# ============================================================
# Build the Word document
# ============================================================

doc = Document()

# Tight margins so 2 cards fit comfortably on an A4 page
for section in doc.sections:
    section.top_margin    = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin   = Inches(0.8)
    section.right_margin  = Inches(0.8)

GREY = RGBColor(0x66, 0x66, 0x66)
DARK = RGBColor(0x22, 0x22, 0x22)

def add_run(p, text, *, size=11, bold=False, italic=False, color=DARK):
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return run

def add_match(match):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    add_run(p, f"{match['name']}", size=11, bold=True)
    add_run(p, f"  \u2014  {match['institution']}", size=10, italic=True, color=GREY)
    if match["score"] is not None:
        add_run(p, f"   (score {match['score']:.2f})", size=9, color=GREY)

    if match["label"]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(8)
        add_run(p, match["label"], size=10, italic=True, color=DARK)

def add_card(card):
    # Name
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    add_run(p, card["name"], size=18, bold=True)

    # Institution
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    add_run(p, card["institution"], size=11, italic=True, color=GREY)

    # Attending
    if card["attending"]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        add_run(p, "Your top matches tonight", size=12, bold=True)
        for m in card["attending"]:
            add_match(m)

    # Reach out
    if card["reach_out"]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after  = Pt(4)
        add_run(p, "Worth reaching out to (not attending)", size=12, bold=True)
        for m in card["reach_out"]:
            add_match(m)

def add_divider():
    """Subtle divider between the two cards on the same page."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(14)
    add_run(p, "\u2500" * 60, size=10, color=GREY)  # horizontal line character

# Lay out: 2 cards per page
for i, card in enumerate(cards):
    if i > 0 and i % 2 == 0:
        # New page for every third, fifth, ... card
        doc.add_page_break()
    elif i > 0:
        # Second card on the same page
        add_divider()
    add_card(card)

doc.save(OUTPUT)
print(f"Wrote {OUTPUT}  ({(len(cards) + 1) // 2} pages, 2 cards per page)")
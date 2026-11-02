"""Stamp every PDF page with Mini-Project-style header/footer."""
from __future__ import annotations

from pathlib import Path

import fitz

PDF = Path(__file__).resolve().parent / "Group4_HK_Parking_Vacancy_Prediction_Final_Report.pdf"

HEADER_LEFT = "Capstone Project Report | Group 4"
HEADER_RIGHT = "Hong Kong Parking Vacancy Prediction"
COLOR_MUTED = (0x56 / 255, 0x65 / 255, 0x73 / 255)
COLOR_RULE = (0x1A / 255, 0x52 / 255, 0x76 / 255)


def stamp(path: Path = PDF) -> Path:
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        left_x, right_x = 50.0, w - 50.0
        # Header text (match Pet Adoption: ~y=26.4, 8pt)
        page.insert_text(
            (left_x, 34.0),
            HEADER_LEFT,
            fontsize=8,
            fontname="helv",
            color=COLOR_MUTED,
        )
        tw = fitz.get_text_length(HEADER_RIGHT, fontname="helv", fontsize=8)
        page.insert_text(
            (right_x - tw, 34.0),
            HEADER_RIGHT,
            fontsize=8,
            fontname="helv",
            color=COLOR_MUTED,
        )
        # Header rule (~y=40)
        page.draw_line(
            fitz.Point(left_x, 40.0),
            fitz.Point(right_x, 40.0),
            color=COLOR_RULE,
            width=1.2,
        )
        # Footer rule (~y=h-35)
        foot_y = h - 35.0
        page.draw_line(
            fitz.Point(left_x, foot_y),
            fitz.Point(right_x, foot_y),
            color=COLOR_RULE,
            width=0.8,
        )
        label = f"Page {i + 1}"
        lw = fitz.get_text_length(label, fontname="helv", fontsize=8)
        page.insert_text(
            ((w - lw) / 2, h - 24.0),
            label,
            fontsize=8,
            fontname="helv",
            color=COLOR_MUTED,
        )
    tmp = path.with_suffix(".stamped.pdf")
    doc.save(tmp, deflate=True)
    doc.close()
    tmp.replace(path)
    return path


if __name__ == "__main__":
    out = stamp()
    print(f"Stamped {out}")

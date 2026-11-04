"""Generate Capstone Final Report (DOCX), styled like the Group 4 Mini Project report."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor, Twips

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
OUT = ROOT / "report" / "Group4_HK_Parking_Vacancy_Prediction_Final_Report.docx"

# Palette aligned with Pet Adoption Mini Project report
COLOR_H1 = RGBColor(0x1A, 0x52, 0x76)
COLOR_H2 = RGBColor(0x28, 0x74, 0xA6)
COLOR_BODY = RGBColor(0x1C, 0x28, 0x33)
COLOR_MUTED = RGBColor(0x56, 0x65, 0x73)
COLOR_RULE = "1A5276"
FONT = "Helvetica"


def f(name: str) -> Path:
    """Prefer normalized *_doc.png assets for python-docx compatibility."""
    doc_png = FIG / f"{name}_doc.png"
    if doc_png.exists():
        return doc_png
    for ext in (".png", ".jpg", ".jpeg"):
        p = FIG / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def set_run_font(run, name=FONT, size=9.5, bold=False, italic=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "PingFang HK")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color


def _set_paragraph_border_top(paragraph, color_hex=COLOR_RULE, size_eighths=12, space=8):
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "single")
    top.set(qn("w:sz"), str(size_eighths))
    top.set(qn("w:space"), str(space))
    top.set(qn("w:color"), color_hex)
    pBdr.append(top)
    pPr.append(pBdr)


def _add_bottom_border_to_paragraph(paragraph, color_hex=COLOR_RULE, size_eighths=8, space=4):
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size_eighths))
    bottom.set(qn("w:space"), str(space))
    bottom.set(qn("w:color"), color_hex)
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_page_number(run):
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_begin)
    run._r.append(instr)
    run._r.append(fld_char_end)


def setup_header_footer(section):
    section.different_first_page_header_footer = False

    header = section.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    hp.clear()
    # usable width ≈ A4 210mm - 18mm*2 = 174mm
    tab_pos = Twips(int(Cm(17.4).twips))
    hp.paragraph_format.tab_stops.add_tab_stop(tab_pos, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.SPACES)
    run_l = hp.add_run("Capstone Project Report | Group 4")
    set_run_font(run_l, size=8, color=COLOR_MUTED)
    hp.add_run("\t")
    run_r = hp.add_run("Hong Kong Parking Vacancy Prediction")
    set_run_font(run_r, size=8, color=COLOR_MUTED)
    _add_bottom_border_to_paragraph(hp, color_hex=COLOR_RULE, size_eighths=8, space=4)

    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    fp.clear()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_border_top(fp, color_hex=COLOR_RULE, size_eighths=8, space=4)
    run_pre = fp.add_run("Page ")
    set_run_font(run_pre, size=8, color=COLOR_MUTED)
    run_num = fp.add_run()
    set_run_font(run_num, size=8, color=COLOR_MUTED)
    add_page_number(run_num)


def add_heading_custom(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    if level == 1:
        _set_paragraph_border_top(p, color_hex=COLOR_RULE, size_eighths=12, space=10)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(8)
    else:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
    color = COLOR_H1 if level == 1 else COLOR_H2
    size = {1: 13, 2: 11, 3: 10}.get(level, 9.5)
    for run in p.runs:
        set_run_font(run, size=size, bold=True, color=color)
    return p


def add_para(
    doc,
    text,
    *,
    size=9.5,
    bold=False,
    italic=False,
    space_after=8,
    space_before=0,
    align="left",
    first_line=False,
    color=None,
):
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = 1.25
    if first_line and align == "justify":
        pf.first_line_indent = Cm(0.4)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic, color=color or COLOR_BODY)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.clear()
    run = p.add_run(text)
    set_run_font(run, size=9.5, color=COLOR_BODY)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.left_indent = Cm(0.6 + level * 0.4)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(text)
    set_run_font(run, size=8, italic=True, color=COLOR_MUTED)
    return p


def add_figure(doc, path, width=5.8, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    if caption:
        add_caption(doc, caption)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        run = p.add_run(h)
        set_run_font(run, size=9, bold=True)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "1A5276")
        shading.set(qn("w:val"), "clear")
        tcPr = hdr[i]._tc.get_or_add_tcPr()
        tcPr.append(shading)
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = table.rows[r_i + 1].cells[c_i]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            set_run_font(run, size=9, color=COLOR_BODY)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return table


def build() -> Path:
    doc = Document()
    section = doc.sections[0]
    # A4, margins ~18mm
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(1.0)
    setup_header_footer(section)

    # Title block (left-aligned, Mini Project style)
    add_para(
        doc,
        "Capstone Project Report",
        size=14,
        bold=False,
        first_line=False,
        space_after=8,
        space_before=6,
        color=COLOR_H2,
    )
    add_para(
        doc,
        "Hong Kong Parking Vacancy Prediction",
        size=22,
        bold=True,
        first_line=False,
        space_after=14,
        color=COLOR_H1,
    )
    add_para(
        doc,
        "Group Number: 4",
        size=14,
        bold=False,
        first_line=False,
        space_after=4,
        color=COLOR_H2,
    )
    add_para(
        doc,
        "Group Members: Lee Ki Yick (Keith) · Chin Ka Wang (Kelvin)",
        size=14,
        bold=False,
        first_line=False,
        space_after=14,
        color=COLOR_H2,
    )

    # 1 Project Details (renumbered from former 2)
    add_heading_custom(doc, "1. Project Details", 1)
    add_heading_custom(doc, "1.1 Project Title", 2)
    add_para(
        doc,
        "Hong Kong Parking Vacancy Prediction (Product name: 「咪再兜」— “stop circling for parking”).",
        align="justify",
    )

    add_heading_custom(doc, "1.2 Project Objectives", 2)
    add_para(doc, "The project objectives are:", first_line=False, space_after=4)
    add_bullet(
        doc,
        "Define a clear research question around forward-looking parking vacancy prediction (30–60 minutes ahead).",
    )
    add_bullet(doc, "Build a reproducible ETL pipeline from official open APIs into a queryable database.")
    add_bullet(
        doc,
        "Establish a strong, interpretable historical baseline and compare Machine Learning models against it.",
    )
    add_bullet(
        doc,
        "Conduct EDA and visualization across Hong Kong districts to reveal vacancy structure by time and place.",
    )
    add_bullet(
        doc,
        "Deliver a usable GenAI product that turns predictions into natural-language, voice-friendly recommendations for drivers.",
    )

    add_heading_custom(doc, "1.3 Research Question", 2)
    add_para(
        doc,
        "Can we predict whether a car park will have private-car vacancies (classification) and the number of vacancies "
        "(regression) 30–60 minutes ahead, using official real-time data and time/location features—and outperform a "
        "same-weekday–same-hour historical average baseline?",
        align="justify",
        italic=True,
        first_line=False,
    )

    # 2 Dataset
    add_heading_custom(doc, "2. Dataset Description", 1)
    add_heading_custom(doc, "2.1 Introduction to the Dataset", 2)
    add_para(
        doc,
        "The primary data source is the Transport Department / data.gov.hk Carpark Information and Vacancy API, which "
        "provides near real-time private-car vacancy counts, park metadata (name, district, coordinates), and fee fields "
        "where available. Complementary weather snapshots come from the Hong Kong Observatory Weather API (temperature, "
        "humidity, and related regional observations). Product-side APIs (DeepSeek Chat, Google Maps Geocoding / Maps "
        "JavaScript, Google Speech Recognition, and Microsoft Edge Text-to-Speech) are used for the interactive agent, "
        "not as training labels.",
        align="justify",
    )
    add_para(
        doc,
        "Geographic scope evolved from an early Kwun Tong / Sha Tin pilot to a territory-wide coverage of roughly 18 "
        "Hong Kong districts and about 587–592 car parks in the working inventory. The modelling and product focus is "
        "car-park vacancy for private cars; on-street metered spaces are out of scope.",
        align="justify",
    )

    add_heading_custom(doc, "2.2 Dataset Characteristics", 2)
    add_table(
        doc,
        ["Characteristic", "Approximate value"],
        [
            ["Car parks (metadata inventory)", "~587–592 parks across ~18 districts"],
            [
                "Historical baseline window",
                "365 days (≈ 34.9k snapshots; ~238k aggregated baseline rows; ~511 parks in baseline build)",
            ],
            [
                "Live / local vacancy snapshots (SQLite backup)",
                "~39k vacancy records; ~69 weather records (growing with collection)",
            ],
            ["Collection cadence", "About every 10 minutes"],
            [
                "Target variables",
                "vacancy count (regression); has_vacancy = vacancy > 0 (classification)",
            ],
            [
                "Key features",
                "hour / half-hour slot, day of week, weekend flag, park_id, district, temperature",
            ],
            [
                "Class balance note",
                "Many parks often have vacancy > 0; binary F1 is high but still useful as a gate for ranking",
            ],
        ],
        col_widths=[2.8, 3.7],
    )

    add_figure(
        doc,
        f("fig02_carparks_by_district"),
        width=5.8,
        caption=(
            "Figure 1. Car-park counts by administrative district (left) and top Google area clusters (right). "
            "Kwun Tong and Sha Tin have the densest coverage in the inventory."
        ),
    )

    add_heading_custom(doc, "2.3 Data Preprocessing", 2)
    add_bullet(
        doc,
        "Filter private-car vacancy fields; treat negative / missing vacancy as unavailable for modelling and ranking.",
    )
    add_bullet(doc, "Join park metadata (district, coordinates, names) to vacancy snapshots on park_id.")
    add_bullet(doc, "Derive temporal features in Hong Kong time (hour, half-hour slot, day of week, weekend).")
    add_bullet(
        doc,
        "Build capacity proxies from observed max vacancy / baseline means where official “space” coverage is sparse (~5%).",
    )
    add_bullet(
        doc,
        "Aggregate a 365-day historical ZIP into a compact vacancy_baselines.parquet summary (park × half-hour × DOW means) for scalable scoring.",
    )
    add_bullet(
        doc,
        "Optional shrinkage toward broader priors when a specific slot has few observations (mirrors production app logic).",
    )
    add_bullet(
        doc,
        "Time-based train/test split (70/30) to avoid future leakage when evaluating Gradient Boosting models.",
    )

    # 3 Methodology
    add_heading_custom(doc, "3. Methodology", 1)
    add_heading_custom(doc, "3.1 Research Approach", 2)
    add_para(
        doc,
        "We treat parking vacancy as a structured spatiotemporal prediction problem rather than a pure “current map” "
        "lookup. The approach combines (i) continuous ETL from official APIs, (ii) a strong historical baseline as the "
        "scientific control, (iii) Gradient Boosting models for lift beyond averages, and (iv) a GenAI agent that "
        "translates scores into driver-facing answers. The rationale is practical: drivers need forward-looking guidance, "
        "and any ML claim must beat an interpretable same-hour baseline built from a full year of history.",
        align="justify",
    )

    add_figure(
        doc,
        f("fig01_pipeline"),
        width=5.9,
        caption=(
            "Figure 2. System pipeline from the project PowerPoint: Collect → Store → Train / Serve → Streamlit app "
            "「咪再兜」(Maps + DeepSeek intent; STT/TTS as UX only)."
        ),
    )

    add_heading_custom(doc, "3.2 Data Collection & ETL", 2)
    add_para(
        doc,
        "Collection scripts and notebooks (collect.ipynb / scripts/collect_to_supabase.py) poll the Transport Department "
        "vacancy API and HKO weather endpoints approximately every 10 minutes. Records are transformed into normalized "
        "tables—carparks, vacancy_records, weather_records—and written to local SQLite (parking.db) as well as mirrored "
        "to Supabase Postgres for cloud reads by the Streamlit app. Historical ZIP imports are summarized into baseline "
        "parquet files to avoid saturating free-tier cloud disk with raw history.",
        align="justify",
    )

    add_heading_custom(doc, "3.3 Analytical Techniques", 2)
    add_para(doc, "Baseline model.", bold=True, first_line=False, space_after=2, color=COLOR_H2, size=11)
    add_para(
        doc,
        "For each park, half-hour slot, and weekday, we compute the mean private-car vacancy from up to 365 days of "
        "history. This baseline is strong and interpretable: it already encodes weekly seasonality and park-specific "
        "capacity behaviour.",
        align="justify",
    )
    add_figure(
        doc,
        f("fig04_baseline_curve"),
        width=5.5,
        caption="Figure 3. Example park-level baseline curve comparing weekday vs weekend mean vacancy across hours of day.",
    )

    add_para(doc, "Machine Learning models.", bold=True, first_line=False, space_after=2, color=COLOR_H2, size=11)
    add_para(
        doc,
        "We train a Gradient Boosting Regressor to predict vacancy count and a Gradient Boosting Classifier to predict "
        "has_vacancy (vacancy > 0). Features include hour, day of week, weekend flag, encoded park/district identifiers, "
        "and temperature when available. Evaluation uses a chronological 70/30 train/test split. Primary metrics are MAE "
        "for regression and F1 for classification, with explicit comparison to the historical baseline MAE.",
        align="justify",
    )

    add_para(doc, "Product / Generative AI layer.", bold=True, first_line=False, space_after=2, color=COLOR_H2, size=11)
    add_para(
        doc,
        "The Streamlit app 「咪再兜」 uses DeepSeek to parse Cantonese/Chinese natural-language intent (place, time, and "
        "query type such as “how easy to park here” vs “where nearby is easier”). Google Geocoding locates the place; "
        "nearby parks are scored with live vacancy plus baseline/ML predictions; Speech Recognition and Edge TTS support "
        "hands-busy driving scenarios. This GenAI layer does not replace the models—it packages predictions into usable guidance.",
        align="justify",
    )

    # 4 Analysis
    add_heading_custom(doc, "4. Analysis and Discussion", 1)
    add_heading_custom(doc, "4.1 Exploratory Insights", 2)
    add_para(
        doc,
        "EDA across all Hong Kong districts shows that vacancy is structured, not random. Capacity-weighted vacancy-rate "
        "heatmaps by district × hour reveal tighter midday conditions and freer nights in many areas, with Central Business "
        "District districts (Central & Western, Yau Tsim Mong, Wan Chai) often showing lower average vacancy rates than "
        "parts of the New Territories and Islands. These patterns motivate including time and location features and explain "
        "why a same-hour baseline is already competitive—and why ML can still add lift by combining park identity with "
        "context features.",
        align="justify",
    )
    add_figure(
        doc,
        f("fig03_district_hour_heatmap"),
        width=5.5,
        caption=(
            "Figure 4. Territory-wide weekday mean vacancy-rate heatmap (district × hour). "
            "Structured diurnal and district differences are visible."
        ),
    )

    add_para(
        doc,
        "Live-vs-baseline spot checks (one park per district) show that historical means usually track live vacancy "
        "directionally, while large facilities can deviate substantially at a given moment—exactly the regime where "
        "forward-looking prediction and nearby ranking help drivers decide.",
        align="justify",
    )
    add_figure(
        doc,
        f("fig05_live_vs_baseline"),
        width=5.4,
        caption="Figure 5. Live private-car vacancy versus same-slot baseline for one representative park per district.",
    )

    add_heading_custom(doc, "4.2 Model Results", 2)
    add_para(
        doc,
        "On the time-based holdout, Gradient Boosting outperforms the same-hour historical baseline on vacancy-count "
        "prediction: regression MAE improves from approximately 17.8 vacancies to 10.40. Classification of whether a park "
        "has any private-car vacancy achieves an F1 of about 0.957. Including temperature yielded a modest MAE improvement "
        "relative to a no-weather variant (~10.73 → 10.40) in early experiments, suggesting weather is a useful but still "
        "under-exploited signal.",
        align="justify",
    )
    add_figure(
        doc,
        f("fig06_model_metrics"),
        width=5.8,
        caption=(
            "Figure 6. Key model results from the PowerPoint (time-based 70/30 split): "
            "Classification F1 ≈ 0.957; Regression MAE = 10.40; Baseline MAE ≈ 17.8."
        ),
    )
    add_figure(
        doc,
        f("fig07_model_mae"),
        width=5.2,
        caption=(
            "Figure 7. Regression MAE comparison from the PowerPoint: "
            "Gradient Boosting versus same-hour historical baseline."
        ),
    )

    add_para(
        doc,
        "Beyond scalar metrics, district-level mean predicted vacancy rankings (e.g., Friday 19:00) illustrate how the "
        "system supports routing decisions: drivers need not only a yes/no vacancy flag, but a comparative view of where "
        "spaces are likelier to be available at a chosen future time.",
        align="justify",
    )
    add_figure(
        doc,
        f("fig08_district_ranking"),
        width=5.0,
        caption="Figure 8. Example Friday 19:00 mean predicted vacancy ranking across all districts.",
    )

    add_heading_custom(doc, "4.3 Potential Reasons for Observed Patterns", 2)
    add_bullet(doc, "Weekly and hourly seasonality dominate parking demand (commuting, shopping, dining).")
    add_bullet(
        doc,
        "District land-use mix (CBD offices vs residential New Territories) shifts when parks fill and empty.",
    )
    add_bullet(doc, "Park-specific capacity and access rules create large level differences that park_id features capture.")
    add_bullet(
        doc,
        "Short-horizon weather effects exist but need longer variation (rain, warnings) to dominate time features.",
    )
    add_bullet(
        doc,
        "High binary F1 partly reflects that many parks often retain some vacancy; ranking and count MAE remain the harder practical problems.",
    )

    add_heading_custom(doc, "4.4 Product Demonstration Evidence", 2)
    add_para(
        doc,
        "The product screenshots below are taken from the live Streamlit Cloud demo "
        "(https://hk-parking-vacancy-prediction.streamlit.app/). They show the full loop from home screen → "
        "natural-language query → AI intent parsing → ranked nearby car parks → map → day-ahead vacancy trend. "
        "The demo query is 「啟德附近下星期五下晝兩點半邊度比較易泊車？」 (Near Kai Tak, where is it easier "
        "to park next Friday at 2:30 p.m.?). DeepSeek parses this into place ≈ Kai Tak, Kowloon, Hong Kong, "
        "date = Friday 13 Nov 2026, and time = 2:30 p.m. The UI then returns the top 10 nearby parks (default when "
        "count is unspecified), with distance and fee fields (fee shown as 未提供 when open data has no rule for the "
        "selected period), a colour-coded map, and a Friday all-day predicted-vacancy trend for the leading parks.",
        align="justify",
    )
    add_figure(
        doc,
        f("fig12_app_home"),
        width=5.5,
        caption="Figure 9. 「咪再兜」 home screen (live Streamlit demo).",
    )
    add_figure(
        doc,
        f("fig09_app_query"),
        width=5.4,
        caption=(
            "Figure 10. Natural-language query and AI intent panel "
            "(Kai Tak · next Friday 2:30 p.m.)."
        ),
    )
    add_figure(
        doc,
        f("fig10_app_results"),
        width=5.4,
        caption=(
            "Figure 11. Ranked nearby car parks for Friday 2:30 p.m. near Kai Tak "
            "(top 10; distance in km; predicted vacancy)."
        ),
    )
    add_figure(
        doc,
        f("fig11_app_map"),
        width=5.4,
        caption=(
            "Figure 12. Map view for the same Kai Tak query "
            "(colour = vacancy tightness; number = predicted vacancy)."
        ),
    )
    add_figure(
        doc,
        f("fig14_app_trend"),
        width=5.4,
        caption=(
            "Figure 13. Friday all-day predicted vacancy trend for leading parks near Kai Tak "
            "(vertical marker at 14:30)."
        ),
    )
    add_para(
        doc,
        "Live app URL: https://hk-parking-vacancy-prediction.streamlit.app/",
        bold=True,
        first_line=False,
        space_after=2,
        color=COLOR_H1,
        size=11,
    )
    add_para(
        doc,
        "GitHub repository: https://github.com/keithlky1992/hk-parking-vacancy-prediction",
        bold=True,
        first_line=False,
        space_after=4,
        color=COLOR_H1,
        size=11,
    )
    add_para(
        doc,
        "Scan the QR code below to open 「咪再兜」 on Streamlit Cloud and try the same natural-language queries.",
        first_line=False,
        space_after=6,
    )
    add_figure(
        doc,
        f("fig15_app_qr"),
        width=1.8,
        caption="Figure 14. QR code for the live Streamlit app (https://hk-parking-vacancy-prediction.streamlit.app/).",
    )

    # 5 Challenges
    add_heading_custom(doc, "5. Challenges and Limitations", 1)
    add_heading_custom(doc, "5.1 Challenges Faced", 2)
    add_bullet(
        doc,
        "Early live collection depended on a continuously running machine; sleep/gaps created missing timestamps.",
    )
    add_bullet(doc, "Cloud free-tier disk limits constrained importing full raw historical vacancy into Supabase.")
    add_bullet(doc, "Official capacity (“space”) fields are sparse, requiring proxy capacity for rate-based EDA.")
    add_bullet(
        doc,
        "Integrating many APIs (TD, HKO, Supabase, DeepSeek, Google, Speech, TTS) increased operational complexity.",
    )
    add_bullet(
        doc,
        "Balancing scientific evaluation (baseline vs ML) with a polished bilingual driving-oriented UX under a short Capstone timeline.",
    )

    add_heading_custom(doc, "5.2 Limitations of Approach / Dataset", 2)
    add_bullet(
        doc,
        "Scope covers car parks only—not on-street meters—so results must not be claimed as “all parking in HK.”",
    )
    add_bullet(
        doc,
        "Early ML training windows were short before the 365-day baseline expansion; some metrics reflect limited diversity.",
    )
    add_bullet(doc, "Weather signals were collected but remain underused beyond temperature in the reported models.")
    add_bullet(doc, "High classification F1 does not imply equally precise count forecasts for every park size.")
    add_bullet(doc, "Speech and LLM components introduce external service dependency and latency.")

    add_heading_custom(doc, "5.3 Potential Areas for Improvement", 2)
    add_bullet(doc, "Move collection to cloud cron / Edge Functions for true 24×7 reliability.")
    add_bullet(doc, "Persist trained model artifacts to storage and automate retraining.")
    add_bullet(doc, "Add rainfall, warning signals, public holidays, and special-event flags.")
    add_bullet(doc, "Introduce multi-class vacancy levels (low / medium / high) alongside regression.")
    add_bullet(doc, "Deeper error analysis by district, hour, and park capacity band.")

    # 6 Future Work
    add_heading_custom(doc, "6. Future Work", 1)
    add_para(
        doc,
        "Ideas considered but not fully implemented—mainly due to time, cloud-disk, and operational constraints—include: "
        "scheduled cloud collection; richer weather and holiday feature sets; storing and versioning trained Gradient "
        "Boosting models; vacancy-level classification; stronger LLM-generated suggestion copy grounded in model "
        "confidence; and systematic residual analysis. The current product already ships a working agent with baseline "
        "scoring; these items are the natural next engineering milestones after Capstone submission.",
        align="justify",
        first_line=False,
    )

    # 7 Conclusion
    add_heading_custom(doc, "7. Conclusion", 1)
    add_para(
        doc,
        "This Capstone project set out to predict Hong Kong private-car parking vacancy 30–60 minutes ahead and to prove "
        "that Machine Learning can beat a strong same-hour historical baseline while packaging results into a usable "
        "driver product. We delivered an end-to-end data science stack—API ETL, SQL/Postgres storage, EDA, baseline and "
        "Gradient Boosting models, and a Generative AI Streamlit agent—aligned with the course technical skills checklist "
        "(Python, SQL, ETL, APIs, EDA, visualization, ML, and GenAI).",
        align="justify",
    )
    add_para(
        doc,
        "Key takeaways: (1) parking vacancy in Hong Kong has learnable structure by district and hour; (2) a year-long "
        "same-park × hour × weekday baseline is a fair and necessary control; (3) Gradient Boosting improved regression "
        "MAE from ~17.8 to 10.40 and reached ~0.957 classification F1 on the reported split; (4) GenAI is most valuable "
        "when it orchestrates maps, voice, and ranked predictions rather than inventing vacancies. Together, these results "
        "affirm the research question and provide a practical foundation for continued data collection and model hardening.",
        align="justify",
    )

    # 8 References
    add_heading_custom(doc, "8. References", 1)
    add_bullet(doc, "Transport Department / data.gov.hk — Carpark Vacancy Data API.")
    add_bullet(doc, "Hong Kong Observatory — Weather information APIs.")
    add_bullet(doc, "Supabase — Postgres backend used for carparks, vacancy_records, and weather_records.")
    add_bullet(doc, "DeepSeek — Chat API for natural-language intent parsing and agent replies.")
    add_bullet(doc, "Google Maps Platform — Geocoding and Maps JavaScript APIs.")
    add_bullet(doc, "Google Speech Recognition API; Microsoft Edge Text-to-Speech (Cantonese output).")
    add_bullet(doc, "scikit-learn — Gradient Boosting Regressor / Classifier.")
    add_bullet(doc, "Streamlit — Product deployment (https://hk-parking-vacancy-prediction.streamlit.app/).")
    add_bullet(
        doc,
        "Project repository: https://github.com/keithlky1992/hk-parking-vacancy-prediction "
        "(explore_api_en.ipynb / explore_api_zh.ipynb, app.py, parking_db.py, scripts/).",
    )

    # 9 Distribution of Work
    add_heading_custom(doc, "9. Distribution of Work", 1)
    add_table(
        doc,
        ["Member", "Primary responsibilities"],
        [
            [
                "Lee Ki Yick (Keith)",
                "Product/app (Streamlit「咪再兜」), GenAI agent wiring (DeepSeek/Maps/Speech/TTS), cloud data layer (Supabase), demo & presentation narrative",
            ],
            [
                "Chin Ka Wang (Kelvin)",
                "Data collection & ETL design, baseline/ML modelling & evaluation, EDA notebooks and charts, methodology documentation",
            ],
            [
                "Shared",
                "Research question framing, report/presentation content, figure selection, final deliverable packaging",
            ],
        ],
        col_widths=[2.2, 4.3],
    )
    add_para(
        doc,
        "Note: Exact effort shares were collaborative; the table summarises primary ownership for assessment clarity.",
        size=8,
        italic=True,
        first_line=False,
        color=COLOR_MUTED,
    )

    # Appendix
    add_heading_custom(doc, "Appendix A. Technical Skills Checklist Mapping", 1)
    add_table(
        doc,
        ["Course skill", "How this project uses it"],
        [
            ["Python programming", "ETL scripts, notebooks, Streamlit app, modelling utilities"],
            ["SQL & database management", "SQLite + Supabase Postgres schemas and queries"],
            ["ETL processes", "Scheduled ~10-min collect/join of vacancy + weather"],
            ["API integration", "TD, HKO, Supabase, DeepSeek, Google Maps, Speech, Edge TTS"],
            ["EDA", "District/hour heatmaps, baseline curves, live-vs-baseline checks"],
            ["Data visualization", "Matplotlib charts embedded in notebooks, slides, and this report"],
            ["Statistical / ML modelling", "Historical baseline; Gradient Boosting regression & classification"],
            ["Generative AI", "DeepSeek intent + natural-language recommendations; voice I/O"],
        ],
        col_widths=[2.3, 4.2],
    )

    add_heading_custom(doc, "Appendix B. How to Reproduce", 1)
    add_bullet(
        doc,
        "Clone the GitHub repository (https://github.com/keithlky1992/hk-parking-vacancy-prediction) "
        "and create a Python environment from requirements.txt.",
    )
    add_bullet(doc, "Configure .env with SUPABASE_URL / keys (and other API keys required by the app).")
    add_bullet(doc, "Run scripts/build_vacancy_baselines.py to regenerate baseline summaries if needed.")
    add_bullet(doc, "Run streamlit run app.py locally, or open the deployed Streamlit Cloud URL.")
    add_bullet(
        doc,
        "Use explore_api_en.ipynb (or explore_api_zh.ipynb) for EDA aligned with the app scoring logic.",
    )
    add_bullet(doc, "Optional: scripts/collect_to_supabase.py for continued ~10-minute collection.")

    add_para(
        doc,
        "— End of Final Report (References, Distribution of Work, and Appendix excluded from the 20-page body limit per Capstone Guideline) —",
        size=8,
        italic=True,
        align="center",
        first_line=False,
        space_after=6,
        color=COLOR_MUTED,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")

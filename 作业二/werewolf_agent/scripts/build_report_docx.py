"""Generate a publication-grade academic Word (.docx) report with architecture diagram, formulas, statistical tests, and appendix."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
PNG_DIR = ROOT / "outputs" / "figures_png"
OUT_DOCX = ROOT.parent / "NLP课程大作业二实验报告_赵梁名.docx"

REPORT_TITLE = "基于冻结大模型与轻量强化学习的狼人杀角色智能体"
FONT_BODY_CN = "宋体"
FONT_HEAD_CN = "黑体"
FONT_LATIN = "Times New Roman"
FONT_MONO = "Consolas"


def find_edge_exe() -> Optional[Path]:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def render_svgs_to_png() -> dict[str, Path]:
    """Render all 6 vector SVG charts to crisp 300-DPI PNGs using headless Edge."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    edge = find_edge_exe()
    if edge is None:
        raise RuntimeError("Microsoft Edge or Chrome not found for SVG->PNG conversion.")

    svg_specs = [
        ("fig0", "fig0_system_architecture.svg", 960, 500),
        ("fig1", "fig1_ppo_learning_curve.svg", 760, 390),
        ("fig2", "fig2_arch_and_symbolic_ablation.svg", 760, 380),
        ("fig3", "fig3_protocol_drift.svg", 820, 380),
        ("fig4", "fig4_m5_nl_winrate_and_cost.svg", 840, 480),
        ("fig5", "fig5_1v8_single_seat.svg", 920, 410),
    ]
    out_paths: dict[str, Path] = {}
    for key, fname, w, h in svg_specs:
        svg_path = FIG_DIR / fname
        png_path = PNG_DIR / fname.replace(".svg", ".png")
        html_path = PNG_DIR / f"_tmp_{key}.html"
        svg_text = svg_path.read_text(encoding="utf-8")
        html_path.write_text(
            f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
            f'html,body{{margin:0;padding:0;background:#ffffff;overflow:hidden;}}'
            f'svg{{display:block;width:{w}px;height:{h}px;}}'
            f'</style></head><body>{svg_text}</body></html>',
            encoding="utf-8",
        )
        cmd = [
            str(edge),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--window-size={w},{h + 24}",
            f"--screenshot={png_path}",
            html_path.as_uri(),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with Image.open(png_path) as im:
            im = im.convert("RGB")
            im.crop((0, 0, min(im.width, w * 2), min(im.height, h * 2))).save(png_path, dpi=(300, 300))
        html_path.unlink(missing_ok=True)
        out_paths[key] = png_path
        print(f"Rendered {png_path.name} ({png_path.stat().st_size} bytes)")
    return out_paths


# -------------------- Word XML Styling Helpers --------------------

def set_run_font(
    run,
    cn_font: str = FONT_BODY_CN,
    size_pt: float = 10.5,
    bold: bool = False,
    italic: bool = False,
    color: Optional[RGBColor] = None,
    superscript: bool = False,
    latin_font: str = FONT_LATIN,
):
    run.font.name = latin_font
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.superscript = superscript
    if color is not None:
        run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), latin_font)
    rFonts.set(qn("w:hAnsi"), latin_font)
    rFonts.set(qn("w:eastAsia"), cn_font)
    rFonts.set(qn("w:cs"), latin_font)


def add_page_number_field(run):
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls("w"))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls("w"))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls("w"))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls("w"))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


def set_paragraph_bottom_border(p, sz: str = "6", color: str = "000000"):
    pPr = p._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="{sz}" w:space="4" w:color="{color}"/></w:pBdr>'
    )
    pPr.append(pBdr)


def configure_document(doc: Document):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.4)
    sec.left_margin = Cm(2.7)
    sec.right_margin = Cm(2.7)
    sec.header_distance = Cm(1.3)
    sec.footer_distance = Cm(1.3)

    normal = doc.styles["Normal"]
    normal.font.name = FONT_LATIN
    normal.font.size = Pt(10.5)
    normal._element.get_or_add_rPr().insert(
        0,
        parse_xml(
            f'<w:rFonts {nsdecls("w")} w:ascii="{FONT_LATIN}" w:hAnsi="{FONT_LATIN}" w:eastAsia="{FONT_BODY_CN}" w:cs="{FONT_LATIN}"/>'
        ),
    )

    header = sec.header
    hp = header.paragraphs[0]
    hp.paragraph_format.space_before = Pt(0)
    hp.paragraph_format.space_after = Pt(4)
    pPr = hp._p.get_or_add_pPr()
    tabs = parse_xml(f'<w:tabs {nsdecls("w")}><w:tab w:val="right" w:pos="{int(15.6 * 567)}"/></w:tabs>')
    pPr.append(tabs)
    set_paragraph_bottom_border(hp, sz="6", color="000000")

    r_left = hp.add_run(REPORT_TITLE)
    set_run_font(r_left, FONT_BODY_CN, 9.0)
    r_tab = hp.add_run("\t")
    set_run_font(r_tab, FONT_BODY_CN, 9.0)
    r_pg = hp.add_run()
    set_run_font(r_pg, FONT_BODY_CN, 9.0)
    add_page_number_field(r_pg)


def add_rich_paragraph(
    doc: Document,
    segments: Sequence[Tuple[str, bool, bool]],
    indent_chars: float = 2.0,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    space_before: float = 0.0,
    space_after: float = 4.0,
    line_spacing: float = 1.28,
    size_pt: float = 10.5,
    cn_font: str = FONT_BODY_CN,
):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if indent_chars > 0:
        pf.first_line_indent = Pt(size_pt * indent_chars)
    for text, is_bold, is_sup in segments:
        r = p.add_run(text)
        set_run_font(
            r,
            cn_font=FONT_HEAD_CN if (is_bold and cn_font == FONT_BODY_CN and len(text) <= 12) else cn_font,
            size_pt=size_pt,
            bold=is_bold,
            superscript=is_sup,
        )
    return p


def add_body(doc: Document, text: str, space_after: float = 4.5):
    """Parse lightweight inline markers: **bold** and ^[1] superscript citations."""
    text = text.replace("`", "")
    tokens = re.split(r"(\*\*.*?\*\*|\^\[[0-9,\-–]+\])", text)
    segs: List[Tuple[str, bool, bool]] = []
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            segs.append((tok[2:-2], True, False))
        elif tok.startswith("^[") and tok.endswith("]"):
            segs.append((tok[1:], False, True))
        else:
            segs.append((tok, False, False))
    return add_rich_paragraph(doc, segs, indent_chars=2.0, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=space_after)


def add_equation(doc: Document, formula: str, eq_num: str):
    """Add a clean academic centered display equation with right-aligned equation number."""
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tblPr = table._tbl.tblPr
    tblPr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'  <w:top w:val="none"/><w:bottom w:val="none"/>'
            f'  <w:left w:val="none"/><w:right w:val="none"/>'
            f'  <w:insideH w:val="none"/><w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
    )
    row = table.rows[0]
    row._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    c0, c1 = row.cells[0], row.cells[1]
    c0.width = Cm(14.0)
    c1.width = Cm(1.6)
    c0.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.space_before = Pt(3)
    p0.paragraph_format.space_after = Pt(3)
    r0 = p0.add_run(formula)
    set_run_font(r0, cn_font=FONT_BODY_CN, size_pt=10.0, italic=False, latin_font="Cambria Math")

    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p1.paragraph_format.space_before = Pt(3)
    p1.paragraph_format.space_after = Pt(3)
    r1 = p1.add_run(f"({eq_num})")
    set_run_font(r1, cn_font=FONT_BODY_CN, size_pt=10.0)


def add_code_box(doc: Document, title: str, content: str):
    """Add a clean shaded academic box for Prompt / Fact-Tracker snippets in the Appendix."""
    p_t = doc.add_paragraph()
    p_t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_t.paragraph_format.space_before = Pt(6)
    p_t.paragraph_format.space_after = Pt(2)
    p_t.paragraph_format.keep_with_next = True
    set_run_font(p_t.add_run(title), cn_font=FONT_HEAD_CN, size_pt=9.5, bold=True)

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tblPr = table._tbl.tblPr
    tblPr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            f'  <w:left w:val="single" w:sz="12" w:space="0" w:color="2563EB"/>'
            f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            f'</w:tblBorders>'
        )
    )
    tblPr.append(
        parse_xml(
            f'<w:tblCellMar {nsdecls("w")}>'
            f'  <w:top w:w="70" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/>'
            f'  <w:left w:w="120" w:type="dxa"/><w:right w:w="100" w:type="dxa"/>'
            f'</w:tblCellMar>'
        )
    )
    cell = table.rows[0].cells[0]
    cell.width = Cm(15.4)
    cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8FAFC"/>'))
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(content.strip())
    set_run_font(r, cn_font=FONT_BODY_CN, size_pt=8.5, latin_font=FONT_MONO)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def add_h1(doc: Document, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(14)
    pf.space_after = Pt(8)
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, cn_font=FONT_HEAD_CN, size_pt=14.0, bold=True)
    return p


def add_h2(doc: Document, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(10)
    pf.space_after = Pt(5)
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, cn_font=FONT_HEAD_CN, size_pt=11.5, bold=True)
    return p


def add_three_line_table(
    doc: Document,
    caption: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_widths_cm: Optional[Sequence[float]] = None,
    highlight_rows: Sequence[int] = (),
    font_size: float = 9.0,
    note: Optional[str] = None,
):
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_before = Pt(8)
    cp.paragraph_format.space_after = Pt(4)
    cp.paragraph_format.keep_with_next = True
    cr = cp.add_run(caption)
    set_run_font(cr, cn_font=FONT_BODY_CN, size_pt=9.5, bold=False)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        f'  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        f'  <w:left w:val="none"/><w:right w:val="none"/>'
        f'  <w:insideH w:val="none"/><w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)
    tblPr.append(
        parse_xml(
            f'<w:tblCellMar {nsdecls("w")}>'
            f'  <w:top w:w="52" w:type="dxa"/><w:bottom w:w="52" w:type="dxa"/>'
            f'  <w:left w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/>'
            f'</w:tblCellMar>'
        )
    )

    def format_cell(cell, text: str, bold: bool = False, align=WD_ALIGN_PARAGRAPH.CENTER, bottom_rule: bool = False, shade: Optional[str] = None):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = align
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.12
        r = p.add_run(text)
        set_run_font(r, cn_font=FONT_HEAD_CN if bold else FONT_BODY_CN, size_pt=font_size, bold=bold)
        tcPr = cell._tc.get_or_add_tcPr()
        if bottom_rule:
            tcPr.append(
                parse_xml(
                    f'<w:tcBorders {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>'
                )
            )
        if shade:
            tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{shade}"/>'))

    hdr_row = table.rows[0]
    trPr = hdr_row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    for j, htxt in enumerate(headers):
        cell = hdr_row.cells[j]
        if col_widths_cm:
            cell.width = Cm(col_widths_cm[j])
        format_cell(cell, htxt, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER, bottom_rule=True)

    for i, rdata in enumerate(rows):
        row = table.rows[i + 1]
        row._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        is_hl = i in highlight_rows
        for j, val in enumerate(rdata):
            cell = row.cells[j]
            if col_widths_cm:
                cell.width = Cm(col_widths_cm[j])
            format_cell(
                cell,
                val,
                bold=is_hl,
                align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER,
                shade="F1F5F9" if is_hl else None,
            )

    if note:
        np_p = doc.add_paragraph()
        np_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        np_p.paragraph_format.space_before = Pt(2)
        np_p.paragraph_format.space_after = Pt(5)
        np_p.paragraph_format.line_spacing = 1.15
        set_run_font(np_p.add_run(note), cn_font=FONT_BODY_CN, size_pt=8.5, color=RGBColor(71, 85, 105))
    else:
        spacer = doc.add_paragraph()
        spacer.paragraph_format.space_before = Pt(0)
        spacer.paragraph_format.space_after = Pt(4)


def add_figure(doc: Document, png_path: Path, caption: str, width_cm: float = 14.8):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(png_path), width=Cm(width_cm))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(8)
    r = p_cap.add_run(caption)
    set_run_font(r, cn_font=FONT_BODY_CN, size_pt=9.5)


def add_reference(doc: Document, idx: int, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.left_indent = Cm(0.85)
    pf.first_line_indent = Cm(-0.85)
    pf.space_before = Pt(1)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.2
    r = p.add_run(f"[{idx}]\t{text}")
    set_run_font(r, cn_font=FONT_BODY_CN, size_pt=9.5)


# -------------------- Build Full Academic Report --------------------

def build_report():
    pngs = render_svgs_to_png()
    doc = Document()
    configure_document(doc)

    # ===== Page 1: Title Block =====
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(14)
    p_title.paragraph_format.space_after = Pt(8)
    set_run_font(p_title.add_run(REPORT_TITLE), FONT_HEAD_CN, 18.0, bold=True)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(10)
    set_run_font(p_sub.add_run("自然语言处理课程实验报告"), FONT_HEAD_CN, 13.0, bold=True)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(0)
    p_meta.paragraph_format.space_after = Pt(5)
    set_run_font(p_meta.add_run("姓名 赵梁名    班级 __________    学号 3126336021"), FONT_BODY_CN, 10.5)

    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.paragraph_format.space_before = Pt(0)
    p_date.paragraph_format.space_after = Pt(14)
    set_run_font(p_date.add_run("2026 年 11 月"), FONT_BODY_CN, 10.5)

    # ===== Abstract & Keywords =====
    p_abs_h = doc.add_paragraph()
    p_abs_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_abs_h.paragraph_format.space_before = Pt(4)
    p_abs_h.paragraph_format.space_after = Pt(6)
    set_run_font(p_abs_h.add_run("摘要"), FONT_HEAD_CN, 12.0, bold=True)

    add_body(
        doc,
        "传统自然语言处理任务通常假设对话参与者诚实且合作，而现实中的多方博弈对话普遍存在信息不对称、隐藏意图与策略性欺骗。针对中文九人预女猎狼人杀场景，如何在**不微调大语言模型权重**且严格控制多智能体在线调用成本的前提下，训练能够胜任不同身份的角色扮演智能体，是一个兼具理论与工程价值的问题。本文提出一种**语言与决策解耦的双系统智能体框架**：底层构建支持纯符号高速推演（357–637 局/秒，零 API 费用）与真实中文自然语言对战的双模统一状态机，严格隔离各角色私有视角；决策层基于 MaKTO 人类对局重构 4,261 步第一人称专家轨迹，在本地 CPU 上训练席位共享信念网络（BeliefMLP）与**席位等变指针策略-价值网络**，经行为克隆冷启动及 4,800 局种群自博弈 Masked PPO 优化（引入角色条件化优势归一化与 KL 锚定抑制语言漂移）；语言层挂载冻结大语言模型（qwen3.5-27b，关闭思考模式），结合跨领域信念适应（BeliefMLP_NL）、System-2 客观事实矛盾核验器与防泄密护栏完成真实博弈。在 352 局零报错真实中文九人局（总费用 14.84 元）的受控评测中，核心 1v8 单席位随机分角主实验表明，本文精简双系统架构（Ours_DualSystem_v2）取得 **40.0% 五角色均权胜率**与 **44.4% 自然牌堆胜率**，较纯大模型基线（26.7% / 37.0%）分别提升 +13.3 与 +7.4 个百分点，好人四角色胜率由 8.3% 提升至 33.3%（配对分歧 7:1，精确 McNemar 检验 p = 0.070），在各自评测口径下达到与冻结大模型结合外部强化学习代表工作相当的参考胜率量级。本文同时通过消融实验系统揭示了语言漂移、自然语言信息瓶颈、第三人称转述误抽取与过度提示过拟合等关键边界现象。",
        space_after=6.0,
    )

    add_rich_paragraph(
        doc,
        [
            ("关键词 ", True, False),
            ("狼人杀智能体；冻结大语言模型；信念追踪；席位等变指针网络；强化学习", False, False),
        ],
        indent_chars=0.0,
        align=WD_ALIGN_PARAGRAPH.LEFT,
        space_after=12.0,
    )

    # ===== Section 1: Introduction =====
    add_h1(doc, "1 引言")
    add_h2(doc, "1.1 研究背景与非合作对话博弈挑战")
    add_body(
        doc,
        "大语言模型（LLM）在问答、摘要与代码生成等合作式语言任务中已取得显著进展，但当交互场景转向反诈识别、商业谈判与社交推理游戏等**非合作多方对话**时，纯语言模型往往暴露出系统性缺陷。中文九人“预女猎”狼人杀（3 名狼人、3 名平民、1 名预言家、1 名女巫、1 名猎人，屠边胜负）是研究非合作对话智能体的理想测试场：游戏规则严格、终局胜负与真实底牌完全客观，却要求智能体在极度不对称的视角下同时完成自然语言理解、谎言辨别、跨轮次逻辑核验与战略性伪装表达。"
    )
    add_body(
        doc,
        "与棋牌或电子游戏相比，狼人杀的独特难点在于**自然语言本身既是观测载体，又是动作空间**。预言家需要通过清晰的逻辑陈述争取好人信任并归票狼人，女巫与猎人需要在隐藏神职防屠神与及时动用技能之间权衡，而狼人则需通过悍跳、倒钩与抿神扰乱好人阵营判断。纯粹依赖大模型固有语言直觉不仅难以形成稳定的均衡策略，还容易出现轻信盲从、阅读幻觉与首日盲目暴露神职底牌等典型失误。"
    )

    add_h2(doc, "1.2 课题演进与低成本解耦动机")
    add_body(
        doc,
        "本课题第一阶段曾构建旁观者视角的狼人杀身份推断系统，在 14 局留出测试集上对比了朴素贝叶斯与多款大语言模型。前期实证暴露出三项关键现象：第一，纯大模型在非合作对话中存在明显的轻信倾向与推理幻觉；第二，过度复杂的提示工程或噪声中间抽取反而会拉低模型准确率；第三，在九人全量上下文对局试点中，仅单日发言即消耗约 49.7 万输入 Token，若直接在线使用纯 LLM 开展多轮强化学习采样，其 API 与算力成本在课程与常规科研条件下难以承受。"
    )
    add_body(
        doc,
        "结合开题汇报关于“为智能体分配真实角色并训练其参与完整对局”的建议，同时坚守**零大模型微调算力费用**与**极低在线推理预算**两项现实约束，本文转向**语言与决策解耦（Decoupled Dual-System）**技术路线：将可训练的信念推理与策略优化全部下沉至本地 CPU 轻量级神经网络中，在零 API 成本的符号模拟器内完成数千局种群自博弈训练，再以 System-2 决策模块的形式挂载到冻结中文大语言模型上参与真实自然语言博弈。"
    )

    # ===== Section 2: Related Work =====
    add_h1(doc, "2 相关工作")
    add_h2(doc, "2.1 端到端大模型微调与自博弈偏好优化")
    add_body(
        doc,
        "近期部分研究尝试通过端到端参数微调直接提升大语言模型的狼人杀博弈水平。Ye 等人提出 MaKTO^[1]，基于 Qwen2.5-72B 先利用人类专家对局进行监督微调（SFT），再通过大规模多智能体随机分角自博弈收集非配对胜负轨迹执行 KTO 偏好优化，在九人中文狼人杀自然牌堆口径下取得 61.0% 的高胜率，并开源了中文九人预女猎对局数据集。Xu 等人提出 LSPO^[2]，将自由发言映射到离散隐策略空间并交替执行反事实遗憾最小化（CFR）与 DPO 微调。Jiang 等人提出的 OMAR^[3]与 Liu 等人提出的 SPIRAL^[4]进一步探索了单模型多角色自博弈强化学习，其中 SPIRAL 提出的**角色条件化优势估计（RAE）**有效缓解了不同阵营基础胜率失衡带来的梯度方差问题，直接启发了本文 PPO 训练中的分角色优势归一化设计。然而，上述端到端路线均依赖 7B–72B 级别模型的多卡集群微调，算力门槛较高。"
    )

    add_h2(doc, "2.2 冻结大模型与外部可训练决策模块")
    add_body(
        doc,
        "为规避微调高成本并克服纯大模型的固有策略偏好（intrinsic biases），另一条主线将冻结语言模型与外部轻量策略网络解耦。Xu 等人（ICML 2024）^[5]指出纯 LLM 缺乏博弈均衡意识，提出由冻结 LLM 提议候选动作、再由外部种群强化学习策略选择器重排的双层架构，在狼人杀中取得好人 30%、狼人 70% 的代表性成绩。腾讯 AI Lab 提出的 Thinker^[6]将系统划分为负责语言感知的 System-1 与负责逻辑决策的 System-2，并依托 1.88 万局人类对局训练外部推理器。Jin 等人（NeurIPS 2024）^[7]在《一夜终极狼人》中引入显式信念建模（Belief Modeling）指导讨论战术意图。此外，ReCon^[8]与 Werewolf Arena^[9]分别从递归换位思考与标准化多模型对战基准角度提供了有益参照。表 1 对比了代表性工作与本文路线的核心差异。"
    )

    add_three_line_table(
        doc,
        "表 1 狼人杀智能体代表性工作与本文技术路线对比",
        ["代表工作", "核心范式", "是否微调 LLM", "显式信念/事实验证", "训练与算力成本"],
        [
            ["MaKTO [1]", "SFT + 多智能体自博弈 KTO", "是（72B）", "隐式文本笔记", "极高（多卡 GPU 集群）"],
            ["LSPO [2] / SPIRAL [4]", "隐空间 CFR+DPO / RAE 自博弈", "是（7B–32B）", "无显式信念头", "高（多卡 GPU 微调）"],
            ["Xu et al. (ICML 2024) [5]", "冻结 LLM 候选 + 外部 RL 选择", "否（冻结）", "无独立信念网络", "低（仅训轻量 RL）"],
            ["Thinker [6] / ONUW-RL [7]", "System-1/2 解耦 + 信念建模", "否/轻量", "显式信念分布", "中（需万局训练数据）"],
            ["本文方法（DualSystem_v2）", "冻结 27B + CPU Belief_NL + 指针 RL", "否（零微调）", "Belief_NL + 事实核验", "极低（本地 CPU + ¥14.8）"],
        ],
        col_widths_cm=[3.5, 3.9, 2.1, 3.0, 3.1],
        highlight_rows=[4],
        font_size=8.8,
    )

    # ===== Section 3: Methodology =====
    add_h1(doc, "3 系统架构与方法设计")
    add_h2(doc, "3.1 双模统一引擎、总体架构与严格视角隔离")
    add_body(
        doc,
        "为同时满足“零成本大规模强化学习”与“真实中文自然语言对战”两类需求，本文独立设计并实现了双模统一的中文九人狼人杀状态机引擎（WerewolfEngine），系统总体架构如图 1 所示：上方为离线符号训练环，在本地 CPU 上完成信念网络与指针策略网络训练并冻结导出权重；下方为在线中文双系统推理环，将冻结大模型（System-1）与本地决策模块（System-2）结合参与真实对局。"
    )
    add_figure(doc, pngs["fig0"], "图 1 语言与决策解耦的双系统狼人杀智能体总体架构（离线符号训练环与在线中文推理环）", width_cm=15.0)

    add_body(
        doc,
        "游戏规则冻结为标准九人预女猎屠边局（3 狼、3 民、1 预、1 女、1 猎，无警长以控制单局轮次与 Token 消耗；女巫仅首夜可自救且同夜不可双药；猎人吃毒不可开枪；平票最多复投 1 次）。引擎在底层强制构造第一人称观测对象（PlayerObservation），严格隔离全局上帝状态与单人私有信息：好人绝不可见狼人夜聊，普通玩家不可见预言家查验或女巫用药记录。引擎定义了 6 类带合法动作掩码的离散决策空间：夜间预言家查验（9 维）、狼人袭击（9 维）、女巫行动（11 维：空过/救/毒 1–9）、猎人开枪（10 维：空过/带 1–9）、白天放逐投票（10 维：弃票/投 1–9）以及白天发言战术意图（64 维 = 1 维低调表水 + 7 类带目标战术 × 9 席位）。在 Symbolic Mode 下，本机单核 CPU 实测吞吐达 357–637 局/秒（1,000 局压测 0 死锁、0 平局、0 API 费用）；在 NL Mode 下，同一状态机无缝接入冻结大模型生成真实中文发言并经混合解析器回写状态。"
    )

    add_h2(doc, "3.2 席位共享信念追踪网络与跨领域适应")
    add_body(
        doc,
        "我们从本地开源 MaKTO 原始日志中重构了 133 局第一人称因果轨迹（严格沿用按整局划分的 107 训练 / 12 验证 / 14 测试集，跨集零重叠，全部 4,261 步动作 0 非法），提取出 1,552 条第一人称信念快照。统一特征编码器为每个决策时点构造 19 维全局上下文 g（天数、存活人数、各角色声称计数、屠边临界标识等）与每席位 24 维个体特征 p_i（存活状态、私有查验/银水、公开身份声明、被发金水/查杀/踩/保及历史票型）。"
    )
    add_body(
        doc,
        "考虑到狼人杀中 1–9 号座位本质上是无序集合，信念追踪器（BeliefMLP，约 2 万参数）采用**席位共享架构**：将第 i 个席位编码为 91 维向量 [g, p_i, mean_j(p_j), max_j(p_j)]，经共享全连接网络 91 → 128 → 64 → 5 输出该席位属于五类身份的后验概率 b_i ∈ Δ^5，实现天然的座位排列等变性与 9 倍样本复用。推理阶段进一步叠加已知硬约束掩码（自身底牌、狼队友、预言家已验、公开翻牌猎人及唯一神职排除）。针对人类文本（MaKTO）与冻结大模型真实对局之间的语言分布偏移，本文进一步合并 MaKTO 训练集（107 局、1,224 快照）与非重叠历史 NL 回放（170 局、3,220 快照，来源于早期 80 局整阵营消融与 90 局未校准 1v8 实验），在本地 CPU 上耗时 15.4 秒完成跨领域适应训练，得到适应版 **BeliefMLP_NL**。"
    )

    add_h2(doc, "3.3 席位等变指针策略网络与种群自博弈强化学习")
    add_body(
        doc,
        "早期原型尝试采用扁平多层感知机（MultiHeadPolicyValueMLP）将 280 维拼接状态直接映射到各决策头，但实验表明扁平网络无法跨座位编号泛化。为此，本文设计了**席位等变指针策略-价值网络（SeatEquivariantPolicyValueNet）**：记第 i 个席位的实体表示为 e_i = [p_i, b_i] ∈ R^{29}，网络对全部 9 个席位共享编码器 f_θ（106 → 96 → 64）与指针打分头，生成席位隐向量 h_i 及第 k 类指向性动作的 Logits z_{i,k}："
    )
    add_equation(doc, "h_i = f_θ([g, e_i, mean_j(e_j), max_j(e_j)]),    z_{i,k} = w_k^T h_i + b_k   (i = 1,…,9)", "1")
    add_body(
        doc,
        "其中 k 覆盖查验、狼刀、毒药、开枪、放逐投票与 7 类带目标发言战术；弃票、空过、解药、低调表水等非指向性动作及状态价值 V_φ(s) 则由全局池化分支 [g, mean_j(e_j), max_j(e_j), mean_j(h_j), max_j(h_j)] 输出，并在 Softmax 前施加合法动作掩码。当输入席位任意置换时，指针打分输出同步等变，有效消除座位编号过拟合。"
    )
    add_body(
        doc,
        "全部训练基于纯 NumPy 手写反向传播与 Adam 优化器在本地 CPU 完成，分为两阶段：（1）**离线行为克隆冷启动（BC）**：在 MaKTO 4,261 步人类专家轨迹上做带合法掩码的加权交叉熵学习（胜方轨迹权重 1.35）；（2）**种群自博弈 Masked PPO**：在符号环境中混合当前策略自博弈、历史快照池与启发式规则对手训练 4,800 局（100 轮 × 48 局）。针对狼人与好人基础胜率失衡问题，引入受 SPIRAL^[4]启发的**角色条件化优势归一化（RAE）**，按当前角色 r(t) 在每个采样批次内独立标准化优势值："
    )
    add_equation(doc, "Â_t = (A_t − μ_{r(t)}) / (σ_{r(t)} + ε),    其中  A_t = G_t − V_φ(s_t)", "2")
    add_body(
        doc,
        "在此基础上，为防止纯符号自博弈中策略演变为偏离人类语义的“暗号协议”，在 Masked PPO 目标函数中引入相对于 BC 锚点策略 π_BC 的 KL 散度正则项，并对发言决策头施加 5 倍锚定强度（β_speech = 5 β_base = 0.025，其余决策头 β_base = 0.005）："
    )
    add_equation(doc, "L(θ, φ) = E_t[ min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t) ] − c_v L_V(φ) + c_e H(π_θ(·|s_t)) − β_d KL(π_θ(·|s_t) || π_BC(·|s_t))", "3")

    add_h2(doc, "3.4 冻结大模型双系统融合、事实核验与护栏")
    add_body(
        doc,
        "直接用符号 RL 动作驱动大模型发言（纯解耦模式）会遭遇自然语言信息瓶颈：离散意图难以保留发言中的细微逻辑漏洞与多轮语境。本文最终构建 **System-1（冻结 LLM）+ System-2（BeliefMLP_NL + 客观事实核验 + Pointer RL）** 双系统融合架构：首先，精简上下文构造器保留最近 8 条原文发言，将单局消耗压缩至 1.5–3 万 Token；混合 NL → State 解析器采用“严格第一人称锚定正则快路径 + 模糊样本 LLM 兜底”同步状态机。其次，System-2 客观事实核验器（fact_tracker.py）自动追踪查验账本并反证逻辑铁狼（例如对好人本人或已翻牌猎人发查杀者直接标为必狼），阻断纯 LLM 常见的“金水误读为查杀”幻觉。在投票、开枪与用药等离散决策阶段，若好人视角已存在存活逻辑铁狼，则由 System-2 铁狼护栏直接优先锁定；否则由 System-1 冻结大模型结合原文、事实核验与高置信信念提示提出 Top-2 候选动作 {c_1, c_2}，再由 System-2 指针 RL 按以下融合准则重排定夺："
    )
    add_equation(doc, "a* = argmax_{a ∈ {c_1, c_2}} [ (1 − λ_r) · s_LLM(a) + λ_r · (log π_RL(a | s) + Δ_fact(a)) ]", "4")
    add_body(
        doc,
        "其中排序先验 s_LLM(c_1) = 0、s_LLM(c_2) = −0.50，Δ_fact(a) 为事实核验器赋予铁狼/铁好人或狼队抿神目标的对数偏置，融合权重按阵营非对称设定：好人侧 λ_r = 0.50 以充分依托 System-2 逻辑与策略先验，狼人侧放松为 λ_r = 0.15 以保留大模型的自然倒钩与语言伪装自由度。发言阶段则由自动 Guardrail 护栏拦截狼人自爆、暴露队友或编造终局等违规输出。表 2 汇总了本系统的关键实现细节、训练数据构成与超参数设置。"
    )

    add_three_line_table(
        doc,
        "表 2 系统关键实现细节、训练数据构成与核心超参数配置",
        ["模块 / 类别", "关键参数与实现细节"],
        [
            ["BeliefMLP_NL 跨领域数据", "训练集：MaKTO 107 局（1,224 快照）+ 非重叠历史 NL 170 局（80 局整阵营 + 90 局未校准 1v8，3,220 快照）；留出验证集：40 局历史 NL（738 快照，5,328 未知席位）；Adam lr = 2×10⁻³, batch = 256, epochs = 45。"],
            ["离线 BC 冷启动", "MaKTO 107 局（4,261 步全阶段轨迹）；Adam lr = 2×10⁻³, weight_decay = 10⁻³, batch = 128, epochs = 35，胜方轨迹损失权重 1.35。"],
            ["种群自博弈 Masked PPO", "100 轮 × 48 局 = 4,800 局（本地 CPU）；γ = 0.97, clip ε = 0.2, ppo_epochs = 3, batch = 256, 初始 lr = 10⁻³（线性衰减至 0.2×）, c_e = 0.015, β_base = 0.005（发言头 5×）；每 15 轮入快照池（上限 8 个，常驻初始 BC）；对手采样 35% 自博弈 / 30% 好人对规则狼 / 20% 狼人对规则好人 / 15% 快照与随机。"],
            ["启发式规则基线（Rule）", "基于第一人称可观测事件维护动态怀疑度：预言家优先报活狼查杀；狼人以 35% 概率单狼悍跳且夜间优先刀公开神职；女巫首夜救人、次夜对怀疑度 ≥ 2.0 目标泼毒；猎人对怀疑度 ≥ 0.5 或残局（≤ 5 人）开枪。"],
            ["冻结 LLM 推理配置", "阿里云百炼 qwen3.5-27b（enable_thinking = False）；发言 temperature = 0.50–0.60, max_tokens = 180；离散决策 JSON temperature = 0.20–0.30, max_tokens = 90–96；top_p 沿用服务端默认。"],
        ],
        col_widths_cm=[3.8, 11.8],
        font_size=8.4,
    )

    # ===== Section 4: Experiments & Results =====
    add_h1(doc, "4 实验设置与结果分析")
    add_h2(doc, "4.1 实验协议、评价指标与工程正确性验证")
    add_body(
        doc,
        "为贴合“为单个智能体随机分配角色并检验其跨角色综合博弈能力”的核心定位，本文确立**“1v8 单席位随机分角为主实验，整阵营对抗与跨模型挑战为补充实验”**的双层评测体系。在 1v8 主实验中，每局仅让 1 个席位运行被测系统，按预言家、女巫、猎人、狼人、平民五类角色分层均衡分配，其余 8 个席位固定为基线对手，且全部对比系统在完全相同的发牌与座位随机种子上配对运行。除五角色均权胜率（5Role）外，本文同步报告与 MaKTO^[1]同口径的**九人自然牌堆加权胜率（Deck9 = (预 + 女 + 猎 + 3×狼 + 3×民)/9）**、好人四角色均值（Good4）、单兵放逐投票准确率、95% Wilson 置信区间、精确配对 McNemar 检验及实测 API 费用。正式评测前，工程全部 **23 项自动化单元测试**（覆盖屠边规则、20 局视角隔离、Belief/Flat-PPO/Pointer-PPO 有限差分梯度校验、第一人称正则防串句、事实核验器与预算熔断）均全部通过。"
    )

    add_h2(doc, "4.2 符号环境训练、架构消融与语言漂移现象")
    add_body(
        doc,
        "在 MaKTO 14 局留出测试集未知席位上，席位共享 BeliefMLP 仅耗时 **2.59 秒 CPU** 即达到 **67.00% 身份预测准确率**（较 28.40% 均匀先验提升 +38.6pp），Macro-F1 为 **64.65%**，Wolf-F1 为 **59.75%**。策略网络架构与符号环境 300 局标准对照消融汇总于表 3，PPO 学习曲线与消融柱状图分别见图 2 与图 3。需要说明的是，图 2 展示的是训练过程中每 480 局在固定验证种子（20000/30000，每阵营 150 局）上的曲线轨迹（第 4,320 局验证峰值综合胜率 72.7%，第 4,800 局终点 71.7%），而表 3 与图 3 报告的是选取验证最优检查点后，在全新独立测试种子（50000/60000，每阵营 300 局）上的无偏评测结果（66.8%），因此两者存在约 4.9 个百分点的合理泛化落差。"
    )

    add_three_line_table(
        doc,
        "表 3 策略网络架构消融与符号环境 300 局标准对照结果（独立测试种子 50000/60000，对阵规则对手）",
        ["智能体 / 网络架构", "MaKTO 测试集 BC 准确率（总/投票/发言）", "好人胜率", "好人投狼率", "狼人胜率", "综合胜率"],
        [
            ["Random（随机基线）", "—", "1.3%", "35.2%", "37.0%", "19.2%"],
            ["Rule（启发式规则基线）", "—", "16.7%", "55.9%", "81.0%", "48.8%"],
            ["Flat MLP（扁平网络 BC+RL）", "24.1% / 28.5% / 19.4%", "10.0%", "39.1%", "39.3%", "24.7%"],
            ["Pointer（仅离线 BC 冷启动）", "47.4% / 60.6% / 34.6%", "6.0%", "34.9%", "64.7%", "35.3%"],
            ["Pointer（Ours，BC+KL 锚定 PPO）", "47.4% / 60.6% / 34.6%", "44.7%", "68.1%", "89.0%", "66.8%"],
            ["Pointer（RL_scratch，无 KL 锚定）", "—", "100.0%*", "98.9%*", "99.3%*", "99.7%*"],
        ],
        col_widths_cm=[4.4, 3.8, 1.7, 1.9, 1.7, 1.9],
        highlight_rows=[4],
        font_size=8.6,
        note="注：* 标示的 RL_scratch（kl_coef = 0）因放弃常规平民发言、全员高频刷神职声明形成“符号暗号协议”而获得极高表面胜率，无法直接接入自然语言环境（见图 4 审计）。",
    )

    add_body(
        doc,
        "表 3 与图 3 验证了两项关键机理：首先，**席位等变性至关重要**——指针网络将 MaKTO 离线模仿准确率翻倍（24.1% → 47.4%，投票达 60.6%），并将自博弈 RL 综合胜率从扁平 MLP 的 24.7% 显著提升至 **66.8%（+42.1pp）**，好人胜率较规则基线提升 +28.0pp。其次，对比有无 KL 锚定的 RL 策略揭示了经典的**语言漂移（Protocol Drift）现象**（图 4）：无 BC 约束的 RL_scratch 虽在符号环境中取得 99.7% 表面胜率，但 30 局发言意图审计显示其几乎完全放弃了平民常规发言（低调表水与认民点踩均为 0%），演化为全员高频刷神职声明的暗号通信；若直接接入冻结 LLM 会产出偏离人类规则的荒谬对话。加入 BC 冷启动与 KL 锚定的 Belief+BC+RL 则在保持自然发言分布的同时取得了稳健的胜率收益。"
    )

    add_figure(doc, pngs["fig1"], "图 2 席位等变指针网络 0–4,800 局种群自博弈 Masked PPO 学习曲线（本地 CPU，验证种子 20000/30000）", width_cm=14.6)
    add_figure(doc, pngs["fig2"], "图 3 符号环境 300 局架构与方法消融对照（独立测试种子 50000/60000，扁平 MLP vs 席位等变指针网络）", width_cm=14.6)
    add_figure(doc, pngs["fig3"], "图 4 30 局发言意图分布审计：无 KL 锚定的语言漂移现象与 BC+RL 锚定效果", width_cm=14.6)

    add_h2(doc, "4.3 核心主实验：1v8 单席位随机分角评测")
    add_body(
        doc,
        "我们首先在符号环境中开展了每系统 1,000 局（共 4,000 局，0 API 费）的 1v8 单席位大样本评测（表 4）。结果表明，Belief+BC+RL 在总胜率（32.1%）、好人四角色均值（18.8%）、猎人（25.0%）、女巫（20.5%）及单兵投准率（71.0%）上均优于随机、规则与纯 BC 基线。"
    )

    add_three_line_table(
        doc,
        "表 4 符号环境 1v8 单席位大样本评测（每系统 1,000 局 vs 8 启发式规则对手，0 API 费）",
        ["单席位系统", "5 角色总胜率", "好人 4 角均值", "预言家", "女巫", "猎人", "平民", "狼人", "单兵投准率"],
        [
            ["Random", "17.9%", "7.5%", "3.0%", "7.5%", "9.0%", "10.5%", "59.5%", "41.7%"],
            ["Rule（规则基线）", "31.6%", "17.6%", "16.5%", "18.0%", "15.5%", "20.5%", "87.5%", "67.6%"],
            ["Belief+BC", "24.4%", "12.9%", "13.0%", "9.0%", "17.0%", "12.5%", "70.5%", "60.4%"],
            ["Belief+BC+RL（Ours）", "32.1%", "18.8%", "14.0%", "20.5%", "25.0%", "15.5%", "85.5%", "71.0%"],
        ],
        col_widths_cm=[3.4, 1.8, 1.9, 1.4, 1.4, 1.4, 1.4, 1.4, 1.5],
        highlight_rows=[3],
        font_size=8.6,
    )

    add_body(
        doc,
        "随后在真实中文自然语言环境（阿里云百炼 qwen3.5-27b，关闭思考模式）开展 1v8 单席位评测。首轮未校准 90 局诊断实验（B0 23.3%、Ours_DualSystem 23.3%，费用 3.57 元）揭示了两个隐藏瓶颈：第一，背景 PureLLM 女巫首日盲目暴露身份率高达 **73%（22/30）**、猎人首日自爆率达 **70%（21/30）**，导致 28/30 局（93.3%）以狼队屠神终结、背景好人胜率被压至 3.3%；第二，仅在 MaKTO 人类数据上训练的 BeliefMLP 在真实 LLM 对局上存在跨领域偏移，未知席位准确率由 67.0% 降至 **50.09%**（NLL 1.4668）。为此，我们一方面统一为全部 9 个席位补充屠边防首日盲跳常识校准博弈环境，另一方面利用表 2 所列 170 局非重叠历史 NL 轨迹耗时 15.4 秒 CPU 完成跨领域适应，训练出 BeliefMLP_NL——在 40 局留出真实 NL 的 5,328 个未知席位上，准确率提升至 **65.69%（+15.6pp）**、Macro-F1 提升至 **67.40%（+17.2pp）**、交叉熵 NLL 降至 **0.7961**。"
    )

    add_body(
        doc,
        "在全新独立随机种子（20000+，与 BeliefMLP_NL 训练集严格零重叠）上，我们对 4 套系统进行了 **120 局均衡化 1v8 真实中文配对消融**（表 5 与图 5），并在表 6 中给出与已发表代表文献的跨研究量级参考对齐。"
    )

    add_three_line_table(
        doc,
        "表 5 均衡校准 1v8 真实中文自然语言四系统配对消融（120 局 vs 8 PureLLM，独立种子 20000+）",
        ["单席位系统", "5 角色均权（95% CI）", "牌堆 Deck9", "好人 4 角", "预/女/猎/民", "狼人", "单兵投准", "30 局费用"],
        [
            ["B0_PureLLM（纯大模型）", "26.7% [14.2, 44.4]", "37.0%", "8.3%", "0/0/33/0%", "100.0%", "80.0%", "¥1.322"],
            ["Ours_DualSystem_v2（精简版）", "40.0% [24.6, 57.7]", "44.4%", "33.3%", "33/33/33/33%", "66.7%", "65.1%", "¥1.291"],
            ["Ours_DualSystem_v2+（+长手册）", "26.7% [14.2, 44.4]", "33.3%", "16.7%", "33/0/17/17%", "66.7%", "71.7%", "¥1.346"],
            ["Ours_DualSystem_v3（+笔记替原文）", "13.3% [5.3, 29.7]", "18.5%", "4.2%", "17/0/0/0%", "50.0%", "40.5%", "¥1.229"],
        ],
        col_widths_cm=[3.9, 2.7, 1.4, 1.3, 2.1, 1.2, 1.4, 1.6],
        highlight_rows=[1],
        font_size=8.2,
        note="注：每系统 30 局（5 角色 × 6 局），括号内为 95% Wilson 置信区间。配对精确 McNemar 检验（Ours_DualSystem_v2 vs B0）：五角色 30 局分歧 7:3（p = 0.344），好人四角色 24 局分歧 7:1（p = 0.070，α = 0.10 水平边缘显著）。",
    )

    add_three_line_table(
        doc,
        "表 6 与已发表狼人杀智能体代表文献的跨研究量级参考对齐",
        ["代表工作", "LLM 底座与训练方式", "好人胜率", "狼人胜率", "5 角色等效", "自然牌堆 Deck9"],
        [
            ["MaKTO (2025) [1]", "Qwen2.5-72B（SFT + 自博弈 KTO 全参微调）", "~60%", "~63%", "~60.6%", "61.0%"],
            ["MaKTO 闭源基线 [1]", "GPT-4o（免训练商用大模型）", "—", "—", "—", "49.5%"],
            ["Xu et al. (ICML 2024) [5]", "冻结 LLM + 外部种群 RL 策略选择器", "30.0%", "70.0%", "38.0%", "43.3%"],
            ["本文 Ours_DualSystem_v2", "冻结 qwen3.5-27b（关思考）+ Belief_NL + 指针 RL", "33.3%", "66.7%", "40.0%", "44.4%"],
            ["本文基线 B0_PureLLM", "冻结 qwen3.5-27b（关思考，无外部训练）", "8.3%", "100.0%", "26.7%", "37.0%"],
        ],
        col_widths_cm=[3.3, 5.1, 1.5, 1.5, 1.8, 2.2],
        highlight_rows=[3],
        font_size=8.5,
        note="注：各文献在人数、规则（是否含警长）、语言及对手池强度上存在差异，本表仅作跨研究量级参考，不构成严格同环境胜负排序。",
    )

    add_body(
        doc,
        "如表 5 与图 5 所示，精简双系统架构 **Ours_DualSystem_v2 取得了全场最高的 40.0% 五角色胜率（较 B0 提升 +13.3pp，相对提升 +50%）与 44.4% 自然牌堆胜率（+7.4pp）**，在预言家、女巫、猎人、平民四个好人角色上均达到 33.3%（好人均值由 8.3% 提升 4 倍至 33.3%，24 局配对分歧 7:1，精确 McNemar 检验 p = 0.070），且 30 局费用（1.291 元）略低于 B0（1.322 元）。受单系统 30 局样本量所限，五角色总体配对检验尚未达到 p < 0.05 的强显著阈值（p = 0.344），但从表 6 量级参考来看，本文在零微调冻结 27B 底座上已达到与同类冻结 LLM+RL 代表工作 Xu et al.（ICML 2024，Deck9 等效 43.3%）相当的参考胜率量级。"
    )
    add_body(
        doc,
        "值得深入剖析表 5 中的两处反差指标：**（1）为何 Ours_DualSystem_v2 的单兵投准率（65.1%）低于 B0（80.0%），好人胜率却高出 4 倍？**拆解原始投票分子与分母可知，两者的有效命中总数完全相同（均为 28 票），且在好人四角色上 v2 的绝对投狼命中数反高于 B0（21/36 对比 18/24）。百分比倒挂源于**存活轮次与分母结构效应**：B0 好人常在前期早早出局（例如 B0 预言家在 6 局中累计仅存活参与了 2 次投票，1/2 = 50.0%），导致好人投票分母偏小且总体投准率被狼人局（10/11 = 90.9%）拉高；相反，v2 好人存活轮次更长、总投票次数增加 50%（36 票对比 24 票，其中预言家 5/8 = 62.5%），虽然中后期残局混淆度更高拉低了比例，但凭借更多的有效归票、预言家稳健带队及女巫高精准毒狼（2/3 = 66.7%）显著提高了好人胜率。**（2）为何 v2 狼人胜率由 6/6（100.0%）回落至 4/6（66.7%）？**失利回放显示，2 局失利均因第 3 天被放逐出局：尽管 v2 已将狼人投票阶段的 RL 权重放松至 0.15，但白天发言阶段注入的 System-2 战术提示在面对偏弱的背景纯大模型好人时仍会略微收窄狼人的自由发挥空间。这一发现与第 6 章提出“按角色不对称门控”的改进方向直接呼应。"
    )

    add_figure(doc, pngs["fig5"], "图 5 1v8 单席位随机分角主实验：符号 1,000 局（左）与均衡中文 NL 120 局四系统配对消融（右）", width_cm=15.0)

    add_h2(doc, "4.4 补充实验：整阵营对抗、跨模型挑战与成本核算")
    add_body(
        doc,
        "在完成均衡 1v8 主实验之前，本文在**早期未校准环境**（尚无首日防屠神常识校准、使用未跨领域适应的 MaKTO BeliefMLP 及早期版 Ours_DualSystem，未引入 fact_tracker）中开展了 120 局同底座（qwen3.5-27b）整阵营对抗消融（6 组 × 20 局）及 20 局跨模型挑战（qwen3.5-27b + Ours_DualSystem 对阵旗舰 qwen3.8-max PureLLM），共计 7 组 140 局（合计实测费用 6.03 元），完整结果见表 7 与图 6。"
    )

    add_three_line_table(
        doc,
        "表 7 补充实验：140 局早期整阵营对抗与跨模型挑战结果（7 组 × 20 局，0 报错，合计费用 6.03 元）",
        ["实验组（底座均为 qwen3.5-27b）", "好人胜率", "狼人胜率", "综合胜率", "好人投狼率", "Guardrail 通过", "20 局费用"],
        [
            ["B0_PureLLM（纯大模型）", "20.0%", "100.0%", "60.0%", "65.8%", "100.0%", "¥0.861"],
            ["B1_BeliefLLM（注入未适应 Belief）", "0.0%", "80.0%", "40.0%", "49.5%", "100.0%", "¥0.816"],
            ["B2_BeliefBC（纯解耦 + 离线 BC）", "10.0%", "40.0%", "25.0%", "53.4%", "100.0%", "¥0.546"],
            ["Ours_v1（纯解耦 + BC+RL）", "40.0%", "30.0%", "35.0%", "51.0%", "100.0%", "¥0.508"],
            ["Ours_v2_Sim2Real（纯解耦 Sim2Real）", "10.0%", "40.0%", "25.0%", "51.9%", "99.4%", "¥0.518"],
            ["Ours_DualSystem（早期双系统融合）", "20.0%", "80.0%", "50.0%", "60.3%", "100.0%", "¥0.933"],
            ["Ours_DualSystem vs qwen3.8-max（跨模型）", "10.0%", "50.0%", "30.0%", "52.2%", "99.4%", "¥1.851"],
        ],
        col_widths_cm=[5.2, 1.5, 1.5, 1.5, 1.8, 1.9, 1.7],
        highlight_rows=[3, 5],
        font_size=8.3,
        note="注：7 组费用之和为 0.861+0.816+0.546+0.508+0.518+0.933+1.851 = 6.033 元（¥6.03）；另含 2 局摸底试点（¥0.057），对应整阵营/试点合计 142 局、¥6.09。",
    )

    add_body(
        doc,
        "表 7 呈现出两项重要规律：第一，在未校准的早期环境中，由于背景 PureLLM 神职首日自爆率超过 70%，任何不受约束的纯大模型狼队均能轻易屠神（B0 狼人 100.0%、综合 60.0%），而早期 Ours_DualSystem 尚未做跨领域信念适应且狼人侧采用较高的 RL 融合权重（0.55），对狼队自由欺骗形成轻微制约（狼人 80.0%、综合 50.0%）。正是该诊断直接催生了主实验中的环境校准、BeliefMLP_NL、fact_tracker 与狼人低权放行设计，使改进版 Ours_DualSystem_v2 在均衡 1v8 主实验中大幅反超 B0。第二，对比纯解耦与双系统可见清晰的**自然语言信息瓶颈**：纯解耦 Ours_v1 虽在好人队取得 40.0% 最高胜率（猎人带狼率 71.4%，20 局仅 0.508 元节省 41%），却因 64 维离散意图丢失了 33.8% 的文本神职自称，导致狼人队胜率仅 30%；升级双系统融合后狼人胜率提升 +50pp 至 80.0%。此外，面对更强旗舰模型 qwen3.8-max，我方综合胜率由 50% 降至 30%，说明底座语言推理能力仍是上限的关键因素。全流程累计 **352 局真实中文九人局（2 局摸底 + 90 局诊断 1v8 + 120 局均衡 1v8 + 140 局整阵营/跨模型）零报错，API 总费用 14.84 元**，本地 CPU 训练 9,600 局自博弈与 5,800 局符号评测完全免费。"
    )

    add_figure(doc, pngs["fig4"], "图 6 补充实验：140 局真实中文整阵营对抗与跨模型挑战的胜率（上）及单局 API 成本（下）", width_cm=14.6)

    # ===== Section 5: Case Studies & Ablation Insights =====
    add_h1(doc, "5 典型案例与关键消融洞察")
    add_h2(doc, "5.1 典型真实中文对局案例剖析")
    add_body(
        doc,
        "**案例一：纯大模型的“幻觉级联与盲从效应”**。在真实对局 Ours_DualSystem_as_wolf_000 中，1/7/8 号为我方双系统狼人，其余 6 名好人为 qwen3.5-27b PureLLM。第 1 天 3 号真预言家明确发言：“我是 3 号预言家，首夜查验 2 号是好人。2 号跳女巫自救逻辑自洽，我认她身份。”紧随其后的 4 号纯 LLM 平民产生明显阅读幻觉，发言称：“3 号查杀 2 号，2 号却跳女巫自救，这逻辑完全矛盾……建议今天先出 3 号！”随后 5 号纯 LLM 猎人不仅盲从该幻觉，还提前亮出猎人底牌站边 4 号。我方三名双系统狼人全程潜伏（Guardrail 100% 通过），借好人内乱全票抗推真预言家 3 号，并连续两夜精准刀杀已暴露的 2 号女巫与 5 号猎人，以三狼全员存活获胜。该案例直观印证了引入 System-2 客观事实核验器的必要性。"
    )
    add_body(
        doc,
        "**案例二：1v8 单席位 Ours_DualSystem_v2 抽中神职稳健带队获胜**。在均衡 1v8 主实验 Ours_DualSystem_v2_seer_00 与 witch_02 中，处于 1v8 孤立席位的我方智能体一方面遵循首日防屠神纪律避免过早自爆，另一方面依靠 BeliefMLP_NL 与客观事实核验器锁定假跳预言家与冲票狼，在女巫局精准保存并命中狼人毒药（精简版 v2 女巫毒狼精准度达 2/3 = 66.7%），在四类好人角色上均实现了从 0% 到 33.3% 的稳定提升。"
    )

    add_h2(doc, "5.2 借鉴 MaKTO 的负消融分析与冻结小模型边界")
    add_body(
        doc,
        "为探索零微调下能否进一步逼近 MaKTO 的 61% 胜率，本文设计了两组借鉴 MaKTO 提示与记忆结构的进阶消融，均产生了清晰的反面实证（表 5）："
    )
    add_body(
        doc,
        "第一，**Ours_DualSystem_v3（以自动抽取笔记替代发言原文 + 四字段 JSON 规划）导致五角色胜率回落至 13.3%、投准率下降 24.6pp（65.1% → 40.5%）**。回放审计定位出双重根因：其一，早期正则解析器在处理第三人称转述句（如“3 号发言：我是好人，2 号跳预言家查杀 4 号”）时发生跨逗号串句误匹配，将 3 号错标为预言家并污染了笔记，诱发女巫误毒真预言家（现已通过严格第一人称锚定正则修复并补全回归测试）；其二，舍弃自然语言原文并用复杂 JSON 强制输出显著损害了关思考 27B 模型的语义理解，与第一阶段“噪声中间抽取反而降分”的发现高度呼应。"
    )
    add_body(
        doc,
        "第二，**Ours_DualSystem_v2+（保留原文、修复正则并叠加长角色手册 ROLE_PLAYBOOK_ZH 与 14 条长上下文）虽将单兵投准率推高至 71.7%（+6.6pp），五角色胜率却回落至 26.7%**。技能日志审计表明，手册中“第 2 夜起果断泼毒避免带毒出局”的强指令诱发女巫在 **6/6 局全部开毒且 3/6 局误毒好人**，使女巫胜率从 33.3% 降至 0.0%；同时冗长提示稀释了关思考 27B 模型的注意力。上述两组对照表明：在冻结中小参数模型下，**简洁结构化约束（Ours_DualSystem_v2）优于冗长规则堆叠**，而要进一步跨越 44.4% 逼近 61%，本质上需要更强推理底座或端到端参数后训练。"
    )

    add_h2(doc, "5.3 交互式心智可视化演示系统")
    add_body(
        doc,
        "为直观呈现智能体在真实中文对局中的内部推理过程，本文构建了零依赖离线交互式 Web 演示看板（werewolf_agent_demo.html）。看板内置 7 局精选真实中文对局回放（首页优先展示 4 局均衡 1v8 预/女/民/狼胜局并高亮被测单席位），支持按事件步进回放、一键切换任意座位的上帝/第一人称视角，实时渲染 BeliefMLP_NL 的 9×5 身份后验概率热力图、状态估值 V(s) 与 System-2 指针 RL 投票候选分布，并同屏集成全部实验矢量图表。"
    )

    # ===== Section 6: Conclusion =====
    add_h1(doc, "6 结论与展望")
    add_body(
        doc,
        "本文针对非合作多轮对话博弈中的高微调成本与纯大模型策略缺陷问题，设计并实现了基于冻结大模型与本地轻量强化学习的狼人杀双系统角色智能体。通过双模统一状态机、席位共享信念追踪与跨领域适应、席位等变指针网络自博弈 PPO 以及 System-2 客观事实核验，本系统在零模型微调费用、总计 14.84 元 API 成本下完成了 352 局真实中文九人局受控评测。主实验表明，精简双系统架构 Ours_DualSystem_v2 在 1v8 单席位随机分角评测中达到 40.0% 五角色胜率与 44.4% 自然牌堆胜率，较纯大模型基线提升 +13.3pp、好人四角色胜率由 8.3% 提升至 33.3%（配对分歧 7:1，p = 0.070），在各自评测口径下达到与同类冻结大模型强化学习代表工作相当的参考胜率量级。同时，本文对语言漂移、信息瓶颈、跨领域信念偏移与过度提示负消融的系统审计，为低成本博弈智能体设计提供了清晰的实证边界。"
    )
    add_body(
        doc,
        "受课程计算预算与时间所限，当前工作仍存在三方面局限：规则暂未纳入警长竞选与警徽流；真实 NL 单组配对样本量（每角色 6 局、每系统 30 局）仍有一定统计方差；冻结 27B 非思考底座在复杂倒钩与多轮隐含语义推理上存在能力上限。未来工作可沿两方向拓展：一是探索基于角色不对称性的自适应双系统门控（好人侧强化 System-2 RL 与事实验证，狼人侧进一步降低提示与策略约束以保留语言灵活性）；二是在少量高质量自博弈轨迹上结合低秩微调（LoRA）或轻量思考蒸馏，进一步缩小与 72B 端到端微调模型之间的胜率差距。"
    )

    # ===== References (GB/T 7714 Verified) =====
    add_h1(doc, "参考文献")
    ordered_refs = [
        "YE R, ZHANG Y, ZHANG Y, et al. Multi-agent KTO: Reinforcing strategic interactions of large language model in language game[J]. arXiv preprint arXiv:2501.14225, 2025.",
        "XU Z, GU W, YU C, et al. Learning strategic language agents in the Werewolf game with iterative latent space policy optimization[C]//Proceedings of the 42nd International Conference on Machine Learning (ICML). PMLR, 2025.",
        "JIANG B, SHI T, KAMOI R, et al. One model, all roles: Multi-turn, multi-agent self-play reinforcement learning for conversational social intelligence[J]. arXiv preprint arXiv:2602.03109, 2026.",
        "LIU B, GUERTLER L, YU S, et al. SPIRAL: Self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning[C]//International Conference on Learning Representations (ICLR). 2026.",
        "XU Z, YU C, FANG F, et al. Language agents with reinforcement learning for strategic play in the Werewolf game[C]//Proceedings of the 41st International Conference on Machine Learning (ICML). PMLR, 2024: 55434–55454.",
        "WU S, ZHU L, YANG T, et al. Enhance reasoning for large language models in the game Werewolf[J]. arXiv preprint arXiv:2402.02330, 2024.",
        "JIN X, WANG Z, DU Y, et al. Learning to discuss strategically: A case study on One Night Ultimate Werewolf[C]//Advances in Neural Information Processing Systems (NeurIPS). 2024, 37.",
        "WANG S, LIU C, ZHENG Z, et al. Avalon's game of thoughts: Battle against deception through recursive contemplation (ReCon)[J]. arXiv preprint arXiv:2310.01320, 2023.",
        "BAILIS S, FRIEDHOFF J, CHEN F. Werewolf Arena: A case study in LLM evaluation via social deduction[J]. arXiv preprint arXiv:2407.13943, 2024.",
    ]
    for idx, ref in enumerate(ordered_refs, start=1):
        add_reference(doc, idx, ref)

    # ===== Appendix (P2 Bonus) =====
    add_h1(doc, "附录：精简双系统提示模板与事实核验注入示例")
    add_body(
        doc,
        "为便于复现并直观呈现“简洁结构化约束优于冗长规则堆叠”的工程实现，下文节选精简双系统（Ours_DualSystem_v2）实际运行的 System-1 候选决策系统提示、System-2 客观事实核验器（fact_tracker.py）注入片段以及 Guardrail 护栏拦截示例。"
    )
    add_code_box(
        doc,
        "附录 A：System-1 候选动作提议系统提示（CANDIDATE_SYSTEM_PROMPT 节选）",
        """你是中文九人预女猎狼人杀的策略助手（3狼3民1预1女1猎，屠边，无警长）。
请结合你的真实身份、私有信息与完整公开对话，从 legal_action_ids 中选出最合理的 1~2 个候选动作（按优先级排序）。
策略提醒：
- 若你是狼人：夜晚优先精准刀死已暴露或疑似神职；白天若队友已被查杀且必出，可适度倒钩隐藏自己；绝不暴露狼队。
- 若你是好人：优先归票查杀狼或行为矛盾最明显的狼人；女巫首夜尽量救人，次夜毒确定度高的狼人；猎人尽量带走最像狼的玩家。
只输出合法 JSON：{"candidates": [最优action_id, 次优action_id], "reason": "一句话理由"}。""",
    )
    add_code_box(
        doc,
        "附录 B：System-2 事实核验板与校准信念注入片段（真预言家视角实测示例）",
        """当前：第2天 | 你的座位：3号 | 你的真实底牌：预言家
【私有】你的查验记录：第1夜验2号=好人；第2夜验7号=狼人
【系统二核验·预言家查验表】D1 3号跳预言家报2号=金水(好人)；D1 7号跳预言家报3号=查杀(狼)
【系统二铁证·存活必狼】[7]号（逻辑铁狼，好人应优先归票/用毒/开枪！）
【系统二铁证·存活好人】[2]号（切勿错投/错毒）
【系统二矛盾提示】我是真预言家，7号对跳预言家必为悍跳狼
【系统二校准信念·最高狼面】7号(狼概率100%)、1号(狼概率62%)、8号(狼概率48%)
→ System-2 决策：触发铁狼强护栏，直接锁定放逐投票 action_id = 7（阻断 LLM 幻觉偏移）。""",
    )
    add_code_box(
        doc,
        "附录 C：防泄密 / 防幻觉 Guardrail 拦截与重试示例（check_speech_guardrail）",
        """[违规初稿（狼人 8号）] “其实我的狼队友 1号 刚才发言很稳，我觉得今天应该先出 3号。”
[Guardrail 检测] passed=False, violations=["wolf_self_or_teammate_leak", "wolf_teammate_1_leak"]
[触发重写提示] “发言违规（['wolf_self_or_teammate_leak']），请重写合法发言：”
[合规终稿] “我是平民，1号逻辑清晰偏正面，3号起跳查验前后漏洞较多，建议今天先出 3号。” → passed=True""",
    )

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    if OUT_DOCX.exists():
        OUT_DOCX.unlink()
        print(f"Deleted old Word report: {OUT_DOCX}")
    doc.save(str(OUT_DOCX))
    print(f"Saved revised Word report: {OUT_DOCX} ({OUT_DOCX.stat().st_size} bytes)")


if __name__ == "__main__":
    build_report()

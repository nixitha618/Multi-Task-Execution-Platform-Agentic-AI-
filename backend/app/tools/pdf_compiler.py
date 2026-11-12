import os
import re
import html
import logging
from typing import Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

logger = logging.getLogger(__name__)

def _format_inline(text: str) -> str:
    """Escapes XML entities then applies inline markdown styling (bold, italic, code)."""
    # 1. Escape XML characters so ReportLab XML parser does not crash
    safe = html.escape(text.strip(), quote=False)

    # 2. Extract inline code first to prevent underscores inside identifiers from turning into italics
    code_blocks = []
    def _save_code(m):
        code_blocks.append(f'<font face="Courier" color="#1e293b"><b>{m.group(1)}</b></font>')
        return f"__CODE_PH_{len(code_blocks)-1}__"

    safe = re.sub(r'`(.+?)`', _save_code, safe)

    # 3. Markdown Bold: **text** or __text__
    safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe)
    safe = re.sub(r'(?<!\w)__(.+?)__(?!\w)', r'<b>\1</b>', safe)

    # 4. Markdown Italic: *text* or (boundary-isolated) _text_
    safe = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', safe)
    safe = re.sub(r'(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)', r'<i>\1</i>', safe)

    # 5. Restore inline code
    for idx, cb in enumerate(code_blocks):
        safe = safe.replace(f"__CODE_PH_{idx}__", cb)

    return safe

def compile_markdown_to_pdf(
    markdown_text: str,
    output_pdf_path: str,
    doc_title: Optional[str] = None
) -> str:
    """
    Compiles a Markdown text string into a beautifully-styled, publication-ready PDF document.
    Handles headings, bullet points, numbered lists, tables, and inline styling.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = getSampleStyleSheet()

    # Custom typography & hierarchy
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=0,
        spaceAfter=8
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2563eb'),
        spaceBefore=9,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=14,
        spaceAfter=3
    )

    story = []

    # Derive or render main document header
    clean_lines = markdown_text.strip().splitlines()
    first_title = doc_title or "Deliverables & Summary Report"

    # If first line of markdown is an H1, adopt it as doc title
    if clean_lines and clean_lines[0].startswith('# '):
        first_title = clean_lines[0][2:].strip()
        clean_lines = clean_lines[1:]

    story.append(Paragraph(_format_inline(first_title), title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceBefore=2, spaceAfter=12))

    try:
        i = 0
        while i < len(clean_lines):
            line = clean_lines[i].strip()
            if not line:
                story.append(Spacer(1, 4))
                i += 1
                continue

            # Markdown Headings
            if line.startswith('# '):
                story.append(Paragraph(_format_inline(line[2:].strip()), h1_style))
                story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#cbd5e1'), spaceBefore=1, spaceAfter=6))
            elif line.startswith('## '):
                story.append(Paragraph(_format_inline(line[3:].strip()), h1_style))
            elif line.startswith('### ') or line.startswith('#### '):
                story.append(Paragraph(_format_inline(line.lstrip('#').strip()), h2_style))
            elif line.startswith('- ') or line.startswith('* '):
                b_text = _format_inline(line[2:].strip())
                story.append(Paragraph(f"&bull; {b_text}", bullet_style))
            elif re.match(r'^\d+\.\s+', line):
                # Numbered item
                num_text = re.sub(r'^\d+\.\s+', '', line)
                num_prefix = re.match(r'^(\d+\.)\s+', line).group(1)
                story.append(Paragraph(f"<b>{num_prefix}</b> {_format_inline(num_text)}", bullet_style))
            elif line.startswith('|') and line.endswith('|'):
                # Parse Markdown Table
                table_lines = []
                while i < len(clean_lines) and clean_lines[i].strip().startswith('|') and clean_lines[i].strip().endswith('|'):
                    table_lines.append(clean_lines[i].strip())
                    i += 1
                i -= 1 # adjust for outer loop increment

                data_rows = [r for r in table_lines if not re.match(r'^\|(?:\s*[:-]+[-| :]*)\|?$', r)]
                table_data = []
                for r in data_rows:
                    cells = [c.strip() for c in r.strip('|').split('|')]
                    row_cells = [Paragraph(_format_inline(c), body_style) for c in cells]
                    table_data.append(row_cells)

                if table_data:
                    num_cols = max(len(row) for row in table_data)
                    avail_width = letter[0] - 108
                    col_width = avail_width / max(num_cols, 1)
                    t = Table(table_data, colWidths=[col_width] * num_cols)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('LEFTPADDING', (0, 0), (-1, -1), 5),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(_format_inline(line), body_style))

            i += 1

        doc.build(story)
        return output_pdf_path

    except Exception as build_err:
        logger.warning(f"Error compiling styled PDF ({build_err}), falling back to robust plain document builder.")
        # Robust plain fallback
        fallback_story = [
            Paragraph(html.escape(first_title), title_style),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2563eb'), spaceBefore=2, spaceAfter=8)
        ]
        for l in markdown_text.splitlines():
            fallback_story.append(Paragraph(html.escape(l.strip() or " "), body_style))
        doc.build(fallback_story)
        return output_pdf_path

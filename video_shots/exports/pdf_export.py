"""PDF export utilities."""

from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from .utils import format_timestamp

def combine_to_pdf(video_name, screenshots, timepoints, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    for shot, tp in zip(screenshots, timepoints):
        c.drawString(2*cm, height - 2*cm, f"{video_name} - {format_timestamp(tp.seconds, precision='ms')}")
        c.drawInlineImage(str(shot), 2*cm, 4*cm, width= width - 4*cm, height=height/2)
        c.showPage()
    c.save()
    return output_path

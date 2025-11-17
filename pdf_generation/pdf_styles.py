"""
PDF Styles Module
Defines all colors, fonts, and paragraph styles for PDF reports
"""

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# Color Definitions
RED6_RED = colors.HexColor('#E30613')
PRIMARY_COLOR = colors.HexColor('#1a1a1a')      # Dark gray (almost black)
SECONDARY_COLOR = colors.HexColor('#6b7280')    # Medium gray
BACKGROUND_LIGHT = colors.HexColor('#f5f5f5')   # Light gray background
DARKENED_RED = colors.HexColor('#a8001a')       # Darkened red (for gradient)
SOFT_DARK_BROWN = colors.HexColor('#4a2c2c')    # Soft dark brown (for gradient)


def get_all_styles():
    """
    Create and return all paragraph styles used in the PDF report
    
    Returns:
        dict: Dictionary containing all style objects
    """
    base_styles = getSampleStyleSheet()
    
    # Title style - Main report title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=base_styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=26,
        spaceAfter=25,
        alignment=TA_LEFT,
        textColor=PRIMARY_COLOR,
        leading=32
    )
    
    # Heading style - Major section headings
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=base_styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=12,
        spaceBefore=20,
        textColor=RED6_RED,
        leading=22,
        alignment=TA_LEFT
    )
    
    # Subheading style - Subsection headings
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=base_styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceAfter=8,
        spaceBefore=15,
        textColor=PRIMARY_COLOR,
        leading=18,
        alignment=TA_LEFT
    )
    
    # Normal text style - Body text
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        spaceAfter=6,
        textColor=PRIMARY_COLOR,
        leading=16,
        alignment=TA_JUSTIFY
    )
    
    # Subtitle style - Secondary titles
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        textColor=SECONDARY_COLOR,
        leading=18,
        alignment=TA_LEFT
    )
    
    # Table of Contents heading style
    toc_style = ParagraphStyle(
        'TOCHeading',
        parent=base_styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=15,
        spaceBefore=10,
        textColor=RED6_RED,
        leading=22,
        alignment=TA_LEFT
    )
    
    # TOC main entry style
    toc_main_style = ParagraphStyle(
        'TOCMain',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        spaceAfter=6,
        leftIndent=10,
        textColor=PRIMARY_COLOR,
        leading=16
    )
    
    # TOC sub entry style
    toc_sub_style = ParagraphStyle(
        'TOCSub',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        spaceAfter=4,
        leftIndent=30,
        textColor=SECONDARY_COLOR,
        leading=14
    )
    
    # Figure caption style
    figure_caption_style = ParagraphStyle(
        'FigureCaption',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        alignment=TA_CENTER,
        textColor=SECONDARY_COLOR,
        spaceAfter=6
    )
    
    return {
        'base': base_styles,
        'title': title_style,
        'heading': heading_style,
        'subheading': subheading_style,
        'normal': normal_style,
        'subtitle': subtitle_style,
        'toc': toc_style,
        'toc_main': toc_main_style,
        'toc_sub': toc_sub_style,
        'figure_caption': figure_caption_style
    }


# Risk level color coding for insurance section
RISK_COLORS = {
    'Low': colors.green,
    'Medium': colors.orange,
    'High': colors.orangered,
    'Critical': colors.red
}

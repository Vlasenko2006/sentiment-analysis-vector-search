"""
PDF Tables Module
Handles creation and styling of tables for PDF reports
"""

from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from .pdf_styles import PRIMARY_COLOR, SECONDARY_COLOR, RISK_COLORS


def create_insurance_breakdown_table(breakdown):
    """
    Create breakdown table for insurance risk factors
    
    Args:
        breakdown: Dictionary containing risk breakdown data
        
    Returns:
        Table: Styled ReportLab Table object
    """
    sentiment_factors = breakdown.get('sentiment_factors', {})
    confidence_factors = breakdown.get('confidence_factors', {})
    sample_factors = breakdown.get('sample_factors', {})
    trend_factors = breakdown.get('trend_factors', {})
    base_rate = breakdown.get('base_rate', 5000)
    
    # Create breakdown table data
    breakdown_data = [
        ['Risk Factor', 'Value', 'Multiplier', 'Impact'],
        ['Base Insurance Rate', f"${base_rate:,.2f}", '1.00', 'Baseline'],
        ['Positive Reviews', f"{sentiment_factors.get('positive_percentage', 0)}%",
         f"{sentiment_factors.get('sentiment_multiplier', 1.0)}", 
         'Higher % reduces risk' if sentiment_factors.get('positive_percentage', 0) > 75 else 'Standard'],
        ['Negative Reviews', f"{sentiment_factors.get('negative_percentage', 0)}%",
         f"{sentiment_factors.get('sentiment_multiplier', 1.0)}", 
         'Higher % increases risk' if sentiment_factors.get('negative_percentage', 0) > 20 else 'Standard'],
        ['Average Confidence', f"{confidence_factors.get('average_confidence', 0):.3f}",
         f"{confidence_factors.get('confidence_multiplier', 1.0)}", 
         'Low confidence increases uncertainty'],
        ['Sample Size', f"{sample_factors.get('total_samples', 0)} reviews",
         f"{sample_factors.get('sample_multiplier', 1.0)}", 
         'Small sample increases risk'],
        ['Sentiment Trend', trend_factors.get('trend_status', 'N/A'), 
         f"{trend_factors.get('trend_multiplier', 1.0)}", 
         'Recent patterns affect risk']
    ]
    
    breakdown_table = Table(breakdown_data, colWidths=[140, 100, 80, 150])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (2, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    return breakdown_table


def create_insurance_result_table(insurance_cost, risk_score, risk_level):
    """
    Create result table for insurance cost estimate
    
    Args:
        insurance_cost: Calculated insurance cost
        risk_score: Risk score (0-100)
        risk_level: Risk level category (Low/Medium/High/Critical)
        
    Returns:
        Table: Styled ReportLab Table object
    """
    risk_color = RISK_COLORS.get(risk_level, colors.grey)
    
    # Create result box
    result_data = [
        ['Assessment Component', 'Result'],
        ['Calculated Insurance Cost', f"${insurance_cost:,.2f}"],
        ['Risk Score', f"{risk_score}/100"],
        ['Risk Level', risk_level]
    ]
    
    result_table = Table(result_data, colWidths=[250, 220])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (1, 2), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, 2), colors.white),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f5f5f5')),  # Light gray background
        ('TEXTCOLOR', (1, 3), (1, 3), risk_color),
        ('GRID', (0, 0), (-1, -1), 1.5, colors.grey)
    ]))
    
    return result_table


def create_title_with_image_table(title_para, image_obj, title_width=5.2, image_width=1.1):
    """
    Create a table with title and image side-by-side
    
    Args:
        title_para: Paragraph object for title
        image_obj: Image object to display
        title_width: Width of title column in inches
        image_width: Width of image column in inches
        
    Returns:
        Table: Styled ReportLab Table object
    """
    data = [[title_para, image_obj]]
    t = Table(data, colWidths=[title_width*inch, image_width*inch])
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    return t


def create_toc_table(toc_entries, toc_main_style, toc_sub_style):
    """
    Create table of contents entries
    
    Args:
        toc_entries: List of (entry_text, style) tuples
        toc_main_style: ParagraphStyle for main entries
        toc_sub_style: ParagraphStyle for sub entries
        
    Returns:
        list: List of Paragraph objects for TOC
    """
    from reportlab.platypus import Paragraph
    
    paragraphs = []
    for entry_text, entry_style in toc_entries:
        paragraphs.append(Paragraph(entry_text, entry_style))
    
    return paragraphs

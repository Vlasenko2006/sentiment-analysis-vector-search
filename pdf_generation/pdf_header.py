"""
PDF Header Module
Handles header stripe drawing and company name loading
"""

import os
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from .pdf_styles import DARKENED_RED, SOFT_DARK_BROWN


def load_company_name():
    """
    Load company name from config_names.yaml
    
    Returns:
        str: Company name from config, or 'Awesome Company' as default
    """
    # Try config file locations (skip -example files)
    config_paths = [
        'config/config_names.yaml',
        'config_names.yaml'
    ]
    
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    company_name = config.get('company_name', 'Awesome Company')
                    print(f"✅ PDF: Loaded company name from {config_path}: {company_name}")
                    return company_name
        except Exception as e:
            print(f"⚠️ PDF: Could not load company name from {config_path}: {e}")
            continue
    
    print("⚠️ PDF: No config_names.yaml found, using default: Awesome Company")
    return 'Awesome Company'


def draw_header_stripe(canvas_obj, doc, company_name=None):
    """
    Draw header stripe with gradient from darkened red to soft dark brown
    
    Args:
        canvas_obj: ReportLab canvas object
        doc: Document object
        company_name: Company name to display (loads from config if None)
    """
    if company_name is None:
        company_name = load_company_name()
    
    canvas_obj.saveState()
    
    # Get page dimensions
    page_width, page_height = A4
    
    # Stripe height is 8% of page height (slimmer, more modern)
    stripe_height = page_height * 0.08
    
    # Draw gradient - create horizontal gradient from left (darkened red) to right (soft dark brown)
    num_steps = 60
    step_width = page_width / num_steps
    
    for i in range(num_steps):
        # Calculate gradient position (0.0 at left to 1.0 at right)
        gradient_pos = i / num_steps
        
        # Interpolate between darkened red and soft dark brown
        r = DARKENED_RED.red + (SOFT_DARK_BROWN.red - DARKENED_RED.red) * gradient_pos
        g = DARKENED_RED.green + (SOFT_DARK_BROWN.green - DARKENED_RED.green) * gradient_pos
        b = DARKENED_RED.blue + (SOFT_DARK_BROWN.blue - DARKENED_RED.blue) * gradient_pos
        
        # Set fill color
        canvas_obj.setFillColorRGB(r, g, b)
        
        # Draw rectangle strip
        x_position = i * step_width
        canvas_obj.rect(x_position, page_height - stripe_height, step_width + 1, stripe_height, fill=1, stroke=0)
    
    # Draw company name in white at the top of the stripe
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont('Helvetica-Bold', 28)  # Slightly smaller, more modern
    
    # Left-aligned text with left margin
    x_position = 40  # Left margin
    y_position = page_height - (stripe_height * 0.6)  # Centered vertically in stripe
    
    canvas_obj.drawString(x_position, y_position, company_name)
    
    # Add "GenAI Sentiment Analysis" subtitle in smaller font
    canvas_obj.setFont('Helvetica', 11)
    canvas_obj.setFillColor(colors.white)
    subtitle_y = y_position - 18
    canvas_obj.drawString(x_position, subtitle_y, "GenAI Sentiment Analysis Bericht")
    
    canvas_obj.restoreState()

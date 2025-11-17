"""
PDF Images Module
Handles image loading and placement in PDF reports
"""

import os
from reportlab.platypus import Image
from reportlab.lib.units import inch


def get_pipeline_diagram(script_dir):
    """
    Get pipeline diagram image
    
    Args:
        script_dir: Directory containing the script
        
    Returns:
        tuple: (image_obj or None, success_message or error_message)
    """
    pipeline_pdf_path = os.path.join(script_dir, 'Images', 'Automated_pipeline.png')
    
    if os.path.exists(pipeline_pdf_path):
        try:
            pipeline_img = Image(pipeline_pdf_path, width=6.5*inch, height=4*inch)
            return pipeline_img, f"✅ Successfully inserted pipeline diagram from {pipeline_pdf_path}"
        except Exception as e:
            return None, f"⚠️ Warning: Could not insert pipeline PDF: {e}"
    else:
        return None, f"⚠️ Warning: Pipeline PDF not found at {pipeline_pdf_path}"


def get_sentiment_icon(script_dir):
    """
    Get sentiment analysis icon image
    
    Args:
        script_dir: Directory containing the script
        
    Returns:
        Image object or None
    """
    sentiment_img_path = os.path.join(script_dir, 'Images', 'Sentiment_analysis.png')
    
    if os.path.exists(sentiment_img_path):
        try:
            return Image(sentiment_img_path, width=0.9*inch, height=0.675*inch)
        except Exception as e:
            print(f"⚠️ Could not load sentiment icon: {e}")
            return None
    return None


def get_confidence_icon(script_dir):
    """
    Get highest confidence comments icon image
    
    Args:
        script_dir: Directory containing the script
        
    Returns:
        Image object or None
    """
    confidence_img_path = os.path.join(script_dir, 'Images', 'Highest_Confidence_Comments.png')
    
    if os.path.exists(confidence_img_path):
        try:
            return Image(confidence_img_path, width=1.4*inch, height=1.05*inch)
        except Exception as e:
            print(f"⚠️ Could not load confidence icon: {e}")
            return None
    return None


def get_summaries_icon(script_dir):
    """
    Get AI-generated summaries icon image
    
    Args:
        script_dir: Directory containing the script
        
    Returns:
        Image object or None
    """
    summaries_img_path = os.path.join(script_dir, 'Images', 'AI_Generated_Sentiment_Summaries.png')
    
    if os.path.exists(summaries_img_path):
        try:
            return Image(summaries_img_path, width=0.9*inch, height=0.675*inch)
        except Exception as e:
            print(f"⚠️ Could not load summaries icon: {e}")
            return None
    return None


def add_visualization(viz_folder, viz_filename, width=7*inch, height=4*inch):
    """
    Load and return a visualization image
    
    Args:
        viz_folder: Directory containing visualizations
        viz_filename: Filename of the visualization
        width: Image width (default 7 inches)
        height: Image height (default 4 inches)
        
    Returns:
        tuple: (Image object or None, exists flag)
    """
    viz_path = os.path.join(viz_folder, viz_filename)
    
    if os.path.exists(viz_path):
        try:
            img = Image(viz_path, width=width, height=height)
            return img, True
        except Exception as e:
            print(f"⚠️ Could not load visualization {viz_filename}: {e}")
            return None, True  # File exists but couldn't load
    return None, False  # File doesn't exist

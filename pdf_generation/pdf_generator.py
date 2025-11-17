"""
PDF Generator Module
Main orchestrator for PDF report generation
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.units import inch

from .pdf_styles import get_all_styles
from .pdf_header import draw_header_stripe, load_company_name
from .pdf_data_loader import extract_source_info_from_db, load_existing_data
from .pdf_sections import (
    create_title_page,
    create_table_of_contents,
    create_executive_summary,
    create_methodology_section,
    create_visualizations_section,
    create_comment_methodology_section,
    create_vector_mean_comments_section,
    create_highest_confidence_section,
    create_llm_summaries_section,
    create_recommendations_section,
    create_insurance_risk_section,
    create_technical_details_section
)

# Check if PDF generation is available
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️ ReportLab not available. Install with: pip install reportlab")
    PDF_AVAILABLE = False


def generate_pdf_report_simple(results_df, representative_results, performance_summary, 
                               db_path, OUTPUT_BASE_DIR, target_url=None, company_name=None):
    """
    Generate comprehensive PDF report of the analysis
    
    Args:
        results_df: DataFrame containing sentiment analysis results
        representative_results: Dict of representative comments by sentiment
        performance_summary: Dict containing performance metrics
        db_path: Path to SQLite database
        OUTPUT_BASE_DIR: Base directory for outputs
        target_url: Target URL for analysis (optional)
        company_name: Company name for branding (optional)
        
    Returns:
        str: Path to generated PDF, or None if generation failed
    """
    
    if not PDF_AVAILABLE:
        print("❌ Cannot generate PDF report. Install reportlab: pip install reportlab")
        return None
    
    # Load company name from config if not provided
    if company_name is None:
        company_name = load_company_name()
    
    # Create PDF file path
    pdf_path = os.path.join(OUTPUT_BASE_DIR, 'visualizations', 'sentiment_analysis_report.pdf')
    
    # Create PDF document with extra top margin for the red stripe (8% of page height + 0.5 inch)
    stripe_height = A4[1] * 0.08  # 8% of page height
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=A4, 
        topMargin=stripe_height + 0.5*inch, 
        bottomMargin=0.5*inch
    )
    story = []
    
    # Get all paragraph styles
    styles = get_all_styles()
    
    # Get script directory for image paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir)  # Go up one level from pdf_generation/
    
    # Determine source information
    if target_url:
        source_info = target_url
    else:
        source_info = extract_source_info_from_db(db_path)
    
    # Build report sections
    create_title_page(story, styles)
    create_table_of_contents(story, styles)
    create_executive_summary(story, styles, results_df, source_info)
    create_methodology_section(story, styles, script_dir)
    
    # Visualizations
    viz_folder = os.path.join(OUTPUT_BASE_DIR, 'visualizations')
    create_visualizations_section(story, styles, viz_folder, script_dir)
    
    # Comment analysis
    create_comment_methodology_section(story, styles)
    create_vector_mean_comments_section(story, styles, representative_results)
    create_highest_confidence_section(story, styles, results_df, script_dir)
    
    # AI-generated content
    create_llm_summaries_section(story, styles, OUTPUT_BASE_DIR, script_dir)
    create_recommendations_section(story, styles, OUTPUT_BASE_DIR)
    
    # Insurance risk assessment
    create_insurance_risk_section(story, styles, OUTPUT_BASE_DIR)
    
    # Technical details
    create_technical_details_section(story, styles, db_path, performance_summary)
    
    # Build PDF with custom header on every page
    try:
        doc.build(
            story, 
            onFirstPage=lambda c, d: draw_header_stripe(c, d, company_name), 
            onLaterPages=lambda c, d: draw_header_stripe(c, d, company_name)
        )
        print(f"📄 PDF report generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        return None


def generate_pdf_fun(DB_PATH, OUTPUT_BASE_DIR, TARGET_URL=None, company_name=None):
    """
    Main entry point for PDF generation from existing analysis data
    
    Args:
        DB_PATH: Path to SQLite database
        OUTPUT_BASE_DIR: Base directory containing analysis outputs
        TARGET_URL: Target URL for analysis (optional)
        company_name: Company name for branding (optional)
    """
    print("📄 Generating PDF Report from Existing Data...")
    
    # Load company name from config if not provided
    if company_name is None:
        company_name = load_company_name()
    
    print(f"🏢 Company Name: {company_name}")
    
    # Load existing analysis data
    results_df, representative_results, performance_summary = load_existing_data(OUTPUT_BASE_DIR)
    
    if results_df is None:
        print("❌ Cannot generate PDF without analysis data")
        return
    
    # Generate PDF report
    pdf_path = generate_pdf_report_simple(
        results_df, 
        representative_results, 
        performance_summary, 
        DB_PATH, 
        OUTPUT_BASE_DIR, 
        TARGET_URL, 
        company_name
    )
    
    if pdf_path:
        print(f"✅ PDF report successfully generated: {pdf_path}")
        print("📋 Report includes:")
        print("   • Executive summary with source information")
        print("   • Methodology and technical details")
        print("   • Sentiment analysis visualizations")
        print("   • Most representative comments by sentiment")
        print("   • Performance metrics and processing statistics")
    else:
        print("❌ PDF report generation failed")

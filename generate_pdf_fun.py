#!/usr/bin/env python3
"""
Quick PDF Report Generator
Generates PDF report from existing analysis data
"""

import os
import pandas as pd
import numpy as np
import json
import sqlite3
import yaml
from datetime import datetime

# PDF report generation
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


def load_company_name():
    """Load company name from config_names.yaml"""
    config_path = 'config_names.yaml'
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('company_name', 'Awesome Company')
    except Exception as e:
        print(f"⚠️ Could not load company name from config: {e}")
    return 'Awesome Company'


def draw_header_stripe(canvas_obj, doc, company_name=None):
    """Draw  header stripe with gradient from darkened red to soft dark brown"""
    if company_name is None:
        company_name = load_company_name()
    
    canvas_obj.saveState()
    
    # Get page dimensions
    page_width, page_height = A4
    
    # Stripe height is 8% of page height (slimmer, more modern)
    stripe_height = page_height * 0.08
    
    # Define gradient colors - darkened red to soft dark brown
    darkened_red = colors.HexColor('#a8001a')   # Darkened red (less bright)
    soft_dark_brown = colors.HexColor('#4a2c2c') # Soft dark brown
    
    # Draw gradient - create horizontal gradient from left (darkened red) to right (soft dark brown)
    num_steps = 60
    step_width = page_width / num_steps
    
    for i in range(num_steps):
        # Calculate gradient position (0.0 at left to 1.0 at right)
        gradient_pos = i / num_steps
        
        # Interpolate between darkened red and soft dark brown
        r = darkened_red.red + (soft_dark_brown.red - darkened_red.red) * gradient_pos
        g = darkened_red.green + (soft_dark_brown.green - darkened_red.green) * gradient_pos
        b = darkened_red.blue + (soft_dark_brown.blue - darkened_red.blue) * gradient_pos
        
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


def extract_source_info_from_db(db_path):
    """Extract source website information from database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get unique file paths to determine source
        cursor.execute("SELECT DISTINCT file_path FROM comment_blocks LIMIT 5")
        file_paths = [row[0] for row in cursor.fetchall()]
        
        # Extract website/source info from file paths
        sources = []
        for path in file_paths:
            filename = os.path.basename(path)
            if 'senso-ji' in filename.lower() or 'asakusa' in filename.lower():
                sources.append("TripAdvisor - Senso-ji Temple, Asakusa")
            elif 'tripadvisor' in filename.lower():
                sources.append("TripAdvisor")
            else:
                # Try to extract meaningful name from filename
                clean_name = filename.replace('%20', ' ').replace('.html', '').replace('.htm', '')
                sources.append(f"Web Source: {clean_name}")
        
        conn.close()
        
        if sources:
            return sources[0]  # Return the first/main source
        else:
            return "Web Source: Unknown"
            
    except Exception as e:
        return f"Database Source: {os.path.basename(db_path)}"

def load_existing_data(OUTPUT_BASE_DIR):
    """Load existing analysis results"""
    try:
        # Load from saved CSV file
        csv_path = os.path.join(OUTPUT_BASE_DIR, "complete_results.csv")
        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(results_df)} results from CSV")
        else:
            print("❌ No existing results CSV found")
            return None, None, None
        
        # Load representative comments
        repr_path = os.path.join(OUTPUT_BASE_DIR, "representative_comments.json")
        if os.path.exists(repr_path):
            with open(repr_path, 'r', encoding='utf-8') as f:
                representative_results = json.load(f)
            print(f"✅ Loaded representative comments")
        else:
            representative_results = {}
        
        # Load performance summary
        perf_path = os.path.join(OUTPUT_BASE_DIR, "performance_summary.json")
        if os.path.exists(perf_path):
            with open(perf_path, 'r', encoding='utf-8') as f:
                performance_summary = json.load(f)
            print(f"✅ Loaded performance summary")
        else:
            performance_summary = {}
        
        return results_df, representative_results, performance_summary
        
    except Exception as e:
        print(f"❌ Error loading existing data: {e}")
        return None, None, None

def generate_pdf_report_simple(results_df, representative_results, performance_summary, db_path, OUTPUT_BASE_DIR, target_url=None, company_name=None):
    """Generate comprehensive PDF report of the analysis"""
    
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
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=stripe_height + 0.5*inch, bottomMargin=0.5*inch)
    story = []
    

    styles = getSampleStyleSheet()
    
   
    red6_red = colors.HexColor('#E30613')           
    primary_color = colors.HexColor('#1a1a1a')      # Dark gray (almost black)
    secondary_color = colors.HexColor('#6b7280')    # Medium gray
    background_light = colors.HexColor('#f5f5f5')   # Light gray background
    

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=26,
        spaceAfter=25,
        alignment=TA_LEFT,
        textColor=primary_color,
        leading=32
    )
    

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=12,
        spaceBefore=20,
        textColor=red6_red,  
        leading=22,
        alignment=TA_LEFT
    )
    
    # Subheading style - Elegant and readable
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceAfter=8,
        spaceBefore=15,
        textColor=primary_color,  # Dark gray for subheadings
        leading=18,
        alignment=TA_LEFT
    )
    
    # Normal text style - Clean body text
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',  # Similar to Open Sans Regular
        fontSize=11,
        spaceAfter=6,
        textColor=primary_color,
        leading=16,
        alignment=TA_JUSTIFY  # Justified alignment for professional look
    )
    
    # Title page
    story.append(Paragraph("Sentiment Analysis Bericht", title_style))
    story.append(Spacer(1, 8))
    
  
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        textColor=secondary_color,
        leading=18,
        alignment=TA_LEFT
    )
    story.append(Paragraph("Powered by GenAI & DistilBERT", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Table of Contents - RIGHT AFTER TITLE
    toc_style = ParagraphStyle(
        'TOCHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=15,
        spaceBefore=10,
        textColor=red6_red,  # 
        leading=22,
        alignment=TA_LEFT
    )
    
    toc_main_style = ParagraphStyle(
        'TOCMain',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        spaceAfter=6,
        leftIndent=10,
        textColor=primary_color,
        leading=16
    )
    
    toc_sub_style = ParagraphStyle(
        'TOCSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        spaceAfter=4,
        leftIndent=30,
        textColor=secondary_color,
        leading=14
    )
    
    story.append(Paragraph("Table of Contents", toc_style))
    story.append(Spacer(1, 10))
    
    # TOC entries with proper hierarchy
    toc_entries = [
        ("1. Executive Summary", toc_main_style),
        ("2. Methodology", toc_main_style),
        ("   2.1 System Overview", toc_sub_style),
        ("   2.2 Processing Pipeline", toc_sub_style),
        ("   2.3 Technical Implementation Details", toc_sub_style),
        ("3. Analysis Visualizations", toc_main_style),
        ("   3.1 Sentiment Analysis Overview", toc_sub_style),
        ("   3.2 Sentiment Word Clouds", toc_sub_style),
        ("   3.3 Word Frequency Analysis", toc_sub_style),
        ("4. Comment Selection Methodology", toc_main_style),
        ("5. Vector-Mean Comments", toc_main_style),
        ("   5.1 Positive Comments", toc_sub_style),
        ("   5.2 Negative Comments", toc_sub_style),
        ("   5.3 Neutral Comments", toc_sub_style),
        ("6. Highest Confidence Comments", toc_main_style),
        ("   6.1 Positive Comments", toc_sub_style),
        ("   6.2 Negative Comments", toc_sub_style),
        ("   6.3 Neutral Comments", toc_sub_style),
        ("7. AI-Generated Sentiment Summaries", toc_main_style),
        ("   7.1 Positive Summary", toc_sub_style),
        ("   7.2 Negative Summary", toc_sub_style),
        ("   7.3 Neutral Summary", toc_sub_style),
        ("8. AI-Generated Recommendations", toc_main_style),
        ("   8.1 Actionable Improvement Suggestions", toc_sub_style),
        ("9. Bankruptcy Insurance Risk Assessment", toc_main_style),
        ("   9.1 Risk Calculation Formula", toc_sub_style),
        ("   9.2 Risk Factors Breakdown", toc_sub_style),
        ("   9.3 Insurance Cost Estimate", toc_sub_style),
        ("10. Technical Details", toc_main_style),
        ("   10.1 Processing Configuration", toc_sub_style),
        ("   10.2 Performance Metrics", toc_sub_style),
        ("   10.3 Database Information", toc_sub_style)
    ]
    
    for entry_text, entry_style in toc_entries:
        story.append(Paragraph(entry_text, entry_style))
    
    story.append(Spacer(1, 20))
    story.append(PageBreak())
    
    # Source information - use TARGET_URL if provided, otherwise fall back to database info
    if target_url:
        source_info = target_url
    else:
        source_info = extract_source_info_from_db(db_path)
    story.append(Paragraph(f"<b>Data Source:</b> {source_info}", normal_style))
    story.append(Spacer(1, 10))
    
    # Analysis date and summary
    story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}", normal_style))
    story.append(Paragraph(f"<b>Total Comments Analyzed:</b> {len(results_df) if results_df is not None else 'N/A'}", normal_style))
    story.append(Paragraph(f"<b>Neural Network Model:</b> DistilBERT-based sentiment classifier", normal_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    if results_df is not None:
        sentiment_dist = results_df['sentiment'].value_counts().to_dict()
        total_comments = len(results_df)
        
        summary_text = f"""
        This report presents a comprehensive sentiment analysis of {total_comments} user comments extracted 
        from {source_info} using advanced neural network classification. The analysis employed DistilBERT 
        for sentiment classification and TF-IDF vectorization with K-means clustering for representative 
        comment selection.
        
        <b>Key Findings:</b><br/>
        • Positive Comments: {sentiment_dist.get('POSITIVE', 0)} ({sentiment_dist.get('POSITIVE', 0)/total_comments*100:.1f}%)<br/>
        • Negative Comments: {sentiment_dist.get('NEGATIVE', 0)} ({sentiment_dist.get('NEGATIVE', 0)/total_comments*100:.1f}%)<br/>
        • Neutral Comments: {sentiment_dist.get('NEUTRAL', 0)} ({sentiment_dist.get('NEUTRAL', 0)/total_comments*100:.1f}%)<br/>
        
        The analysis utilized vector search and clustering algorithms to identify the most representative 
        comments from each sentiment category, providing insights into common themes and user experiences.
        """
    else:
        summary_text = "Analysis data could not be loaded. Please check data files."
    
    story.append(Paragraph(summary_text, normal_style))
    story.append(PageBreak())
    
    # Methodology
    story.append(Paragraph("Methodology", heading_style))
    
    # Pipeline Overview
    pipeline_overview = """
    <b>2.1 System Overview:</b><br/><br/>
    
    This system implements a comprehensive AI pipeline utilizing three complementary neural network 
    architectures (BERT, BART, and LLaMA) to extract, analyze, and synthesize user-generated feedback 
    into actionable insights. The pipeline processes any type of categorizable information, from 
    restaurant reviews (helping service providers improve their offerings) to creditworthiness 
    assessment and risk estimation for individuals or organizations.<br/><br/>
    
    Beyond generating summaries, the system delivers two critical metrics:<br/>
    <b>1. Semantic Vector Analysis:</b> Identifies representative comments (positive, negative, and 
    neutral) that encapsulate the most frequently encountered themes across the entire dataset.<br/>
    <b>2. Confidence-Based Selection:</b> Extracts comments with the highest sentiment confidence 
    scores, revealing the most emphatic responses (highly satisfied, strongly dissatisfied, or 
    distinctly neutral perspectives).<br/><br/>
    """
    
    story.append(Paragraph(pipeline_overview, normal_style))
    story.append(Spacer(1, 15))
    story.append(Spacer(1, 15))
    # ==================== INSERT PIPELINE DIAGRAM HERE ====================
    # Try to insert the Automated_pipeline.pdf as an image/figure
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_pdf_path = os.path.join(script_dir, 'Images', 'Automated_pipeline.png')
    
    # Check if the file exists
    if os.path.exists(pipeline_pdf_path):
        try:
            # Add a centered caption/title for the figure
            figure_caption_style = ParagraphStyle(
                'FigureCaption',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=11,
                alignment=TA_CENTER,
                textColor=secondary_color,
                spaceAfter=6
            )
            
            
            # Insert the PDF as an image (ReportLab can handle PDF pages)
            # Adjust width and height as needed - here using 6.5 inches wide
            pipeline_img = Image(pipeline_pdf_path, width=6.5*inch, height=4*inch)
            story.append(pipeline_img)
            story.append(Spacer(1, 15))

            story.append(Paragraph("Automated Processing Pipeline Architecture", figure_caption_style))
            story.append(Spacer(1, 8))
            
            print(f"✅ Successfully inserted pipeline diagram from {pipeline_pdf_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not insert pipeline PDF: {e}")
            story.append(Paragraph(f"<i>[Pipeline diagram not available: {str(e)}]</i>", styles['Italic']))
            story.append(Spacer(1, 10))
    else:
        print(f"⚠️ Warning: Pipeline PDF not found at {pipeline_pdf_path}")
        story.append(Paragraph("<i>[Pipeline diagram file not found]</i>", styles['Italic']))
        story.append(Spacer(1, 10))
    # ======================================================================
    
    # Continue with Processing Pipeline section - MOVE TO NEXT PAGE
    story.append(PageBreak())
    story.append(Paragraph("<b>2.2 Processing Pipeline:</b>", normal_style))
    story.append(Spacer(1, 8))
    
    pipeline_stages = """
    <b>Stage 1 - Data Acquisition:</b> Target webpage URL → Automated text retrieval from web content<br/><br/>
    
    <b>Stage 2 - Content Extraction:</b> Raw HTML → Clean, structured comment blocks<br/><br/>
    
    <b>Stage 3 - Sentiment Classification:</b> BERT-based LLM (DistilBERT via HuggingFace) → 
    Categorization into Positive/Negative/Neutral with confidence scores<br/><br/>
    
    <b>Stage 4 - Representative Discovery:</b> TF-IDF Vectorization + K-means Clustering → 
    Identification of thematically central comments<br/><br/>
    
    <b>Stage 5 - Summarization:</b> LLaMA LLM (via Groq API) → Concise synthesis of sentiment patterns 
    and key themes<br/><br/>
    
    <b>Stage 6 - Recommendation Generation:</b> LLaMA LLM analysis of positive/negative summaries → 
    Actionable improvement suggestions<br/><br/>
    
    This multi-stage architecture ensures robust analysis through specialized neural networks, 
    each optimized for its specific task in the pipeline.
    """
    
    story.append(Paragraph(pipeline_stages, normal_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("2.3 Technical Implementation Details", subheading_style))
    story.append(Spacer(1, 10))
    
    methodology_text = """
    <b>1. Data Extraction:</b> Comments were pre-filtered using neural network-based quality scoring 
    to identify genuine user-generated content versus website boilerplate.<br/><br/>
    
    <b>2. Sentiment Classification:</b> DistilBERT model processed each comment to classify sentiment 
    as Positive, Negative, or Neutral with confidence scores.<br/><br/>
    
    <b>3. Vector Search & Clustering:</b> TF-IDF vectorization transformed text into numerical 
    representations, followed by K-means clustering to group similar comments.<br/><br/>
    
    <b>4. Representative Selection:</b> For each cluster, the comment closest to the centroid 
    was selected as the most representative example of that theme.<br/><br/>
    
    <b>5. AI-Powered Summarization:</b> LLaMA 3.1 model (8B parameters) analyzed representative 
    comments to generate concise summaries highlighting key patterns and themes.<br/><br/>
    
    <b>6. Recommendation Synthesis:</b> LLaMA model processed positive and negative summaries to 
    produce actionable improvement recommendations tailored to the specific domain.
    """
    
    story.append(Paragraph(methodology_text, normal_style))
    story.append(Spacer(1, 20))
    
    # Add visualizations with descriptions
    viz_folder = os.path.join(OUTPUT_BASE_DIR, 'visualizations')
    
    # 3.1 Sentiment Analysis Overview with icon next to title
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sentiment_img_path = os.path.join(script_dir, 'Images', 'Sentiment_analysis.png')
    
    # Create title with icon on the same line (50% bigger: 0.6 -> 0.9 inch)
    if os.path.exists(sentiment_img_path):
        try:
            sentiment_img = Image(sentiment_img_path, width=0.9*inch, height=0.675*inch)
            title_para = Paragraph("3.1 Sentiment Analysis Overview", subheading_style)
            data = [[title_para, sentiment_img]]
            t = Table(data, colWidths=[5.2*inch, 1.1*inch])
            t.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ]))
            story.append(t)
        except Exception as e:
            story.append(Paragraph("3.1 Sentiment Analysis Overview", subheading_style))
    else:
        story.append(Paragraph("3.1 Sentiment Analysis Overview", subheading_style))
    
    story.append(Spacer(1, 8))
    
    # Add descriptive text
    sentiment_analysis_description = """
    The sentiment analysis overview provides a comprehensive visual breakdown of the comment distribution. 
    The pie chart displays the proportional fractions of positive, neutral, and negative comments, offering 
    an immediate understanding of overall sentiment balance. The sentiment trends over time chart illustrates 
    how positive and negative sentiments fluctuate across the analysis period, revealing temporal patterns 
    and shifts in user opinions. Additionally, the confidence score distribution histogram shows the reliability 
    of sentiment classifications, with higher scores indicating greater model certainty.
    """
    story.append(Paragraph(sentiment_analysis_description, normal_style))
    story.append(Spacer(1, 15))
    
    # Add full-size visualization from OUTPUT_BASE_DIR
    viz_path = os.path.join(viz_folder, 'sentiment_analysis_overview.png')
    if os.path.exists(viz_path):
        try:
            img = Image(viz_path, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 15))
        except Exception as e:
            story.append(Paragraph(f"[Visualization not available: {str(e)}]", styles['Italic']))
    
    # 3.2 Sentiment Word Clouds
    viz_path = os.path.join(viz_folder, 'sentiment_wordclouds.png')
    if os.path.exists(viz_path):
        story.append(Paragraph("3.2 Sentiment Word Clouds", subheading_style))
        try:
            img = Image(viz_path, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 15))
        except Exception as e:
            story.append(Paragraph(f"[Visualization not available: {str(e)}]", styles['Italic']))
    
    # 3.3 Word Frequency Analysis
    viz_path = os.path.join(viz_folder, 'word_frequency_analysis.png')
    if os.path.exists(viz_path):
        story.append(Paragraph("3.3 Word Frequency Analysis", subheading_style))
        try:
            img = Image(viz_path, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 15))
        except Exception as e:
            story.append(Paragraph(f"[Visualization not available: {str(e)}]", styles['Italic']))
    
    story.append(PageBreak())
    
    # Explanatory Section - How Comments Were Selected
    story.append(Paragraph("Comment Selection Methodology", heading_style))
    story.append(Spacer(1, 10))
    
    methodology_explanation = """
    This report displays two types of representative comments for each sentiment category:<br/><br/>
    
    <b>1. Vector-Mean Comments (Cluster Centroids):</b><br/>
    These comments were selected using TF-IDF vectorization and K-means clustering. For each sentiment, 
    comments were grouped into 10 clusters based on semantic similarity. The comment closest to each 
    cluster's centroid (mathematical center in vector space) represents the most typical expression of 
    that theme. We display the top cluster's representative comment here.<br/><br/>
    
    <b>2. Highest Confidence Comments:</b><br/>
    These are the comments where the DistilBERT sentiment classifier had the highest confidence score 
    (closest to 1.0). These represent the clearest, most unambiguous examples of each sentiment, where 
    the neural network was most certain about the classification.<br/><br/>
    
    <b>3. AI-Generated Summaries:</b><br/>
    Using the Groq API with Llama 3.1 model, we generated concise 2-3 sentence summaries that synthesize 
    the main themes and patterns across all representative comments in each sentiment category.
    """
    
    story.append(Paragraph(methodology_explanation, normal_style))
    story.append(Spacer(1, 20))
    story.append(PageBreak())
    
    # Vector-Mean Comments Section
    story.append(Paragraph("5. Vector-Mean Comments", heading_style))
    story.append(Spacer(1, 12))

    sentiment_subsections = {
        'POSITIVE': '5.1',
        'NEGATIVE': '5.2',
        'NEUTRAL': '5.3'
    }

    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        if sentiment in representative_results and len(representative_results[sentiment]) > 0:
            subsection_num = sentiment_subsections[sentiment]
            story.append(Paragraph(f"{subsection_num} {sentiment.title()} Comments", subheading_style))
            story.append(Spacer(1, 6))
            
            # Get only the top representative comment (highest confidence)
            best_rep = representative_results[sentiment][0]  # Top 1
            
            # Truncate comment to fit page properly (max 300 characters)
            comment_text = str(best_rep.get('text', ''))
            if len(comment_text) > 300:
                comment_text = comment_text[:300] + "..."
            
            # Add confidence score
            confidence = best_rep.get('confidence', 0)
            cluster_info = f"Cluster {best_rep.get('cluster_id', 'N/A')} (Size: {best_rep.get('cluster_size', 'N/A')})"
            
            story.append(Paragraph(f"<b>Confidence:</b> {confidence:.3f}", normal_style))
            story.append(Paragraph(f"<b>Cluster Info:</b> {cluster_info}", normal_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Comment:</b> {comment_text}", normal_style))
            story.append(Spacer(1, 12))
    
    # Highest Confidence Comments Section
    story.append(PageBreak())
    
    # Add title with side image (30% smaller: 2.0 -> 1.4 inch)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    confidence_img_path = os.path.join(script_dir, 'Images', 'Highest_Confidence_Comments.png')
    
    if os.path.exists(confidence_img_path):
        try:
            confidence_img = Image(confidence_img_path, width=1.4*inch, height=1.05*inch)
            title_para = Paragraph("6. Highest Confidence Comments", heading_style)
            data = [[title_para, confidence_img]]
            t = Table(data, colWidths=[4.8*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ]))
            story.append(t)
        except Exception as e:
            story.append(Paragraph("6. Highest Confidence Comments", heading_style))
    else:
        story.append(Paragraph("6. Highest Confidence Comments", heading_style))
    
    story.append(Spacer(1, 12))
    
    confidence_subsections = {
        'POSITIVE': '6.1',
        'NEGATIVE': '6.2',
        'NEUTRAL': '6.3'
    }
    
    if results_df is not None:
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            sentiment_data = results_df[results_df['sentiment'] == sentiment]
            if len(sentiment_data) > 0:
                subsection_num = confidence_subsections[sentiment]
                story.append(Paragraph(f"{subsection_num} {sentiment.title()} Comments", subheading_style))
                story.append(Spacer(1, 6))
                
                # Find comment with highest confidence
                highest_conf_idx = sentiment_data['confidence'].idxmax()
                highest_conf_comment = sentiment_data.loc[highest_conf_idx]
                
                # Truncate comment
                comment_text = str(highest_conf_comment['text'])
                if len(comment_text) > 300:
                    comment_text = comment_text[:300] + "..."
                
                confidence = highest_conf_comment['confidence']
                
                story.append(Paragraph(f"<b>Confidence Score:</b> {confidence:.4f}", normal_style))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Comment:</b> {comment_text}", normal_style))
                story.append(Spacer(1, 12))
    
    # LLM-Generated Summaries Section
    story.append(PageBreak())
    
    # Add title with icon on the same line (50% bigger: 0.6 -> 0.9 inch)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summaries_img_path = os.path.join(script_dir, 'Images', 'AI_Generated_Sentiment_Summaries.png')
    
    if os.path.exists(summaries_img_path):
        try:
            summaries_img = Image(summaries_img_path, width=0.9*inch, height=0.675*inch)
            title_para = Paragraph("7. AI-Generated Sentiment Summaries", heading_style)
            data = [[title_para, summaries_img]]
            t = Table(data, colWidths=[5.2*inch, 1.1*inch])
            t.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ]))
            story.append(t)
        except Exception as e:
            story.append(Paragraph("7. AI-Generated Sentiment Summaries", heading_style))
    else:
        story.append(Paragraph("7. AI-Generated Sentiment Summaries", heading_style))
    
    story.append(Spacer(1, 12))
    
    summary_subsections = {
        'positive': '7.1',
        'negative': '7.2',
        'neutral': '7.3'
    }
    
    # Load LLM summaries from files
    for sentiment in ['positive', 'negative', 'neutral']:
        summary_file = os.path.join(OUTPUT_BASE_DIR, sentiment, f'{sentiment}_summary.json')
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                subsection_num = summary_subsections[sentiment]
                story.append(Paragraph(f"{subsection_num} {sentiment.title()} Summary", subheading_style))
                story.append(Spacer(1, 6))
                
                summary_text = summary_data.get('summary', 'No summary available')
                llm_model = summary_data.get('model_used', 'Unknown')
                num_comments = summary_data.get('num_comments_analyzed', 'N/A')
                
                story.append(Paragraph(f"<b>Model Used:</b> {llm_model}", normal_style))
                story.append(Paragraph(f"<b>Comments Analyzed:</b> {num_comments}", normal_style))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Summary:</b> {summary_text}", normal_style))
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"<i>Summary for {sentiment} not available</i>", styles['Italic']))
                story.append(Spacer(1, 12))
    
    # AI-Generated Recommendations Section
    story.append(PageBreak())
    story.append(Paragraph("8. AI-Generated Recommendations", heading_style))
    story.append(Spacer(1, 12))
    
    # Load recommendations from file
    recommendation_file = os.path.join(OUTPUT_BASE_DIR, 'recommendation', 'recommendation.json')
    if os.path.exists(recommendation_file):
        try:
            with open(recommendation_file, 'r', encoding='utf-8') as f:
                recommendation_data = json.load(f)
            
            recommendation_text = recommendation_data.get('recommendation', 'No recommendations available')
            llm_model = recommendation_data.get('model_used', 'Unknown')
            timestamp = recommendation_data.get('generated_timestamp', 'N/A')
            
            story.append(Paragraph("8.1 Actionable Improvement Suggestions", subheading_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Based on Analysis of Positive and Negative Feedback</b>", normal_style))
            story.append(Paragraph(f"<b>Model Used:</b> {llm_model}", normal_style))
            story.append(Paragraph(f"<b>Generated:</b> {timestamp}", normal_style))
            story.append(Spacer(1, 10))
            
            # Split recommendation text into paragraphs for better formatting
            for para in recommendation_text.split('\n\n'):
                if para.strip():
                    story.append(Paragraph(para, normal_style))
                    story.append(Spacer(1, 8))
            
        except Exception as e:
            story.append(Paragraph(f"<i>Recommendations not available: {str(e)}</i>", styles['Italic']))
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("<i>No recommendations file found</i>", styles['Italic']))
        story.append(Spacer(1, 12))
    
    # Bankruptcy Insurance Risk Assessment Section
    story.append(PageBreak())
    story.append(Paragraph("9. Bankruptcy Insurance Risk Assessment", heading_style))
    story.append(Spacer(1, 12))
    
    # Load insurance risk calculation
    insurance_file = os.path.join(OUTPUT_BASE_DIR, 'insurance_risk.json')
    print(f"🔍 DEBUG: Looking for insurance file at: {insurance_file}")
    print(f"🔍 DEBUG: File exists: {os.path.exists(insurance_file)}")
    if os.path.exists(insurance_file):
        try:
            with open(insurance_file, 'r', encoding='utf-8') as f:
                insurance_data = json.load(f)
            
            insurance_cost = insurance_data.get('insurance_cost', 0)
            risk_level = insurance_data.get('risk_level', 'Unknown')
            risk_score = insurance_data.get('risk_score', 0)
            breakdown = insurance_data.get('breakdown', {})
            
            # Risk assessment overview
            story.append(Paragraph("9.1 Risk Calculation Formula", subheading_style))
            story.append(Spacer(1, 6))
            
            formula_text = """
            This assessment calculates bankruptcy insurance cost for loan applications based on customer sentiment analysis.
            The formula considers multiple risk factors:<br/><br/>
            
            <b>Insurance Cost = Base Rate × Sentiment Multiplier × Confidence Multiplier × Sample Size Multiplier × Trend Multiplier</b><br/><br/>
            
            Where:<br/>
            • <b>Sentiment Multiplier</b>: Increases with negative reviews (1.0 + negative_ratio × 2.5)<br/>
            • <b>Confidence Multiplier</b>: Accounts for prediction uncertainty (1.5 - average_confidence × 0.5)<br/>
            • <b>Sample Size Multiplier</b>: Adjusts for data sufficiency (higher for small samples)<br/>
            • <b>Trend Multiplier</b>: Reflects recent sentiment changes (0.9 to 1.4 based on trends)
            """
            story.append(Paragraph(formula_text, normal_style))
            story.append(Spacer(1, 12))
            
            # Risk factors breakdown
            story.append(Paragraph("9.2 Risk Factors Breakdown", subheading_style))
            story.append(Spacer(1, 6))
            
            sentiment_factors = breakdown.get('sentiment_factors', {})
            confidence_factors = breakdown.get('confidence_factors', {})
            sample_factors = breakdown.get('sample_factors', {})
            trend_factors = breakdown.get('trend_factors', {})
            base_rate = breakdown.get('base_rate', 5000)
            
            # Create breakdown table
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
            story.append(breakdown_table)
            story.append(Spacer(1, 12))
            
            # Insurance cost estimate
            story.append(Paragraph("9.3 Insurance Cost Estimate", subheading_style))
            story.append(Spacer(1, 6))
            
            # Risk level color coding
            risk_colors = {
                'Low': colors.green,
                'Medium': colors.orange,
                'High': colors.orangered,
                'Critical': colors.red
            }
            risk_color = risk_colors.get(risk_level, colors.grey)
            
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
            story.append(result_table)
            story.append(Spacer(1, 12))
            
            # Interpretation
            interpretation = f"""
            <b>Interpretation:</b><br/>
            Based on the sentiment analysis of customer reviews, the estimated bankruptcy insurance cost for this restaurant 
            is <b>${insurance_cost:,.2f}</b> with a <b>{risk_level}</b> risk level (score: {risk_score}/100).<br/><br/>
            
            This assessment considers the overall sentiment distribution ({sentiment_factors.get('positive_percentage', 0)}% positive, 
            {sentiment_factors.get('negative_percentage', 0)}% negative), the confidence of sentiment predictions 
            (avg: {confidence_factors.get('average_confidence', 0):.1%}), and recent sentiment trends 
            ({trend_factors.get('trend_status', 'stable').lower()}).
            """
            story.append(Paragraph(interpretation, normal_style))
            story.append(Spacer(1, 8))
            
        except Exception as e:
            story.append(Paragraph(f"<i>Insurance risk assessment not available: {str(e)}</i>", styles['Italic']))
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("<i>Insurance risk assessment file not found</i>", styles['Italic']))
        story.append(Spacer(1, 12))
    
    # Technical Analysis Section
    
    # Technical Details
    story.append(PageBreak())
    story.append(Paragraph("10. Technical Details", heading_style))
    
    tech_details = f"""
    <b>10.1 Processing Configuration:</b><br/>
    • TF-IDF Features: 1000<br/>
    • Minimum Document Frequency: 4<br/>
    • Maximum Document Frequency: 0.8<br/>
    • Clusters per Sentiment: 10<br/>
    • Confidence Threshold: 0.8<br/><br/>
    
    <b>10.2 Performance Metrics:</b><br/>
    • Average Sentiment Confidence: {performance_summary.get('score_distribution', {}).get('avg_sentiment_confidence', 'N/A')}<br/>
    • Total Samples Processed: {performance_summary.get('total_samples', 'N/A')}<br/>
    • Processing Time: {performance_summary.get('processing_time_minutes', 'N/A')} minutes<br/><br/>
    
    <b>10.3 Database Information:</b><br/>
    • Source Database: {os.path.basename(db_path)}<br/>
    • Analysis Timestamp: {datetime.now().isoformat()}
    """
    
    story.append(Paragraph(tech_details, normal_style))
    
    # Build PDF with custom header on every page
    try:
        doc.build(story, onFirstPage=lambda c, d: draw_header_stripe(c, d, company_name), onLaterPages=lambda c, d: draw_header_stripe(c, d, company_name))
        print(f"📄 PDF report generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        return None

def generate_pdf_fun(DB_PATH, OUTPUT_BASE_DIR, TARGET_URL=None, company_name=None):
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
    pdf_path = generate_pdf_report_simple(results_df, representative_results, performance_summary, DB_PATH, OUTPUT_BASE_DIR, TARGET_URL, company_name)
    
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

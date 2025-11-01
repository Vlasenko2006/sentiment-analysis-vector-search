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
from datetime import datetime

# PDF report generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️ ReportLab not available. Install with: pip install reportlab")
    PDF_AVAILABLE = False

# Configuration
OUTPUT_BASE_DIR = "/Users/andreyvlasenko/tst/Request/my_volume/sentiment_analysis"
DB_PATH = "/Users/andreyvlasenko/tst/Request/filtered_reviews.db"

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

def load_existing_data():
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

def generate_pdf_report_simple(results_df, representative_results, performance_summary, db_path):
    """Generate comprehensive PDF report of the analysis"""
    
    if not PDF_AVAILABLE:
        print("❌ Cannot generate PDF report. Install reportlab: pip install reportlab")
        return None
    
    # Create PDF file path
    pdf_path = os.path.join(OUTPUT_BASE_DIR, 'visualizations', 'sentiment_analysis_report.pdf')
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#000080')  # Navy blue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#800000')  # Maroon
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=15,
        textColor=colors.HexColor('#000080')  # Navy blue
    )
    
    # Title page
    story.append(Paragraph("RoBERTa Sentiment Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Source information
    source_info = extract_source_info_from_db(db_path)
    story.append(Paragraph(f"<b>Data Source:</b> {source_info}", styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Analysis date and summary
    story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Comments Analyzed:</b> {len(results_df) if results_df is not None else 'N/A'}", styles['Normal']))
    story.append(Paragraph(f"<b>Neural Network Model:</b> DistilBERT-based sentiment classifier", styles['Normal']))
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
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(PageBreak())
    
    # Methodology
    story.append(Paragraph("Methodology", heading_style))
    methodology_text = """
    <b>1. Data Extraction:</b> Comments were pre-filtered using neural network-based quality scoring 
    to identify genuine user-generated content versus website boilerplate.<br/><br/>
    
    <b>2. Sentiment Classification:</b> DistilBERT model processed each comment to classify sentiment 
    as Positive, Negative, or Neutral with confidence scores.<br/><br/>
    
    <b>3. Vector Search & Clustering:</b> TF-IDF vectorization transformed text into numerical 
    representations, followed by K-means clustering to group similar comments.<br/><br/>
    
    <b>4. Representative Selection:</b> For each cluster, the comment closest to the centroid 
    was selected as the most representative example of that theme.
    """
    
    story.append(Paragraph(methodology_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add visualizations if they exist
    viz_folder = os.path.join(OUTPUT_BASE_DIR, 'visualizations')
    viz_files = ['sentiment_analysis_overview.png', 'sentiment_wordclouds.png', 'word_frequency_analysis.png']
    
    for viz_file in viz_files:
        viz_path = os.path.join(viz_folder, viz_file)
        if os.path.exists(viz_path):
            story.append(Paragraph(f"Analysis Visualization: {viz_file.replace('_', ' ').title()}", subheading_style))
            try:
                # Add image with appropriate sizing
                img = Image(viz_path, width=7*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 15))
            except Exception as e:
                story.append(Paragraph(f"[Visualization not available: {str(e)}]", styles['Italic']))
    
    story.append(PageBreak())
    
    # Representative Comments Section
    story.append(Paragraph("Most Representative Comments", heading_style))
    story.append(Spacer(1, 12))

    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        if sentiment in representative_results and len(representative_results[sentiment]) > 0:
            story.append(Paragraph(f"Most Representative {sentiment.title()} Comment", subheading_style))
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
            
            story.append(Paragraph(f"<b>Confidence:</b> {confidence:.3f}", styles['Normal']))
            story.append(Paragraph(f"<b>Cluster Info:</b> {cluster_info}", styles['Normal']))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Comment:</b> {comment_text}", styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Technical Analysis Section
    
    # Technical Details
    story.append(PageBreak())
    story.append(Paragraph("Technical Details", heading_style))
    
    tech_details = f"""
    <b>Processing Configuration:</b><br/>
    • TF-IDF Features: 1000<br/>
    • Minimum Document Frequency: 4<br/>
    • Maximum Document Frequency: 0.8<br/>
    • Clusters per Sentiment: 10<br/>
    • Confidence Threshold: 0.8<br/><br/>
    
    <b>Performance Metrics:</b><br/>
    • Average Sentiment Confidence: {performance_summary.get('score_distribution', {}).get('avg_sentiment_confidence', 'N/A')}<br/>
    • Total Samples Processed: {performance_summary.get('total_samples', 'N/A')}<br/>
    • Processing Time: {performance_summary.get('processing_time_minutes', 'N/A')} minutes<br/><br/>
    
    <b>Database Information:</b><br/>
    • Source Database: {os.path.basename(db_path)}<br/>
    • Analysis Timestamp: {datetime.now().isoformat()}
    """
    
    story.append(Paragraph(tech_details, styles['Normal']))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"📄 PDF report generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        return None

def main():
    print("📄 Generating PDF Report from Existing Data...")
    
    # Load existing analysis data
    results_df, representative_results, performance_summary = load_existing_data()
    
    if results_df is None:
        print("❌ Cannot generate PDF without analysis data")
        return
    
    # Generate PDF report
    pdf_path = generate_pdf_report_simple(results_df, representative_results, performance_summary, DB_PATH)
    
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

if __name__ == "__main__":
    main()
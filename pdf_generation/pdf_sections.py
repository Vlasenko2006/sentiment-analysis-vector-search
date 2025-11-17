"""
PDF Sections Module
Handles creation of all major report sections
"""

import os
import json
from datetime import datetime
from reportlab.platypus import Paragraph, Spacer, PageBreak
from .pdf_images import (
    get_pipeline_diagram, get_sentiment_icon, get_confidence_icon,
    get_summaries_icon, add_visualization
)
from .pdf_tables import (
    create_title_with_image_table, create_insurance_breakdown_table,
    create_insurance_result_table
)


def create_title_page(story, styles):
    """Create title page with TOC"""
    story.append(Paragraph("Sentiment Analysis Bericht", styles['title']))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Powered by GenAI & DistilBERT", styles['subtitle']))
    story.append(Spacer(1, 20))


def create_table_of_contents(story, styles):
    """Create table of contents"""
    story.append(Paragraph("Table of Contents", styles['toc']))
    story.append(Spacer(1, 10))
    
    # TOC entries with proper hierarchy
    toc_entries = [
        ("1. Executive Summary", styles['toc_main']),
        ("2. Methodology", styles['toc_main']),
        ("   2.1 System Overview", styles['toc_sub']),
        ("   2.2 Processing Pipeline", styles['toc_sub']),
        ("   2.3 Technical Implementation Details", styles['toc_sub']),
        ("3. Analysis Visualizations", styles['toc_main']),
        ("   3.1 Sentiment Analysis Overview", styles['toc_sub']),
        ("   3.2 Sentiment Word Clouds", styles['toc_sub']),
        ("   3.3 Word Frequency Analysis", styles['toc_sub']),
        ("4. Comment Selection Methodology", styles['toc_main']),
        ("5. Vector-Mean Comments", styles['toc_main']),
        ("   5.1 Positive Comments", styles['toc_sub']),
        ("   5.2 Negative Comments", styles['toc_sub']),
        ("   5.3 Neutral Comments", styles['toc_sub']),
        ("6. Highest Confidence Comments", styles['toc_main']),
        ("   6.1 Positive Comments", styles['toc_sub']),
        ("   6.2 Negative Comments", styles['toc_sub']),
        ("   6.3 Neutral Comments", styles['toc_sub']),
        ("7. AI-Generated Sentiment Summaries", styles['toc_main']),
        ("   7.1 Positive Summary", styles['toc_sub']),
        ("   7.2 Negative Summary", styles['toc_sub']),
        ("   7.3 Neutral Summary", styles['toc_sub']),
        ("8. AI-Generated Recommendations", styles['toc_main']),
        ("   8.1 Actionable Improvement Suggestions", styles['toc_sub']),
        ("9. Bankruptcy Insurance Risk Assessment", styles['toc_main']),
        ("   9.1 Risk Calculation Formula", styles['toc_sub']),
        ("   9.2 Risk Factors Breakdown", styles['toc_sub']),
        ("   9.3 Insurance Cost Estimate", styles['toc_sub']),
        ("10. Technical Details", styles['toc_main']),
        ("   10.1 Processing Configuration", styles['toc_sub']),
        ("   10.2 Performance Metrics", styles['toc_sub']),
        ("   10.3 Database Information", styles['toc_sub'])
    ]
    
    for entry_text, entry_style in toc_entries:
        story.append(Paragraph(entry_text, entry_style))
    
    story.append(Spacer(1, 20))
    story.append(PageBreak())


def create_executive_summary(story, styles, results_df, source_info):
    """Create executive summary section"""
    story.append(Paragraph(f"<b>Data Source:</b> {source_info}", styles['normal']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['normal']))
    story.append(Paragraph(f"<b>Total Comments Analyzed:</b> {len(results_df) if results_df is not None else 'N/A'}", styles['normal']))
    story.append(Paragraph(f"<b>Neural Network Model:</b> DistilBERT-based sentiment classifier", styles['normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Executive Summary", styles['heading']))
    
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
    
    story.append(Paragraph(summary_text, styles['normal']))
    story.append(PageBreak())


def create_methodology_section(story, styles, script_dir):
    """Create methodology section with pipeline diagram"""
    story.append(Paragraph("Methodology", styles['heading']))
    
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
    
    story.append(Paragraph(pipeline_overview, styles['normal']))
    story.append(Spacer(1, 15))
    
    # Try to insert pipeline diagram
    pipeline_img, message = get_pipeline_diagram(script_dir)
    if pipeline_img:
        story.append(pipeline_img)
        story.append(Spacer(1, 15))
        story.append(Paragraph("Automated Processing Pipeline Architecture", styles['figure_caption']))
        story.append(Spacer(1, 8))
        print(message)
    else:
        story.append(Paragraph(f"<i>[Pipeline diagram not available]</i>", styles['base']['Italic']))
        story.append(Spacer(1, 10))
        print(message)
    
    # Continue with Processing Pipeline section
    story.append(PageBreak())
    story.append(Paragraph("<b>2.2 Processing Pipeline:</b>", styles['normal']))
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
    
    story.append(Paragraph(pipeline_stages, styles['normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("2.3 Technical Implementation Details", styles['subheading']))
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
    
    story.append(Paragraph(methodology_text, styles['normal']))
    story.append(Spacer(1, 20))


def create_visualizations_section(story, styles, viz_folder, script_dir):
    """Create visualizations section"""
    # 3.1 Sentiment Analysis Overview with icon
    sentiment_img = get_sentiment_icon(script_dir)
    if sentiment_img:
        try:
            title_para = Paragraph("3.1 Sentiment Analysis Overview", styles['subheading'])
            t = create_title_with_image_table(title_para, sentiment_img)
            story.append(t)
        except Exception:
            story.append(Paragraph("3.1 Sentiment Analysis Overview", styles['subheading']))
    else:
        story.append(Paragraph("3.1 Sentiment Analysis Overview", styles['subheading']))
    
    story.append(Spacer(1, 8))
    
    sentiment_analysis_description = """
    The sentiment analysis overview provides a comprehensive visual breakdown of the comment distribution.
    The pie chart displays the proportional fractions of positive, neutral, and negative comments, offering
    an immediate understanding of overall sentiment balance. The sentiment trends over time chart illustrates
    how positive and negative sentiments fluctuate across the analysis period, revealing temporal patterns
    and shifts in user opinions. Additionally, the confidence score distribution histogram shows the reliability
    of sentiment classifications, with higher scores indicating greater model certainty.
    """
    story.append(Paragraph(sentiment_analysis_description, styles['normal']))
    story.append(Spacer(1, 15))
    
    # Add visualizations
    viz_img, exists = add_visualization(viz_folder, 'sentiment_analysis_overview.png')
    if viz_img:
        story.append(viz_img)
        story.append(Spacer(1, 15))
    
    # 3.2 Sentiment Word Clouds
    viz_img, exists = add_visualization(viz_folder, 'sentiment_wordclouds.png')
    if exists:
        story.append(Paragraph("3.2 Sentiment Word Clouds", styles['subheading']))
        if viz_img:
            story.append(viz_img)
        story.append(Spacer(1, 15))
    
    # 3.3 Word Frequency Analysis
    viz_img, exists = add_visualization(viz_folder, 'word_frequency_analysis.png')
    if exists:
        story.append(Paragraph("3.3 Word Frequency Analysis", styles['subheading']))
        if viz_img:
            story.append(viz_img)
        story.append(Spacer(1, 15))
    
    story.append(PageBreak())


def create_comment_methodology_section(story, styles):
    """Create comment selection methodology explanation"""
    story.append(Paragraph("Comment Selection Methodology", styles['heading']))
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
    
    story.append(Paragraph(methodology_explanation, styles['normal']))
    story.append(Spacer(1, 20))
    story.append(PageBreak())


def create_vector_mean_comments_section(story, styles, representative_results):
    """Create vector-mean comments section"""
    story.append(Paragraph("5. Vector-Mean Comments", styles['heading']))
    story.append(Spacer(1, 12))

    sentiment_subsections = {
        'POSITIVE': '5.1',
        'NEGATIVE': '5.2',
        'NEUTRAL': '5.3'
    }

    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        if sentiment in representative_results and len(representative_results[sentiment]) > 0:
            subsection_num = sentiment_subsections[sentiment]
            story.append(Paragraph(f"{subsection_num} {sentiment.title()} Comments", styles['subheading']))
            story.append(Spacer(1, 6))
            
            # Get only the top representative comment
            best_rep = representative_results[sentiment][0]
            
            # Truncate comment to fit page properly
            comment_text = str(best_rep.get('text', ''))
            if len(comment_text) > 300:
                comment_text = comment_text[:300] + "..."
            
            confidence = best_rep.get('confidence', 0)
            cluster_info = f"Cluster {best_rep.get('cluster_id', 'N/A')} (Size: {best_rep.get('cluster_size', 'N/A')})"
            
            story.append(Paragraph(f"<b>Confidence:</b> {confidence:.3f}", styles['normal']))
            story.append(Paragraph(f"<b>Cluster Info:</b> {cluster_info}", styles['normal']))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Comment:</b> {comment_text}", styles['normal']))
            story.append(Spacer(1, 12))


def create_highest_confidence_section(story, styles, results_df, script_dir):
    """Create highest confidence comments section"""
    story.append(PageBreak())
    
    # Add title with icon
    confidence_img = get_confidence_icon(script_dir)
    if confidence_img:
        try:
            title_para = Paragraph("6. Highest Confidence Comments", styles['heading'])
            t = create_title_with_image_table(title_para, confidence_img, title_width=4.8, image_width=1.5)
            story.append(t)
        except Exception:
            story.append(Paragraph("6. Highest Confidence Comments", styles['heading']))
    else:
        story.append(Paragraph("6. Highest Confidence Comments", styles['heading']))
    
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
                story.append(Paragraph(f"{subsection_num} {sentiment.title()} Comments", styles['subheading']))
                story.append(Spacer(1, 6))
                
                # Find comment with highest confidence
                highest_conf_idx = sentiment_data['confidence'].idxmax()
                highest_conf_comment = sentiment_data.loc[highest_conf_idx]
                
                comment_text = str(highest_conf_comment['text'])
                if len(comment_text) > 300:
                    comment_text = comment_text[:300] + "..."
                
                confidence = highest_conf_comment['confidence']
                
                story.append(Paragraph(f"<b>Confidence Score:</b> {confidence:.4f}", styles['normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Comment:</b> {comment_text}", styles['normal']))
                story.append(Spacer(1, 12))


def create_llm_summaries_section(story, styles, OUTPUT_BASE_DIR, script_dir):
    """Create LLM-generated summaries section"""
    story.append(PageBreak())
    
    # Add title with icon
    summaries_img = get_summaries_icon(script_dir)
    if summaries_img:
        try:
            title_para = Paragraph("7. AI-Generated Sentiment Summaries", styles['heading'])
            t = create_title_with_image_table(title_para, summaries_img)
            story.append(t)
        except Exception:
            story.append(Paragraph("7. AI-Generated Sentiment Summaries", styles['heading']))
    else:
        story.append(Paragraph("7. AI-Generated Sentiment Summaries", styles['heading']))
    
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
                story.append(Paragraph(f"{subsection_num} {sentiment.title()} Summary", styles['subheading']))
                story.append(Spacer(1, 6))
                
                summary_text = summary_data.get('summary', 'No summary available')
                llm_model = summary_data.get('model_used', 'Unknown')
                num_comments = summary_data.get('num_comments_analyzed', 'N/A')
                
                story.append(Paragraph(f"<b>Model Used:</b> {llm_model}", styles['normal']))
                story.append(Paragraph(f"<b>Comments Analyzed:</b> {num_comments}", styles['normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Summary:</b> {summary_text}", styles['normal']))
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"<i>Summary for {sentiment} not available</i>", styles['base']['Italic']))
                story.append(Spacer(1, 12))


def create_recommendations_section(story, styles, OUTPUT_BASE_DIR):
    """Create AI-generated recommendations section"""
    story.append(PageBreak())
    story.append(Paragraph("8. AI-Generated Recommendations", styles['heading']))
    story.append(Spacer(1, 12))
    
    recommendation_file = os.path.join(OUTPUT_BASE_DIR, 'recommendation', 'recommendation.json')
    if os.path.exists(recommendation_file):
        try:
            with open(recommendation_file, 'r', encoding='utf-8') as f:
                recommendation_data = json.load(f)
            
            recommendation_text = recommendation_data.get('recommendation', 'No recommendations available')
            llm_model = recommendation_data.get('model_used', 'Unknown')
            timestamp = recommendation_data.get('generated_timestamp', 'N/A')
            
            story.append(Paragraph("8.1 Actionable Improvement Suggestions", styles['subheading']))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Based on Analysis of Positive and Negative Feedback</b>", styles['normal']))
            story.append(Paragraph(f"<b>Model Used:</b> {llm_model}", styles['normal']))
            story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['normal']))
            story.append(Spacer(1, 10))
            
            # Split recommendation text into paragraphs
            for para in recommendation_text.split('\n\n'):
                if para.strip():
                    story.append(Paragraph(para, styles['normal']))
                    story.append(Spacer(1, 8))
            
        except Exception as e:
            story.append(Paragraph(f"<i>Recommendations not available: {str(e)}</i>", styles['base']['Italic']))
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("<i>No recommendations file found</i>", styles['base']['Italic']))
        story.append(Spacer(1, 12))


def create_insurance_risk_section(story, styles, OUTPUT_BASE_DIR):
    """Create bankruptcy insurance risk assessment section"""
    story.append(PageBreak())
    story.append(Paragraph("9. Bankruptcy Insurance Risk Assessment", styles['heading']))
    story.append(Spacer(1, 12))
    
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
            
            # Risk calculation formula
            story.append(Paragraph("9.1 Risk Calculation Formula", styles['subheading']))
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
            story.append(Paragraph(formula_text, styles['normal']))
            story.append(Spacer(1, 12))
            
            # Risk factors breakdown
            story.append(Paragraph("9.2 Risk Factors Breakdown", styles['subheading']))
            story.append(Spacer(1, 6))
            
            breakdown_table = create_insurance_breakdown_table(breakdown)
            story.append(breakdown_table)
            story.append(Spacer(1, 12))
            
            # Insurance cost estimate
            story.append(Paragraph("9.3 Insurance Cost Estimate", styles['subheading']))
            story.append(Spacer(1, 6))
            
            result_table = create_insurance_result_table(insurance_cost, risk_score, risk_level)
            story.append(result_table)
            story.append(Spacer(1, 12))
            
            # Interpretation
            sentiment_factors = breakdown.get('sentiment_factors', {})
            confidence_factors = breakdown.get('confidence_factors', {})
            trend_factors = breakdown.get('trend_factors', {})
            
            interpretation = f"""
            <b>Interpretation:</b><br/>
            Based on the sentiment analysis of customer reviews, the estimated bankruptcy insurance cost for this restaurant
            is <b>${insurance_cost:,.2f}</b> with a <b>{risk_level}</b> risk level (score: {risk_score}/100).<br/><br/>
            
            This assessment considers the overall sentiment distribution ({sentiment_factors.get('positive_percentage', 0)}% positive,
            {sentiment_factors.get('negative_percentage', 0)}% negative), the confidence of sentiment predictions
            (avg: {confidence_factors.get('average_confidence', 0):.1%}), and recent sentiment trends
            ({trend_factors.get('trend_status', 'stable').lower()}).
            """
            story.append(Paragraph(interpretation, styles['normal']))
            story.append(Spacer(1, 8))
            
        except Exception as e:
            story.append(Paragraph(f"<i>Insurance risk assessment not available: {str(e)}</i>", styles['base']['Italic']))
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("<i>Insurance risk assessment file not found</i>", styles['base']['Italic']))
        story.append(Spacer(1, 12))


def create_technical_details_section(story, styles, db_path, performance_summary):
    """Create technical details section"""
    story.append(PageBreak())
    story.append(Paragraph("10. Technical Details", styles['heading']))
    
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
    
    story.append(Paragraph(tech_details, styles['normal']))

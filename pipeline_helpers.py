#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for sentiment analysis pipeline
Breaks down the main analysis pipeline into smaller, focused functions
"""

import os
import shutil
import json
from pdf_generation import generate_pdf_fun as pdf_gen_func
import logging
import traceback
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def initialize_mlflow_tracking(
    job_id: str,
    search_method: str,
    company_name: str,
    custom_prompt: Optional[str],
    MLFLOW_AVAILABLE: bool,
    SentimentExperimentTracker,
    SENTENCE_LENGTH: int,
    SEPARATOR_KEYWORDS: str
) -> Tuple[object, datetime]:
    """
    Initialize MLflow experiment tracker if available
    
    Returns:
        Tuple of (tracker, start_time)
    """
    tracker = None
    start_time = datetime.now()
    
    if MLFLOW_AVAILABLE:
        tracker = SentimentExperimentTracker(experiment_name="sentiment-analysis-production")
        
        run_tags = {
            'job_id': job_id,
            'search_method': search_method,
            'company_name': company_name,
            'has_custom_prompt': 'yes' if custom_prompt else 'no'
        }
        tracker.start_run(run_name=f"analysis_{job_id[:8]}", tags=run_tags)
        
        tracker.log_parameters({
            'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
            'search_method': search_method,
            'sentence_length': SENTENCE_LENGTH,
            'separator_keywords': SEPARATOR_KEYWORDS,
            'llm_provider': 'groq',
            'has_custom_prompt': custom_prompt is not None,
            'company_name': company_name
        })
    
    return tracker, start_time


def setup_analysis_directories(job_id: str) -> Tuple[str, str, str]:
    """
    Create necessary directories for analysis
    
    Returns:
        Tuple of (cache_folder, output_base_dir, db_path)
    """
    cache_folder = f"cache/{job_id}"
    output_base_dir = f"my_volume/sentiment_analysis/{job_id}"
    db_path = f"filtered_reviews_{job_id}.db"
    
    os.makedirs(cache_folder, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)
    
    return cache_folder, output_base_dir, db_path


def prepare_html_content(
    job_id: str,
    search_method: str,
    search_input: Optional[str],
    url: Optional[str],
    html_path: Optional[str],
    cache_folder: str,
    process_search_method,
    download_page_fun
) -> None:
    """
    Process search method and prepare HTML content for analysis
    """
    logger.info(f"Job {job_id}: Search method: {search_method}, input: {search_input}")
    
    target_file = process_search_method(search_method, search_input)
    logger.info(f"Job {job_id}: Target file: {target_file}")
    
    # Copy target file to cache folder if it exists
    if os.path.exists(target_file):
        shutil.copy(target_file, cache_folder)
        logger.info(f"Job {job_id}: Copied {target_file} to cache")
    elif os.path.exists(f"cache/{target_file}"):
        shutil.copy(f"cache/{target_file}", cache_folder)
        logger.info(f"Job {job_id}: Copied from cache/{target_file}")
    
    # Download or use provided HTML
    if url and url != "demo":
        logger.info(f"Job {job_id}: Downloading from URL: {url}")
        download_page_fun(cache_folder, url)
    elif html_path:
        logger.info(f"Job {job_id}: Using provided HTML")
        shutil.copy(html_path, cache_folder)


def execute_sentiment_analysis(
    job_id: str,
    cache_folder: str,
    output_base_dir: str,
    db_path: str,
    base_config: dict,
    KEY_POSITIVE_WORDS: list,
    KEY_NEUTRAL_WORDS: list,
    KEY_NEGATIVE_WORDS: list,
    SENTENCE_LENGTH: int,
    SEPARATOR_KEYWORDS: str,
    extract_text_fun,
    Context_analyzer_RoBERTa_fun
) -> None:
    """
    Execute the core sentiment analysis steps
    """
    # Extract text
    logger.info(f"Job {job_id}: Extracting text")
    extract_text_fun(SEPARATOR_KEYWORDS, cache_folder)
    
    # Configure analysis
    config = base_config.copy()
    config['key_positive_words'] = KEY_POSITIVE_WORDS
    config['key_neutral_words'] = KEY_NEUTRAL_WORDS
    config['key_negative_words'] = KEY_NEGATIVE_WORDS
    config['sentence_length'] = SENTENCE_LENGTH
    config['output_base_dir'] = output_base_dir
    config['path_db'] = db_path
    
    # Run sentiment analysis
    logger.info(f"Job {job_id}: Running sentiment analysis")
    Context_analyzer_RoBERTa_fun(**config)


def generate_ai_summaries(
    job_id: str,
    output_base_dir: str,
    custom_prompt: Optional[str],
    GROQ_API_KEY: str,
    DEFAULT_PROMPT: str,
    summarize_sentiments_fun,
    recommendation_fun
) -> None:
    """
    Generate AI summaries and recommendations
    """
    # Generate summaries
    logger.info(f"Job {job_id}: Generating summaries")
    summarize_sentiments_fun(output_base_dir, GROQ_API_KEY, llm_method="groq")
    
    # Generate recommendations
    logger.info(f"Job {job_id}: Generating recommendations")
    prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT
    recommendation_fun(prompt, output_base_dir, GROQ_API_KEY, llm_method="groq")


def calculate_and_save_insurance_risk(
    job_id: str,
    output_base_dir: str,
    calculate_insurance_risk
) -> None:
    """
    Calculate insurance risk assessment and save results
    """
    logger.info(f"Job {job_id}: Calculating insurance risk assessment")
    try:
        perf_summary_path = f"{output_base_dir}/performance_summary.json"
        sentiment_trends_path = f"{output_base_dir}/sentiment_trends.json"
        
        if os.path.exists(perf_summary_path) and os.path.exists(sentiment_trends_path):
            with open(perf_summary_path, 'r') as f:
                performance_data = json.load(f)
            with open(sentiment_trends_path, 'r') as f:
                trends_data = json.load(f)
            
            insurance_result = calculate_insurance_risk(
                performance_summary=performance_data,
                sentiment_trends=trends_data,
                base_rate=5000.0
            )
            
            insurance_path = f"{output_base_dir}/insurance_risk.json"
            with open(insurance_path, 'w') as f:
                json.dump(insurance_result, f, indent=2)
            
            logger.info(f"Job {job_id}: Insurance assessment complete - Risk Level: {insurance_result['risk_level']}, Cost: ${insurance_result['insurance_cost']:,.2f}")
        else:
            logger.warning(f"Job {job_id}: Could not calculate insurance risk - missing data files")
    except Exception as e:
        logger.error(f"Job {job_id}: Error calculating insurance risk: {e}")
        logger.error(f"Job {job_id}: Traceback: {traceback.format_exc()}")


def generate_and_copy_pdf(
    job_id: str,
    db_path: str,
    output_base_dir: str,
    company_name: str,
    generate_pdf_fun=None
) -> None:
    """
    Generate PDF report and copy to visualizations folder
    """
    logger.info(f"Job {job_id}: Generating PDF")
    
    # Use provided function or default to pdf_generation module
    pdf_func = generate_pdf_fun if generate_pdf_fun is not None else pdf_gen_func
    pdf_func(db_path, output_base_dir, company_name=company_name)
    
    # Copy PDF to root visualizations folder for frontend access
    source_pdf = os.path.join(output_base_dir, 'visualizations', 'sentiment_analysis_report.pdf')
    dest_pdf = os.path.join('my_volume', 'sentiment_analysis', 'visualizations', 'sentiment_analysis_report.pdf')
    
    os.makedirs(os.path.dirname(dest_pdf), exist_ok=True)
    
    if os.path.exists(source_pdf):
        shutil.copy2(source_pdf, dest_pdf)
        logger.info(f"Job {job_id}: Copied PDF to {dest_pdf}")
    else:
        logger.warning(f"Job {job_id}: Source PDF not found at {source_pdf}")


def finalize_job_success(
    job_id: str,
    jobs_db: dict,
    emails: Optional[list],
    output_base_dir: str,
    company_name: str,
    SMTP_CONFIG: dict,
    send_email,
    send_report_email_fun,
    tracker,
    search_method: str,
    start_time: datetime,
    MLFLOW_AVAILABLE: bool,
    mlflow_logger
) -> None:
    """
    Finalize successful job: send emails, log to MLflow, update status
    """
    # Send email if provided
    if emails:
        send_email(job_id, emails, output_base_dir, company_name, jobs_db, SMTP_CONFIG, send_report_email_fun)
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE and tracker:
        mlflow_logger(job_id, output_base_dir, search_method, start_time, tracker)
        tracker.end_run(status="FINISHED")
    
    # Update job status
    jobs_db[job_id]['status'] = 'completed'
    jobs_db[job_id]['progress'] = 100
    jobs_db[job_id]['message'] = 'Analysis complete!'
    jobs_db[job_id]['pdf_url'] = f"/api/results/{job_id}/pdf"
    jobs_db[job_id]['results_url'] = f"/api/results/{job_id}/data"
    
    logger.info(f"Job {job_id}: Completed successfully")


def handle_job_failure(
    job_id: str,
    jobs_db: dict,
    error: Exception,
    tracker,
    MLFLOW_AVAILABLE: bool
) -> None:
    """
    Handle job failure: log error, update status
    """
    logger.error(f"Job {job_id}: Error - {str(error)}")
    
    # Log error to MLflow if available
    if MLFLOW_AVAILABLE and tracker:
        tracker.log_error(error)
        tracker.end_run(status="FAILED")
    
    jobs_db[job_id]['status'] = 'failed'
    jobs_db[job_id]['error'] = str(error)
    jobs_db[job_id]['message'] = f'Error: {str(error)}'

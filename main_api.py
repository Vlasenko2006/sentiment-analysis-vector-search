#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI wrapper for sentiment analysis pipeline
Provides REST API endpoints for containerized deployment
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import os
from datetime import datetime
import logging
import uvicorn

# Import your existing functions
from extract_text_fun import extract_text_fun
from Context_analyzer_RoBERTa_fun import Context_analyzer_RoBERTa_fun
from summarize_sentiments_fun import summarize_sentiments_fun
from recommendation_fun import recommendation_fun
from pdf_generation import generate_pdf_fun
from download_page_fun import download_page_fun
from search_methods_fun import process_search_method
from send_report_email_fun import send_report_email_fun
from send_email import send_email
from chatbot_analyzer import ResultsChatbot
from insurance_calculator import calculate_insurance_risk
from routes import Routes
from cleanup_old_jobs import cleanup_old_jobs
from pipeline_helpers import (
    initialize_mlflow_tracking,
    setup_analysis_directories,
    prepare_html_content,
    execute_sentiment_analysis,
    generate_ai_summaries,
    calculate_and_save_insurance_risk,
    generate_and_copy_pdf,
    finalize_job_success,
    handle_job_failure
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional MLflow imports - handle gracefully if not available
try:
    from mlflow_tracking import SentimentExperimentTracker
    from mlflow_logger import mlflow_logger
    MLFLOW_AVAILABLE = True
    logger.info("MLflow tracking enabled")
except ImportError as e:
    MLFLOW_AVAILABLE = False
    logger.warning(f"MLflow not available, tracking disabled: {e}")
    SentimentExperimentTracker = None
    mlflow_logger = None

app = FastAPI(
    title="Sentiment Analysis API",
    description="AI-powered restaurant review sentiment analysis",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage (use Redis in production)
jobs_db: Dict[str, dict] = {}

# Load all configurations
from config.config import load_all_configs
config_data = load_all_configs()

base_config = config_data['base_config']
key_config = config_data['key_config']
names_config = config_data['names_config']
SEPARATOR_KEYWORDS = config_data['SEPARATOR_KEYWORDS']
KEY_POSITIVE_WORDS = config_data['KEY_POSITIVE_WORDS']
KEY_NEUTRAL_WORDS = config_data['KEY_NEUTRAL_WORDS']
KEY_NEGATIVE_WORDS = config_data['KEY_NEGATIVE_WORDS']
SENTENCE_LENGTH = config_data['SENTENCE_LENGTH']
DEFAULT_PROMPT = config_data['DEFAULT_PROMPT']
GROQ_API_KEY = config_data['GROQ_API_KEY']
SMTP_CONFIG = config_data['SMTP_CONFIG']


def run_analysis_pipeline(
    job_id: str, 
    url: Optional[str] = None, 
    html_path: Optional[str] = None,
    emails: Optional[list] = None,
    custom_prompt: Optional[str] = None,
    search_method: str = "demo",
    search_input: Optional[str] = None,
    company_name: str = "Awesome Company"
):
    """
    Run the complete sentiment analysis pipeline
    
    Args:
        job_id: Unique job identifier
        url: URL to download (deprecated, use search_method)
        html_path: Path to pre-downloaded HTML
        emails: List of customer emails for report delivery (or single email string)
        custom_prompt: Custom LLM prompt for recommendations
        search_method: One of 'keywords', 'urls', or 'demo'
        search_input: User input for search (keywords or URLs)
        company_name: Company name for report branding
    """
    tracker = None
    try:
        # Initialize MLflow tracking
        tracker, start_time = initialize_mlflow_tracking(
            job_id, search_method, company_name, custom_prompt,
            MLFLOW_AVAILABLE, SentimentExperimentTracker,
            SENTENCE_LENGTH, SEPARATOR_KEYWORDS
        )
        
        jobs_db[job_id]['status'] = 'running'
        jobs_db[job_id]['progress'] = 10
        jobs_db[job_id]['message'] = 'Initializing...'
        
        # Cleanup old job folders
        cleanup_old_jobs(logger, max_age_days=7)
        
        # Setup directories
        cache_folder, output_base_dir, db_path = setup_analysis_directories(job_id)
        
        # Prepare HTML content
        jobs_db[job_id]['progress'] = 15
        jobs_db[job_id]['message'] = 'Processing search method...'
        prepare_html_content(
            job_id, search_method, search_input, url, html_path,
            cache_folder, process_search_method, download_page_fun
        )
        
        # Extract text and run sentiment analysis
        jobs_db[job_id]['progress'] = 30
        jobs_db[job_id]['message'] = 'Extracting reviews...'
        execute_sentiment_analysis(
            job_id, cache_folder, output_base_dir, db_path,
            base_config, KEY_POSITIVE_WORDS, KEY_NEUTRAL_WORDS,
            KEY_NEGATIVE_WORDS, SENTENCE_LENGTH, SEPARATOR_KEYWORDS,
            extract_text_fun, Context_analyzer_RoBERTa_fun
        )
        
        # Generate AI summaries and recommendations
        jobs_db[job_id]['progress'] = 60
        jobs_db[job_id]['message'] = 'Generating AI summaries...'
        generate_ai_summaries(
            job_id, output_base_dir, custom_prompt,
            GROQ_API_KEY, DEFAULT_PROMPT,
            summarize_sentiments_fun, recommendation_fun
        )
        
        # Calculate insurance risk
        jobs_db[job_id]['progress'] = 85
        jobs_db[job_id]['message'] = 'Calculating insurance risk...'
        calculate_and_save_insurance_risk(
            job_id, output_base_dir, calculate_insurance_risk
        )
        
        # Generate PDF report
        jobs_db[job_id]['progress'] = 90
        jobs_db[job_id]['message'] = 'Generating PDF report...'
        generate_and_copy_pdf(
            job_id, db_path, output_base_dir, company_name, generate_pdf_fun
        )
        
        # Finalize: send emails, log to MLflow, update status
        finalize_job_success(
            job_id, jobs_db, emails, output_base_dir, company_name,
            SMTP_CONFIG, send_email, send_report_email_fun,
            tracker, search_method, start_time,
            MLFLOW_AVAILABLE, mlflow_logger
        )
        
    except Exception as e:
        handle_job_failure(job_id, jobs_db, e, tracker, MLFLOW_AVAILABLE)


# Chatbot storage (stores chatbot instances per job)
chatbots: Dict[str, ResultsChatbot] = {}

# Initialize routes and register with app
routes_handler = Routes(
    jobs_db=jobs_db,
    chatbots=chatbots,
    run_analysis_pipeline=run_analysis_pipeline,
    names_config=names_config,
    base_config=base_config,
    key_config=key_config
)
app.include_router(routes_handler.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

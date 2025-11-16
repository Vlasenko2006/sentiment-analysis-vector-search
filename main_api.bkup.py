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
import yaml
import shutil
import json
from datetime import datetime
import logging
import traceback
import uvicorn

# Import your existing functions
from extract_text_fun import extract_text_fun
from Context_analyzer_RoBERTa_fun import Context_analyzer_RoBERTa_fun
from summarize_sentiments_fun import summarize_sentiments_fun
from recommendation_fun import recommendation_fun
from generate_pdf_fun import generate_pdf_fun
from download_page_fun import download_page_fun
from search_methods_fun import process_search_method
from send_report_email_fun import send_report_email_fun
from send_email import send_email
from chatbot_analyzer import ResultsChatbot
from insurance_calculator import calculate_insurance_risk
from routes import Routes
from cleanup_old_jobs import cleanup_old_jobs
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

# Load config
with open('config.yaml', 'r', encoding='utf-8') as f:
    base_config = yaml.safe_load(f)

# Load sensitive config from config_key.yaml
try:
    with open('config_key.yaml', 'r', encoding='utf-8') as f:
        key_config = yaml.safe_load(f)
except FileNotFoundError:
    logger.warning("config_key.yaml not found, using environment variables")
    key_config = {}

# Load company names config from config_names.yaml
try:
    with open('config_names.yaml', 'r', encoding='utf-8') as f:
        names_config = yaml.safe_load(f)
except FileNotFoundError:
    logger.warning("config_names.yaml not found, using default company name")
    names_config = {}

# Constants
SEPARATOR_KEYWORDS = key_config.get('analysis', {}).get('separator_keywords')

if not SEPARATOR_KEYWORDS:
    raise ValueError("separator_keywords must be set in config_key.yaml")

# Load analysis parameters from config_key.yaml
KEY_POSITIVE_WORDS = key_config.get('analysis', {}).get('key_positive_words')
KEY_NEUTRAL_WORDS = key_config.get('analysis', {}).get('key_neutral_words')
KEY_NEGATIVE_WORDS = key_config.get('analysis', {}).get('key_negative_words')
SENTENCE_LENGTH = key_config.get('analysis', {}).get('sentence_length')
DEFAULT_PROMPT = key_config.get('analysis', {}).get('default_prompt')

if not all([KEY_POSITIVE_WORDS, KEY_NEUTRAL_WORDS, KEY_NEGATIVE_WORDS, SENTENCE_LENGTH, DEFAULT_PROMPT]):
    raise ValueError("Analysis parameters (key words, sentence_length, default_prompt) must be set in config_key.yaml")

# Load Groq API key from config_key.yaml
GROQ_API_KEY = key_config.get('groq', {}).get('api_key')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in config_key.yaml")

# Load email configuration from config_key.yaml
email_config = key_config.get('email', {})
SMTP_SERVER = email_config.get('smtp_server')
SMTP_PORT = email_config.get('smtp_port')
SENDER_EMAIL = email_config.get('sender_email')
SENDER_PASSWORD = email_config.get('sender_password')

if not all([SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD]):
    raise ValueError("Email configuration (smtp_server, smtp_port, sender_email, sender_password) must be set in config_key.yaml")

# SMTP configuration dictionary for email sending
SMTP_CONFIG = {
    'server': SMTP_SERVER,
    'port': SMTP_PORT,
    'email': SENDER_EMAIL,
    'password': SENDER_PASSWORD
}





def run_analysis_pipeline(
    job_id: str, 
    url: Optional[str] = None, 
    html_path: Optional[str] = None,
    emails: Optional[list] = None,  # Changed to support multiple emails
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
    try:
        # Initialize MLflow experiment tracker (if available)
        tracker = None
        start_time = datetime.now()
        
        if MLFLOW_AVAILABLE:
            tracker = SentimentExperimentTracker(experiment_name="sentiment-analysis-production")
            
            # Start MLflow run
            run_tags = {
                'job_id': job_id,
                'search_method': search_method,
                'company_name': company_name,
                'has_custom_prompt': 'yes' if custom_prompt else 'no'
            }
            tracker.start_run(run_name=f"analysis_{job_id[:8]}", tags=run_tags)
            
            # Log parameters
            tracker.log_parameters({
                'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
                'search_method': search_method,
                'sentence_length': SENTENCE_LENGTH,
                'separator_keywords': SEPARATOR_KEYWORDS,
                'llm_provider': 'groq',
                'has_custom_prompt': custom_prompt is not None,
                'company_name': company_name
            })
        
        jobs_db[job_id]['status'] = 'running'
        jobs_db[job_id]['progress'] = 10
        jobs_db[job_id]['message'] = 'Initializing...'
        
        # Cleanup old job folders (older than 7 days)
        cleanup_old_jobs(logger, max_age_days=7)
        
        # Setup directories
        cache_folder = f"cache/{job_id}"
        output_base_dir = f"my_volume/sentiment_analysis/{job_id}"
        db_path = f"filtered_reviews_{job_id}.db"
        
        os.makedirs(cache_folder, exist_ok=True)
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Determine TARGET_URL based on search method
        jobs_db[job_id]['progress'] = 15
        jobs_db[job_id]['message'] = 'Processing search method...'
        logger.info(f"Job {job_id}: Search method: {search_method}, input: {search_input}")
        
        target_file = process_search_method(search_method, search_input)
        logger.info(f"Job {job_id}: Target file: {target_file}")
        
        # Copy target file to cache folder if it exists in current directory
        if os.path.exists(target_file):
            shutil.copy(target_file, cache_folder)
            logger.info(f"Job {job_id}: Copied {target_file} to cache")
        elif os.path.exists(f"cache/{target_file}"):
            shutil.copy(f"cache/{target_file}", cache_folder)
            logger.info(f"Job {job_id}: Copied from cache/{target_file}")
        
        # Download or use provided HTML
        jobs_db[job_id]['progress'] = 20
        jobs_db[job_id]['message'] = 'Downloading page...'
        
        if url and url != "demo":
            logger.info(f"Job {job_id}: Downloading from URL: {url}")
            download_page_fun(cache_folder, url)
        elif html_path:
            logger.info(f"Job {job_id}: Using provided HTML")
            shutil.copy(html_path, cache_folder)
        
        # Extract text
        jobs_db[job_id]['progress'] = 30
        jobs_db[job_id]['message'] = 'Extracting reviews...'
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
        jobs_db[job_id]['progress'] = 40
        jobs_db[job_id]['message'] = 'Analyzing sentiment...'
        logger.info(f"Job {job_id}: Running sentiment analysis")
        Context_analyzer_RoBERTa_fun(**config)
        
        # Generate summaries
        jobs_db[job_id]['progress'] = 60
        jobs_db[job_id]['message'] = 'Generating AI summaries...'
        logger.info(f"Job {job_id}: Generating summaries")
        summarize_sentiments_fun(output_base_dir, GROQ_API_KEY, llm_method="groq")
        
        # Generate recommendations
        jobs_db[job_id]['progress'] = 75
        jobs_db[job_id]['message'] = 'Creating recommendations...'
        logger.info(f"Job {job_id}: Generating recommendations")
        prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT
        recommendation_fun(prompt, output_base_dir, GROQ_API_KEY, llm_method="groq")
        
        # Calculate insurance risk assessment
        jobs_db[job_id]['progress'] = 85
        jobs_db[job_id]['message'] = 'Calculating insurance risk...'
        logger.info(f"Job {job_id}: Calculating insurance risk assessment")
        try:
            # Load performance summary and sentiment trends
            perf_summary_path = f"{output_base_dir}/performance_summary.json"
            sentiment_trends_path = f"{output_base_dir}/sentiment_trends.json"
            
            if os.path.exists(perf_summary_path) and os.path.exists(sentiment_trends_path):
                with open(perf_summary_path, 'r') as f:
                    performance_data = json.load(f)
                with open(sentiment_trends_path, 'r') as f:
                    trends_data = json.load(f)
                
                # Calculate insurance risk
                insurance_result = calculate_insurance_risk(
                    performance_summary=performance_data,
                    sentiment_trends=trends_data,
                    base_rate=5000.0  # Base insurance cost in USD
                )
                
                # Save insurance risk assessment
                insurance_path = f"{output_base_dir}/insurance_risk.json"
                with open(insurance_path, 'w') as f:
                    json.dump(insurance_result, f, indent=2)
                
                logger.info(f"Job {job_id}: Insurance assessment complete - Risk Level: {insurance_result['risk_level']}, Cost: ${insurance_result['insurance_cost']:,.2f}")
            else:
                logger.warning(f"Job {job_id}: Could not calculate insurance risk - missing data files")
        except Exception as e:
            logger.error(f"Job {job_id}: Error calculating insurance risk: {e}")
            logger.error(f"Job {job_id}: Traceback: {traceback.format_exc()}")
        
        # Generate PDF
        jobs_db[job_id]['progress'] = 90
        jobs_db[job_id]['message'] = 'Generating PDF report...'
        logger.info(f"Job {job_id}: Generating PDF")
        generate_pdf_fun(db_path, output_base_dir, company_name=company_name)
        
        # Copy PDF to root visualizations folder for frontend access
        source_pdf = os.path.join(output_base_dir, 'visualizations', 'sentiment_analysis_report.pdf')
        dest_pdf = os.path.join('my_volume', 'sentiment_analysis', 'visualizations', 'sentiment_analysis_report.pdf')
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_pdf), exist_ok=True)
        
        if os.path.exists(source_pdf):
            shutil.copy2(source_pdf, dest_pdf)
            logger.info(f"Job {job_id}: Copied PDF to {dest_pdf}")
        else:
            logger.warning(f"Job {job_id}: Source PDF not found at {source_pdf}")
        
        # Send email if email address(es) provided
        if emails:
            send_email(job_id, emails, output_base_dir, company_name, jobs_db, SMTP_CONFIG, send_report_email_fun)
        
        # Log metrics and artifacts to MLflow (if available)
        if MLFLOW_AVAILABLE and tracker:
            mlflow_logger(job_id, output_base_dir, search_method, start_time, tracker)
            # End MLflow run successfully
            tracker.end_run(status="FINISHED")
        
        # Complete
        jobs_db[job_id]['status'] = 'completed'
        jobs_db[job_id]['progress'] = 100
        jobs_db[job_id]['message'] = 'Analysis complete!'
        jobs_db[job_id]['pdf_url'] = f"/api/results/{job_id}/pdf"
        jobs_db[job_id]['results_url'] = f"/api/results/{job_id}/data"
        
        logger.info(f"Job {job_id}: Completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id}: Error - {str(e)}")
        # Log error to MLflow (if available)
        if MLFLOW_AVAILABLE and 'tracker' in locals() and tracker:
            tracker.log_error(e)
            tracker.end_run(status="FAILED")

        jobs_db[job_id]['status'] = 'failed'
        jobs_db[job_id]['error'] = str(e)
        jobs_db[job_id]['message'] = f'Error: {str(e)}'


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


# debugging wrapper for chatbot endpoints
@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    """Log all chatbot-related requests"""
    if "/chat" in str(request.url):
        debug_dir = "debuglogs"
        os.makedirs(debug_dir, exist_ok=True)
        
        log_file = os.path.join(debug_dir, f"python-api-{datetime.now().strftime('%Y-%m-%d')}.log")
        
        # Log request
        log_entry = f"\n{'='*80}\n"
        log_entry += f"[{datetime.now().isoformat()}] REQUEST: {request.method} {request.url}\n"
        log_entry += f"Headers: {dict(request.headers)}\n"
        
        if request.method == "POST":
            body = await request.body()
            log_entry += f"Body: {body.decode()}\n"
            
            # Recreate request with body
            async def receive():
                return {"type": "http.request", "body": body}
            request = Request(request.scope, receive)
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
        
        # Process request
        response = await call_next(request)
        
        # Log response
        log_entry = f"RESPONSE: Status {response.status_code}\n"
        log_entry += f"{'='*80}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
        
        return response
    
    return await call_next(request)

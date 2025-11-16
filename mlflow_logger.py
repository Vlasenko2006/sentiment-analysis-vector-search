#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLflow logging utilities for sentiment analysis pipeline
"""

import os
import json
import logging
import traceback
from datetime import datetime
from mlflow_tracking import SentimentExperimentTracker

logger = logging.getLogger(__name__)


def mlflow_logger(job_id: str, output_base_dir: str, search_method: str, start_time: datetime, tracker: SentimentExperimentTracker):
    """
    Log metrics and artifacts to MLflow
    
    Args:
        job_id: Unique job identifier
        output_base_dir: Base directory for analysis outputs
        search_method: Search method used
        start_time: Analysis start time
        tracker: MLflow experiment tracker instance
    """
    try:
        # Log sentiment distribution to MLflow
        if os.path.exists(f"{output_base_dir}/performance_summary.json"):
            with open(f"{output_base_dir}/performance_summary.json", 'r') as f:
                perf_data = json.load(f)
                sentiment_dist = perf_data.get('sentiment_distribution', {})
                
                tracker.log_sentiment_distribution(
                    positive=sentiment_dist.get('POSITIVE', 0),
                    negative=sentiment_dist.get('NEGATIVE', 0),
                    neutral=sentiment_dist.get('NEUTRAL', 0)
                )
                
                # Log dataset info
                tracker.log_dataset_info(
                    num_reviews=perf_data.get('total_samples', 0),
                    source=search_method
                )
        
        # Log processing time
        end_time = datetime.now()
        tracker.log_processing_time(start_time, end_time)
        
        # Log API usage (estimated)
        tracker.log_groq_usage(num_api_calls=3)  # summary + recommendation + chatbot prep
        
        # Log artifacts
        pdf_path = f"{output_base_dir}/visualizations/sentiment_analysis_report.pdf"
        if os.path.exists(pdf_path):
            tracker.log_artifact(pdf_path, "reports")
        
        # Log JSON results
        for sentiment in ['positive', 'negative', 'neutral']:
            summary_path = f"{output_base_dir}/{sentiment}/{sentiment}_summary.json"
            if os.path.exists(summary_path):
                tracker.log_artifact(summary_path, "summaries")
        
        # Log recommendation
        rec_path = f"{output_base_dir}/recommendation/recommendation.json"
        if os.path.exists(rec_path):
            tracker.log_artifact(rec_path, "recommendations")
        
        logger.info(f"Job {job_id}: MLflow logging completed")
        
    except Exception as e:
        logger.error(f"Job {job_id}: Error logging to MLflow: {e}")
        logger.error(f"Job {job_id}: MLflow traceback: {traceback.format_exc()}")

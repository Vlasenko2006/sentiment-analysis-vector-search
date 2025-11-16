"""
MLflow Experiment Tracking Module
Tracks ML experiments, parameters, metrics, and artifacts
"""

import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import json
import os
from datetime import datetime

class SentimentExperimentTracker:
    """
    MLflow tracker for sentiment analysis experiments
    
    Tracks:
    - Model parameters (model_name, thresholds, etc.)
    - Metrics (review counts, sentiment distribution, processing time)
    - Artifacts (PDFs, JSON results, plots)
    """
    
    def __init__(self, experiment_name: str = "sentiment-analysis"):
        """Initialize MLflow experiment tracker"""
        # Set MLflow tracking URI (can be remote server in production)
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
        except Exception as e:
            print(f"Error setting up MLflow: {e}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """Start a new MLflow run"""
        run_tags = tags or {}
        run_tags['timestamp'] = datetime.now().isoformat()
        
        self.active_run = mlflow.start_run(run_name=run_name, tags=run_tags)
        return self.active_run
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log model parameters"""
        for key, value in params.items():
            if value is not None:
                mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics (review counts, sentiment scores, etc.)"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log artifact (PDF, JSON, plot)"""
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path)
    
    def log_dict_as_json(self, data: Dict, filename: str):
        """Log dictionary as JSON artifact"""
        temp_file = f"/tmp/{filename}"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(temp_file)
        os.remove(temp_file)
    
    def log_model_info(self, model_name: str, model_version: str = "latest"):
        """Log model information"""
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
    
    def log_dataset_info(self, num_reviews: int, source: str = "unknown"):
        """Log dataset information"""
        mlflow.log_param("num_reviews", num_reviews)
        mlflow.log_param("data_source", source)
    
    def log_processing_time(self, start_time: datetime, end_time: datetime):
        """Log processing time"""
        duration = (end_time - start_time).total_seconds()
        mlflow.log_metric("processing_time_seconds", duration)
        mlflow.log_metric("processing_time_minutes", duration / 60)
    
    def log_sentiment_distribution(self, positive: int, negative: int, neutral: int):
        """Log sentiment distribution"""
        total = positive + negative + neutral
        if total > 0:
            mlflow.log_metric("total_reviews", total)
            mlflow.log_metric("positive_count", positive)
            mlflow.log_metric("negative_count", negative)
            mlflow.log_metric("neutral_count", neutral)
            mlflow.log_metric("positive_ratio", positive / total)
            mlflow.log_metric("negative_ratio", negative / total)
            mlflow.log_metric("neutral_ratio", neutral / total)
    
    def log_groq_usage(self, num_api_calls: int, estimated_tokens: int = 0):
        """Log LLM API usage"""
        mlflow.log_metric("groq_api_calls", num_api_calls)
        if estimated_tokens > 0:
            mlflow.log_metric("estimated_tokens", estimated_tokens)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the run"""
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def end_run(self, status: str = "FINISHED"):
        """End the current run"""
        mlflow.set_tag("status", status)
        mlflow.end_run()
    
    def log_error(self, error: Exception):
        """Log error information"""
        mlflow.set_tag("status", "FAILED")
        mlflow.log_param("error_type", type(error).__name__)
        mlflow.log_param("error_message", str(error))


# Convenience function for quick tracking
def track_sentiment_run(
    job_id: str,
    params: Dict,
    metrics: Dict,
    artifacts: Dict[str, str],
    tags: Optional[Dict] = None
):
    """
    Quick function to track a complete sentiment analysis run
    
    Args:
        job_id: Unique job identifier
        params: Dictionary of parameters (model, thresholds, etc.)
        metrics: Dictionary of metrics (counts, ratios, timing)
        artifacts: Dictionary of artifact paths {name: path}
        tags: Optional tags for categorization
    """
    tracker = SentimentExperimentTracker()
    
    run_tags = tags or {}
    run_tags['job_id'] = job_id
    
    with tracker.start_run(run_name=f"job_{job_id}", tags=run_tags):
        tracker.log_parameters(params)
        tracker.log_metrics(metrics)
        
        for artifact_name, artifact_path in artifacts.items():
            if os.path.exists(artifact_path):
                tracker.log_artifact(artifact_path)
        
        tracker.end_run("FINISHED")

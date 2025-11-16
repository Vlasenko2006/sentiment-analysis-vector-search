from fastapi import APIRouter, BackgroundTasks, HTTPException, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime
from typing import Optional, Dict, Callable
import os
import json
import uuid
import logging

from models import AnalysisRequest, JobStatus, ChatRequest, ChatResponse
from chatbot_analyzer import ResultsChatbot

logger = logging.getLogger(__name__)


class Routes:
    """
    Routes class to encapsulate all API endpoints with access to shared state
    """
    def __init__(
        self,
        jobs_db: Dict,
        chatbots: Dict,
        run_analysis_pipeline: Callable,
        names_config: dict,
        base_config: dict,
        key_config: dict
    ):
        self.jobs_db = jobs_db
        self.chatbots = chatbots
        self.run_analysis_pipeline = run_analysis_pipeline
        self.names_config = names_config
        self.base_config = base_config
        self.key_config = key_config
        self.router = APIRouter()
        self._setup_routes()
    
    def get_company_name(self, company_name_param: Optional[str] = None) -> str:
        """Helper to resolve company name from various sources"""
        return (
            company_name_param
            or self.names_config.get('company_name')
            or self.base_config.get('company_name')
            or self.key_config.get('company_name')
            or os.getenv('COMPANY_NAME')
            or 'Awesome Company'
        )
    
    def _setup_routes(self):
        """Register all route handlers"""
        
        @self.router.get("/")
        async def root():
            """Root endpoint with API documentation"""
            return {
                "message": "Sentiment Analysis API",
                "version": "1.0",
                "endpoints": {
                    "analyze": "/api/analyze",
                    "status": "/api/status/{job_id}",
                    "results": "/api/results/{job_id}",
                    "health": "/health"
                }
            }
        
        @self.router.get("/health")
        async def health_check():
            """Health check endpoint for Docker"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.router.get("/api/config")
        async def get_config():
            """Get frontend configuration including company name"""
            company_name = self.get_company_name()
            return {
                "company_name": company_name
            }
        
        @self.router.post("/api/analyze", response_model=JobStatus)
        async def analyze_reviews(
            request: AnalysisRequest,
            background_tasks: BackgroundTasks,
            company_name: Optional[str] = None
        ):
            """
            Start sentiment analysis job
            
            Accepts:
            - url: TripAdvisor/Yelp URL (optional, deprecated)
            - html_content: Raw HTML content (optional)
            - email: Single email or comma-separated emails for report delivery (optional)
            - emails: List of email addresses (alternative to email field)
            - customPrompt: Custom LLM prompt (optional)
            - searchMethod: 'keywords', 'urls', or 'demo' (default: 'demo')
            """
            # Resolve company name
            company_name = self.get_company_name(company_name)

            job_id = str(uuid.uuid4())
            
            # Initialize job
            self.jobs_db[job_id] = {
                'job_id': job_id,
                'status': 'queued',
                'progress': 0,
                'message': 'Job queued',
                'created_at': datetime.now().isoformat()
            }
            
            # Handle HTML content
            html_path = None
            if request.html_content:
                cache_folder = f"cache/{job_id}"
                os.makedirs(cache_folder, exist_ok=True)
                html_path = f"{cache_folder}/reviews.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(request.html_content)
            
            # Process email addresses - support both formats
            email_input = None
            if request.emails:
                # Use the emails list if provided
                email_input = request.emails
            elif request.email:
                # Pass email string as-is (can be single or comma-separated)
                email_input = request.email
            
            # Determine search_input based on search method and URL
            search_input = None
            if request.url and request.url != "demo":
                search_input = request.url
            
            # Start analysis in background with all parameters
            background_tasks.add_task(
                self.run_analysis_pipeline,
                job_id,
                str(request.url) if request.url else None,
                html_path,
                email_input,  # Pass email(s) - can be string or list
                request.customPrompt,
                request.searchMethod or "demo",
                search_input,
                company_name,
            )
            
            return JobStatus(
                job_id=job_id,
                status='queued',
                progress=0,
                message='Analysis job queued'
            )
        
        @self.router.get("/api/status/{job_id}", response_model=JobStatus)
        async def get_job_status(job_id: str):
            """Get status of analysis job"""
            if job_id not in self.jobs_db:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs_db[job_id]
            return JobStatus(**job)
        
        @self.router.get("/api/results/{job_id}/pdf")
        async def get_pdf_report(job_id: str):
            """Download PDF report"""
            if job_id not in self.jobs_db:
                raise HTTPException(status_code=404, detail="Job not found")
            
            if self.jobs_db[job_id]['status'] != 'completed':
                raise HTTPException(status_code=400, detail="Analysis not completed")
            
            pdf_path = f"my_volume/sentiment_analysis/{job_id}/visualizations/sentiment_analysis_report.pdf"
            
            if not os.path.exists(pdf_path):
                raise HTTPException(status_code=404, detail="PDF not found")
            
            return FileResponse(
                pdf_path,
                media_type='application/pdf',
                filename=f'sentiment_report_{job_id}.pdf'
            )
        
        @self.router.get("/api/results/{job_id}/data")
        async def get_results_data(job_id: str):
            """Get analysis results as JSON"""
            output_dir = f"my_volume/sentiment_analysis/{job_id}"
            
            # Check if results exist on disk (even if not in memory after restart)
            if not os.path.exists(output_dir):
                raise HTTPException(status_code=404, detail="Job not found")
            
            # If in memory, check status
            if job_id in self.jobs_db and self.jobs_db[job_id]['status'] != 'completed':
                raise HTTPException(status_code=400, detail="Analysis not completed")
            
            # Load results
            results = {}
            
            # Load trends
            trends_path = f"{output_dir}/sentiment_trends.json"
            if os.path.exists(trends_path):
                with open(trends_path, 'r') as f:
                    results['trends'] = json.load(f)
            
            # Load summaries
            for sentiment in ['positive', 'negative', 'neutral']:
                summary_path = f"{output_dir}/{sentiment}/{sentiment}_summary.json"
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        results[f'{sentiment}_summary'] = json.load(f)
            
            # Load recommendations
            rec_path = f"{output_dir}/recommendation/recommendation.json"
            if os.path.exists(rec_path):
                with open(rec_path, 'r') as f:
                    results['recommendations'] = json.load(f)
            
            # Load statistics from performance_summary.json
            perf_path = f"{output_dir}/performance_summary.json"
            if os.path.exists(perf_path):
                with open(perf_path, 'r') as f:
                    perf_data = json.load(f)
                    # Extract statistics in the format expected by frontend
                    sentiment_dist = perf_data.get('sentiment_distribution', {})
                    results['statistics'] = {
                        'total_reviews': perf_data.get('total_samples', 0),
                        'positive': sentiment_dist.get('POSITIVE', 0),
                        'negative': sentiment_dist.get('NEGATIVE', 0),
                        'neutral': sentiment_dist.get('NEUTRAL', 0)
                    }
            
            return JSONResponse(content=results)
        
        @self.router.post("/api/upload")
        async def upload_html_file(
            file: UploadFile = File(...),
            background_tasks: BackgroundTasks = None
        ):
            """Upload HTML file for analysis"""
            if not file.filename.endswith('.html'):
                raise HTTPException(status_code=400, detail="Only HTML files allowed")
            
            job_id = str(uuid.uuid4())
            cache_folder = f"cache/{job_id}"
            os.makedirs(cache_folder, exist_ok=True)
            
            # Save uploaded file
            file_path = f"{cache_folder}/{file.filename}"
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            # Initialize job
            self.jobs_db[job_id] = {
                'job_id': job_id,
                'status': 'queued',
                'progress': 0,
                'message': 'File uploaded, analysis queued',
                'created_at': datetime.now().isoformat()
            }
            
            # Start analysis
            company_name = self.get_company_name()
            background_tasks.add_task(
                self.run_analysis_pipeline,
                job_id,
                None,
                file_path,
                None,  # email
                None,  # custom_prompt
                'demo',
                None,  # search_input
                company_name
            )
            
            return JobStatus(
                job_id=job_id,
                status='queued',
                progress=0,
                message='File uploaded and queued for analysis'
            )
        
        @self.router.post("/api/results/{job_id}/chat", response_model=ChatResponse)
        async def chat_with_results(job_id: str, request: ChatRequest):
            """
            Interactive chatbot for querying analysis results
            
            Ask natural language questions about the sentiment analysis:
            - "What are customers complaining about?"
            - "Show me examples of negative feedback"
            - "What should we fix first?"
            - etc.
            """
            # Check if job exists and is complete
            if job_id not in self.jobs_db:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs_db[job_id]
            if job['status'] != 'completed':
                raise HTTPException(
                    status_code=400, 
                    detail=f"Analysis not complete yet. Current status: {job['status']}"
                )
            
            # Check if results exist
            results_path = f"my_volume/sentiment_analysis/{job_id}"
            if not os.path.exists(results_path):
                raise HTTPException(status_code=404, detail="Analysis results not found")
            
            # Initialize chatbot if not already created
            if job_id not in self.chatbots:
                groq_api_key = os.getenv('GROQ_API_KEY')
                if not groq_api_key:
                    raise HTTPException(
                        status_code=500, 
                        detail="GROQ_API_KEY not configured on server"
                    )
                
                try:
                    self.chatbots[job_id] = ResultsChatbot(job_id, results_path, groq_api_key)
                    logger.info(f"Created new chatbot for job {job_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize chatbot: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to initialize chatbot: {str(e)}")
            
            # Get answer from chatbot
            try:
                chatbot = self.chatbots[job_id]
                answer = chatbot.ask(request.question, include_history=request.include_history)
                
                # Get suggested questions for first interaction
                suggested = None
                if len(chatbot.conversation_history) <= 2:  # First question
                    suggested = chatbot.get_suggested_questions()
                
                return ChatResponse(
                    job_id=job_id,
                    question=request.question,
                    answer=answer,
                    suggested_questions=suggested
                )
                
            except Exception as e:
                logger.error(f"Error in chatbot interaction: {e}")
                raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")
        
        @self.router.get("/api/results/{job_id}/chat/suggestions")
        async def get_chat_suggestions(job_id: str):
            """Get suggested questions for the chatbot"""
            if job_id not in self.jobs_db:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs_db[job_id]
            if job['status'] != 'completed':
                raise HTTPException(status_code=400, detail="Analysis not complete yet")
            
            # Initialize chatbot if needed
            if job_id not in self.chatbots:
                results_path = f"my_volume/sentiment_analysis/{job_id}"
                groq_api_key = os.getenv('GROQ_API_KEY')
                
                if not groq_api_key:
                    raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
                
                try:
                    self.chatbots[job_id] = ResultsChatbot(job_id, results_path, groq_api_key)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to initialize chatbot: {str(e)}")
            
            suggestions = self.chatbots[job_id].get_suggested_questions()
            
            return {
                "job_id": job_id,
                "suggestions": suggestions
            }
        
        @self.router.delete("/api/results/{job_id}/chat/history")
        async def clear_chat_history(job_id: str):
            """Clear conversation history for a job's chatbot"""
            if job_id not in self.chatbots:
                raise HTTPException(status_code=404, detail="No active chatbot for this job")
            
            self.chatbots[job_id].clear_history()
            
            return {
                "job_id": job_id,
                "message": "Conversation history cleared"
            }
        
        @self.router.post("/api/debug/log")
        async def debug_log(request: Request):
            """Receive debug logs from frontend"""
            try:
                data = await request.json()
                
                # Log to file
                debug_dir = "debuglogs"
                os.makedirs(debug_dir, exist_ok=True)
                
                log_file = os.path.join(debug_dir, f"python-backend-{datetime.now().strftime('%Y-%m-%d')}.log")
                log_line = f"[{data.get('timestamp')}] FRONTEND-{data.get('component')} - {data.get('action')}: {json.dumps(data.get('data'))}\n"
                
                with open(log_file, 'a') as f:
                    f.write(log_line)
                
                logger.info(f"DEBUG [{data.get('component')}] {data.get('action')}: {data.get('data')}")
                
                return {"received": True}
            except Exception as e:
                logger.error(f"Error logging debug: {e}")
                return {"received": False, "error": str(e)}

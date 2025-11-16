#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic models for Sentiment Analysis API
Data validation and serialization schemas
"""

from pydantic import BaseModel
from typing import Optional


class AnalysisRequest(BaseModel):
    """Request model for sentiment analysis"""
    url: Optional[str] = None
    html_content: Optional[str] = None
    email: Optional[str] = None  # Single email or comma-separated emails
    emails: Optional[list] = None  # Alternative: list of emails
    customPrompt: Optional[str] = None  # Custom LLM prompt
    searchMethod: Optional[str] = "demo"  # keywords, urls, or demo


class JobStatus(BaseModel):
    """Status model for analysis job"""
    job_id: str
    status: str
    progress: Optional[int] = 0
    message: Optional[str] = None
    pdf_url: Optional[str] = None
    results_url: Optional[str] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chatbot interaction"""
    question: str
    include_history: Optional[bool] = True


class ChatResponse(BaseModel):
    """Response model from chatbot"""
    job_id: str
    question: str
    answer: str
    suggested_questions: Optional[list] = None

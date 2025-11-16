#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Middleware for FastAPI application
Handles request/response logging and debugging
"""

import os
from datetime import datetime
from fastapi import Request


async def debug_middleware(request: Request, call_next):
    """
    Log all chatbot-related requests for debugging
    
    Logs request details (method, URL, headers, body) and response status
    to daily log files in the debuglogs directory.
    """
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

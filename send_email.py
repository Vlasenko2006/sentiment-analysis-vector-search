#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email sending functionality for sentiment analysis reports
"""

import logging

logger = logging.getLogger(__name__)


def send_email(job_id: str, emails, output_base_dir: str, company_name: str, 
               jobs_db: dict, smtp_config: dict, send_report_email_fun):
    """
    Send email with PDF report to recipients
    
    Args:
        job_id: Job identifier for logging
        emails: Single email, comma-separated emails, or list of emails
        output_base_dir: Directory containing the PDF report
        company_name: Company name for email subject
        jobs_db: Job database to update progress
        smtp_config: Dictionary with SMTP configuration (server, port, email, password)
        send_report_email_fun: Function to send the actual email
    """
    jobs_db[job_id]['progress'] = 95
    jobs_db[job_id]['message'] = 'Sending email...'
    
    # Convert to list only for logging
    if isinstance(emails, str):
        if ',' in emails:
            email_list = [e.strip() for e in emails.split(',')]
        else:
            email_list = [emails]
    else:
        email_list = emails
    
    logger.info(f"Job {job_id}: Sending report to {len(email_list)} recipient(s): {', '.join(email_list)}")
    try:
        pdf_path = f"{output_base_dir}/visualizations/sentiment_analysis_report.pdf"
        send_report_email_fun(
            emails,  # Pass as-is: string (single/comma-separated) or list
            f"Sentiment Analysis Report from {company_name} Platform",
            pdf_path,
            smtp_config['server'],
            smtp_config['port'],
            smtp_config['email'],
            smtp_config['password']
        )
        logger.info(f"Job {job_id}: Email sent successfully to {len(email_list)} recipient(s)")
    except Exception as email_error:
        logger.warning(f"Job {job_id}: Email sending failed - {str(email_error)}")
        # Don't fail the job if email sending fails

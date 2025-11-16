#!/usr/bin/env python3
"""
Email sender script for RoBERTa sentiment analysis report
Sends PDF report to customer with professional email formatting
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime



def create_email_message(SENDER_EMAIL,CUSTOMER_EMAIL,SUBJECT):
    """Create the email "message with PDF attachment"""
    
    # Create message container
    msg = MIMEMultipart()
    
    # Email headers
    msg['From'] = SENDER_EMAIL
    msg['To'] = CUSTOMER_EMAIL
    msg['Subject'] = SUBJECT
    
    # Email body with HTML formatting (matching PDF style)
    html_body = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #1a1a1a;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #0066cc 0%, #004999 100%);
                color: white;
                padding: 30px 20px;
                border-radius: 8px 8px 0 0;
                text-align: center;
            }}
            .header h1 {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-weight: bold;
                font-size: 28px;
                margin: 0;
                padding: 0;
            }}
            .content {{
                background-color: #ffffff;
                padding: 30px;
                border: 1px solid #e0e0e0;
            }}
            .greeting {{
                font-size: 16px;
                color: #1a1a1a;
                font-weight: bold;
                margin-bottom: 15px;
            }}
            .main-text {{
                font-size: 14px;
                color: #1a1a1a;
                margin-bottom: 20px;
            }}
            .section-title {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-weight: bold;
                font-size: 16px;
                color: #0066cc;
                margin-top: 25px;
                margin-bottom: 15px;
            }}
            .feature-list {{
                list-style: none;
                padding-left: 0;
            }}
            .feature-list li {{
                font-size: 13px;
                color: #4a4a4a;
                padding: 8px 0;
                padding-left: 25px;
                position: relative;
            }}
            .feature-list li:before {{
                content: "✓";
                position: absolute;
                left: 0;
                color: #0066cc;
                font-weight: bold;
                font-size: 16px;
            }}
            .footer {{
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 0 0 8px 8px;
                border: 1px solid #e0e0e0;
                border-top: none;
                font-size: 12px;
                color: #4a4a4a;
            }}
            .signature {{
                margin-top: 25px;
                font-size: 14px;
                color: #1a1a1a;
            }}
            .meta {{
                font-size: 11px;
                color: #666;
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #ddd;
            }}
            .highlight {{
                color: #0066cc;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>RoBERTa Sentiment Analysis Report</h1>
        </div>
        
        <div class="content">
            <div class="greeting">Dear Nataliya,</div>
            
            <div class="main-text">
                I hope this email finds you well.
            </div>
            
            <div class="main-text">
                I am pleased to inform you that your <span class="highlight">RoBERTa sentiment analysis report</span> 
                is now ready and complete.
            </div>
            
            <div class="section-title">This comprehensive report includes:</div>
            
            <ul class="feature-list">
                <li>Detailed sentiment analysis results from your filtered review data</li>
                <li>Neural network-based comment classification using DistilBERT</li>
                <li>Vector search clustering and representative comment identification</li>
                <li>AI-generated sentiment summaries and actionable recommendations</li>
                <li>Sentiment trends visualization over time</li>
                <li>Comprehensive visualizations and statistical insights</li>
                <li>Technical analysis methodology and performance metrics</li>
            </ul>
            
            <div class="main-text" style="margin-top: 25px;">
                The report has been generated using advanced machine learning techniques and provides 
                <span class="highlight">actionable insights</span> from your review dataset.
            </div>
            
            <div class="main-text">
                <strong>Please find the complete PDF report attached to this email.</strong>
            </div>
            
            <div class="main-text">
                If you have any questions about the analysis or need further clarification on any 
                aspect of the report, please don't hesitate to reach out.
            </div>
            
            <div class="signature">
                <strong>Best regards,</strong><br>
                RoBERTa Analysis Team
            </div>
        </div>
        
        <div class="footer">
            <div class="meta">
                <strong>Report Details:</strong><br>
                Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}<br>
                Analysis Method: DistilBERT sentiment classification with TF-IDF vectorization<br>
                AI Summaries: Groq LLM (llama-3.1-8b-instant)
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create plain text version as fallback
    plain_body = f"""Dear Nataliya,

I hope this email finds you well.

I am pleased to inform you that your RoBERTa sentiment analysis report is now ready and complete.

This comprehensive report includes:
• Detailed sentiment analysis results from your filtered review data
• Neural network-based comment classification using DistilBERT
• Vector search clustering and representative comment identification
• AI-generated sentiment summaries and actionable recommendations
• Sentiment trends visualization over time
• Comprehensive visualizations and statistical insights
• Technical analysis methodology and performance metrics

The report has been generated using advanced machine learning techniques and provides actionable insights from your review dataset.

Please find the complete PDF report attached to this email.

If you have any questions about the analysis or need further clarification on any aspect of the report, please don't hesitate to reach out.

Best regards,
RoBERTa Analysis Team

---
Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis method: DistilBERT sentiment classification with TF-IDF vectorization
AI Summaries: Groq LLM (llama-3.1-8b-instant)
"""
    
    # Attach both plain and HTML versions (HTML will be preferred by email clients)
    msg.attach(MIMEText(plain_body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))
    
    return msg

def attach_pdf_report(msg,PDF_PATH):
    """Attach PDF report to email message"""
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF report not found at {PDF_PATH}")
        print("Please run generate_pdf_only.py first to create the report.")
        return False
    
    try:
        # Open PDF file in binary mode
        with open(PDF_PATH, "rb") as attachment:
            # Instance of MIMEBase and named as part
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        
        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)
        
        # Add header as key/value pair to attachment part
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= "RoBERTa_Sentiment_Analysis_Report.pdf"',
        )
        
        # Attach the part to message
        msg.attach(part)
        
        file_size = os.path.getsize(PDF_PATH) / (1024 * 1024)  # Size in MB
        print(f"✅ PDF report attached successfully ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Error attaching PDF: {e}")
        return False

def send_email(CUSTOMER_EMAIL,
                      SUBJECT,
                      PDF_PATH,
                      SMTP_SERVER,
                      SMTP_PORT,
                      SENDER_EMAIL,
                      SENDER_PASSWORD):
    """Send the email with PDF attachment"""
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF report not found at: {PDF_PATH}")
        print("Please run generate_pdf_only.py first to create the report.")
        return False
    
    try:
        # Create email message
        print("📧 Creating email message...")
        msg = create_email_message(SENDER_EMAIL,CUSTOMER_EMAIL,SUBJECT)
        
        # Attach PDF
        print("📎 Attaching PDF report...")
        if not attach_pdf_report(msg, PDF_PATH):
            return False
        
        # Note: Actual email sending is commented out for security
        # Uncomment and configure SMTP settings to actually send
        
        print("📧 Email prepared successfully!")
        print(f"   To: {CUSTOMER_EMAIL}")
        print(f"   Subject: {SUBJECT}")
        print(f"   Attachment: RoBERTa_Sentiment_Analysis_Report.pdf")
        print(f"   PDF Path: {PDF_PATH}")
        
        # Actual SMTP sending (uncomment to use):
  
        print("📤 Connecting to SMTP server...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Enable security
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        # Send email
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, CUSTOMER_EMAIL, text)
        server.quit()
        
        print("✅ Email sent successfully!")

        
        print("\n⚠️  NOTE: Email sending is disabled for security.")
        print("   To actually send the email:")
        print("   1. Configure SENDER_EMAIL and SENDER_PASSWORD")
        print("   2. Uncomment the SMTP sending code section")
        print("   3. For Gmail, use an 'App Password' instead of your regular password")
        
        return True
        
    except Exception as e:
        print(f"❌ Error preparing email: {e}")
        return False

def send_report_email_fun(CUSTOMER_EMAIL,
                      SUBJECT,
                      PDF_PATH,
                      SMTP_SERVER,
                      SMTP_PORT,
                      SENDER_EMAIL,
                      SENDER_PASSWORD
                      ):
    """Main function"""
    print("🤖 RoBERTa Report Email Sender")
    print("=" * 50)
    
    # Check if all required files exist
    print(f"📋 Checking PDF report: {PDF_PATH}")
    
    if os.path.exists(PDF_PATH):
        file_size = os.path.getsize(PDF_PATH) / (1024 * 1024)
        print(f"✅ PDF report found ({file_size:.2f} MB)")
    else:
        print(f"❌ PDF report not found!")
        print("Please run generate_pdf_only.py first.")
        return
    
    # Send email
    success = send_email(CUSTOMER_EMAIL,
                          SUBJECT,
                          PDF_PATH,
                          SMTP_SERVER,
                          SMTP_PORT,
                          SENDER_EMAIL,
                          SENDER_PASSWORD
                          )
    
    if success:
        print("\n🎉 Email preparation completed successfully!")
        print("Configure SMTP settings and uncomment sending code to actually send.")
    else:
        print("\n❌ Email preparation failed.")


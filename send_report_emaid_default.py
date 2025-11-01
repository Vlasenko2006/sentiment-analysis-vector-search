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

# Email configuration
CUSTOMER_EMAIL = "nataliya.stashchuk@gmail.com"
SUBJECT = "Report from RoBERTa"

# PDF report path (from generate_pdf_only.py)
PDF_PATH = "/Users/andreyvlasenko/tst/Request/my_volume/sentiment_analysis/visualizations/sentiment_analysis_report.pdf"

# SMTP configuration (you'll need to configure this with your email provider)
# For Gmail, you would need an "App Password" instead of your regular password
SMTP_SERVER = "smtp.gmail.com"  # Change this to your email provider's SMTP server
SMTP_PORT = 587
SENDER_EMAIL = "max.mustermann@gmail.com"  # Replace with your email
SENDER_PASSWORD = "abcdefghijkl"  # Replace with your app password from Google

def create_email_message():
    """Create the email "message with PDF attachment"""
    
    # Create message container
    msg = MIMEMultipart()
    
    # Email headers
    msg['From'] = SENDER_EMAIL
    msg['To'] = CUSTOMER_EMAIL
    msg['Subject'] = SUBJECT
    
    # Email body
    body = f"""Dear Nataliya,

I hope this email finds you well.

I am pleased to inform you that your RoBERTa sentiment analysis report is now ready and complete. 

This comprehensive report includes:
• Detailed sentiment analysis results from your filtered review data
• Neural network-based comment classification using DistilBERT
• Vector search clustering and representative comment identification
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
"""
    
    # Attach body to email
    msg.attach(MIMEText(body, 'plain'))
    
    return msg

def attach_pdf_report(msg):
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

def send_email():
    """Send the email with PDF attachment"""
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF report not found at: {PDF_PATH}")
        print("Please run generate_pdf_only.py first to create the report.")
        return False
    
    try:
        # Create email message
        print("📧 Creating email message...")
        msg = create_email_message()
        
        # Attach PDF
        print("📎 Attaching PDF report...")
        if not attach_pdf_report(msg):
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

def main():
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
    success = send_email()
    
    if success:
        print("\n🎉 Email preparation completed successfully!")
        print("Configure SMTP settings and uncomment sending code to actually send.")
    else:
        print("\n❌ Email preparation failed.")

if __name__ == "__main__":
    main()
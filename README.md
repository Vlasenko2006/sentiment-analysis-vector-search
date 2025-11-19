# 🎯 Sentiment Analysis Platform

**Production-Ready ML System for Customer Review Analysis**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![.NET](https://img.shields.io/badge/.NET-8.0-512BD4?logo=.net)](https://dotnet.microsoft.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)

---

# Roberta — ELT Frontend-Backend System for Content Analysis

## Summary
Roberta is an ELT frontend-backend universal platform that processes information from both web sources and existing databases. It runs a three-model AI pipeline to analyze content (for example, customer reviews), extracts the most relevant and representative pieces of information (e.g., the most critical and the most typical comments), summarizes findings, and produces text and visual reports — for example, suggestions to improve your business, risk assessments, or insurance‑relevant conclusions.

## User flow
- Enter your email.
- Provide a website or list of websites of interest (e.g., review sites).
- Click "Send report."

## Top of the RoBERTa's front page

<p align="center">
<img src="https://github.com/Vlasenko2006/sentiment-analysis-vector-search/blob/main/Images/RoBERTa.png" alt="RoBERTa front page" width="90%">
</p>

## Report & architecture overview
- When processing finishes, Roberta emails you a report with charts (for example, trends in positive vs. negative comments).
- While the report is being generated you can preview an example report and its structure.

## Report sample (pages 7 and 14)

<table align="center">
  <tr>
    <td><img src="https://github.com/Vlasenko2006/sentiment-analysis-vector-search/blob/main/Images/Report1.png" alt="Report 1" width="100%"></td>
    <td><img src="https://github.com/Vlasenko2006/sentiment-analysis-vector-search/blob/main/Images/Report2.png" alt="Report 2" width="100%"></td>
  </tr>
</table>

## How it works (technical flow)
- The .NET frontend forwards the request to a Python script that performs the main processing. The Python pipeline:
  - Loads the specified sites and performs ELT processing.
  - Detects and extracts reviews from the raw text using rule-based methods and the DistilBERT model.
  - Performs vector and semantic analysis to identify the most representative and the most salient comments.
  - Runs semantic analysis using an LLM (Llama).
  - Generates textual summaries grouped by sentiment category.
  - Produces and sends to the user comprehensive .pdf report with:
    - Sentiment distribution charts
    - Key insights
    - Example reviews
    - Prioritized recommendations for service improvement
    - Risk assesment
  - Sends recommendations to the user by email.

## Interactive results
Roberta includes a Results Chatbot that uses RAG (Retrieval‑Augmented Generation) over the analysis results, so you can ask for clarifications, explanations of individual comments, or details about the analysis methodology at any time.

## Key features
1. Flexible — configurable for many business use cases (e.g., restaurants, insurance valuation, creditworthiness assessments).  
2. Asynchronous processing via FastAPI.  
3. Production ready: fully dockerized and scalable — already deployed on AWS (in future on Azure).  
4. CI/CD ready: automated testing; integration with MLflow (with 15+ metrics) and pytest (21+ pytests).  
5. Caching support using SQLite or Redis.

---

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│                      http://localhost:3000                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (nginx:alpine)                      │
│                                                                 │
│  • Serves static HTML/CSS/JavaScript                           │
│  • User interface for submitting analysis jobs                 │
│  • Real-time status updates                                    │
│  • Results visualization                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /api/analyze
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              .NET API GATEWAY (ASP.NET Core 8.0)               │
│                    http://localhost:5000                        │
│                                                                 │
│  • Entry point for all API requests                            │
│  • Request validation and routing                              │
│  • Acts as reverse proxy to Python backend                     │
│  • Health check aggregation                                    │
│  • Future: Load balancing, rate limiting, auth                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ Proxy to Python
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PYTHON BACKEND (FastAPI + Uvicorn)                │
│                    http://localhost:8000                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Load DistilBERT Model (transformers)                │   │
│  │    • distilbert-base-uncased-finetuned-sst-2-english   │   │
│  │    • 66M parameters, 256MB model size                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │ 2. Sentiment Analysis Pipeline                         │   │
│  │    • Tokenize reviews                                  │   │
│  │    • Run inference (positive/negative/neutral)         │   │
│  │    • Aggregate sentiment distribution                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │ 3. LLM Summarization (Groq API)                        │   │
│  │    • Llama 3.1 70B model                               │   │
│  │    • Generate insights and recommendations             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │ 4. MLflow Experiment Tracking                          │   │
│  │    • Log parameters (model, thresholds, settings)      │   │
│  │    • Log metrics (sentiment ratios, processing time)   │   │
│  │    • Store artifacts (PDF reports, JSON summaries)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │ 5. Report Generation                                   │   │
│  │    • Create PDF with matplotlib/reportlab              │   │
│  │    • Generate JSON summaries                           │   │
│  │    • Email delivery (optional)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │ 6. Cache Results (Redis - Ready)                       │   │
│  │    • Cache sentiment predictions                       │   │
│  │    • Cache LLM responses                               │   │
│  │    • Reduce API costs and latency                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Results
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MLFLOW TRACKING SERVER                      │
│                    http://localhost:5002                        │
│                                                                 │
│  • Experiment tracking UI                                      │
│  • Compare runs side-by-side                                   │
│  • View metrics, parameters, artifacts                         │
│  • Download PDF reports and data                               │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow Explained

1. **User** opens frontend at localhost:3000
2. **Frontend** sends analysis request to .NET API (localhost:5000)
3. **.NET API** validates request and proxies to Python backend (localhost:8000)
4. **Python Backend** executes ML pipeline:
   - Loads DistilBERT model
   - Analyzes sentiment of each review
   - Generates LLM summaries via Groq
   - Logs everything to MLflow
   - Creates PDF report
   - Returns results
5. **Response** flows back: Python → .NET → Frontend → User
6. **User** can view MLflow experiments at localhost:5002

---

## ✨ Complete Feature List

### 🎯 Sentiment Analysis Core

| # | Feature | Description |
|---|---------|-------------|
| 1 | **DistilBERT Classification** | State-of-the-art transformer (66M params, SST-2 fine-tuned) |
| 2 | **Multi-Class Sentiment** | Positive, Negative, Neutral detection with confidence scores |
| 3 | **Batch Processing** | Efficient parallel analysis of 100+ reviews |
| 4 | **Confidence Thresholds** | Configurable sensitivity for classification |
| 5 | **Sentiment Distribution** | Aggregate statistics across all reviews |



### 📈 MLflow Experiment Tracking

| # | Feature | Description |
|---|---------|-------------|
| 17 | **Parameter Logging** | Model, thresholds, settings (15+ per run) |
| 18 | **Metric Tracking** | Sentiment ratios, processing time, API usage (10+ metrics) |
| 19 | **Artifact Storage** | PDFs, JSONs, recommendations versioned |
| 20 | **Run Comparison** | Side-by-side experiment analysis |
| 21 | **Experiment History** | Complete audit trail of all analyses |
| 22 | **Reproducibility** | Every run fully reproducible with logged params |

### 🧪 Testing & Quality

| # | Feature | Description |
|---|---------|-------------|
| 23 | **Unit Test Suite** | 21+ tests covering API and MLflow |
| 24 | **API Endpoint Tests** | Health, config, analyze, status validated |
| 25 | **MLflow Tests** | Parameter/metric/artifact logging verified |
| 26 | **Coverage Reports** | pytest-cov for code coverage measurement |
| 27 | **Async Testing** | pytest-asyncio for async endpoint tests |
| 28 | **CI/CD Ready** | Test automation for GitHub Actions |

### 🏗️ Architecture & Infrastructure

| # | Feature | Description |
|---|---------|-------------|
| 29 | **Microservices Design** | Frontend, .NET API, Python backend separation |
| 30 | **Docker Containerization** | All services in isolated containers |
| 31 | **Docker Compose** | One-command deployment |
| 32 | **Health Checks** | Automated service monitoring |
| 33 | **API Gateway Pattern** | .NET as entry point with routing |
| 34 | **Horizontal Scaling** | Easy to add more Python workers |

### ⚡ Performance & Optimization

Platform normaly runs on AWS micro3 instance (this AWS offers you for free).

---

## 🚀 Quick Start on a Local Machine (your laptop)

### Prerequisites

- Docker Desktop installed and running
- 8GB RAM minimum
- 10GB free disk space

### One-Command Deployment

```bash
# Set PATH (macOS)
export PATH="/usr/local/bin:$PATH"

# Navigate to project
cd /path/to/Request

# Build and start
docker compose up -d

# Access services
open http://localhost:3000  # Frontend (or port 3001, not sure. check the dockerfile)
open http://localhost:8000/docs  # API Docs Bthw. it might be 8001, check it in the dockerfile
open http://localhost:5002  # MLflow (after setup)
```

---

## 📊 Technology Stack

### Backend
- **Python 3.10** - ML backend runtime
- **FastAPI 0.104** - Modern async web framework
- **DistilBERT** - Transformer model (Hugging Face)
- **Groq API** - LLM integration (Llama 3.1)
- **MLflow 2.9+** - Experiment tracking
- **.NET 8.0** - API gateway (ASP.NET Core)

### Frontend
- **nginx:alpine** - Web server
- **HTML/CSS/JS** - User interface

### Infrastructure
- **Docker Compose** - Multi-container orchestration
- **pytest 7.4+** - Testing framework
- **Redis 5.0+** - Caching (ready)

---

## ☁️ Deployment on AWS EC2 Deployment Guide

Deploy the sentiment analysis platform to AWS EC2 for production-like hosting with public access.

### Prerequisites

- AWS account with Free Tier
- Basic AWS knowledge (EC2, Security Groups)
- SSH key pair for EC2 access

### Step-by-Step Deployment

#### 1. Launch EC2 Instance

**Instance Configuration:**
- **Instance Type**: `t3.micro` (1 vCPU, 1GB RAM) - Free Tier eligible
- **AMI**: Ubuntu 24.04 LTS
- **Storage**: 20GB EBS volume (within Free Tier 30GB limit)
- **Region**: Choose closest to your location (e.g., eu-north-1)

**Security Group Rules:**
```
Inbound:
- SSH (22)        → Your IP only (e.g., 95.91.224.181/32)
- HTTP (3000)     → 0.0.0.0/0 (public access to frontend)

Outbound:
- All traffic     → 0.0.0.0/0 (default)
```

⚠️ **Security**: Do NOT expose ports 5000 or 8000 to internet - keep APIs internal!

#### 2. Configure Elastic IP (Permanent Address)

```bash
# In AWS Console:
1. EC2 → Elastic IPs → "Allocate Elastic IP address"
2. Select new IP → Actions → "Associate Elastic IP address"
3. Choose your EC2 instance
4. Note the IP (e.g., 13.48.16.109)
```

**Benefits:**
- ✅ Permanent IP that survives instance restarts
- ✅ Free while associated with running instance
- ✅ Can point domain names to it

#### 3. Setup Swap Memory (Critical for 1GB RAM)

The DistilBERT model and Docker builds require more than 1GB RAM. Add swap:

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@YOUR_ELASTIC_IP

# Create 2GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify swap is active
free -h
# Output should show 2.0Gi swap

# Make swap permanent (survives reboots)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Why needed:**
- Docker builds fail with OOM (Out of Memory) without swap
- Model loading requires ~1.5GB memory
- Build process peaks at ~2GB total

#### 4. Install Docker & Docker Compose

```bash
# Update system
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (no sudo needed)
sudo usermod -aG docker ubuntu
exit  # Log out and back in

# Verify installation
docker --version
docker compose version
```

#### 5. Clone Repository & Configure

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-vector-search.git
cd sentiment-analysis-vector-search

# Switch to deployment branch
git checkout web-service

# Create config file with API keys
nano config_key.yaml
```

**config_key.yaml template:**
```yaml
# Sensitive Configuration - DO NOT COMMIT TO GIT

# Groq API Configuration
groq:
  api_key: "YOUR_GROQ_API_KEY"  # Get from https://console.groq.com

# Email Configuration (optional)
email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "your@email.com"
  sender_password: "your_app_password"

# Analysis Parameters
analysis:
  key_positive_words:
    - "excellent"
    - "amazing"
    - "wonderful"
  
  key_neutral_words:
    - "location"
    - "place"
  
  key_negative_words:
    - "terrible"
    - "awful"
  
  separator_keywords:
    - "•"
    - "Written "
    - "Reviewed "
    - "Visited "
  
  sentence_length: 4
  default_prompt: "Provide 3 actionable recommendations for improvement."
```

#### 6. Fix Dockerfile Memory Issues

⚠️ **Critical**: Comment out model download in `Dockerfile.python` to prevent OOM during build:

```dockerfile
# Lines 22-25 - COMMENT THESE OUT:
# RUN python -c "from transformers import pipeline; \
#     pipe = pipeline('sentiment-analysis', \
#     model='distilbert-base-uncased-finetuned-sst-2-english'); \
#     print('Model downloaded successfully')"
```
Then **after your images are built and containers running** run this fix only once:

```
docker exec sentiment-python-v2 python -c "
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model.save_pretrained('./my_volume/hf_model')
tokenizer.save_pretrained('./my_volume/hf_model')"
```
I would be grateful, if you suggest me better options to solve tight memory issue on free trial :)



**Why:** Model downloads during runtime (first API call), not build time. This prevents build failures.

#### 7. Build and Start Services

```bash
# Export API key for docker-compose
export GROQ_API_KEY=$(grep 'api_key:' config_key.yaml | awk '{print $2}' | tr -d '"')

# Build and start all containers
docker compose up -d --build

# Wait for services to start (~30 seconds)
sleep 30

# Check container status
docker ps
# All should show "healthy" status

# View logs
docker compose logs -f
```

#### 8. Verify Deployment

```bash
# Test Python API
curl http://localhost:8000/health
# {"status":"healthy","timestamp":"..."}

# Test .NET API
curl http://localhost:5000/health
# {"status":"Healthy",...}

# Test Frontend
curl -I http://localhost:3000
# HTTP/1.1 200 OK
```

**Access from browser:**
- Frontend: `http://YOUR_ELASTIC_IP:3000`
- API Docs: `http://YOUR_ELASTIC_IP:8000/docs` (if you expose port 8000 - not recommended)


#### 10. Auto-Start on Reboot (Optional)

Services restart automatically thanks to `restart: unless-stopped` in docker-compose.yml.

For guaranteed startup after EC2 reboot, create systemd service:

```bash
sudo nano /etc/systemd/system/sentiment-app.service
```

```ini
[Unit]
Description=Sentiment Analysis Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/sentiment-analysis-vector-search
ExecStartPre=/bin/bash -c 'export GROQ_API_KEY=$(grep "api_key:" config_key.yaml | awk "{print \$2}" | tr -d "\"")'
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=ubuntu

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable sentiment-app.service
sudo systemctl start sentiment-app.service
```


### Maintenance Commands

```bash
# Restart services
docker compose restart

# Stop services
docker compose down

# View logs
docker compose logs -f python-service
docker compose logs -f dotnet-api

# Check disk space
df -h

# Check memory usage
free -h

# Update code from GitHub
git pull origin web-service
docker compose up -d --build
```

### Troubleshooting

**Container exits immediately:**
```bash
# Check logs
docker logs sentiment-python --tail 50

# Common issues:
# - Missing config_key.yaml
# - GROQ_API_KEY not set
# - OOM during build (need swap)
```

**Out of disk space:**
```bash
# Clean up Docker
docker system prune -a

# Increase EBS volume in AWS Console
# Then expand filesystem:
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
```

**Port already in use:**
```bash
# Check what's using the port
sudo lsof -i :3000
sudo lsof -i :8000

# Kill process or stop conflicting service
```
---

## 📜 License

Educational and portfolio purposes.

**Model Licenses**:
- DistilBERT: Apache 2.0 (Hugging Face)
- Groq API: Commercial (requires API key)

---

## 🙏 Acknowledgments

- **Hugging Face** for DistilBERT and transformers
- **MLflow** for experiment tracking
- **FastAPI** for modern Python APIs
- **Groq** for LLM API access

---

**Built with ❤️ for production ML deployment and data engineering interviews**

*Last updated: November 2025*

---

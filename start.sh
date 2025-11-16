#!/bin/bash

# Startup script for local containerized development
# This script builds and starts all services

set -e

echo "🚀 Starting Sentiment Analysis Containerized Stack"
echo "=================================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Copying from .env.example"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it with your API keys."
        echo "Press Enter after updating .env file..."
        read
    else
        echo "❌ Error: .env.example not found. Please create .env file manually."
        exit 1
    fi
fi

# Stop any running containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build images
echo "🔨 Building Docker images..."
docker-compose build --no-cache

# Start services
echo "▶️  Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to become healthy..."
sleep 10

# Check Python service health
echo "🔍 Checking Python service..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Python service is healthy"
else
    echo "❌ Python service is not responding"
    docker-compose logs python-service
fi

# Check .NET API health
echo "🔍 Checking .NET API..."
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ .NET API is healthy"
else
    echo "❌ .NET API is not responding"
    docker-compose logs dotnet-api
fi

# Check Frontend health
echo "🔍 Checking Frontend..."
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend is not responding"
    docker-compose logs frontend
fi

echo ""
echo "=================================================="
echo "✨ Services are running!"
echo ""
echo "📊 Available Endpoints:"
echo "  - Frontend UI: http://localhost:3000"
echo "  - .NET API: http://localhost:5000"
echo "  - Python Service: http://localhost:8000"
echo "  - Swagger UI: http://localhost:5000/swagger"
echo ""
echo "🌐 Open the frontend in your browser:"
echo "  open http://localhost:3000    # macOS"
echo "  xdg-open http://localhost:3000  # Linux"
echo ""
echo "📝 Example Usage:"
echo "  1. Open http://localhost:3000"
echo "  2. Enter your email"
echo "  3. Choose search method (demo mode is pre-selected)"
echo "  4. Click 'Start Analysis'"
echo "  5. Watch the progress bar"
echo "  6. Download PDF when complete"
echo ""
echo "🔍 View logs:"
echo "  docker-compose logs -f"
echo ""
echo "🛑 Stop services:"
echo "  docker-compose down"
echo "=================================================="

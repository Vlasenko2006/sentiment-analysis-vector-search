#!/bin/bash
# Frontend fix applied: Changed requestData.searchInput to requestData.url
# API expects: { email, url: "demo", searchMethod: "demo" }

echo "🔧 Rebuilding frontend with fixed app.js..."
cd /Users/andreyvlasenko/tst/Request

docker compose build frontend
if [ $? -eq 0 ]; then
    echo "✅ Build successful"
    
    echo "🚀 Restarting frontend container..."
    docker compose up -d frontend
    
    if [ $? -eq 0 ]; then
        echo "✅ Frontend restarted"
        echo ""
        echo "📋 Next steps:"
        echo "1. Open http://localhost:3000"
        echo "2. Hard refresh browser: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)"
        echo "3. Open browser console (F12)"
        echo "4. Fill email, keep 'demo mode' selected"
        echo "5. Click 'Start Analysis'"
        echo "6. Watch console for: 'Submitting analysis request: {email, url: demo, searchMethod: demo}'"
        echo "7. Progress bar should appear and analysis should start"
        echo "8. After ~2-3 minutes, chatbot button should appear!"
    else
        echo "❌ Failed to restart frontend"
    fi
else
    echo "❌ Build failed"
fi

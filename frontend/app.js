// Configuration
const API_BASE_URL = 'http://localhost:5000/api/sentiment';
const POLL_INTERVAL = 3000; // Poll every 3 seconds

// Global state
let currentJobId = null;
let pollInterval = null;

// DOM Elements
const analysisForm = document.getElementById('analysisForm');
const submitBtn = document.getElementById('submitBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const taskSection = document.getElementById('taskSection');
const customTaskTextarea = document.getElementById('customTask');

// Radio buttons and conditional inputs
const radioKeywords = document.getElementById('option-keywords');
const radioUrls = document.getElementById('option-urls');
const radioDemo = document.getElementById('option-demo');
const inputKeywords = document.getElementById('input-keywords');
const inputUrls = document.getElementById('input-urls');

// Progress elements
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// Result buttons
const downloadPdfBtn = document.getElementById('downloadPdfBtn');
const viewDataBtn = document.getElementById('viewDataBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const retryBtn = document.getElementById('retryBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    console.log('Frontend initialized');
});

// Setup Event Listeners
function setupEventListeners() {
    // Form submission
    analysisForm.addEventListener('submit', handleFormSubmit);

    // Radio button changes
    radioKeywords.addEventListener('change', handleSearchMethodChange);
    radioUrls.addEventListener('change', handleSearchMethodChange);
    radioDemo.addEventListener('change', handleSearchMethodChange);

    // Result buttons
    downloadPdfBtn.addEventListener('click', downloadPdf);
    viewDataBtn.addEventListener('click', viewJsonData);
    newAnalysisBtn.addEventListener('click', resetForm);
    retryBtn.addEventListener('click', resetForm);
}

// Handle search method radio button changes
function handleSearchMethodChange() {
    // Hide all conditional inputs
    inputKeywords.style.display = 'none';
    inputUrls.style.display = 'none';

    // Clear required attributes
    document.getElementById('keywords').removeAttribute('required');
    document.getElementById('urls').removeAttribute('required');

    // Show relevant input and update task section
    if (radioKeywords.checked) {
        inputKeywords.style.display = 'block';
        document.getElementById('keywords').setAttribute('required', 'required');
        customTaskTextarea.disabled = false;
    } else if (radioUrls.checked) {
        inputUrls.style.display = 'block';
        document.getElementById('urls').setAttribute('required', 'required');
        customTaskTextarea.disabled = false;
    } else if (radioDemo.checked) {
        // Demo mode - disable custom task
        customTaskTextarea.disabled = true;
        customTaskTextarea.value = '';
    }
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();

    // Get form values
    const email = document.getElementById('email').value.trim();
    const searchMethod = document.querySelector('input[name="searchMethod"]:checked').value;
    const customTask = document.getElementById('customTask').value.trim();

    window.debugLogger?.log('APP', '← INPUT: Form Submit', { 
        email, 
        searchMethod, 
        customTask, 
        path: 'user→form' 
    });

    // Validate email
    if (!validateEmail(email)) {
        window.debugLogger?.log('ERROR', 'Form Validation Failed', { 
            reason: 'Invalid email', 
            email, 
            path: 'form validation' 
        });
        showError('Please enter a valid email address.');
        return;
    }

    // Prepare request data
    let requestData = {
        email: email,
        searchMethod: searchMethod
    };

    // Add search input based on method
    if (searchMethod === 'keywords') {
        const keywords = document.getElementById('keywords').value.trim();
        if (!keywords) {
            window.debugLogger?.log('ERROR', 'Form Validation Failed', { 
                reason: 'Missing keywords', 
                path: 'form validation' 
            });
            showError('Please enter search keywords.');
            return;
        }
        requestData.url = keywords;
    } else if (searchMethod === 'urls') {
        const urls = document.getElementById('urls').value.trim();
        if (!urls) {
            window.debugLogger?.log('ERROR', 'Form Validation Failed', { 
                reason: 'Missing URLs', 
                path: 'form validation' 
            });
            showError('Please enter at least one URL.');
            return;
        }
        requestData.url = urls;
    } else {
        // Demo mode
        requestData.url = 'demo';
    }

    // Add custom task if provided
    if (customTask && searchMethod !== 'demo') {
        requestData.customPrompt = customTask;
    }

    window.debugLogger?.log('APP', '→ OUTPUT: Prepared Request Data', { 
        requestData, 
        path: 'form→startAnalysis' 
    });

    // Start analysis
    await startAnalysis(requestData);
}

// Start analysis
async function startAnalysis(requestData) {
    window.debugLogger?.log('APP', '← INPUT: Start Analysis', { 
        requestData, 
        path: 'handleFormSubmit→startAnalysis' 
    });

    try {
        // Hide form and error sections
        analysisForm.style.display = 'none';
        errorSection.style.display = 'none';
        resultsSection.style.display = 'none';

        // Show progress section
        progressSection.style.display = 'block';
        resetProgress();

        // Disable submit button
        submitBtn.disabled = true;

        const apiPayload = {
            url: requestData.url || 'demo',
            htmlContent: null,
            email: requestData.email,
            customPrompt: requestData.customPrompt || null,
            searchMethod: requestData.searchMethod
        };

        const apiUrl = `${API_BASE_URL}/analyze`;
        
        window.debugLogger?.log('APP', '→ OUTPUT: API Request', { 
            url: apiUrl, 
            method: 'POST',
            payload: apiPayload, 
            path: 'app.js→.NET API' 
        });

        // Call API
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(apiPayload)
        });

        window.debugLogger?.log('APP', `← INPUT: API Response Status ${response.status}`, { 
            ok: response.ok, 
            status: response.status, 
            statusText: response.statusText,
            path: '.NET API→app.js' 
        });

        if (!response.ok) {
            const errorText = await response.text();
            window.debugLogger?.log('ERROR', 'API Request Failed', { 
                status: response.status, 
                error: errorText,
                path: '.NET API→app.js' 
            });
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        window.debugLogger?.log('APP', '← INPUT: API Response Data', { 
            jobId: data.jobId || data.job_id, 
            data, 
            path: '.NET API→app.js' 
        });

        currentJobId = data.jobId || data.job_id;  // Support both camelCase and snake_case

        window.debugLogger?.log('APP', '→ OUTPUT: Starting Polling', { 
            jobId: currentJobId, 
            path: 'app.js→startPolling' 
        });

        // Start polling for status
        startPolling();

    } catch (error) {
        window.debugLogger?.log('ERROR', 'Analysis Start Failed', { 
            error: error.message, 
            stack: error.stack,
            path: 'app.js error' 
        });
        console.error('Error starting analysis:', error);
        showError(`Failed to start analysis: ${error.message}`);
        analysisForm.style.display = 'block';
        progressSection.style.display = 'none';
        submitBtn.disabled = false;
    }
}

// Start polling for job status
function startPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    pollInterval = setInterval(checkJobStatus, POLL_INTERVAL);
    checkJobStatus(); // Check immediately
}

// Check job status
async function checkJobStatus() {
    if (!currentJobId) {
        window.debugLogger?.log('ERROR', 'No Job ID', { 
            currentJobId, 
            path: 'checkJobStatus' 
        });
        stopPolling();
        return;
    }

    try {
        const statusUrl = `${API_BASE_URL}/status/${currentJobId}`;
        
        window.debugLogger?.log('APP', '→ OUTPUT: Status Check Request', { 
            url: statusUrl, 
            jobId: currentJobId,
            path: 'app.js→.NET API' 
        });

        const response = await fetch(statusUrl);

        window.debugLogger?.log('APP', `← INPUT: Status Response ${response.status}`, { 
            ok: response.ok, 
            status: response.status,
            path: '.NET API→app.js' 
        });

        if (!response.ok) {
            window.debugLogger?.log('ERROR', 'Status Check Failed', { 
                status: response.status, 
                jobId: currentJobId,
                path: '.NET API→app.js' 
            });
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        window.debugLogger?.log('APP', '← INPUT: Status Data', { 
            status: data.status, 
            progress: data.progress, 
            data,
            path: '.NET API→app.js' 
        });

        // Update progress
        updateProgress(data.progress, data.status);

        // Check if completed
        if (data.status === 'completed') {
            window.debugLogger?.log('APP', 'Analysis Completed', { 
                jobId: currentJobId, 
                progress: data.progress,
                path: 'app.js' 
            });
            stopPolling();
            showResults(data);
        } else if (data.status === 'failed') {
            window.debugLogger?.log('ERROR', 'Analysis Failed', { 
                jobId: currentJobId, 
                message: data.message,
                path: 'app.js' 
            });
            stopPolling();
            showError(data.message || 'Analysis failed. Please try again.');
        }

    } catch (error) {
        window.debugLogger?.log('ERROR', 'Status Check Exception', { 
            error: error.message, 
            jobId: currentJobId,
            stack: error.stack,
            path: 'app.js error' 
        });
        console.error('Error checking job status:', error);
        stopPolling();
        showError(`Failed to check job status: ${error.message}`);
    }
}

// Stop polling
function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Update progress bar and steps
function updateProgress(progress, status) {
    window.debugLogger?.log('PROGRESS', '← INPUT: Update Progress', { 
        progress, 
        status, 
        path: 'checkJobStatus→updateProgress' 
    });

    // Update progress bar
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `${progress}%`;

    window.debugLogger?.log('PROGRESS', '→ OUTPUT: Progress Bar Updated', { 
        width: `${progress}%`, 
        text: `${progress}%`,
        path: 'updateProgress→DOM' 
    });

    // Update steps based on progress
    const steps = [
        { id: 'step-init', threshold: 10 },
        { id: 'step-download', threshold: 20 },
        { id: 'step-extract', threshold: 30 },
        { id: 'step-analyze', threshold: 40 },
        { id: 'step-summarize', threshold: 60 },
        { id: 'step-recommend', threshold: 75 },
        { id: 'step-pdf', threshold: 90 },
        { id: 'step-complete', threshold: 100 }
    ];

    steps.forEach(step => {
        const stepElement = document.getElementById(step.id);
        if (progress >= step.threshold) {
            stepElement.classList.add('completed');
            stepElement.classList.remove('active');
        } else if (progress >= step.threshold - 10) {
            stepElement.classList.add('active');
            stepElement.classList.remove('completed');
        } else {
            stepElement.classList.remove('active', 'completed');
        }
    });

    window.debugLogger?.log('PROGRESS', '→ OUTPUT: Steps Updated', { 
        progress, 
        completedSteps: steps.filter(s => progress >= s.threshold).map(s => s.id),
        path: 'updateProgress→DOM' 
    });
}

// Reset progress
function resetProgress() {
    progressFill.style.width = '0%';
    progressText.textContent = '0%';

    // Reset all steps
    const steps = document.querySelectorAll('.step');
    steps.forEach(step => {
        step.classList.remove('active', 'completed');
    });
}

// Show results
function showResults(jobData) {
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Store job ID for download
    downloadPdfBtn.dataset.jobId = currentJobId;
    viewDataBtn.dataset.jobId = currentJobId;
    
    // Save completed job ID to localStorage for chatbot
    localStorage.setItem('lastCompletedJob', currentJobId);
    
    // Initialize chatbot for results interaction
    if (typeof initializeChatbot === 'function') {
        initializeChatbot(currentJobId);
    }

    // Fetch and display quick stats
    fetchQuickStats(currentJobId);
}

// Fetch quick statistics
async function fetchQuickStats(jobId) {
    try {
        // Temporary fix: Call Python API directly to get statistics
        // (DOTNET gateway strips the statistics field)
        const pythonUrl = `http://localhost:8000/api/results/${jobId}/data`;
        const response = await fetch(pythonUrl);
        
        if (!response.ok) {
            console.warn('Could not fetch statistics');
            return;
        }

        const data = await response.json();
        console.log('Analysis data with statistics:', data);

        // Display statistics
        displayQuickStats(data);

    } catch (error) {
        console.error('Error fetching statistics:', error);
    }
}

// Display quick statistics
function displayQuickStats(data) {
    const statsContainer = document.getElementById('quickStats');
    
    // Extract statistics
    const stats = data.statistics || {};
    
    const html = `
        <h4>📊 Quick Statistics</h4>
        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-value">${stats.total_reviews || 0}</span>
                <span class="stat-label">Total Reviews</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" style="color: #28a745;">${stats.positive || 0}</span>
                <span class="stat-label">Positive (${Math.round((stats.positive || 0) / (stats.total_reviews || 1) * 100)}%)</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" style="color: #dc3545;">${stats.negative || 0}</span>
                <span class="stat-label">Negative (${Math.round((stats.negative || 0) / (stats.total_reviews || 1) * 100)}%)</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" style="color: #6c757d;">${stats.neutral || 0}</span>
                <span class="stat-label">Neutral (${Math.round((stats.neutral || 0) / (stats.total_reviews || 1) * 100)}%)</span>
            </div>
        </div>
    `;
    
    statsContainer.innerHTML = html;
}

// Download PDF
async function downloadPdf() {
    const jobId = downloadPdfBtn.dataset.jobId;
    if (!jobId) return;

    try {
        downloadPdfBtn.disabled = true;
        downloadPdfBtn.innerHTML = '<span class="button-icon">⏳</span> Downloading...';

        const response = await fetch(`${API_BASE_URL}/results/${jobId}/pdf`);
        
        if (!response.ok) {
            throw new Error('Failed to download PDF');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sentiment_report_${jobId}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        downloadPdfBtn.disabled = false;
        downloadPdfBtn.innerHTML = '<span class="button-icon">📄</span> Download PDF Report';

    } catch (error) {
        console.error('Error downloading PDF:', error);
        alert('Failed to download PDF. Please try again.');
        downloadPdfBtn.disabled = false;
        downloadPdfBtn.innerHTML = '<span class="button-icon">📄</span> Download PDF Report';
    }
}

// View JSON data
async function viewJsonData() {
    const jobId = viewDataBtn.dataset.jobId;
    if (!jobId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/results/${jobId}/data`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch data');
        }

        const data = await response.json();
        
        // Open in new window with formatted JSON
        const newWindow = window.open('', '_blank');
        newWindow.document.write(`
            <html>
            <head>
                <title>Analysis Data - Job ${jobId}</title>
                <style>
                    body { 
                        font-family: 'Courier New', monospace; 
                        padding: 20px; 
                        background: #1e1e1e; 
                        color: #d4d4d4;
                    }
                    pre { 
                        white-space: pre-wrap; 
                        word-wrap: break-word; 
                    }
                </style>
            </head>
            <body>
                <h2>Analysis Results - Job ${jobId}</h2>
                <pre>${JSON.stringify(data, null, 2)}</pre>
            </body>
            </html>
        `);

    } catch (error) {
        console.error('Error viewing data:', error);
        alert('Failed to load data. Please try again.');
    }
}

// Reset form for new analysis
function resetForm() {
    // Reset form
    analysisForm.reset();
    analysisForm.style.display = 'block';
    
    // Hide other sections
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Enable submit button
    submitBtn.disabled = false;
    
    // Clear job ID
    currentJobId = null;
    
    // Stop polling
    stopPolling();
    
    // Reset search method to demo
    radioDemo.checked = true;
    handleSearchMethodChange();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show error
function showError(message) {
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'block';
    
    document.getElementById('errorText').textContent = message;
    
    submitBtn.disabled = false;
    stopPolling();
}

// Validate email (supports single or comma-separated emails)
function validateEmail(emailString) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    
    // Check if it contains commas (multiple emails)
    if (emailString.includes(',')) {
        const emails = emailString.split(',').map(e => e.trim());
        // Validate each email
        return emails.every(email => emailRegex.test(email)) && emails.length > 0;
    }
    
    // Single email validation
    return emailRegex.test(emailString);
}

// Log frontend version
console.log('🎯 Sentiment Analysis Frontend v1.0 - Ready!');
console.log('API Base URL:', API_BASE_URL);

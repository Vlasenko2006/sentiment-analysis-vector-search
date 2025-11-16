// Chatbot functionality for Results interaction
// Note: API_BASE_URL is defined in app.js
// This module extends the main app with chatbot functionality

let chatbotJobId = null;
let chatbotInitialized = false;
let sendButtonClickCount = 0;
let sendButtonClickTimes = [];

// Initialize chatbot - can be called on page load or after analysis
function initializeChatbot(jobId) {
    window.debugLogger?.log('CHATBOT', 'initializeChatbot called', { jobId, previousJobId: chatbotJobId });
    
    chatbotJobId = jobId;
    
    // Show the toggle button
    const toggleBtn = document.getElementById('toggleChatBtn');
    if (toggleBtn) {
        toggleBtn.style.display = 'flex';
        window.debugLogger?.log('CHATBOT', 'Toggle button shown', { jobId });
    } else {
        window.debugLogger?.log('CHATBOT', 'Toggle button NOT FOUND', { jobId });
    }
    
    // Load suggested questions if we have a job ID
    if (jobId) {
        loadSuggestedQuestions(jobId);
    } else {
        window.debugLogger?.log('CHATBOT', 'No job ID - skipping suggestions load', {});
    }
    
    // Setup event listeners only once
    if (!chatbotInitialized) {
        setupChatbotListeners();
        chatbotInitialized = true;
        window.debugLogger?.log('CHATBOT', 'Event listeners initialized', {});
    }
}

// Initialize chatbot on page load (always show button)
document.addEventListener('DOMContentLoaded', () => {
    window.debugLogger?.log('CHATBOT', 'DOMContentLoaded - checking localStorage', {});
    
    // Check if there's a recent completed job in localStorage
    const recentJobId = localStorage.getItem('lastCompletedJob');
    window.debugLogger?.log('CHATBOT', 'localStorage check', { recentJobId });
    
    if (recentJobId) {
        initializeChatbot(recentJobId);
    } else {
        // Still show button, user can enter job ID manually
        initializeChatbot(null);
    }
});

// Setup all chatbot event listeners
function setupChatbotListeners() {
    window.debugLogger?.log('CHATBOT', 'Setting up event listeners', {});
    
    const toggleBtn = document.getElementById('toggleChatBtn');
    const closeBtn = document.getElementById('closeChatBtn');
    const sendBtn = document.getElementById('sendChatBtn');
    const chatInput = document.getElementById('chatInput');
    
    if (!toggleBtn || !closeBtn || !sendBtn || !chatInput) {
        window.debugLogger?.log('CHATBOT', 'ERROR: Missing DOM elements', {
            toggleBtn: !!toggleBtn,
            closeBtn: !!closeBtn,
            sendBtn: !!sendBtn,
            chatInput: !!chatInput
        });
        return;
    }
    
    // Toggle chatbot visibility
    toggleBtn.addEventListener('click', () => {
        const widget = document.getElementById('chatbotWidget');
        const isVisible = widget.style.display === 'flex';
        widget.style.display = isVisible ? 'none' : 'flex';
        
        window.debugLogger?.log('CHATBOT', 'Toggle button clicked', { 
            isVisible: !isVisible,
            currentJobId: chatbotJobId 
        });
        
        if (!isVisible) {
            chatInput.focus();
        }
    });
    
    // Close chatbot
    closeBtn.addEventListener('click', () => {
        document.getElementById('chatbotWidget').style.display = 'none';
        window.debugLogger?.log('CHATBOT', 'Chat closed', {});
    });
    
    // Send message on button click
    sendBtn.addEventListener('click', () => {
        sendButtonClickCount++;
        const clickTime = new Date().toISOString();
        sendButtonClickTimes.push(clickTime);
        
        window.debugLogger?.log('CHATBOT', '🔘 SEND BUTTON CLICKED', { 
            clickNumber: sendButtonClickCount,
            clickTime: clickTime,
            totalClicks: sendButtonClickCount,
            allClickTimes: sendButtonClickTimes,
            currentJobId: chatbotJobId,
            path: 'user→sendButton'
        });
        
        sendMessage();
    });
    
    // Send message on Enter key
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendButtonClickCount++;
            const clickTime = new Date().toISOString();
            sendButtonClickTimes.push(clickTime);
            
            window.debugLogger?.log('CHATBOT', '⌨️ ENTER KEY PRESSED (Send)', { 
                clickNumber: sendButtonClickCount,
                clickTime: clickTime,
                totalClicks: sendButtonClickCount,
                allClickTimes: sendButtonClickTimes,
                currentJobId: chatbotJobId,
                path: 'user→enterKey'
            });
            
            sendMessage();
        }
    });
    
    // Handle suggested question clicks
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('suggestion-chip')) {
            const question = e.target.textContent;
            chatInput.value = question;
            window.debugLogger?.log('CHATBOT', 'Suggestion clicked', { question, currentJobId: chatbotJobId });
            sendMessage();
        }
    });
}

// Send message to chatbot
async function sendMessage() {
    window.debugLogger?.log('CHATBOT', 'sendMessage called', { 
        jobId: chatbotJobId,
        hasJobId: !!chatbotJobId 
    });
    
    const chatInput = document.getElementById('chatInput');
    const question = chatInput.value.trim();
    
    window.debugLogger?.log('CHATBOT', '← INPUT: User Question', { 
        question, 
        jobId: chatbotJobId,
        hasJobId: !!chatbotJobId,
        path: 'user→chatbot.js'
    });
    
    if (!question) {
        window.debugLogger?.log('ERROR', 'Empty Question', { path: 'chatbot validation' });
        return;
    }
    
    // Check if we have a job ID
    if (!chatbotJobId) {
        window.debugLogger?.log('ERROR', 'No Job ID Available', { 
            question, 
            localStorage: localStorage.getItem('lastCompletedJob'),
            path: 'chatbot validation' 
        });
        addMessageToChat('Please complete an analysis first before using the chatbot.', 'assistant');
        return;
    }
    
    // Clear input and disable while processing
    chatInput.value = '';
    chatInput.disabled = true;
    document.getElementById('sendChatBtn').disabled = true;
    
    window.debugLogger?.log('CHATBOT', 'Input cleared, button disabled', { 
        path: 'sendMessage' 
    });
    
    // Add user message to chat
    addMessageToChat(question, 'user');
    
    window.debugLogger?.log('CHATBOT', 'User message added, creating typing indicator', { 
        path: 'sendMessage' 
    });
    
    // Show typing indicator
    const typingIndicator = addTypingIndicator();
    
    window.debugLogger?.log('CHATBOT', 'Typing indicator created, preparing API call', { 
        hasIndicator: !!typingIndicator,
        path: 'sendMessage' 
    });
    
    try {
        const apiUrl = `${API_BASE_URL}/results/${chatbotJobId}/chat`;
        const requestBody = {
            question: question,
            include_history: true
        };
        
        window.debugLogger?.log('CHATBOT', '→ OUTPUT: API Request', { 
            url: apiUrl, 
            method: 'POST',
            body: requestBody,
            jobId: chatbotJobId,
            path: 'chatbot.js→.NET API'
        });
        
        // Send to API
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        window.debugLogger?.log('CHATBOT', '← INPUT: API Response Status', { 
            status: response.status,
            ok: response.ok,
            statusText: response.statusText,
            path: '.NET API→chatbot.js'
        });
        
        if (response.ok) {
            const data = await response.json();
            
            window.debugLogger?.log('CHATBOT', '← INPUT: API Response Data', { 
                answer: data.answer?.substring(0, 100) + '...', 
                hasSuggestions: !!data.suggested_questions,
                data,
                path: '.NET API→chatbot.js'
            });
            
            // Remove typing indicator
            typingIndicator.remove();
            
            window.debugLogger?.log('CHATBOT', '→ OUTPUT: Display Answer', { 
                answerLength: data.answer?.length,
                path: 'chatbot.js→DOM'
            });
            
            // Add assistant response
            addMessageToChat(data.answer, 'assistant');
            
            // Update suggested questions if provided
            if (data.suggested_questions) {
                displaySuggestedQuestions(data.suggested_questions);
            }
        } else {
            const errorText = await response.text();
            window.debugLogger?.log('ERROR', 'Chatbot API Error', { 
                status: response.status,
                errorText,
                path: '.NET API→chatbot.js'
            });
            
            let errorMessage = 'An error occurred';
            try {
                const errorJson = JSON.parse(errorText);
                errorMessage = errorJson.detail || errorJson.error || errorJson.message || errorMessage;
            } catch (e) {
                errorMessage = errorText || errorMessage;
            }
            
            window.debugLogger?.log('CHATBOT', 'Parsed error message', { errorMessage });
            
            typingIndicator.remove();
            addMessageToChat(`Sorry, I encountered an error: ${errorMessage}`, 'assistant');
        }
    } catch (error) {
        window.debugLogger?.log('CHATBOT', 'Exception in sendMessage', { 
            error: error.message,
            stack: error.stack 
        });
        
        console.error('Chat error:', error);
        typingIndicator.remove();
        addMessageToChat('Sorry, I could not process your question. Please try again.', 'assistant');
    } finally {
        // Re-enable input
        chatInput.disabled = false;
        document.getElementById('sendChatBtn').disabled = false;
        chatInput.focus();
    }
}

// Load suggested questions for a job
async function loadSuggestedQuestions(jobId) {
    window.debugLogger?.log('CHATBOT', 'Loading suggestions', { jobId });
    
    try {
        const apiUrl = `${API_BASE_URL}/results/${jobId}/chat/suggestions`;
        window.debugLogger?.log('CHATBOT', 'Fetching suggestions', { url: apiUrl });
        
        const response = await fetch(apiUrl);
        
        window.debugLogger?.log('CHATBOT', 'Suggestions response', { 
            status: response.status,
            ok: response.ok 
        });
        
        if (response.ok) {
            const data = await response.json();
            window.debugLogger?.log('CHATBOT', 'Suggestions received', { data });
            
            if (data.suggestions && data.suggestions.length > 0) {
                displaySuggestedQuestions(data.suggestions);
            }
        }
    } catch (error) {
        window.debugLogger?.log('CHATBOT', 'Error loading suggestions', { 
            error: error.message 
        });
        console.error('Error loading suggestions:', error);
    }
}

// Display suggested questions
function displaySuggestedQuestions(suggestions) {
    window.debugLogger?.log('CHATBOT', 'Displaying suggestions', { 
        count: suggestions.length,
        suggestions 
    });
    
    const container = document.getElementById('suggestedQuestions');
    if (!container) return;
    
    container.innerHTML = '';
    suggestions.slice(0, 6).forEach(suggestion => {
        const chip = document.createElement('button');
        chip.className = 'suggestion-chip';
        chip.textContent = suggestion;
        container.appendChild(chip);
    });
}

// Add message to chat UI
function addMessageToChat(text, sender) {
    window.debugLogger?.log('CHATBOT', 'Adding message to chat', { 
        sender, 
        textLength: text.length 
    });
    
    const messagesContainer = document.getElementById('chatMessages');
    
    if (!messagesContainer) {
        window.debugLogger?.log('ERROR', 'Chat container not found', { 
            elementId: 'chatMessages',
            path: 'addMessageToChat' 
        });
        console.error('CHATBOT ERROR: Element with id "chatMessages" not found!');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message message-${sender}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    
    messageDiv.appendChild(bubble);
    messagesContainer.appendChild(messageDiv);
    
    window.debugLogger?.log('CHATBOT', '→ OUTPUT: Message added to DOM', { 
        sender,
        path: 'addMessageToChat→DOM' 
    });
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Add typing indicator
function addTypingIndicator() {
    window.debugLogger?.log('CHATBOT', 'Adding typing indicator', {});
    
    const messagesContainer = document.getElementById('chatMessages');
    
    if (!messagesContainer) {
        window.debugLogger?.log('ERROR', 'Chat container not found', { 
            elementId: 'chatMessages',
            path: 'addTypingIndicator' 
        });
        console.error('CHATBOT ERROR: Element with id "chatMessages" not found!');
        return null;
    }
    
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'chat-message message-assistant';
    indicatorDiv.id = 'typingIndicator';
    
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    
    indicatorDiv.appendChild(indicator);
    messagesContainer.appendChild(indicatorDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return indicatorDiv;
}

// Debug Logger for Frontend
const DEBUG = true;

function debugLog(component, action, data) {
    if (!DEBUG) return;
    
    const timestamp = new Date().toISOString();
    const logEntry = {
        timestamp,
        component,
        action,
        data
    };
    
    console.log(`[${timestamp}] [${component}] ${action}:`, data);
    
    // Send to backend for logging
    fetch('http://localhost:5001/api/debug/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logEntry)
    }).catch(e => console.error('Failed to send debug log:', e));
}

window.debugLog = debugLog;

// Debug Logger - Centralized logging with INPUT/OUTPUT tracking
class DebugLogger {
    constructor() {
        this.logs = [];
        this.componentId = 'FRONTEND-DEBUGLOGGER';
        this.programName = 'debug-logger.js';
        this.enabled = true;
        this.logToFile = true;
        console.log(`%c[${this.componentId}] ✓ Initialized - Program: ${this.programName}`, 'color: #06D6A0; font-weight: bold');
    }

    log(component, action, data) {
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            componentId: `FRONTEND-${component}`,
            programName: this.getProgramName(component),
            component,
            action,
            data: this.sanitizeData(data),
            path: data?.path || 'unknown'
        };
        
        this.logs.push(logEntry);
        
        // Console log with color coding and INPUT/OUTPUT prefix
        const color = this.getComponentColor(component);
        const prefix = action.includes('OUTPUT') || action.includes('→') ? '→ OUTPUT' : 
                      action.includes('INPUT') || action.includes('←') ? '← INPUT' : '';
        console.log(
            `%c[${logEntry.componentId}] ${prefix} ${action}`,
            `color: ${color}; font-weight: bold;`,
            data
        );
        
        // Send to server for file logging
        if (this.logToFile) {
            this.sendToServer(logEntry);
        }
    }

    getProgramName(component) {
        const programMap = {
            'APP': 'app.js',
            'CHATBOT': 'chatbot.js',
            'API': 'API Request',
            'ERROR': 'Error Handler',
            'PROGRESS': 'Progress Bar'
        };
        return programMap[component] || 'frontend';
    }

    sanitizeData(data) {
        try {
            return JSON.parse(JSON.stringify(data));
        } catch (e) {
            return String(data);
        }
    }

    getComponentColor(component) {
        const colors = {
            'APP': '#06D6A0',
            'CHATBOT': '#9D4EDD',
            'API': '#FF6B35',
            'ERROR': '#EF476F',
            'PROGRESS': '#118AB2'
        };
        return colors[component] || '#4a5568';
    }

    async sendToServer(logEntry) {
        try {
            const response = await fetch('http://localhost:5000/api/debug/log', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(logEntry)
            });

            if (!response.ok) {
                console.error(`%c[${this.componentId}] ERROR: Server returned ${response.status}`, 
                    'color: #EF476F; font-weight: bold',
                    { path: 'frontend→.NET API', status: response.status });
            }
        } catch (error) {
            console.error(`%c[${this.componentId}] ERROR: Failed to send log to server`, 
                'color: #EF476F; font-weight: bold',
                { error: error.message, path: 'frontend→.NET API' });
        }
    }

    downloadLogs() {
        const dataStr = JSON.stringify(this.logs, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `debug-logs-${Date.now()}.json`;
        link.click();
        console.log(`%c[${this.componentId}] Downloaded ${this.logs.length} logs`, 'color: #06D6A0; font-weight: bold');
    }

    clear() {
        this.logs = [];
        console.clear();
        console.log(`%c[${this.componentId}] Logs cleared`, 'color: #06D6A0; font-weight: bold');
    }
}

// Global debug logger instance
window.debugLogger = new DebugLogger();

// Add download button to page
document.addEventListener('DOMContentLoaded', () => {
    const debugBtn = document.createElement('button');
    debugBtn.innerHTML = '📊 Debug';
    debugBtn.style.cssText = 'position:fixed;top:10px;right:10px;z-index:9999;background:#333;color:white;border:none;padding:8px 12px;border-radius:5px;cursor:pointer;font-size:12px;';
    debugBtn.onclick = () => window.debugLogger.downloadLogs();
    document.body.appendChild(debugBtn);
    
    console.log('%c[FRONTEND-DEBUGLOGGER] Download button added to page', 'color: #06D6A0; font-weight: bold');
});

// Simple connection test script
console.log('=== CONNECTION TEST STARTING ===');

// Test 1: Health endpoint
fetch('http://localhost:5001/health')
    .then(r => {
        console.log('✅ Health endpoint reachable:', r.status);
        return r.json();
    })
    .then(data => console.log('Health data:', data))
    .catch(e => console.error('❌ Health endpoint failed:', e.message));

// Test 2: Analyze endpoint
setTimeout(() => {
    fetch('http://localhost:5001/api/sentiment/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: 'demo', searchMethod: 'demo', email: 'test@test.com' })
    })
    .then(r => {
        console.log('✅ Analyze endpoint reachable:', r.status);
        return r.json();
    })
    .then(data => {
        console.log('Analyze response:', data);
        // Test 3: Status check
        if (data.job_id) {
            return fetch(`http://localhost:5001/api/sentiment/status/${data.job_id}`);
        }
    })
    .then(r => r ? r.json() : null)
    .then(data => data ? console.log('Status check:', data) : null)
    .catch(e => console.error('❌ Analyze/Status failed:', e.message));
}, 1000);

console.log('API_BASE_URL should be: http://localhost:5001/api/sentiment');

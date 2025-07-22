// XAI Scoring Framework JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Set up event listeners
    setupFormHandlers();
    setupRangeSliders();
    setupChatInterface();
    
    // Check data status on load
    checkDataStatus();
}

function setupFormHandlers() {
    // Load data button handler
    const loadDataBtn = document.getElementById('loadDataBtn');
    if (loadDataBtn) {
        loadDataBtn.addEventListener('click', handleLoadData);
    }
    
    // Scoring form handler
    const scoringForm = document.getElementById('scoringForm');
    if (scoringForm) {
        scoringForm.addEventListener('submit', handleDatasetScoring);
    }
}

function setupRangeSliders() {
    // Update range slider values display
    const rangeSliders = document.querySelectorAll('input[type="range"]');
    rangeSliders.forEach(slider => {
        slider.addEventListener('input', function() {
            updateRangeValue(this);
        });
        // Initialize values
        updateRangeValue(slider);
    });
}

function updateRangeValue(slider) {
    const value = slider.value;
    const label = slider.previousElementSibling;
    if (label && label.tagName === 'LABEL') {
        const currentText = label.textContent;
        const baseText = currentText.replace(/:\s*\d+\.?\d*/, '');
        label.textContent = `${baseText}: ${parseFloat(value).toFixed(2)}`;
    }
}

function setupChatInterface() {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    
    if (chatForm) {
        chatForm.addEventListener('submit', handleChatSubmit);
    }
    
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleChatSubmit(e);
            }
        });
    }
}

async function handleLoadData() {
    const loadDataBtn = document.getElementById('loadDataBtn');
    const statusDiv = document.getElementById('loadDataStatus');
    const dataStatusDiv = document.getElementById('dataStatus');
    
    // Show loading state
    loadDataBtn.disabled = true;
    loadDataBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
    showStatus(statusDiv, 'Loading benchmark data...', 'info');
    
    try {
        const response = await fetch('/load-data', {
            method: 'GET'
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showStatus(statusDiv, data.message, 'success');
            updateDataStatus(true);
            enableScoringSection();
            
            // Update button
            loadDataBtn.innerHTML = '<i class="fas fa-check me-2"></i>Data Loaded';
            loadDataBtn.className = 'btn btn-success btn-lg';
        } else {
            showStatus(statusDiv, data.error || 'Failed to load data', 'danger');
            updateDataStatus(false);
            
            // Reset button
            loadDataBtn.disabled = false;
            loadDataBtn.innerHTML = '<i class="fas fa-download me-2"></i>Load Benchmark Data';
        }
    } catch (error) {
        showStatus(statusDiv, 'An error occurred while loading data.', 'danger');
        updateDataStatus(false);
        
        // Reset button
        loadDataBtn.disabled = false;
        loadDataBtn.innerHTML = '<i class="fas fa-download me-2"></i>Load Benchmark Data';
        console.error('Load data error:', error);
    }
}

async function checkDataStatus() {
    try {
        const response = await fetch('/data-status');
        const data = await response.json();
        
        if (data.data_loaded) {
            updateDataStatus(true);
            enableScoringSection();
            
            // Update button if data is already loaded
            const loadDataBtn = document.getElementById('loadDataBtn');
            if (loadDataBtn) {
                loadDataBtn.innerHTML = '<i class="fas fa-check me-2"></i>Data Loaded';
                loadDataBtn.className = 'btn btn-success btn-lg';
                loadDataBtn.disabled = true;
            }
        } else {
            updateDataStatus(false);
        }
    } catch (error) {
        console.error('Error checking data status:', error);
        updateDataStatus(false);
    }
}

function updateDataStatus(loaded) {
    const dataStatusDiv = document.getElementById('dataStatus');
    if (dataStatusDiv) {
        if (loaded) {
            dataStatusDiv.innerHTML = '<span class="badge bg-success"><i class="fas fa-check me-1"></i>Data Ready</span>';
        } else {
            dataStatusDiv.innerHTML = '<span class="badge bg-warning"><i class="fas fa-exclamation-triangle me-1"></i>Data Not Loaded</span>';
        }
    }
}

async function handleDatasetScoring(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const statusDiv = document.getElementById('scoringStatus');
    const resultsDiv = document.getElementById('scoringResults');
    
    // Show loading state
    showStatus(statusDiv, 'Processing dataset and generating scores...', 'info');
    resultsDiv.innerHTML = '';
    
    try {
        const response = await fetch('/score_dataset', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showStatus(statusDiv, 'Scoring completed successfully!', 'success');
            // Store the full response for recommendations
            window.scoringResponse = data;
            displayScoringResults(data.results, data.features);
        } else {
            showStatus(statusDiv, data.error, 'danger');
        }
    } catch (error) {
        showStatus(statusDiv, 'An error occurred while scoring the dataset.', 'danger');
        console.error('Scoring error:', error);
    }
}

async function handleChatSubmit(event) {
    event.preventDefault();
    
    const chatInput = document.getElementById('chatInput');
    const question = chatInput.value.trim();
    
    if (!question) return;
    
    // Add user message to chat
    addChatMessage(question, 'user');
    chatInput.value = '';
    
    // Show typing indicator
    const typingIndicator = addTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        typingIndicator.remove();
        
        if (response.ok) {
            addChatMessage(data.response, 'assistant');
        } else {
            addChatMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        }
    } catch (error) {
        typingIndicator.remove();
        addChatMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        console.error('Chat error:', error);
    }
}

function addChatMessage(message, type) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = message;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing-indicator';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<i class="fas fa-ellipsis-h"></i> Typing...';
    
    typingDiv.appendChild(contentDiv);
    chatMessages.appendChild(typingDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingDiv;
}

function showStatus(element, message, type) {
    element.innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
}

function enableScoringSection() {
    const scoreBtn = document.getElementById('scoreBtn');
    if (scoreBtn) {
        scoreBtn.disabled = false;
    }
}

function displayScoringResults(results, features) {
    const resultsDiv = document.getElementById('scoringResults');
    
    console.log('Results:', results);
    console.log('Features:', features);
    
    if (!results || Object.keys(results).length === 0) {
        resultsDiv.innerHTML = '<div class="alert alert-warning">No results available.</div>';
        return;
    }
    
    let html = `
        <div class="results-card">
            <h4><i class="fas fa-chart-line me-2"></i>Scoring Results</h4>
            
            <div class="row">
                <div class="col-md-6">
                    <h5>Dataset Features</h5>
                    <div class="metric-item">
                        <div class="metric-label">Number of Features</div>
                        <div class="metric-value">${features.feature_count || 'N/A'}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Number of Samples</div>
                        <div class="metric-value">${features.size || 'N/A'}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Numeric Features</div>
                        <div class="metric-value">${features.numeric_features || 'N/A'}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Categorical Features</div>
                        <div class="metric-value">${features.cat_features || 'N/A'}</div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <h5>XAI Method Scores</h5>
    `;
    
    // Display XAI method scores
    if (results && typeof results === 'object') {
        const methods = Object.keys(results).filter(key => key !== 'recommended_method' && key !== 'recommended_ai' && key !== 'similar_datasets');
        
        methods.forEach((method, index) => {
            const methodData = results[method];
            if (methodData && typeof methodData === 'object') {
                html += `
                    <div class="metric-item">
                        <div class="metric-label">${index + 1}. ${method}</div>
                        <div class="metric-value">Score: ${methodData.overall_score ? methodData.overall_score.toFixed(3) : 'N/A'}</div>
                        <small class="text-muted">
                            Fidelity: ${methodData.avg_fidelity ? methodData.avg_fidelity.toFixed(3) : 'N/A'} | 
                            Stability: ${methodData.avg_stability ? methodData.avg_stability.toFixed(3) : 'N/A'} | 
                            Rating: ${methodData.avg_rating ? methodData.avg_rating.toFixed(3) : 'N/A'}
                        </small>
                    </div>
                `;
            }
        });
    }
    
    // Add recommendations if available
    if (window.scoringResponse && window.scoringResponse.recommended_method) {
        html += `
            <div class="alert alert-success mt-3">
                <h6><i class="fas fa-star me-2"></i>Recommended XAI Method</h6>
                <strong>${window.scoringResponse.recommended_method}</strong>
            </div>
        `;
    }
    
    if (window.scoringResponse && window.scoringResponse.recommended_ai) {
        html += `
            <div class="alert alert-info mt-2">
                <h6><i class="fas fa-robot me-2"></i>Recommended AI Model</h6>
                <strong>${window.scoringResponse.recommended_ai}</strong>
            </div>
        `;
    }
    
    html += `
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-12">
                    <h5>Detailed XAI Method Metrics</h5>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>XAI Method</th>
                                    <th>Fidelity</th>
                                    <th>Stability</th>
                                    <th>User Rating</th>
                                    <th>Simplicity</th>
                                    <th>Domain Bonus</th>
                                    <th>Overall Score</th>
                                </tr>
                            </thead>
                            <tbody>
    `;
    
    // Display detailed metrics table
    if (results && typeof results === 'object') {
        const methods = Object.keys(results).filter(key => key !== 'recommended_method' && key !== 'recommended_ai' && key !== 'similar_datasets');
        
        methods.forEach(method => {
            const methodData = results[method];
            if (methodData && typeof methodData === 'object') {
                html += `
                    <tr>
                        <td><strong>${method}</strong></td>
                        <td>${methodData.avg_fidelity ? methodData.avg_fidelity.toFixed(3) : 'N/A'}</td>
                        <td>${methodData.avg_stability ? methodData.avg_stability.toFixed(3) : 'N/A'}</td>
                        <td>${methodData.avg_rating ? methodData.avg_rating.toFixed(3) : 'N/A'}</td>
                        <td>${methodData.avg_simplicity ? methodData.avg_simplicity.toFixed(3) : 'N/A'}</td>
                        <td>${methodData.domain_bonus ? methodData.domain_bonus.toFixed(3) : 'N/A'}</td>
                        <td><strong>${methodData.overall_score ? methodData.overall_score.toFixed(3) : 'N/A'}</strong></td>
                    </tr>
                `;
            }
        });
    }
    
    html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
}

// Utility functions
function formatNumber(num) {
    return parseFloat(num).toFixed(3);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
}); 
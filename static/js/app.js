// XAI Scoring Framework JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    setupFormHandlers();
    setupRangeSliders();
    setupChatInterface();
    checkDataStatus().then(function(loaded) {
        if (!loaded) { handleLoadData(); }
    });
}

function setupFormHandlers() {
    // Load data button handler
    const loadDataBtn = document.getElementById('loadDataBtn');
    if (loadDataBtn) {
        loadDataBtn.addEventListener('click', handleLoadData);
    }

    // Data type change: fetch datasets and update file accept
    const dataTypeSelect = document.getElementById('data_type');
    if (dataTypeSelect) {
        dataTypeSelect.addEventListener('change', function() {
            fetchDatasets(this.value);
            updateFileAccept(this.value);
            toggleMetadataSections(this.value);
        });
    }

    // Input mode toggle: upload vs benchmark
    document.querySelectorAll('input[name="input_mode"]').forEach(function(radio) {
        radio.addEventListener('change', function() {
            toggleInputMode(this.value);
        });
    });

    // Initial state
    toggleInputMode(document.querySelector('input[name="input_mode"]:checked')?.value || 'upload');
    updateFileAccept(document.getElementById('data_type')?.value || 'tabular');
    toggleMetadataSections(document.getElementById('data_type')?.value || 'tabular');

    // Scoring form handler
    const scoringForm = document.getElementById('scoringForm');
    if (scoringForm) {
        scoringForm.addEventListener('submit', handleDatasetScoring);
    }
}

function toggleInputMode(mode) {
    const uploadSection = document.getElementById('uploadSection');
    const benchmarkSection = document.getElementById('benchmarkSection');
    const fileInput = document.getElementById('file_input');
    const datasetSelect = document.getElementById('dataset_id');
    if (mode === 'upload') {
        if (uploadSection) uploadSection.style.display = '';
        if (benchmarkSection) benchmarkSection.style.display = 'none';
        if (fileInput) fileInput.required = true;
        if (datasetSelect) datasetSelect.required = false;
    } else {
        if (uploadSection) uploadSection.style.display = 'none';
        if (benchmarkSection) benchmarkSection.style.display = '';
        if (fileInput) fileInput.required = false;
        if (datasetSelect) datasetSelect.required = true;
    }
}

function updateFileAccept(dataType) {
    const fileInput = document.getElementById('file_input');
    if (!fileInput) return;
    const help = document.getElementById('fileHelp');
    if (dataType === 'image') {
        fileInput.accept = '.zip,.csv';
        if (help) help.textContent = 'ZIP of images (.png, .jpg, .jpeg) or CSV with image paths';
    } else if (dataType === 'tabular') {
        fileInput.accept = '.csv,.xlsx,.xls';
        if (help) help.textContent = 'CSV or Excel (.csv, .xlsx, .xls)';
    } else {
        fileInput.accept = '.csv,.txt';
        if (help) help.textContent = 'CSV file (text: one text column; timeseries: time series columns)';
    }
}

function toggleMetadataSections(dataType) {
    const imgSection = document.getElementById('imageMetadataSection');
    const tsSection = document.getElementById('timeseriesMetadataSection');
    if (imgSection) imgSection.style.display = dataType === 'image' ? '' : 'none';
    if (tsSection) tsSection.style.display = dataType === 'timeseries' ? '' : 'none';
}

async function fetchDatasets(dataType) {
    const select = document.getElementById('dataset_id');
    if (!select) return;
    select.innerHTML = '<option value="">-- Loading --</option>';
    try {
        const response = await fetch(`/datasets?data_type=${encodeURIComponent(dataType || 'tabular')}`);
        const datasets = await response.json();
        select.innerHTML = '<option value="">-- Select dataset --</option>';
        (datasets || []).forEach(function(d) {
            const opt = document.createElement('option');
            opt.value = d.dataset_id;
            opt.textContent = (d.dataset_name || d.dataset_id) + (d.domain ? ' (' + d.domain + ')' : '');
            select.appendChild(opt);
        });
    } catch (e) {
        select.innerHTML = '<option value="">-- Error loading --</option>';
        console.error('fetchDatasets error:', e);
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
            fetchDatasets(document.getElementById('data_type')?.value || 'tabular');
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
            fetchDatasets(document.getElementById('data_type')?.value || 'tabular');
            const loadDataBtn = document.getElementById('loadDataBtn');
            if (loadDataBtn) {
                loadDataBtn.innerHTML = '<i class="fas fa-check me-2"></i>Data Loaded';
                loadDataBtn.className = 'btn btn-success btn-lg';
                loadDataBtn.disabled = true;
            }
            return true;
        } else {
            updateDataStatus(false);
            return false;
        }
    } catch (error) {
        console.error('Error checking data status:', error);
        updateDataStatus(false);
        return false;
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

    const mode = document.querySelector('input[name="input_mode"]:checked')?.value || 'upload';
    const fileInput = document.getElementById('file_input');
    const datasetSelect = document.getElementById('dataset_id');

    if (mode === 'upload' && (!fileInput || !fileInput.files || !fileInput.files.length)) {
        showStatus(document.getElementById('scoringStatus'), 'Please select a file to upload.', 'danger');
        return;
    }
    if (mode === 'benchmark' && (!datasetSelect || !datasetSelect.value)) {
        showStatus(document.getElementById('scoringStatus'), 'Please select a benchmark dataset.', 'danger');
        return;
    }

    const formData = new FormData(event.target);
    const statusDiv = document.getElementById('scoringStatus');
    const resultsDiv = document.getElementById('scoringResults');

    // Show loading state
    showStatus(statusDiv, mode === 'upload' ? 'Extracting features and generating scores...' : 'Processing dataset and generating scores...', 'info');
    resultsDiv.innerHTML = '';

    try {
        const response = await fetch('/score_dataset', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            if (data.message) {
                showStatus(statusDiv, data.message, 'warning');
            } else {
                showStatus(statusDiv, 'Scoring completed successfully!', 'success');
            }
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
    
    features = features || {};
    results = results || {};
    const hasResults = Object.keys(results).length > 0;

    function featureItem(label, value) {
        const v = value !== undefined && value !== null ? value : 'N/A';
        return `<div class="metric-item"><div class="metric-label">${label}</div><div class="metric-value">${v}</div></div>`;
    }

    let featureHtml = '<h5>Dataset Info</h5>';
    if (features.dataset_name || features.domain) {
        featureHtml += featureItem('Dataset', features.dataset_name || 'N/A');
        featureHtml += featureItem('Domain', features.domain || 'N/A');
        featureHtml += featureItem('Data Type', features.data_type || 'N/A');
    }
    if (features.data_type === 'image' && (features.size || features.image_width)) {
        featureHtml += featureItem('Sample Size', features.size);
        featureHtml += featureItem('Image Size', features.image_width && features.image_height ? `${features.image_width}x${features.image_height}` : 'N/A');
        featureHtml += featureItem('Channels', features.channels);
        featureHtml += featureItem('Num Classes', features.num_classes);
    }
    if (features.data_type === 'text' && (features.size || features.avg_doc_length)) {
        featureHtml += featureItem('Sample Size', features.size);
        featureHtml += featureItem('Avg Doc Length', features.avg_doc_length);
        featureHtml += featureItem('Max Length', features.max_length);
        featureHtml += featureItem('Vocab Size', features.vocab_size);
    }
    if (features.data_type === 'timeseries' && (features.size || features.series_length)) {
        featureHtml += featureItem('Sample Size', features.size);
        featureHtml += featureItem('Series Length', features.series_length);
        featureHtml += featureItem('Num Channels', features.num_channels);
        featureHtml += featureItem('Num Classes', features.num_classes);
    }
    if (features.feature_count !== undefined || features.size !== undefined) {
        featureHtml += featureItem('Number of Features', features.feature_count);
        featureHtml += featureItem('Number of Samples', features.size);
        featureHtml += featureItem('Numeric Features', features.numeric_features);
        featureHtml += featureItem('Categorical Features', features.cat_features);
    }

    let html = `
        <div class="results-card">
            <h4><i class="fas fa-chart-line me-2"></i>Scoring Results</h4>
            ${!hasResults && window.scoringResponse && window.scoringResponse.message ? `<div class="alert alert-warning">${window.scoringResponse.message}</div>` : ''}
            <div class="row">
                <div class="col-md-6">
                    ${featureHtml}
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
    
    // Display similar datasets used for recommendation
    if (window.scoringResponse && window.scoringResponse.similar_datasets && window.scoringResponse.similar_datasets.length > 0) {
        html += `
            <div class="mt-3">
                <h6><i class="fas fa-database me-2"></i>Similar Benchmark Datasets</h6>
                <p class="text-muted small">Datasets used to compute recommendations:</p>
                <table class="table table-sm table-bordered">
                    <thead><tr><th>Dataset ID</th><th>Dataset Name</th><th>Similarity Score</th></tr></thead>
                    <tbody>
        `;
        window.scoringResponse.similar_datasets.forEach(function(item) {
            const dsId = item[0], sim = item[1], name = item[2] || 'Unknown';
            html += `<tr><td>${dsId}</td><td>${name}</td><td>${typeof sim === 'number' ? sim.toFixed(4) : sim}</td></tr>`;
        });
        html += `
                    </tbody>
                </table>
            </div>
        `;
    }
    
    html += `
                </div>
            </div>
    `;

    if (hasResults && results && typeof results === 'object') {
        const methods = Object.keys(results).filter(key => key !== 'recommended_method' && key !== 'recommended_ai' && key !== 'similar_datasets');
        html += `
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
        html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += `
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
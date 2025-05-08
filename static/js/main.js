// Main JavaScript for Speech-to-Text Transcriber

// DOM Elements
const engineSelect = document.getElementById('engineSelect');
const modelSizeSelect = document.getElementById('modelSizeSelect');
const languageSelect = document.getElementById('languageSelect');
const modelSizeGroup = document.getElementById('modelSizeGroup');
const languageGroup = document.getElementById('languageGroup');
const debugOutput = document.getElementById('debugOutput');

// Recording Elements
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const resetButton = document.getElementById('resetButton');
const recordingStatus = document.getElementById('recordingStatus');
const audioPlayer = document.getElementById('audioPlayer');
const recordingSpinner = document.getElementById('recordingSpinner');
const recordingResultCard = document.getElementById('recordingResultCard');
const recordingResult = document.getElementById('recordingResult');
const recordingDownloadBtn = document.getElementById('recordingDownloadBtn');

// Upload Elements
const fileUpload = document.getElementById('fileUpload');
const uploadButton = document.getElementById('uploadButton');
const uploadSpinner = document.getElementById('uploadSpinner');
const uploadResultCard = document.getElementById('uploadResultCard');
const uploadResult = document.getElementById('uploadResult');
const uploadDownloadBtn = document.getElementById('uploadDownloadBtn');

// YouTube Elements
const youtubeUrl = document.getElementById('youtubeUrl');
const channelId = document.getElementById('channelId');
const maxVideos = document.getElementById('maxVideos');
const youtubeButton = document.getElementById('youtubeButton');
const youtubeSpinner = document.getElementById('youtubeSpinner');
const youtubeResults = document.getElementById('youtubeResults');

// Recording variables
let mediaRecorder;
let audioChunks = [];
let audioBlob;
let audioUrl;
let stream;

// Event Listeners
document.addEventListener('DOMContentLoaded', initApp);
engineSelect.addEventListener('change', handleEngineChange);
recordButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);
resetButton.addEventListener('click', resetRecording);
uploadButton.addEventListener('click', handleFileUpload);
youtubeButton.addEventListener('click', handleYoutubeTranscription);

// Initialize the application
function initApp() {
    // Set up initial UI state
    handleEngineChange();
    
    // Add debug info
    logDebug('Application initialized');
    
    // Fetch available engines and languages
    fetchEngineInfo();
}

// Fetch engine information
function fetchEngineInfo() {
    fetch('/api/engines')
        .then(response => response.json())
        .then(data => {
            // Store the language data
            window.whisperLanguages = data.whisper_languages || [];
            window.springlabLanguages = data.springlab_languages || [];
            
            // Update language dropdown with initial engine
            updateLanguageOptions();
            
            logDebug('Engine info loaded: ' + 
                window.whisperLanguages.length + ' Whisper languages, ' + 
                window.springlabLanguages.length + ' SpringLab languages');
        })
        .catch(error => {
            logDebug('Error loading engine info: ' + error.message);
        });
}

// Update language options based on selected engine
function updateLanguageOptions() {
    // Clear existing options
    languageSelect.innerHTML = '';
    
    const engine = engineSelect.value;
    let languages = [];
    
    if (engine === 'Whisper (Local)') {
        languages = window.whisperLanguages || [];
    } else { // SpringLab API
        languages = window.springlabLanguages || [];
    }
    
    // Add options to select
    languages.forEach(lang => {
        const option = document.createElement('option');
        option.value = lang.code;
        option.textContent = lang.name;
        languageSelect.appendChild(option);
    });
    
    // If no languages loaded yet, add a default option
    if (languages.length === 0) {
        const option = document.createElement('option');
        option.value = 'english';
        option.textContent = 'English';
        languageSelect.appendChild(option);
    }
}

// Handle engine change
function handleEngineChange() {
    const engine = engineSelect.value;
    
    if (engine === 'Whisper (Local)') {
        modelSizeGroup.style.display = 'block';
        languageGroup.style.display = 'block'; // Show language selection for Whisper too
        
        logDebug('Switched to Whisper (Local) engine');
    } else { // SpringLab API
        modelSizeGroup.style.display = 'none';
        languageGroup.style.display = 'block';
        
        logDebug('Switched to SpringLab API (best for Indian languages)');
    }
    
    // Update language options
    updateLanguageOptions();
}

// Start recording function
async function startRecording() {
    try {
        // Check if mediaDevices API is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("navigator.mediaDevices is not available in this environment. Please try on an HTTPS page with a browser that supports this API.");
        }
        
        audioChunks = [];
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioUrl = URL.createObjectURL(audioBlob);
            audioPlayer.src = audioUrl;
            audioPlayer.style.display = 'block';
            
            // Send to server for transcription
            transcribeRecording(audioBlob);
        };
        
        mediaRecorder.start();
        recordButton.disabled = true;
        stopButton.disabled = false;
        resetButton.disabled = true;
        
        // Show recording status with timer
        let seconds = 0;
        recordingStatus.textContent = `Recording... (${seconds}s)`;
        window.recordingTimer = setInterval(() => {
            seconds++;
            recordingStatus.textContent = `Recording... (${seconds}s)`;
        }, 1000);
        
        logDebug('Recording started');
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        recordingStatus.textContent = 'Error: Could not access microphone';
        recordingStatus.classList.add('text-danger');
        
        // Display more detailed error message
        recordingResultCard.style.display = 'block';
        recordingResult.innerHTML = `<div class="alert alert-danger">
            <strong>Microphone Error:</strong> ${error.message}<br><br>
            <strong>Solutions:</strong>
            <ul>
                <li>Make sure you're on a secure (HTTPS) connection</li>
                <li>Allow microphone permissions in your browser</li>
                <li>Try using a different browser (Chrome or Firefox recommended)</li>
                <li>If using a mobile device, try on a desktop computer</li>
            </ul>
        </div>`;
        
        logDebug('Microphone error: ' + error.message);
    }
}

// Stop recording function
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        clearInterval(window.recordingTimer);
        recordingStatus.textContent = 'Recording stopped. Transcribing...';
        recordButton.disabled = false;
        stopButton.disabled = true;
        resetButton.disabled = false;
        
        logDebug('Recording stopped');
    }
}

// Reset recording function
function resetRecording() {
    audioPlayer.style.display = 'none';
    audioPlayer.src = '';
    recordingStatus.textContent = '';
    recordButton.disabled = false;
    stopButton.disabled = true;
    resetButton.disabled = true;
    recordingResultCard.style.display = 'none';
    
    logDebug('Recording reset');
}

// Transcribe recorded audio
function transcribeRecording(blob) {
    // Show loading spinner
    recordingSpinner.style.display = 'block';
    recordingResultCard.style.display = 'none';
    
    // Get settings
    const engine = engineSelect.value;
    const modelSize = modelSizeSelect.value;
    const language = languageSelect.value;
    
    // Create form data for API request
    const formData = new FormData();
    formData.append('audio_file', blob, 'recording.wav');
    formData.append('engine', engine);
    formData.append('language', language);
    formData.append('model_size', modelSize);
    
    // Log request details
    logDebug(`Sending transcription request:
    - Engine: ${engine}
    - Model Size: ${modelSize}
    - Language: ${language}
    - Audio Size: ${Math.round(blob.size / 1024)} KB`);
    
    // Send to API
    fetch('/api/transcribe', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        // Check if response is ok
        if (!response.ok) {
            throw new Error(`Server returned status: ${response.status}`);
        }
        
        // Check content type to ensure it's JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error(`Expected JSON response but got ${contentType || 'unknown content type'}`);
        }
        
        return response.json();
    })
    .then(data => {
        // Hide spinner
        recordingSpinner.style.display = 'none';
        
        if (data.success) {
            // Show result
            recordingResultCard.style.display = 'block';
            recordingResult.innerHTML = `<pre class="mb-0">${data.text}</pre>`;
            
            // Set download link
            if (data.download_url) {
                recordingDownloadBtn.href = data.download_url;
            }
            
            logDebug('Transcription complete: ' + data.text.length + ' characters');
        } else {
            // Show error
            recordingResultCard.style.display = 'block';
            recordingResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            logDebug('Transcription error: ' + data.error);
        }
    })
    .catch(error => {
        // Hide spinner
        recordingSpinner.style.display = 'none';
        
        // Show error
        recordingResultCard.style.display = 'block';
        recordingResult.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        logDebug('API error: ' + error.message);
    });
}

// Handle file upload
function handleFileUpload() {
    // Check if file is selected
    if (!fileUpload.files || fileUpload.files.length === 0) {
        alert('Please select a file to upload');
        return;
    }
    
    const file = fileUpload.files[0];
    
    // Show loading spinner
    uploadSpinner.style.display = 'block';
    uploadResultCard.style.display = 'none';
    
    // Get settings
    const engine = engineSelect.value;
    const modelSize = modelSizeSelect.value;
    const language = languageSelect.value;
    
    // Create form data for API request
    const formData = new FormData();
    formData.append('file', file);
    formData.append('engine', engine);
    formData.append('language', language);
    formData.append('model_size', modelSize);
    
    // Log request details
    logDebug(`Uploading file:
    - Name: ${file.name}
    - Size: ${Math.round(file.size / 1024)} KB
    - Type: ${file.type}
    - Engine: ${engine}
    - Model Size: ${modelSize}
    - Language: ${language}`);
    
    // Send to API
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        // Check if response is ok
        if (!response.ok) {
            throw new Error(`Server returned status: ${response.status}`);
        }
        
        // Check content type to ensure it's JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error(`Expected JSON response but got ${contentType || 'unknown content type'}`);
        }
        
        return response.json();
    })
    .then(data => {
        // Hide spinner
        uploadSpinner.style.display = 'none';
        
        if (data.success) {
            // Show result
            uploadResultCard.style.display = 'block';
            uploadResult.innerHTML = `<pre class="mb-0">${data.text}</pre>`;
            
            // Set download link
            if (data.download_url) {
                uploadDownloadBtn.href = data.download_url;
            }
            
            logDebug('Transcription complete: ' + data.text.length + ' characters');
        } else {
            // Show error
            uploadResultCard.style.display = 'block';
            uploadResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            logDebug('Transcription error: ' + data.error);
        }
    })
    .catch(error => {
        // Hide spinner
        uploadSpinner.style.display = 'none';
        
        // Show error
        uploadResultCard.style.display = 'block';
        uploadResult.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        logDebug('API error: ' + error.message);
    });
}

// Handle YouTube transcription
function handleYoutubeTranscription() {
    // Get YouTube URL or channel
    const url = youtubeUrl.value.trim();
    const channel = channelId.value.trim();
    const max = maxVideos.value;
    
    if (!url && !channel) {
        alert('Please enter a YouTube URL or channel ID');
        return;
    }
    
    // Show loading spinner
    youtubeButton.disabled = true;
    youtubeSpinner.style.display = 'block';
    youtubeResults.innerHTML = '';
    
    // Create status element
    const statusElement = document.createElement('div');
    statusElement.className = 'alert alert-info mt-3';
    statusElement.innerHTML = `
        <div class="d-flex align-items-center">
            <strong>Initializing...</strong>
            <div class="spinner-border spinner-border-sm ms-auto" role="status" aria-hidden="true"></div>
        </div>
        <div class="progress mt-2" style="height: 5px;">
            <div class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
    `;
    youtubeResults.appendChild(statusElement);
    
    // Get settings
    const engine = engineSelect.value;
    const modelSize = modelSizeSelect.value;
    const language = languageSelect.value;
    
    // Create form data for API request
    const formData = new FormData();
    if (url) formData.append('url', url);
    if (channel) formData.append('channel_id', channel);
    formData.append('max_videos', max);
    formData.append('engine', engine);
    formData.append('language', language);
    formData.append('model_size', modelSize);
    
    // Log request details
    logDebug(`YouTube request:
    - URL: ${url || 'N/A'}
    - Channel: ${channel || 'N/A'}
    - Max Videos: ${max}
    - Engine: ${engine}
    - Model Size: ${modelSize}
    - Language: ${language}`);
    
    // If using single URL mode, start the status polling
    let statusInterval = null;
    if (url) {
        statusInterval = startStatusPolling(url, statusElement);
    }
    
    // Send to API
    fetch('/api/youtube', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        // Check if response is ok
        if (!response.ok) {
            throw new Error(`Server returned status: ${response.status}`);
        }
        
        // Check content type to ensure it's JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error(`Expected JSON response but got ${contentType || 'unknown content type'}`);
        }
        
        return response.json();
    })
    .then(data => {
        // Stop status polling
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        
        // Remove status element
        if (statusElement && statusElement.parentNode) {
            statusElement.parentNode.removeChild(statusElement);
        }
        
        // Hide spinner and enable button
        youtubeButton.disabled = false;
        youtubeSpinner.style.display = 'none';
        
        if (data.success) {
            // Show result
            if (data.text) {
                // Single video result
                const card = document.createElement('div');
                card.className = 'card mb-3';
                card.innerHTML = `
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">YouTube Video Transcription</h5>
                        <a href="${data.download_url}" class="btn btn-sm btn-primary">
                            <i class="fas fa-download"></i> Download
                        </a>
                    </div>
                    <div class="card-body">
                        <pre class="mb-0">${data.text}</pre>
                    </div>
                `;
                youtubeResults.appendChild(card);
                
                logDebug('YouTube transcription complete: ' + data.text.length + ' characters');
            } else if (data.results) {
                // Channel results
                for (const [title, text] of Object.entries(data.results)) {
                    const card = document.createElement('div');
                    card.className = 'card mb-3';
                    card.innerHTML = `
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">${title}</h5>
                            <a href="${data.download_urls[title]}" class="btn btn-sm btn-primary">
                                <i class="fas fa-download"></i> Download
                            </a>
                        </div>
                        <div class="card-body">
                            <pre class="mb-0">${text}</pre>
                        </div>
                    `;
                    youtubeResults.appendChild(card);
                }
                
                logDebug('YouTube channel transcription complete: ' + 
                    Object.keys(data.results).length + ' videos');
            }
        } else {
            // Show error
            const errorCard = document.createElement('div');
            errorCard.className = 'alert alert-danger';
            errorCard.textContent = data.error;
            youtubeResults.appendChild(errorCard);
            
            logDebug('YouTube transcription error: ' + data.error);
        }
    })
    .catch(error => {
        // Stop status polling
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        
        // Hide spinner and enable button
        youtubeButton.disabled = false;
        youtubeSpinner.style.display = 'none';
        
        // Show error
        const errorCard = document.createElement('div');
        errorCard.className = 'alert alert-danger';
        errorCard.textContent = 'Error: ' + error.message;
        youtubeResults.appendChild(errorCard);
        
        logDebug('YouTube API error: ' + error.message);
    });
}

// Function to poll for status updates
function startStatusPolling(url, statusElement) {
    // Initial state
    let lastStatus = null;
    
    // Get progress bar and status text
    const progressBar = statusElement.querySelector('.progress-bar');
    const statusText = statusElement.querySelector('strong');
    
    // Start polling
    const interval = setInterval(() => {
        fetch(`/api/youtube-status?url=${encodeURIComponent(url)}`)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.status) {
                    const status = data.status;
                    
                    // If status changed
                    if (!lastStatus || 
                        lastStatus.status !== status.status || 
                        lastStatus.message !== status.message ||
                        lastStatus.progress !== status.progress) {
                        
                        // Update UI
                        if (status.progress !== null && progressBar) {
                            progressBar.style.width = `${status.progress}%`;
                            progressBar.setAttribute('aria-valuenow', status.progress);
                        }
                        
                        if (status.message && statusText) {
                            statusText.textContent = status.message;
                        } else if (status.status && statusText) {
                            // Default messages based on status
                            const messages = {
                                'downloading': 'Downloading YouTube audio...',
                                'processing': 'Processing audio...',
                                'transcribing': 'Transcribing audio...',
                                'completed': 'Transcription complete!',
                                'failed': 'Transcription failed'
                            };
                            statusText.textContent = messages[status.status] || 'Processing...';
                        }
                        
                        // If completed or failed, stop polling
                        if (status.status === 'completed' || status.status === 'failed') {
                            clearInterval(interval);
                            
                            // Update UI for completion/failure
                            statusElement.className = status.status === 'completed' 
                                ? 'alert alert-success mt-3' 
                                : 'alert alert-danger mt-3';
                                
                            // Remove spinner
                            const spinner = statusElement.querySelector('.spinner-border');
                            if (spinner) {
                                spinner.remove();
                            }
                        }
                        
                        // Log status change
                        const progressStr = status.progress !== null ? ` (${status.progress}%)` : '';
                        logDebug(`YouTube status: ${status.status}${progressStr} - ${status.message || ''}`);
                        
                        // Save for comparison
                        lastStatus = status;
                    }
                }
            })
            .catch(error => {
                console.error('Status polling error:', error);
            });
    }, 2000); // Poll every 2 seconds
    
    return interval;
}

// Utility Functions
function logDebug(message) {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.innerHTML = `<small>${timestamp}</small>: ${message}`;
    
    // Prepend to debug output
    debugOutput.insertBefore(logEntry, debugOutput.firstChild);
    
    // Also log to console
    console.log(`[${timestamp}] ${message}`);
} 
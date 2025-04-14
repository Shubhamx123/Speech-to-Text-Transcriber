from flask import Flask, request, jsonify, send_file, Response
import os
import tempfile
import uuid
import json
import time
import threading
from werkzeug.utils import secure_filename
import traceback
import logging
import sys
from logging.handlers import RotatingFileHandler

# Set up logging
def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration for the application
    
    Args:
        log_level: The logging level to use (default: INFO)
    
    Returns:
        A configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('stt_app')
    logger.setLevel(log_level)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler for all logs
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logging configuration completed")
    return logger

# Set up logging
logger = setup_logging()

# Import local modules
from whisper import WhisperTranscriber, WHISPER_AVAILABLE
from bhashini import BhashiniASR, BHASHINI_AVAILABLE
from spring_lab import SpringLabASR, SPRING_LAB_AVAILABLE

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "stt_uploads")
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'mp4', 'avi', 'mov', 'mkv', 'm4a'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"Upload directory created at: {UPLOAD_FOLDER}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Store active transcription jobs
transcription_jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logger.info("Index page requested")
    engines = {
        'whisper': WHISPER_AVAILABLE,
        'spring_lab': SPRING_LAB_AVAILABLE,
        'bhashini': BHASHINI_AVAILABLE,
    }
    
    whisper_models = ['tiny', 'base', 'small', 'medium', 'large'] if WHISPER_AVAILABLE else []
    
    languages = []
    if SPRING_LAB_AVAILABLE:
        try:
            spring_asr = SpringLabASR()
            languages = spring_asr.get_supported_languages()
            logger.info(f"Loaded {len(languages)} languages from SPRING Lab")
        except Exception as e:
            logger.error(f"Error loading SPRING Lab languages: {e}")
            languages = ["english", "hindi", "tamil", "telugu"] # Fallback
    
    # Check if any engines are available
    any_engine_available = any(engines.values())
    
    # Create html content directly
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech-to-Text Transcription</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        .hidden { display: none; }
        .card { margin-bottom: 20px; }
        #result-container { max-height: 400px; overflow-y: auto; }
        #status-container { margin-top: 20px; }
        .control-btn { 
            width: 180px; 
            margin: 5px 10px;
            padding: 8px 0;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4 text-center">Speech-to-Text Transcription</h1>
"""
    
    # Add warning for Whisper not available
    if not engines['whisper']:
        html_content += """
        <div class="alert alert-danger mb-4">
            <h5><i class="bi bi-exclamation-triangle"></i> Whisper Engine Not Available</h5>
            <p><strong>Error detected:</strong> The logs show a module error: <code>module 'whisper' has no attribute 'load_model'</code></p>
            <p>This means you might have the wrong whisper package installed.</p>
            <p><strong>To fix this:</strong> Run the <code>fix_whisper.py</code> file that was just created in your project folder:</p>
            <p><code>python fix_whisper.py</code></p>
            <p>After installation, restart the application to use Whisper transcription.</p>
        </div>
"""

    # Add warning for no engines available
    if not any_engine_available:
        html_content += """
        <div class="alert alert-danger mb-4">
            <h5><i class="bi bi-exclamation-triangle"></i> No Transcription Engines Available</h5>
            <p>No transcription engines are available. Please install at least one of the following:</p>
            <ul>
                <li>OpenAI Whisper - Run <code>install_whisper.bat</code></li>
                <li>SPRING Lab</li>
                <li>Bhashini</li>
            </ul>
            <p>After installation, restart the application.</p>
        </div>
"""

    # Start form container
    html_content += """
        <div class="card">
            <div class="card-header">
                <h5>Transcription Options</h5>
            </div>
            <div class="card-body">
                <form id="transcription-form">
                    <div class="mb-3">
                        <label for="engine" class="form-label">Transcription Engine:</label>
                        <select class="form-select" id="engine" name="engine" required>
"""

    # Add engine options
    if engines['whisper']:
        html_content += '                            <option value="whisper">OpenAI Whisper</option>\n'
    if engines['spring_lab']:
        html_content += '                            <option value="spring_lab">SPRING Lab</option>\n'
    if engines['bhashini']:
        html_content += '                            <option value="bhashini">Bhashini</option>\n'

    # Continue with the form
    html_content += """
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="source_type" class="form-label">Source Type:</label>
                        <select class="form-select" id="source_type" name="source_type" required>
                            <option value="file">File Upload</option>
                            <option value="youtube">YouTube URL</option>
                            <option value="live">Live Recording</option>
                        </select>
                    </div>
                    
                    <div id="file-input" class="mb-3">
                        <label for="file" class="form-label">Audio/Video File:</label>
                        <input type="file" class="form-control" id="file" name="file" accept="audio/*,video/*">
                        <div class="form-text">Supported formats: MP3, WAV, OGG, FLAC, M4A, MP4, AVI, MOV, MKV</div>
                    </div>
                    
                    <div id="youtube-input" class="mb-3 hidden">
                        <label for="youtube_url" class="form-label">YouTube URL:</label>
                        <input type="url" class="form-control" id="youtube_url" name="youtube_url" placeholder="https://www.youtube.com/watch?v=...">
                    </div>
                    
                    <div id="live-input" class="mb-3 hidden">
                        <label for="live_duration" class="form-label">Maximum Recording Duration (seconds):</label>
                        <input type="number" class="form-control" id="live_duration" name="live_duration" value="10" min="1" max="60">
                    </div>
                    
                    <div class="mb-3">
                        <label for="language" class="form-label">Language:</label>
                        <select class="form-select" id="language" name="language">
                            <option value="">Auto-detect (where supported)</option>
"""

    # Add language options
    for lang in languages:
        html_content += f'                            <option value="{lang}">{lang.capitalize()}</option>\n'

    # Continue with form
    html_content += """
                        </select>
                    </div>
                    
                    <div id="bhashini-options" class="mb-3 hidden">
                        <label for="api_key" class="form-label">Bhashini API Key:</label>
                        <input type="text" class="form-control" id="api_key" name="api_key" placeholder="Enter your Bhashini API key">
                        <div class="form-text">Get your API key from <a href="https://bhashini.gov.in/developer" target="_blank">Bhashini Developer Portal</a></div>
                    </div>
                    
                    <div id="whisper-options" class="mb-3">
                        <label for="whisper_model" class="form-label">Whisper Model:</label>
                        <select class="form-select" id="whisper_model" name="whisper_model">
"""

    # Add whisper model options
    for model in whisper_models:
        selected = ' selected' if model == 'base' else ''
        html_content += f'                            <option value="{model}"{selected}>{model.capitalize()}</option>\n'

    # Complete the form with the buttons
    html_content += """
                        </select>
                        <div class="form-text">Larger models are more accurate but slower and require more resources</div>
                    </div>
                    
                    <div class="mb-3">
                        <div id="recording-timer" class="mb-2 p-2 bg-light border rounded" style="display: none;">
                            <strong>Recording: <span id="recording-seconds">0</span> seconds</strong>
                        </div>
                        
                        <div class="d-flex justify-content-center mt-4">
                            <button type="submit" class="btn btn-primary control-btn mx-2" id="submit-btn">Start Transcription</button>
                            <button type="button" class="btn btn-danger control-btn mx-2" id="stop-btn">Stop</button>
                            <button type="button" class="btn btn-secondary control-btn mx-2" id="reset-btn">Reset</button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
        
        <div id="status-container" class="card hidden">
            <div class="card-header">
                <h5>Transcription Status</h5>
            </div>
            <div class="card-body">
                <div class="progress mb-3">
                    <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
                </div>
                <p id="status-message">Processing your audio...</p>
                <p id="time-elapsed">Time elapsed: 0 seconds</p>
            </div>
        </div>
        
        <div id="result-container" class="card hidden">
            <div class="card-header">
                <h5>Transcription Result</h5>
            </div>
            <div class="card-body">
                <div id="result-content" class="mb-3"></div>
                <button id="copy-button" class="btn btn-sm btn-secondary">Copy to Clipboard</button>
                <button id="download-button" class="btn btn-sm btn-secondary">Download</button>
                <button id="new-transcription" class="btn btn-sm btn-primary">New Transcription</button>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize UI on page load
        window.addEventListener('DOMContentLoaded', (event) => {
            // Disable stop button initially
            document.getElementById('stop-btn').disabled = true;
        });
        
        // Show/hide source type inputs
        document.getElementById('source_type').addEventListener('change', function() {
            document.getElementById('file-input').classList.add('hidden');
            document.getElementById('youtube-input').classList.add('hidden');
            document.getElementById('live-input').classList.add('hidden');
            
            const sourceType = this.value;
            document.getElementById(sourceType + '-input').classList.remove('hidden');
        });
        
        // Show/hide engine specific options
        document.getElementById('engine').addEventListener('change', function() {
            document.getElementById('bhashini-options').classList.add('hidden');
            document.getElementById('whisper-options').classList.add('hidden');
            
            const engine = this.value;
            if (engine === 'bhashini') {
                document.getElementById('bhashini-options').classList.remove('hidden');
            }
            
            if (engine === 'whisper') {
                document.getElementById('whisper-options').classList.remove('hidden');
            }
        });
        
        // Live recording variables
        let recordingTimerId = null;
        let recordingSeconds = 0;
        let currentJobId = null;
        
        // Handle form submission (Start Transcription)
        document.getElementById('transcription-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const sourceType = formData.get('source_type');
            
            // Validate form
            if (sourceType === 'file' && !formData.get('file').size) {
                alert('Please select a file to upload');
                return;
            }
            
            if (sourceType === 'youtube' && !formData.get('youtube_url')) {
                alert('Please enter a YouTube URL');
                return;
            }
            
            if (formData.get('engine') === 'bhashini' && !formData.get('api_key')) {
                alert('Bhashini API key is required');
                return;
            }
            
            // Show status container
            document.getElementById('status-container').classList.remove('hidden');
            document.getElementById('result-container').classList.add('hidden');
            
            // For live recording, start the timer
            if (sourceType === 'live') {
                // Show and start the recording timer
                document.getElementById('recording-timer').style.display = 'block';
                recordingSeconds = 0;
                document.getElementById('recording-seconds').textContent = recordingSeconds;
                
                // Start the timer
                recordingTimerId = setInterval(function() {
                    recordingSeconds++;
                    document.getElementById('recording-seconds').textContent = recordingSeconds;
                }, 1000);
                
                document.getElementById('status-message').textContent = 'Recording in progress...';
                
                // Disable the start button and enable the stop button
                document.getElementById('submit-btn').disabled = true;
                document.getElementById('stop-btn').disabled = false;
            }
            
            // Submit form
            fetch('/transcribe', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('status-message').textContent = 'Error: ' + data.error;
                    document.getElementById('progress-bar').classList.remove('progress-bar-animated');
                    document.getElementById('progress-bar').classList.add('bg-danger');
                    stopRecording();
                } else {
                    currentJobId = data.job_id;
                    pollJobStatus(data.job_id);
                }
            })
            .catch(error => {
                document.getElementById('status-message').textContent = 'Error: ' + error;
                document.getElementById('progress-bar').classList.remove('progress-bar-animated');
                document.getElementById('progress-bar').classList.add('bg-danger');
                stopRecording();
            });
        });
        
        // Stop button handler
        document.getElementById('stop-btn').addEventListener('click', function() {
            if (!currentJobId) return;
            
            // Send stop request
            fetch('/stop_recording/' + currentJobId, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('status-message').textContent = 'Processing recording...';
            })
            .catch(error => {
                console.error('Error stopping recording:', error);
            });
            
            stopRecording();
        });
        
        // Reset button handler
        document.getElementById('reset-btn').addEventListener('click', function() {
            // Stop any ongoing recording
            if (currentJobId) {
                try {
                    fetch('/stop_recording/' + currentJobId, { method: 'POST' });
                } catch (e) { 
                    console.error("Error stopping recording:", e);
                }
            }
            
            // Reset UI
            stopRecording();
            document.getElementById('status-container').classList.add('hidden');
            document.getElementById('result-container').classList.add('hidden');
            document.getElementById('recording-timer').style.display = 'none';
            document.getElementById('submit-btn').disabled = false;
            
            // Reset form
            document.getElementById('transcription-form').reset();
        });
        
        function stopRecording() {
            // Stop the timer
            if (recordingTimerId) {
                clearInterval(recordingTimerId);
                recordingTimerId = null;
            }
            
            // Reset UI elements
            document.getElementById('submit-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            document.getElementById('recording-timer').style.display = 'none';
        }
        
        // Poll job status
        function pollJobStatus(jobId) {
            let startTime = Date.now();
            let interval = setInterval(() => {
                fetch('/job_status/' + jobId)
                .then(response => response.json())
                .then(data => {
                    const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
                    document.getElementById('time-elapsed').textContent = 'Time elapsed: ' + elapsedSeconds + ' seconds';
                    
                    if (data.status === 'completed') {
                        clearInterval(interval);
                        document.getElementById('status-container').classList.add('hidden');
                        document.getElementById('result-container').classList.remove('hidden');
                        displayResult(data.result);
                        
                        // Reset recording controls if this was a live recording job
                        if (jobId === currentJobId) {
                            stopRecording();
                            currentJobId = null;
                        }
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        document.getElementById('status-message').textContent = 'Error: ' + (data.result?.error || 'Unknown error');
                        document.getElementById('progress-bar').classList.remove('progress-bar-animated');
                        document.getElementById('progress-bar').classList.add('bg-danger');
                        
                        // Reset recording controls if this was a live recording job
                        if (jobId === currentJobId) {
                            stopRecording();
                            currentJobId = null;
                        }
                    } else if (data.status === 'recording') {
                        document.getElementById('status-message').textContent = 'Recording in progress...';
                        document.getElementById('submit-btn').disabled = true;
                        document.getElementById('stop-btn').disabled = false;
                    } else {
                        document.getElementById('status-message').textContent = 'Processing...';
                    }
                })
                .catch(error => {
                    console.error('Error polling job status:', error);
                });
            }, 1000);
        }
        
        // Display transcription result
        function displayResult(result) {
            const resultContent = document.getElementById('result-content');
            resultContent.innerHTML = '';
            
            // Handle different result formats based on the engine
            if (typeof result === 'string') {
                resultContent.innerHTML = '<pre>' + result + '</pre>';
            } else if (result.error) {
                resultContent.innerHTML = '<div class="alert alert-danger">' + result.error + '</div>';
            } else if (result.text) {
                // Whisper
                resultContent.innerHTML = '<h6>Transcription:</h6><pre>' + result.text + '</pre>';
                if (result.language) {
                    let langInfo = '<p>Detected language: ' + result.language;
                    if (result.original_detected_language) {
                        langInfo += ' (originally detected as: ' + result.original_detected_language + ', auto-corrected)';
                    }
                    langInfo += '</p>';
                    resultContent.innerHTML += langInfo;
                }
            } else if (result.transcript) {
                // SPRING Lab
                resultContent.innerHTML = '<h6>Transcription:</h6><pre>' + result.transcript + '</pre>';
                if (result.detected_language) {
                    resultContent.innerHTML += '<p>Detected language: ' + result.detected_language + '</p>';
                }
                if (result.time_taken) {
                    resultContent.innerHTML += '<p>Time taken: ' + result.time_taken + ' seconds</p>';
                }
            } else if (result.whisper || result.spring_lab || result.bhashini) {
                // Transkriptor (multiple engines)
                resultContent.innerHTML = '<h6>Comparison Results:</h6>';
                
                if (result.whisper) {
                    let whisperHeader = 'Whisper';
                    if (result.whisper.original_detected_language) {
                        whisperHeader += ' (Auto-corrected from ' + result.whisper.original_detected_language + ' to ' + result.whisper.language + ')';
                    } else if (result.whisper.language) {
                        whisperHeader += ' (' + result.whisper.language + ')';
                    }
                    resultContent.innerHTML += '<div class="card mb-3"><div class="card-header">' + whisperHeader + '</div><div class="card-body"><pre>' + 
                        (result.whisper.text || JSON.stringify(result.whisper, null, 2)) + '</pre></div></div>';
                }
                
                if (result.spring_lab) {
                    let springText = '<div class="card mb-3"><div class="card-header">SPRING Lab';
                    if (result.spring_lab.detected_language) {
                        springText += ' (Detected: ' + result.spring_lab.detected_language + ')';
                    }
                    springText += '</div><div class="card-body"><pre>' + 
                        (result.spring_lab.transcript || JSON.stringify(result.spring_lab, null, 2)) + '</pre></div></div>';
                    resultContent.innerHTML += springText;
                }
                
                if (result.bhashini) {
                    resultContent.innerHTML += '<div class="card mb-3"><div class="card-header">Bhashini</div><div class="card-body"><pre>' + 
                        (typeof result.bhashini === 'string' ? result.bhashini : JSON.stringify(result.bhashini, null, 2)) + '</pre></div></div>';
                }
            } else {
                // Unknown format, just stringify the JSON
                resultContent.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
            }
        }
        
        // Copy result to clipboard
        document.getElementById('copy-button').addEventListener('click', function() {
            const resultText = document.getElementById('result-content').innerText;
            navigator.clipboard.writeText(resultText).then(() => {
                alert('Copied to clipboard!');
            });
        });
        
        // Download result
        document.getElementById('download-button').addEventListener('click', function() {
            const resultText = document.getElementById('result-content').innerText;
            const blob = new Blob([resultText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'transcription_result.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
        
        // New transcription
        document.getElementById('new-transcription').addEventListener('click', function() {
            document.getElementById('transcription-form').reset();
            document.getElementById('result-container').classList.add('hidden');
            document.getElementById('file-input').classList.remove('hidden');
            document.getElementById('youtube-input').classList.add('hidden');
            document.getElementById('live-input').classList.add('hidden');
            document.getElementById('progress-bar').classList.remove('bg-danger');
            document.getElementById('progress-bar').classList.add('progress-bar-animated');
            document.getElementById('submit-btn').disabled = false;
        });
    </script>
</body>
</html>
"""
    
    return html_content

@app.route('/transcribe', methods=['POST'])
def transcribe():
    engine = request.form.get('engine')
    source_type = request.form.get('source_type')
    language = request.form.get('language', '')
    api_key = request.form.get('api_key', '')
    whisper_model = request.form.get('whisper_model', 'base')
    youtube_url = request.form.get('youtube_url', '')
    
    job_id = str(uuid.uuid4())
    result_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_result.json")
    
    logger.info(f"New transcription job started. ID: {job_id}, Engine: {engine}, Source: {source_type}, Language: {language if language else 'auto-detect'}")
    
    if source_type == 'file':
        # Check if a file was uploaded
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(file_path)
        logger.info(f"File saved at: {file_path}")
        
        # Start transcription in background
        threading.Thread(target=process_file_transcription, 
                        args=(engine, file_path, language, api_key, whisper_model, job_id, result_file)).start()
        
    elif source_type == 'youtube':
        if not youtube_url:
            logger.error("YouTube URL is required")
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        logger.info(f"YouTube URL: {youtube_url}")
        # Start transcription in background
        threading.Thread(target=process_youtube_transcription,
                        args=(engine, youtube_url, language, api_key, whisper_model, job_id, result_file)).start()
    
    elif source_type == 'live':
        # For live recording with start/stop functionality
        duration = int(request.form.get('live_duration', 60))  # Default to 60 seconds max
        logger.info(f"Live recording started. Max duration: {duration} seconds")
        
        # Store job info before starting thread
        transcription_jobs[job_id] = {
            'status': 'recording',
            'start_time': time.time(),
            'engine': engine,
            'source_type': source_type,
            'result_file': result_file,
            'should_stop': False
        }
        
        # Start transcription in background
        threading.Thread(target=process_live_transcription,
                        args=(engine, duration, language, api_key, whisper_model, job_id, result_file)).start()
    
    else:
        logger.error(f"Invalid source type: {source_type}")
        return jsonify({'error': 'Invalid source type'}), 400
    
    # Store job info if not already stored (for live recording)
    if job_id not in transcription_jobs:
        transcription_jobs[job_id] = {
            'status': 'processing',
            'start_time': time.time(),
            'engine': engine,
            'source_type': source_type,
            'result_file': result_file
        }
    
    return jsonify({'job_id': job_id})

@app.route('/stop_recording/<job_id>', methods=['POST'])
def stop_recording(job_id):
    """Stop an ongoing recording job"""
    if job_id not in transcription_jobs:
        logger.error(f"Job not found: {job_id}")
        return jsonify({'error': 'Job not found'}), 404
    
    job = transcription_jobs[job_id]
    if job['status'] != 'recording':
        logger.warning(f"Cannot stop job {job_id} because it's not in recording state")
        return jsonify({'error': 'Job is not in recording state'}), 400
    
    # Set the stop flag for the job
    job['should_stop'] = True
    logger.info(f"Stop signal sent for job {job_id}")
    
    return jsonify({'status': 'stopping'})

@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    if job_id not in transcription_jobs:
        logger.error(f"Job not found: {job_id}")
        return jsonify({'error': 'Job not found'}), 404
    
    job = transcription_jobs[job_id]
    logger.debug(f"Job status requested for {job_id}: {job['status']}")
    
    if job['status'] == 'completed' and os.path.exists(job['result_file']):
        with open(job['result_file'], 'r', encoding='utf-8') as f:
            result = json.load(f)
        return jsonify({
            'status': 'completed',
            'result': result
        })
    
    return jsonify({
        'status': job['status'],
        'elapsed_time': round(time.time() - job['start_time'], 1)
    })

def process_file_transcription(engine, file_path, language, api_key, whisper_model, job_id, result_file):
    try:
        logger.info(f"Starting file transcription. Job: {job_id}, Engine: {engine}, File: {file_path}")
        result = {}
        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if engine == 'whisper' and WHISPER_AVAILABLE:
            logger.info(f"Using Whisper with model: {whisper_model}")
            transcriber = WhisperTranscriber(model_name=whisper_model)
            if language:
                transcriber.language = language
                logger.info(f"Language set to: {language}")
                # Important: Set task to "transcribe" to preserve original language
                transcriber.task = "transcribe"
                logger.info("Task set to 'transcribe' to preserve original language")
            
            if is_video:
                logger.info("Processing video file")
                result = transcriber.transcribe_video_file(file_path)
            else:
                logger.info("Processing audio file")
                result = transcriber.transcribe_audio_file(file_path)
                
        elif engine == 'spring_lab' and SPRING_LAB_AVAILABLE:
            logger.info("Using SPRING Lab ASR")
            transcriber = SpringLabASR()
            if language:
                transcriber.language = language
                logger.info(f"Language set to: {language}")
            else:
                # Empty string for auto-detection
                transcriber.language = ""
                logger.info("Language set to auto-detect")
                
            if is_video:
                logger.info("Processing video file")
                result = transcriber.transcribe_video_file(file_path)
            else:
                logger.info("Processing audio file")
                result = transcriber.transcribe_audio_file(file_path)
                
        elif engine == 'bhashini' and BHASHINI_AVAILABLE:
            logger.info("Using Bhashini ASR")
            transcriber = BhashiniASR(api_key=api_key)
            if language:
                transcriber.source_language = language
                logger.info(f"Language set to: {language}")
                
            if is_video:
                logger.info("Processing video file")
                result = transcriber.transcribe_video_file(file_path)
            else:
                logger.info("Processing audio file")
                result = transcriber.transcribe_audio_file(file_path)
                
        else:
            error_msg = f"Selected engine '{engine}' is not available"
            logger.error(error_msg)
            result = {"error": error_msg}
        
        # Save result
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Transcription result saved to: {result_file}")
            
        # Update job status
        transcription_jobs[job_id]['status'] = 'completed'
        logger.info(f"Job {job_id} completed successfully")
        
        # Clean up
        try:
            os.remove(file_path)
            logger.debug(f"Temporary file removed: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {e}")
            
    except Exception as e:
        error_msg = f"Error in file transcription: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e)}, f)
        transcription_jobs[job_id]['status'] = 'error'
        logger.error(f"Job {job_id} failed")

def process_youtube_transcription(engine, youtube_url, language, api_key, whisper_model, job_id, result_file):
    try:
        logger.info(f"Starting YouTube transcription. Job: {job_id}, Engine: {engine}, URL: {youtube_url}")
        result = {}
        
        if engine == 'whisper' and WHISPER_AVAILABLE:
            logger.info(f"Using Whisper with model: {whisper_model}")
            transcriber = WhisperTranscriber(model_name=whisper_model)
            if language:
                transcriber.language = language
                logger.info(f"Language set to: {language}")
                # Important: Set task to "transcribe" to preserve original language
                transcriber.task = "transcribe"
                logger.info("Task set to 'transcribe' to preserve original language")
            result = transcriber.transcribe_youtube_video(youtube_url)
                
        elif engine == 'spring_lab' and SPRING_LAB_AVAILABLE:
            logger.info("Using SPRING Lab ASR")
            transcriber = SpringLabASR()
            if language:
                transcriber.language = language
                logger.info(f"Language set to: {language}")
            result = transcriber.transcribe_youtube_video(youtube_url)
                
        elif engine == 'bhashini' and BHASHINI_AVAILABLE:
            logger.info("Using Bhashini ASR")
            transcriber = BhashiniASR(api_key=api_key)
            if language:
                transcriber.source_language = language
                logger.info(f"Language set to: {language}")
            result = transcriber.transcribe_youtube_video(youtube_url)
                
        else:
            error_msg = f"Selected engine '{engine}' is not available"
            logger.error(error_msg)
            result = {"error": error_msg}
        
        # Save result
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Transcription result saved to: {result_file}")
            
        # Update job status
        transcription_jobs[job_id]['status'] = 'completed'
        logger.info(f"Job {job_id} completed successfully")
            
    except Exception as e:
        error_msg = f"Error in YouTube transcription: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e)}, f)
        transcription_jobs[job_id]['status'] = 'error'
        logger.error(f"Job {job_id} failed")

def process_live_transcription(engine, duration, language, api_key, whisper_model, job_id, result_file):
    try:
        logger.info(f"Starting live transcription. Job: {job_id}, Engine: {engine}, Max Duration: {duration}s")
        result = {}
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, f"recording_{job_id}.wav")
        
        # Get the transcriber based on the engine
        transcriber = None
        if engine == 'whisper' and WHISPER_AVAILABLE:
            logger.info(f"Using Whisper with model: {whisper_model}")
            transcriber = WhisperTranscriber(model_name=whisper_model)
            if language:
                transcriber.language = language
                logger.info(f"Language set to: {language}")
                # Important: Set task to "transcribe" to preserve original language
                transcriber.task = "transcribe"
                logger.info("Task set to 'transcribe' to preserve original language")
                
        elif engine == 'spring_lab' and SPRING_LAB_AVAILABLE:
            logger.info("Using SPRING Lab ASR")
            transcriber = SpringLabASR()
            if language:
                transcriber.language = language
                logger.info(f"Language set to: {language}")
                
        elif engine == 'bhashini' and BHASHINI_AVAILABLE:
            logger.info("Using Bhashini ASR")
            transcriber = BhashiniASR(api_key=api_key)
            if language:
                transcriber.source_language = language
                logger.info(f"Language set to: {language}")
                
        else:
            error_msg = f"Selected engine '{engine}' is not available"
            logger.error(error_msg)
            result = {"error": error_msg}
            raise ValueError(error_msg)
        
        # Record audio using the selected transcriber
        if transcriber:
            # Use custom recording with stop capability
            result = custom_record_and_transcribe(transcriber, job_id, temp_wav, duration)
            
        # Save result
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Transcription result saved to: {result_file}")
            
        # Update job status
        transcription_jobs[job_id]['status'] = 'completed'
        logger.info(f"Job {job_id} completed successfully")
            
    except Exception as e:
        error_msg = f"Error in live transcription: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e)}, f)
        transcription_jobs[job_id]['status'] = 'error'
        logger.error(f"Job {job_id} failed")

def custom_record_and_transcribe(transcriber, job_id, temp_wav, max_duration=60, sample_rate=16000):
    """
    Custom recording function that can be stopped by setting should_stop flag
    """
    import sounddevice as sd
    import numpy as np
    import wave
    
    logger.info(f"Starting recording for job {job_id}")
    
    # Create an array to store the recording
    max_frames = max_duration * sample_rate
    recording = np.zeros((max_frames, 1), dtype=np.float32)
    frames_recorded = 0
    
    # Define callback function for audio stream
    def callback(indata, frames, time, status):
        nonlocal frames_recorded
        if status:
            logger.warning(f"Audio callback status: {status}")
        remaining = max_frames - frames_recorded
        if remaining <= 0:
            return
        
        # Store the new frames
        frames_to_add = min(remaining, len(indata))
        recording[frames_recorded:frames_recorded + frames_to_add] = indata[:frames_to_add]
        frames_recorded += frames_to_add
    
    # Start recording
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            # Wait until recording should stop or max duration is reached
            while frames_recorded < max_frames:
                sd.sleep(100)  # Check every 100ms
                if job_id in transcription_jobs and transcription_jobs[job_id].get('should_stop', False):
                    logger.info(f"Stopping recording for job {job_id}")
                    break
    except Exception as e:
        logger.error(f"Error during recording: {e}")
        return {"error": f"Recording error: {str(e)}"}
    
    # Only process if we have recorded something
    if frames_recorded == 0:
        return {"error": "No audio recorded"}
    
    # Trim the recording to actual length
    trimmed_recording = recording[:frames_recorded]
    
    # Save to WAV file
    try:
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes((trimmed_recording * 32767).astype(np.int16).tobytes())
        
        # Transcribe
        logger.info(f"Transcribing {frames_recorded/sample_rate:.2f} seconds of audio")
        result = transcriber.transcribe_audio_file(temp_wav)
        
        # Clean up
        try:
            os.remove(temp_wav)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_wav}: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Error saving or transcribing audio: {e}")
        return {"error": f"Processing error: {str(e)}"}

if __name__ == '__main__':
    # No need to create template directory since we're not using templates anymore
    
    logger.info("Starting web server at http://127.0.0.1:5000")
    app.run(debug=True) 
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import tempfile
import time
import base64
from transcriber import (
    safe_transcribe_with_whisper,
    enhanced_transcribe_with_whisper,
    transcribe_with_springlab,
    transcribe_youtube_video,
    transcribe_youtube_channel,
    get_available_languages
)
from utils import save_text_file, convert_to_wav, select_available_gpu, download_youtube_audio
from celery_worker import celery_app
import uuid

# Check for CUDA availability and print GPU info
import torch
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"CUDA available with {gpu_count} GPUs:")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
        print(f"  GPU {i}: {gpu_name} with {gpu_mem:.1f} GB memory")
else:
    print("CUDA not available, running in CPU mode only")

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create jobs directory for storing task status
JOBS_DIR = os.path.join(OUTPUT_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory status tracking for YouTube transcriptions
youtube_transcription_status = {}

# Create Flask app
app = Flask(__name__, static_url_path='/static')

# Set max content length (1GB)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024

# Default configuration
DEFAULT_ENGINE = "Whisper (Local)"
DEFAULT_MODEL_SIZE = "small"
DEFAULT_LANGUAGE = "english"

# Define Celery tasks
@celery_app.task
def transcribe_youtube_task(task_id, url, engine, model_size, language):
    """Background task to transcribe YouTube video"""
    try:
        print(f"[Celery] Starting YouTube transcription task {task_id}")
        
        # Update job status
        update_job_status(task_id, "downloading", "Downloading YouTube audio...")
        
        # Download YouTube audio
        method = "whisper" if engine == "Whisper (Local)" else "springlab"
        audio_path = download_youtube_audio(url)
        
        # Validate the downloaded file
        from pydub import AudioSegment
        
        # Check if file exists
        if not os.path.exists(audio_path):
            update_job_status(task_id, "failed", "Downloaded YouTube audio file not found")
            return {"success": False, "error": "Error: Downloaded YouTube audio file not found"}
        
        # Check file size
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb < 0.5:  # Less than 500 KB
            update_job_status(task_id, "failed", f"Downloaded YouTube audio too small ({file_size_mb:.2f} MB)")
            return {"success": False, "error": f"Audio file too small ({file_size_mb:.2f} MB)"}
        
        # Check audio duration
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_sec = len(audio) / 1000
            if duration_sec < 5:  # Less than 5 seconds
                update_job_status(task_id, "failed", f"Downloaded audio is too short ({duration_sec:.2f} seconds)")
                return {"success": False, "error": f"Audio is too short ({duration_sec:.2f} seconds)"}
        except Exception as e:
            print(f"[Celery] Warning: Could not check audio duration: {str(e)}")
            # Continue anyway as this is an optional check
        
        # Update job status
        update_job_status(task_id, "transcribing", "Transcribing audio...")
        
        # Transcribe the audio
        start_time = time.time()
        if method == "whisper":
            text = enhanced_transcribe_with_whisper(audio_path, model_size, language)
        else:
            text = transcribe_with_springlab(audio_path, language)
        
        # Create a downloadable file
        output_file = save_text_file(text, OUTPUT_DIR, f"youtube_{task_id}")
        elapsed_time = time.time() - start_time
        
        # Update job status
        update_job_status(task_id, "completed", "Transcription complete", 
                         {"text": text, "download_url": f"/output/{output_file}"})
                         
        print(f"[Celery] YouTube transcription task {task_id} completed in {elapsed_time:.2f} seconds")
        return {"success": True, "text": text, "download_url": f"/output/{output_file}"}
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[Celery] Error in YouTube transcription task {task_id}: {error_msg}")
        print(traceback.format_exc())
        update_job_status(task_id, "failed", error_msg)
        return {"success": False, "error": error_msg}

# Helper function to update job status
def update_job_status(task_id, status, message, data=None):
    """Update the status of a background job"""
    status_data = {
        "task_id": task_id,
        "status": status,  # pending, downloading, transcribing, completed, failed
        "message": message,
        "timestamp": time.time(),
        "data": data
    }
    
    # Save to a JSON file
    import json
    status_file = os.path.join(JOBS_DIR, f"{task_id}.json")
    with open(status_file, 'w') as f:
        json.dump(status_data, f)
    
    return status_data

# Helper function to update YouTube transcription status
def update_youtube_status(url, status, progress=None, message=None):
    """Update the status of a YouTube transcription"""
    status_data = {
        "url": url,
        "status": status,  # downloading, processing, transcribing, completed, failed
        "progress": progress,  # Optional percentage complete
        "message": message,
        "timestamp": time.time()
    }
    
    # Store in memory
    youtube_transcription_status[url] = status_data
    
    return status_data

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/output/<path:filename>')
def download_file(filename):
    """Download a file from the output directory."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """API endpoint for transcribing audio from browser recording."""
    try:
        # Get request data
        audio_file = request.files.get('audio_file')
        engine = request.form.get('engine', DEFAULT_ENGINE)
        language = request.form.get('language', DEFAULT_LANGUAGE)
        model_size = request.form.get('model_size', DEFAULT_MODEL_SIZE)
        
        if not audio_file:
            return jsonify({"success": False, "error": "No audio file provided"})
        
        # Save uploaded file temporarily with a more generic suffix
        uploaded_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        uploaded_temp.close()
        audio_file.save(uploaded_temp.name)
        
        # Record file size for logging
        file_size_mb = os.path.getsize(uploaded_temp.name) / (1024 * 1024)
        print(f"[API] Received audio file: {file_size_mb:.2f} MB")
        
        # Convert uploaded file to proper WAV
        try:
            wav_path = convert_to_wav(uploaded_temp.name)
        except Exception as e:
            # Clean up the temporary file if conversion fails
            if os.path.exists(uploaded_temp.name):
                os.remove(uploaded_temp.name)
            print(f"[API] Error converting audio file: {str(e)}")
            return jsonify({"success": False, "error": f"Invalid recording file: {str(e)}"})
        
        # Clean up uploaded file (original .webm)
        if os.path.exists(uploaded_temp.name):
            os.remove(uploaded_temp.name)
        
        # Transcribe the audio
        try:
            if engine == "Whisper (Local)":
                print(f"[API] Starting enhanced transcription with model_size={model_size}, language={language}")
                text = enhanced_transcribe_with_whisper(wav_path, model_size, language)
            else:
                print(f"[API] Starting SpringLab transcription with language={language}")
                text = transcribe_with_springlab(wav_path, language)
                
            # Create a downloadable file
            filename = "recording_transcription.txt"
            save_text_file(text, OUTPUT_DIR, "recording")
            
            # Clean up the temporary WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path)
                
            return jsonify({
                "success": True, 
                "text": text,
                "download_url": f"/output/{filename}"
            })
        except Exception as e:
            # Clean up temporary files
            if os.path.exists(wav_path):
                os.remove(wav_path)
            print(f"[API] Transcription error: {str(e)}")
            return jsonify({"success": False, "error": str(e)})
    
    except Exception as e:
        print(f"[API] General error in transcribe API: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for transcribing uploaded audio/video files."""
    try:
        # Get request data
        file = request.files.get('file')
        engine = request.form.get('engine', DEFAULT_ENGINE)
        language = request.form.get('language', DEFAULT_LANGUAGE)
        model_size = request.form.get('model_size', DEFAULT_MODEL_SIZE)
        
        if not file:
            return jsonify({"success": False, "error": "No file provided"})
        
        # Save the file
        file_path = os.path.join(OUTPUT_DIR, file.filename)
        file.save(file_path)
        
        # Record file size for logging
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"[API] Received uploaded file: {file.filename} ({file_size_mb:.2f} MB)")
        
        # Transcribe the file
        try:
            start_time = time.time()
            if engine == "Whisper (Local)":
                print(f"[API] Starting enhanced transcription with model_size={model_size}, language={language}")
                text = enhanced_transcribe_with_whisper(file_path, model_size, language)
            else:
                print(f"[API] Starting SpringLab transcription with language={language}")
                text = transcribe_with_springlab(file_path, language)
                
            elapsed_time = time.time() - start_time
            print(f"[API] Transcription completed in {elapsed_time:.2f} seconds")
                
            # Create a downloadable file
            output_file = save_text_file(text, OUTPUT_DIR, file.filename)
            
            return jsonify({
                "success": True, 
                "text": text,
                "download_url": f"/output/{output_file}"
            })
        except Exception as e:
            print(f"[API] Transcription error: {str(e)}")
            return jsonify({"success": False, "error": str(e)})
    
    except Exception as e:
        print(f"[API] Upload request error: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/youtube', methods=['POST'])
def api_youtube():
    """API endpoint for transcribing YouTube videos."""
    try:
        # Get request data
        url = request.form.get('url')
        channel_id = request.form.get('channel_id')
        
        # Safe conversion of max_videos to handle empty string or invalid values
        max_videos_str = request.form.get('max_videos', '5')
        try:
            # If empty or just whitespace, use default value 5
            max_videos = int(max_videos_str) if max_videos_str.strip() else 5
        except (ValueError, TypeError):
            # If conversion fails (not a number), use default value 5
            max_videos = 5
            
        engine = request.form.get('engine', DEFAULT_ENGINE)
        language = request.form.get('language', DEFAULT_LANGUAGE)
        model_size = request.form.get('model_size', DEFAULT_MODEL_SIZE)
        
        print(f"[API] YouTube transcription request: url={url}, channel_id={channel_id}")
        print(f"[API] Engine: {engine}, Language: {language}, Model size: {model_size}, Max videos: {max_videos}")
        
        if not url and not channel_id:
            return jsonify({"success": False, "error": "No YouTube URL or channel ID provided"})
        
        # Transcribe from YouTube (using the extended timeout)
        try:
            start_time = time.time()
            if url:
                # Transcribe single video
                method = "whisper" if engine == "Whisper (Local)" else "springlab"
                print(f"[API] Transcribing YouTube video URL: {url} with {method}")
                
                # Initialize status
                update_youtube_status(url, "downloading", 0, "Downloading YouTube audio...")
                
                # Download YouTube audio
                try:
                    audio_path = download_youtube_audio(url)
                    
                    # Validate the downloaded file
                    from pydub import AudioSegment
                    
                    # Check if file exists
                    if not os.path.exists(audio_path):
                        print(f"[API] Error: Downloaded YouTube audio file not found: {audio_path}")
                        update_youtube_status(url, "failed", 100, "Downloaded YouTube audio file not found")
                        return jsonify({
                            "success": False, 
                            "error": "Error: Downloaded YouTube audio file not found. The video may be unavailable or region-restricted."
                        })
                    
                    # Check file size
                    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                    print(f"[API] YouTube audio file size: {file_size_mb:.2f} MB")
                    if file_size_mb < 0.5:  # Less than 500 KB
                        print(f"[API] Error: Downloaded YouTube audio too small ({file_size_mb:.2f} MB)")
                        update_youtube_status(url, "failed", 100, f"Downloaded YouTube audio too small ({file_size_mb:.2f} MB)")
                        return jsonify({
                            "success": False, 
                            "error": f"Error: Downloaded YouTube audio is too small ({file_size_mb:.2f} MB). The video may be silent or corrupted."
                        })
                    
                    # Check audio duration
                    try:
                        audio = AudioSegment.from_file(audio_path)
                        duration_sec = len(audio) / 1000
                        print(f"[API] YouTube audio duration: {duration_sec:.2f} seconds")
                        if duration_sec < 5:  # Less than 5 seconds
                            print(f"[API] Error: Downloaded audio is too short ({duration_sec:.2f} seconds)")
                            update_youtube_status(url, "failed", 100, f"Downloaded audio is too short ({duration_sec:.2f} seconds)")
                            return jsonify({
                                "success": False, 
                                "error": f"Error: Downloaded audio is too short ({duration_sec:.2f} seconds). The video may be invalid or have no audio track."
                            })
                    except Exception as e:
                        print(f"[API] Warning: Could not check audio duration: {str(e)}")
                        # Continue anyway as this is an optional check
                    
                    # If all validation checks pass, proceed with transcription
                    print(f"[API] YouTube audio validation passed, proceeding with transcription")
                    update_youtube_status(url, "transcribing", 30, "Audio downloaded successfully. Starting transcription...")
                except Exception as e:
                    print(f"[API] YouTube download error: {str(e)}")
                    update_youtube_status(url, "failed", 100, f"Error downloading YouTube video: {str(e)}")
                    return jsonify({
                        "success": False, 
                        "error": f"Error downloading YouTube video: {str(e)}"
                    })
                
                # Now proceed with transcription
                try:
                    if method == "whisper":
                        print(f"[API] Starting enhanced Whisper transcription with model_size={model_size}, language={language}")
                        update_youtube_status(url, "transcribing", 50, "Transcribing audio with Whisper...")
                        
                        # Create a progress callback for Whisper
                        def progress_callback(message, progress):
                            print(f"[API] Transcription progress: {progress}% - {message}")
                            update_youtube_status(url, "transcribing", progress, message)
                            
                        text = enhanced_transcribe_with_whisper(audio_path, model_size, language, progress_callback)
                    else:
                        print(f"[API] Starting SpringLab transcription with language={language}")
                        update_youtube_status(url, "transcribing", 50, "Transcribing audio with SpringLab API...")
                        
                        # Create a progress callback for SpringLab
                        def progress_callback(message, progress):
                            print(f"[API] Transcription progress: {progress}% - {message}")
                            update_youtube_status(url, "transcribing", progress, message)
                            
                        text = transcribe_with_springlab(audio_path, language)
                    
                    # Create a downloadable file
                    output_file = save_text_file(text, OUTPUT_DIR, "youtube_video")
                    elapsed_time = time.time() - start_time
                    print(f"[API] YouTube transcription complete: {len(text)} characters in {elapsed_time:.2f} seconds")
                    
                    # Update status
                    update_youtube_status(url, "completed", 100, "Transcription complete")
                    
                    return jsonify({
                        "success": True, 
                        "text": text,
                        "download_url": f"/output/{output_file}"
                    })
                except Exception as e:
                    import traceback
                    print(f"[API] YouTube transcription error: {str(e)}")
                    print(traceback.format_exc())
                    # Update status
                    update_youtube_status(url, "failed", 100, f"Error transcribing YouTube audio: {str(e)}")
                    return jsonify({
                        "success": False, 
                        "error": f"Error transcribing YouTube audio: {str(e)}"
                    })
            else:
                # Transcribe channel videos
                engine_name = "whisper" if engine == "Whisper (Local)" else "springlab"
                print(f"[API] Transcribing YouTube channel: {channel_id} with {engine_name}")
                results = transcribe_youtube_channel(
                    channel_id, max_videos, engine_name, model_size, language
                )
                
                download_urls = {}
                for title, text in results.items():
                    output_file = save_text_file(text, OUTPUT_DIR, title)
                    download_urls[title] = f"/output/{output_file}"
                
                elapsed_time = time.time() - start_time
                print(f"[API] YouTube channel transcription complete: {len(results)} videos in {elapsed_time:.2f} seconds")
                return jsonify({
                    "success": True, 
                    "results": results,
                    "download_urls": download_urls
                })
        except Exception as e:
            import traceback
            print(f"[API] YouTube transcription error: {str(e)}")
            print(traceback.format_exc())
            if url:
                update_youtube_status(url, "failed", 100, f"Transcription error: {str(e)}")
            return jsonify({"success": False, "error": str(e)})
    
    except Exception as e:
        print(f"[API] General error in YouTube API route: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/job/<task_id>', methods=['GET'])
def api_job_status(task_id):
    """API endpoint to check the status of a background job."""
    try:
        status_file = os.path.join(JOBS_DIR, f"{task_id}.json")
        
        if not os.path.exists(status_file):
            return jsonify({"success": False, "error": "Job not found"})
            
        # Read job status
        import json
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            
        return jsonify({"success": True, "job": status_data})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/youtube-status', methods=['GET'])
def api_youtube_status():
    """API endpoint to check the status of YouTube transcriptions."""
    try:
        url = request.args.get('url')
        
        if not url or url not in youtube_transcription_status:
            return jsonify({
                "success": False, 
                "error": "No status found for the provided URL",
                "status": "unknown"
            })
            
        # Return the current status
        return jsonify({
            "success": True, 
            "status": youtube_transcription_status[url]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "status": "error"})

@app.route('/api/engines', methods=['GET'])
def api_engines():
    """API endpoint to get available engines and languages."""
    # Get available languages from transcriber
    whisper_languages, springlab_languages = get_available_languages()
    
    # Convert to format needed by frontend
    whisper_lang_list = [{"code": code, "name": name} for code, name in whisper_languages.items()]
    springlab_lang_list = [{"code": code, "name": name} for code, name in springlab_languages.items()]
    
    # Get GPU information
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
            free_mem = total_mem - allocated_mem
            gpu_info.append({
                "id": i,
                "name": gpu_name,
                "total_memory": f"{total_mem:.2f} GB",
                "free_memory": f"{free_mem:.2f} GB",
                "usage": f"{allocated_mem/total_mem*100:.1f}%"
            })
    
    return jsonify({
        "engines": [
            {
                "id": "Whisper (Local)", 
                "name": "Whisper (Local)", 
                "model_sizes": ["small", "medium", "large"]
            },
            {
                "id": "SpringLab API", 
                "name": "SpringLab API (Best for Indian languages)"
            }
        ],
        "whisper_languages": whisper_lang_list,
        "springlab_languages": springlab_lang_list,
        "gpu_info": gpu_info
    })

@app.route('/api/system-info', methods=['GET'])
def api_system_info():
    """API endpoint to get system information."""
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
            free_mem = total_mem - allocated_mem
            gpu_info.append({
                "id": i,
                "name": gpu_name,
                "total_memory": f"{total_mem:.2f} GB",
                "free_memory": f"{free_mem:.2f} GB",
                "usage": f"{allocated_mem/total_mem*100:.1f}%"
            })
    
    return jsonify({
        "gpu_info": gpu_info,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.time()
    })

if __name__ == "__main__":
    try:
        # Install cryptography package if not present (needed for adhoc SSL)
        try:
            import cryptography
        except ImportError:
            import subprocess
            import sys
            print("Installing cryptography package for SSL support...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography"])
            print("Cryptography package installed successfully.")
        
        print("Starting Flask app with HTTPS (self-signed certificate)...")
        app.run(host="0.0.0.0", port=8080, ssl_context="adhoc")
    except Exception as e:
        print(f"Error starting server with SSL: {str(e)}")
        print("Falling back to HTTP (microphone recording will not work)...")
        app.run(host="0.0.0.0", port=8080) 
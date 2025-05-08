# transcriber.py

import whisper
import tempfile
import os
import sounddevice as sd
import numpy as np
import threading
import queue
import scipy.io.wavfile
import requests
import shutil
from utils import (
    convert_to_wav, 
    download_youtube_audio, 
    chunk_audio, 
    chunk_audio_by_time,
    merge_transcriptions,
    select_available_gpu,
    release_gpu
)
import json
import sys
import torch
from pydub import AudioSegment
from pydub.silence import detect_silence

# Audio Chunker class for handling long audio files
class AudioChunker:
    def __init__(self, file_path, min_chunk_size=20000, max_chunk_size=60000, silence_thresh=-40, min_silence_len=700):
        """
        Initialize the audio chunker.
        
        Args:
            file_path: Path to audio file
            min_chunk_size: Minimum chunk size in milliseconds
            max_chunk_size: Maximum chunk size in milliseconds
            silence_thresh: Silence threshold in dB
            min_silence_len: Minimum silence length in ms to consider as a split point
        """
        self.file_path = file_path
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
        self.temp_dirs = []
        
    def chunk_audio(self):
        """
        Split audio file into chunks based on silence detection.
        
        Returns:
            List of paths to chunked audio files
        """
        print(f"[Chunker] Chunking audio file: {self.file_path}")
        try:
            # Ensure we have a WAV file
            wav_path = convert_to_wav(self.file_path)
            
            # Load the audio file
            audio = AudioSegment.from_file(wav_path)
            duration_sec = len(audio) / 1000
            print(f"[Chunker] Loaded audio duration: {duration_sec:.2f} seconds")
            
            # If audio is longer than 30 minutes, use time-based chunking directly
            if duration_sec > 1800:  # 30 minutes
                print("[Chunker] Long audio detected. Using time-based chunking (60 seconds each).")
                return chunk_audio_by_time(wav_path, chunk_size_ms=60000)
            
            # If audio is already short enough, return it as is
            if len(audio) <= self.max_chunk_size:
                print(f"[Chunker] Audio is already short ({len(audio)/1000:.1f}s), no chunking needed")
                return [wav_path]
                
            # Create a directory for chunks
            chunk_dir = tempfile.mkdtemp(prefix="audio_chunks_")
            self.temp_dirs.append(chunk_dir)
            print(f"[Chunker] Created temporary directory for chunks: {chunk_dir}")
            
            # Detect silence points
            print(f"[Chunker] Detecting silence points with threshold: {self.silence_thresh}dB, min length: {self.min_silence_len}ms")
            silence_points = detect_silence(audio, min_silence_len=self.min_silence_len, silence_thresh=self.silence_thresh)
            
            if not silence_points:
                print("[Chunker] No silence points detected, using time-based chunking")
                # If no silence detected, use time-based chunking
                temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
                self.temp_dirs.append(temp_dir)
                
                # Split by time
                num_chunks = max(1, len(audio) // self.max_chunk_size + (1 if len(audio) % self.max_chunk_size > 0 else 0))
                chunk_paths = []
                
                print(f"[Chunker] Creating {num_chunks} time-based chunks")
                for i in range(num_chunks):
                    start_ms = i * self.max_chunk_size
                    end_ms = min(start_ms + self.max_chunk_size, len(audio))
                    
                    chunk_audio = audio[start_ms:end_ms]
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                    chunk_audio.export(chunk_path, format="wav")
                    chunk_paths.append(chunk_path)
                
                print(f"[Chunker] Created {len(chunk_paths)} time-based chunks")
                return chunk_paths
            
            # Create optimal chunks that respect silence boundaries and size limits
            chunks = []
            start_pos = 0
            current_chunk_size = 0
            chunk_paths = []
            
            print(f"[Chunker] Creating optimal chunks from {len(silence_points)} silence points")
            for i, (silence_start, silence_end) in enumerate(silence_points):
                # If adding this segment would exceed max_chunk_size, create a chunk
                if start_pos < silence_start and (silence_start - start_pos) + current_chunk_size > self.max_chunk_size:
                    # Save current chunk if it's big enough
                    if current_chunk_size >= self.min_chunk_size:
                        chunk_audio = audio[start_pos:silence_start]
                        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunks)}.wav")
                        chunk_audio.export(chunk_path, format="wav")
                        chunk_paths.append(chunk_path)
                        chunks.append((start_pos, silence_start))
                        
                        # Reset for next chunk
                        start_pos = silence_start
                        current_chunk_size = 0
                
                # Add silence duration to current chunk size
                current_chunk_size += (silence_end - silence_start)
                
                # If we reach the last silence or processed enough content
                if i == len(silence_points) - 1 or current_chunk_size >= self.min_chunk_size:
                    # Find next suitable start point
                    next_start = silence_end
                    
                    # Create a chunk from start_pos to next_start
                    if next_start > start_pos:
                        chunk_audio = audio[start_pos:next_start]
                        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunks)}.wav")
                        chunk_audio.export(chunk_path, format="wav")
                        chunk_paths.append(chunk_path)
                        chunks.append((start_pos, next_start))
                    
                    # Reset for next chunk
                    start_pos = next_start
                    current_chunk_size = 0
            
            # Add any remaining audio
            if start_pos < len(audio):
                chunk_audio = audio[start_pos:]
                chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunks)}.wav")
                chunk_audio.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)
                chunks.append((start_pos, len(audio)))
            
            print(f"[Chunker] Created {len(chunk_paths)} chunks")
            for i, path in enumerate(chunk_paths):
                chunk_duration = AudioSegment.from_file(path).duration_seconds
                print(f"[Chunker] Chunk {i}: {chunk_duration:.1f} seconds ({os.path.getsize(path)/1024/1024:.1f} MB)")
                
            return chunk_paths
            
        except Exception as e:
            print(f"[Chunker] Error during chunking: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to chunk audio: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary directories and files."""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"[Chunker] Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    print(f"[Chunker] Error cleaning up temp directory: {str(e)}")

# Global variables for recording
audio_queue = queue.Queue()
recording_thread = None
is_recording = False
recorded_frames = []

# Cache for loaded Whisper models
_whisper_models = {}

# Safe wrapper for Whisper transcription
def safe_transcribe_with_whisper(file_path, model_size="small", language=None):
    """
    Safe transcription with Whisper model. 
    Catches fatal errors like SystemExit and returns user-friendly errors.
    """
    try:
        return transcribe_with_whisper(file_path, model_size, language)
    except SystemExit as e:
        print(f"[Fatal Error] SystemExit during Whisper transcription: {str(e)}")
        return "Whisper encountered a fatal error. Please try a smaller model or a different audio file."
    except Exception as e:
        import traceback
        print(f"[Error] Whisper transcription failed: {str(e)}")
        print(traceback.format_exc())
        return f"Whisper transcription failed: {str(e)}"

# Enhanced version using chunking and GPU
def enhanced_transcribe_with_whisper(file_path, model_size="medium", language="auto", progress_callback=None):
    """
    Enhanced version of whisper transcription that handles chunking for long audio files
    
    Args:
        file_path: Path to audio file
        model_size: Size of Whisper model
        language: Language code or "auto" for auto-detection
        progress_callback: Optional callback function to report progress (takes status message and progress percentage)
    """
    print(f"[Enhanced Whisper] Processing file: {file_path}")

    try:
        # Step 1: Split the audio into chunks
        if progress_callback:
            progress_callback("Analyzing audio file...", 35)
            
        chunker = AudioChunker(file_path)
        chunk_files = chunker.chunk_audio()
        
        if not chunk_files:
            raise RuntimeError(f"[Enhanced Whisper] Failed to chunk audio file: {file_path}")
        
        print(f"[Enhanced Whisper] File split into {len(chunk_files)} chunks")
        
        # Step 2: Select available GPU
        if progress_callback:
            progress_callback("Preparing transcription model...", 40)
            
        device_id = select_available_gpu()
        if device_id is not None:
            device = f"cuda:{device_id}"
            print(f"[Enhanced Whisper] Using GPU device: {device}")
        else:
            device = "cpu"
            print("[Enhanced Whisper] No GPU available, using CPU")
        
        # Step 3: Load model (with half precision if using GPU)
        print(f"Loading Whisper model: {model_size} on {device}")
        fp16 = device.startswith("cuda")
        model = whisper.load_model(model_size, device=device)
        
        # Step 4: Process each chunk and merge
        transcriptions = []
        for i, chunk_file in enumerate(chunk_files):
            if progress_callback:
                # Calculate progress percentage based on chunks
                chunk_progress = 45 + (i / len(chunk_files)) * 45  # 45% to 90%
                progress_callback(f"Transcribing part {i+1} of {len(chunk_files)}...", chunk_progress)
                
            print(f"[Enhanced Whisper] Processing chunk {i+1}/{len(chunk_files)}: {chunk_file}")
            
            # Process with appropriate options
            options = {
                "fp16": fp16,
                "language": language if language != "auto" else None,
            }
            
            if language == "auto":
                # First detect language, then transcribe with that language
                initial_result = model.transcribe(chunk_file, **options)
                detected_lang = initial_result["language"]
                print(f"[Enhanced Whisper] Detected language: {detected_lang}")
                options["language"] = detected_lang
                result = model.transcribe(chunk_file, **options)
            else:
                result = model.transcribe(chunk_file, **options)
            
            # Only append the text field, not the entire result dictionary
            transcriptions.append(result["text"])
            print(f"[Enhanced Whisper] Chunk {i+1} text: {result['text'][:50]}...")
        
        # Step 5: Merge the transcriptions
        if transcriptions:
            merged_result = merge_transcriptions(transcriptions)
            return merged_result
        else:
            raise RuntimeError("[Enhanced Whisper] No transcriptions were generated")
    
    except Exception as e:
        import traceback
        print(f"[Enhanced Whisper] Fatal error during chunking or transcription: {str(e)}")
        print(traceback.format_exc())
        raise RuntimeError(f"Enhanced Whisper failed - input may be too large or file corrupted: {str(e)}")
        
    finally:
        # Clean up any temporary files
        if 'chunker' in locals():
            chunker.cleanup()
        
        # Release GPU if we were using one
        if 'device' in locals() and device.startswith("cuda"):
            device_id = int(device.split(":")[-1])
            try:
                release_gpu(device_id)
                print(f"[Enhanced Whisper] Released GPU {device_id}")
            except:
                # This might fail if release_gpu isn't implemented yet
                pass

# Get available language options for both engines
def get_available_languages():
    """
    Returns available languages for both transcription engines.
    
    Returns:
        Tuple containing (whisper_languages, springlab_languages)
    """
    # Whisper languages (from whisper.tokenizer.LANGUAGES)
    whisper_languages = {
        'auto': 'Auto-detect',
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'mr': 'Marathi',
        'bn': 'Bengali', 
        'gu': 'Gujarati',
        'pa': 'Punjabi',
        'ur': 'Urdu',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ru': 'Russian'
    }
    
    # SpringLab languages (based on available options)
    springlab_languages = {
        'english': 'English',
        'hindi': 'Hindi',
        'tamil': 'Tamil',
        'telugu': 'Telugu',
        'kannada': 'Kannada',
        'malayalam': 'Malayalam',
        'marathi': 'Marathi',
        'bengali': 'Bengali',
        'gujarati': 'Gujarati',
        'punjabi': 'Punjabi',
        'urdu': 'Urdu'
    }
    
    return whisper_languages, springlab_languages

# Load Whisper Model with GPU support
def load_whisper_model(model_size="small", device="cuda"):
    """Load and cache Whisper model based on size and device."""
    global _whisper_models
    
    model_key = f"{model_size}_{device}"
    if model_key not in _whisper_models:
        print(f"Loading Whisper model: {model_size} on {device}")
        _whisper_models[model_key] = whisper.load_model(model_size, device=device)
    else:
        print(f"Using cached Whisper model: {model_size} on {device}")
    
    return _whisper_models[model_key]

# Original Transcribe using Whisper (kept for backward compatibility)
def transcribe_with_whisper(file_path, model_size="small", language=None):
    """
    Transcribe audio using OpenAI's Whisper (local).
    
    Args:
        file_path: Path to audio/video file
        model_size: Size of Whisper model ("small", "medium", "large")
        language: Language code to force (None for auto-detection)
    
    Returns:
        Transcribed text
    """
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = select_available_gpu()
    
    if device == "cuda" and gpu_id is not None:
        device = f"cuda:{gpu_id}"
        print(f"[Whisper] Using GPU device: {device}")
    else:
        print(f"[Whisper] Using device: {device}")
    
    # Load model with GPU support if available
    model = load_whisper_model(model_size, device)
    
    # Ensure input is WAV format
    wav_path = convert_to_wav(file_path)
    
    # Prepare transcription options
    transcribe_options = {
        "fp16": device.startswith("cuda"),  # Use fp16 for GPU
    }
    
    # Add language if specified (not auto-detect)
    if language and language.lower() != "auto":
        print(f"[Whisper] Forcing language: {language}")
        transcribe_options["language"] = language.lower()
    else:
        print("[Whisper] Using auto language detection")
    
    # Perform transcription with options
    result = model.transcribe(wav_path, **transcribe_options)
    
    # Log detected language if using auto-detection
    if not language or language.lower() == "auto":
        detected_lang = result.get("language", "unknown")
        print(f"[Whisper] Detected language: {detected_lang}")
    
    return result['text']

# Transcribe using SpringLab API
def transcribe_with_springlab(file_path, language="english"):
    """
    Transcribe audio using SpringLab API (IIT Madras).
    
    Args:
        file_path: Path to audio/video file
        language: Language of the audio (default: english)
    
    Returns:
        Transcribed text
    """
    import time

    # Debug info
    print(f"[SpringLab] Starting SpringLab transcription with {language} language")
    print(f"[SpringLab] Input file: {file_path}")
    
    # Updated endpoint based on the documentation
    url = "https://asr.iitm.ac.in/internal/asr/decode"
    print(f"[SpringLab] API endpoint: {url}")
    
    # Ensure input is WAV format
    start_time = time.time()
    wav_path = convert_to_wav(file_path)
    conversion_time = time.time() - start_time
    print(f"[SpringLab] Converted to WAV: {wav_path} (took {conversion_time:.2f}s)")

    # Prepare and send API request
    files = {
        'file': open(wav_path, 'rb'),
        'language': (None, language.lower()),
        'vtt': (None, 'false')
    }
    
    try:
        print(f"[SpringLab] Sending request to API with {language.lower()} language")
        api_start_time = time.time()
        response = requests.post(url, files=files)
        api_time = time.time() - api_start_time
        
        print(f"[SpringLab] API response received in {api_time:.2f}s")
        print(f"[SpringLab] Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"[SpringLab] Response data: {data}")
            
            if data.get("status") == "success":
                transcript = data.get("transcript", "")
                print(f"[SpringLab] Success! Transcript length: {len(transcript)}")
                return transcript
            else:
                error_reason = data.get('reason', 'Unknown error')
                print(f"[SpringLab] API returned error: {error_reason}")
                if "No speech detected" in error_reason:
                    return "No speech was detected in the recording. Please speak louder or adjust your microphone."
                return f"SpringLab API error: {error_reason}"
        else:
            print(f"[SpringLab] API error: {response.status_code} - {response.text}")
            return f"SpringLab API error: Status code {response.status_code} - {response.text}"
    except Exception as e:
        print(f"[SpringLab] Exception during API call: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"SpringLab API request failed: {e}"
    finally:
        # Close the file handle
        files['file'].close()
        print("[SpringLab] File handle closed")

# Audio callback function for recording
def audio_callback(indata, frames, time, status):
    """Callback for sounddevice to capture audio data."""
    global recorded_frames
    if status:
        print(f"Error in audio recording: {status}")
    if is_recording:
        recorded_frames.append(indata.copy())

# Start recording from microphone
def start_recording():
    """Start recording audio from microphone."""
    global recording_thread, is_recording, recorded_frames
    
    # Reset recorded frames
    recorded_frames = []
    is_recording = True
    
    # Start recording in a separate thread
    def record_audio():
        # Find a valid input device (microphone)
        input_device = None
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_device = idx
                print(f"Using input device: {dev['name']}")
                break

        if input_device is None:
            is_recording = False
            raise RuntimeError("No input (microphone) device found.")

        # Open stream with the correct input device
        with sd.InputStream(device=input_device, callback=audio_callback, channels=1, samplerate=16000, dtype='int16'):
            while is_recording:
                sd.sleep(100)
    
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    
    return True

# Stop recording from microphone
def stop_recording():
    """
    Stop recording and return the recorded audio data.
    
    Returns:
        numpy array of audio data
    """
    global is_recording, recording_thread, recorded_frames
    
    if not is_recording:
        return None
    
    # Stop recording
    is_recording = False
    
    # Wait for recording thread to finish
    if recording_thread and recording_thread.is_alive():
        recording_thread.join(timeout=1.0)
    
    # Check if we have recorded frames
    if not recorded_frames:
        return None
    
    # Combine all recorded frames
    audio_data = np.concatenate(recorded_frames, axis=0)
    return audio_data

# Transcribe recorded audio
def transcribe_recording(audio_data, engine="whisper", model_size="small", language="english"):
    """
    Transcribe the recorded audio data.
    
    Args:
        audio_data: Numpy array of audio data
        engine: Transcription engine ("whisper" or "springlab")
        model_size: Size of Whisper model (if using Whisper)
        language: Language for SpringLab API (if using SpringLab)
    
    Returns:
        Transcribed text
    """
    if audio_data is None or len(audio_data) == 0:
        return "No audio recorded or recording was too short."
    
    # Save to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        scipy.io.wavfile.write(temp_wav.name, 16000, audio_data)
        temp_wav_path = temp_wav.name
    
    # Transcribe using selected engine
    try:
        if engine == "whisper":
            text = enhanced_transcribe_with_whisper(temp_wav_path, model_size, language)
        else:
            text = transcribe_with_springlab(temp_wav_path, language)
        
        # Return a message if the transcription is empty
        if not text or text.strip() == "":
            if engine == "whisper":
                return "No speech detected. Please speak louder or adjust your microphone."
            return text
            
        return text
    except Exception as e:
        return f"Transcription error: {str(e)}"
    finally:
        # Clean up temp file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

# Transcribe YouTube single video
def transcribe_youtube_video(url, method="whisper", model_size="small", language="english", progress_callback=None):
    """
    Download and transcribe a YouTube video.
    
    Args:
        url: YouTube video URL
        method: Transcription method ("whisper" or "springlab")
        model_size: Size of Whisper model (if using "whisper")
        language: Language for SpringLab API (if using SpringLab)
        progress_callback: Optional callback function to report progress (takes status message and progress percentage)
    
    Returns:
        Transcribed text
    """
    # Download YouTube audio
    if progress_callback:
        progress_callback("Downloading YouTube audio...", 10)
        
    audio_path = download_youtube_audio(url)
    
    # Validate the downloaded file
    import os
    from pydub import AudioSegment
    
    # Check if file exists
    if not os.path.exists(audio_path):
        if progress_callback:
            progress_callback("Error: Downloaded YouTube audio file not found", 100)
        raise RuntimeError(f"Downloaded YouTube audio file not found: {audio_path}")
    
    # Check file size
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"[YouTube] Audio file size: {file_size_mb:.2f} MB")
    if file_size_mb < 0.5:  # Less than 500 KB
        if progress_callback:
            progress_callback(f"Error: Downloaded YouTube audio too small ({file_size_mb:.2f} MB)", 100)
        raise RuntimeError(f"Downloaded YouTube audio too small ({file_size_mb:.2f} MB)")
    
    # Check audio duration
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_sec = len(audio) / 1000
        print(f"[YouTube] Audio duration: {duration_sec:.2f} seconds")
        if duration_sec < 5:  # Less than 5 seconds
            if progress_callback:
                progress_callback(f"Error: Downloaded audio is too short ({duration_sec:.2f} seconds)", 100)
            raise RuntimeError(f"Downloaded audio is too short ({duration_sec:.2f} seconds)")
    except Exception as e:
        print(f"[YouTube] Warning: Could not check audio duration: {str(e)}")
        # Continue anyway as this is an optional check
    
    # If all validation checks pass, proceed with transcription
    print(f"[YouTube] Audio validation passed, proceeding with transcription")
    if progress_callback:
        progress_callback("Audio downloaded and validated. Preparing for transcription...", 30)
    
    # Transcribe using the specified method
    if method == "whisper":
        return enhanced_transcribe_with_whisper(audio_path, model_size, language, progress_callback)
    else:
        if progress_callback:
            progress_callback("Transcribing with SpringLab API...", 50)
        text = transcribe_with_springlab(audio_path, language)
        if progress_callback:
            progress_callback("Transcription complete!", 100)
        return text

# Transcribe multiple videos from a channel
def transcribe_youtube_channel(channel_id, max_videos, engine, model_size="small", language="english"):
    """
    Download and transcribe multiple videos from a YouTube channel.
    
    Args:
        channel_id: YouTube channel ID
        max_videos: Maximum number of videos to transcribe
        engine: Transcription engine ("whisper" or "springlab")
        model_size: Size of Whisper model (if using Whisper)
        language: Language for SpringLab API (if using SpringLab)
    
    Returns:
        Dictionary mapping video titles to transcribed text
    """
    audio_files = download_youtube_audio(channel_id, channel_mode=True, max_videos=max_videos)
    results = {}
    for title, path in audio_files.items():
        if engine == "whisper":
            text = enhanced_transcribe_with_whisper(path, model_size, language)
        else:
            text = transcribe_with_springlab(path, language)
        results[title] = text
    return results 

# Transcriber class for OOP usage
class Transcriber:
    def __init__(self):
        self.whisper_model = None
        
    def transcribe(self, audio_path, engine="Whisper (Local)", model_size="small", language="english"):
        """
        Transcribe audio file using the selected engine.
        
        Args:
            audio_path: Path to audio file
            engine: Transcription engine ("Whisper (Local)" or "SpringLab API")
            model_size: Size of Whisper model (small, medium, large)
            language: Language code or name
            
        Returns:
            Transcribed text
        """
        if engine == "Whisper (Local)":
            return enhanced_transcribe_with_whisper(audio_path, model_size, language)
        else:
            return transcribe_with_springlab(audio_path, language) 
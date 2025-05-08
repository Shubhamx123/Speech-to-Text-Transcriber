# utils.py

import os
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
import yt_dlp
import numpy as np
import math
import torch
import threading

# GPU locking mechanism to prevent race conditions
_gpu_lock = threading.Lock()
_gpu_busy = set()

# Convert audio/video file to WAV format
def convert_to_wav(input_path):
    """
    Convert any audio/video file to WAV format for processing.
    
    Args:
        input_path: Path to input audio/video file
        
    Returns:
        Path to converted WAV file
    """
    if input_path.lower().endswith(".wav"):
        return input_path  # Already WAV format, no conversion needed
        
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_file(input_path)
        # Create temporary WAV file
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        # Export to WAV format
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        raise Exception(f"Error converting file to WAV: {str(e)}")

# New function: Chunk audio file based on silences
def chunk_audio(audio_path, min_chunk_size=20000, max_chunk_size=60000, silence_thresh=-40, min_silence_len=700):
    """
    Split audio file into chunks based on silence detection.
    
    Args:
        audio_path: Path to audio file
        min_chunk_size: Minimum chunk size in milliseconds (default: 20 seconds)
        max_chunk_size: Maximum chunk size in milliseconds (default: 60 seconds)
        silence_thresh: Silence threshold in dB (default: -40dB)
        min_silence_len: Minimum silence length in ms to consider as a split point
        
    Returns:
        List of paths to chunked audio files
    """
    print(f"[Chunker] Chunking audio file: {audio_path}")
    try:
        # Ensure we have a WAV file
        wav_path = convert_to_wav(audio_path)
        # Load the audio file
        audio = AudioSegment.from_file(wav_path)
        
        # If audio is already short enough, return it as is
        if len(audio) <= max_chunk_size:
            print(f"[Chunker] Audio is already short ({len(audio)/1000:.1f}s), no chunking needed")
            return [wav_path]
            
        # Create a directory for chunks
        chunk_dir = tempfile.mkdtemp(prefix="audio_chunks_")
        print(f"[Chunker] Created temporary directory for chunks: {chunk_dir}")
        
        # Detect silence points
        print(f"[Chunker] Detecting silence points with threshold: {silence_thresh}dB, min length: {min_silence_len}ms")
        silence_points = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        
        if not silence_points:
            print("[Chunker] No silence points detected, using time-based chunking")
            # If no silence detected, use time-based chunking
            return chunk_audio_by_time(wav_path, chunk_size_ms=max_chunk_size)
        
        # Create optimal chunks that respect silence boundaries and size limits
        chunks = []
        start_pos = 0
        current_chunk_size = 0
        chunk_paths = []
        
        print(f"[Chunker] Creating optimal chunks from {len(silence_points)} silence points")
        for i, (silence_start, silence_end) in enumerate(silence_points):
            # If adding this segment would exceed max_chunk_size, create a chunk
            if start_pos < silence_start and (silence_start - start_pos) + current_chunk_size > max_chunk_size:
                # Save current chunk if it's big enough
                if current_chunk_size >= min_chunk_size:
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
            if i == len(silence_points) - 1 or current_chunk_size >= min_chunk_size:
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
        # If chunking fails, return the original file
        return [audio_path]

# Alternative chunking based purely on time
def chunk_audio_by_time(audio_path, chunk_size_ms=60000):
    """
    Split audio file into chunks of equal time duration.
    
    Args:
        audio_path: Path to audio file
        chunk_size_ms: Chunk size in milliseconds (default: 60 seconds)
        
    Returns:
        List of paths to chunked audio files
    """
    print(f"[Chunker] Time-based chunking of audio file: {audio_path}")
    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        
        # If audio is already short enough, return it as is
        if len(audio) <= chunk_size_ms:
            print(f"[Chunker] Audio is already short ({len(audio)/1000:.1f}s), no chunking needed")
            return [audio_path]
        
        # Create a directory for chunks
        chunk_dir = tempfile.mkdtemp(prefix="audio_chunks_")
        print(f"[Chunker] Created temporary directory for chunks: {chunk_dir}")
        
        # Calculate number of chunks
        num_chunks = math.ceil(len(audio) / chunk_size_ms)
        chunk_paths = []
        
        # Split audio into chunks
        for i in range(num_chunks):
            start_ms = i * chunk_size_ms
            end_ms = min(start_ms + chunk_size_ms, len(audio))
            
            chunk_audio = audio[start_ms:end_ms]
            chunk_path = os.path.join(chunk_dir, f"chunk_{i}.wav")
            chunk_audio.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)
        
        print(f"[Chunker] Created {len(chunk_paths)} time-based chunks")
        return chunk_paths
        
    except Exception as e:
        print(f"[Chunker] Error during time-based chunking: {str(e)}")
        import traceback
        traceback.print_exc()
        # If chunking fails, return the original file
        return [audio_path]

# Merge transcriptions from multiple chunks
def merge_transcriptions(transcriptions):
    """
    Merge multiple chunk transcriptions into a single cohesive transcript.
    
    Args:
        transcriptions: List of transcription text strings
        
    Returns:
        Merged transcription text
    """
    if not transcriptions:
        return ""
    
    if len(transcriptions) == 1:
        return transcriptions[0]
    
    # Join all transcriptions with a space in between
    print(f"[Merge] Merging {len(transcriptions)} text chunks")
    merged = " ".join(transcriptions)
    
    # Clean up potential duplicate phrases at chunk boundaries
    # This is a simple approach - more sophisticated NLP could be used here
    print(f"[Merge] Final merged text length: {len(merged)} characters")
    return merged

# Function to select available GPU
def select_available_gpu():
    """
    Selects the GPU with the most available memory and locks it for exclusive use.
    Returns the GPU ID (int) or None if no GPU is available.
    """
    if not torch.cuda.is_available():
        print("[GPU Selector] CUDA not available")
        return None
    
    with _gpu_lock:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("[GPU Selector] No GPUs detected")
            return None
            
        # Check which GPUs are already locked
        available_gpus = []
        for gpu_id in range(num_gpus):
            # Skip if this GPU is already locked by another process
            if gpu_id in _gpu_busy:
                continue
                
            # Get memory information
            try:
                free_memory = round(torch.cuda.get_device_properties(gpu_id).total_memory / 1e9, 2)
                used_memory = round(torch.cuda.memory_allocated(gpu_id) / 1e9, 2)
                available_memory = round(free_memory - used_memory, 2)
                
                print(f"[GPU Selector] GPU {gpu_id}: {available_memory}GB free out of {free_memory}GB")
                available_gpus.append((available_memory, gpu_id))
            except Exception as e:
                print(f"[GPU Selector] Error checking GPU {gpu_id}: {str(e)}")
        
        if not available_gpus:
            print("[GPU Selector] All GPUs are busy or unavailable")
            return None
            
        # Select GPU with most available memory
        best_gpu = max(available_gpus, key=lambda x: x[0])[1]
        
        # Lock this GPU
        _gpu_busy.add(best_gpu)
        print(f"[GPU Selector] Selected GPU {best_gpu} and locked it for exclusive use")
        return best_gpu

def release_gpu(gpu_id):
    """
    Releases a previously locked GPU.
    """
    with _gpu_lock:
        if gpu_id in _gpu_busy:
            _gpu_busy.remove(gpu_id)
            print(f"[GPU Selector] Released GPU {gpu_id}")
        else:
            print(f"[GPU Selector] Warning: Attempted to release GPU {gpu_id} which was not locked")

# Download YouTube audio (single video or multiple from channel)
def download_youtube_audio(url_or_channel_id, channel_mode=False, max_videos=5):
    """
    Download audio from YouTube videos.
    
    Args:
        url_or_channel_id: YouTube video URL or channel ID
        channel_mode: If True, treat url_or_channel_id as a channel ID
        max_videos: Maximum number of videos to download (if channel_mode=True)
        
    Returns:
        If channel_mode=False: Path to downloaded audio
        If channel_mode=True: Dictionary mapping video titles to audio paths
    """
    print(f"[YouTube] Starting download: {url_or_channel_id}")
    print(f"[YouTube] Channel mode: {channel_mode}, Max videos: {max_videos}")
    
    # Create temporary directory for downloads
    output_dir = tempfile.mkdtemp()
    print(f"[YouTube] Output directory: {output_dir}")
    
    # Use sanitized filenames (only alphanumeric characters)
    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')
    print(f"[YouTube] Output template: {output_template}")
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': False,
        'noplaylist': not channel_mode,
        'extract_flat': 'in_playlist' if channel_mode else False,
        'playlistend': max_videos if channel_mode else None,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'verbose': True
    }

    audio_files = {}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if channel_mode:
                # Handle channel - download multiple videos
                channel_url = f"https://www.youtube.com/channel/{url_or_channel_id}"
                print(f"[YouTube] Extracting channel info: {channel_url}")
                info = ydl.extract_info(channel_url, download=False)
                entries = info.get('entries', [])[:max_videos]
                
                print(f"[YouTube] Found {len(entries)} videos in channel")
                for i, entry in enumerate(entries):
                    print(f"[YouTube] Downloading video {i+1}/{len(entries)}: {entry.get('title', entry.get('id'))}")
                    ydl.download([entry['url']])
                    video_id = entry.get('id')
                    title = entry.get('title', video_id)
                    audio_path = os.path.join(output_dir, f"{video_id}.wav")
                    
                    # Verify file exists
                    if os.path.exists(audio_path):
                        print(f"[YouTube] Downloaded successfully: {audio_path} ({os.path.getsize(audio_path)} bytes)")
                        audio_files[title] = audio_path
                    else:
                        print(f"[YouTube] Failed to download: {audio_path}")
            else:
                # Handle single video
                try:
                    # Clean URL by removing timestamp parameters
                    if '&t=' in url_or_channel_id:
                        url_or_channel_id = url_or_channel_id.split('&t=')[0]
                    if '?t=' in url_or_channel_id:
                        url_or_channel_id = url_or_channel_id.split('?t=')[0]
                    
                    # Remove any other query parameters that might cause issues
                    if '&' in url_or_channel_id:
                        base_url = url_or_channel_id.split('?')[0]
                        query_params = url_or_channel_id.split('?')[1].split('&')
                        # Keep only the video ID parameter (v=)
                        video_param = next((p for p in query_params if p.startswith('v=')), None)
                        if video_param:
                            url_or_channel_id = f"{base_url}?{video_param}"
                    
                    print(f"[YouTube] Cleaned URL: {url_or_channel_id}")
                    
                    # First try to get info without downloading
                    info = ydl.extract_info(url_or_channel_id, download=False)
                    video_id = info.get('id')
                    title = info.get('title', video_id)
                    
                    print(f"[YouTube] Video info retrieved: ID={video_id}, Title={title}")
                    
                    # Then download the audio
                    ydl.download([url_or_channel_id])
                    
                    # Look for the downloaded file
                    audio_path = os.path.join(output_dir, f"{video_id}.wav")
                    
                    # Verify file exists
                    if os.path.exists(audio_path):
                        print(f"[YouTube] Downloaded successfully: {audio_path} ({os.path.getsize(audio_path)/1024/1024:.2f} MB)")
                        audio_files = audio_path
                    else:
                        # Try to find any WAV file that was created
                        print(f"[YouTube] File not found: {audio_path}")
                        print(f"[YouTube] Directory contents: {os.listdir(output_dir)}")
                        wav_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
                        if wav_files:
                            audio_path = os.path.join(output_dir, wav_files[0])
                            print(f"[YouTube] Found alternative file: {audio_path}")
                            audio_files = audio_path
                        else:
                            raise Exception(f"No WAV file found in {output_dir} after download")
                
                except Exception as e:
                    print(f"[YouTube] Error during first download attempt: {str(e)}")
                    # Fallback method: try with generic options
                    try:
                        print(f"[YouTube] Trying fallback download method...")
                        fallback_opts = {
                            'format': 'bestaudio/best',
                            'outtmpl': os.path.join(output_dir, 'audio.%(ext)s'),
                            'postprocessors': [{
                                'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'wav',
                            }],
                        }
                        with yt_dlp.YoutubeDL(fallback_opts) as ydl2:
                            ydl2.download([url_or_channel_id])
                            
                        # Look for any WAV file
                        wav_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
                        if wav_files:
                            audio_path = os.path.join(output_dir, wav_files[0])
                            print(f"[YouTube] Fallback download successful: {audio_path}")
                            audio_files = audio_path
                        else:
                            raise Exception("Fallback download failed to produce a WAV file")
                    except Exception as e2:
                        print(f"[YouTube] Fallback download also failed: {str(e2)}")
                        raise Exception(f"All download attempts failed: {str(e)} | {str(e2)}")
                
        return audio_files
    except Exception as e:
        import traceback
        print(f"[YouTube] Error: {str(e)}")
        print(traceback.format_exc())
        raise Exception(f"Error downloading YouTube audio: {str(e)}")

# Save text to file
def save_text_file(text, output_dir, base_name):
    """
    Save transcribed text to file and return the filename.
    
    Args:
        text: Transcribed text to save
        output_dir: Directory to save the file in
        base_name: Base name for the file
        
    Returns:
        Filename (without path)
    """
    # Create a safe filename from the base name
    safe_base = "".join(c for c in base_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    filename = f"{safe_base}_transcription.txt"
    file_path = os.path.join(output_dir, filename)
    
    # Write text to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    return filename 
import os
import tempfile
import time
import logging
import json
import requests
import sounddevice as sd
import numpy as np
import threading
import wave
import subprocess
from utils import convert_to_wav, download_youtube_audio, download_all_channel_videos, get_channel_info

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('transcriber')

# Check if Whisper is available
WHISPER_AVAILABLE = True
try:
    import whisper
    whisper.load_model  # Test if the function exists
except (ImportError, AttributeError):
    logger.error("Failed to import OpenAI Whisper package. Whisper transcription will be disabled.")
    logger.error("To enable Whisper, install it separately with: pip install openai-whisper")
    WHISPER_AVAILABLE = False

# SpringLab API endpoint
SPRING_LAB_API_URL = "https://asr.iitm.ac.in/internal/asr/decode"

# Supported languages by SpringLab
SPRING_LAB_LANGUAGES = [
    "english", "hindi", "tamil", "telugu", "kannada", "gujarati", 
    "marathi", "malayalam", "bengali", "urdu", "odia", "punjabi", "assamese"
]

class WhisperTranscriber:
    """Class for transcribing audio using OpenAI's Whisper model."""
    
    def __init__(self, model_name="base"):
        """Initialize the transcriber with the specified model.
        
        Args:
            model_name (str): Name of the Whisper model to use
                             (tiny, base, small, medium, large)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper is not available. Please install it with: pip install openai-whisper")
        
        self.model_name = model_name
        self.model = None
        self.language = None  # Auto-detect language if None
        self.task = "transcribe"  # Can be "transcribe" or "translate" (to English)
        
        logger.info(f"Initializing Whisper with model: {model_name}")
    
    def _load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model loaded: {self.model_name}")
    
    def transcribe_audio_file(self, file_path):
        """Transcribe an audio file using Whisper.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            dict: Transcription results
        """
        start_time = time.time()
        logger.info(f"Transcribing file: {file_path}")
        
        # Load the model if not already loaded
        self._load_model()
        
        # Convert to WAV if needed
        wav_file = convert_to_wav(file_path)
        
        # Transcribe the audio
        kwargs = {}
        if self.language:
            kwargs['language'] = self.language
        
        result = self.model.transcribe(
            wav_file,
            task=self.task,
            **kwargs
        )
        
        # If the input file was converted, and it's not the same as the original,
        # remove the temporary file
        if wav_file != file_path and os.path.exists(wav_file):
            try:
                os.remove(wav_file)
            except:
                pass
        
        # Calculate time taken
        time_taken = time.time() - start_time
        
        # Add extra information to the result
        result['time_taken'] = time_taken
        
        logger.info(f"Transcription completed in {time_taken:.2f} seconds")
        
        return result
    
    def transcribe_video_file(self, file_path):
        """Transcribe a video file by extracting the audio first.
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            dict: Transcription results
        """
        logger.info(f"Extracting audio from video: {file_path}")
        
        # Create a temporary file for the audio
        temp_dir = tempfile.gettempdir()
        temp_audio = os.path.join(temp_dir, f"video_audio_{os.path.basename(file_path)}.wav")
        
        try:
            # Extract audio using FFmpeg
            cmd = [
                'ffmpeg',
                '-y',
                '-i', file_path,
                '-vn',
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                temp_audio
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Transcribe the extracted audio
            result = self.transcribe_audio_file(temp_audio)
            
            # Clean up
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing video: {e}")
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            raise
    
    def transcribe_youtube_video(self, url):
        """Download and transcribe a YouTube video.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            dict: Transcription results
        """
        logger.info(f"Downloading and transcribing YouTube video: {url}")
        
        try:
            # Download the audio
            audio_path = download_youtube_audio(url)
            
            # Transcribe the audio
            result = self.transcribe_audio_file(audio_path)
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing YouTube video: {e}")
            raise
    
    def get_channel_info(self, channel_id):
        """Get information about a YouTube channel.
        
        Args:
            channel_id (str): YouTube channel ID or URL
            
        Returns:
            dict: Channel information
        """
        try:
            return get_channel_info(channel_id)
        except Exception as e:
            logger.error(f"Error getting channel info: {e}")
            raise
    
    def transcribe_youtube_channel(self, channel_id, limit=5, max_chars=5000, 
                                 start_index=0, batch_size=10, 
                                 progress_callback=None, sort_by="newest"):
        """Download and transcribe videos from a YouTube channel with pagination.
        
        Args:
            channel_id (str): YouTube channel ID or URL
            limit (int): Maximum number of videos to transcribe
            max_chars (int): Maximum characters to include per transcription
            start_index (int): Starting index for pagination
            batch_size (int): Number of videos to process in each batch
            progress_callback (callable): Function to call with progress updates
            sort_by (str): Sort order - "newest", "oldest", or "popular"
            
        Returns:
            list: List of transcription results
        """
        logger.info(f"Transcribing up to {limit} videos from channel: {channel_id}")
        
        # Get channel info to show total videos count
        try:
            channel_info = self.get_channel_info(channel_id)
            logger.info(f"Channel: {channel_info['title']}, Total videos: {channel_info['video_count']}")
            
            # If progress callback exists, send initial channel info
            if progress_callback:
                progress_callback(0, limit, None, "channel_info", channel_info)
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            channel_info = {"title": "Unknown Channel", "video_count": "Unknown"}
            
            # If progress callback exists, send error channel info
            if progress_callback:
                progress_callback(0, limit, None, "channel_info", channel_info)
        
        # Calculate how many batches we'll need
        remaining = min(limit, channel_info.get('video_count', limit))
        current_index = start_index
        results = []
        
        # Define progress update function
        def update_download_progress(index, total, url, status, error=None):
            if progress_callback:
                # Convert local batch index to global index
                global_index = current_index + index
                progress_callback(global_index, limit, url, f"download_{status}", error)
        
        # Process in batches
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            logger.info(f"Processing batch of {current_batch_size} videos starting at index {current_index}")
            
            # If progress callback exists, send batch start info
            if progress_callback:
                progress_callback(current_index, limit, None, "batch_start", {
                    "batch_index": current_index // batch_size,
                    "batch_size": current_batch_size,
                    "start_index": current_index
                })
            
            try:
                # Download batch of videos
                audio_files = download_all_channel_videos(
                    channel_id, 
                    limit=current_batch_size,
                    start_index=current_index,
                    sort_by=sort_by,
                    progress_callback=update_download_progress
                )
                
                # Transcribe each audio file in this batch
                for i, audio_path in enumerate(audio_files):
                    try:
                        # If progress callback exists, update transcription progress
                        if progress_callback:
                            global_index = current_index + i
                            progress_callback(global_index, limit, audio_path, "transcribing")
                            
                        # Transcribe the audio
                        result = self.transcribe_audio_file(audio_path)
                        
                        # Add video index information
                        result['video_index'] = current_index + i
                        
                        # Limit the transcription text if needed
                        if len(result.get('text', '')) > max_chars:
                            result['text'] = result['text'][:max_chars] + '...'
                        
                        results.append(result)
                        
                        # If progress callback exists, update completion
                        if progress_callback:
                            global_index = current_index + i
                            progress_callback(global_index, limit, audio_path, "completed", result)
                        
                        # Clean up
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                            
                    except Exception as e:
                        logger.error(f"Error transcribing {audio_path}: {e}")
                        
                        # Add error result
                        error_result = {
                            'error': str(e),
                            'video_index': current_index + i
                        }
                        results.append(error_result)
                        
                        # If progress callback exists, update error
                        if progress_callback:
                            global_index = current_index + i
                            progress_callback(global_index, limit, audio_path, "error", str(e))
                
                # Update counters for next batch
                current_index += current_batch_size
                remaining -= current_batch_size
                
                # If progress callback exists, send batch end info
                if progress_callback:
                    progress_callback(current_index, limit, None, "batch_end", {
                        "batch_index": current_index // batch_size - 1,
                        "completed": len(results),
                        "remaining": remaining
                    })
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                
                # If progress callback exists, send batch error info
                if progress_callback:
                    progress_callback(current_index, limit, None, "batch_error", str(e))
                
                # Add batch error result
                batch_error = {
                    'error': f"Batch processing error: {str(e)}",
                    'video_index': current_index
                }
                results.append(batch_error)
                
                # Move to next batch
                current_index += current_batch_size
                remaining -= current_batch_size
            
        return results
    
    def record_and_transcribe(self, duration=5, sample_rate=16000):
        """Record audio and transcribe it.
        
        Args:
            duration (int): Duration to record in seconds
            sample_rate (int): Sample rate for recording
            
        Returns:
            dict: Transcription results
        """
        logger.info(f"Recording audio for {duration} seconds")
        
        # Create an array to store the recording
        recording = np.zeros((duration * sample_rate, 1), dtype=np.float32)
        
        # Record audio
        sd.rec(
            frames=recording.shape[0],
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            out=recording
        )
        sd.wait()
        
        # Save the recording to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")
        
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        
        # Transcribe the recording
        result = self.transcribe_audio_file(temp_wav)
        
        # Clean up
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        return result

class SpringLabASR:
    """Class for transcribing audio using SpringLab ASR API."""
    
    def __init__(self):
        """Initialize the SpringLab ASR transcriber."""
        self.api_url = SPRING_LAB_API_URL
        self.language = None  # Auto-detect language if None
        logger.info("Initializing SpringLab ASR")
    
    def get_supported_languages(self):
        """Return list of supported languages."""
        return SPRING_LAB_LANGUAGES
    
    def get_channel_info(self, channel_id):
        """Get information about a YouTube channel.
        
        Args:
            channel_id (str): YouTube channel ID or URL
            
        Returns:
            dict: Channel information
        """
        try:
            return get_channel_info(channel_id)
        except Exception as e:
            logger.error(f"Error getting channel info: {e}")
            raise
    
    def transcribe_audio_file(self, file_path):
        """Transcribe an audio file using SpringLab ASR API.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            dict: Transcription results
        """
        start_time = time.time()
        logger.info(f"Transcribing file with SpringLab: {file_path}")
        
        # Convert to WAV if needed
        wav_file = convert_to_wav(file_path)
        
        try:
            # Read the audio file
            with open(wav_file, 'rb') as f:
                audio_data = f.read()
            
            # Prepare API request
            files = {'file': ('audio.wav', audio_data, 'audio/wav')}
            data = {}
            
            if self.language:
                data['lang'] = self.language
            
            # Make API request
            logger.info(f"Sending request to SpringLab API with language: {self.language}")
            response = requests.post(self.api_url, files=files, data=data)
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                logger.info(f"SpringLab API response: {result}")
                
                # Add time taken
                time_taken = time.time() - start_time
                result['time_taken'] = time_taken
                
                logger.info(f"Transcription completed in {time_taken:.2f} seconds")
                return result
            else:
                error_msg = f"API request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
            
        except Exception as e:
            logger.error(f"Error transcribing with SpringLab: {e}")
            return {"error": str(e)}
            
        finally:
            # Clean up
            if wav_file != file_path and os.path.exists(wav_file):
                try:
                    os.remove(wav_file)
                except:
                    pass
    
    def transcribe_video_file(self, file_path):
        """Transcribe a video file by extracting the audio first.
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            dict: Transcription results
        """
        logger.info(f"Extracting audio from video: {file_path}")
        
        # Create a temporary file for the audio
        temp_dir = tempfile.gettempdir()
        temp_audio = os.path.join(temp_dir, f"video_audio_{os.path.basename(file_path)}.wav")
        
        try:
            # Extract audio using FFmpeg
            cmd = [
                'ffmpeg',
                '-y',
                '-i', file_path,
                '-vn',
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                temp_audio
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Transcribe the extracted audio
            result = self.transcribe_audio_file(temp_audio)
            
            # Clean up
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing video: {e}")
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            raise
    
    def transcribe_youtube_video(self, url):
        """Download and transcribe a YouTube video.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            dict: Transcription results
        """
        logger.info(f"Downloading and transcribing YouTube video: {url}")
        
        try:
            # Download the audio
            audio_path = download_youtube_audio(url)
            
            # Transcribe the audio
            result = self.transcribe_audio_file(audio_path)
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing YouTube video: {e}")
            raise
    
    def transcribe_youtube_channel(self, channel_id, limit=5, max_chars=5000, 
                                 start_index=0, batch_size=10, 
                                 progress_callback=None, sort_by="newest"):
        """Download and transcribe videos from a YouTube channel with pagination.
        
        Args:
            channel_id (str): YouTube channel ID or URL
            limit (int): Maximum number of videos to transcribe
            max_chars (int): Maximum characters to include per transcription
            start_index (int): Starting index for pagination
            batch_size (int): Number of videos to process in each batch
            progress_callback (callable): Function to call with progress updates
            sort_by (str): Sort order - "newest", "oldest", "popular"
            
        Returns:
            list: List of transcription results
        """
        logger.info(f"Transcribing up to {limit} videos from channel: {channel_id}")
        
        # Get channel info to show total videos count
        try:
            channel_info = self.get_channel_info(channel_id)
            logger.info(f"Channel: {channel_info['title']}, Total videos: {channel_info['video_count']}")
            
            # If progress callback exists, send initial channel info
            if progress_callback:
                progress_callback(0, limit, None, "channel_info", channel_info)
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            channel_info = {"title": "Unknown Channel", "video_count": "Unknown"}
            
            # If progress callback exists, send error channel info
            if progress_callback:
                progress_callback(0, limit, None, "channel_info", channel_info)
        
        # Calculate how many batches we'll need
        remaining = min(limit, channel_info.get('video_count', limit))
        current_index = start_index
        results = []
        
        # Define progress update function
        def update_download_progress(index, total, url, status, error=None):
            if progress_callback:
                # Convert local batch index to global index
                global_index = current_index + index
                progress_callback(global_index, limit, url, f"download_{status}", error)
        
        # Process in batches
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            logger.info(f"Processing batch of {current_batch_size} videos starting at index {current_index}")
            
            # If progress callback exists, send batch start info
            if progress_callback:
                progress_callback(current_index, limit, None, "batch_start", {
                    "batch_index": current_index // batch_size,
                    "batch_size": current_batch_size,
                    "start_index": current_index
                })
            
            try:
                # Download batch of videos
                audio_files = download_all_channel_videos(
                    channel_id, 
                    limit=current_batch_size,
                    start_index=current_index,
                    sort_by=sort_by,
                    progress_callback=update_download_progress
                )
                
                # Transcribe each audio file in this batch
                for i, audio_path in enumerate(audio_files):
                    try:
                        # If progress callback exists, update transcription progress
                        if progress_callback:
                            global_index = current_index + i
                            progress_callback(global_index, limit, audio_path, "transcribing")
                            
                        # Transcribe the audio
                        result = self.transcribe_audio_file(audio_path)
                        
                        # Add video index information
                        result['video_index'] = current_index + i
                        
                        # Limit the transcription text if needed
                        if 'transcript' in result and len(result['transcript']) > max_chars:
                            result['transcript'] = result['transcript'][:max_chars] + '...'
                        
                        results.append(result)
                        
                        # If progress callback exists, update completion
                        if progress_callback:
                            global_index = current_index + i
                            progress_callback(global_index, limit, audio_path, "completed", result)
                        
                        # Clean up
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                            
                    except Exception as e:
                        logger.error(f"Error transcribing {audio_path}: {e}")
                        
                        # Add error result
                        error_result = {
                            'error': str(e),
                            'video_index': current_index + i
                        }
                        results.append(error_result)
                        
                        # If progress callback exists, update error
                        if progress_callback:
                            global_index = current_index + i
                            progress_callback(global_index, limit, audio_path, "error", str(e))
                
                # Update counters for next batch
                current_index += current_batch_size
                remaining -= current_batch_size
                
                # If progress callback exists, send batch end info
                if progress_callback:
                    progress_callback(current_index, limit, None, "batch_end", {
                        "batch_index": current_index // batch_size - 1,
                        "completed": len(results),
                        "remaining": remaining
                    })
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                
                # If progress callback exists, send batch error info
                if progress_callback:
                    progress_callback(current_index, limit, None, "batch_error", str(e))
                
                # Add batch error result
                batch_error = {
                    'error': f"Batch processing error: {str(e)}",
                    'video_index': current_index
                }
                results.append(batch_error)
                
                # Move to next batch
                current_index += current_batch_size
                remaining -= current_batch_size
            
        return results
    
    def record_and_transcribe(self, duration=5, sample_rate=16000):
        """Record audio and transcribe it.
        
        Args:
            duration (int): Duration to record in seconds
            sample_rate (int): Sample rate for recording
            
        Returns:
            dict: Transcription results
        """
        logger.info(f"Recording audio for SpringLab transcription: {duration} seconds")
        
        # Create an array to store the recording
        recording = np.zeros((duration * sample_rate, 1), dtype=np.float32)
        
        # Record audio
        sd.rec(
            frames=recording.shape[0],
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            out=recording
        )
        sd.wait()
        
        # Save the recording to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")
        
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        
        # Transcribe the recording
        result = self.transcribe_audio_file(temp_wav)
        
        # Clean up
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        return result

def live_record_with_callback(callback, stop_event, max_duration=30, sample_rate=16000):
    """Record audio with a callback to update UI.
    
    Args:
        callback (function): Function to call with recording progress updates
        stop_event (threading.Event): Event to signal when to stop recording
        max_duration (int): Maximum duration to record in seconds
        sample_rate (int): Sample rate for recording
        
    Returns:
        str: Path to the recorded audio file
    """
    rec_logger = logging.getLogger('feature.recording.live')
    rec_logger.debug(f"Starting live recording with max_duration={max_duration}, sample_rate={sample_rate}")
    
    # Create an array to store the recording
    max_frames = max_duration * sample_rate
    recording = np.zeros((max_frames, 1), dtype=np.float32)
    frames_recorded = 0
    
    # Define callback function for audio stream
    def audio_callback(indata, frames, time, status):
        nonlocal frames_recorded
        if status:
            rec_logger.warning(f"Audio callback status: {status}")
        
        if indata is None:
            rec_logger.error("Audio callback received None indata")
            return
            
        if len(indata) == 0:
            rec_logger.warning("Audio callback received empty indata")
            return
            
        rec_logger.debug(f"Audio callback: received {len(indata)} frames, max={indata.max()}, min={indata.min()}")
        
        remaining = max_frames - frames_recorded
        if remaining <= 0:
            rec_logger.debug("Max frames reached, ignoring new audio data")
            return
        
        # Store the new frames
        frames_to_add = min(remaining, len(indata))
        recording[frames_recorded:frames_recorded + frames_to_add] = indata[:frames_to_add]
        frames_recorded += frames_to_add
        
        # Call the progress callback
        try:
            callback(frames_recorded / sample_rate)
        except Exception as e:
            rec_logger.error(f"Error in progress callback: {e}", exc_info=True)
    
    # Check audio devices before recording
    try:
        devices = sd.query_devices()
        rec_logger.debug(f"Available audio devices: {devices}")
        
        default_device = sd.query_devices(kind='input')
        rec_logger.debug(f"Default input device: {default_device}")
        
        if default_device is None:
            rec_logger.error("No default input device found")
    except Exception as e:
        rec_logger.error(f"Error querying audio devices: {e}", exc_info=True)
    
    # Create a temporary file for the recording
    temp_dir = tempfile.gettempdir()
    temp_wav = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")
    rec_logger.debug(f"Temporary WAV file: {temp_wav}")
    
    # Start recording
    try:
        rec_logger.debug("Opening audio input stream")
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback) as stream:
            rec_logger.info(f"Recording started, max duration: {max_duration}s")
            
            # Record until the stop event is set or max duration is reached
            seconds_recorded = 0
            while not stop_event.is_set() and seconds_recorded < max_duration:
                time.sleep(0.1)
                seconds_recorded = frames_recorded / sample_rate
                
                # Periodically log recording progress
                if int(seconds_recorded) % 1 == 0:  # Log every second
                    rec_logger.debug(f"Recording progress: {seconds_recorded:.2f}s / {max_duration}s")
                
                # Call the progress callback periodically
                try:
                    callback(seconds_recorded)
                except Exception as e:
                    rec_logger.error(f"Error in progress callback: {e}", exc_info=True)
                
                # If we've recorded enough frames, break
                if frames_recorded >= max_frames:
                    rec_logger.debug("Max recording duration reached")
                    break
            
            # Log recording completion reason
            if stop_event.is_set():
                rec_logger.debug("Recording stopped by user")
            elif seconds_recorded >= max_duration:
                rec_logger.debug("Recording stopped due to max duration")
            
            # Check if we recorded anything
            if frames_recorded == 0:
                rec_logger.error("No audio frames were recorded")
                raise RuntimeError("No audio was recorded")
            
            # Trim the recording to the actual length
            actual_recording = recording[:frames_recorded]
            rec_logger.debug(f"Trimmed recording to {frames_recorded} frames ({seconds_recorded:.2f}s)")
                
            # Save the recording
            try:
                rec_logger.debug(f"Saving recording to {temp_wav}")
                with wave.open(temp_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    audio_data = (actual_recording * 32767).astype(np.int16).tobytes()
                    wf.writeframes(audio_data)
                
                # Check if file was created successfully
                if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                    rec_logger.info(f"Recording saved: {temp_wav} ({frames_recorded / sample_rate:.2f}s, {os.path.getsize(temp_wav)} bytes)")
                else:
                    rec_logger.error(f"Failed to save recording: file {temp_wav} does not exist or is empty")
            except Exception as e:
                rec_logger.error(f"Error saving recording: {e}", exc_info=True)
                raise
            
            return temp_wav
            
    except Exception as e:
        rec_logger.error(f"Error during recording: {e}", exc_info=True)
        raise 
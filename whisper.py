# Make this module importable as whisper_transcriber
import sys
import os
import tempfile
import argparse
import numpy as np
import wave
import sys
import logging
import subprocess
import ssl
import time

# Fix SSL errors by disabling certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('whisper_module')

def download_whisper_model_manually(model_name="base", max_retries=3):
    """Manually download Whisper model using requests instead of the default downloader"""
    import requests
    from tqdm import tqdm
    
    # Model URLs and expected sizes (approximately)
    model_urls = {
        "tiny": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.pt",
        "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0575f39d86581/large.pt"
    }
    
    model_sizes = {
        "tiny": 150_000_000,   # ~150MB
        "base": 150_000_000,   # ~150MB
        "small": 500_000_000,  # ~500MB
        "medium": 1_500_000_000, # ~1.5GB
        "large": 3_000_000_000   # ~3GB
    }
    
    if model_name not in model_urls:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(model_urls.keys())}")
    
    # Create directory if it doesn't exist
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_path = os.path.join(cache_dir, f"{model_name}.pt")
    
    # Skip download if the file already exists and has a reasonable size
    if os.path.exists(model_path) and os.path.getsize(model_path) > model_sizes[model_name] * 0.9:
        logger.info(f"Model {model_name} already exists at {model_path}")
        return model_path
    
    logger.info(f"Downloading model {model_name} to {model_path}")
    
    # Try multiple times in case of network issues
    for attempt in range(max_retries):
        try:
            # Stream download with progress bar
            response = requests.get(
                model_urls[model_name], 
                stream=True, 
                verify=False,
                timeout=30,  # 30 second timeout
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f, tqdm(
                desc=f"Downloading {model_name} (Attempt {attempt+1}/{max_retries})",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024*1024):
                    size = f.write(data)
                    bar.update(size)
            
            # Verify the download size
            if os.path.exists(model_path) and os.path.getsize(model_path) > model_sizes[model_name] * 0.9:
                logger.info(f"Successfully downloaded model {model_name} to {model_path}")
                return model_path
            else:
                logger.warning(f"Downloaded file seems incomplete. Size: {os.path.getsize(model_path)} bytes")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying download (Attempt {attempt+2}/{max_retries})...")
                    time.sleep(2)  # Short delay before retry
                continue
                
        except Exception as e:
            logger.error(f"Download attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} download attempts failed.")
                raise
    
    # If we got here, all attempts failed
    raise RuntimeError(f"Failed to download model {model_name} after {max_retries} attempts")

def fix_whisper_installation():
    """Try to fix the whisper installation automatically"""
    try:
        logger.info("Attempting to fix Whisper installation automatically...")
        
        # Uninstall any existing whisper package
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "whisper"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        # Install the correct openai-whisper package
        result = subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            logger.info("Successfully installed openai-whisper")
            
            # Also install dependencies
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "sounddevice", "pytube", "pydub"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            
            logger.info("Please restart your application for changes to take effect")
            return True
        else:
            logger.error(f"Failed to install openai-whisper: {result.stderr.decode()}")
            return False
    except Exception as e:
        logger.error(f"Error trying to fix installation: {e}")
        return False

# Try to import whisper
try:
    import whisper
    # Test if the module has the necessary functions
    if not hasattr(whisper, 'load_model'):
        logger.warning("Found whisper module but it doesn't have load_model function")
        # Try to import specifically openai-whisper
        try:
            logger.info("Trying to import openai_whisper directly...")
            import openai_whisper
            # If successful, replace the whisper module with openai_whisper
            whisper = openai_whisper
            logger.info("Successfully imported openai_whisper")
            WHISPER_AVAILABLE = True
        except ImportError:
            logger.warning("Could not import openai_whisper, trying manual fixes...")
            
            # This might be a different whisper package, try to force it
            import importlib
            import site
            # Check if we're in a virtual environment
            venv_path = os.environ.get('VIRTUAL_ENV')
            if venv_path:
                search_paths = [os.path.join(venv_path, 'lib', 'site-packages')]
            else:
                search_paths = site.getsitepackages()
            
            # Manually search for whisper in site-packages
            whisper_path = None
            for path in search_paths:
                potential_path = os.path.join(path, 'whisper')
                if os.path.exists(potential_path) and os.path.isdir(potential_path):
                    whisper_path = potential_path
                    break
                
            if whisper_path:
                logger.info(f"Found whisper at {whisper_path}, trying to import manually")
                sys.path.insert(0, os.path.dirname(whisper_path))
                import whisper
                importlib.reload(whisper)
            
            if not hasattr(whisper, 'load_model'):
                # Define our own basic load_model function to avoid crashes
                logger.warning("Creating fallback load_model function")
                def fallback_load_model(name):
                    raise RuntimeError("OpenAI Whisper not properly installed. Run: pip install openai-whisper")
                
                whisper.load_model = fallback_load_model
                WHISPER_AVAILABLE = False
                logger.error("Invalid whisper package. Please install the correct OpenAI Whisper package with: pip install openai-whisper")
            else:
                WHISPER_AVAILABLE = True
                logger.info("OpenAI Whisper package fixed and loaded successfully")
    else:
        WHISPER_AVAILABLE = True
        logger.info("OpenAI Whisper package loaded successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.error("Failed to import OpenAI Whisper package. Please install it with: pip install openai-whisper")
    
# Try to import other dependencies
try:
    import torch
    import sounddevice as sd
    from pytube import YouTube
    from pydub import AudioSegment
    DEPENDENCIES_AVAILABLE = True
    logger.info("All dependencies loaded successfully")
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    logger.error(f"Failed to import dependency: {e}")
    logger.error("Please install all dependencies with: pip install torch sounddevice pytube pydub")

def load_model_direct(model_name="base"):
    """
    Load a model directly using torch, bypassing whisper's download mechanism
    """
    import torch
    import whisper
    from whisper.model import Whisper, ModelDimensions
    
    # Make sure the model is downloaded
    model_path = download_whisper_model_manually(model_name)
    
    logger.info(f"Loading model directly from {model_path}")
    
    # Define model dimensions based on model size
    dims = {
        "tiny": ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4, n_vocab=51864, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4),
        "base": ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=512, n_audio_head=8, n_audio_layer=6, n_vocab=51864, n_text_ctx=448, n_text_state=512, n_text_head=8, n_text_layer=6),
        "small": ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=768, n_audio_head=12, n_audio_layer=12, n_vocab=51864, n_text_ctx=448, n_text_state=768, n_text_head=12, n_text_layer=12),
        "medium": ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=1024, n_audio_head=16, n_audio_layer=24, n_vocab=51864, n_text_ctx=448, n_text_state=1024, n_text_head=16, n_text_layer=24),
        "large": ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=1280, n_audio_head=20, n_audio_layer=32, n_vocab=51864, n_text_ctx=448, n_text_state=1280, n_text_head=20, n_text_layer=32),
    }
    
    if model_name not in dims:
        raise ValueError(f"Model {model_name} not recognized.")
    
    # Create the model with the specified dimensions
    model = Whisper(dims[model_name])
    
    # Load the checkpoint file
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Load model weights
    model.load_state_dict(checkpoint)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Return wrapped model like whisper.load_model does
    return whisper.model.Whisper(model.encoder, model.decoder)

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        """
        Initialize Whisper transcriber
        Args:
            model_name: Whisper model name ("tiny", "base", "small", "medium", "large")
        """
        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available, attempting to fix...")
            if fix_whisper_installation():
                logger.info("Installation fixed, please restart the application")
                print("\n" + "=" * 80)
                print("Whisper has been installed correctly but you need to restart the application.")
                print("Please close the application and start it again.")
                print("=" * 80 + "\n")
            raise ImportError(
                "OpenAI Whisper package is not installed or is the wrong version. "
                "Please run: pip install openai-whisper"
            )
            
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Some dependencies are missing. "
                "Please run: pip install torch sounddevice pytube pydub"
            )
            
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            # First try using our manual downloader and loader to avoid SSL issues
            try:
                logger.info("Using custom downloader to avoid SSL issues...")
                # Install tqdm if it's not available
                try:
                    import tqdm
                except ImportError:
                    logger.info("Installing tqdm for download progress...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                # Install requests if it's not available
                try:
                    import requests
                except ImportError:
                    logger.info("Installing requests for downloading...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Try direct loading method first
                try:
                    self.model = load_model_direct(model_name)
                    logger.info(f"Successfully loaded Whisper model directly: {model_name}")
                except Exception as direct_load_error:
                    logger.warning(f"Direct loading failed: {direct_load_error}, trying standard loading")
                    
                    # Download the model manually 
                    model_path = download_whisper_model_manually(model_name)
                    
                    # Load the model using whisper's method
                    logger.info(f"Loading model from {model_path}")
                    self.model = whisper.load_model(model_name)
                    logger.info(f"Successfully loaded Whisper model: {model_name}")
            except Exception as e:
                logger.warning(f"Custom download failed: {e}, falling back to default loader")
                # Fall back to default loader
                self.model = whisper.load_model(model_name)
                logger.info(f"Successfully loaded Whisper model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            logger.info("This might be caused by the wrong whisper package, attempting to fix...")
            if fix_whisper_installation():
                logger.info("Installation fixed, please restart the application")
                print("\n" + "=" * 80)
                print("Whisper has been installed correctly but you need to restart the application.")
                print("Please close the application and start it again.")
                print("=" * 80 + "\n")
            raise
            
        self.language = None  # Will auto-detect by default
        self.task = "transcribe"  # Default to transcribe, not translate
        
    def transcribe_audio_file(self, audio_path):
        """
        Transcribe audio file using Whisper
        Args:
            audio_path: Path to audio file
        Returns:
            Dictionary with transcription and metadata
        """
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            raise FileNotFoundError(f"File not found: {audio_path}")
        
        logger.info(f"Transcribing file: {audio_path}")
        
        # If language is explicitly specified, use it directly
        if self.language:
            options = {
                "language": self.language,
                "task": self.task  # Use the current task
            }
            logger.info(f"Using specified language: {self.language}")
            
            # Perform transcription
            try:
                result = self.model.transcribe(audio_path, **options)
                logger.info(f"Transcription completed with specified language: {self.language}")
                return result
            except Exception as e:
                logger.error(f"Error transcribing audio file with specified language: {e}")
                return {"error": str(e)}
        
        # Auto-detection mode - use a two-pass approach
        try:
            # First pass: Initial transcription with auto-detection
            logger.info("First pass: Auto-detecting language")
            # Force the task to be transcribe not translate
            first_result = self.model.transcribe(audio_path, task="transcribe")
            detected_lang = first_result.get('language')
            detected_text = first_result.get('text', '')
            
            logger.info(f"First pass detected language: {detected_lang}")
            
            # Special case for Hindi that Whisper often incorrectly identifies as English
            # Check for Hindi words/patterns in Latin script
            if detected_lang == 'en':
                hindi_patterns = ['hai', 'hain', 'aur', 'main', 'mein', 'kya', 'nahin', 'nahi', 'mere', 'tumhara', 'hamara']
                pattern_count = 0
                for pattern in hindi_patterns:
                    if pattern in detected_text.lower().split():
                        pattern_count += 1
                
                # If multiple Hindi patterns are found, likely Hindi being mistranscribed as English
                if pattern_count >= 2:
                    logger.info(f"Found {pattern_count} Hindi patterns in English text, likely Hindi in Latin script")
                    # Try a second pass forcing Hindi
                    second_pass_options = {
                        "language": "hi",  # Force Hindi
                        "task": "transcribe"  # Important: Keep original language, don't translate
                    }
                    logger.info("Second pass: Re-transcribing with Hindi")
                    second_result = self.model.transcribe(audio_path, **second_pass_options)
                    second_result['language'] = 'hi'
                    second_result['original_detected_language'] = detected_lang
                    logger.info("Second pass transcription completed with Hindi")
                    return second_result
            
            # Language correction rules based on common misdetections
            language_corrections = {
                # Language mappings for commonly confused languages
                ('ur', lambda t: any(p in t.lower() for p in ['om', 'namah', 'shanti', 'dharma', 'karma', 'satyam'])): 'sa',  # Sanskrit
                ('hi', lambda t: any(p in t.lower() for p in ['allah', 'bismillah', 'quran', 'muslim'])): 'ur',  # Urdu 
                ('en', lambda t: any(p in t.lower() for p in ['नमस्ते', 'भारत', 'हिंदी'])): 'hi',  # Hindi
                ('hi', lambda t: any(p in t.lower() for p in ['தமிழ்', 'சென்னை', 'வணக்கம்'])): 'ta',  # Tamil
                ('ta', lambda t: any(p in t.lower() for p in ['తెలుగు', 'నమస్కారం', 'అన్నం'])): 'te',  # Telugu
                ('ar', lambda t: 'کتاب' in t or 'فارسی' in t): 'fa',  # Persian vs Arabic
                ('hi', lambda t: 'অ' in t or 'বাংলা' in t): 'bn',  # Bengali
                ('hi', lambda t: 'संस्कृत' in t): 'sa',  # Sanskrit from Devanagari detection
            }
            
            # Check for corrections
            corrected_lang = None
            for (source_lang, condition_func), target_lang in language_corrections.items():
                if detected_lang == source_lang and condition_func(detected_text):
                    corrected_lang = target_lang
                    logger.info(f"Language correction: {source_lang} -> {target_lang} based on text patterns")
                    break
            
            # Second pass with corrected language if needed
            if corrected_lang:
                logger.info(f"Second pass: Re-transcribing with corrected language: {corrected_lang}")
                options = {
                    "language": corrected_lang,
                    "task": self.task  # Important: Keep original language, don't translate
                }
                second_result = self.model.transcribe(audio_path, **options)
                second_result['language'] = corrected_lang  # Ensure the result shows the corrected language
                second_result['original_detected_language'] = detected_lang  # Keep original detection for reference
                logger.info(f"Transcription completed with corrected language: {corrected_lang}")
                return second_result
            
            # No correction needed
            logger.info(f"Language auto-detection completed: {detected_lang}")
            return first_result
            
        except Exception as e:
            logger.error(f"Error in language auto-detection: {e}")
            return {"error": str(e)}
    
    def transcribe_youtube_video(self, youtube_url):
        """
        Download audio from YouTube video and transcribe
        Args:
            youtube_url: URL of YouTube video
        Returns:
            Dictionary with transcription and metadata
        """
        logger.info(f"Downloading audio from YouTube: {youtube_url}")
        
        # Handle YouTube Shorts URLs by converting them to standard format
        if '/shorts/' in youtube_url:
            video_id = youtube_url.split('/shorts/')[1].split('?')[0]
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Converted YouTube Shorts URL to standard format: {youtube_url}")
        
        temp_dir = tempfile.gettempdir()
        temp_audio = os.path.join(temp_dir, f"youtube_audio_{os.getpid()}.mp4")
        
        try:
            # Method 1: Try yt-dlp first as it's more reliable
            try:
                # Install yt-dlp if it's not available
                try:
                    import yt_dlp
                except ImportError:
                    logger.info("Installing yt-dlp for YouTube downloads...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Download using yt-dlp
                logger.info("Using yt-dlp for download...")
                subprocess.run([
                    sys.executable, "-m", "yt_dlp", 
                    "-f", "bestaudio", 
                    "-o", temp_audio, 
                    youtube_url
                ], check=True)
                logger.info(f"Successfully downloaded audio with yt-dlp to {temp_audio}")
            except Exception as ytdlp_error:
                logger.warning(f"yt-dlp download failed: {ytdlp_error}")
                
                # Method 2: Try PyTube as fallback
                try:
                    logger.info("Falling back to PyTube...")
                    # Try to update pytube first
                    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pytube"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    from pytube import YouTube
                    yt = YouTube(youtube_url)
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    if audio_stream:
                        audio_stream.download(output_path=temp_dir, filename=os.path.basename(temp_audio))
                        logger.info(f"Successfully downloaded audio with PyTube to {temp_audio}")
                    else:
                        raise Exception("No audio stream found")
                except Exception as pytube_error:
                    logger.warning(f"PyTube download failed: {pytube_error}")
                    
                    # Method 3: Try a direct URL if it's a common video site
                    try:
                        video_id = youtube_url.split("v=")[1].split("&")[0]
                        direct_url = f"https://www.youtube.com/watch?v={video_id}"
                        
                        logger.info(f"Trying direct request to: {direct_url}")
                        
                        # Let user know we're still trying
                        print(f"YouTube download issues - trying alternative method. Please wait...")
                        
                        # Use yt-dlp as a command line tool as last resort
                        subprocess.run([
                            "yt-dlp", 
                            "-f", "bestaudio", 
                            "-o", temp_audio, 
                            youtube_url
                        ], check=True)
                        
                        if not os.path.exists(temp_audio):
                            raise Exception("Download with yt-dlp command line failed")
                    except Exception as direct_error:
                        logger.error(f"All download methods failed: {direct_error}")
                        return {"error": f"Failed to download YouTube video: Unable to access {youtube_url}"}
            
            # Convert to WAV
            temp_wav = self._convert_to_wav(temp_audio)
            
            # Transcribe the audio
            result = self.transcribe_audio_file(temp_wav)
            
            # Clean up
            try:
                os.remove(temp_audio)
                os.remove(temp_wav)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Error transcribing YouTube video: {e}")
            return {"error": f"Failed to transcribe YouTube video: {e}"}
    
    def _convert_to_wav(self, audio_path):
        """
        Convert audio file to WAV format
        Args:
            audio_path: Path to input audio file
        Returns:
            Path to WAV file
        """
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"temp_audio_{os.getpid()}.wav")
        
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio.export(output_path, format="wav")
            logger.info(f"Converted audio to WAV: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {e}")
            raise
    
    def transcribe_video_file(self, video_path):
        """
        Extract audio from video file and transcribe
        Args:
            video_path: Path to video file
        Returns:
            Dictionary with transcription and metadata
        """
        if not os.path.exists(video_path):
            logger.error(f"File not found: {video_path}")
            raise FileNotFoundError(f"File not found: {video_path}")
        
        # Extract audio to temporary WAV file
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, f"video_audio_{os.getpid()}.wav")
        
        try:
            # Extract audio using pydub
            logger.info(f"Extracting audio from video: {video_path}")
            video_audio = AudioSegment.from_file(video_path)
            video_audio = video_audio.set_channels(1)  # Convert to mono
            video_audio = video_audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            video_audio.export(temp_wav, format="wav")
            
            # Transcribe
            result = self.transcribe_audio_file(temp_wav)
            
            # Clean up
            try:
                os.remove(temp_wav)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Error transcribing video file: {e}")
            return {"error": f"Failed to transcribe video file: {e}"}
    
    def record_and_transcribe(self, duration=5, sample_rate=16000):
        """
        Record audio from microphone and transcribe
        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
        Returns:
            Transcription result
        """
        logger.info(f"Recording for {duration} seconds...")
        
        try:
            # Record audio
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            
            # Save to WAV file
            temp_dir = tempfile.gettempdir()
            # Use PID in filename to avoid conflicts
            temp_wav = os.path.join(temp_dir, f"recording_{os.getpid()}.wav")
            
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes((recording * 32767).astype(np.int16).tobytes())
            
            logger.info(f"Audio saved to {temp_wav}")
            
            # Transcribe
            result = self.transcribe_audio_file(temp_wav)
            
            # Clean up
            try:
                os.remove(temp_wav)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Error in record_and_transcribe: {e}")
            return {"error": f"Recording and transcription failed: {e}"}
    
    def live_transcription(self, chunk_duration=10, total_duration=None):
        """
        Perform live transcription in chunks
        Args:
            chunk_duration: Duration of each recording chunk in seconds
            total_duration: Total duration to record (None for indefinite)
        """
        logger.info("Starting live transcription. Press Ctrl+C to stop.")
        
        try:
            chunks_processed = 0
            while total_duration is None or chunks_processed * chunk_duration < total_duration:
                logger.info(f"Recording chunk {chunks_processed + 1}...")
                result = self.record_and_transcribe(duration=chunk_duration)
                
                if "text" in result:
                    logger.info(f"Transcription: {result['text']}")
                    if "language" in result:
                        logger.info(f"Detected language: {result['language']}")
                else:
                    logger.warning("No transcription available")
                
                chunks_processed += 1
                
        except KeyboardInterrupt:
            logger.info("Live transcription stopped by user.")
            
    def save_transcript_to_file(self, transcript, output_file):
        """
        Save transcript to text file
        Args:
            transcript: Transcription result from Whisper
            output_file: Path to output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                if isinstance(transcript, dict) and "text" in transcript:
                    f.write(transcript["text"])
                elif isinstance(transcript, str):
                    f.write(transcript)
                else:
                    logger.error("Invalid transcript format")
                    raise ValueError("Invalid transcript format")
            
            logger.info(f"Transcript saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            raise


# Helper function to check if OpenAI Whisper is properly installed
def check_whisper_installation():
    """Check if OpenAI Whisper is properly installed and provide instructions if not"""
    if not WHISPER_AVAILABLE:
        print("=" * 80)
        print("ERROR: OpenAI Whisper package is not installed or cannot be imported.")
        print("\nThe application found a whisper package, but it doesn't have the required functions.")
        print("This usually happens when the wrong whisper package is installed.")
        
        try:
            # Attempt to fix automatically
            print("\nAttempting to fix the issue automatically...")
            if fix_whisper_installation():
                print("\nFix successful! Please restart your application to use Whisper.")
            else:
                print("\nAutomatic fix failed. Please run the following commands manually:")
                print("1. pip uninstall -y whisper")
                print("2. pip install -U openai-whisper")
                print("3. pip install -U torch sounddevice pytube pydub")
        except Exception as e:
            print(f"\nError during automatic fix: {e}")
            print("\nPlease manually run the following commands:")
            print("1. pip uninstall -y whisper")
            print("2. pip install -U openai-whisper")
            print("3. pip install -U torch sounddevice pytube pydub")
            
        print("\nAfter installation, restart your application.")
        print("=" * 80)
        return False
    return True


if __name__ == "__main__":
    # Check if OpenAI Whisper is installed before proceeding
    if not check_whisper_installation():
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="OpenAI Whisper Transcription Tool")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="base",
                        help="Whisper model to use")
    parser.add_argument("--source", choices=["youtube", "video", "audio", "live"], required=True,
                        help="Source type (youtube, video, audio, or live)")
    parser.add_argument("--input", help="Input file path or YouTube URL")
    parser.add_argument("--language", help="Language code (optional, will auto-detect if not specified)")
    parser.add_argument("--output", help="Output file to save transcript")
    parser.add_argument("--live-duration", type=int, default=None,
                        help="Total duration for live transcription in seconds (default: indefinite)")
    
    args = parser.parse_args()
    
    try:
        # Check for CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info("CUDA is available. Using GPU for faster transcription.")
        else:
            logger.info("CUDA is not available. Using CPU for transcription (slower).")
        
        # Initialize transcriber
        transcriber = WhisperTranscriber(model_name=args.model)
        if args.language:
            transcriber.language = args.language
        
        # Process based on source type
        if args.source == "youtube":
            if not args.input:
                parser.error("--input YouTube URL is required for YouTube source")
            result = transcriber.transcribe_youtube_video(args.input)
            if "text" in result:
                print(f"\nTranscription: {result['text']}")
                if args.output:
                    transcriber.save_transcript_to_file(result, args.output)
            else:
                print(f"Transcription failed: {result.get('error', 'Unknown error')}")
            
        elif args.source == "video":
            if not args.input:
                parser.error("--input video file path is required for video source")
            result = transcriber.transcribe_video_file(args.input)
            if "text" in result:
                print(f"\nTranscription: {result['text']}")
                if args.output:
                    transcriber.save_transcript_to_file(result, args.output)
            else:
                print(f"Transcription failed: {result.get('error', 'Unknown error')}")
            
        elif args.source == "audio":
            if not args.input:
                parser.error("--input audio file path is required for audio source")
            result = transcriber.transcribe_audio_file(args.input)
            if "text" in result:
                print(f"\nTranscription: {result['text']}")
                if args.output:
                    transcriber.save_transcript_to_file(result, args.output)
            else:
                print(f"Transcription failed: {result.get('error', 'Unknown error')}")
            
        elif args.source == "live":
            transcriber.live_transcription(total_duration=args.live_duration)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        if "No module named 'whisper'" in str(e):
            check_whisper_installation()

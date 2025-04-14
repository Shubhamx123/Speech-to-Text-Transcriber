import requests
import json
import os
import tempfile
import time
from pydub import AudioSegment
import pytube
import sounddevice as sd
import numpy as np
import wave
import argparse
import logging
import subprocess
import sys

# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('spring_lab_module')

# Check if all required dependencies are available
try:
    # Just testing imports
    import requests
    from pydub import AudioSegment
    import pytube
    import sounddevice as sd
    import numpy as np
    
    SPRING_LAB_AVAILABLE = True
    logger.info("Spring Lab dependencies loaded successfully")
except ImportError as e:
    SPRING_LAB_AVAILABLE = False
    logger.error(f"Spring Lab unavailable - missing dependency: {e}")

class SpringLabASR:
    def __init__(self):
        """
        Initialize Spring Lab ASR client
        """
        self.api_url = "https://asr.iitm.ac.in/internal/asr/decode"
        
        # Default language
        self.language = "english"
        # Whether to request VTT captions
        self.generate_vtt = False
        
    def transcribe_audio_file(self, audio_path):
        """
        Transcribe audio file using Spring Lab ASR API
        Args:
            audio_path: Path to audio file
        Returns:
            Dictionary with transcription and metadata
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")
        
        print(f"Transcribing file: {audio_path}")
        
        # If language is not specified or empty, try auto-detection by attempting multiple languages
        if not self.language:
            logger.info("No language specified, attempting auto-detection")
            # Try the most common languages
            test_languages = ["hindi", "english", "tamil"]
            best_result = None
            best_confidence = 0
            
            for lang in test_languages:
                logger.info(f"Trying transcription with language: {lang}")
                
                # Prepare the request
                files = {
                    'file': open(audio_path, 'rb'),
                    'language': (None, lang),
                    'vtt': (None, str(self.generate_vtt).lower())
                }
                
                # Make the API request
                try:
                    response = requests.post(self.api_url, files=files)
                    files['file'].close()
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Estimate confidence based on transcript length and word count
                        transcript = result.get('transcript', '')
                        word_count = len(transcript.split())
                        confidence = word_count
                        
                        logger.info(f"Language {lang} produced {word_count} words")
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = result
                            best_result['detected_language'] = lang
                except Exception as e:
                    logger.warning(f"Error trying language {lang}: {e}")
                    continue
            
            if best_result:
                logger.info(f"Auto-detected language: {best_result['detected_language']}")
                return best_result
            else:
                # Fallback to English if auto-detection failed
                logger.warning("Auto-detection failed, falling back to English")
                self.language = "english"
        
        # Standard transcription with specified language
        # Prepare the request
        files = {
            'file': open(audio_path, 'rb'),
            'language': (None, self.language),
            'vtt': (None, str(self.generate_vtt).lower())
        }
        
        # Make the API request
        try:
            response = requests.post(self.api_url, files=files)
            
            # Close the file
            files['file'].close()
            
            # Check if request was successful
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return {"status": "failure", "reason": f"HTTP error {response.status_code}"}
            
            result = response.json()
            return result
            
        except Exception as e:
            print(f"Error calling Spring Lab ASR API: {e}")
            return {"status": "failure", "reason": str(e)}
    
    def transcribe_youtube_video(self, youtube_url):
        """
        Download audio from YouTube video and transcribe
        Args:
            youtube_url: URL of YouTube video
        Returns:
            Transcription result
        """
        logger.info(f"Downloading audio from YouTube: {youtube_url}")
        
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
                    print("Trying PyTube as fallback...")
                    # First, try to update pytube
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
            temp_wav = os.path.join(temp_dir, f"youtube_audio_{os.getpid()}.wav")
            try:
                self._convert_to_wav(temp_audio, temp_wav)
            except Exception as e:
                logger.error(f"Error converting to WAV: {e}")
                return {"error": f"Failed to convert YouTube video audio: {e}"}
            
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
    
    def transcribe_video_file(self, video_path):
        """
        Extract audio from video file and transcribe
        Args:
            video_path: Path to video file
        Returns:
            Dictionary with transcription and metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")
        
        # The Spring Lab ASR API accepts video files directly
        return self.transcribe_audio_file(video_path)
    
    def record_and_transcribe(self, duration=5, sample_rate=16000):
        """
        Record audio from microphone and transcribe
        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
        Returns:
            Dictionary with transcription and metadata
        """
        print(f"Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # Save to WAV file
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, "recording.wav")
        
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        
        # Transcribe
        result = self.transcribe_audio_file(temp_wav)
        
        # Clean up
        os.remove(temp_wav)
        
        return result
    
    def live_transcription(self, chunk_duration=10, total_duration=None):
        """
        Perform live transcription in chunks
        Args:
            chunk_duration: Duration of each recording chunk in seconds
            total_duration: Total duration to record (None for indefinite)
        """
        print("Starting live transcription. Press Ctrl+C to stop.")
        print(f"Language: {self.language}")
        
        try:
            chunks_processed = 0
            while total_duration is None or chunks_processed * chunk_duration < total_duration:
                print(f"\nRecording chunk {chunks_processed + 1}...")
                result = self.record_and_transcribe(duration=chunk_duration)
                
                if result.get("status") == "success":
                    print(f"Transcription: {result.get('transcript', 'No transcript available')}")
                    print(f"Time taken: {result.get('time_taken', 'N/A')} seconds")
                else:
                    print(f"Transcription failed: {result.get('reason', 'Unknown error')}")
                
                chunks_processed += 1
                
        except KeyboardInterrupt:
            print("\nLive transcription stopped.")
    
    def get_supported_languages(self):
        """Return list of supported languages"""
        return [
            "bengali", "english", "gujarati", "hindi", "kannada", 
            "malayalam", "marathi", "odia", "punjabi", 
            "sanskrit", "tamil", "telugu", "urdu"
        ]

    def _convert_to_wav(self, input_path, output_path=None):
        """
        Convert audio file to WAV format for transcription
        Args:
            input_path: Path to input audio file
            output_path: Path to output WAV file (optional)
        Returns:
            Path to WAV file
        """
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"temp_audio_{os.getpid()}.wav")
        
        try:
            logger.info(f"Converting {input_path} to WAV at {output_path}")
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio.export(output_path, format="wav")
            logger.info(f"Successfully converted to WAV")
            return output_path
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spring Lab ASR Tool")
    parser.add_argument("--source", choices=["youtube", "video", "audio", "live"], required=True, 
                        help="Source type (youtube, video, audio, or live)")
    parser.add_argument("--input", help="Input file path or YouTube URL")
    parser.add_argument("--language", default="english", 
                        help="Source language (e.g., english, hindi, tamil)")
    parser.add_argument("--vtt", action="store_true", help="Generate WebVTT captions")
    parser.add_argument("--live-duration", type=int, default=None, 
                        help="Total duration for live transcription in seconds (default: indefinite)")
    
    args = parser.parse_args()
    
    try:
        asr = SpringLabASR()
        asr.language = args.language.lower()
        asr.generate_vtt = args.vtt
        
        if args.source == "youtube":
            if not args.input:
                parser.error("--input YouTube URL is required for YouTube source")
            result = asr.transcribe_youtube_video(args.input)
            if result.get("status") == "success":
                print(f"\nTranscription: {result.get('transcript')}")
                print(f"Time taken: {result.get('time_taken')} seconds")
                if args.vtt and 'vtt' in result:
                    print(f"\nWebVTT captions: {result.get('vtt')}")
            else:
                print(f"Transcription failed: {result.get('reason', 'Unknown error')}")
            
        elif args.source == "video":
            if not args.input:
                parser.error("--input video file path is required for video source")
            result = asr.transcribe_video_file(args.input)
            if result.get("status") == "success":
                print(f"\nTranscription: {result.get('transcript')}")
                print(f"Time taken: {result.get('time_taken')} seconds")
                if args.vtt and 'vtt' in result:
                    print(f"\nWebVTT captions: {result.get('vtt')}")
            else:
                print(f"Transcription failed: {result.get('reason', 'Unknown error')}")
            
        elif args.source == "audio":
            if not args.input:
                parser.error("--input audio file path is required for audio source")
            result = asr.transcribe_audio_file(args.input)
            if result.get("status") == "success":
                print(f"\nTranscription: {result.get('transcript')}")
                print(f"Time taken: {result.get('time_taken')} seconds")
                if args.vtt and 'vtt' in result:
                    print(f"\nWebVTT captions: {result.get('vtt')}")
            else:
                print(f"Transcription failed: {result.get('reason', 'Unknown error')}")
            
        elif args.source == "live":
            asr.live_transcription(total_duration=args.live_duration)
            
    except Exception as e:
        print(f"Error: {e}")

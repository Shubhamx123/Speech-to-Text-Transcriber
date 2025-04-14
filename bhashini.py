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

# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bhashini_module')

# Check if all required dependencies are available
try:
    # Just testing imports
    import requests
    import json
    from pydub import AudioSegment
    import pytube
    import sounddevice as sd
    import numpy as np
    import wave
    
    BHASHINI_AVAILABLE = True
    logger.info("Bhashini dependencies loaded successfully")
except ImportError as e:
    BHASHINI_AVAILABLE = False
    logger.error(f"Bhashini unavailable - missing dependency: {e}")

class BhashiniASR:
    def __init__(self, api_key=None):
        """
        Initialize Bhashini ASR client
        Args:
            api_key: API key for Bhashini services (get from https://bhashini.gov.in/developer)
        """
        self.api_key = api_key or os.environ.get("BHASHINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Get it from https://bhashini.gov.in/developer")
        
        self.base_url = "https://dhruva-api.bhashini.gov.in"
        self.pipeline_url = f"{self.base_url}/services/inference/pipeline"
        self.asr_url = f"{self.base_url}/services/inference/asr"
        
        # Default configurations
        self.source_language = "hi"  # Hindi by default
        self.target_language = "en"  # English by default
        
    def _get_auth_headers(self):
        """Return headers with API key authentication"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _call_asr_api(self, audio_path):
        """
        Call Bhashini ASR API with audio file
        Args:
            audio_path: Path to audio file (wav format recommended)
        Returns:
            Transcription result
        """
        with open(audio_path, "rb") as audio_file:
            files = {"audio": audio_file}
            
            payload = {
                "language": {
                    "sourceLanguage": self.source_language
                },
                "config": {
                    "audioFormat": "wav"
                }
            }
            
            # Prepare the multipart form data
            data = {"task": json.dumps(payload)}
            
            response = requests.post(
                self.asr_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
                
            result = response.json()
            return result
    
    def transcribe_audio_file(self, audio_path):
        """
        Transcribe audio file
        Args:
            audio_path: Path to audio file
        Returns:
            Transcription text
        """
        # Convert to WAV if not already
        if not audio_path.lower().endswith(".wav"):
            temp_wav = self._convert_to_wav(audio_path)
            result = self._call_asr_api(temp_wav)
            os.remove(temp_wav)
        else:
            result = self._call_asr_api(audio_path)
            
        if not result:
            return "Transcription failed"
            
        try:
            return result['output'][0]['source']
        except (KeyError, IndexError):
            return "Unable to extract transcription from response"
    
    def transcribe_youtube_video(self, youtube_url):
        """
        Transcribe audio from YouTube video
        Args:
            youtube_url: URL of YouTube video
        Returns:
            Transcription text
        """
        print(f"Downloading audio from YouTube: {youtube_url}")
        temp_dir = tempfile.gettempdir()
        temp_audio = os.path.join(temp_dir, "youtube_audio.mp4")
        
        try:
            # Download YouTube video audio
            yt = pytube.YouTube(youtube_url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_stream.download(output_path=temp_dir, filename="youtube_audio.mp4")
            
            # Convert to WAV
            wav_path = self._convert_to_wav(temp_audio)
            
            # Transcribe
            result = self.transcribe_audio_file(wav_path)
            
            # Clean up
            os.remove(temp_audio)
            os.remove(wav_path)
            
            return result
        except Exception as e:
            print(f"Error transcribing YouTube video: {e}")
            return f"Failed to transcribe YouTube video: {e}"
    
    def _convert_to_wav(self, audio_path):
        """
        Convert audio file to WAV format (required by Bhashini API)
        Args:
            audio_path: Path to input audio file
        Returns:
            Path to WAV file
        """
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "temp_audio.wav")
        
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
        audio.export(output_path, format="wav")
        
        return output_path
    
    def record_and_transcribe(self, duration=5, sample_rate=16000):
        """
        Record audio from microphone and transcribe
        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
        Returns:
            Transcription text
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
    
    def transcribe_video_file(self, video_path):
        """
        Extract audio from video file and transcribe
        Args:
            video_path: Path to video file
        Returns:
            Transcription text
        """
        # Extract audio to temporary WAV file
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, "video_audio.wav")
        
        try:
            # Extract audio using pydub
            video_audio = AudioSegment.from_file(video_path)
            video_audio = video_audio.set_channels(1)  # Convert to mono
            video_audio = video_audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            video_audio.export(temp_wav, format="wav")
            
            # Transcribe
            result = self.transcribe_audio_file(temp_wav)
            
            # Clean up
            os.remove(temp_wav)
            
            return result
        except Exception as e:
            print(f"Error transcribing video file: {e}")
            return f"Failed to transcribe video file: {e}"
    
    def live_transcription(self, chunk_duration=10, total_duration=None):
        """
        Perform live transcription in chunks
        Args:
            chunk_duration: Duration of each recording chunk in seconds
            total_duration: Total duration to record (None for indefinite)
        """
        print("Starting live transcription. Press Ctrl+C to stop.")
        
        try:
            chunks_processed = 0
            while total_duration is None or chunks_processed * chunk_duration < total_duration:
                print(f"\nRecording chunk {chunks_processed + 1}...")
                transcription = self.record_and_transcribe(duration=chunk_duration)
                print(f"Transcription: {transcription}")
                chunks_processed += 1
                
        except KeyboardInterrupt:
            print("\nLive transcription stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bhashini ASR Tool")
    parser.add_argument("--api-key", help="Bhashini API Key")
    parser.add_argument("--source", choices=["youtube", "video", "audio", "live"], required=True, 
                        help="Source type (youtube, video, audio, or live)")
    parser.add_argument("--input", help="Input file path or YouTube URL")
    parser.add_argument("--language", default="hi", help="Source language code (default: hi)")
    parser.add_argument("--live-duration", type=int, default=None, 
                        help="Total duration for live transcription in seconds (default: indefinite)")
    
    args = parser.parse_args()
    
    try:
        bhashini = BhashiniASR(api_key=args.api_key)
        bhashini.source_language = args.language
        
        if args.source == "youtube":
            if not args.input:
                parser.error("--input YouTube URL is required for YouTube source")
            result = bhashini.transcribe_youtube_video(args.input)
            print(f"Transcription: {result}")
            
        elif args.source == "video":
            if not args.input:
                parser.error("--input video file path is required for video source")
            result = bhashini.transcribe_video_file(args.input)
            print(f"Transcription: {result}")
            
        elif args.source == "audio":
            if not args.input:
                parser.error("--input audio file path is required for audio source")
            result = bhashini.transcribe_audio_file(args.input)
            print(f"Transcription: {result}")
            
        elif args.source == "live":
            bhashini.live_transcription(total_duration=args.live_duration)
            
    except Exception as e:
        print(f"Error: {e}")

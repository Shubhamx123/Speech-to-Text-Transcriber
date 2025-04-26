import os
import tempfile
import subprocess
import logging
import uuid
import re
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('utils')

# Check if FFmpeg is available
FFMPEG_AVAILABLE = True
try:
    # Test if FFmpeg is installed
    subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
except (FileNotFoundError, subprocess.SubprocessError):
    logger.error("FFmpeg is not available. Audio/video conversion features will be limited.")
    FFMPEG_AVAILABLE = False

# Try to import external libraries with fallbacks
try:
    from pydub import AudioSegment
except ImportError:
    logger.error("Failed to import pydub. Audio processing will be limited.")
    # Create dummy AudioSegment to avoid errors
    class DummyAudioSegment:
        @staticmethod
        def from_file(*args, **kwargs):
            return DummyAudioSegment()
        def export(self, *args, **kwargs):
            return None
        frame_rate = 0
    AudioSegment = DummyAudioSegment

try:
    import yt_dlp
except ImportError:
    logger.error("Failed to import yt_dlp. YouTube download will use fallback methods.")
    yt_dlp = None

try:
    import pytube
except ImportError:
    logger.error("Failed to import pytube. YouTube features will be limited.")
    # Create dummy pytube to avoid errors
    class DummyYouTube:
        def __init__(self, *args, **kwargs):
            pass
        streams = None
    class DummyChannel:
        def __init__(self, *args, **kwargs):
            self.channel_name = "Unknown"
            self.channel_id = "unknown"
            self.channel_url = ""
            self.video_urls = []
    class DummyPytube:
        YouTube = DummyYouTube
        Channel = DummyChannel
    pytube = DummyPytube

def create_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def download_youtube_audio(url, output_dir=None):
    """Download audio from a YouTube video.
    
    Args:
        url (str): YouTube video URL
        output_dir (str, optional): Directory to save the file
        
    Returns:
        str: Path to the downloaded audio file
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    # Generate a unique filename
    temp_filename = f"youtube_audio_{uuid.uuid4().hex[:8]}"
    output_path = os.path.join(output_dir, f"{temp_filename}.mp3")
    
    logger.info(f"Downloading audio from: {url}")
    
    try:
        # Try with yt-dlp first (more robust)
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_dir, temp_filename),
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            # The file will be saved with .mp3 extension due to the postprocessor
            return os.path.join(output_dir, f"{temp_filename}.mp3")
    
    except Exception as e:
        logger.warning(f"yt-dlp download failed: {e}. Trying PyTube as fallback...")
        
        try:
            # Fallback to pytube
            yt = pytube.YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            downloaded_file = audio_stream.download(output_path=output_dir, filename=temp_filename)
            
            # Convert to mp3 if needed
            if not downloaded_file.endswith('.mp3'):
                mp3_file = convert_to_mp3(downloaded_file, output_path)
                os.remove(downloaded_file)  # Remove the original file
                return mp3_file
            return downloaded_file
            
        except Exception as e2:
            logger.error(f"Both downloaders failed. Error: {e2}")
            raise Exception(f"Failed to download YouTube audio: {e2}")

def get_channel_info(channel_id):
    """Get information about a YouTube channel including total videos count.
    
    Args:
        channel_id (str): YouTube channel ID or URL
        
    Returns:
        dict: Channel information including total videos count and title
    """
    # Extract channel ID if full URL is provided
    if 'youtube.com' in channel_id:
        if '/channel/' in channel_id:
            channel_id = channel_id.split('/channel/')[1].split('/')[0]
        elif '/c/' in channel_id or '/user/' in channel_id:
            try:
                # Try to resolve the custom URL to a channel ID
                channel = pytube.Channel(channel_id)
                channel_id = channel.channel_id
            except Exception as e:
                logger.error(f"Failed to resolve channel URL: {e}")
                raise Exception(f"Invalid channel URL: {channel_id}")
    
    try:
        # Get channel using pytube
        channel = pytube.Channel(f"https://www.youtube.com/channel/{channel_id}")
        
        # Get channel info
        channel_info = {
            'title': channel.channel_name,
            'channel_id': channel_id,
            'url': channel.channel_url,
        }
        
        # Count total videos (this could take time for large channels)
        # We'll estimate by getting the initial batch
        video_urls = []
        video_count = 0
        
        # Use yt-dlp to get a more accurate video count
        try:
            ydl_opts = {
                'extract_flat': True,
                'skip_download': True,
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(f"https://www.youtube.com/channel/{channel_id}/videos", download=False)
                if 'entries' in result:
                    video_count = len(result['entries'])
                    # For large channels, this may not get all videos
                    if 'estimated_entries' in result:
                        video_count = result['estimated_entries'] or video_count
        except Exception as e:
            logger.warning(f"yt-dlp count failed: {e}, using PyTube fallback")
            # Fallback to pytube
            try:
                # This is slower and less reliable for large channels
                video_urls = list(channel.video_urls)
                video_count = len(video_urls)
            except Exception as e2:
                logger.error(f"Failed to count videos: {e2}")
                video_count = 0
        
        channel_info['video_count'] = video_count
        return channel_info
    except Exception as e:
        logger.error(f"Failed to get channel info: {e}")
        raise Exception(f"Failed to fetch info from channel {channel_id}: {e}")

def get_channel_videos(channel_id, limit=10, start_index=0, sort_by="newest"):
    """Get videos from a YouTube channel with pagination support.
    
    Args:
        channel_id (str): YouTube channel ID or URL
        limit (int): Maximum number of videos to fetch
        start_index (int): Starting index for pagination
        sort_by (str): Sort order - "newest", "oldest", or "popular"
        
    Returns:
        list: List of video URLs
    """
    # Extract channel ID if full URL is provided
    if 'youtube.com' in channel_id:
        if '/channel/' in channel_id:
            channel_id = channel_id.split('/channel/')[1].split('/')[0]
        elif '/c/' in channel_id or '/user/' in channel_id:
            try:
                # Try to resolve the custom URL to a channel ID
                channel = pytube.Channel(channel_id)
                channel_id = channel.channel_id
            except Exception as e:
                logger.error(f"Failed to resolve channel URL: {e}")
                raise Exception(f"Invalid channel URL: {channel_id}")
    
    try:
        # Use yt-dlp for more reliable pagination and sorting
        ydl_opts = {
            'extract_flat': True,
            'skip_download': True,
            'quiet': True,
            'playlistend': start_index + limit,
            'playliststart': start_index + 1,  # yt-dlp uses 1-based indexing
        }
        
        # Add sorting options
        if sort_by == "oldest":
            ydl_opts['playlistreverse'] = True
        elif sort_by == "popular":
            # YouTube doesn't have a direct API for popularity, 
            # would need to sort after fetching
            pass
        
        video_urls = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"https://www.youtube.com/channel/{channel_id}/videos", download=False)
            if 'entries' in result:
                for entry in result['entries']:
                    if entry:
                        video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
        
        # If we couldn't get videos or got fewer than expected, try pytube as fallback
        if not video_urls:
            logger.warning(f"yt-dlp returned no videos, using PyTube fallback")
            try:
                # Get channel videos using pytube (less reliable for pagination)
                channel = pytube.Channel(f"https://www.youtube.com/channel/{channel_id}")
                all_videos = list(channel.video_urls)
                
                # Sort if needed
                if sort_by == "oldest":
                    all_videos.reverse()
                
                # Apply pagination
                end_index = min(start_index + limit, len(all_videos))
                video_urls = all_videos[start_index:end_index]
            except Exception as e:
                logger.error(f"PyTube fallback failed: {e}")
                raise Exception(f"Failed to fetch videos from channel {channel_id}: {e}")
        
        logger.info(f"Found {len(video_urls)} videos in channel {channel_id} (start_index={start_index}, limit={limit})")
        return video_urls
    except Exception as e:
        logger.error(f"Failed to get channel videos: {e}")
        raise Exception(f"Failed to fetch videos from channel {channel_id}: {e}")

def download_all_channel_videos(channel_id, output_dir=None, limit=5, start_index=0, 
                              sort_by="newest", progress_callback=None):
    """Download audio from videos in a YouTube channel with progress tracking.
    
    Args:
        channel_id (str): YouTube channel ID or URL
        output_dir (str, optional): Directory to save the files
        limit (int): Maximum number of videos to download
        start_index (int): Starting index for pagination
        sort_by (str): Sort order - "newest", "oldest", or "popular"
        progress_callback (callable): Function to call with progress updates
        
    Returns:
        list: List of paths to the downloaded audio files
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    videos = get_channel_videos(channel_id, limit, start_index, sort_by)
    downloaded_files = []
    
    for i, url in enumerate(videos):
        try:
            logger.info(f"Downloading video {i+1}/{len(videos)}: {url}")
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i, len(videos), url, "downloading")
                
            file_path = download_youtube_audio(url, output_dir)
            downloaded_files.append(file_path)
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i, len(videos), url, "completed")
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i, len(videos), url, "error", str(e))
    
    return downloaded_files

def convert_to_wav(input_path, sample_rate=16000):
    """Convert audio file to WAV format.
    
    Args:
        input_path (str): Path to input audio file
        sample_rate (int): Sample rate for the output WAV file
        
    Returns:
        str: Path to the converted WAV file
    """
    try:
        # If FFmpeg is not available, return the original path
        if not FFMPEG_AVAILABLE:
            logger.warning("FFmpeg not available, returning original audio path")
            return input_path
            
        # Get the file extension without the dot
        file_ext = os.path.splitext(input_path)[1][1:].lower()
        
        # If already a WAV file with correct sample rate, return the path
        if file_ext == 'wav':
            try:
                # Check the sample rate using pydub
                audio = AudioSegment.from_file(input_path, format='wav')
                if audio.frame_rate == sample_rate:
                    return input_path
            except Exception:
                # If there's an error checking the sample rate, convert anyway
                pass
        
        # Create a temporary file for the output
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"{uuid.uuid4().hex[:8]}.wav")
        
        # Convert using FFmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-i', input_path,  # Input file
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono audio
            '-vn',  # No video
            output_path  # Output file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
        # Return original path as fallback
        return input_path
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        # Return original path as fallback
        return input_path

def convert_to_mp3(input_path, output_path=None):
    """Convert audio file to MP3 format.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str, optional): Path for the output MP3 file
        
    Returns:
        str: Path to the converted MP3 file
    """
    try:
        if output_path is None:
            output_dir = os.path.dirname(input_path)
            output_name = f"{os.path.splitext(os.path.basename(input_path))[0]}.mp3"
            output_path = os.path.join(output_dir, output_name)
        
        # Convert using FFmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-i', input_path,  # Input file
            '-vn',  # No video
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-b:a', '192k',  # Bitrate
            output_path  # Output file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
        raise Exception(f"Failed to convert to MP3: {e}")
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        raise Exception(f"Failed to convert to MP3: {e}")

def extract_audio_from_video(video_path):
    """Extract audio from video file.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        str: Path to the extracted audio file
    """
    try:
        output_dir = os.path.dirname(video_path)
        output_path = os.path.join(output_dir, f"{uuid.uuid4().hex[:8]}.mp3")
        
        # Extract audio using FFmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-i', video_path,  # Input file
            '-vn',  # No video
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-b:a', '192k',  # Bitrate
            output_path  # Output file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Extracted audio from {video_path} to {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg extraction failed: {e.stderr.decode() if e.stderr else str(e)}")
        raise Exception(f"Failed to extract audio: {e}")
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise Exception(f"Failed to extract audio: {e}")

def save_transcript(text, source_name="transcript", output_dir=None):
    """Save transcription text to a file.
    
    Args:
        text (str): Transcription text
        source_name (str): Name to use for the file
        output_dir (str, optional): Directory to save the file
        
    Returns:
        str: Path to the saved file
    """
    if output_dir is None:
        output_dir = create_output_dir()
    
    # Clean up the source name to use as a filename
    clean_name = re.sub(r'[^\w\s-]', '', source_name).strip().lower()
    clean_name = re.sub(r'[-\s]+', '-', clean_name)
    
    # Add timestamp for uniqueness
    timestamp = uuid.uuid4().hex[:6]
    filename = f"{clean_name}_{timestamp}.txt"
    
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Saved transcript to {file_path}")
    return file_path

def cleanup_temp_files(file_list):
    """Clean up temporary files.
    
    Args:
        file_list (list): List of file paths to clean up
    """
    for file_path in file_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove {file_path}: {e}") 
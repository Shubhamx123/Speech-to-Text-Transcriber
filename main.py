import streamlit as st
import os
import tempfile
import time
import threading
import logging
import base64
import uuid
import sys
import argparse

# Set up basic logging first
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('streamlit_app')

# Import UI components with try/except
try:
    import streamlit.components.v1 as components
except ImportError as e:
    logger.error(f"Failed to import streamlit components: {e}")
    components = None

# Set more specific loggers
file_logger = logging.getLogger('feature.file_upload') 
youtube_logger = logging.getLogger('feature.youtube')
recording_logger = logging.getLogger('feature.recording')

# Try importing our modules with error handling
try:
    from transcriber import WhisperTranscriber, SpringLabASR, WHISPER_AVAILABLE, SPRING_LAB_LANGUAGES, live_record_with_callback
    from utils import download_youtube_audio, download_all_channel_videos, save_transcript, create_output_dir, get_channel_info
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    logger.error(f"Module import error: {e}")
    WHISPER_AVAILABLE = False
    SPRING_LAB_LANGUAGES = ["english"]
    
    # Define placeholder functions to avoid errors
    def live_record_with_callback(*args, **kwargs):
        logger.error("live_record_with_callback not available")
        return None
        
    # Create empty classes for fallback
    class WhisperTranscriber:
        def __init__(self, *args, **kwargs):
            pass
            
    class SpringLabASR:
        def __init__(self, *args, **kwargs):
            pass
            
    def download_youtube_audio(*args, **kwargs):
        return None
        
    def download_all_channel_videos(*args, **kwargs):
        return []
        
    def save_transcript(*args, **kwargs):
        return None
        
    def create_output_dir():
        return "output"
        
    def get_channel_info(*args, **kwargs):
        return {"title": "Error", "video_count": 0}

# Create output directory
output_dir = create_output_dir()

# Configure Streamlit page
st.set_page_config(
    page_title="Speech-to-Text Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if argument is provided for specifying port
if len(sys.argv) > 1:
    port_arg = [arg for arg in sys.argv if "--server.port" in arg]
    if not port_arg:
        logger.warning("Port argument not provided. Using default.")
        logger.warning("Use: streamlit run main.py --server.port=8080")

# Initialize session state variables
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = None
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = "idle"  # idle, recording, stopped
if 'recording_time' not in st.session_state:
    st.session_state.recording_time = 0
if 'recording_path' not in st.session_state:
    st.session_state.recording_path = None
if 'stop_recording' not in st.session_state:
    st.session_state.stop_recording = threading.Event()
if 'filename' not in st.session_state:
    st.session_state.filename = "transcription"

def update_recording_progress(seconds):
    """Update recording progress in session state"""
    try:
        recording_logger.debug(f"Updating recording progress: {seconds:.2f} seconds")
        st.session_state.recording_time = seconds
    except Exception as e:
        recording_logger.error(f"Error updating recording progress: {e}", exc_info=True)

def start_recording(duration=10):
    """Start audio recording"""
    try:
        # Reset stop event
        recording_logger.debug(f"Starting recording with duration: {duration} seconds")
        recording_logger.debug(f"Current session state keys: {list(st.session_state.keys())}")
        
        # Make sure stop_recording is initialized
        if 'stop_recording' not in st.session_state:
            recording_logger.debug("Initializing stop_recording event")
            st.session_state.stop_recording = threading.Event()
        else:
            recording_logger.debug("stop_recording already initialized")
        
        st.session_state.recording_state = "recording"
        st.session_state.recording_time = 0
        
        # Start recording in a thread
        def record_thread():
            try:
                recording_logger.debug("Recording thread started")
                
                # Check audio devices
                try:
                    import sounddevice as sd
                    devices = sd.query_devices()
                    recording_logger.debug(f"Audio devices: {devices}")
                    default_input = sd.query_devices(kind='input')
                    recording_logger.debug(f"Default input device: {default_input}")
                except Exception as e:
                    recording_logger.error(f"Error checking audio devices: {e}", exc_info=True)
                
                recording_path = live_record_with_callback(
                    callback=update_recording_progress,
                    stop_event=st.session_state.stop_recording,
                    max_duration=duration
                )
                recording_logger.debug(f"Recording completed, path: {recording_path}")
                st.session_state.recording_path = recording_path
                st.session_state.recording_state = "stopped"
            except Exception as e:
                recording_logger.error(f"Recording error: {e}", exc_info=True)
                st.session_state.recording_state = "error"
        
        thread = threading.Thread(target=record_thread)
        thread.daemon = True  # Make thread daemon so it doesn't block app shutdown
        thread.start()
        recording_logger.debug(f"Recording thread launched: {thread.name}")
    except Exception as e:
        recording_logger.error(f"Error setting up recording: {e}", exc_info=True)

def stop_recording():
    """Stop ongoing recording"""
    try:
        recording_logger.debug("Attempting to stop recording")
        if st.session_state.recording_state == "recording":
            recording_logger.debug("Setting stop_recording event")
            st.session_state.stop_recording.set()
        else:
            recording_logger.debug(f"Not stopping - current state: {st.session_state.recording_state}")
    except Exception as e:
        recording_logger.error(f"Error stopping recording: {e}", exc_info=True)

def get_download_link(text, filename="transcription.txt", link_text="Download Transcription"):
    """Generate a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def display_transcription_result(result, engine):
    """Display transcription result based on the engine used"""
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    # Extract text based on engine
    if engine == "whisper":
        if "text" in result:
            st.session_state.transcribed_text = result["text"]
            
            # Show language and timing info
            col1, col2 = st.columns(2)
            if "language" in result:
                col1.info(f"Detected language: {result['language']}")
            if "time_taken" in result:
                col2.info(f"Processing time: {result['time_taken']:.2f} seconds")
    
    elif engine == "spring_lab":
        if "transcript" in result:
            st.session_state.transcribed_text = result["transcript"]
            
            # Show language and timing info
            col1, col2 = st.columns(2)
            if "detected_language" in result:
                col1.info(f"Detected language: {result['detected_language']}")
            if "time_taken" in result:
                col2.info(f"Processing time: {result['time_taken']:.2f} seconds")
    
    # Store the entire result for reference
    st.session_state.transcription_result = result
    
    # Display text area with result
    st.text_area(
        "Transcription Result", 
        value=st.session_state.transcribed_text,
        height=300
    )
    
    # Download options
    col1, col2 = st.columns(2)
    
    # Option 1: Direct download
    filename = f"{st.session_state.filename}_{uuid.uuid4().hex[:6]}.txt"
    col1.markdown(
        get_download_link(st.session_state.transcribed_text, filename),
        unsafe_allow_html=True
    )
    
    # Option 2: Save to output directory
    if col2.button("Save to Output Folder"):
        saved_path = save_transcript(
            st.session_state.transcribed_text,
            st.session_state.filename,
            output_dir
        )
        st.success(f"Saved to: {saved_path}")

# Main app UI
st.title("🎙️ Speech-to-Text Transcriber")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Engine selection
    engine_options = []
    if WHISPER_AVAILABLE:
        engine_options.append("whisper")
    engine_options.append("spring_lab")
    
    if not engine_options:
        st.error("No transcription engines are available!")
        st.stop()
    
    engine = st.selectbox(
        "Transcription Engine",
        options=engine_options,
        index=0
    )
    
    # Whisper model selection
    whisper_model = "base"
    if engine == "whisper":
        whisper_model = st.selectbox(
            "Whisper Model",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to base
            help="Larger models are more accurate but slower and require more resources"
        )
    elif not WHISPER_AVAILABLE and "spring_lab" in engine_options:
        st.warning("OpenAI Whisper is not available in this environment. Using SpringLab ASR instead.")
        st.info("To use Whisper locally, install it with: pip install openai-whisper")
    
    # Language selection
    language = ""
    if engine == "spring_lab":
        language = st.selectbox(
            "Language",
            options=[""] + SPRING_LAB_LANGUAGES,
            index=0,
            help="Leave empty for auto-detection"
        )
    
    # Information about engines
    st.markdown("---")
    st.subheader("About Engines:")
    st.info("""
    **Whisper**: OpenAI's local speech recognition model
    
    **SpringLab**: IIT Madras ASR API with multiple Indian language support
    """)

# Create tabs for different input types
tab1, tab2, tab3 = st.tabs(["📁 File Upload", "🎬 YouTube", "🎤 Live Recording"])

# Tab 1: File Upload
with tab1:
    st.header("Upload Audio/Video File")
    uploaded_file = st.file_uploader(
        "Choose an audio or video file", 
        type=["mp3", "wav", "ogg", "flac", "mp4", "avi", "mov", "mkv", "m4a"]
    )
    
    if uploaded_file is not None:
        # Save filename for later use
        st.session_state.filename = os.path.splitext(uploaded_file.name)[0]
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_file_path = tmp.name
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        if st.button("Transcribe File"):
            with st.spinner("Transcribing..."):
                try:
                    # Get the right transcriber
                    if engine == "whisper":
                        transcriber = WhisperTranscriber(model_name=whisper_model)
                        if language:
                            transcriber.language = language
                        
                        # Check if it's a video file
                        if uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            result = transcriber.transcribe_video_file(temp_file_path)
                        else:
                            result = transcriber.transcribe_audio_file(temp_file_path)
                    
                    else:  # spring_lab
                        transcriber = SpringLabASR()
                        if language:
                            transcriber.language = language
                            
                        # Check if it's a video file
                        if uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            result = transcriber.transcribe_video_file(temp_file_path)
                        else:
                            result = transcriber.transcribe_audio_file(temp_file_path)
                    
                    # Display results
                    display_transcription_result(result, engine)
                    
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
                    logger.error(f"Transcription error: {e}")
                
                finally:
                    # Clean up
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass

# Tab 2: YouTube
with tab2:
    st.header("YouTube Transcription")
    
    # Select between single video or channel
    yt_mode = st.radio(
        "Select YouTube Source",
        options=["Single Video", "Channel (Multiple Videos)"]
    )
    
    if yt_mode == "Single Video":
        youtube_url = st.text_input(
            "Enter YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        if youtube_url:
            st.session_state.filename = f"youtube_video_{youtube_url.split('=')[-1]}"
            
            if st.button("Transcribe YouTube Video"):
                with st.spinner("Downloading and transcribing video..."):
                    try:
                        # Get the right transcriber
                        if engine == "whisper":
                            transcriber = WhisperTranscriber(model_name=whisper_model)
                            if language:
                                transcriber.language = language
                            result = transcriber.transcribe_youtube_video(youtube_url)
                        
                        else:  # spring_lab
                            transcriber = SpringLabASR()
                            if language:
                                transcriber.language = language
                            result = transcriber.transcribe_youtube_video(youtube_url)
                        
                        # Display results
                        display_transcription_result(result, engine)
                        
                    except Exception as e:
                        st.error(f"YouTube transcription failed: {str(e)}")
                        logger.error(f"YouTube transcription error: {e}")
    
    else:  # Channel mode
        channel_id = st.text_input(
            "Enter YouTube Channel ID or URL",
            placeholder="Channel ID or full URL"
        )
        
        # Create columns for configuration
        col1, col2 = st.columns(2)
        
        with col1:
            max_videos = st.slider(
                "Maximum number of videos to transcribe",
                min_value=1,
                max_value=100,
                value=5
            )
            
            sorting = st.selectbox(
                "Sort videos by",
                options=["newest", "oldest", "popular"],
                index=0
            )
        
        with col2:
            batch_size = st.slider(
                "Videos per batch",
                min_value=1,
                max_value=20,
                value=5,
                help="Processing videos in batches helps manage resources and provides better progress tracking"
            )
            
            start_index = st.number_input(
                "Start from video #",
                min_value=0,
                value=0,
                help="Skip initial videos (0 = start from beginning)"
            )
        
        if channel_id:
            st.session_state.filename = f"youtube_channel_{channel_id.split('/')[-1]}"
            
            # Initialize session state for channel processing
            if 'channel_info' not in st.session_state:
                st.session_state.channel_info = None
            if 'channel_processing' not in st.session_state:
                st.session_state.channel_processing = False
            if 'channel_results' not in st.session_state:
                st.session_state.channel_results = []
            if 'channel_progress' not in st.session_state:
                st.session_state.channel_progress = {
                    'total_videos': 0,
                    'processed': 0,
                    'current_video': None,
                    'status': 'idle'
                }
            
            # Get channel info first if we have a channel ID but no info
            if (st.session_state.channel_info is None or 
                st.session_state.channel_info.get('channel_id') != channel_id) and channel_id:
                try:
                    # Get the right transcriber
                    if engine == "whisper":
                        transcriber = WhisperTranscriber(model_name=whisper_model)
                    else:  # spring_lab
                        transcriber = SpringLabASR()
                        
                    # Get channel info
                    with st.spinner("Fetching channel information..."):
                        channel_info = transcriber.get_channel_info(channel_id)
                        st.session_state.channel_info = channel_info
                        
                        # Display channel info
                        st.success(f"Channel: {channel_info['title']}")
                        st.info(f"Total videos: {channel_info['video_count']}")
                except Exception as e:
                    st.error(f"Error fetching channel info: {str(e)}")
                    logger.error(f"Channel info error: {e}")
            
            # Display channel info if available
            if st.session_state.channel_info:
                st.subheader(f"Channel: {st.session_state.channel_info['title']}")
                total_videos = st.session_state.channel_info.get('video_count', 'Unknown')
                
                if max_videos > total_videos and isinstance(total_videos, int):
                    st.warning(f"Requested {max_videos} videos but channel only has {total_videos} videos")
                    max_videos = total_videos
            
            # Create progress containers (we'll update these during processing)
            progress_container = st.empty()
            current_video_container = st.empty()
            results_container = st.empty()
            
            # Reset button if already processing
            if st.session_state.channel_processing:
                if st.button("Reset Processing"):
                    st.session_state.channel_processing = False
                    st.session_state.channel_results = []
                    st.session_state.channel_progress = {
                        'total_videos': 0,
                        'processed': 0,
                        'current_video': None,
                        'status': 'idle'
                    }
                    st.experimental_rerun()
                    
                # Display current progress
                with progress_container.container():
                    progress_value = 0
                    if st.session_state.channel_progress['total_videos'] > 0:
                        progress_value = st.session_state.channel_progress['processed'] / st.session_state.channel_progress['total_videos']
                        
                    progress_bar = st.progress(progress_value)
                    st.markdown(f"**Status:** {st.session_state.channel_progress['status']}")
                    st.markdown(f"**Processed:** {st.session_state.channel_progress['processed']} / {st.session_state.channel_progress['total_videos']} videos")
                    
                with current_video_container.container():
                    if st.session_state.channel_progress['current_video']:
                        st.markdown(f"**Currently processing:** {st.session_state.channel_progress['current_video']}")
            
            # Start processing button
            elif st.button("Transcribe Channel Videos"):
                # Mark as processing
                st.session_state.channel_processing = True
                st.session_state.channel_results = []
                st.session_state.channel_progress = {
                    'total_videos': max_videos,
                    'processed': 0,
                    'current_video': None,
                    'status': 'starting'
                }
                
                # Define progress callback function
                def update_progress(index, total, url, status, data=None):
                    """Update progress state for UI updates"""
                    # Update progress info
                    if status == "channel_info":
                        st.session_state.channel_info = data
                    elif status == "download_downloading":
                        st.session_state.channel_progress['status'] = f"Downloading video {index+1}/{total}"
                        st.session_state.channel_progress['current_video'] = url
                    elif status == "transcribing":
                        st.session_state.channel_progress['status'] = f"Transcribing video {index+1}/{total}"
                    elif status == "completed":
                        st.session_state.channel_progress['processed'] += 1
                        if isinstance(data, dict) and 'text' in data:
                            # Add video info
                            data['video_url'] = url
                            data['index'] = index
                            st.session_state.channel_results.append(data)
                    elif status == "error":
                        st.session_state.channel_progress['processed'] += 1
                        # Add error info
                        error_info = {
                            'error': data,
                            'video_url': url,
                            'index': index
                        }
                        st.session_state.channel_results.append(error_info)
                    elif status == "batch_end":
                        st.session_state.channel_progress['status'] = f"Completed batch {data['batch_index']+1}"
                
                # Get the right transcriber
                if engine == "whisper":
                    transcriber = WhisperTranscriber(model_name=whisper_model)
                    if language:
                        transcriber.language = language
                else:  # spring_lab
                    transcriber = SpringLabASR()
                    if language:
                        transcriber.language = language
                
                # Create a background thread to process videos
                def process_channel():
                    try:
                        results = transcriber.transcribe_youtube_channel(
                            channel_id, 
                            limit=max_videos,
                            start_index=start_index,
                            batch_size=batch_size,
                            sort_by=sorting,
                            progress_callback=update_progress
                        )
                        
                        # Process is complete
                        st.session_state.channel_progress['status'] = "completed"
                        
                        # Combine results to display
                        combine_channel_results()
                            
                    except Exception as e:
                        logger.error(f"Channel transcription error: {e}")
                        st.session_state.channel_progress['status'] = f"Error: {str(e)}"
                
                # Start processing thread
                processing_thread = threading.Thread(target=process_channel)
                processing_thread.daemon = True
                processing_thread.start()
                
                # Force a rerun to show progress
                st.experimental_rerun()
                
            # Function to combine and display results
            def combine_channel_results():
                """Combine all channel results into a single text for display and download"""
                if not st.session_state.channel_results:
                    return
                
                combined_text = f"Transcription of {st.session_state.channel_info.get('title', 'YouTube Channel')}\n"
                combined_text += f"Total videos processed: {len(st.session_state.channel_results)}\n\n"
                
                for i, result in enumerate(st.session_state.channel_results):
                    video_num = result.get('index', i) + 1
                    combined_text += f"--- Video {video_num} ---\n"
                    
                    if 'error' in result:
                        combined_text += f"Error: {result['error']}\n"
                    elif engine == "whisper" and "text" in result:
                        combined_text += result["text"]
                    elif engine == "spring_lab" and "transcript" in result:
                        combined_text += result["transcript"]
                    else:
                        combined_text += f"Error: Unknown result format"
                    
                    combined_text += "\n\n"
                
                # Save the combined text
                st.session_state.transcribed_text = combined_text
            
            # Display the combined results
            if st.session_state.channel_processing and st.session_state.channel_results:
                combine_channel_results()
                
                with results_container.container():
                    st.subheader("Transcription Results")
                    
                    # Display text area with result
                    st.text_area(
                        "Combined Transcription", 
                        value=st.session_state.transcribed_text,
                        height=400
                    )
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    
                    # Option 1: Direct download
                    filename = f"{st.session_state.filename}_{uuid.uuid4().hex[:6]}.txt"
                    col1.markdown(
                        get_download_link(st.session_state.transcribed_text, filename),
                        unsafe_allow_html=True
                    )
                    
                    # Option 2: Save to output directory
                    if col2.button("Save to Output Folder"):
                        saved_path = save_transcript(
                            st.session_state.transcribed_text,
                            st.session_state.filename,
                            output_dir
                        )
                        st.success(f"Saved to: {saved_path}")
                        
                    # Display individual video results
                    st.subheader("Individual Video Results")
                    for i, result in enumerate(st.session_state.channel_results):
                        video_num = result.get('index', i) + 1
                        
                        with st.expander(f"Video {video_num}"):
                            if 'error' in result:
                                st.error(f"Error: {result['error']}")
                            else:
                                if engine == "whisper" and "text" in result:
                                    text = result["text"]
                                elif engine == "spring_lab" and "transcript" in result:
                                    text = result["transcript"]
                                else:
                                    text = "No transcript available"
                                
                                st.text_area(
                                    f"Video {video_num} Transcript",
                                    value=text,
                                    height=200
                                )
                                
                                # Show metadata
                                cols = st.columns(3)
                                if "time_taken" in result:
                                    cols[0].info(f"Processing time: {result['time_taken']:.2f} seconds")
                                if "language" in result:
                                    cols[1].info(f"Detected language: {result['language']}")
                                if "video_url" in result:
                                    cols[2].markdown(f"[Open Video]({result['video_url']})")
                                    
                                # Individual download
                                single_filename = f"video_{video_num}_{uuid.uuid4().hex[:6]}.txt"
                                st.markdown(
                                    get_download_link(text, single_filename, f"Download Video {video_num} Transcript"),
                                    unsafe_allow_html=True
                                )

# Tab 3: Live Recording
with tab3:
    st.header("Live Microphone Recording")
    recording_logger.debug("Rendering Live Recording tab")
    
    # Add helper text about browser permissions
    st.info("""
    ⚠️ **Browser Microphone Access Required**
    
    This feature requires microphone access in your browser. If you encounter issues:
    1. Make sure you allow microphone access when prompted
    2. If recording doesn't work, use the File Upload tab as an alternative
    """)
    
    # Option to use file upload as backup
    with st.expander("Having trouble with browser recording? Use file upload instead"):
        backup_file = st.file_uploader(
            "Upload a recorded audio file", 
            type=["mp3", "wav", "ogg", "webm", "m4a"]
        )
        if backup_file is not None:
            # Save file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(backup_file.name)[1]) as tmp:
                tmp.write(backup_file.getvalue())
                temp_file_path = tmp.name
            
            st.success(f"File uploaded: {backup_file.name}")
            st.session_state.recording_path = temp_file_path
    
    # Recording duration
    col1, col2 = st.columns([3, 1])
    with col1:
        duration = st.slider(
            "Recording Duration (seconds)", 
            min_value=5, 
            max_value=60, 
            value=10
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        use_browser_recording = st.checkbox("Use browser recording", value=True)
        
    recording_logger.debug(f"Selected recording duration: {duration}")
    
    if use_browser_recording:
        # Display simple HTML recorder
        st.markdown("## Browser Microphone Recorder")
        st.markdown("Click the button below and allow microphone access when prompted.")
        
        # Simple recorder that doesn't rely on component communication
        simple_recorder_html = f"""
        <div style="text-align: center; padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
            <button id="startBtn" style="background-color: #FF4B4B; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                Start Recording
            </button>
            <button id="stopBtn" style="background-color: blue; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; display: none;">
                Stop Recording
            </button>
            <div id="status" style="margin-top: 10px; font-weight: bold;"></div>
            <div id="timer" style="font-size: 24px; margin: 10px 0; display: none;">00:00</div>
            <audio id="audio" controls style="width: 100%; margin-top: 10px; display: none;"></audio>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const status = document.getElementById('status');
                const timer = document.getElementById('timer');
                const audio = document.getElementById('audio');
                
                let mediaRecorder;
                let chunks = [];
                let timerInterval;
                let startTime;
                
                function updateTimer() {{
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
                    const secs = (elapsed % 60).toString().padStart(2, '0');
                    timer.textContent = `${{mins}}:${{secs}}`;
                    
                    if (elapsed >= {duration}) {{
                        stopRecording();
                    }}
                }}
                
                async function startRecording() {{
                    chunks = [];
                    status.textContent = "Requesting microphone access...";
                    
                    try {{
                        const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                        
                        try {{
                            mediaRecorder = new MediaRecorder(stream);
                        }} catch (e) {{
                            console.error('MediaRecorder error:', e);
                            status.textContent = "Your browser doesn't support audio recording. Please try another browser.";
                            return;
                        }}
                        
                        mediaRecorder.ondataavailable = (e) => {{
                            chunks.push(e.data);
                        }};
                        
                        mediaRecorder.onstop = () => {{
                            const blob = new Blob(chunks, {{ 'type' : 'audio/webm; codecs=opus' }});
                            const url = URL.createObjectURL(blob);
                            audio.src = url;
                            audio.style.display = 'block';
                            status.textContent = "Recording ready! Download and upload using the controls below.";
                            
                            // Create download link
                            const downloadLink = document.createElement('a');
                            downloadLink.href = url;
                            downloadLink.download = 'recording.webm';
                            downloadLink.style.display = 'block';
                            downloadLink.style.margin = '10px auto';
                            downloadLink.style.padding = '8px 15px';
                            downloadLink.style.backgroundColor = '#4CAF50';
                            downloadLink.style.color = 'white';
                            downloadLink.style.textAlign = 'center';
                            downloadLink.style.textDecoration = 'none';
                            downloadLink.style.borderRadius = '4px';
                            downloadLink.textContent = 'Download Recording';
                            
                            // Add download link after audio element
                            audio.parentNode.insertBefore(downloadLink, audio.nextSibling);
                            
                            // Add instructions
                            const instructions = document.createElement('p');
                            instructions.textContent = "After downloading, upload the file using the 'Upload a recorded audio file' option above.";
                            instructions.style.marginTop = '10px';
                            downloadLink.parentNode.insertBefore(instructions, downloadLink.nextSibling);
                        }};
                        
                        mediaRecorder.start(1000);
                        status.textContent = "Recording... (speak now)";
                        startBtn.style.display = 'none';
                        stopBtn.style.display = 'inline-block';
                        timer.style.display = 'block';
                        startTime = Date.now();
                        timerInterval = setInterval(updateTimer, 1000);
                    }} catch (err) {{
                        console.error('Error accessing microphone:', err);
                        status.textContent = `Error: ${{err.message || 'Microphone access denied'}}`;
                    }}
                }}
                
                function stopRecording() {{
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
                        mediaRecorder.stop();
                        mediaRecorder.stream.getTracks().forEach(track => track.stop());
                        clearInterval(timerInterval);
                        startBtn.style.display = 'inline-block';
                        stopBtn.style.display = 'none';
                    }}
                }}
                
                startBtn.addEventListener('click', startRecording);
                stopBtn.addEventListener('click', stopRecording);
            }});
        </script>
        """
        
        components.html(simple_recorder_html, height=400)
        
        st.markdown("""
        ### How to use:
        1. Click 'Start Recording' and allow microphone access
        2. Speak into your microphone
        3. Click 'Stop Recording' when done (or wait for the timer to complete)
        4. Use the download button to save your recording
        5. Upload the file using the 'Upload a recorded audio file' option at the top of this tab
        """)
    
    # Transcribe button for the uploaded/recorded file
    if st.session_state.recording_path:
        if st.button("Transcribe Recording"):
            recording_logger.debug(f"Transcribing file: {st.session_state.recording_path}")
            with st.spinner("Transcribing..."):
                try:
                    # Get the right transcriber
                    if engine == "whisper":
                        recording_logger.debug(f"Using Whisper model: {whisper_model}")
                        transcriber = WhisperTranscriber(model_name=whisper_model)
                        if language:
                            transcriber.language = language
                        result = transcriber.transcribe_audio_file(st.session_state.recording_path)
                    
                    else:  # spring_lab
                        recording_logger.debug(f"Using SpringLab with language: {language}")
                        transcriber = SpringLabASR()
                        if language:
                            transcriber.language = language
                        result = transcriber.transcribe_audio_file(st.session_state.recording_path)
                    
                    # Use a meaningful filename
                    st.session_state.filename = f"recording_{int(time.time())}"
                    
                    # Display results
                    recording_logger.debug(f"Transcription result: {result}")
                    display_transcription_result(result, engine)
                    
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
                    recording_logger.error(f"Transcription error: {e}", exc_info=True)

# Run with custom port
if __name__ == "__main__":
    # Note: The actual port setting is done when running the app with:
    # streamlit run main.py --server.port=8080
    logger.info("Starting Speech-to-Text Transcriber app")
    logger.info("To set a specific port, use: streamlit run main.py --server.port=8080") 
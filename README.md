# Speech-to-Text Transcriber

A Streamlit application for speech-to-text transcription with multiple transcription engines and input options.

## Features

- **Multiple Transcription Engines**: 
  - OpenAI Whisper (local)
  - SpringLab ASR (API-based)

- **Multiple Input Methods**:
  - File Upload (audio/video)
  - YouTube Video URL
  - YouTube Channel (batch processing)
  - Live Microphone Recording

- **Supported Languages**:
  - Whisper: 100+ languages with auto-detection
  - SpringLab: Support for Indian languages (Hindi, Tamil, Telugu, etc.)

## Live Demo

Try the app at [Streamlit Cloud](https://speech-to-text-transcriber.streamlit.app)

## How to Use

1. Select a transcription engine in the sidebar
2. Choose your input method (File, YouTube, or Recording)
3. Configure options as needed
4. Start transcription
5. Download or save your transcription results

## Local Setup

```bash
# Clone the repository
git clone https://github.com/username/Speech-to-Text-Transcriber.git
cd Speech-to-Text-Transcriber

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## Deploying to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Sign in with GitHub
4. Select this repository
5. Set main file path to: `main.py`
6. Click "Deploy"

Your app will be available at a URL like `https://username-speech-to-text-transcriber.streamlit.app`

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- Internet connection (for YouTube and SpringLab features)

## License

MIT 
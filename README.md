# Speech-to-Text Transcription System

A comprehensive speech-to-text transcription system with multiple engines and a web interface.

## Features

### Multiple Transcription Engines
- **Whisper**: OpenAI's open-source speech recognition model
  - Multiple model sizes (tiny, base, small, medium, large)
  - Advanced language detection and auto-correction
  - Offline capability
- **SPRING Lab**: IIT Madras speech recognition API
  - Real-time transcription
  - Indian language support
  - WebVTT caption generation
- **Bhashini**: Indian government's multilingual ASR platform
  - Support for Indian languages
  - API-based transcription

### Input Sources
- Audio files (.mp3, .wav, .ogg, .flac, .m4a)
- Video files (.mp4, .avi, .mov, .mkv)
- YouTube videos (including Shorts)
- Live microphone recording

### Web Interface
- User-friendly interface with Bootstrap UI
- Real-time transcription progress tracking
- Multiple engine comparison
- Features:
  - Copy results to clipboard
  - Download transcriptions
  - Live recording with start/stop controls
  - Progress bar and status updates
  - Language selection
  - Model selection for Whisper

### Output Formats
- Plain text transcription
- JSON format with metadata
- WebVTT captions (SPRING Lab)
- Detailed language detection information

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install FFmpeg (required for audio processing):
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

## Usage

### Web Interface
1. Start the server:
   ```bash
   python app.py
   ```
2. Open http://127.0.0.1:5000 in your browser
3. Select transcription engine, input source, and options
4. Start transcription and monitor progress

### Command Line Usage

#### Whisper
```bash
python whisper.py --model [tiny|base|small|medium|large] --source [youtube|video|audio|live] --input [FILE_PATH/URL] --language [LANGUAGE_CODE] --output [OUTPUT_FILE]
```

Features:
- Automatic language detection
- Language correction for commonly confused languages
- Special handling for Hindi/English confusion
- CUDA support for GPU acceleration

#### SPRING Lab
```bash
python spring_lab.py --source [youtube|video|audio|live] --input [FILE_PATH/URL] --language [LANGUAGE_CODE] --vtt
```

Supported languages:
- Bengali, English, Gujarati, Hindi, Kannada
- Malayalam, Marathi, Odia, Punjabi
- Sanskrit, Tamil, Telugu, Urdu

#### Bhashini
```bash
python bhashini.py --source [youtube|video|audio|live] --input [FILE_PATH/URL] --language [LANGUAGE_CODE] --api-key [YOUR_API_KEY]
```

Required:
- Get API key from [Bhashini Developer Portal](https://bhashini.gov.in/developer)
- Set as `--api-key` parameter or `BHASHINI_API_KEY` environment variable

### Language Support

#### Whisper
- Supports 96+ languages with automatic detection
- Advanced language correction for:
  - Hindi transcribed as English
  - Sanskrit/Urdu confusion
  - Various Indian languages
  - Persian/Arabic differentiation

#### SPRING Lab
- Specialized in Indian languages
- Automatic language detection
- High accuracy for regional languages

#### Bhashini
- Focused on Indian languages
- Check [Bhashini's documentation](https://bhashini.gov.in/services) for current language list

## Error Handling
- Automatic retry for YouTube downloads
- Multiple download methods (yt-dlp, pytube, direct)
- Detailed logging in logs/app.log
- User-friendly error messages in web interface

## License
This project is open-source under the MIT License.

## Contributing
Contributions are welcome! Please feel free to submit pull requests. 
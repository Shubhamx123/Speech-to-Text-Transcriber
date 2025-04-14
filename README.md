# Speech-to-Text Transcription Tools

This project provides multiple speech-to-text (STT) transcription tools that can be used independently or compared side-by-side.

## Features

- **Multiple Transcription Engines**:
  - **Bhashini**: Indian government's multilingual ASR platform
  - **SPRING Lab**: IIT Madras speech recognition API
  - **Whisper**: OpenAI's open-source speech recognition model (offline capability)
  - **Transkriptor**: Unified interface to compare all engines

- **Input Sources**:
  - Audio files (.mp3, .wav, etc.)
  - Video files (.mp4, etc.)
  - YouTube videos (via URL)
  - Live microphone recording

- **Output Formats**:
  - Console output
  - Text files
  - JSON format
  - WebVTT captions (Bhashini and SPRING Lab)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install FFmpeg (required for audio processing):
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

## Usage

### Bhashini ASR

```bash
python bhashini.py --source [youtube|video|audio|live] --input [FILE_PATH/URL] --language [LANGUAGE_CODE] --api-key [YOUR_API_KEY]
```

Required:
- Get an API key from [Bhashini Developer Portal](https://bhashini.gov.in/developer)
- Set it as `--api-key` parameter or as `BHASHINI_API_KEY` environment variable

### SPRING Lab ASR

```bash
python spring_lab.py --source [youtube|video|audio|live] --input [FILE_PATH/URL] --language [LANGUAGE_CODE] --vtt
```

Note: SPRING Lab API is free to use but may have usage limitations.

### Whisper

```bash
python whisper.py --model [tiny|base|small|medium|large] --source [youtube|video|audio|live] --input [FILE_PATH/URL] --language [LANGUAGE_CODE] --output [OUTPUT_FILE]
```

Note: Whisper works offline but requires downloading models. Larger models provide better accuracy but require more resources.

### Transkriptor (Unified Interface)

```bash
python transkriptor.py --engine [bhashini|spring_lab|whisper|all] --source [youtube|video|audio|live] --input [FILE_PATH/URL] --language [LANGUAGE_CODE] --api-key [BHASHINI_API_KEY] --model [WHISPER_MODEL] --output [OUTPUT_FILE]
```

The "all" engine option will compare results from all available engines.

## Examples

### Transcribe a YouTube video with Whisper

```bash
python whisper.py --source youtube --input "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output transcript.txt
```

### Live transcription with all engines

```bash
python transkriptor.py --engine all --source live --language hindi --live-duration 60
```

### Transcribe a video file with Bhashini

```bash
python bhashini.py --source video --input my_video.mp4 --language hi --api-key YOUR_API_KEY
```

## Supported Languages

### Bhashini
Various Indian languages including Hindi, Tamil, Telugu, etc. (Check [Bhashini's documentation](https://bhashini.gov.in/services))

### SPRING Lab
Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu

### Whisper
Supports 96+ languages with automatic language detection

## License

This project is open-source under the MIT License. 
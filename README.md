# Robust Speech-to-Text Transcriber

A powerful and flexible speech-to-text transcription system supporting multiple transcription engines, input methods, and languages.

## Features

### Multiple Transcription Engines
* **OpenAI Whisper (Local)**
  - Runs entirely on your machine
  - Multiple model sizes (small, medium, large)
  - GPU acceleration support
* **SpringLab ASR (API)**
  - Specialized for Indian languages
  - High accuracy for regional languages

### Input Methods
* **File Upload**
  - Support for various audio formats (WAV, MP3, OGG, FLAC)
  - Video file support (MP4, AVI, MOV)
  - Automatic format conversion
* **YouTube Integration**
  - Single video transcription
  - Batch processing for entire channels
  - Automatic audio extraction
* **Live Recording**
  - Browser-based microphone recording
  - Real-time audio capture
  - Secure HTTPS connection

### Language Support
* **Whisper Engine**
  - 100+ languages supported
  - Automatic language detection
  - Multilingual transcription
* **SpringLab Engine**
  - Specialized Indian language support
  - High accuracy for regional languages

## Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Speech-to-Text-Transcriber.git
cd Speech-to-Text-Transcriber
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Generate SSL certificate (required for HTTPS):
```bash
mkdir certs
openssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your environment variables in `.env`:
```ini
# Server Settings
HOST=<your-host>          # Default: 0.0.0.0
PORT=<your-port>          # Default: 8000

# SSL Configuration
SSL_CERT_PATH=<path>      # Path to SSL certificate
SSL_KEY_PATH=<path>       # Path to SSL private key

# API Configuration
SPRINGLAB_API_KEY=<key>   # Your SpringLab API key

# Application Settings
MAX_UPLOAD_SIZE_MB=1024   # Maximum upload size in MB
GPU_ENABLED=true         # Enable GPU acceleration if available
```

3. Generate SSL certificate (required for HTTPS/microphone access):
```bash
# Using the provided script (recommended):
./generate_cert.sh

# Or manually:
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -nodes \
  -out certs/cert.pem \
  -keyout certs/key.pem \
  -days 365 \
  -subj "/C=US/ST=State/L=City/O=Org/CN=localhost"
```

## Running the Application

1. Start all services:
```bash
./start_all.sh
```

2. Access the web interface:
```
https://localhost:your_port
```

3. Stop all services:
```bash
./stop_all.sh
```

## Project Structure
```
project/
├── app.py                 # Main Flask application
├── transcriber.py         # Transcription engines
├── utils.py              # Helper functions
├── celery_worker.py      # Background task worker
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── start_all.sh         # Service startup script
├── stop_all.sh          # Service shutdown script
├── templates/           # HTML templates
├── static/             # Frontend assets
├── certs/              # SSL certificates
├── logs/              # Application logs
└── output/            # Transcription results
```

## Security Notes

- Never commit `.env` files
- Keep SSL certificates private
- Protect API keys
- Use environment variables for sensitive data
- Generate new SSL certificates for production

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Whisper team for the amazing speech recognition model
- SpringLab for their Indian language ASR capabilities
- FFmpeg project for audio processing capabilities
- All contributors who have helped improve this project 
# Video Language & Accent Analysis Agent

A comprehensive AI-powered application that analyzes video content to detect spoken language and classify English accents with confidence scoring.

## Demo

**[Try the App Here](https://video-accent-analyzer.streamlit.app/)**

## 📋 Overview

This application performs intelligent two-stage analysis of video content:

1. **Language Detection**: Identifies the primary spoken language using SpeechBrain's VoxLingua107 model
2. **Accent Classification**: For English content, classifies regional accents (British, American, Australian, etc.) with confidence scoring

## Features

- **Multi-Language Detection**: Supports 107+ languages
- **English Accent Classification**: Identifies British, American, Australian, Canadian, Irish, New Zealand, and other regional accents
- **Confidence Scoring**: Provides detailed confidence percentages for all predictions
- **Mixed Language Handling**: Detects and analyzes videos with multiple languages
- **Video Source Support**: Works with YouTube URLs and direct video links
- **Real-time Processing**: Streamlit interface with progress indicators
- **Intelligent Chunking**: Processes audio in optimized segments for better accuracy

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for model downloads

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Alfred9/video-language-accent-analysis.git
   cd video-language-accent-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies** (if running locally):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install ffmpeg
   
   # On macOS
   brew install ffmpeg
   
   # On Windows
   # Download FFmpeg from https://ffmpeg.org/download.html
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## Dependencies

### Core Libraries
- `streamlit` - Web application framework
- `yt-dlp` - Video downloading and processing
- `moviepy` - Video/audio manipulation
- `speechbrain` - AI models for speech analysis
- `torch` & `torchaudio` - PyTorch framework
- `librosa` - Audio analysis
- `pydub` - Audio processing
- `ffmpeg-python` - Media processing

### Model Dependencies
- `speechbrain/lang-id-voxlingua107-ecapa` - Language identification
- `Jzuluaga/accent-id-commonaccent_ecapa` - Accent classification

*See `requirements.txt` for complete dependency list with versions*

## Usage Guide

### Basic Usage

1. **Open the application** in your web browser
2. **Enter a video URL** in the input field:
   - YouTube videos: `https://youtube.com/watch?v=VIDEO_ID`
   - Direct MP4 links: `https://example.com/video.mp4`
3. **Click "Analyze"** to start processing
4. **Wait for results** - processing typically takes 30-60 seconds

### Supported Input Formats

- **YouTube URLs**: Standard YouTube video links
- **Direct Video URLs**: MP4, AVI, MOV formats
- **Video Length**: Optimized for videos up to 4 minutes
- **Audio Quality**: Works with various audio bitrates

### Example URLs for Testing

```
# English (American accent)
https://youtube.com/watch?v=EXAMPLE1

# English (British accent)  
https://www.youtube.com/watch?v=As_bK8LipBY&pp=ygUKZ3JhbmQgdG91cg%3D%3D

# Mixed language content
https://youtube.com/watch?v=EXAMPLE3

# Non-English content
https://youtube.com/watch?v=EXAMPLE4
```

*Replace with actual working video URLs for testing*

## 📊 Output Analysis

### English Content with Accent Detection
```
Language Detected: English (95.2% confidence)
Detected Accent: American
English Accent Confidence: 87.3%
Summary: The detected accent is American with a confidence of 87.3%...
```

### Mixed Language Content
```
Primary Language: Spanish (78.4% confidence)
English Content: 35.2% of audio
English Accent Detected: British
English Accent Confidence: 72.1%
Summary: Mixed language detected: Primarily Spanish with 35.2% English content...
```

### Non-English Content
```
Language Detected: French (92.7% confidence)
Summary: Non-English language detected: French with 92.7% confidence...
```

## 🔧 Technical Architecture

### Processing Pipeline

1. **Video Download**: Uses yt-dlp to fetch video content
2. **Audio Extraction**: MoviePy extracts WAV audio at optimal quality
3. **Audio Chunking**: Splits into 15-30 second segments for analysis
4. **Language Detection**: VoxLingua107 model analyzes each chunk
5. **Accent Analysis**: CommonAccent model processes English segments
6. **Confidence Calculation**: Softmax probabilities provide confidence scores
7. **Result Aggregation**: Combines chunk-level predictions into final output

### Model Information

- **Language Model**: `speechbrain/lang-id-voxlingua107-ecapa`
  - Supports 107 languages
  - ECAPA-TDNN architecture
  - Trained on VoxLingua107 dataset

- **Accent Model**: `Jzuluaga/accent-id-commonaccent_ecapa`
  - Focuses on English accent varieties
  - CommonAccent dataset training
  - Regional accent classification

## 🚀 Deployment

### Streamlit Cloud Deployment

This application is deployed on Streamlit Cloud with the following configuration:

- **Python Version**: 3.9+
- **Memory Requirements**: 2GB+ recommended
- **Processing Time**: 30-120 seconds per video
- **Concurrent Users**: Supports multiple simultaneous analyses

### Local Development

For local development and testing:

```bash
# Development mode with debug logging
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run app.py --logger.level=debug
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Troubleshooting

### Common Issues

**Video Download Fails**:
- Ensure the URL is publicly accessible
- Check internet connection
- Try a different video URL

**Audio Extraction Errors**:
- Verify the video has an audio track
- Check if the video is too long (>4 minutes)

**Model Loading Issues**:
- Ensure stable internet connection for initial model download
- Check available disk space (models require ~2GB)

**Low Confidence Scores**:
- Audio quality may be poor
- Background noise or music interference
- Very short speech segments

### Performance Tips

- Use videos with clear, single-speaker audio for best results
- Avoid videos with heavy background music
- Shorter videos (1-3 minutes) process faster
- Good internet connection improves model loading time

## 📞 Support

For issues, questions, or feature requests, please:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed description
4. Include error logs and video URL (if applicable)

---

**Built with ❤️ using SpeechBrain, Streamlit, and modern AI technologies**

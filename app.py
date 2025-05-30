from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yt_dlp
import moviepy.editor as mp
import os
import tempfile
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier, LangID
import torch
import torchaudio
import librosa
import numpy as np
import logging
from pydub import AudioSegment
import ffmpeg

app = FastAPI()
templates = Jinja2Templates(directory="templates")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set torchaudio backend
torchaudio.set_audio_backend("soundfile")
logger.info("Torchaudio backend set to soundfile.")

# Load models at startup
logger.info("Loading accent model...")
accent_model = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_ecapa",
    savedir="pretrained_models/accent-id-commonaccent_ecapa"
)
logger.info("Loading language model...")
lang_model = LangID.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="pretrained_models/lang-id-voxlingua107-ecapa"
)

def probe_video(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream['r_frame_rate'] if video_stream and 'r_frame_rate' in video_stream else None
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg probe failed: {str(e)}")
        return None

def download_and_extract_audio(video_url):
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
            'quiet': True,
            'max_duration': 240,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_path = ydl.prepare_filename(info)
        
        fps = probe_video(video_path)
        logger.info(f"Video FPS: {fps}")
        
        try:
            video_clip = mp.VideoFileClip(video_path)
        except Exception as e:
            if 'video_fps' in str(e):
                video_clip = mp.VideoFileClip(video_path, fps_source='mp4')
            else:
                raise
        
        if video_clip.audio is None:
            raise ValueError("No audio track.")
        audio_path = os.path.join(temp_dir, "audio.wav")
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        return audio_path

def is_speech(audio_path, threshold=0.01):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        return np.mean(rms) > threshold
    except Exception as e:
        logger.error(f"Speech detection failed: {str(e)}")
        return False

def split_audio(audio_path, chunk_length_ms=30000):
    try:
        audio = AudioSegment.from_wav(audio_path)
        chunks = []
        temp_dir = os.path.dirname(audio_path)
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_path = os.path.join(temp_dir, f"chunk_{i//1000}.wav")
            chunk.export(chunk_path, format="wav")
            if is_speech(chunk_path):
                chunks.append(chunk_path)
            else:
                logger.info(f"Chunk {chunk_path} skipped: no speech detected.")
        logger.info(f"Split audio into {len(chunks)} valid chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        return [audio_path]

def detect_language(audio_path):
    try:
        out_prob, score, index, text_lab = lang_model.classify_file(audio_path)
        language = text_lab[0].lower()  # e.g., 'en:english', 'fr:french'
        confidence = float(score) * 100
        logger.info(f"Detected language: {language}, confidence: {confidence:.2f}%")
        return language.startswith('en'), confidence
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return False, 0.0

def analyze_accent(audio_path):
    is_english, lang_confidence = detect_language(audio_path)
    if not is_english:
        raise ValueError(f"Non-English speech detected (confidence: {lang_confidence:.2f}%). Please provide a video with English speech.")

    audio_chunks = split_audio(audio_path)
    if not audio_chunks:
        raise ValueError("No valid speech chunks detected.")

    predictions = []
    
    for chunk_path in audio_chunks:
        try:
            out_prob, score, index, text_lab = accent_model.classify_file(chunk_path)
            predicted_class = text_lab[0].lower()
            confidence = float(score) * 100
            
            accent_map = {
                "england": "British",
                "scotland": "British",
                "wales": "British",
                "us": "American",
                "australia": "Australian",
                "canada": "Canadian",
                "ireland": "Irish",
                "newzealand": "New Zealander",
                "bermuda": "Bermudian",
                "hongkong": "Hong Kong",
                "indian": "Indian",
                "malaysia": "Malaysian",
                "philippines": "Philippine",
                "singapore": "Singaporean",
                "southatlandtic": "South Atlantic",
                "african": "African"
            }
            mapped_accent = accent_map.get(predicted_class, "Other")
            
            is_english_accent = mapped_accent in ["British", "American", "Australian", "Canadian", "Irish", "New Zealander"]
            english_confidence = confidence if is_english_accent else (100 - confidence)
            
            predictions.append({
                "accent": mapped_accent,
                "english_confidence": english_confidence,
                "confidence": confidence,
                "language_code": predicted_class
            })
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_path}: {str(e)}")
            continue
    
    if not predictions:
        raise ValueError("No valid accent predictions from audio chunks.")
    
    accents = [p["accent"] for p in predictions]
    most_common_accent = max(set(accents), key=accents.count)
    english_confidences = [p["english_confidence"] for p in predictions if p["accent"] != "Other"]
    avg_english_confidence = np.mean(english_confidences) if english_confidences else 0.0
    avg_confidence = np.mean([p["confidence"] for p in predictions])
    language_codes = [p["language_code"] for p in predictions]
    
    logger.info(f"Predicted language codes: {language_codes}")
    
    summary = (
        f"The detected accent is {most_common_accent} with a confidence of {avg_confidence:.2f}%. "
        f"The likelihood of an English-native accent (British, American, Australian, Canadian, Irish, New Zealander) is {avg_english_confidence:.2f}%. "
    )
    if most_common_accent == "Other":
        summary += "The speaker's accent is less likely to be a native English accent."
    
    return {
        "accent": most_common_accent,
        "english_confidence": avg_english_confidence,
        "summary": summary,
        "raw_language_codes": language_codes
    }

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_video(video_url: str = Form(...)):
    try:
        logger.info(f"Analyzing video: {video_url}")
        audio_path = download_and_extract_audio(video_url)
        result = analyze_accent(audio_path)
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

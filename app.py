import streamlit as st
import yt_dlp
import moviepy.editor as mp
import os
import tempfile
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio
import librosa
import numpy as np
import logging
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set torchaudio backend
try:
    torchaudio.set_audio_backend("soundfile")
    logger.info("Torchaudio backend set to soundfile.")
except Exception as e:
    logger.warning(f"Failed to set torchaudio backend: {str(e)}")

# Cache model
@st.cache_resource
def load_accent_model():
    logger.info("Loading accent model...")
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="pretrained_models/lang-id-voxlingua107-ecapa"
        )
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# Function to download video and extract audio
def download_and_extract_audio(video_url):
    try:
        logger.info(f"Downloading video: {video_url}")
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio/best[ext=mp4]',
            'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
            'quiet': True,
            'max_duration': 240,  # 4 minutes
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_path = ydl.prepare_filename(info)
            st.write(f"Downloaded video: {info['title']}")
        logger.info("Extracting audio...")
        video_clip = mp.VideoFileClip(video_path)
        if video_clip.audio is None:
            raise ValueError("No audio track.")
        audio_path = os.path.join(temp_dir, "audio.wav")
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        logger.info("Audio extracted successfully.")
        return audio_path, temp_dir
    except Exception as e:
        logger.error(f"Error downloading or extracting audio: {str(e)}")
        st.error(f"Error downloading or extracting audio: {str(e)}")
        return None, None

# Function to split audio into chunks
def split_audio(audio_path, chunk_length_ms=30000):  # 30 seconds
    try:
        audio = AudioSegment.from_wav(audio_path)
        chunks = []
        temp_dir = os.path.dirname(audio_path)
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_path = os.path.join(temp_dir, f"chunk_{i//1000}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        logger.info(f"Split audio into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        return [audio_path]  # Fallback to original audio

# Function to analyze accent
def analyze_accent(audio_path):
    try:
        logger.info("Analyzing accent...")
        model = load_accent_model()
        
        # Split audio into chunks
        audio_chunks = split_audio(audio_path)
        predictions = []
        
        for chunk_path in audio_chunks:
            logger.info(f"Processing chunk: {chunk_path}")
            signal, fs = librosa.load(chunk_path, sr=16000)
            signal_tensor = torch.tensor(signal).float()
            output = model.classify_batch(signal_tensor.unsqueeze(0))
            probabilities = output[0].exp().cpu().numpy()
            predicted_class = output[3][0]
            
            accent_map = {
                "en-US": "American",
                "en-GB": "British",
                "en-AU": "Australian",
                "hi-IN": "Indian",
                "es-ES": "Spanish",
                "fr-FR": "French",
                "en": "British"
            }
            default_accent = "Other"
            predicted_accent = accent_map.get(predicted_class, default_accent)
            confidence = float(probabilities[0][model.hparams.label_encoder.encode_label(predicted_class)] * 100)
            
            is_english_accent = predicted_class in ["en-US", "en-GB", "en-AU", "en"]
            english_confidence = confidence if is_english_accent else (100 - confidence)
            
            predictions.append({
                "accent": predicted_accent,
                "english_confidence": english_confidence,
                "confidence": confidence,
                "language_code": predicted_class
            })
        
        # Aggregate predictions (majority vote for accent, average confidence)
        if not predictions:
            raise ValueError("No valid predictions from audio chunks.")
        
        accents = [p["accent"] for p in predictions]
        most_common_accent = max(set(accents), key=accents.count)
        avg_english_confidence = np.mean([p["english_confidence"] for p in predictions if p["accent"] != "Other"])
        avg_confidence = np.mean([p["confidence"] for p in predictions])
        language_codes = [p["language_code"] for p in predictions]
        
        st.write(f"Raw model outputs (language codes): {language_codes}")
        logger.info(f"Predicted language codes: {language_codes}")
        
        summary = (
            f"The detected accent is {most_common_accent} with a confidence of {avg_confidence:.2f}%. "
            f"The likelihood of an English-native accent (American, British, or Australian) is {avg_english_confidence:.2f}%. "
        )
        if most_common_accent == "Other":
            summary += "The speaker's accent is less likely to be a native English accent."
        
        logger.info(f"Accent analysis complete: {most_common_accent}, {avg_english_confidence:.2f}%")
        return most_common_accent, avg_english_confidence, summary
    except Exception as e:
        logger.error(f"Error analyzing accent: {str(e)}")
        st.error(f"Error analyzing accent: {str(e)}")
        return None, None, None
    finally:
        # Clean up chunks
        for chunk_path in audio_chunks:
            try:
                if os.path.exists(chunk_path) and chunk_path != audio_path:
                    os.remove(chunk_path)
            except:
                pass

# Streamlit UI
st.title("Video Accent Analysis Agent")
st.markdown("Upload a public video URL (e.g., YouTube or direct MP4) to analyze the speaker's accent.")

video_url = st.text_input("Enter video URL:")
if st.button("Analyze"):
    if video_url:
        with st.spinner("Downloading video and extracting audio..."):
            audio_path, temp_dir = download_and_extract_audio(video_url)
        if audio_path:
            with st.spinner("Analyzing accent..."):
                accent, confidence, summary = analyze_accent(audio_path)
                if accent and confidence:
                    st.success("Analysis complete!")
                    st.write(f"**Detected Accent**: {accent}")
                    st.write(f"**English Accent Confidence**: {confidence:.2f}%")
                    st.write(f"**Summary**:")
                    st.markdown(summary)
                    try:
                        os.remove(audio_path)
                        os.remove(os.path.join(temp_dir, "video.mp4"))
                        os.rmdir(temp_dir)
                    except Exception as e:
                        logger.warning(f"Cleanup failed: {str(e)}")
                else:
                    st.error("Failed to analyze the accent. Please try another video.")
    else:
        st.warning("Please enter a valid video URL.")

import streamlit as st
import yt_dlp
import moviepy.editor as mp
import os
import tempfile
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import torch
import librosa
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache model to reduce memory usage
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
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
            'quiet': True,
            'max_duration': 60,  # Limit to 60 seconds
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

# Function to analyze accent
def analyze_accent(audio_path):
    try:
        logger.info("Analyzing accent...")
        model = load_accent_model()
        signal, fs = librosa.load(audio_path, sr=16000)
        signal_tensor = torch.tensor(signal).float()
        output = model.classify_batch(signal_tensor.unsqueeze(0))
        probabilities = output[0].exp().cpu().numpy()
        predicted_class = output[3][0]
        
        st.write(f"Raw model output (language code): {predicted_class}")
        logger.info(f"Predicted language code: {predicted_class}")
        
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
        
        summary = (
            f"The detected accent is {predicted_accent} with a confidence of {confidence:.2f}%. "
            f"The likelihood of an English-native accent (American, British, or Australian) is {english_confidence:.2f}%. "
        )
        if not is_english_accent:
            summary += "The speaker's accent is less likely to be a native English accent."
        
        logger.info(f"Accent analysis complete: {predicted_accent}, {english_confidence:.2f}%")
        return predicted_accent, english_confidence, summary
    except Exception as e:
        logger.error(f"Error analyzing accent: {str(e)}")
        st.error(f"Error analyzing accent: {str(e)}")
        return None, None, None

# Streamlit UI
st.title("Video Accent Analysis Tool")
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
                    except:
                        pass
                else:
                    st.error("Failed to analyze the accent. Please try another video.")
    else:
        st.warning("Please enter a valid video URL.")

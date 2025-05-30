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

# Function to download video and extract audio
def download_and_extract_audio(video_url):
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]',  # Prioritize single MP4 stream with audio
            'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
            'quiet': True,
        }
        
        # Download video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_path = ydl.prepare_filename(info)
            st.write(f"Downloaded video: {info['title']}")
        
        # Load video with moviepy
        video_clip = mp.VideoFileClip(video_path)
        
        # Check if audio track exists
        if video_clip.audio is None:
            raise ValueError("The downloaded video has no audio track.")
        
        # Extract audio
        audio_path = os.path.join(temp_dir, "audio.wav")
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        return audio_path, temp_dir
    except Exception as e:
        st.error(f"Error downloading or extracting audio: {str(e)}")
        return None, None

# Function to analyze accent
def analyze_accent(audio_path):
    try:
        # Load pre-trained accent classification model
        model = EncoderClassifier.from_hparams(
            source="speechbrain/accent-classifier-ecapa",
            savedir="pretrained_models/accent-classifier-ecapa"
        )
        
        # Load audio
        signal, fs = librosa.load(audio_path, sr=16000)
        
        # Convert to tensor
        signal_tensor = torch.tensor(signal).float()
        
        # Analyze accent
        output = model.classify_batch(signal_tensor.unsqueeze(0))
        probabilities = output[0].exp().cpu().numpy()  # Get probabilities
        predicted_class = output[3].item()  # Get predicted class index
        labels = model.hparams.label_encoder._lb.classes_  # Get class labels
        
        # Map labels to readable accent names
        accent_map = {
            0: "American",
            1: "British",
            2: "Australian",
            3: "Indian",
            4: "Other"
        }
        predicted_accent = accent_map.get(predicted_class, "Unknown")
        confidence = float(probabilities[0][predicted_class] * 100)
        
        # Check if English accent (non-Indian, non-Other)
        is_english_accent = predicted_class in [0, 1, 2]
        english_confidence = confidence if is_english_accent else (100 - confidence)
        
        # Generate summary
        summary = (
            f"The detected accent is {predicted_accent} with a confidence of {confidence:.2f}%. "
            f"The likelihood of an English-native accent (American, British, or Australian) is {english_confidence:.2f}%. "
        )
        if not is_english_accent:
            summary += "The speaker's accent is less likely to be a native English accent."
        
        return predicted_accent, english_confidence, summary
    except Exception as e:
        st.error(f"Error analyzing accent: {str(e)}")
        return None, None, None

# Streamlit UI
st.title("Video Accent Analysis Tool")
st.markdown("Upload a public video URL (e.g., YouTube or direct MP4) to analyze the speaker's accent.")

# Input field for video URL
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
                    st.write("**Summary**:")
                    st.markdown(summary)
                    
                    # Clean up temporary files
                    try:
                        os.remove(audio_path)
                        os.remove(video_path)
                        os.rmdir(temp_dir)
                    except:
                        pass
                else:
                    st.error("Failed to analyze the accent. Please try another video.")
    else:
        st.warning("Please enter a valid video URL.")

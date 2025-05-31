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
import ffmpeg

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set torchaudio backend
try:
    torchaudio.set_audio_backend("soundfile")
    logger.info("Torchaudio backend set to soundfile.")
except Exception as e:
    logger.warning(f"Failed to set torchaudio backend: {str(e)}")

# Cache models
@st.cache_resource
def load_language_model():
    logger.info("Loading language identification model...")
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="pretrained_models/lang-id-voxlingua107-ecapa"
        )
        logger.info("Language model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load language model: {str(e)}")
        raise

@st.cache_resource
def load_accent_model():
    logger.info("Loading accent model...")
    try:
        model = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir="pretrained_models/accent-id-commonaccent_ecapa"
        )
        logger.info("Accent model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load accent model: {str(e)}")
        raise

# Function to probe video metadata
def probe_video(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream['r_frame_rate'] if video_stream and 'r_frame_rate' in video_stream else None
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg probe failed: {str(e)}")
        return None

# Function to download video and extract audio
def download_and_extract_audio(video_url):
    try:
        logger.info(f"Downloading video: {video_url}")
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
            'quiet': True,
            'max_duration': 240,  # 4 minutes
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_path = ydl.prepare_filename(info)
        
        logger.info("Probing video metadata...")
        fps = probe_video(video_path)
        logger.info(f"Video FPS: {fps}")
        
        logger.info("Extracting audio...")
        try:
            video_clip = mp.VideoFileClip(video_path)
        except Exception as e:
            logger.error(f"MoviePy failed: {str(e)}")
            if 'video_fps' in str(e):
                video_clip = mp.VideoFileClip(video_path, fps_source='mp4')
            else:
                raise
        
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
def split_audio(audio_path, chunk_length_ms=30000):
    try:
        audio_chunk = AudioSegment.from_wav(audio_path)
        chunks = []
        temp_dir = os.path.dirname(audio_path)
        for i in range(0, len(audio_chunk), chunk_length_ms):
            chunk = audio_chunk[i:i + chunk_length_ms]
            chunk_path = os.path.join(temp_dir, f"chunk_{i//1000}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        logger.info(f"Split audio into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        return [audio_path]

# Fixed language detection function
def detect_language_with_softmax(audio_path):
    try:
        logger.info("Starting language detection...")
        lang_model = load_language_model()
        
        audio_chunks = split_audio(audio_path, chunk_length_ms=15000)
        language_predictions = []
        
        for chunk_path in audio_chunks:
            logger.info(f"Processing chunk for language: {chunk_path}")
            try:
                out_prob, score, index, text_lab = lang_model.classify_file(chunk_path)
                detected_language = text_lab[0].lower()
                
                # Use softmax to convert logits to probabilities
                import torch.nn.functional as F
                if torch.is_tensor(out_prob):
                    probs = F.softmax(out_prob, dim=-1)
                    confidence = float(probs[0][index[0]]) * 100
                else:
                    # Fallback if not a tensor
                    confidence = float(score) * 100 if float(score) > 0 else abs(float(score)) * 100
                
                language_predictions.append({
                    "language": detected_language,
                    "confidence": confidence
                })
            except Exception as e:
                logger.error(f"Error processing language chunk {chunk_path}: {str(e)}")
                continue
        
        # Rest of the function remains the same...
        if not language_predictions:
            raise ValueError("No valid language predictions from audio chunks.")
        
        languages = [p["language"] for p in language_predictions]
        most_common_language = max(set(languages), key=languages.count)
        avg_confidence = np.mean([p["confidence"] for p in language_predictions])
        
        english_chunks = [p for p in language_predictions if p["language"] == "en"]
        english_ratio = len(english_chunks) / len(language_predictions)
        
        logger.info(f"Language detection complete: {most_common_language}, confidence: {avg_confidence:.2f}%")
        logger.info(f"English ratio: {english_ratio:.2f}")
        
        # Clean up
        try:
            for chunk_path in audio_chunks:
                if os.path.exists(chunk_path) and chunk_path != audio_path:
                    os.remove(chunk_path)
        except Exception as e:
            logger.warning(f"Language chunk cleanup failed: {str(e)}")
        
        return most_common_language, avg_confidence, english_ratio, language_predictions
        
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return None, None, None, None

#Accent analysis function
def analyze_accent(audio_path):
    try:
        logger.info("Starting accent analysis...")
        model = load_accent_model()
        
        audio_chunks = split_audio(audio_path)
        predictions = []
        
        for chunk_path in audio_chunks:
            logger.info(f"Processing chunk: {chunk_path}")
            try:
                out_prob, score, index, text_lab = model.classify_file(chunk_path)
                predicted_class = text_lab[0].lower()                
                # Fix: Get actual probability instead of raw score
                if hasattr(out_prob, 'numpy'):
                    prob_array = out_prob.numpy()
                else:
                    prob_array = out_prob
                
                confidence = float(prob_array[index[0]]) * 100
                
                # Map model labels to desired accents
                accent_map = {
                    "england": "British",
                    "scotland": "British", 
                    "wales": "British",
                    "us": "American",
                    "australia": "Australian",
                    "canada": "Canadian",
                    "ireland": "Irish",
                    "newzealand": "New Zealand",
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
                
                # Calculate English accent confidence using model probabilities
                english_speaking_regions = [
                    "england", "scotland", "wales", "us", "australia", 
                    "canada", "ireland", "newzealand"
                ]
                
                # Get probability distribution and sum probabilities for English-speaking regions
                english_accent_probability = 0.0
                try:
                    # Sum probabilities for all English-speaking regions
                    for i in range(len(text_lab)):
                        if text_lab[i].lower() in english_speaking_regions:
                            english_accent_probability += float(prob_array[i])
                    
                    english_accent_probability *= 100  # Convert to percentage
                    
                except Exception as prob_error:
                    logger.warning(f"Could not extract probabilities: {prob_error}")
                    # Fallback to original logic if probability extraction fails
                    is_english_accent = mapped_accent in ["British", "American", "Australian", "Canadian", "Irish", "New Zealand"]
                    english_accent_probability = confidence if is_english_accent else (100 - confidence)
                
                predictions.append({
                    "accent": mapped_accent,
                    "english_confidence": english_accent_probability,
                    "confidence": confidence,
                    "language_code": predicted_class
                })
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_path}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from audio chunks.")
        
        accents = [p["accent"] for p in predictions]
        most_common_accent = max(set(accents), key=accents.count)
        
        english_confidences = [p["english_confidence"] for p in predictions]
        avg_english_confidence = np.mean(english_confidences) if english_confidences else 0.0
        avg_confidence = np.mean([p["confidence"] for p in predictions])
        language_codes = [p["language_code"] for p in predictions]
        
        logger.info(f"Predicted language codes: {language_codes}")
        
        summary = (
            f"The detected accent is {most_common_accent} with a confidence of {avg_confidence:.2f}%. "
            f"The likelihood of an English-native accent (British, American, Australian, Irish, Canadian, New Zealand) is {avg_english_confidence:.2f}%. "
        )
        if avg_english_confidence < 50:
            summary += "The speaker's accent is less likely to be a native English accent."
        
        logger.info(f"Accent analysis complete: {most_common_accent}, {avg_english_confidence:.2f}%")
        return most_common_accent, avg_english_confidence, summary
    except Exception as e:
        logger.error(f"Accent analysis failed: {str(e)}")
        st.error(f"Accent analysis failed: {str(e)}")
        return None, None, None
    finally:
        try:
            for chunk_path in audio_chunks:
                if os.path.exists(chunk_path) and chunk_path != audio_path:
                    os.remove(chunk_path)
        except Exception as e:
            logger.warning(f"Chunk cleanup failed: {str(e)}")
            
# Main analysis function
def analyze_video_language_and_accent(audio_path):
    # First, detect language
    language, lang_confidence, english_ratio, lang_predictions = detect_language(audio_path)
    
    if not language:
        return None, None, "Language detection failed."
    
    # Language mapping for display
    language_names = {
        "en": "English", "fr": "French", "es": "Spanish", "de": "German", 
        "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
        "th": "Thai", "vi": "Vietnamese", "tr": "Turkish", "pl": "Polish",
        "nl": "Dutch", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
        "fi": "Finnish", "he": "Hebrew", "cs": "Czech", "hu": "Hungarian"
    }
    
    language_display = language_names.get(language, language.capitalize())
    
    # Check if primarily English
    if language == "en" and english_ratio >= 0.6:  # At least 60% of chunks detected as English
        logger.info("English detected, proceeding with accent analysis...")
        accent, accent_confidence, accent_summary = analyze_accent(audio_path)
        
        if accent and accent_confidence is not None:
            return {
                "type": "english_with_accent",
                "language": language_display,
                "language_confidence": lang_confidence,
                "accent": accent,
                "accent_confidence": accent_confidence,
                "summary": accent_summary
            }
        else:
            return {
                "type": "english_only",
                "language": language_display,
                "language_confidence": lang_confidence,
                "summary": f"English detected with {lang_confidence:.2f}% confidence, but accent analysis failed."
            }
    
    # Check for mixed languages (some English detected)
    elif english_ratio > 0.2:  # Some English detected (>20% of chunks)
        logger.info("Mixed language detected, performing accent analysis on English portions...")
        accent, accent_confidence, accent_summary = analyze_accent(audio_path)
        
        return {
            "type": "mixed_language",
            "primary_language": language_display,
            "language_confidence": lang_confidence,
            "english_ratio": english_ratio,
            "accent": accent if accent else "Unable to determine",
            "accent_confidence": accent_confidence if accent_confidence else 0,
            "summary": f"Mixed language detected: Primarily {language_display} ({lang_confidence:.1f}% confidence) with {english_ratio*100:.1f}% English content. " +
                      (accent_summary if accent else "English accent could not be determined.")
        }
    
    # Non-English language
    else:
        return {
            "type": "non_english",
            "language": language_display,
            "language_confidence": lang_confidence,
            "summary": f"Non-English language detected: {language_display} with {lang_confidence:.2f}% confidence. No accent analysis performed as this is not primarily English speech."
        }

# Streamlit UI
st.title("Video Language & Accent Analysis Agent")
st.markdown("Upload a public video URL (e.g., YouTube or direct MP4) to first detect the language, then analyze English accents if applicable.")

video_url = st.text_input("Enter video URL:")
if st.button("Analyze"):
    if video_url:
        with st.spinner("Downloading video and extracting audio..."):
            audio_path, temp_dir = download_and_extract_audio(video_url)
        
        if audio_path:
            with st.spinner("Analyzing language and accent..."):
                result = analyze_video_language_and_accent(audio_path)
                
                if result:
                    st.success("Analysis complete!")
                    
                    if result["type"] == "english_with_accent":
                        st.write(f"**Language Detected**: {result['language']} ({result['language_confidence']:.1f}% confidence)")
                        st.write(f"**Detected Accent**: {result['accent']}")
                        st.write(f"**English Accent Confidence**: {result['accent_confidence']:.2f}%")
                        st.write(f"**Summary**: {result['summary']}")
                        
                    elif result["type"] == "mixed_language":
                        st.write(f"**Primary Language**: {result['primary_language']} ({result['language_confidence']:.1f}% confidence)")
                        st.write(f"**English Content**: {result['english_ratio']*100:.1f}% of audio")
                        if result['accent'] != "Unable to determine":
                            st.write(f"**English Accent Detected**: {result['accent']}")
                            st.write(f"**English Accent Confidence**: {result['accent_confidence']:.2f}%")
                        st.write(f"**Summary**: {result['summary']}")
                        
                    elif result["type"] == "non_english":
                        st.warning("Non-English Language Detected")
                        st.write(f"**Language Detected**: {result['language']} ({result['language_confidence']:.1f}% confidence)")
                        st.write(f"**Summary**: {result['summary']}")
                        
                    else:  # english_only with failed accent analysis
                        st.write(f"**Language Detected**: {result['language']} ({result['language_confidence']:.1f}% confidence)")
                        st.write(f"**Summary**: {result['summary']}")
                    
                    # Cleanup
                    try:
                        os.remove(audio_path)
                        video_file = os.path.join(temp_dir, "video.mp4")
                        if os.path.exists(video_file):
                            os.remove(video_file)
                        os.rmdir(temp_dir)
                    except Exception as e:
                        logger.warning(f"Cleanup failed: {str(e)}")
                else:
                    st.error("Failed to analyze the audio. Please try another video.")
    else:
        st.warning("Please enter a valid video URL.")

import gradio as gr
from TTS.api import TTS
import tempfile
import os
from pydub import AudioSegment
import torch

# Load XTTS-v2 model once (CPU mode - works on Railway free tier)
print("Loading XTTS-v2 model... (this may take 20-60 seconds first time)")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
print("Model loaded successfully!")

def clone_and_generate(reference_audio, text, language="en"):
    if not reference_audio or not text.strip():
        return None, "❌ Please upload your voice audio and enter text."
    
    # Create temp output files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    mp3_path = wav_path.replace(".wav", ".mp3")
    
    try:
        # Generate speech using your reference voice
        tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language=language,
            file_path=wav_path,
            split_sentences=True
        )
        
        # Convert WAV to MP3
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="192k")
        
        return mp3_path, "✅ Voice cloned successfully! Download your MP3 below."
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="My Voice Cloner - TTS Tool", theme=gr.themes.Dark()) as demo:
    gr.Markdown("# 🎤 My Personal Voice Cloner\nUpload your voice • Type text • Get MP3 in your voice")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Upload your voice reference (6-30 seconds recommended)")
            audio_input = gr.Audio(type="filepath", label="Your Voice Audio", sources=["upload"])
            
            text_input = gr.Textbox(
                label="2. Enter the text you want spoken",
                placeholder="Hello, this is my cloned voice speaking...",
                lines=4
            )
            
            lang_input = gr.Dropdown(
                choices=[
                    ("English", "en"), ("Urdu", "ur"), ("Hindi", "hi"), ("Arabic", "ar"),
                    ("Spanish", "es"), ("French", "fr"), ("German", "de"), ("Italian", "it"),
                    ("Portuguese", "pt"), ("Chinese", "zh-cn")
                ],
                value="en",
                label="3. Language"
            )
            
            generate_btn = gr.Button("🚀 Generate Speech in My Voice", variant="primary", size="large")
        
        with gr.Column():
            gr.Markdown("### Result")
            output_audio = gr.Audio(label="Preview", type="filepath")
            status = gr.Markdown()
            download_file = gr.File(label="Download MP3", visible=True)

    generate_btn.click(
        fn=clone_and_generate,
        inputs=[audio_input, text_input, lang_input],
        outputs=[download_file, status]
    )

    gr.Markdown("### Tips\n• Best results with clear, quiet 10-20 second voice sample\n• First generation may take 30-60 seconds (model loads)\n• Deployed on Railway • 100% free & private")

demo.launch()

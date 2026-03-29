import gradio as gr
import torch
from TTS.api import TTS
import tempfile
import os
from pydub import AudioSegment

# Load the model (CPU mode - safe for Railway free tier)
print("Loading XTTS-v2 model... This may take 30-90 seconds on first run.")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
print(f"✅ Model loaded successfully on {device.upper()}!")

def generate_speech(reference_audio, text, language="en"):
    if not reference_audio:
        return None, "❌ Please upload a reference audio of your voice."
    if not text or not text.strip():
        return None, "❌ Please enter some text to speak."

    try:
        # Temporary files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        mp3_path = wav_path.replace(".wav", ".mp3")

        # Generate speech with voice cloning
        tts.tts_to_file(
            text=text.strip(),
            speaker_wav=reference_audio,
            language=language,
            file_path=wav_path,
            split_sentences=True
        )

        # Convert to MP3 (better quality & smaller size)
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="192k")

        # Clean up wav
        if os.path.exists(wav_path):
            os.unlink(wav_path)

        return mp3_path, "✅ Success! Your voice has been cloned. Download the MP3 below."

    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}"

# ====================== Gradio UI ======================
with gr.Blocks(title="My Voice Cloner", theme=gr.themes.Dark()) as demo:
    gr.Markdown("# 🎙️ My Personal Voice Cloner\nUpload **your voice** → Type text → Get MP3 in **your exact voice**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Reference Audio (your voice)")
            audio_input = gr.Audio(
                type="filepath",
                label="Reference Voice (6-30 seconds best)",
                sources=["upload", "microphone"]
            )

            text_input = gr.Textbox(
                label="2. Text to Speak",
                placeholder="Hello everyone, this is my cloned voice speaking right now...",
                lines=5
            )

            lang_input = gr.Dropdown(
                label="3. Language",
                choices=[
                    ("English", "en"), ("Urdu", "ur"), ("Hindi", "hi"), ("Arabic", "ar"),
                    ("Spanish", "es"), ("French", "fr"), ("German", "de")
                ],
                value="en"
            )

            generate_btn = gr.Button("🚀 Generate in My Voice", variant="primary", size="large")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_audio = gr.Audio(label="Preview", type="filepath")
            status_text = gr.Markdown()
            download_btn = gr.File(label="⬇️ Download MP3")

    generate_btn.click(
        fn=generate_speech,
        inputs=[audio_input, text_input, lang_input],
        outputs=[download_btn, status_text]
    )

    gr.Markdown("""
    **Tips:**
    - Use a clear, quiet recording of your voice (10-20 seconds is ideal)
    - First generation is slow (model loading)
    - Works best with natural speaking voice
    """)

# Launch for Railway
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)

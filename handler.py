import runpod
import torch
import torchaudio
import base64
import io
import os

os.environ["NO_TORCH_COMPILE"] = "1"

# Login to HF at startup (for model downloads)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)

from generator import load_csm_1b

# Load model once at cold start
print("Loading CSM-1B model...")
generator = load_csm_1b(device="cuda")
print("Model loaded successfully.")


def handler(job):
    input_data = job["input"]

    text = input_data.get("text", "Hello from Sesame.")
    speaker = input_data.get("speaker", 0)
    max_audio_length_ms = input_data.get("max_audio_length_ms", 10000)

    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=[],
        max_audio_length_ms=max_audio_length_ms,
    )

    buffer = io.BytesIO()
    torchaudio.save(
        buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav"
    )
    buffer.seek(0)
    audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "audio_base64": audio_b64,
        "sample_rate": generator.sample_rate,
        "format": "wav",
    }


runpod.serverless.start({"handler": handler})

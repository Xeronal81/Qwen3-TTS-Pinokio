# coding=utf-8
import os
import sys
import warnings
import json
from datetime import datetime

# Suppress common warnings
warnings.filterwarnings("ignore", message=".*Min value of input waveform.*")
warnings.filterwarnings("ignore", message=".*Max value of input waveform.*")
warnings.filterwarnings("ignore", message=".*Trying to convert audio automatically.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

# Only attempt flash-attn install on Linux (HF Spaces)
if sys.platform == "linux":
    import subprocess
    subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, scan_cache_dir

# Voice files directory
VOICE_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_voices")
os.makedirs(VOICE_FILES_DIR, exist_ok=True)

# Whisper model for transcription
whisper_model = None


def get_device():
    """Get the best available device for this platform."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_whisper_model():
    """Load Whisper tiny model for transcription."""
    global whisper_model
    if whisper_model is None:
        import whisper
        device = get_device()
        whisper_model = whisper.load_model("tiny", device=device)
    return whisper_model


def unload_whisper():
    """Force unload whisper model from GPU."""
    global whisper_model
    if whisper_model is not None:
        # Move to CPU first, then delete
        try:
            whisper_model.cpu()
        except:
            pass
        whisper_model = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def transcribe_audio(audio):
    """Transcribe audio using Whisper tiny."""
    global whisper_model
    if audio is None:
        return "Please upload audio first."
    
    try:
        sr, wav = audio
        # Convert to float32 and normalize properly
        wav = wav.astype(np.float32)
        
        # Check if audio needs normalization (int16 range is -32768 to 32767)
        max_val = np.abs(wav).max()
        if max_val > 1.0:
            wav = wav / max_val  # Normalize to [-1, 1] range
        
        # Whisper expects 16kHz mono
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        
        model = get_whisper_model()
        result = model.transcribe(wav, fp16=torch.cuda.is_available())
        text = result["text"].strip()
        
        # Unload whisper to free GPU memory
        unload_whisper()
        
        return text
    except Exception as e:
        # Still try to unload on error
        unload_whisper()
        return f"Transcription error: {str(e)}"

# Global model holders - keyed by (model_type, model_size)
loaded_models = {}

# Global LLM model holder
llm_model = None
llm_tokenizer = None

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]

# Available models configuration
AVAILABLE_MODELS = {
    "VoiceDesign": {
        "sizes": ["1.7B"],
        "description": "Create custom voices using natural language descriptions"
    },
    "Base": {
        "sizes": ["0.6B", "1.7B"],
        "description": "Voice cloning from reference audio"
    },
    "CustomVoice": {
        "sizes": ["0.6B", "1.7B"],
        "description": "TTS with predefined speakers and style instructions"
    }
}

# Available LLM models for conversation generation
AVAILABLE_LLMS = {
    "Qwen3-4B-Instruct": {
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "description": "Qwen3 4B - Great for conversations, matches TTS family"
    }
}


def get_model_repo_id(model_type: str, model_size: str) -> str:
    """Get HuggingFace repo ID for a model."""
    return f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"


def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return snapshot_download(get_model_repo_id(model_type, model_size))


def check_model_downloaded(model_type: str, model_size: str) -> bool:
    """Check if a model is already downloaded in the cache."""
    try:
        cache_info = scan_cache_dir()
        repo_id = get_model_repo_id(model_type, model_size)
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
        return False
    except Exception:
        return False


def get_downloaded_models_status() -> str:
    """Get status of all available models."""
    lines = ["### Model Download Status\n"]
    for model_type, info in AVAILABLE_MODELS.items():
        lines.append(f"**{model_type}** - {info['description']}")
        for size in info["sizes"]:
            status = "✅ Downloaded" if check_model_downloaded(model_type, size) else "⬜ Not downloaded"
            lines.append(f"  - {size}: {status}")
        lines.append("")
    return "\n".join(lines)


def download_model(model_type: str, model_size: str, progress=gr.Progress()):
    """Download a specific model."""
    if model_size not in AVAILABLE_MODELS.get(model_type, {}).get("sizes", []):
        return f"❌ Invalid combination: {model_type} {model_size}", get_downloaded_models_status()
    
    repo_id = get_model_repo_id(model_type, model_size)
    
    if check_model_downloaded(model_type, model_size):
        return f"✅ {model_type} {model_size} is already downloaded!", get_downloaded_models_status()
    
    try:
        progress(0, desc=f"Downloading {model_type} {model_size}...")
        snapshot_download(repo_id)
        progress(1, desc="Complete!")
        return f"✅ Successfully downloaded {model_type} {model_size}!", get_downloaded_models_status()
    except Exception as e:
        return f"❌ Error downloading {model_type} {model_size}: {str(e)}", get_downloaded_models_status()


def get_available_sizes(model_type: str):
    """Get available sizes for a model type."""
    return gr.update(choices=AVAILABLE_MODELS.get(model_type, {}).get("sizes", []), value=AVAILABLE_MODELS.get(model_type, {}).get("sizes", ["1.7B"])[0])


def get_model(model_type: str, model_size: str):
    """Get or load a model by type and size."""
    global loaded_models
    key = (model_type, model_size)
    if key not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        model_path = get_model_path(model_type, model_size)
        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.bfloat16,
#           attn_implementation="flash_attention_2",
        )
    return loaded_models[key]


def get_loaded_models_status() -> str:
    """Get status of currently loaded models in memory."""
    if not loaded_models:
        return "No models currently loaded in memory."
    
    lines = ["**Currently loaded models:**"]
    for (model_type, model_size) in loaded_models.keys():
        lines.append(f"- {model_type} ({model_size})")
    return "\n".join(lines)


def load_model_manual(model_type: str, model_size: str, progress=gr.Progress()):
    """Manually load a model into memory."""
    if model_size not in AVAILABLE_MODELS.get(model_type, {}).get("sizes", []):
        return f"❌ Invalid combination: {model_type} {model_size}", get_loaded_models_status()
    
    key = (model_type, model_size)
    if key in loaded_models:
        return f"✅ {model_type} {model_size} is already loaded!", get_loaded_models_status()
    
    try:
        progress(0, desc=f"Loading {model_type} {model_size}...")
        get_model(model_type, model_size)
        progress(1, desc="Complete!")
        return f"✅ Successfully loaded {model_type} {model_size}!", get_loaded_models_status()
    except Exception as e:
        return f"❌ Error loading {model_type} {model_size}: {str(e)}", get_loaded_models_status()


def unload_model(model_type: str, model_size: str):
    """Unload a specific model from memory."""
    global loaded_models
    key = (model_type, model_size)
    
    if key not in loaded_models:
        return f"⚠️ {model_type} {model_size} is not loaded.", get_loaded_models_status()
    
    try:
        del loaded_models[key]
        torch.cuda.empty_cache()
        return f"✅ Unloaded {model_type} {model_size} and freed GPU memory.", get_loaded_models_status()
    except Exception as e:
        return f"❌ Error unloading: {str(e)}", get_loaded_models_status()


def unload_all_models():
    """Unload all models from memory."""
    global loaded_models

    if not loaded_models:
        return "⚠️ No models are currently loaded.", get_loaded_models_status()

    try:
        count = len(loaded_models)
        loaded_models.clear()
        torch.cuda.empty_cache()
        return f"✅ Unloaded {count} model(s) and freed GPU memory.", get_loaded_models_status()
    except Exception as e:
        return f"❌ Error unloading: {str(e)}", get_loaded_models_status()


# ============================================
# LLM Model Functions (for Conversation Mode)
# ============================================

def check_llm_downloaded(llm_name: str) -> bool:
    """Check if an LLM is already downloaded in the cache."""
    try:
        cache_info = scan_cache_dir()
        repo_id = AVAILABLE_LLMS.get(llm_name, {}).get("repo_id", "")
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
        return False
    except Exception:
        return False


def get_llm_download_status() -> str:
    """Get status of LLM models."""
    lines = ["### LLM Models (for Conversation Mode)\n"]
    for llm_name, info in AVAILABLE_LLMS.items():
        status = "✅ Downloaded" if check_llm_downloaded(llm_name) else "⬜ Not downloaded"
        lines.append(f"**{llm_name}** - {info['description']}")
        lines.append(f"  - Status: {status}")
        lines.append("")
    return "\n".join(lines)


def download_llm(llm_name: str, progress=gr.Progress()):
    """Download an LLM model."""
    if llm_name not in AVAILABLE_LLMS:
        return f"❌ Unknown LLM: {llm_name}", get_llm_download_status()

    repo_id = AVAILABLE_LLMS[llm_name]["repo_id"]

    if check_llm_downloaded(llm_name):
        return f"✅ {llm_name} is already downloaded!", get_llm_download_status()

    try:
        progress(0, desc=f"Downloading {llm_name}...")
        snapshot_download(repo_id)
        progress(1, desc="Complete!")
        return f"✅ Successfully downloaded {llm_name}!", get_llm_download_status()
    except Exception as e:
        return f"❌ Error downloading {llm_name}: {str(e)}", get_llm_download_status()


def get_llm_loaded_status() -> str:
    """Get status of loaded LLM."""
    global llm_model
    if llm_model is None:
        return "No LLM currently loaded."

    # Get device info
    try:
        device = next(llm_model.parameters()).device
        device_str = str(device)
        if "cuda" in device_str:
            device_info = f"🟢 GPU ({device_str})"
        elif "mps" in device_str:
            device_info = f"🟢 Apple Silicon ({device_str})"
        else:
            device_info = f"🟡 CPU ({device_str})"
    except:
        device_info = "Unknown"

    return f"**Loaded LLM:** Qwen3-4B-Instruct\n**Device:** {device_info}"


def load_llm(llm_name: str, progress=gr.Progress()):
    """Load an LLM into memory."""
    global llm_model, llm_tokenizer

    if llm_name not in AVAILABLE_LLMS:
        return f"❌ Unknown LLM: {llm_name}", get_llm_loaded_status()

    if llm_model is not None:
        return f"✅ LLM is already loaded!", get_llm_loaded_status()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo_id = AVAILABLE_LLMS[llm_name]["repo_id"]
        progress(0, desc=f"Loading {llm_name}...")

        print(f"\n{'='*50}")
        print(f"🤖 Loading LLM: {llm_name}")
        print(f"{'='*50}")

        # Determine device
        if torch.cuda.is_available():
            device_map = "cuda"
            print(f"🎮 Using CUDA GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
            print(f"🍎 Using Apple Silicon MPS")
        else:
            device_map = "cpu"
            print(f"💻 Using CPU (no GPU available)")

        llm_tokenizer = AutoTokenizer.from_pretrained(repo_id)
        llm_model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16 if device_map != "cpu" else torch.float32,
            device_map=device_map if device_map != "cpu" else None,
        )

        # Move to device if CPU (device_map doesn't work for CPU)
        if device_map == "cpu":
            llm_model = llm_model.to("cpu")

        progress(1, desc="Complete!")
        print(f"✅ LLM loaded successfully on {device_map}!")
        print(f"{'='*50}\n")

        return f"✅ Successfully loaded {llm_name} on {device_map.upper()}!", get_llm_loaded_status()
    except Exception as e:
        print(f"❌ Error loading LLM: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error loading {llm_name}: {str(e)}", get_llm_loaded_status()


def unload_llm():
    """Unload the LLM from memory."""
    global llm_model, llm_tokenizer

    if llm_model is None:
        return "⚠️ No LLM is currently loaded.", get_llm_loaded_status()

    try:
        del llm_model
        del llm_tokenizer
        llm_model = None
        llm_tokenizer = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return "✅ Unloaded LLM and freed GPU memory.", get_llm_loaded_status()
    except Exception as e:
        return f"❌ Error unloading: {str(e)}", get_llm_loaded_status()


def generate_llm_response(messages: list, max_tokens: int = 150) -> str:
    """Generate a response from the LLM."""
    global llm_model, llm_tokenizer

    if llm_model is None or llm_tokenizer is None:
        raise RuntimeError("LLM not loaded. Please load the LLM first.")

    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking for faster responses
    )

    inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


import re

def chunk_text(text: str, max_chars: int = 200) -> list:
    """
    Split text into chunks without cutting words.
    Tries to split on sentence boundaries first, then falls back to word boundaries.
    """
    text = text.strip()
    if not text:
        return []
    
    if len(text) <= max_chars:
        return [text]
    
    # Sentence-ending punctuation patterns
    sentence_endings = re.compile(r'(?<=[.!?。！？])\s+')
    
    # Split into sentences first
    sentences = sentence_endings.split(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If single sentence is too long, split by words
        if len(sentence) > max_chars:
            # Flush current chunk first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by words
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk = current_chunk + " " + word if current_chunk else word
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
        else:
            # Try to add sentence to current chunk
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Italian", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_voice_design(text, language, voice_description, seed):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        # Handle seed - if -1 (auto), generate one
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        set_seed(seed)
        
        tts = get_model("VoiceDesign", "1.7B")
        
        print(f"\n{'='*50}")
        print(f"🎨 Voice Design Generation")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"📝 Text length: {len(text)} chars")
        
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        
        total_duration = len(wavs[0]) / sr
        print(f"\n{'='*50}")
        print(f"✅ Complete! Duration: {total_duration:.2f}s")
        print(f"{'='*50}\n")
        
        status = f"Generated {total_duration:.1f}s of audio | Seed: {seed}"
        return (sr, wavs[0]), status
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return None, f"Error: {type(e).__name__}: {e}"


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size, max_chunk_chars, chunk_gap, seed):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

    try:
        from tqdm import tqdm
        
        # Handle seed - if -1 (auto), generate one and use it for all chunks
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        
        tts = get_model("Base", model_size)
        chunks = chunk_text(target_text.strip(), max_chars=int(max_chunk_chars))
        
        print(f"\n{'='*50}")
        print(f"🎭 Voice Clone Generation ({model_size})")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"📝 Text length: {len(target_text)} chars → {len(chunks)} chunk(s)")
        print(f"⏱️ Chunk gap: {chunk_gap}s")
        
        all_wavs = []
        sr = None
        for i, chunk in enumerate(tqdm(chunks, desc="Generating chunks", unit="chunk")):
            # Set seed before each chunk to ensure consistency
            set_seed(seed)
            
            print(f"\n🔊 Chunk {i+1}/{len(chunks)} [Seed: {seed}]: \"{chunk[:50]}{'...' if len(chunk) > 50 else ''}\"")
            wavs, sr = tts.generate_voice_clone(
                text=chunk,
                language=language,
                ref_audio=audio_tuple,
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
            all_wavs.append(wavs[0])
            print(f"   ✅ Generated {len(wavs[0])/sr:.2f}s of audio")
        
        # Concatenate all audio chunks with gap (silence) between them
        if len(all_wavs) > 1 and chunk_gap > 0:
            gap_samples = int(sr * chunk_gap)
            silence = np.zeros(gap_samples, dtype=np.float32)
            chunks_with_gaps = []
            for i, wav in enumerate(all_wavs):
                chunks_with_gaps.append(wav)
                if i < len(all_wavs) - 1:  # Don't add gap after last chunk
                    chunks_with_gaps.append(silence)
            final_wav = np.concatenate(chunks_with_gaps)
        else:
            final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
        
        total_duration = len(final_wav) / sr
        print(f"\n{'='*50}")
        print(f"✅ Complete! Total duration: {total_duration:.2f}s")
        print(f"{'='*50}\n")
        
        status = f"Generated {len(chunks)} chunk(s), {total_duration:.1f}s total | Seed: {seed}" if len(chunks) > 1 else f"Generated {total_duration:.1f}s of audio | Seed: {seed}"
        return (sr, final_wav), status
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return None, f"Error: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, model_size, seed):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    try:
        # Handle seed - if -1 (auto), generate one
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        set_seed(seed)
        
        tts = get_model("CustomVoice", model_size)
        
        print(f"\n{'='*50}")
        print(f"🗣️ Custom Voice Generation ({model_size})")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"👤 Speaker: {speaker}")
        print(f"📝 Text length: {len(text)} chars")
        
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        
        total_duration = len(wavs[0]) / sr
        print(f"\n{'='*50}")
        print(f"✅ Complete! Duration: {total_duration:.2f}s")
        print(f"{'='*50}\n")
        
        status = f"Generated {total_duration:.1f}s of audio | Seed: {seed}"
        return (sr, wavs[0]), status
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return None, f"Error: {type(e).__name__}: {e}"


# ============================================
# Voice Save/Load Functions
# ============================================

def get_saved_voices_list():
    """Get list of saved voice files."""
    voices = []
    if os.path.exists(VOICE_FILES_DIR):
        for f in os.listdir(VOICE_FILES_DIR):
            if f.endswith('.npz'):
                voices.append(f[:-4])  # Remove .npz extension
    return sorted(voices)


def get_voice_files_info():
    """Get formatted info about saved voice files."""
    voices = get_saved_voices_list()
    if not voices:
        return "No saved voices yet."

    lines = ["**Saved Voices:**"]
    for voice in voices:
        filepath = os.path.join(VOICE_FILES_DIR, f"{voice}.npz")
        try:
            data = np.load(filepath, allow_pickle=True)
            x_vector_only = bool(data.get('x_vector_only_mode', False))
            mode = "x-vector only" if x_vector_only else "full (with text)"
            ref_text = str(data.get('ref_text', ''))[:50]
            if ref_text and len(str(data.get('ref_text', ''))) > 50:
                ref_text += "..."
            lines.append(f"- **{voice}** ({mode})")
            if ref_text and not x_vector_only:
                lines.append(f"  Text: \"{ref_text}\"")
        except Exception as e:
            lines.append(f"- **{voice}** (error reading)")
    return "\n".join(lines)


def save_voice_file(ref_audio, ref_text, use_xvector_only, voice_name, model_size):
    """Save voice embedding from reference audio to a file."""
    if not voice_name or not voice_name.strip():
        return "Error: Please enter a name for the voice file.", get_voice_files_info(), gr.update()

    # Sanitize voice name
    voice_name = voice_name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return "Error: Reference audio is required.", get_voice_files_info(), gr.update()

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return "Error: Reference text is required when 'Use x-vector only' is not enabled.", get_voice_files_info(), gr.update()

    try:
        import librosa

        tts = get_model("Base", model_size)
        wav, sr = audio_tuple

        print(f"\n{'='*50}")
        print(f"💾 Saving Voice: {voice_name}")
        print(f"{'='*50}")
        print(f"🎙️ Audio: {len(wav)/sr:.2f}s at {sr}Hz")
        print(f"📝 X-vector only: {use_xvector_only}")

        # Extract speaker embedding
        wav_resample = wav
        speaker_encoder_sr = tts.model.speaker_encoder_sample_rate
        if sr != speaker_encoder_sr:
            wav_resample = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=speaker_encoder_sr)

        spk_emb = tts.model.extract_speaker_embedding(audio=wav_resample, sr=speaker_encoder_sr)
        # Convert bfloat16 to float32 before numpy (numpy doesn't support bfloat16)
        spk_emb_cpu = spk_emb.cpu()
        if spk_emb_cpu.dtype == torch.bfloat16:
            spk_emb_np = spk_emb_cpu.float().numpy()
        else:
            spk_emb_np = spk_emb_cpu.numpy()

        # Extract speech codes if not x-vector only mode
        ref_code_np = None
        if not use_xvector_only:
            enc = tts.model.speech_tokenizer.encode([wav], sr=sr)
            ref_code = enc.audio_codes[0]
            # Speech codes are typically integers, but handle bfloat16 just in case
            ref_code_cpu = ref_code.cpu()
            if ref_code_cpu.dtype == torch.bfloat16:
                ref_code_np = ref_code_cpu.float().numpy()
            else:
                ref_code_np = ref_code_cpu.numpy()

        # Save to file
        filepath = os.path.join(VOICE_FILES_DIR, f"{voice_name}.npz")
        save_dict = {
            'ref_spk_embedding': spk_emb_np,
            'x_vector_only_mode': use_xvector_only,
            'ref_text': ref_text.strip() if ref_text else "",
            'model_size': model_size,
            'sample_rate': sr,
            'created_at': datetime.now().isoformat(),
        }
        if ref_code_np is not None:
            save_dict['ref_code'] = ref_code_np

        np.savez(filepath, **save_dict)

        print(f"✅ Saved to: {filepath}")
        print(f"{'='*50}\n")

        # Update dropdown choices
        new_choices = get_saved_voices_list()
        return f"✅ Voice saved as '{voice_name}'!", get_voice_files_info(), gr.update(choices=new_choices, value=voice_name)
    except Exception as e:
        print(f"❌ Error saving voice: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"Error saving voice: {type(e).__name__}: {e}", get_voice_files_info(), gr.update()


def load_voice_and_generate(voice_file, target_text, language, model_size, max_chunk_chars, chunk_gap, seed):
    """Load a saved voice file and generate speech."""
    if not voice_file:
        return None, "Error: Please select a voice file."

    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    filepath = os.path.join(VOICE_FILES_DIR, f"{voice_file}.npz")
    if not os.path.exists(filepath):
        return None, f"Error: Voice file '{voice_file}' not found."

    try:
        from tqdm import tqdm

        # Load voice data
        data = np.load(filepath, allow_pickle=True)
        spk_emb_np = data['ref_spk_embedding']
        x_vector_only = bool(data.get('x_vector_only_mode', False))
        ref_text = str(data.get('ref_text', ''))
        ref_code_np = data.get('ref_code', None)

        # Handle seed
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)

        tts = get_model("Base", model_size)

        # Convert back to tensors
        device = tts.device
        spk_emb = torch.from_numpy(spk_emb_np).to(device)
        ref_code = None
        if ref_code_np is not None:
            ref_code = torch.from_numpy(ref_code_np).to(device)

        # Chunk the text
        chunks = chunk_text(target_text.strip(), max_chars=int(max_chunk_chars))

        print(f"\n{'='*50}")
        print(f"🎭 Generate from Saved Voice: {voice_file}")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"📝 Text length: {len(target_text)} chars → {len(chunks)} chunk(s)")
        print(f"📂 X-vector only: {x_vector_only}")

        all_wavs = []
        sr = None

        for i, chunk in enumerate(tqdm(chunks, desc="Generating chunks", unit="chunk")):
            set_seed(seed)

            print(f"\n🔊 Chunk {i+1}/{len(chunks)} [Seed: {seed}]: \"{chunk[:50]}{'...' if len(chunk) > 50 else ''}\"")

            # Build voice clone prompt
            voice_clone_prompt = {
                'ref_code': [ref_code] if not x_vector_only else [None],
                'ref_spk_embedding': [spk_emb],
                'x_vector_only_mode': [x_vector_only],
                'icl_mode': [not x_vector_only],
            }

            # Build input text
            input_text = f"<|im_start|>assistant\n{chunk}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = tts.processor(text=input_text, return_tensors="pt", padding=True)["input_ids"].to(device)
            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

            # Build ref_ids if not x-vector only
            ref_ids = None
            if not x_vector_only and ref_text:
                ref_text_formatted = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
                ref_tok = tts.processor(text=ref_text_formatted, return_tensors="pt", padding=True)["input_ids"].to(device)
                ref_tok = ref_tok.unsqueeze(0) if ref_tok.dim() == 1 else ref_tok
                ref_ids = [ref_tok]

            # Generate
            talker_codes_list, _ = tts.model.generate(
                input_ids=[input_ids],
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt,
                languages=[language],
                non_streaming_mode=True,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                repetition_penalty=1.05,
                max_new_tokens=2048,
            )

            # Decode with ref_code prepended if ICL mode
            codes = talker_codes_list[0]
            if not x_vector_only and ref_code is not None:
                codes_for_decode = torch.cat([ref_code.to(codes.device), codes], dim=0)
            else:
                codes_for_decode = codes

            wavs_all, fs = tts.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])
            sr = fs

            # Cut off the reference portion if ICL mode
            wav = wavs_all[0]
            if not x_vector_only and ref_code is not None:
                ref_len = int(ref_code.shape[0])
                total_len = int(codes_for_decode.shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wav = wav[cut:]

            all_wavs.append(wav)
            print(f"   ✅ Generated {len(wav)/sr:.2f}s of audio")

        # Concatenate chunks with gaps
        if len(all_wavs) > 1 and chunk_gap > 0:
            gap_samples = int(sr * chunk_gap)
            silence = np.zeros(gap_samples, dtype=np.float32)
            chunks_with_gaps = []
            for i, wav in enumerate(all_wavs):
                chunks_with_gaps.append(wav)
                if i < len(all_wavs) - 1:
                    chunks_with_gaps.append(silence)
            final_wav = np.concatenate(chunks_with_gaps)
        else:
            final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]

        total_duration = len(final_wav) / sr
        print(f"\n{'='*50}")
        print(f"✅ Complete! Total duration: {total_duration:.2f}s")
        print(f"{'='*50}\n")

        status = f"Generated {len(chunks)} chunk(s), {total_duration:.1f}s total | Voice: {voice_file} | Seed: {seed}" if len(chunks) > 1 else f"Generated {total_duration:.1f}s | Voice: {voice_file} | Seed: {seed}"
        return (sr, final_wav), status
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {type(e).__name__}: {e}"


def delete_voice_file(voice_file):
    """Delete a saved voice file."""
    if not voice_file:
        return "Error: Please select a voice file to delete.", get_voice_files_info(), gr.update()

    filepath = os.path.join(VOICE_FILES_DIR, f"{voice_file}.npz")
    if not os.path.exists(filepath):
        return f"Error: Voice file '{voice_file}' not found.", get_voice_files_info(), gr.update()

    try:
        os.remove(filepath)
        new_choices = get_saved_voices_list()
        new_value = new_choices[0] if new_choices else None
        return f"✅ Deleted voice file '{voice_file}'.", get_voice_files_info(), gr.update(choices=new_choices, value=new_value)
    except Exception as e:
        return f"Error deleting: {e}", get_voice_files_info(), gr.update()


# ============================================
# Conversation Mode Functions
# ============================================

def generate_conversation_turn(
    conversation_history: list,
    speaker_name: str,
    speaker_personality: str,
    other_speaker_name: str,
    topic: str,
    is_first_turn: bool = False
) -> str:
    """Generate a single conversation turn using the LLM."""

    system_prompt = f"""You are {speaker_name}. {speaker_personality}

You are having a natural conversation with {other_speaker_name} about: {topic}

Rules:
- Respond naturally as {speaker_name} would, staying in character
- Keep responses concise (1-3 sentences)
- Be engaging and conversational
- React to what the other person said
- Don't use quotation marks or speaker labels
- Just respond with your dialogue directly"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for turn in conversation_history:
        if turn["speaker"] == speaker_name:
            messages.append({"role": "assistant", "content": turn["text"]})
        else:
            messages.append({"role": "user", "content": turn["text"]})

    # If first turn, add a prompt to start
    if is_first_turn:
        messages.append({"role": "user", "content": f"Start the conversation about {topic}. Say something to begin."})

    response = generate_llm_response(messages, max_tokens=100)

    # Clean up response
    response = response.strip().strip('"').strip("'")
    # Remove any speaker labels that might have been generated
    if response.startswith(f"{speaker_name}:"):
        response = response[len(f"{speaker_name}:"):].strip()

    return response


def generate_voice_for_text(voice_file: str, text: str, model_size: str = "1.7B") -> tuple:
    """Generate audio for text using a saved voice file."""
    filepath = os.path.join(VOICE_FILES_DIR, f"{voice_file}.npz")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Voice file '{voice_file}' not found.")

    # Load voice data
    data = np.load(filepath, allow_pickle=True)
    spk_emb_np = data['ref_spk_embedding']
    x_vector_only = bool(data.get('x_vector_only_mode', False))
    ref_text = str(data.get('ref_text', ''))
    ref_code_np = data.get('ref_code', None)

    tts = get_model("Base", model_size)
    device = tts.device

    # Convert back to tensors
    spk_emb = torch.from_numpy(spk_emb_np).to(device)
    ref_code = None
    if ref_code_np is not None:
        ref_code = torch.from_numpy(ref_code_np).to(device)

    # Build voice clone prompt
    voice_clone_prompt = {
        'ref_code': [ref_code] if not x_vector_only else [None],
        'ref_spk_embedding': [spk_emb],
        'x_vector_only_mode': [x_vector_only],
        'icl_mode': [not x_vector_only],
    }

    # Build input text
    input_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tts.processor(text=input_text, return_tensors="pt", padding=True)["input_ids"].to(device)
    input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    # Build ref_ids if not x-vector only
    ref_ids = None
    if not x_vector_only and ref_text:
        ref_text_formatted = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ref_tok = tts.processor(text=ref_text_formatted, return_tensors="pt", padding=True)["input_ids"].to(device)
        ref_tok = ref_tok.unsqueeze(0) if ref_tok.dim() == 1 else ref_tok
        ref_ids = [ref_tok]

    # Generate
    talker_codes_list, _ = tts.model.generate(
        input_ids=[input_ids],
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt,
        languages=["Auto"],
        non_streaming_mode=True,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        max_new_tokens=2048,
    )

    # Decode with ref_code prepended if ICL mode
    codes = talker_codes_list[0]
    if not x_vector_only and ref_code is not None:
        codes_for_decode = torch.cat([ref_code.to(codes.device), codes], dim=0)
    else:
        codes_for_decode = codes

    wavs_all, sr = tts.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])

    # Cut off the reference portion if ICL mode
    wav = wavs_all[0]
    if not x_vector_only and ref_code is not None:
        ref_len = int(ref_code.shape[0])
        total_len = int(codes_for_decode.shape[0])
        cut = int(ref_len / max(total_len, 1) * wav.shape[0])
        wav = wav[cut:]

    return wav, sr


def run_conversation(
    voice_a: str,
    voice_b: str,
    name_a: str,
    name_b: str,
    personality_a: str,
    personality_b: str,
    topic: str,
    num_turns: int,
    pause_between: float,
    model_size: str,
    progress=gr.Progress()
):
    """Run a full conversation between two AI personas."""
    global llm_model

    if llm_model is None:
        return None, "❌ Error: Please load the LLM first in the Models tab.", ""

    if not voice_a or not voice_b:
        return None, "❌ Error: Please select both voice files.", ""

    if not name_a or not name_b:
        return None, "❌ Error: Please enter names for both speakers.", ""

    if not topic:
        return None, "❌ Error: Please enter a conversation topic.", ""

    print(f"\n{'='*60}")
    print(f"🎭 Starting Conversation Mode")
    print(f"{'='*60}")
    print(f"👤 Speaker A: {name_a} ({voice_a})")
    print(f"👤 Speaker B: {name_b} ({voice_b})")
    print(f"📝 Topic: {topic}")
    print(f"🔄 Turns: {num_turns}")
    print(f"{'='*60}\n")

    conversation_history = []
    all_audio = []
    transcript_lines = []
    sr = None

    speakers = [
        {"name": name_a, "personality": personality_a, "voice": voice_a},
        {"name": name_b, "personality": personality_b, "voice": voice_b},
    ]

    total_steps = num_turns * 2  # Each turn has LLM generation + TTS
    current_step = 0

    for turn_idx in range(num_turns):
        for speaker_idx, speaker in enumerate(speakers):
            other_speaker = speakers[1 - speaker_idx]

            # Generate dialogue
            progress(current_step / total_steps, desc=f"🤖 {speaker['name']} is thinking...")
            print(f"🤖 Generating dialogue for {speaker['name']}...")

            try:
                text = generate_conversation_turn(
                    conversation_history=conversation_history,
                    speaker_name=speaker["name"],
                    speaker_personality=speaker["personality"],
                    other_speaker_name=other_speaker["name"],
                    topic=topic,
                    is_first_turn=(turn_idx == 0 and speaker_idx == 0)
                )
            except Exception as e:
                print(f"❌ LLM Error: {e}")
                return None, f"❌ LLM Error: {str(e)}", "\n".join(transcript_lines)

            print(f"   💬 {speaker['name']}: {text}")

            conversation_history.append({"speaker": speaker["name"], "text": text})
            transcript_lines.append(f"**{speaker['name']}:** {text}")

            current_step += 1

            # Generate audio
            progress(current_step / total_steps, desc=f"🎙️ {speaker['name']} is speaking...")
            print(f"🎙️ Generating audio for {speaker['name']}...")

            try:
                wav, sr = generate_voice_for_text(speaker["voice"], text, model_size)
                all_audio.append(wav)

                # Add pause between speakers
                if pause_between > 0:
                    pause_samples = int(sr * pause_between)
                    all_audio.append(np.zeros(pause_samples, dtype=np.float32))

                print(f"   ✅ Generated {len(wav)/sr:.2f}s of audio")
            except Exception as e:
                print(f"❌ TTS Error: {e}")
                return None, f"❌ TTS Error: {str(e)}", "\n".join(transcript_lines)

            current_step += 1

    # Concatenate all audio
    if all_audio:
        final_audio = np.concatenate(all_audio)
        total_duration = len(final_audio) / sr

        print(f"\n{'='*60}")
        print(f"✅ Conversation complete!")
        print(f"📊 Total duration: {total_duration:.2f}s")
        print(f"💬 Total turns: {num_turns * 2}")
        print(f"{'='*60}\n")

        status = f"✅ Generated {num_turns * 2} turns, {total_duration:.1f}s total"
        transcript = "\n\n".join(transcript_lines)

        return (sr, final_audio), status, transcript

    return None, "❌ No audio generated.", ""


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        primary_hue="indigo",
        secondary_hue="slate",
    )

    css = """
    .gradio-container {
        max-width: 100% !important;
        padding: 0 2rem !important;
    }
    .header-container {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .header-container h1 {
        color: white !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .header-container p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
    }
    .feature-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: white;
    }
    .tab-content {
        min-height: 500px !important;
    }
    .tabitem {
        min-height: 500px !important;
    }
    """

    with gr.Blocks(title="Qwen3-TTS Demo") as demo:
        gr.HTML(
            """
            <div class="header-container">
                <h1>🎙️ Qwen3-TTS</h1>
                <p>High-Quality Text-to-Speech with Voice Cloning & Design</p>
                <div style="margin-top: 1rem;">
                    <span class="feature-badge">🎨 Voice Design</span>
                    <span class="feature-badge">🎭 Voice Clone</span>
                    <span class="feature-badge">🗣️ Custom Voices</span>
                    <span class="feature-badge">📝 Long Text Chunking</span>
                </div>
            </div>
            """
        )

        with gr.Tabs():
            # Tab 0: Model Management (Collapsible sections)
            with gr.Tab("⚙️ Models"):
                with gr.Accordion("📥 Download Models", open=True):
                    gr.Markdown("*💡 Tip: Models can be downloaded here or will auto-download when you generate in any tab.*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                download_model_type = gr.Dropdown(
                                    label="Type",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                download_model_size = gr.Dropdown(
                                    label="Size",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            download_btn = gr.Button("Download", variant="primary", size="sm")
                            download_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            models_status = gr.Markdown(value=get_downloaded_models_status)
                
                download_model_type.change(
                    get_available_sizes,
                    inputs=[download_model_type],
                    outputs=[download_model_size],
                )
                
                download_btn.click(
                    download_model,
                    inputs=[download_model_type, download_model_size],
                    outputs=[download_status, models_status],
                )
                
                with gr.Accordion("🚀 Load Models to GPU", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                load_model_type = gr.Dropdown(
                                    label="Type",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                load_model_size = gr.Dropdown(
                                    label="Size",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            load_btn = gr.Button("Load to GPU", variant="primary", size="sm")
                            load_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            load_refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                            load_loaded_status = gr.Markdown(value=get_loaded_models_status)
                
                load_model_type.change(
                    get_available_sizes,
                    inputs=[load_model_type],
                    outputs=[load_model_size],
                )
                
                load_refresh_btn.click(
                    lambda: get_loaded_models_status(),
                    inputs=[],
                    outputs=[load_loaded_status],
                )
                
                load_btn.click(
                    load_model_manual,
                    inputs=[load_model_type, load_model_size],
                    outputs=[load_status, load_loaded_status],
                )
                
                with gr.Accordion("🗑️ Unload Models", open=False):
                    gr.Markdown("*💡 Tip: Click 'Refresh Status' to see models loaded from other tabs.*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                unload_model_type = gr.Dropdown(
                                    label="Type",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                unload_model_size = gr.Dropdown(
                                    label="Size",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            with gr.Row():
                                unload_btn = gr.Button("Unload Selected", variant="secondary", size="sm")
                                unload_all_btn = gr.Button("Unload All", variant="stop", size="sm")
                            unload_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                            loaded_status = gr.Markdown(value=get_loaded_models_status)
                
                unload_model_type.change(
                    get_available_sizes,
                    inputs=[unload_model_type],
                    outputs=[unload_model_size],
                )
                
                refresh_btn.click(
                    lambda: get_loaded_models_status(),
                    inputs=[],
                    outputs=[loaded_status],
                )
                
                unload_btn.click(
                    unload_model,
                    inputs=[unload_model_type, unload_model_size],
                    outputs=[unload_status, loaded_status],
                )
                
                unload_all_btn.click(
                    unload_all_models,
                    inputs=[],
                    outputs=[unload_status, loaded_status],
                )

                # LLM Models section (for Conversation Mode)
                with gr.Accordion("🤖 LLM Models (for Conversation Mode)", open=False):
                    gr.Markdown("*💡 Download and load the LLM to enable AI-powered conversation generation.*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            llm_dropdown = gr.Dropdown(
                                label="LLM Model",
                                choices=list(AVAILABLE_LLMS.keys()),
                                value="Qwen3-4B-Instruct",
                                interactive=True,
                            )
                            with gr.Row():
                                llm_download_btn = gr.Button("📥 Download", variant="primary", size="sm")
                                llm_load_btn = gr.Button("🚀 Load", variant="secondary", size="sm")
                                llm_unload_btn = gr.Button("🗑️ Unload", variant="stop", size="sm")
                            llm_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            llm_download_status = gr.Markdown(value=get_llm_download_status)
                            llm_loaded_status = gr.Markdown(value=get_llm_loaded_status)

                llm_download_btn.click(
                    download_llm,
                    inputs=[llm_dropdown],
                    outputs=[llm_status, llm_download_status],
                )

                llm_load_btn.click(
                    load_llm,
                    inputs=[llm_dropdown],
                    outputs=[llm_status, llm_loaded_status],
                )

                llm_unload_btn.click(
                    unload_llm,
                    inputs=[],
                    outputs=[llm_status, llm_loaded_status],
                )

            # Tab 1: Voice Design
            with gr.Tab("🎨 Voice Design"):
                gr.Markdown("*ℹ️ Voice Design generates unique voices from descriptions. Max ~2048 tokens (~300-500 chars recommended). No chunking - for longer texts use Voice Clone or Custom Voice.*")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        design_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=6,
                            placeholder="Enter the text you want to convert to speech (keep under ~500 chars)...",
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                        )
                        design_instruct = gr.Textbox(
                            label="Voice Description",
                            lines=3,
                            placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                        )
                        with gr.Row():
                            design_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            design_seed = gr.Number(
                                label="Seed (-1 = Auto)",
                                value=-1,
                                precision=0,
                            )
                        design_btn = gr.Button("🎙️ Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct, design_seed],
                    outputs=[design_audio_out, design_status],
                )

            # Tab 2: Voice Clone (with sub-tabs)
            with gr.Tab("🎭 Voice Clone"):
                with gr.Tabs():
                    # Sub-tab 1: Clone & Generate (original functionality)
                    with gr.Tab("Clone & Generate"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                clone_ref_audio = gr.Audio(
                                    label="Reference Audio",
                                    type="numpy",
                                )
                                with gr.Row():
                                    clone_ref_text = gr.Textbox(
                                        label="Reference Text",
                                        lines=2,
                                        placeholder="Transcript of reference audio...",
                                        scale=3,
                                    )
                                    transcribe_btn = gr.Button("🎤 Transcribe", scale=1)
                                clone_xvector = gr.Checkbox(
                                    label="X-vector only (no text needed, lower quality)",
                                    value=False,
                                )
                                clone_target_text = gr.Textbox(
                                    label="Target Text",
                                    lines=5,
                                    placeholder="Text to synthesize with cloned voice...",
                                )
                                with gr.Row():
                                    clone_language = gr.Dropdown(
                                        label="Language",
                                        choices=LANGUAGES,
                                        value="Auto",
                                        interactive=True,
                                    )
                                    clone_model_size = gr.Dropdown(
                                        label="Size",
                                        choices=MODEL_SIZES,
                                        value="1.7B",
                                        interactive=True,
                                    )
                                with gr.Row():
                                    clone_chunk_size = gr.Slider(
                                        label="Chunk Size",
                                        minimum=50,
                                        maximum=500,
                                        value=200,
                                        step=10,
                                    )
                                    clone_chunk_gap = gr.Slider(
                                        label="Chunk Gap (s)",
                                        minimum=0.0,
                                        maximum=3.0,
                                        value=0.0,
                                        step=0.01,
                                    )
                                with gr.Row():
                                    clone_seed = gr.Number(
                                        label="Seed (-1 = Auto)",
                                        value=-1,
                                        precision=0,
                                    )
                                clone_btn = gr.Button("🎙️ Clone & Generate", variant="primary", size="lg")

                            with gr.Column(scale=1):
                                clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                                clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

                        transcribe_btn.click(
                            transcribe_audio,
                            inputs=[clone_ref_audio],
                            outputs=[clone_ref_text],
                        )

                        clone_btn.click(
                            generate_voice_clone,
                            inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size, clone_chunk_size, clone_chunk_gap, clone_seed],
                            outputs=[clone_audio_out, clone_status],
                        )

                    # Sub-tab 2: Save / Load Voice
                    with gr.Tab("Save / Load Voice (Save/Load Cloned Voice)"):
                        with gr.Row(equal_height=True):
                            # Left column: Save Voice
                            with gr.Column(scale=1):
                                gr.Markdown("### 💾 Save Voice")
                                gr.Markdown("*Upload reference audio and text, choose whether to use x-vector only, and then save a reusable voice prompt file.*")

                                save_ref_audio = gr.Audio(
                                    label="Reference Audio",
                                    type="numpy",
                                )
                                with gr.Row():
                                    save_ref_text = gr.Textbox(
                                        label="Reference Text (audio text)",
                                        lines=2,
                                        placeholder="Required if 'use x-vector only' is not selected.",
                                        scale=3,
                                    )
                                    save_transcribe_btn = gr.Button("🎤 Transcribe", scale=1)
                                save_xvector = gr.Checkbox(
                                    label="Use x-vector only (using only the speaker vector has limited effect, but you don't need to pass in the reference audio text)",
                                    value=False,
                                )
                                with gr.Row():
                                    save_voice_name = gr.Textbox(
                                        label="Voice Name",
                                        placeholder="Enter a name for this voice...",
                                        scale=2,
                                    )
                                    save_model_size = gr.Dropdown(
                                        label="Model Size",
                                        choices=MODEL_SIZES,
                                        value="1.7B",
                                        interactive=True,
                                        scale=1,
                                    )
                                save_btn = gr.Button("💾 Save Voice File", variant="primary", size="lg")
                                save_status = gr.Textbox(label="Status", lines=1, interactive=False)

                                # Voice File management
                                gr.Markdown("---")
                                gr.Markdown("### 📁 Voice File")
                                voice_files_info = gr.Markdown(value=get_voice_files_info)
                                with gr.Row():
                                    refresh_voices_btn = gr.Button("🔄 Refresh", size="sm")
                                    delete_voice_dropdown = gr.Dropdown(
                                        label="Select to delete",
                                        choices=get_saved_voices_list(),
                                        interactive=True,
                                        scale=2,
                                    )
                                    delete_voice_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")

                            # Right column: Load Voice & Generate
                            with gr.Column(scale=1):
                                gr.Markdown("### 🔊 Load Voice & Generate")
                                gr.Markdown("*Upload a previously saved voice file, then synthesize new text.*")

                                load_voice_dropdown = gr.Dropdown(
                                    label="Upload Prompt File",
                                    choices=get_saved_voices_list(),
                                    interactive=True,
                                )
                                load_target_text = gr.Textbox(
                                    label="Target Text (Text to be synthesized)",
                                    lines=5,
                                    placeholder="Enter text to synthesize.",
                                )
                                with gr.Row():
                                    load_language = gr.Dropdown(
                                        label="Language",
                                        choices=LANGUAGES,
                                        value="Auto",
                                        interactive=True,
                                    )
                                    load_model_size = gr.Dropdown(
                                        label="Model Size",
                                        choices=MODEL_SIZES,
                                        value="1.7B",
                                        interactive=True,
                                    )
                                with gr.Row():
                                    load_chunk_size = gr.Slider(
                                        label="Chunk Size",
                                        minimum=50,
                                        maximum=500,
                                        value=200,
                                        step=10,
                                    )
                                    load_chunk_gap = gr.Slider(
                                        label="Chunk Gap (s)",
                                        minimum=0.0,
                                        maximum=3.0,
                                        value=0.0,
                                        step=0.01,
                                    )
                                load_seed = gr.Number(
                                    label="Seed (-1 = Auto)",
                                    value=-1,
                                    precision=0,
                                )
                                load_generate_btn = gr.Button("🎙️ Generate", variant="primary", size="lg")

                                load_audio_out = gr.Audio(label="Output Audio (Synthesized Result)", type="numpy")
                                load_status = gr.Textbox(label="Status", lines=2, interactive=False)

                        # Event handlers for Save / Load Voice tab
                        save_transcribe_btn.click(
                            transcribe_audio,
                            inputs=[save_ref_audio],
                            outputs=[save_ref_text],
                        )

                        save_btn.click(
                            save_voice_file,
                            inputs=[save_ref_audio, save_ref_text, save_xvector, save_voice_name, save_model_size],
                            outputs=[save_status, voice_files_info, load_voice_dropdown],
                        ).then(
                            lambda: gr.update(choices=get_saved_voices_list()),
                            outputs=[delete_voice_dropdown],
                        )

                        refresh_voices_btn.click(
                            lambda: (get_voice_files_info(), gr.update(choices=get_saved_voices_list()), gr.update(choices=get_saved_voices_list())),
                            outputs=[voice_files_info, load_voice_dropdown, delete_voice_dropdown],
                        )

                        delete_voice_btn.click(
                            delete_voice_file,
                            inputs=[delete_voice_dropdown],
                            outputs=[save_status, voice_files_info, load_voice_dropdown],
                        ).then(
                            lambda: gr.update(choices=get_saved_voices_list()),
                            outputs=[delete_voice_dropdown],
                        )

                        load_generate_btn.click(
                            load_voice_and_generate,
                            inputs=[load_voice_dropdown, load_target_text, load_language, load_model_size, load_chunk_size, load_chunk_gap, load_seed],
                            outputs=[load_audio_out, load_status],
                        )

            # Tab 3: Custom Voice TTS
            with gr.Tab("🗣️ Custom Voice"):
                gr.Markdown("*ℹ️ Custom Voice uses predefined speakers. Max ~2048 tokens (~300-500 chars recommended). For longer texts use Voice Clone.*")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=6,
                            placeholder="Enter the text you want to convert to speech (keep under ~500 chars)...",
                            value="Hello! Welcome to the Text-to-Speech system. This is a demo of our TTS capabilities."
                        )
                        with gr.Row():
                            tts_speaker = gr.Dropdown(
                                label="Speaker",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                            tts_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                        tts_instruct = gr.Textbox(
                            label="Style Instruction (Optional, 1.7B only)",
                            lines=2,
                            placeholder="e.g., Speak in a cheerful and energetic tone",
                        )
                        with gr.Row():
                            tts_model_size = gr.Dropdown(
                                label="Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                            tts_seed = gr.Number(
                                label="Seed (-1 = Auto)",
                                value=-1,
                                precision=0,
                            )
                        tts_btn = gr.Button("🎙️ Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size, tts_seed],
                    outputs=[tts_audio_out, tts_status],
                )

            # Tab 4: Conversation Mode
            with gr.Tab("💬 Conversation Mode"):
                gr.Markdown("""
                ### AI-Powered Conversation Generator
                *Create dynamic conversations between two AI personas using saved voice profiles.*

                **Requirements:**
                1. Download & Load the LLM in the Models tab
                2. Save at least 2 voice profiles in Voice Clone → Save/Load Voice
                """)

                with gr.Row(equal_height=True):
                    # Left column: Speaker A
                    with gr.Column(scale=1):
                        gr.Markdown("### 👤 Speaker A")
                        conv_voice_a = gr.Dropdown(
                            label="Voice Profile",
                            choices=get_saved_voices_list(),
                            interactive=True,
                        )
                        conv_name_a = gr.Textbox(
                            label="Name",
                            placeholder="e.g., Alex",
                            value="Alex",
                        )
                        conv_personality_a = gr.Textbox(
                            label="Personality",
                            lines=3,
                            placeholder="Describe this character's personality...",
                            value="You are a curious and enthusiastic tech enthusiast who loves discussing new innovations. You ask thoughtful questions and share interesting facts.",
                        )

                    # Middle column: Speaker B
                    with gr.Column(scale=1):
                        gr.Markdown("### 👤 Speaker B")
                        conv_voice_b = gr.Dropdown(
                            label="Voice Profile",
                            choices=get_saved_voices_list(),
                            interactive=True,
                        )
                        conv_name_b = gr.Textbox(
                            label="Name",
                            placeholder="e.g., Jordan",
                            value="Jordan",
                        )
                        conv_personality_b = gr.Textbox(
                            label="Personality",
                            lines=3,
                            placeholder="Describe this character's personality...",
                            value="You are a pragmatic realist who likes to consider both sides of any topic. You're friendly but sometimes play devil's advocate to spark interesting discussion.",
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        conv_topic = gr.Textbox(
                            label="Conversation Topic",
                            lines=2,
                            placeholder="What should they talk about?",
                            value="The future of artificial intelligence and how it will change everyday life",
                        )
                    with gr.Column(scale=1):
                        conv_turns = gr.Slider(
                            label="Number of Turns (exchanges)",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                        )
                        conv_pause = gr.Slider(
                            label="Pause Between Speakers (s)",
                            minimum=0.0,
                            maximum=3.0,
                            value=0.5,
                            step=0.1,
                        )
                        conv_model_size = gr.Dropdown(
                            label="TTS Model Size",
                            choices=MODEL_SIZES,
                            value="1.7B",
                            interactive=True,
                        )

                with gr.Row():
                    conv_refresh_btn = gr.Button("🔄 Refresh Voice List", size="sm")
                    conv_generate_btn = gr.Button("🎭 Generate Conversation", variant="primary", size="lg")

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        conv_audio_out = gr.Audio(label="Generated Conversation", type="numpy")
                        conv_status = gr.Textbox(label="Status", lines=2, interactive=False)
                    with gr.Column(scale=1):
                        conv_transcript = gr.Markdown(label="Transcript", value="*Transcript will appear here...*")

                # Event handlers
                conv_refresh_btn.click(
                    lambda: (gr.update(choices=get_saved_voices_list()), gr.update(choices=get_saved_voices_list())),
                    outputs=[conv_voice_a, conv_voice_b],
                )

                conv_generate_btn.click(
                    run_conversation,
                    inputs=[
                        conv_voice_a, conv_voice_b,
                        conv_name_a, conv_name_b,
                        conv_personality_a, conv_personality_b,
                        conv_topic, conv_turns, conv_pause, conv_model_size
                    ],
                    outputs=[conv_audio_out, conv_status, conv_transcript],
                )

    return demo, theme, css


if __name__ == "__main__":
    demo, theme, css = build_ui()
    demo.launch(theme=theme, css=css)

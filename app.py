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

# Flash-attn is installed via torch.js during Pinokio install - no runtime install needed

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, scan_cache_dir

# Voice files directory
VOICE_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_voices")
os.makedirs(VOICE_FILES_DIR, exist_ok=True)

# Whisper model for transcription
whisper_model = None


def get_whisper_model():
    """Load Whisper tiny model for transcription."""
    global whisper_model
    if whisper_model is None:
        import whisper
        whisper_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
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
        spk_emb_np = spk_emb.cpu().float().numpy()

        # Extract speech codes if not x-vector only mode
        ref_code_np = None
        if not use_xvector_only:
            enc = tts.model.speech_tokenizer.encode([wav], sr=sr)
            ref_code = enc.audio_codes[0]
            # Convert to float32 if needed before numpy
            ref_code_np = ref_code.cpu().float().numpy()

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

    return demo, theme, css


if __name__ == "__main__":
    demo, theme, css = build_ui()
    demo.launch(theme=theme, css=css)

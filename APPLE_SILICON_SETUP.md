# Apple Silicon Setup Guide

## 🎉 Your project now supports Apple Silicon with MPS acceleration!

The error "Torch not compiled with CUDA enabled" should now be resolved. The app will automatically use your Mac's GPU via Metal Performance Shaders (MPS).

## Quick Start

### 1. Reset Your Installation

Since you had a previous installation with CPU-only PyTorch, you need to reset:

**Option A: Using Pinokio UI (Recommended)**
1. Click the "Reset" button in the Pinokio launcher
2. Click "Install" to reinstall with MPS support

**Option B: Manual Reset**
```bash
cd /Users/red/pinokio/api/Qwen3-TTS-Pinokio.git/app
rm -rf venv
```

Then run the install script again from Pinokio.

### 2. Verify MPS Support

After installation, test that MPS is working:

```bash
cd /Users/red/pinokio/api/Qwen3-TTS-Pinokio.git/app
source venv/bin/activate
python test_mps.py
```

You should see:
```
🚀 Using device: mps
✅ All MPS tests passed!
Your Apple Silicon GPU is ready for TTS generation!
```

### 3. Launch the App

Start the app as usual. You should now see:
```
🚀 Using device: mps
```

No more "CUDA not compiled" errors! 🎉

## What Changed?

### Automatic Device Detection
The app now intelligently detects your hardware:
- ✅ Apple Silicon (M1/M2/M3/M4) → Uses **MPS** for TTS models
- ✅ NVIDIA GPU → Uses **CUDA**  
- ✅ No GPU → Falls back to **CPU**

**Note on Whisper**: The transcription feature uses Whisper on CPU (not MPS) because Whisper relies on sparse tensors which aren't fully supported on MPS yet. This only affects transcription - all TTS generation runs on GPU with full MPS acceleration.

### PyTorch Installation
- Old: CPU-only PyTorch (no GPU support)
- New: Standard PyTorch with MPS support built-in

### Code Changes
All hardcoded CUDA references have been replaced with automatic device detection:
- Model loading (MPS accelerated)
- Memory management (CUDA and MPS)
- Whisper transcription (CPU to avoid sparse tensor issues)
- Seed setting (cross-platform)

## Performance

With MPS acceleration, you should see:
- **3-10x faster** inference compared to CPU
- **Smooth generation** without errors
- **Lower CPU usage** as work moves to GPU

## Troubleshooting

### Transcription Feature

If transcription doesn't work or shows errors:

**This is now fixed!** Whisper runs on CPU automatically because:
- Whisper uses sparse tensors
- MPS doesn't fully support sparse tensor operations yet
- Running on CPU avoids the error while TTS models still use GPU

You'll see this message when transcribing:
```
📝 Loading Whisper model on cpu (sparse tensors not supported on MPS)
```

This is normal and expected behavior. Transcription will be slightly slower but still functional.

### If you still see CUDA errors:

1. **Check you've reset the installation**
   ```bash
   cd app
   source venv/bin/activate
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```
   Should print: `True`

2. **Verify PyTorch version**
   ```bash
   pip show torch
   ```
   Should be version 2.7.0 or higher

3. **Force reinstall PyTorch**
   ```bash
   cd app
   source venv/bin/activate
   pip uninstall torch torchvision torchaudio -y
   pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
   ```

### If MPS shows as unavailable:

1. **Check macOS version** - MPS requires macOS 12.3 or later
2. **Verify you're on Apple Silicon**
   ```bash
   uname -m
   ```
   Should print: `arm64`

3. **Update PyTorch**
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

## Monitor GPU Usage

To see your GPU working:

1. Open **Activity Monitor** (Applications → Utilities)
2. Go to **Window** → **GPU History**
3. Start generating audio
4. You should see GPU usage spike during generation

## Files Created/Modified

### New Files
- `APPLE_SILICON.md` - Detailed Apple Silicon guide
- `APPLE_SILICON_SETUP.md` - This file
- `CHANGELOG_APPLE_SILICON.md` - Technical changelog
- `app/test_mps.py` - MPS verification script

### Modified Files
- `app/app.py` - Added device detection and MPS support
- `torch.js` - Updated PyTorch installation for MPS
- `app/README.md` - Added Apple Silicon documentation

## Support

If you encounter any issues:

1. Run the test script: `python test_mps.py`
2. Check the detailed guide: `APPLE_SILICON.md`
3. Review the changelog: `CHANGELOG_APPLE_SILICON.md`

## Enjoy! 🚀

Your Qwen3-TTS app now has full Apple Silicon support with GPU acceleration!

# Apple Silicon (M1/M2/M3/M4) Support

This project now fully supports Apple Silicon Macs with **hardware acceleration via MPS (Metal Performance Shaders)**.

## What Changed

### Automatic Device Detection
The app now automatically detects and uses the best available device:
- **MPS** for Apple Silicon Macs (M1/M2/M3/M4)
- **CUDA** for NVIDIA GPUs
- **CPU** as fallback

### Performance
On Apple Silicon Macs, the models will run on the GPU using Metal Performance Shaders, providing significant speedup compared to CPU-only execution.

## Usage

No special configuration needed! Just:

1. **Install** - Run the install script
2. **Start** - Launch the app
3. The app will automatically detect your Apple Silicon GPU and use MPS acceleration

You'll see this message when the app starts:
```
🚀 Using device: mps
```

## Technical Details

### PyTorch with MPS Support
- The installer now installs the standard PyTorch build (not CPU-only)
- PyTorch 2.7.0+ includes native MPS support for Apple Silicon
- Models are automatically loaded to MPS device

### Supported Operations
- **TTS Model Inference** - VoiceDesign, Base, CustomVoice run on MPS (GPU accelerated)
- **Audio Transcription** - Whisper runs on CPU (see note below)
- **All TTS generation modes** - Fully supported with MPS acceleration

### Limitations & Workarounds

#### Whisper Transcription (CPU Only)
Whisper uses sparse tensors internally, which aren't fully supported on MPS yet. The app automatically runs Whisper on CPU to avoid errors. This is a minor performance impact since:
- Whisper is only used during transcription (not TTS generation)
- The "tiny" model is fast even on CPU
- Whisper auto-unloads after transcription to free memory

#### Other Limitations
- **Flash Attention 2** - Not available on MPS (automatically disabled)
- **FP16 precision** - May have reduced accuracy on MPS (using bfloat16 instead)
- **Some operations** - May fall back to CPU if not MPS-optimized

## Troubleshooting

### Transcription Errors (Sparse Tensor Issues)
If you see errors like `'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseMPS' backend`:

**This is now fixed!** The app automatically runs Whisper on CPU to avoid sparse tensor issues on MPS. You should see this message when transcribing:
```
📝 Loading Whisper model on cpu (sparse tensors not supported on MPS)
```

The transcription will be slightly slower on CPU, but TTS generation still uses full GPU acceleration.

### If you see "CUDA not compiled" error
This means you have an old installation. To fix:

1. **Uninstall old PyTorch**:
   ```bash
   cd app
   source venv/bin/activate
   pip uninstall torch torchvision torchaudio -y
   ```

2. **Reinstall with MPS support**:
   - Click "Reset" in the Pinokio UI to reset dependencies
   - Click "Install" again to reinstall with MPS support

### Check Your Device
You can verify MPS is available by running:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### Performance Tips
- Ensure you have at least 8GB of unified memory for 0.6B models
- 16GB+ recommended for 1.7B models
- Close other GPU-intensive apps for best performance
- Use the model management tab to preload/unload models as needed

## Verification

After installation, you should see:
```
🚀 Using device: mps
```

When generating audio, check the terminal output - successful MPS usage will show smooth generation without "CUDA not compiled" errors.

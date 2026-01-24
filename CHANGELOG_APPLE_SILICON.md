# Changelog: Apple Silicon Support

## Summary
Added full Apple Silicon (M1/M2/M3/M4) support with MPS (Metal Performance Shaders) hardware acceleration. The app now automatically detects and uses the best available device (MPS, CUDA, or CPU).

## Changes Made

### 1. app/app.py - Core Application Updates

#### Added Device Detection Functions
- `get_device()` - Automatically detects best device (MPS → CUDA → CPU)
- `get_device_type()` - Returns device type for device-specific operations
- `empty_cache()` - Cross-platform GPU cache clearing (supports both CUDA and MPS)
- `supports_fp16()` - Checks if device supports FP16 precision

#### Updated Model Loading
- **Before**: Hardcoded `device_map="cuda"`
- **After**: Dynamic device mapping based on available hardware
  ```python
  device_map = DEVICE if DEVICE == "cpu" else "auto"
  ```

#### Updated Whisper Integration
- **Fixed sparse tensor error on MPS**: Whisper now always runs on CPU
- Reason: Whisper uses sparse tensors which aren't fully supported on MPS
- Workaround: Force `device="cpu"` for Whisper model loading
- Transcription now uses `fp16=False` since it runs on CPU
- Added informative message when loading Whisper on CPU

#### Updated Seed Setting
- Added MPS-specific handling in `set_seed()` function
- CUDA-specific settings only apply when using NVIDIA GPU

#### Updated Memory Management
- All `torch.cuda.empty_cache()` calls replaced with `empty_cache()` function
- Works with both CUDA and MPS devices

### 2. torch.js - PyTorch Installation Script

#### Apple Silicon Installation Updated
- **Before**: Installed CPU-only PyTorch
  ```javascript
  "message": "uv pip install torch==2.7.0... --index-url https://download.pytorch.org/whl/cpu"
  ```
- **After**: Installs standard PyTorch with MPS support
  ```javascript
  "message": "uv pip install torch==2.7.0... --force-reinstall --no-deps"
  ```

### 3. Documentation

#### Created APPLE_SILICON.md
- Comprehensive guide for Apple Silicon users
- Explains MPS acceleration and benefits
- Troubleshooting section for common issues
- Performance tips and memory requirements

#### Updated app/README.md
- Added Apple Silicon badge and feature
- Updated requirements section to include MPS
- Added Apple Silicon installation instructions
- Clarified GPU vs CPU requirements

#### Created test_mps.py
- Test script to verify MPS support
- Checks PyTorch version and MPS availability
- Tests basic MPS tensor operations
- Provides clear pass/fail output

## Technical Details

### Device Detection Logic
```
1. Check if CUDA is available → use "cuda"
2. Check if MPS is available → use "mps"
3. Fallback to "cpu"
```

### Automatic Model Placement
- Models automatically move to the detected device
- No manual device specification needed
- Works seamlessly across all platforms

### Memory Management
- Proper cache clearing for both CUDA and MPS
- Unified `empty_cache()` function
- Prevents memory leaks on all platforms

## Testing Recommendations

### Before Starting the App
Run the test script to verify MPS support:
```bash
cd app
source venv/bin/activate
python test_mps.py
```

Expected output:
```
🚀 Using device: mps
✅ All MPS tests passed!
```

### First Run
1. Start the app: `python app.py`
2. Look for: `🚀 Using device: mps` in the console
3. Try generating with any model
4. Should see smooth generation without errors

### Verify GPU Usage
Monitor GPU usage with Activity Monitor:
1. Open Activity Monitor
2. Go to "Window" → "GPU History"
3. Generate audio - should see GPU activity spike

## Known Limitations

1. **Whisper Transcription**: Runs on CPU (not MPS) due to sparse tensor limitations
   - Whisper uses sparse tensors internally
   - MPS doesn't fully support sparse tensor operations
   - Workaround: Force Whisper to use CPU device
   - Impact: Minor - transcription is less frequent than generation
   
2. **Flash Attention**: Not available on MPS (automatically disabled)

3. **FP16 Precision**: Less stable on MPS, using bfloat16 instead

4. **Some Operations**: May fall back to CPU if not MPS-optimized

## Rollback Instructions

If you need to revert to CPU-only mode:

1. Set environment variable:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

2. Or modify `app.py` to force CPU:
   ```python
   DEVICE = "cpu"  # Force CPU mode
   ```

## Future Enhancements

Potential improvements for Apple Silicon support:
- [ ] MPS-specific optimizations
- [ ] Better memory management for unified memory
- [ ] Performance benchmarks vs CUDA
- [ ] Support for quantized models on MPS

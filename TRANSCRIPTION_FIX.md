# Transcription Feature - Apple Silicon Fix

## Problem
When clicking "Transcribe" on Apple Silicon, you got this error:
```
Transcription error: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' 
with arguments from the 'SparseMPS' backend.
```

## Root Cause
- Whisper (the transcription model) uses **sparse tensors** internally
- PyTorch's MPS backend doesn't fully support sparse tensor operations yet
- This caused the sparse tensor operation to fail on MPS

## Solution
The app now **automatically runs Whisper on CPU** instead of MPS.

### What Changed
```python
# Before (tried to use MPS)
whisper_model = whisper.load_model("tiny", device=DEVICE)  # DEVICE could be "mps"

# After (always uses CPU)
whisper_device = "cpu"
whisper_model = whisper.load_model("tiny", device=whisper_device)
```

### What You'll See
When you click "Transcribe", you'll see this message in the console:
```
📝 Loading Whisper model on cpu (sparse tensors not supported on MPS)
```

This is **normal and expected**!

## Performance Impact

### Minimal Impact
- **Transcription**: Runs on CPU (slightly slower, but still fast with "tiny" model)
- **TTS Generation**: Still runs on MPS GPU (full acceleration)

### Why This Is OK
1. Transcription is **infrequent** - you only transcribe once per reference audio
2. The "tiny" Whisper model is **fast on CPU** - usually completes in 1-2 seconds
3. Whisper **auto-unloads** after transcription to free memory
4. TTS models (which you use more often) **still get full GPU acceleration**

## Testing

Try the transcribe feature now:

1. Go to the "🎭 Voice Clone" tab
2. Upload a reference audio file
3. Click "🎤 Transcribe"
4. You should see the transcription appear without errors!

## Technical Details

### Why Not Fix Sparse Tensors on MPS?
This is a PyTorch/Apple limitation:
- Sparse tensors on MPS are under development
- Full support will come in future PyTorch versions
- Running on CPU is the safest workaround for now

### Will This Be Fixed?
Once PyTorch adds full sparse tensor support for MPS, we can update to:
```python
whisper_device = DEVICE  # Will use MPS when sparse tensors are supported
```

For now, CPU is the reliable choice for Whisper.

## Summary

✅ **Transcription now works** on Apple Silicon  
✅ **No more sparse tensor errors**  
✅ **TTS generation still GPU accelerated**  
✅ **Minimal performance impact**  

The fix ensures reliability while maintaining excellent performance for TTS generation!

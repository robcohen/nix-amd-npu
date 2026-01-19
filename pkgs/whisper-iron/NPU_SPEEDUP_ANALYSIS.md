# Can We Make NPU Faster for Whisper?

## Current State
- **CPU time**: 17.3s for 10s audio
- **NPU overhead**: ~5.4s (2679 kernel calls × 2ms each)
- **NPU compute**: Unknown (never measured separately due to overhead)

## The Core Problem

The current IRON approach calls the NPU **2679 times** per inference:
- Each call has ~2ms overhead (host-device sync, kernel dispatch)
- Even if NPU compute were instant, we'd still have 5.4s overhead

## What Would Make NPU Faster

### Option 1: Fused Kernels (Hard)
Combine multiple operations into single kernel:
```
Current:  Q_proj → K_proj → V_proj → Q@K^T → softmax → @V → out_proj
Fused:    attention_layer(x, weights) → output
```

**Challenges:**
- AIE tiles have only 16KB local memory
- Whisper uses 384×384 weight matrices (576KB each!)
- Would need complex tiling and streaming
- IRON has bugs that make complex kernels hard (local+local, fused-madd)

### Option 2: Keep Data on NPU (Not Currently Possible)
Avoid host-device sync between layers.

**Challenges:**
- IRON requires `_sync_to_device()` before each kernel
- No API to chain kernels without returning to host
- Would need custom runtime, not IRON

### Option 3: Use AMD's Official Tools (Different Approach)
AMD provides Ryzen AI Software for NPU inference:
- Uses ONNX models with Vitis AI quantization
- Pre-compiled kernels optimized for NPU
- Handles data movement automatically

**This is likely the only practical path to NPU speedup.**

## Honest Assessment

| Approach | Effort | Likely Speedup |
|----------|--------|----------------|
| Current IRON | Done | 0.3x (slower) |
| Cache fix + prewarm | 1 hour | 0.5x (still slower) |
| Fused kernels in IRON | Weeks | Maybe 1-2x |
| AMD Ryzen AI (ONNX) | Days | 2-5x (estimated) |
| CPU (numpy) | Done | 1x (baseline) |

## Recommendation

**For production**: Use AMD Ryzen AI Software with ONNX export
- Export Whisper to ONNX
- Quantize with Vitis AI
- Run with `onnxruntime-vitisai`

**For research/learning**: Current IRON approach is valuable for:
- Understanding NPU architecture
- Prototyping custom kernels
- Learning MLIR-AIE compilation

## What We Learned

1. **NPU kernel overhead is the bottleneck**, not compute
2. **IRON is research-grade**, not production-ready
3. **Fused kernels are essential** for NPU performance
4. **AMD's official tools** are the practical path forward

## Next Steps if Pursuing NPU

1. Export Whisper to ONNX format
2. Install Ryzen AI Software (RyzenAI-SW)
3. Quantize model with Vitis AI
4. Run with onnxruntime-vitisai provider
5. Benchmark against CPU

This would bypass IRON entirely and use AMD's optimized runtime.

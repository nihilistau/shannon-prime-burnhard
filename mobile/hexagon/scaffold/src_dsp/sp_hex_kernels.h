// Shannon-Prime VHT2 - Hexagon DSP scaffold (scalar reference kernels).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#ifndef SP_HEX_KERNELS_H
#define SP_HEX_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// In-place VHT2 forward butterfly. Self-inverse: applying twice yields the
// original. n must be a power of 2 and >= 8. Scalar fp32 reference; an HVX
// kernel will replace this once perf work begins (see TODO in the .c file).
void sp_hex_vht2_f32(float *data, int n);

// Banded quantize: VHT2 coefficients → packed bytes (per-band scales then
// int8 codes). Wire format matches the SP math core's sp_band_quantize.
// out_len is the number of bytes written; caller pre-allocates the buffer
// per the SP packed-size formula.
//
// Returns 0 on success, negative on error (e.g., head_dim invalid).
//
// SCAFFOLD STUB: currently returns -1 (not implemented). Wire to the math
// core or implement on-DSP scalar reference once needed.
int sp_hex_band_quantize_scalar(const float *coeffs, int head_dim,
                                 unsigned char *out, int out_capacity,
                                 int *out_len);

// Banded dequantize. max_bands < 0 means full reconstruction; 0 <= max_bands
// < n_bands triggers the partial path. Wire format matches sp_band_dequantize
// in the math core.
//
// SCAFFOLD STUB: currently returns -1.
int sp_hex_band_dequantize_scalar(const unsigned char *in, int in_len,
                                   int head_dim, int max_bands,
                                   float *out_coeffs);

#ifdef __cplusplus
}
#endif

#endif // SP_HEX_KERNELS_H

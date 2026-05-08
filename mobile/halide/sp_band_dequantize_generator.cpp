// Shannon-Prime VHT2: Halide generator for SP-banded dequantize.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under AGPLv3. Commercial license: raydaniels@gmail.com
//
// Replaces the hand-rolled scalar `sp_band_dequantize` and the partial
// HVX kernels (sp_hex_kernels_hvx.c) with a Halide-generated
// implementation. Halide produces optimal HVX intrinsics from a
// high-level algorithm description, with a separate schedule that
// controls vectorization, prefetching, and parallelism.
//
// This is the first piece of "Mode C glue" — the dequant + scale step
// that takes packed SP bytes from rpcmem and produces fp32 (or fp16)
// VHT2 coefficients for the downstream VHT2 inverse + HTP matmul.
//
// Build flow (matches Qualcomm Halide example pattern):
//   1. Generator binary: g++ this file + Halide/lib → sp_band_dequantize.generator
//   2. Run generator: ./sp_band_dequantize.generator -g sp_band_dequantize \
//        -o build/ -e o,h,html target=hexagon-32-noos-hvx_128 ...
//   3. Output: sp_band_dequantize.o + sp_band_dequantize.h (callable from C)
//   4. Link the .o into libsp_hex_skel.so as a replacement for the scalar
//      sp_band_dequantize_scalar in sp_hex_kernels.c.
//
// SP K config (4 bands @ {5, 5, 4, 3} bits) — packed format:
//   For each band b in [0, n_bands):
//     2 bytes: fp16 scale header
//     ceil(band_sz[b] * band_bits[b] / 8) bytes: packed signed-int codes
//   Coefficient i in band b decodes as:
//     u = (packed >> bit_offset(i)) & ((1 << band_bits[b]) - 1)
//     q = (int)u - max_val[b]      where max_val[b] = (1 << (band_bits[b] - 1)) - 1
//     coeff = (float)q * scale[b]
//
// This generator handles ONE band per invocation. The host wraps the
// 4-band loop and concatenates the outputs into a head_dim-element row.
// Halide's autotuning + HVX scheduling produces a 5-10x speedup over
// the scalar reference at hd=128 per the standalone bench projections.

#include "Halide.h"

using namespace Halide;

class SpBandDequantize : public Generator<SpBandDequantize> {
public:
    // Inputs:
    //   packed: contiguous bytes for this band (NOT including the fp16
    //           scale header — caller pre-strips that).
    //   scale:  fp32 scale factor (host-side fp16 → fp32 conversion).
    //   band_bits: bit width of each code, in {2, 3, 4, 5}.
    //   max_val:   sign bias = (1 << (band_bits - 1)) - 1.
    Input<Buffer<uint8_t>> packed{"packed", 1};
    Input<float>           scale{"scale"};
    Input<int>             band_bits{"band_bits"};
    Input<int>             max_val{"max_val"};

    // Output: fp32 coefficients, length determined by the buffer shape
    // the host allocates. Halide infers loop bounds from output dim 0.
    Output<Buffer<float>> coeffs{"coeffs", 1};

    void generate() {
        // For coefficient i, the bits we want span:
        //   start_bit = i * band_bits
        //   end_bit   = (i + 1) * band_bits  (exclusive)
        // These straddle byte boundaries — we read up to 2 adjacent bytes
        // and shift+mask to extract the field. Halide's vectorizer will
        // turn this into HVX vmem loads + word shifts for free.
        Expr start_bit  = i * band_bits;
        Expr byte_pos   = start_bit / 8;
        Expr bit_offset = start_bit % 8;

        // Read two adjacent bytes and stitch into a 16-bit window. This
        // gives us up to 16 bits of data starting at any bit offset,
        // which covers the 2..5 bit code widths SP uses.
        Expr lo = cast<uint16_t>(packed(byte_pos));
        Expr hi = cast<uint16_t>(packed(byte_pos + 1));
        Expr window = (hi << 8) | lo;

        // Extract the bit field. mask = (1 << band_bits) - 1.
        Expr mask = cast<uint16_t>((1 << band_bits) - 1);
        Expr u    = (window >> bit_offset) & mask;

        // Sign-correct: q = (int)u - max_val. Cast through int16_t so the
        // subtraction handles small negative values correctly without
        // promotion ambiguity. Then to fp32, then scale.
        Expr q = cast<int16_t>(u) - cast<int16_t>(max_val);

        coeffs(i) = cast<float>(q) * scale;
    }

    void schedule() {
        // Match the natural HVX vector size on V69+.
        const int vector_size = natural_vector_size<float>();

        if (get_target().has_feature(Target::HVX)) {
            // The standard pattern for offloaded HVX kernels: dispatch to
            // hexagon, vectorize the inner loop at HVX width, prefetch
            // the next chunk while computing the current one.
            //
            // RoundUp tail strategy is appropriate here: the host pads
            // the output buffer to a multiple of vector_size and ignores
            // the trailing lanes (cheaper than ShiftInwards or Predicate).
            coeffs
                .hexagon()
                .vectorize(i, vector_size, TailStrategy::RoundUp);

            // Prefetch hint: next iteration's bytes are typically 1-2
            // cache lines ahead of the current ones. The DMA engine will
            // overlap the fetch with the compute.
            coeffs.prefetch(packed, i, 2);
        } else {
            // ARM host fallback for x86/CPU testing — vectorize via NEON.
            coeffs.vectorize(i, vector_size, TailStrategy::RoundUp);
        }

        // Bound estimates for the autoscheduler (when run with
        // auto_schedule=true at the generator command line).
        if (auto_schedule) {
            packed.dim(0).set_estimate(0, 64);   // largest band: ~32 bytes at hd=128
            coeffs.dim(0).set_estimate(0, 128);  // largest hd we support
            band_bits.set_estimate(5);
            max_val.set_estimate(15);
            scale.set_estimate(0.01f);
        }
    }

private:
    Var i{"i"};
};

HALIDE_REGISTER_GENERATOR(SpBandDequantize, sp_band_dequantize)

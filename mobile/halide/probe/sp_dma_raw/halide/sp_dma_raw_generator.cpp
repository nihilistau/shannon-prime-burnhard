// Shannon-Prime Mode D Stage 1 probe — Halide generator.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3.
//
// 2D variant of sp_band_dequantize_generator.cpp adapted for the Halide
// UBWCDMA pipeline. The 1D version assumes the host loops over bands
// and calls the kernel per-band; this version takes all bands as rows
// of a 2D image and lets Halide schedule across them, which lets us
// drive UBWCDMA reads/writes the same way the dma_raw_blur_rw_async
// example does.
//
// Layout:
//   input  : uint8[packed_band_stride, n_bands]  — y indexes band, x indexes byte
//   output : float[coeffs_stride, n_bands]       — y indexes band, x indexes coeff
//   bits, max_val, scale: 1D arrays indexed by y (band)
//
// This is a probe, not a production kernel: the goal is a binary answer
// to "does Halide DMA + RAW format + non-image bytes + fp32 output work
// on V69". Throughput is secondary. The schedule mirrors the upstream
// dma_raw_blur_rw_async example so we exercise the same DMA codepath.

#include "Halide.h"

using namespace Halide;

class SpDmaRawProbe : public Generator<SpDmaRawProbe> {
public:
    // Packed input bytes. dim 0 = byte offset within band, dim 1 = band index.
    Input<Buffer<uint8_t>> packed{"packed", 2};

    // Per-band metadata as 1D buffers (length == n_bands).
    Input<Buffer<float>>   scale_per_band{"scale_per_band", 1};
    Input<Buffer<int32_t>> bits_per_band{"bits_per_band", 1};
    Input<Buffer<int32_t>> max_val_per_band{"max_val_per_band", 1};

    // Output coefficients. dim 0 = coefficient within band, dim 1 = band index.
    Output<Buffer<float>>  coeffs{"coeffs", 2};

    void generate() {
        Var x{"x"}, y{"y"};

        // For each (band y, coefficient x): compute the bit field offset
        // within the packed bytes for that band's row, then extract.
        Expr band_bits = bits_per_band(y);
        Expr max_val   = max_val_per_band(y);
        Expr scale     = scale_per_band(y);

        Expr start_bit  = x * band_bits;
        Expr byte_pos   = start_bit / 8;
        Expr bit_offset = start_bit % 8;

        // Halide's bounds inference can't statically prove `byte_pos` and
        // `byte_pos + 1` stay within `packed.dim(0)` because both depend
        // on the runtime value of `band_bits`. Wrap the input with a
        // constant-exterior boundary condition so out-of-range reads
        // return 0; the host packs strictly inside the dim 0 extent so
        // this only affects the trailing window-read on the last byte
        // (where the extra byte falls past the band's packed-stride).
        Func packed_safe = BoundaryConditions::constant_exterior(packed, 0);

        // 16-bit window over two adjacent packed bytes — covers up to
        // 16-bit code widths starting at any bit offset (we use 2..5).
        Expr lo = cast<uint16_t>(packed_safe(byte_pos,     y));
        Expr hi = cast<uint16_t>(packed_safe(byte_pos + 1, y));
        Expr window = (hi << 8) | lo;

        Expr mask = cast<uint16_t>((1 << band_bits) - 1);
        Expr u    = (window >> bit_offset) & mask;

        // Sign-correct via the band's max_val bias, then to fp32 and scale.
        Expr q = cast<int16_t>(u) - cast<int16_t>(max_val);
        coeffs(x, y) = cast<float>(q) * scale;
    }

    void schedule() {
        Var x{"x"}, y{"y"};
        const int vector_size = natural_vector_size<float>();

        if (get_target().has_feature(Target::HVX)) {
            // The schedule mirrors the dma_raw_blur_rw_async example so
            // we light up the same UBWCDMA codepath. We compute coeffs
            // at the band granularity, vectorize x at HVX width, and
            // prefetch the next band while computing the current one.
            //
            // Note: we DON'T schedule the input as a hexagon_ubwcdma_read
            // here — that's done at the dsp/_run.cpp level via
            // halide_hexagon_dma_prepare_for_copy_to_host on the input
            // halide_buffer_t. The generator just operates on the
            // already-DMA'd L2 buffer.
            coeffs
                .compute_root()
                .hexagon()
                .vectorize(x, vector_size, TailStrategy::RoundUp)
                .parallel(y);

            coeffs.prefetch(packed, y, 1);
        } else {
            coeffs
                .compute_root()
                .vectorize(x, vector_size, TailStrategy::RoundUp)
                .parallel(y);
        }

        if (auto_schedule) {
            packed.dim(0).set_estimate(0, 32);   // typical packed-byte stride per band
            packed.dim(1).set_estimate(0, 4);    // typical n_bands = 4
            scale_per_band.dim(0).set_estimate(0, 4);
            bits_per_band.dim(0).set_estimate(0, 4);
            max_val_per_band.dim(0).set_estimate(0, 4);
            coeffs.dim(0).set_estimate(0, 32);   // coeffs per band
            coeffs.dim(1).set_estimate(0, 4);
        }
    }
};

HALIDE_REGISTER_GENERATOR(SpDmaRawProbe, sp_dma_raw_halide)

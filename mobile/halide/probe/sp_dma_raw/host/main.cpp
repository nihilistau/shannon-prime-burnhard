// Shannon-Prime Mode D Stage 1 probe — host driver.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3.
//
// This is the host (ARM Android) side of the SP-packed-bytes UBWCDMA
// probe. It:
//
//   1. Generates a known coefficient sequence per band.
//   2. Quantizes those coefficients to {3, 4, 5}-bit codes per the
//      SP K config (band_bits = {5, 5, 4, 3}).
//   3. Packs the codes contiguously per band (no fp16 scale header —
//      the scale travels as a separate IDL parameter in this probe).
//   4. Calls sp_dma_raw_run() over FastRPC.
//   5. Receives the fp32 dequantized output from the DSP.
//   6. Compares to a scalar reference computed locally; reports
//      max-error / RMS / pass-fail, and the timing the DSP measured.
//
// Patterned after dma_raw_blur_rw_async/host/main.cpp's verify() pattern.
//
// usage: ./test-sp-dma-raw [iterations]

// We deliberately don't include "buffer.h" / "ion_allocation.h" — those
// would route through alloc_ion(), which on modern Android (Q+) fails
// for shell-user apps because SELinux blocks direct /dev/dma_heap/qcom,system
// ioctls. We use rpcmem_alloc instead — the cdsprpc daemon (running as
// `system`) handles the dma_heap ioctl on our behalf.
#include "rpcmem.h"
#include "remote.h"            // remote_session_control + DSPRPC_CONTROL_UNSIGNED_MODULE
#include "sp_dma_raw.h"        // FastRPC stub (generated from .idl)
#include "test_report.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// SP K config we're probing. {5,5,4,3}-bit bands at hd=128 split into
// 4 equal bands of 32 coefficients each.
static constexpr int    N_BANDS              = 4;
static constexpr int    COEFFS_PER_BAND[]    = {32, 32, 32, 32};
static constexpr int    BITS_PER_BAND[]      = { 5,  5,  4,  3};
static constexpr float  SCALE_PER_BAND[]     = {0.04f, 0.02f, 0.015f, 0.01f};
// max_val[b] = (1 << (bits-1)) - 1
static constexpr int    MAX_VAL_PER_BAND[]   = {15, 15, 7, 3};

// Ground-truth coefficient generator. Uses a deterministic pattern
// (sine + small phase offset per band) so the values are spread across
// the dynamic range of each quantizer.
static float gen_coeff(int band, int i) {
    // amplitude near MAX_VAL[b] * SCALE[b] so we exercise saturation
    // without clipping it
    float amp = (MAX_VAL_PER_BAND[band] - 1) * SCALE_PER_BAND[band];
    float phase = 0.3f * band + 0.05f * i;
    return amp * std::sin(phase);
}

// Quantize a single coefficient to a code in [-max_val, +max_val].
static int quantize(float x, float scale, int max_val) {
    int q = (int)std::lround(x / scale);
    if (q >  max_val) q =  max_val;
    if (q < -max_val) q = -max_val;
    return q;
}

// Pack the codes for one band into bytes. Returns ceil(n*bits/8).
// Bit order: code i occupies bits [i*bits, (i+1)*bits) of the byte stream
// in little-endian-within-byte order, which matches what the Halide
// generator's `(window >> bit_offset) & mask` recovers.
static int pack_band(const int *codes, int n, int bits, int max_val,
                     uint8_t *out, int out_capacity) {
    const int n_bytes = (n * bits + 7) / 8;
    if (n_bytes > out_capacity) {
        std::fprintf(stderr, "pack_band: out_capacity %d < %d\n", out_capacity, n_bytes);
        return -1;
    }
    std::memset(out, 0, n_bytes);
    for (int i = 0; i < n; ++i) {
        int signed_q = codes[i];
        // Shift to unsigned [0, 2*max_val] for packing — matches the
        // Halide kernel's `q = (int)u - max_val` decode step.
        int u = signed_q + max_val;
        int start_bit = i * bits;
        int byte_pos  = start_bit / 8;
        int bit_off   = start_bit % 8;
        // 16-bit window write so we can straddle byte boundaries
        uint16_t window = (uint16_t)out[byte_pos] | ((uint16_t)out[byte_pos + 1] << 8);
        window |= (uint16_t)(u << bit_off);
        out[byte_pos]     = (uint8_t)(window & 0xFF);
        out[byte_pos + 1] = (uint8_t)((window >> 8) & 0xFF);
    }
    return n_bytes;
}

// Scalar SP dequantize reference — what the DSP output should match.
static float dequantize(int unsigned_code, int max_val, float scale) {
    int q = unsigned_code - max_val;
    return (float)q * scale;
}

int main(int argc, char *argv[]) {
    int iterations = (argc > 1) ? std::atoi(argv[1]) : 100;

    rpcmem_init();

    // Enable UNSIGNED PD on the cDSP. The phone is bootloader-locked and
    // we don't have a Qualcomm-issued testsig for it; without unsigned PD
    // the cDSP loader rejects our (unsigned) skel.so + oemconfig.so with
    // "signature verify start failed". Mirrors the working S22U test's
    // pattern in S22U_ext.c.
    {
        struct remote_rpc_control_unsigned_module data;
        data.domain = 3;  // CDSP_DOMAIN_ID
        data.enable = 1;
        int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                        (void*)&data, sizeof(data));
        if (rc != 0) {
            std::fprintf(stderr, "remote_session_control(UNSIGNED_PD) failed: 0x%x\n", rc);
            return 1;
        }
    }

    // Power on HVX + turbo
    int rc_pwr = sp_dma_raw_power_on_hvx();
    if (rc_pwr != 0) {
        std::fprintf(stderr, "power_on_hvx failed: 0x%x\n", rc_pwr);
        return 1;
    }
    sp_dma_raw_set_hvx_perf_mode_turbo();

    // Pick a packed_band_stride that's a multiple of 128 (vector size)
    // and large enough for the widest band: 32 coeffs * 5 bits = 20 bytes.
    // Pad to 128 to give the schedule room.
    const int packed_band_stride = 128;
    const int coeffs_stride      = 32;  // == max coeffs per band

    // Allocate via rpcmem. cdsprpc handles the dma_heap ioctl as `system`
    // user; direct /dev/dma_heap access from shell-user apps is SELinux-
    // blocked on production stock Android. RPCMEM_HEAP_ID_SYSTEM maps to
    // the same qcom,system dmabuf heap, just routed through the daemon.
    const int heap_id = RPCMEM_HEAP_ID_SYSTEM;
    const size_t packed_bytes = (size_t)packed_band_stride * N_BANDS;
    const size_t coeffs_bytes = (size_t)coeffs_stride * N_BANDS * sizeof(float);

    uint8_t *packed_buf =
        (uint8_t*)rpcmem_alloc(heap_id, RPCMEM_DEFAULT_FLAGS, packed_bytes);
    float *coeffs_buf =
        (float*)rpcmem_alloc(heap_id, RPCMEM_DEFAULT_FLAGS, coeffs_bytes);
    if (!packed_buf || !coeffs_buf) {
        std::fprintf(stderr, "rpcmem_alloc failed: packed=%p coeffs=%p\n",
                     packed_buf, coeffs_buf);
        return 1;
    }
    std::memset(packed_buf, 0, packed_bytes);
    std::memset(coeffs_buf, 0, coeffs_bytes);

    // Ground-truth + reference output we expect.
    std::vector<float> ref_coeffs(N_BANDS * coeffs_stride, 0.0f);
    for (int b = 0; b < N_BANDS; ++b) {
        int codes[64] = {0};
        for (int i = 0; i < COEFFS_PER_BAND[b]; ++i) {
            float x = gen_coeff(b, i);
            codes[i] = quantize(x, SCALE_PER_BAND[b], MAX_VAL_PER_BAND[b]);
            // What we EXPECT the DSP to produce after dequantize.
            ref_coeffs[b * coeffs_stride + i] =
                dequantize(codes[i] + MAX_VAL_PER_BAND[b],
                           MAX_VAL_PER_BAND[b], SCALE_PER_BAND[b]);
        }
        int n_bytes = pack_band(codes, COEFFS_PER_BAND[b],
                                BITS_PER_BAND[b], MAX_VAL_PER_BAND[b],
                                packed_buf + b * packed_band_stride,
                                packed_band_stride);
        if (n_bytes < 0) return 1;
        std::printf("band %d: %d coeffs * %d bits -> %d bytes (capacity %d)\n",
                    b, COEFFS_PER_BAND[b], BITS_PER_BAND[b],
                    n_bytes, packed_band_stride);
    }

    // Send to DSP. NB: qaic-generated `uint64` resolves to `unsigned long
    // long` per the SDK's AEEStdDef.h, while `uint64_t` from <stdint.h> is
    // `unsigned long` on aarch64 NDK (LP64 model) — same width, distinct
    // C++ type. Use unsigned long long here so the FastRPC stub signature
    // matches without a cast.
    unsigned long long avg_time = 0;
    int rc = sp_dma_raw_run(
        packed_buf, packed_band_stride * N_BANDS,
        packed_band_stride,
        N_BANDS,
        /*is_input_ubwc=*/0,
        SCALE_PER_BAND,    N_BANDS,
        BITS_PER_BAND,     N_BANDS,
        MAX_VAL_PER_BAND,  N_BANDS,
        COEFFS_PER_BAND,   N_BANDS,
        coeffs_stride,
        coeffs_buf, coeffs_stride * N_BANDS,
        iterations,
        &avg_time);

    sp_dma_raw_power_off_hvx();

    if (rc != 0) {
        std::fprintf(stderr, "PROBE FAIL: sp_dma_raw_run returned %d\n", rc);
        std::fprintf(stderr,
            "Most likely culprits in order of probability:\n"
            "  1. prepare_for_copy_to_device rejects fp32 RAW (output type)\n"
            "  2. prepare_for_copy_to_host rejects packed-bytes RAW (input layout)\n"
            "  3. UBWCDMA descriptor issue (frame size / stride alignment)\n"
            "  4. ION allocation alignment (need 128-byte aligned for HVX)\n"
            "Check FARF logs from DSP for the failing call site.\n");
        TestReport tr("sp_dma_raw_probe", 0, "microseconds",
                      Mode::Device_Standalone, Result::Fail);
        tr.print();
        rpcmem_free(packed_buf);
        rpcmem_free(coeffs_buf);
        rpcmem_deinit();
        return rc;
    }

    // Verify: max error per band, overall RMS, # mismatches > 1e-3.
    int total = N_BANDS * coeffs_stride;
    float max_err = 0.0f;
    double sse = 0.0;
    int n_mismatch = 0;
    const float TOL = 1e-5f;  // pure dequantize, no rounding loss expected
    for (int b = 0; b < N_BANDS; ++b) {
        float band_max = 0.0f;
        for (int i = 0; i < COEFFS_PER_BAND[b]; ++i) {
            int idx = b * coeffs_stride + i;
            float got = coeffs_buf[idx];
            float ref = ref_coeffs[idx];
            float err = std::fabs(got - ref);
            if (err > band_max) band_max = err;
            if (err > max_err)  max_err  = err;
            sse += (double)err * err;
            if (err > TOL) ++n_mismatch;
        }
        std::printf("band %d max_err = %.3e\n", b, band_max);
    }
    double rms = std::sqrt(sse / total);
    std::printf("overall: max_err=%.3e rms=%.3e mismatches=%d/%d\n",
                max_err, rms, n_mismatch, total);
    std::printf("DSP avg_time = %llu us over %d iterations\n",
                (unsigned long long)avg_time, iterations);

    Result r = (n_mismatch == 0) ? Result::Pass : Result::Fail;
    TestReport tr("sp_dma_raw_probe", avg_time, "microseconds",
                  Mode::Device_Standalone, r);
    tr.print();

    rpcmem_free(packed_buf);
    rpcmem_free(coeffs_buf);
    rpcmem_deinit();
    return (r == Result::Pass) ? 0 : 1;
}

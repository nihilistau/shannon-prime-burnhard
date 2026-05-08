// Shannon-Prime Mode D Stage 1 probe — DSP-side FastRPC entry point.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3.
//
// This is the cDSP side of the SP-packed-bytes UBWCDMA probe. The
// host packs SP K rows into a uint8 buffer formatted as
// `n_bands rows × packed_band_stride bytes`; we wrap it as a 2D
// halide_buffer_t, prepare it for UBWCDMA copy-to-host, allocate the
// fp32 output buffer, prepare it for UBWCDMA copy-to-device, and
// invoke the Halide pipeline.
//
// Patterned after dma_raw_blur_rw_async_run.cpp. Differences:
//   - Output is fp32 (not uint8) — output halide_buffer_t.type changes.
//   - Three additional 1D halide_buffer_t scalars carry per-band metadata.
//   - Format is hard-pinned to halide_hexagon_fmt_RawData / eDmaFmt_RawData
//     (the packed bytes have no sensible NV12/P010 interpretation).
//
// The PROBE OUTCOME (the binary thing we're answering) is whether the
// halide_hexagon_dma_prepare_for_copy_to_device call succeeds when the
// output is f32 — i.e., whether RAW format has any opinion about the
// element type, or only about the layout/stride.

#include "HAP_farf.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "hexagon_benchmark.h"
#include "hexagon_types.h"
#include "hvx_interface.h"
#include "qurt_error.h"
#include "sysmon_cachelock.h"
#include "dma_def.h"
#include "dma_types.h"
#include "dmaWrapper.h"

#include "sp_dma_raw.h"          // FastRPC skel (generated from .idl)
#include "sp_dma_raw_halide.h"   // Halide AOT pipeline

#include "HalideRuntime.h"
#include "HalideRuntimeHexagonDma.h"
#include "HalideRuntimeHexagonHost.h"
#include "qurt.h"
#include "qurt_hvx.h"
#include "ubwcdma_utils.h"
#include <stdlib.h>
#include <string.h>

#ifndef SIMULATOR
#undef FARF_LOW
#define FARF_LOW 1
#undef FARF_HIGH
#define FARF_HIGH 1
#endif
#define DESIRED_NUM_THREADS 4

int sp_dma_raw_power_on_hvx(void)            { return power_on_hvx(); }
int sp_dma_raw_power_off_hvx(void)           { return power_off_hvx(); }
int sp_dma_raw_set_hvx_perf_mode_turbo(void) { return set_hvx_perf_mode_turbo(); }

int sp_dma_raw_getLinearFrameSize(int width, int height, int stride, int /*format*/, int *size) {
    if (!size) return -1;
    *size = getFrameSize(width, height, stride, eDmaFmt_RawData, FALSE);
    return 0;
}
int sp_dma_raw_getUBWCFrameSize(int width, int height, int stride, int /*format*/, int *size) {
    if (!size) return -1;
    *size = getFrameSize(width, height, stride, eDmaFmt_RawData, TRUE);
    return 0;
}

// IDL signature (matches sp_dma_raw.idl `run`):
//   long run(in u8_buffer packed_in,
//            in long packed_band_stride,
//            in long n_bands,
//            in long is_input_ubwc,
//            in band_scales_t scale_per_band,
//            in band_int_t    bits_per_band,
//            in band_int_t    max_val_per_band,
//            in band_int_t    coeffs_per_band,
//            in long          coeffs_stride,
//            rout f32_buffer  coeffs_out,
//            in long          iterations,
//            rout unsigned long long avg_time);
//
// FastRPC IDL maps `sequence<octet>` to (uint8_t* + len), `sequence<float>`
// to (float* + len), `sequence<long>` to (int32_t* + len).
int sp_dma_raw_run(
    const uint8_t *packed_in, int packed_in_len,
    int packed_band_stride,
    int n_bands,
    int is_input_ubwc,
    const float *scale_per_band, int scale_per_band_len,
    const int *bits_per_band,    int bits_per_band_len,
    const int *max_val_per_band, int max_val_per_band_len,
    const int *coeffs_per_band,  int coeffs_per_band_len,
    int coeffs_stride,
    float *coeffs_out, int coeffs_out_len,
    int iterations,
    uint64_t *avg_time)
{
    FARF(HIGH, "sp_dma_raw_run: bands=%d stride=%d ubwc=%d iters=%d\n",
         n_bands, packed_band_stride, is_input_ubwc, iterations);

    if (scale_per_band_len   != n_bands ||
        bits_per_band_len    != n_bands ||
        max_val_per_band_len != n_bands ||
        coeffs_per_band_len  != n_bands) {
        FARF(HIGH, "sp_dma_raw_run: per-band metadata length mismatch\n");
        return -1;
    }
    if (packed_in_len < n_bands * packed_band_stride) {
        FARF(HIGH, "sp_dma_raw_run: packed_in too small (%d < %d*%d)\n",
             packed_in_len, n_bands, packed_band_stride);
        return -1;
    }
    if (coeffs_out_len < n_bands * coeffs_stride) {
        FARF(HIGH, "sp_dma_raw_run: coeffs_out too small\n");
        return -1;
    }

    const int width  = packed_band_stride;   // RAW frame width  (bytes)
    const int height = n_bands;              // RAW frame height (rows)
    (void)width; (void)height;

    // ------------------------------------------------------------------
    // ALL declarations hoisted to top so the cleanup gotos don't cross
    // any C++ object initializers (gcc/clang reject that as ill-formed).
    // ------------------------------------------------------------------
    halide_buffer_t input  = {0};
    halide_buffer_t output = {0};
    halide_buffer_t scale_buf = {0}, bits_buf = {0}, max_val_buf = {0};
    halide_dimension_t in_dim[2]  = {{0, packed_band_stride, 1},
                                     {0, n_bands, packed_band_stride}};
    halide_dimension_t out_dim[2] = {{0, coeffs_stride, 1},
                                     {0, n_bands, coeffs_stride}};
    halide_dimension_t s_dim[1]   = {{0, n_bands, 1}};
    halide_dimension_t b_dim[1]   = {{0, n_bands, 1}};
    halide_dimension_t m_dim[1]   = {{0, n_bands, 1}};
    void *dma_engine = nullptr;
    int   nRet       = 0;
    int   result     = 0;
    int   old_threads = 0;
    bool  input_prepared  = false;
    bool  output_prepared = false;
    bool  input_wrapped   = false;
    bool  output_wrapped  = false;

    // ------------------------------------------------------------------
    // 1. Wrap the packed input as a 2D halide_buffer_t (uint8, RAW).
    // ------------------------------------------------------------------
    input.type = halide_type_t(halide_type_uint, 8, 1);
    input.dimensions = 2;
    input.dim = in_dim;

    output.type = halide_type_t(halide_type_float, 32, 1);
    output.dimensions = 2;
    output.dim = out_dim;

    scale_buf.type   = halide_type_t(halide_type_float, 32, 1);
    bits_buf.type    = halide_type_t(halide_type_int,   32, 1);
    max_val_buf.type = halide_type_t(halide_type_int,   32, 1);
    scale_buf.dimensions = bits_buf.dimensions = max_val_buf.dimensions = 1;
    scale_buf.dim   = s_dim;
    bits_buf.dim    = b_dim;
    max_val_buf.dim = m_dim;
    scale_buf.host   = (uint8_t*)scale_per_band;
    bits_buf.host    = (uint8_t*)bits_per_band;
    max_val_buf.host = (uint8_t*)max_val_per_band;

    nRet = halide_hexagon_dma_allocate_engine(0, &dma_engine);
    if (nRet) { FARF(HIGH, "dma_allocate_engine: %d\n", nRet); return nRet; }
    halide_hexagon_dma_power_mode_voting(0, halide_hexagon_power_nominal);

    nRet = halide_hexagon_dma_device_wrap_native(0, &input, (uint64_t)packed_in);
    if (nRet) { FARF(HIGH, "wrap_native(in): %d\n", nRet); goto cleanup; }
    input_wrapped = true;
    input.flags |= halide_buffer_flag_device_dirty;

    nRet = halide_hexagon_dma_prepare_for_copy_to_host(
        0, &input, dma_engine, (bool)is_input_ubwc, halide_hexagon_fmt_RawData);
    if (nRet) { FARF(HIGH, "prepare_to_host: %d\n", nRet); goto cleanup; }
    input_prepared = true;

    // ------------------------------------------------------------------
    // 2. Wrap the fp32 output. THIS is the probe — does
    //    prepare_for_copy_to_device accept a 4-byte-per-element RAW
    //    frame, or does it complain about layout assumptions?
    // ------------------------------------------------------------------
    nRet = halide_hexagon_dma_device_wrap_native(0, &output, (uint64_t)coeffs_out);
    if (nRet) { FARF(HIGH, "wrap_native(out): %d\n", nRet); goto cleanup; }
    output_wrapped = true;
    output.flags |= halide_buffer_flag_device_dirty;

    nRet = halide_hexagon_dma_prepare_for_copy_to_device(
        0, &output, dma_engine, /*is_ubwc=*/false, halide_hexagon_fmt_RawData);
    if (nRet) {
        FARF(HIGH, "PROBE RESULT: prepare_to_device(fp32 RAW) FAILED: %d\n", nRet);
        goto cleanup;
    }
    output_prepared = true;
    FARF(HIGH, "PROBE RESULT: prepare_to_device(fp32 RAW) succeeded\n");

    // ------------------------------------------------------------------
    // 3. Run the pipeline. Time it.
    // ------------------------------------------------------------------
    old_threads = halide_set_num_threads(DESIRED_NUM_THREADS);
    *avg_time = benchmark(iterations, [&]() {
        result = sp_dma_raw_halide(
            &input, &scale_buf, &bits_buf, &max_val_buf, &output);
        if (result != 0) {
            FARF(HIGH, "sp_dma_raw_halide: %d\n", result);
        }
    });
    halide_set_num_threads(old_threads);

cleanup:
    if (output_prepared) halide_hexagon_dma_unprepare(0, &output);
    if (output_wrapped)  halide_hexagon_dma_device_detach_native(0, &output);
    if (input_prepared)  halide_hexagon_dma_unprepare(0, &input);
    if (input_wrapped)   halide_hexagon_dma_device_detach_native(0, &input);
    if (dma_engine)      halide_hexagon_dma_deallocate_engine(0, dma_engine);
    halide_hexagon_dma_deinit(0);

    FARF(HIGH, "sp_dma_raw_run: nRet=%d result=%d avg_time=%llu\n",
         nRet, result, (unsigned long long)*avg_time);
    return (nRet != 0) ? nRet : result;
}

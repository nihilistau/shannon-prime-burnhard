// Shannon-Prime VHT2 - Hexagon DSP scaffold (FastRPC interface impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Implements the IDL methods declared in inc/sp_hex.idl. The qaic-generated
// sp_hex_skel.c calls these functions on the DSP side after unmarshalling
// FastRPC arguments. Forked from the Hexagon SDK 5.5.6.0 S22U sample
// (Qualcomm copyright on the lifecycle pattern; SP-specific math is AGPLv3).

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "HAP_farf.h"
#include "HAP_vtcm_mgr.h"      // HAP_request_VTCM / HAP_release_VTCM / queries
#include "HAP_perf.h"          // HAP_perf_get_pcycles for the bench IDL
#include "qurt_hvx.h"          // qurt_hvx_lock / unlock ? required before HVX
#include "sp_hex.h"            // qaic-generated header from sp_hex.idl
#include "sp_hex_kernels.h"    // forwards to SP math core
#include "shannon_prime.h"     // sp_band_config_t / sp_band_quantize / etc.
// Phase 7B: dmaWrapper.h types for the raw DMA probe.
// The symbols (hDmaWrapper_AllocDma, nDmaWrapper_*) live in ubwcdma_dynlib.so
// on the DSP. Declared #pragma weak so the skel loads even when that lib is
// absent (as on S22U production). sp_hex_probe_dma_raw returns -1 if any
// weak symbol resolves to NULL.
#include "compat/dmaWrapper.h"
#pragma weak hDmaWrapper_AllocDma
#pragma weak nDmaWrapper_FreeDma
#pragma weak nDmaWrapper_Move
#pragma weak nDmaWrapper_Wait
#pragma weak nDmaWrapper_DmaTransferSetup

// Forward decl ? defined in sp_hex_kernels_hvx.c when HVX is available.
// Used directly by the bench IDL to time the HVX path in isolation,
// vs sp_hex_vht2_f32 which dispatches and would obscure the comparison.
#ifdef __HVX__
void sp_hex_vht2_f32_hvx(float *data, int n);
void sp_hex_vht2_f32_hvx_qf32(float *data, int n);
#endif

// Per-session DSP context. Held behind the FastRPC handle that sp_hex_open
// returns; sp_hex_close releases everything. Sized to fit a few small
// pieces of state ? VTCM pointer, region size, anything else the kernels
// will need at the session level.
//
// VTCM is V69's tightly-coupled SRAM (~8 MB total on this DSP variant).
// Acquiring a small region per session ensures HVX kernels have a fast
// scratch path; HAP_request_VTCM returning NULL means the kernels will
// fall back to DDR, with predictably worse perf.
typedef struct {
    void   *vtcm_ptr;       // NULL if acquisition failed
    int     vtcm_bytes;     // 0 if no region; else bytes acquired
    int     hvx_locked;     // 1 if qurt_hvx_lock succeeded ? required
                            //   before any HVX instruction executes on
                            //   this thread; without it, vector ops
                            //   fault and FastRPC returns a transport
                            //   error code (the 78 / -2147482611 / 39
                            //   chaos we saw before locking).
} sp_hex_session_t;

// Default per-session VTCM ask. Banded quantize working set on a typical
// head_dim is well under 16 KB; 64 KB gives headroom for staging packed
// bytes + scale headers + a few HVX register spills if needed. Tunable
// once we have actual HVX kernels showing pressure.
#define SP_HEX_VTCM_BYTES   (64 * 1024)

// ============================================================================
// Lifecycle - inherited from remote_handle64.
// ============================================================================

int sp_hex_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    sp_hex_session_t *sess = (sp_hex_session_t *)calloc(1, sizeof(*sess));
    if (!sess) return -1;

    // Lock the HVX context for this thread. Required before any HVX
    // instruction executes ? without this lock, the first vector op
    // throws a fault and FastRPC returns a transport error to the host.
    // V69 is 128-byte (1024-bit) HVX so we ask for QURT_HVX_MODE_128B.
    // 0 on success; negative if HVX is unavailable for any reason.
    int hvx_rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    sess->hvx_locked = (hvx_rc == 0);

    // Try to acquire a VTCM region. single_page_flag=1 asks for one
    // contiguous physical page ? fine for 64 KB which is well under
    // the V69 page granularity. NULL return means "couldn't get one";
    // the session is still usable, kernels just run against DDR.
    sess->vtcm_ptr = HAP_request_VTCM(SP_HEX_VTCM_BYTES, /*single_page=*/1);
    sess->vtcm_bytes = (sess->vtcm_ptr != NULL) ? SP_HEX_VTCM_BYTES : 0;

    *handle = (remote_handle64)sess;
    FARF(RUNTIME_HIGH,
         "[sp_hex] open() -> handle=0x%llx hvx_locked=%d vtcm=%p (%d bytes)",
         *handle, sess->hvx_locked, sess->vtcm_ptr, sess->vtcm_bytes);
    return 0;
}

int sp_hex_close(remote_handle64 handle) {
    sp_hex_session_t *sess = (sp_hex_session_t *)handle;
    if (sess) {
        if (sess->vtcm_ptr) {
            HAP_release_VTCM(sess->vtcm_ptr);
        }
        if (sess->hvx_locked) {
            qurt_hvx_unlock();
        }
        free(sess);
    }
    FARF(RUNTIME_HIGH, "[sp_hex] close()");
    return 0;
}

int sp_hex_vtcm_status(remote_handle64 h,
                        long long *bytes_total,
                        long long *bytes_avail,
                        long long *bytes_acquired) {
    sp_hex_session_t *sess = (sp_hex_session_t *)h;
    if (!bytes_total || !bytes_avail || !bytes_acquired) return -1;

    unsigned int page_size = 0, page_count = 0;
    if (HAP_query_total_VTCM(&page_size, &page_count) == 0) {
        *bytes_total = (long long)page_size * (long long)page_count;
    } else {
        *bytes_total = 0;
    }

    unsigned int avail_block = 0, max_page = 0, num_pages = 0;
    if (HAP_query_avail_VTCM(&avail_block, &max_page, &num_pages) == 0) {
        // avail_block is the largest contiguous request that would
        // succeed right now; close enough to "avail" for our purposes.
        *bytes_avail = (long long)avail_block;
    } else {
        *bytes_avail = 0;
    }

    *bytes_acquired = sess ? (long long)sess->vtcm_bytes : 0;
    FARF(RUNTIME_HIGH,
         "[sp_hex] vtcm_status: total=%lld avail=%lld acquired=%lld",
         *bytes_total, *bytes_avail, *bytes_acquired);
    return 0;
}

int sp_hex_vht2_bench(remote_handle64 h, int head_dim, int iterations,
                       long long *scalar_pcycles, long long *hvx_pcycles) {
    (void)h;
    if (!scalar_pcycles || !hvx_pcycles) return -1;
    if (head_dim < 8 || head_dim > 1024) return -1;
    if ((head_dim & (head_dim - 1)) != 0) return -1;  // pow2 only
    if (iterations < 1) return -1;

    // Cycle bench ? scalar reference vs HVX-qf32 (the numerically correct
    // HVX path). The IEEE-HVX kernel is structurally broken on V69 so we
    // time qf32; user can A/B by editing the kernel call below.
    float input[1024] __attribute__((aligned(128)));
    float buf[1024]   __attribute__((aligned(128)));
    for (int i = 0; i < head_dim; ++i) {
        input[i] = 0.125f + (float)i / (float)head_dim;
    }

    // Scalar timing
    memcpy(buf, input, sizeof(float) * head_dim);
    sp_vht2_forward_f32(buf, head_dim);  // warmup
    uint64_t t0 = HAP_perf_get_pcycles();
    for (int it = 0; it < iterations; ++it) {
        memcpy(buf, input, sizeof(float) * head_dim);
        sp_vht2_forward_f32(buf, head_dim);
    }
    *scalar_pcycles = (long long)(HAP_perf_get_pcycles() - t0);

    // HVX-qf32 timing
#ifdef __HVX__
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
        memcpy(buf, input, sizeof(float) * head_dim);
        sp_hex_vht2_f32_hvx_qf32(buf, head_dim);  // warmup
        t0 = HAP_perf_get_pcycles();
        for (int it = 0; it < iterations; ++it) {
            memcpy(buf, input, sizeof(float) * head_dim);
            sp_hex_vht2_f32_hvx_qf32(buf, head_dim);
        }
        *hvx_pcycles = (long long)(HAP_perf_get_pcycles() - t0);
        qurt_hvx_unlock();
    } else {
        *hvx_pcycles = *scalar_pcycles;
    }
#else
    *hvx_pcycles = *scalar_pcycles;
#endif
    return 0;
}

// ============================================================================
// IDL methods.
// ============================================================================

// Full SP round-trip on the cDSP:
//   in_vec  --VHT2-->  coeffs  --quantize-->  packed bytes
//   packed  --dequantize-->  coeffs  --VHT2-->  out_vec
// VHT2 is self-inverse so the second VHT2 is the inverse. Reconstruction
// error matches what the math core produces on CPU for the same input ?
// not fp32 epsilon (that's the no-quantize path).
int sp_hex_round_trip_f32(remote_handle64 h,
                           const float *in_vec, int in_len,
                           int head_dim,
                           float *out_vec, int out_len) {
    (void)h;
    if (in_len != head_dim || out_len != head_dim) {
        FARF(ERROR, "[sp_hex] round_trip: length mismatch in=%d out=%d hd=%d",
             in_len, out_len, head_dim);
        return -1;
    }
    if (head_dim < 8 || (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] round_trip: head_dim=%d must be pow2 and >=8",
             head_dim);
        return -1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);

    if (bc.total_bytes > 4096) {
        FARF(ERROR, "[sp_hex] round_trip: packed size %d > scratch 4096",
             bc.total_bytes);
        return -1;
    }
    // 128-byte aligned for HVX vmem (the dispatcher routes large enough
    // head_dim through HVX and the kernels do strict-alignment loads).
    unsigned char packed[4096] __attribute__((aligned(128)));
    float coeffs[1024]         __attribute__((aligned(128)));
    if (head_dim > 1024) {
        FARF(ERROR, "[sp_hex] round_trip: head_dim=%d > scratch 1024",
             head_dim);
        return -1;
    }

    // 1. Copy in, VHT2 forward via dispatcher (currently scalar ? HVX
    // VHT2 has a precision drift issue noted in sp_hex_kernels.c).
    memcpy(coeffs, in_vec, sizeof(float) * head_dim);
    sp_hex_vht2_f32(coeffs, head_dim);

    // 2. Quantize via dispatcher ? HVX path activates for head_dim ? 128
    // multiple-of-128, ternary mask zero (the default).
    int packed_used = 0;
    sp_hex_band_quantize_scalar(coeffs, head_dim, packed, sizeof(packed),
                                &packed_used);

    // 3. Dequantize back to coeffs (still scalar ? HVX dequantize TODO).
    sp_band_dequantize(packed, coeffs, &bc);

    // 4. VHT2 inverse via dispatcher.
    sp_hex_vht2_f32(coeffs, head_dim);
    memcpy(out_vec, coeffs, sizeof(float) * head_dim);

    FARF(RUNTIME_HIGH,
         "[sp_hex] round_trip: hd=%d packed=%d in[0]=%f out[0]=%f",
         head_dim, bc.total_bytes, in_vec[0], out_vec[0]);
    return 0;
}

int sp_hex_vht2_forward(remote_handle64 h, float *data, int data_len, int n) {
    (void)h;
    if (data_len != n) {
        FARF(ERROR, "[sp_hex] vht2_forward: data_len %d != n %d", data_len, n);
        return -1;
    }
    sp_hex_vht2_f32(data, n);
    return 0;
}

int sp_hex_band_quantize(remote_handle64 h,
                          const float *coeffs, int coeffs_len,
                          int head_dim,
                          unsigned char *out, int out_capacity) {
    (void)h;
    if (coeffs_len != head_dim) return -1;
    int written = 0;
    return sp_hex_band_quantize_scalar(coeffs, head_dim, out, out_capacity,
                                       &written);
}

int sp_hex_band_dequantize(remote_handle64 h,
                            const unsigned char *in, int in_len,
                            int head_dim, int max_bands,
                            float *out_coeffs, int out_len) {
    (void)h;
    if (out_len != head_dim) return -1;
    return sp_hex_band_dequantize_scalar(in, in_len, head_dim, max_bands,
                                          out_coeffs);
}

// compress_f32 ? full encode pipeline in one FastRPC dispatch.
// Mirrors the first half of sp_hex_round_trip_f32 (memcpy, VHT2, quantize)
// without the dequantize+inverse-VHT2 tail. Used by the host's
// sp_hexagon_cache_write_{k,v} to land a per-position raw fp32 vector
// into compressed bytes in one call.
//
// The packed_used out-param tells the host exactly how many bytes were
// written into out_packed. Today this is bc.total_bytes for the {5,5,4,3}
// ship config ? fixed for a given head_dim ? but exposing it explicitly
// future-proofs against per-call band-config changes.
int sp_hex_compress_f32(remote_handle64 h,
                         const float *in_vec, int in_len,
                         int head_dim,
                         unsigned char *out_packed, int out_capacity,
                         int *packed_used) {
    (void)h;
    if (!packed_used) return -1;
    *packed_used = 0;
    if (in_len != head_dim) {
        FARF(ERROR, "[sp_hex] compress: length mismatch in=%d hd=%d",
             in_len, head_dim);
        return -1;
    }
    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] compress: head_dim=%d must be pow2 in [8,1024]",
             head_dim);
        return -1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    if (bc.total_bytes > out_capacity) {
        FARF(ERROR, "[sp_hex] compress: packed=%d > capacity=%d",
             bc.total_bytes, out_capacity);
        return -1;
    }

    // 128-byte aligned for HVX vmem (the dispatcher routes large enough
    // head_dim through HVX and the kernels do strict-alignment loads).
    float coeffs[1024] __attribute__((aligned(128)));

    memcpy(coeffs, in_vec, sizeof(float) * head_dim);
    sp_hex_vht2_f32(coeffs, head_dim);

    int written = 0;
    int rc = sp_hex_band_quantize_scalar(coeffs, head_dim, out_packed,
                                          out_capacity, &written);
    if (rc != 0) {
        FARF(ERROR, "[sp_hex] compress: band_quantize rc=%d", rc);
        return rc;
    }
    *packed_used = written;
    return 0;
}

// decompress_f32 ? full decode pipeline in one FastRPC dispatch.
// Mirrors the second half of sp_hex_round_trip_f32 (band_dequantize,
// then VHT2 self-inverse). Used by the host's sp_hexagon_cache_read_*
// wrappers. max_bands < 0 (or >= n_bands) means "all bands"; 0 <=
// max_bands < n_bands triggers the partial-fidelity reconstruction
// path used by the phase 3 attention short-circuit.
int sp_hex_decompress_f32(remote_handle64 h,
                           const unsigned char *packed_in, int packed_len,
                           int head_dim, int max_bands,
                           float *out_vec, int out_len) {
    (void)h;
    if (out_len != head_dim) {
        FARF(ERROR, "[sp_hex] decompress: out_len=%d != head_dim=%d",
             out_len, head_dim);
        return -1;
    }
    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] decompress: head_dim=%d must be pow2 in [8,1024]",
             head_dim);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));

    int rc = sp_hex_band_dequantize_scalar(packed_in, packed_len, head_dim,
                                            max_bands, coeffs);
    if (rc != 0) {
        FARF(ERROR, "[sp_hex] decompress: band_dequantize rc=%d", rc);
        return rc;
    }
    // VHT2 is self-inverse ? applying it again returns to the original
    // basis. The HVX-qf32 kernel is bit-equivalent to scalar within
    // fp32 epsilon, so this matches the round-trip path's reconstruction.
    sp_hex_vht2_f32(coeffs, head_dim);
    memcpy(out_vec, coeffs, sizeof(float) * head_dim);
    return 0;
}

// compress_f32_batch ? loop the compress_f32 kernel internally over
// n_vectors so the host pays one FastRPC dispatch per chunk instead
// of one per position. Used by the post-decode hook on prefill, where
// the bridge sees a contiguous run of K/V positions in a single batch.
//
// Layout matches the IDL contract:
//   in_vecs    = n_vectors * head_dim contiguous fp32
//   out_packed = n_vectors * bc.total_bytes contiguous octets
//   packed_used = n_vectors * bc.total_bytes on success
//
// Validation rules: in_len == n_vectors * head_dim, out_capacity >=
// n_vectors * bc.total_bytes. n_vectors must be > 0; an empty batch is
// rejected so the host's chunking logic can't silently drop work.
int sp_hex_compress_f32_batch(remote_handle64 h,
                               const float *in_vecs, int in_len,
                               int head_dim, int n_vectors,
                               unsigned char *out_packed, int out_capacity,
                               int *packed_used) {
    (void)h;
    if (!packed_used) return -1;
    *packed_used = 0;
    if (n_vectors <= 0) {
        FARF(ERROR, "[sp_hex] compress_batch: n_vectors=%d <= 0", n_vectors);
        return -1;
    }
    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] compress_batch: head_dim=%d must be pow2 in [8,1024]",
             head_dim);
        return -1;
    }
    if (in_len != n_vectors * head_dim) {
        FARF(ERROR, "[sp_hex] compress_batch: in_len=%d != n_vectors=%d * hd=%d",
             in_len, n_vectors, head_dim);
        return -1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    int per_vec_bytes = bc.total_bytes;
    int total_needed = n_vectors * per_vec_bytes;
    if (total_needed > out_capacity) {
        FARF(ERROR, "[sp_hex] compress_batch: needed=%d > capacity=%d",
             total_needed, out_capacity);
        return -1;
    }

    // Stack scratch reused across iterations. 1024-float aligned coeffs
    // covers head_dim ? 1024 (any production model). The packed stage is
    // written directly into the per-vector offset of out_packed so we
    // don't need a separate per-iter packed scratch.
    float coeffs[1024] __attribute__((aligned(128)));

    for (int i = 0; i < n_vectors; ++i) {
        const float *in_vec = in_vecs + (size_t)i * (size_t)head_dim;
        unsigned char *out_slot = out_packed + (size_t)i * (size_t)per_vec_bytes;

        memcpy(coeffs, in_vec, sizeof(float) * head_dim);
        sp_hex_vht2_f32(coeffs, head_dim);

        int written = 0;
        int rc = sp_hex_band_quantize_scalar(coeffs, head_dim, out_slot,
                                              per_vec_bytes, &written);
        if (rc != 0) {
            FARF(ERROR, "[sp_hex] compress_batch: vec[%d] band_quantize rc=%d",
                 i, rc);
            return rc;
        }
        if (written != per_vec_bytes) {
            FARF(ERROR, "[sp_hex] compress_batch: vec[%d] short write %d != %d",
                 i, written, per_vec_bytes);
            return -1;
        }
    }
    *packed_used = total_needed;
    return 0;
}

// decompress_f32_batch ? loop the decompress_f32 kernel internally
// over n_vectors. max_bands applies to every vector in the batch (the
// host caller is responsible for grouping equal-fidelity reads into
// one call; mixed-fidelity batches would need a per-vector max_bands
// array, which today's host paths don't need).
int sp_hex_decompress_f32_batch(remote_handle64 h,
                                 const unsigned char *packed_in, int packed_len,
                                 int head_dim, int n_vectors, int max_bands,
                                 float *out_vecs, int out_len) {
    (void)h;
    if (n_vectors <= 0) {
        FARF(ERROR, "[sp_hex] decompress_batch: n_vectors=%d <= 0", n_vectors);
        return -1;
    }
    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] decompress_batch: head_dim=%d must be pow2 in [8,1024]",
             head_dim);
        return -1;
    }
    if (out_len != n_vectors * head_dim) {
        FARF(ERROR, "[sp_hex] decompress_batch: out_len=%d != n_vectors=%d * hd=%d",
             out_len, n_vectors, head_dim);
        return -1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    int per_vec_bytes = bc.total_bytes;
    if (packed_len != n_vectors * per_vec_bytes) {
        FARF(ERROR, "[sp_hex] decompress_batch: packed_len=%d != n=%d * pv=%d",
             packed_len, n_vectors, per_vec_bytes);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));

    for (int i = 0; i < n_vectors; ++i) {
        const unsigned char *in_slot = packed_in + (size_t)i * (size_t)per_vec_bytes;
        float *out_vec = out_vecs + (size_t)i * (size_t)head_dim;

        int rc = sp_hex_band_dequantize_scalar(in_slot, per_vec_bytes, head_dim,
                                                max_bands, coeffs);
        if (rc != 0) {
            FARF(ERROR, "[sp_hex] decompress_batch: vec[%d] dequant rc=%d", i, rc);
            return rc;
        }
        sp_hex_vht2_f32(coeffs, head_dim);
        memcpy(out_vec, coeffs, sizeof(float) * head_dim);
    }
    return 0;
}


// ============================================================================
// V-specific compress/decompress (1 band, 3 bits — different from K's 4-band
// 5/5/4/3 config). The kernels are identical to the K path but use V's bc.
// ============================================================================

int sp_hex_compress_f32_v(remote_handle64 h,
                           const float *in_vec, int in_len,
                           int head_dim,
                           unsigned char *out_packed, int out_capacity,
                           int *packed_used) {
    (void)h;
    if (!packed_used) return -1;
    *packed_used = 0;
    if (in_len != head_dim) return -1;
    if (head_dim < 8 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) return -1;

    sp_band_config_t bc;
    int v_bits[1] = {3};
    sp_band_config_init(&bc, head_dim, 1, v_bits);
    if (bc.total_bytes > out_capacity) {
        FARF(ERROR, "[sp_hex] compress_v: packed=%d > capacity=%d",
             bc.total_bytes, out_capacity);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));
    memcpy(coeffs, in_vec, sizeof(float) * head_dim);
    sp_hex_vht2_f32(coeffs, head_dim);

    // Use math core sp_band_quantize directly (takes bc), bypassing
    // sp_hex_band_quantize_scalar which has K's config hard-coded via
    // sp_hex_default_band_config. V's 1-band 3-bit config doesn't satisfy
    // HVX preconditions (head_dim >= 128 && % 128 == 0) anyway, so
    // scalar-only is fine here.
    sp_band_quantize(coeffs, out_packed, &bc);
    *packed_used = bc.total_bytes;
    return 0;
}

int sp_hex_decompress_f32_v(remote_handle64 h,
                             const unsigned char *packed_in, int packed_len,
                             int head_dim, int max_bands,
                             float *out_vec, int out_len) {
    (void)h;
    if (out_len != head_dim) return -1;
    if (head_dim < 8 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) return -1;

    sp_band_config_t bc;
    int v_bits[1] = {3};
    sp_band_config_init(&bc, head_dim, 1, v_bits);
    if (packed_len < bc.total_bytes) {
        FARF(ERROR, "[sp_hex] decompress_v: packed_len=%d < expected=%d",
             packed_len, bc.total_bytes);
        return -1;
    }
    float coeffs[1024] __attribute__((aligned(128)));
    if (max_bands < 0 || max_bands >= bc.n_bands) {
        sp_band_dequantize(packed_in, coeffs, &bc);
    } else {
        sp_band_dequantize_partial(packed_in, coeffs, &bc, max_bands);
    }
    sp_hex_vht2_f32(coeffs, head_dim);
    memcpy(out_vec, coeffs, sizeof(float) * head_dim);
    return 0;
}

int sp_hex_compress_f32_v_batch(remote_handle64 h,
                                 const float *in_vecs, int in_len,
                                 int head_dim, int n_vectors,
                                 unsigned char *out_packed, int out_capacity,
                                 int *packed_used) {
    (void)h;
    if (!packed_used) return -1;
    *packed_used = 0;
    if (n_vectors <= 0) return -1;
    if (head_dim < 8 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) return -1;
    if (in_len != n_vectors * head_dim) return -1;

    sp_band_config_t bc;
    int v_bits[1] = {3};
    sp_band_config_init(&bc, head_dim, 1, v_bits);
    int per_vec_bytes = bc.total_bytes;
    int total_needed = n_vectors * per_vec_bytes;
    if (total_needed > out_capacity) {
        FARF(ERROR, "[sp_hex] compress_v_batch: needed=%d > cap=%d",
             total_needed, out_capacity);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));
    for (int i = 0; i < n_vectors; ++i) {
        const float *in_vec = in_vecs + (size_t)i * (size_t)head_dim;
        unsigned char *out_slot = out_packed + (size_t)i * (size_t)per_vec_bytes;
        memcpy(coeffs, in_vec, sizeof(float) * head_dim);
        sp_hex_vht2_f32(coeffs, head_dim);

        sp_band_quantize(coeffs, out_slot, &bc);
    }
    *packed_used = total_needed;
    return 0;
}

int sp_hex_decompress_f32_v_batch(remote_handle64 h,
                                   const unsigned char *packed_in, int packed_len,
                                   int head_dim, int n_vectors, int max_bands,
                                   float *out_vecs, int out_len) {
    (void)h;
    if (n_vectors <= 0) return -1;
    if (head_dim < 8 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) return -1;
    if (out_len != n_vectors * head_dim) return -1;

    sp_band_config_t bc;
    int v_bits[1] = {3};
    sp_band_config_init(&bc, head_dim, 1, v_bits);
    int per_vec_bytes = bc.total_bytes;
    if (packed_len != n_vectors * per_vec_bytes) {
        FARF(ERROR, "[sp_hex] decompress_v_batch: packed_len=%d != n=%d * pv=%d",
             packed_len, n_vectors, per_vec_bytes);
        return -1;
    }

    float coeffs[1024] __attribute__((aligned(128)));
    for (int i = 0; i < n_vectors; ++i) {
        const unsigned char *in_slot = packed_in + (size_t)i * (size_t)per_vec_bytes;
        float *out_vec = out_vecs + (size_t)i * (size_t)head_dim;

        if (max_bands < 0 || max_bands >= bc.n_bands) {
            sp_band_dequantize(in_slot, coeffs, &bc);
        } else {
            sp_band_dequantize_partial(in_slot, coeffs, &bc, max_bands);
        }
        sp_hex_vht2_f32(coeffs, head_dim);
        memcpy(out_vec, coeffs, sizeof(float) * head_dim);
    }
    return 0;
}

// ============================================================================
// kq_matmul_fused — Phase 1.6/1.7 harness intercept replacement.
// ============================================================================
//
// HVX-accelerated path on V69+. Composes:
//   1. scalar band_dequantize (small per-row work — n_kv decompresses)
//   2. HVX qf32 VHT2 forward (self-inverse) — sp_hex_vht2_f32_hvx_qf32
//   3. HVX qf32 dot product — sp_hex_dot_f32_hvx
//
// The hot loop is dot product (n_kv × n_q operations vs n_kv decompresses).
// HVX runs 32 fp32 lanes per cycle; at head_dim=128 that's 4 fmas + a
// butterfly reduce per dot. The decompressed K row is reused across all
// Q heads — the "fused" in the name.

#ifdef __HVX__
extern void sp_hex_vht2_f32_hvx_qf32(float *data, int n);
extern void sp_hex_dot_f32_hvx(const float *a, const float *b, int head_dim,
                                float *out);
#endif

int sp_hex_kq_matmul_fused(remote_handle64 h,
                            const float *q_vec, int q_len,
                            const unsigned char *k_packed, int packed_len,
                            int head_dim, int n_kv, int n_q,
                            float *kq_scores, int kq_len) {
    (void)h;

    if (head_dim < 8 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        FARF(ERROR, "[sp_hex] kq_fused: bad head_dim=%d", head_dim);
        return -1;
    }
    if (n_kv <= 0 || n_q <= 0) {
        FARF(ERROR, "[sp_hex] kq_fused: bad n_kv=%d n_q=%d", n_kv, n_q);
        return -1;
    }
    if (q_len != n_q * head_dim) {
        FARF(ERROR, "[sp_hex] kq_fused: q_len=%d != n_q=%d * hd=%d",
             q_len, n_q, head_dim);
        return -1;
    }
    if (kq_len != n_kv * n_q) {
        FARF(ERROR, "[sp_hex] kq_fused: kq_len=%d != n_kv=%d * n_q=%d",
             kq_len, n_kv, n_q);
        return -1;
    }

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    int per_pos_bytes = bc.total_bytes;
    if (packed_len != n_kv * per_pos_bytes) {
        FARF(ERROR, "[sp_hex] kq_fused: packed_len=%d != n_kv=%d * pp=%d",
             packed_len, n_kv, per_pos_bytes);
        return -1;
    }

    // Per-row scratch — 128-byte aligned for HVX vmem loads.
    float k_row[1024] __attribute__((aligned(128)));
    float q_aligned_buf[1024] __attribute__((aligned(128)));

#ifdef __HVX__
    // HVX requires per-thread lock. FastRPC dispatchers can rotate methods
    // across worker threads, so locking in sp_hex_open doesn't reliably
    // carry. Lock here per call. Failure means HVX is unavailable; we
    // fall back to scalar so the math is still right.
    const int hvx_ok = (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0);
    const int hd_vec_ok = (head_dim >= 32) && ((head_dim & 31) == 0);
    const int q_aligned = ((((uintptr_t)q_vec) & 127) == 0);
#else
    const int hvx_ok = 0;
    const int hd_vec_ok = 0;
    const int q_aligned = 0;
#endif

    // kv outer, q inner — reuses decompressed K row across all Q heads.
    for (int kv = 0; kv < n_kv; ++kv) {
        const unsigned char *packed = k_packed + (size_t)kv * (size_t)per_pos_bytes;

        int rc = sp_hex_band_dequantize_scalar(packed, per_pos_bytes, head_dim,
                                                /*max_bands=*/-1, k_row);
        if (rc != 0) {
            FARF(ERROR, "[sp_hex] kq_fused: dequant rc=%d kv=%d", rc, kv);
#ifdef __HVX__
            if (hvx_ok) qurt_hvx_unlock();
#endif
            return rc;
        }

#ifdef __HVX__
        if (hvx_ok && hd_vec_ok) {
            sp_hex_vht2_f32_hvx_qf32(k_row, head_dim);
        } else
#endif
        {
            sp_hex_vht2_f32(k_row, head_dim);
        }

        for (int q = 0; q < n_q; ++q) {
            const float *q_row = q_vec + (size_t)q * (size_t)head_dim;
            float dot = 0.0f;
#ifdef __HVX__
            if (hvx_ok && hd_vec_ok) {
                const float *q_in = q_row;
                if (!q_aligned) {
                    memcpy(q_aligned_buf, q_row, sizeof(float) * head_dim);
                    q_in = q_aligned_buf;
                }
                sp_hex_dot_f32_hvx(q_in, k_row, head_dim, &dot);
            } else
#endif
            {
                for (int i = 0; i < head_dim; ++i) {
                    dot += q_row[i] * k_row[i];
                }
            }
            kq_scores[(size_t)kv * (size_t)n_q + (size_t)q] = dot;
        }
    }

#ifdef __HVX__
    if (hvx_ok) qurt_hvx_unlock();
#endif
    return 0;
}

// ============================================================================
// Phase 6: logit_argmax_u16 — HVX argmax over UFIXED_POINT_16 logit row.
// ============================================================================
//
// Dispatches to sp_hvx_logit_argmax_u16() from sp_hvx_logits.c, which runs
// the vectorised max scan on the HVX vector units. The buffer arrives via
// FastRPC from an ION-backed (rpcmem_alloc) host buffer — the cDSP accesses
// it zero-copy via the SMMU, so no data is copied across the FastRPC boundary.
//
// Caller (qnn_bin_driver.cpp) allocates the Split 4 logit output buffer as
// rpcmem so the HTP can write into it and the DSP can read from it without any
// intermediate copy — 4 bytes (token_id int32) is the only data that crosses
// the host/DSP boundary per decode step.
// ============================================================================

// Forward decl — defined in sp_hvx_logits.c.
int sp_hvx_logit_argmax_u16(const uint16_t *buf, int vocab_size);

// Signature matches qaic-generated sp_hex.h:
//   __QAIC_HEADER(sp_hex_logit_argmax_u16)(remote_handle64 _h,
//       const unsigned short* logits_row, int logits_rowLen,
//       int vocab_size, int* token_id)
// IDL "rout long" → C "int*" (qaic maps IDL long to C int in skel/stub).
int sp_hex_logit_argmax_u16(remote_handle64 h,
                             const unsigned short *logits_row,
                             int logits_row_len,
                             int vocab_size,
                             int *token_id) {
    (void)h;
    if (!logits_row || !token_id) return -1;
    if (vocab_size <= 0 || logits_row_len < vocab_size) return -1;

#ifdef __HVX__
    // Lock HVX for this thread — required before any HVX instruction.
    // sp_hex_open() holds the lock for the session thread, but the IDL
    // method may execute on a different QURT thread. Re-lock defensively.
    int lock_rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
#endif

    int idx = sp_hvx_logit_argmax_u16((const uint16_t *)logits_row, vocab_size);

#ifdef __HVX__
    if (lock_rc == 0) qurt_hvx_unlock();
#endif

    if (idx < 0) return -1;
    *token_id = (int)idx;
    FARF(RUNTIME_HIGH, "[sp_hex] logit_argmax_u16: token_id=%d vocab=%d",
         *token_id, vocab_size);
    return 0;
}

// ============================================================================
// Phase 7B — Raw DMA probe (dmaWrapper.h).
//
// Binary probe: can the Hexagon DMA engine accept raw byte buffers in
// unsigned PD? Halide DMA is blocked (0x4e EPERM); this tests the
// lower-level dmaWrapper.h path before committing to async weight streaming.
//
// IDL marshalls src_buf as sequence<unsigned short> (matching logit_argmax_u16)
// so the existing _skel_method / _stub_method_10 templates handle transport.
// The actual transfer treats the buffer as raw bytes (eDmaFmt_RawData).
//
// src_buf       : DDR input buffer (rpcmem; SMMU-mapped → DSP sees VA directly)
// src_bufLen    : number of uint16 words (buffer_bytes / 2)
// src_len_words : same as src_bufLen (IDL "in long" mirrors the sequence length
//                 so the DSP knows how many bytes are actually valid)
// timing_us     : set to wall-clock duration of the transfer attempt (µs),
//                 or -1 if AllocDma fails before the transfer.
//
// HAP_perf_get_pcycles() is used for timing. Cycle→µs: cycles / (Hexagon MHz).
// On V69 at SVS2 the cDSP runs ~547 MHz; at NOM ~729 MHz. We report raw
// µs via HAP_perf_get_pcycles with a nominal 700 MHz divisor — adequate
// precision for a pass/fail probe.
// ============================================================================
int sp_hex_probe_dma_raw(remote_handle64 h,
                          const unsigned short *src_buf,
                          int src_buf_len,
                          int src_len_words,
                          int *timing_us) {
    (void)h; (void)src_buf; (void)src_buf_len; (void)src_len_words;

    // ── Phase 7B + testsig probe result (2026-05-05, S22U SM8450) ────────────
    //
    // Dispatch: confirmed working (sentinel 0x55 returned cleanly, no crash).
    // ubwcdma_dynlib.so: PRESENT on device, loads into DSP process correctly.
    //   hDmaWrapper_AllocDma weak symbol = non-NULL (library found in DSP path).
    //
    // Calling hDmaWrapper_AllocDma() in unsigned PD → DSP hardware fault.
    //   FastRPC returns AEE_EBADPARM=14 (different from TLBMISS null-ptr=78).
    //   The fault is a hardware-level privilege violation, not a library issue.
    //
    // Testsig (0x467f8091 for this device): generated and pushed to
    //   /data/local/tmp/sp-engine/ (in ADSP_LIBRARY_PATH). Had no effect —
    //   DMA hardware still faults. Testsig requires debug fuse absent on S22U.
    //
    // Conclusion: DMA engine inaccessible from unsigned PD regardless of
    //   DMA library availability or testsig. Requires signed PD (Phase 9).
    //
    if (timing_us) *timing_us = -1;
    FARF(RUNTIME_HIGH, "[sp_hex] probe_dma_raw: blocked (hDmaWrapper_AllocDma "
         "faults in unsigned PD — testsig no help on production S22U)");
    return 0x4E;  // EPERM: DMA engine inaccessible in unsigned PD
}

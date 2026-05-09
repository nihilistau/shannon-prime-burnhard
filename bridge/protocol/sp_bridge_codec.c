// Shannon-Prime Bridge: residue codec — implementation. See sp_bridge_proto.h.
//
// Wraps the core VHT2 banded quantizer (sp_band_quantize / sp_band_dequantize)
// to compress an [n_kv x row_dim] float buffer for transmission over the
// USB-C bulk endpoint. Uses a single uniform band configuration per call —
// callers that want richer (per-tier) layouts should call this multiple
// times with different tier_bits and stack the resulting buffers.

#include "sp_bridge_proto.h"
#include "shannon_prime.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Build a 4-band config where every band uses the requested bits-per-elem.
// 4 bands matches the default ship-path layout (K bits {5,5,4,3}); using a
// uniform value here keeps the header small without losing the multi-band
// structure that VHT2 expects.
static void bridge_build_band_cfg(sp_band_config_t *bc, int row_dim,
                                  uint16_t tier_bits) {
    int bits[SP_MAX_BANDS];
    int n_bands = 4;
    for (int i = 0; i < n_bands; ++i) bits[i] = (int)tier_bits;
    sp_band_config_init(bc, row_dim, n_bands, bits);
}

int sp_bridge_pack_residue(const float *src, uint32_t n_kv, uint32_t row_dim,
                           uint16_t tier_bits,
                           uint8_t **out_bytes, uint32_t *out_n_bytes) {
    if (!src || !out_bytes || !out_n_bytes) return -1;
    if (n_kv == 0 || row_dim == 0) return -2;
    if (tier_bits == 0 || tier_bits > 8) return -3;

    sp_band_config_t bc;
    bridge_build_band_cfg(&bc, (int)row_dim, tier_bits);
    if (bc.total_bytes <= 0) return -4;

    size_t total = (size_t)n_kv * (size_t)bc.total_bytes;
    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) return -5;

    for (uint32_t r = 0; r < n_kv; ++r) {
        sp_band_quantize(src + (size_t)r * row_dim,
                         buf + (size_t)r * bc.total_bytes, &bc);
    }

    *out_bytes   = buf;
    *out_n_bytes = (uint32_t)total;
    return 0;
}

int sp_bridge_unpack_residue(const uint8_t *bytes, uint32_t n_bytes,
                             uint32_t n_kv, uint32_t row_dim,
                             uint16_t tier_bits,
                             float **out_floats) {
    if (!bytes || !out_floats) return -1;
    if (n_kv == 0 || row_dim == 0) return -2;
    if (tier_bits == 0 || tier_bits > 8) return -3;

    sp_band_config_t bc;
    bridge_build_band_cfg(&bc, (int)row_dim, tier_bits);
    if (bc.total_bytes <= 0) return -4;
    if ((size_t)n_kv * (size_t)bc.total_bytes != n_bytes) return -6;

    float *out = (float *)malloc((size_t)n_kv * row_dim * sizeof(float));
    if (!out) return -5;

    for (uint32_t r = 0; r < n_kv; ++r) {
        sp_band_dequantize(bytes + (size_t)r * bc.total_bytes,
                           out + (size_t)r * row_dim, &bc);
    }

    *out_floats = out;
    return 0;
}

void sp_bridge_free(void *p) {
    if (p) free(p);
}

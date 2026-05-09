// Shannon-Prime Bridge: Wire protocol extensions for desktop ↔ phone fabric.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Layered on top of the existing sp_sidecar_proto.h framing
// (magic + cmd + payload_len + bytes). This header adds the message types
// that the heartbeat manager and the residue-migration path use, and
// declares the codec helpers that sp_bridge_codec.c implements.
//
// Wire commands defined here intentionally start at 0x0100 to avoid
// colliding with the existing sp_sidecar_cmd_t values (which top out at
// 0x00FF for SP_CMD_SHUTDOWN).

#ifndef SP_BRIDGE_PROTO_H
#define SP_BRIDGE_PROTO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "../../desktop/beast_canyon/sp_sidecar_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Bridge command codes (extend sp_sidecar_cmd_t)
// ============================================================================

enum {
    // Heartbeat: payload = u64 monotonic counter; phone echoes back the
    // same counter as the SP_RESP_OK payload. The heartbeat manager
    // uses this for liveness, not the existing SP_CMD_PING (which has
    // no payload and can't detect dropped frames mid-stream).
    SP_CMD_BRIDGE_HEARTBEAT  = 0x0100,

    // Phone-drafted speculative tokens (phone → desktop, push-style).
    // Payload follows sp_bridge_draft_pkt_t below.
    SP_CMD_BRIDGE_DRAFT_PUSH = 0x0110,

    // Residue assignment (desktop → phone). Payload = sp_bridge_residue_pkt_t
    // header followed by the VHT2-banded compressed bytes.
    SP_CMD_BRIDGE_RESIDUE_ASSIGN = 0x0120,

    // Residue result (phone → desktop). Payload = sp_bridge_residue_pkt_t
    // header followed by the result bytes (same banded format).
    SP_CMD_BRIDGE_RESIDUE_RESULT = 0x0121,
};

// ============================================================================
// Packet shapes
// ============================================================================

// Draft tokens pushed from phone to desktop. n_tokens may be up to
// SP_SIDECAR_MAX_DRAFT (8). logprobs are fp32 sum-log-prob per token.
typedef struct {
    uint32_t batch_id;
    uint16_t n_tokens;
    uint16_t reserved;
    // followed by:
    //   int32_t  token_ids[n_tokens]
    //   float    logprobs[n_tokens]
} sp_bridge_draft_pkt_t;

// Residue assignment / result header. The compressed payload follows.
// `tier_bits` must match what the receiver expects (matches sp_band_quantize
// configuration on the desktop side).
typedef struct {
    uint32_t batch_id;
    uint16_t expert_id;
    uint16_t tier_bits;        // bits-per-element of the banded quantizer
    uint32_t n_kv;             // logical row count
    uint32_t n_compressed_bytes; // payload length following this header
} sp_bridge_residue_pkt_t;

// ============================================================================
// Codec — pack/unpack residue payloads (sp_bridge_codec.c)
// ============================================================================

// Pack a row-major float buffer of [n_kv * row_dim] elements into
// banded-quantized bytes. Caller owns *out_bytes (free with sp_bridge_free).
// Returns 0 on success; negative on error.
int sp_bridge_pack_residue(const float *src, uint32_t n_kv, uint32_t row_dim,
                           uint16_t tier_bits,
                           uint8_t **out_bytes, uint32_t *out_n_bytes);

// Inverse: decompress *bytes back into [n_kv * row_dim] floats. Caller
// owns *out_floats (free with sp_bridge_free).
int sp_bridge_unpack_residue(const uint8_t *bytes, uint32_t n_bytes,
                             uint32_t n_kv, uint32_t row_dim,
                             uint16_t tier_bits,
                             float **out_floats);

// Free a buffer returned by either pack or unpack.
void sp_bridge_free(void *p);

#ifdef __cplusplus
}
#endif

#endif // SP_BRIDGE_PROTO_H

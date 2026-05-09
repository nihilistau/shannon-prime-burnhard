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

    // Speculative oracle packet (phone → desktop, fire-and-forget).
    // Fixed 16-byte payload — see burnhard_oracle_packet_t below. The
    // phone-side draft model emits one of these per locally-decoded
    // token; the desktop's expert prefetcher consumes them async.
    SP_CMD_BRIDGE_ORACLE     = 0x0130,
};

// ============================================================================
// Oracle packet — fixed 16 bytes. Sent by the phone-side draft engine
// every time it decodes a token.  The desktop reads `expert_id_1/2` and
// kicks off an mmap prefetch of those expert blocks before the host's
// MoE router gets to that layer.
//
// Wire layout matches a packed 16 B struct so we can `memcpy` directly
// off the recv buffer with zero parsing.  No alignment between this
// struct and the wire bytes — receiver must byte-copy into a local
// instance before reading fields if the recv buffer isn't aligned.
// ============================================================================
typedef struct
#if defined(__GNUC__) || defined(__clang__)
__attribute__((packed))
#endif
{
    uint32_t token_id;       // Speculative token predicted by the S22
    uint16_t expert_id_1;    // Top-1 predicted MoE expert
    uint16_t expert_id_2;    // Top-2 predicted MoE expert
    uint8_t  confidence;     // 0..255, oracle certainty (0 = no signal)
    uint8_t  layer_hint;     // 0..n_layer-1, layer the prediction targets
    uint16_t batch_id;       // Caller-managed correlation ID
    uint8_t  padding[6];     // Pad to 16 bytes for AVX/cacheline alignment
} burnhard_oracle_packet_t;

#if defined(_MSC_VER)
#pragma pack(push, 1)
typedef struct {
    uint32_t token_id;
    uint16_t expert_id_1;
    uint16_t expert_id_2;
    uint8_t  confidence;
    uint8_t  layer_hint;
    uint16_t batch_id;
    uint8_t  padding[6];
} burnhard_oracle_packet_msvc_t;
#pragma pack(pop)
// On MSVC the __attribute__((packed)) above is a no-op; use this
// alternate type at the wire boundary if you need guaranteed 16 B.
#endif

// Compile-time guard so a future field addition can't silently grow
// the wire packet without bumping the protocol version.
#define BURNHARD_ORACLE_PKT_BYTES 16

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

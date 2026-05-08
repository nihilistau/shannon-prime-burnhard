// Shannon-Prime Beast Canyon — Sidecar Wire Protocol
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Binary protocol between Beast Canyon (host) and S22U (phone sidecar)
// over ADB port-forwarded TCP. All integers are little-endian.
//
// The phone runs a small draft model (e.g. Qwen3-0.6B litertlm) for
// speculative token prediction. Beast Canyon sends the current token,
// the phone drafts N tokens and sends them back. Beast Canyon verifies
// the drafts against the full model — accepted tokens skip full forward
// passes, giving a throughput multiplier.
//
// Message framing: [magic:u32][cmd:u32][payload_len:u32][payload...]
//                  [magic:u32][status:u32][payload_len:u32][payload...]
//
// The sidecar also handles Prime-PE offload (positional encoding
// computation on Hexagon DSP) when the draft model isn't active.

#ifndef SP_SIDECAR_PROTO_H
#define SP_SIDECAR_PROTO_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#else
#  include <sys/socket.h>
#  include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Wire protocol magic (ASCII "SP22" little-endian)
#define SP_SIDECAR_MAGIC    0x32325053u

// ============================================================================
// Commands (host → phone)
// ============================================================================

typedef enum {
    // Heartbeat / ping — phone responds with SP_RESP_OK
    SP_CMD_PING          = 0x0001,

    // Load draft model: payload = null-terminated GGUF path on phone
    SP_CMD_LOAD_DRAFT    = 0x0010,

    // Prefill draft model: payload = [n_tokens:u32][token_ids:i32*n]
    SP_CMD_PREFILL       = 0x0011,

    // Draft N tokens: payload = [current_token:i32][n_draft:u32]
    // Response: [n_drafted:u32][draft_ids:i32*n]
    SP_CMD_DRAFT         = 0x0012,

    // Accept/resync after verification:
    //   payload = [n_accepted:u32][verified_ids:i32*n_accepted]
    // If n_accepted < n_drafted, the last (n_drafted - n_accepted) drafts
    // are discarded and the phone resyncs from verified_ids[n_accepted-1].
    SP_CMD_ACCEPT        = 0x0013,

    // Reset draft model state (new conversation)
    SP_CMD_RESET         = 0x0014,

    // Prime-PE offload: payload = [dim:u32][hidden:f32*dim]
    // Response: [dim:u32][pe_out:f32*dim]
    SP_CMD_PRIME_PE      = 0x0020,

    // Query phone stats (battery, thermal, draft accuracy)
    SP_CMD_STATS         = 0x0030,

    // Shutdown sidecar server
    SP_CMD_SHUTDOWN      = 0x00FF,
} sp_sidecar_cmd_t;

// ============================================================================
// Responses (phone → host)
// ============================================================================

typedef enum {
    SP_RESP_OK           = 0x0000,
    SP_RESP_ERROR        = 0x0001,
    SP_RESP_NOT_LOADED   = 0x0002,  // Draft model not loaded
    SP_RESP_BUSY         = 0x0003,  // Still processing previous command
} sp_sidecar_resp_t;

// ============================================================================
// Wire message header (12 bytes)
// ============================================================================

typedef struct {
    uint32_t magic;         // SP_SIDECAR_MAGIC
    uint32_t cmd_or_status; // sp_sidecar_cmd_t or sp_sidecar_resp_t
    uint32_t payload_len;   // Bytes following this header
} sp_sidecar_hdr_t;

// ============================================================================
// Draft result (returned by sp_beast_sidecar_draft)
// ============================================================================

#define SP_SIDECAR_MAX_DRAFT 8

typedef struct {
    int32_t  tokens[SP_SIDECAR_MAX_DRAFT];  // Drafted token IDs
    int      n_drafted;                      // How many were actually drafted
    uint64_t draft_us;                       // Time on phone (microseconds)
    uint64_t round_trip_us;                  // Host-measured round trip
} sp_sidecar_draft_result_t;

// ============================================================================
// Phone stats (response to SP_CMD_STATS)
// ============================================================================

typedef struct {
    float    battery_pct;       // 0.0–100.0
    float    thermal_celsius;   // SoC temperature
    float    draft_accuracy;    // Running hit rate
    uint32_t total_drafted;     // Total tokens drafted
    uint32_t total_accepted;    // Total tokens accepted
} sp_sidecar_phone_stats_t;

// ============================================================================
// Helper: send/recv full message over socket
// ============================================================================

// Send a complete message. Returns 0 on success, -1 on error.
static inline int sp_sidecar_send(int fd, uint32_t cmd, const void* payload, uint32_t len) {
    sp_sidecar_hdr_t hdr;
    hdr.magic = SP_SIDECAR_MAGIC;
    hdr.cmd_or_status = cmd;
    hdr.payload_len = len;

    // Send header
    const char* p = (const char*)&hdr;
    size_t rem = sizeof(hdr);
    while (rem > 0) {
        int sent = send(fd, p, (int)rem, 0);
        if (sent <= 0) return -1;
        p += sent;
        rem -= (size_t)sent;
    }

    // Send payload
    if (len > 0 && payload) {
        p = (const char*)payload;
        rem = len;
        while (rem > 0) {
            int sent = send(fd, p, (int)rem, 0);
            if (sent <= 0) return -1;
            p += sent;
            rem -= (size_t)sent;
        }
    }
    return 0;
}

// Receive exactly `n` bytes. Returns 0 on success, -1 on error/disconnect.
static inline int sp_sidecar_recv_exact(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    size_t rem = n;
    while (rem > 0) {
        int got = recv(fd, p, (int)rem, 0);
        if (got <= 0) return -1;
        p += got;
        rem -= (size_t)got;
    }
    return 0;
}

// Receive a response header + payload. Caller must free(*out_payload) if non-NULL.
// Returns the status code, or -1 on network error.
static inline int sp_sidecar_recv_msg(int fd, void** out_payload, uint32_t* out_len) {
    sp_sidecar_hdr_t hdr;
    if (sp_sidecar_recv_exact(fd, &hdr, sizeof(hdr)) != 0) return -1;
    if (hdr.magic != SP_SIDECAR_MAGIC) return -1;

    if (out_len) *out_len = hdr.payload_len;

    if (hdr.payload_len > 0) {
        void* buf = malloc(hdr.payload_len);
        if (!buf) return -1;
        if (sp_sidecar_recv_exact(fd, buf, hdr.payload_len) != 0) {
            free(buf);
            return -1;
        }
        if (out_payload) *out_payload = buf;
        else free(buf);
    } else {
        if (out_payload) *out_payload = NULL;
    }

    return (int)hdr.cmd_or_status;
}

#ifdef __cplusplus
}
#endif

#endif // SP_SIDECAR_PROTO_H

// Shannon-Prime Bridge: heartbeat manager (desktop-side).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Spawns a low-priority worker thread that pings the phone over the
// existing sidecar socket once per second. Tracks ack/nack counts and
// publishes a stable liveness state with hysteresis:
//
//   ONLINE   ←→   DEGRADED   ←→   OFFLINE
//
// Transitions:
//   ONLINE → DEGRADED   after  2 consecutive missed beats
//   DEGRADED → OFFLINE  after  5 consecutive missed beats (cumulative)
//   OFFLINE → ONLINE    after  3 consecutive ACKs (debounce)
//
// On every state change we fire the user-supplied callback. The callback
// receives the new state plus a free-form context pointer. The callback
// is invoked from the heartbeat thread — keep it short and non-blocking.

#ifndef SP_BRIDGE_HEARTBEAT_H
#define SP_BRIDGE_HEARTBEAT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SP_BRIDGE_OFFLINE  = 0,
    SP_BRIDGE_DEGRADED = 1,
    SP_BRIDGE_ONLINE   = 2,
} sp_bridge_state_t;

typedef void (*sp_bridge_state_cb)(sp_bridge_state_t new_state, void *ctx);

typedef struct sp_bridge_heartbeat sp_bridge_heartbeat_t;

// Construct + start a heartbeat manager. `socket_fd` is the live ADB-
// forwarded TCP socket from sp_beast_sidecar_connect. `period_ms` is the
// inter-beat interval (1000 is reasonable). `cb`/`cb_ctx` are optional.
// Returns NULL on failure (out of memory, thread spawn failed).
sp_bridge_heartbeat_t *sp_bridge_heartbeat_start(int socket_fd,
                                                 uint32_t period_ms,
                                                 sp_bridge_state_cb cb,
                                                 void *cb_ctx);

// Snapshot of current liveness state. Cheap; safe to call from any thread.
sp_bridge_state_t sp_bridge_heartbeat_state(const sp_bridge_heartbeat_t *hb);

// Counters since start. ack/nack are atomic loads; safe from any thread.
uint64_t sp_bridge_heartbeat_acks(const sp_bridge_heartbeat_t *hb);
uint64_t sp_bridge_heartbeat_misses(const sp_bridge_heartbeat_t *hb);

// Stop the worker thread, free resources. Idempotent.
void sp_bridge_heartbeat_stop(sp_bridge_heartbeat_t *hb);

const char *sp_bridge_state_name(sp_bridge_state_t s);

#ifdef __cplusplus
}
#endif

#endif // SP_BRIDGE_HEARTBEAT_H

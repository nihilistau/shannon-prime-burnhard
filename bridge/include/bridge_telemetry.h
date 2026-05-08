/*
 * bridge_telemetry.h — Desktop ↔ S22 Ultra heartbeat over ADB/USB-C
 *
 * The bridge runs ADB port-forwarding (adb forward tcp:8090 tcp:8090)
 * to create a localhost TCP tunnel to the phone. The desktop is the
 * initiator; the phone runs a lightweight daemon.
 *
 * Protocol:
 *   Desktop sends PING (4 bytes: "BPNG")
 *   Phone replies PONG (4 bytes: "BPOK") + 8 bytes payload (phone timestamp)
 *   If no PONG within timeout → declare sidecar dead → reclaim tasks
 *
 * Lifecycle:
 *   bridge_start()   — spawn heartbeat thread
 *   bridge_is_alive() — check last status (lock-free)
 *   bridge_stop()    — join thread, close socket
 */

#ifndef BRIDGE_TELEMETRY_H
#define BRIDGE_TELEMETRY_H

#include "burnhard_config.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Callback: fired on the heartbeat thread when sidecar state changes. */
typedef void (*bridge_state_cb)(bool alive, void *userdata);

typedef struct bridge_ctx bridge_ctx_t;

/*
 * Start the heartbeat loop.
 * cfg→bridge_port / heartbeat_interval_ms / heartbeat_timeout_ms are used.
 * on_change is called (from heartbeat thread) whenever alive flips.
 * Returns NULL on failure.
 */
bridge_ctx_t *bridge_start(const burnhard_config_t *cfg,
                           bridge_state_cb on_change,
                           void *userdata);

/* Lock-free check: is the sidecar responding right now? */
bool bridge_is_alive(const bridge_ctx_t *ctx);

/* Get round-trip latency of last successful ping (microseconds). */
uint64_t bridge_last_rtt_us(const bridge_ctx_t *ctx);

/* Cleanly stop the heartbeat thread and free resources. */
void bridge_stop(bridge_ctx_t *ctx);

/* ── ADB helpers ──────────────────────────────────────────────────── */

/* Run `adb forward tcp:port tcp:port`. Returns 0 on success. */
int bridge_adb_forward(uint16_t port);

/* Run `adb forward --remove tcp:port`. */
int bridge_adb_unforward(uint16_t port);

#ifdef __cplusplus
}
#endif

#endif /* BRIDGE_TELEMETRY_H */

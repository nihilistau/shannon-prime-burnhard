/*
 * burnhard_init.c — Dynamic feature gating & startup sequence
 *
 * This is the first thing that runs. It:
 *   1. Probes hardware → selects operating mode
 *   2. Opens the reservoir on the best available tier
 *   3. Starts the bridge heartbeat (if sidecar-capable mode)
 *   4. Registers a state-change callback to handle hot disconnect/reconnect
 *   5. Logs the full capability snapshot
 *
 * After init returns, the caller has:
 *   - A valid burnhard_state_t with mode, reservoir, and bridge context
 *   - Speculative decode routing configured (sidecar or local)
 *   - A guarantee that mode transitions are handled automatically
 */

#include "burnhard_config.h"
#include "burnhard_reservoir.h"
#include "bridge_telemetry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Global engine state ──────────────────────────────────────────── */

typedef struct {
    burnhard_config_t  cfg;
    reservoir_t        reservoir;
    bridge_ctx_t      *bridge;
    bool               initialized;
} burnhard_state_t;

/* Singleton — one engine per process */
static burnhard_state_t g_state = {0};


/* ── Bridge callback: sidecar state changed ───────────────────────── */

static void on_sidecar_state_change(bool alive, void *userdata) {
    burnhard_state_t *st = (burnhard_state_t *)userdata;
    burnhard_config_t *cfg = &st->cfg;

    if (alive) {
        /*
         * Sidecar just came back online.
         * If we were DESKTOP_SOLO, upgrade to FULL_PULSE.
         * Route speculative drafting to the phone.
         */
        fprintf(stderr, "[init] Sidecar reconnected — upgrading to FULL_PULSE\n");
        cfg->caps.sidecar_responsive = true;
        cfg->mode = BURNHARD_MODE_FULL_PULSE;
        cfg->spec_use_sidecar = true;
    } else {
        /*
         * Sidecar dropped. Downgrade to DESKTOP_SOLO.
         * Reclaim speculative tasks — desktop drafts locally now.
         */
        fprintf(stderr, "[init] Sidecar disconnected — falling back to DESKTOP_SOLO\n");
        cfg->caps.sidecar_responsive = false;
        cfg->mode = BURNHARD_MODE_DESKTOP_SOLO;
        cfg->spec_use_sidecar = false;

        /* TODO: signal the speculative decode thread to switch to local drafting */
    }

    fprintf(stderr, "[init] Mode is now: %s\n", burnhard_mode_str(cfg->mode));
}


/* ── Public init/shutdown ─────────────────────────────────────────── */

int burnhard_init(void) {
    if (g_state.initialized) {
        fprintf(stderr, "[init] Already initialized (mode=%s)\n",
                burnhard_mode_str(g_state.cfg.mode));
        return 0;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "  ____                  _   _               _\n");
    fprintf(stderr, " | __ ) _   _ _ __ _ __| | | | __ _ _ __ __| |\n");
    fprintf(stderr, " |  _ \\| | | | '__| '_ \\ |_| |/ _` | '__/ _` |\n");
    fprintf(stderr, " | |_) | |_| | |  | | | |  _  | (_| | | | (_| |\n");
    fprintf(stderr, " |____/ \\__,_|_|  |_| |_|_| |_|\\__,_|_|  \\__,_|\n");
    fprintf(stderr, "  Shannon-Prime Distributed Engine v0.1.0\n\n");

    memset(&g_state, 0, sizeof(g_state));

    /* ── Step 1: Detect hardware ──────────────────────────────────── */
    fprintf(stderr, "[init] Step 1: Probing hardware...\n");

    /* Allow env overrides for Optane/NVMe paths */
    g_state.cfg.optane_dev_path = getenv("BURNHARD_OPTANE_DEV");
    g_state.cfg.nvme_cache_path = getenv("BURNHARD_NVME_CACHE");

    /* If no NVMe cache path set, use a default */
    if (!g_state.cfg.nvme_cache_path) {
#ifdef _WIN32
        g_state.cfg.nvme_cache_path = "C:\\burnhard_reservoir.bin";
#else
        g_state.cfg.nvme_cache_path = "/tmp/burnhard_reservoir.bin";
#endif
    }

    int rc = burnhard_detect(&g_state.cfg);
    if (rc != 0) {
        fprintf(stderr, "[init] FATAL: hardware detection failed\n");
        return -1;
    }

    burnhard_dump_caps(&g_state.cfg);

    /* ── Step 2: Open reservoir ───────────────────────────────────── */
    fprintf(stderr, "[init] Step 2: Opening reservoir (%llu MB requested)...\n",
            (unsigned long long)g_state.cfg.reservoir_size_mb);

    size_t res_bytes = (size_t)g_state.cfg.reservoir_size_mb * 1024ULL * 1024ULL;
    /* Cap at 4 GB for initial scaffold to avoid OOM */
    if (res_bytes > 4ULL * 1024 * 1024 * 1024) {
        res_bytes = 4ULL * 1024 * 1024 * 1024;
        fprintf(stderr, "[init] Capped reservoir at 4 GB for initial scaffold\n");
    }

    rc = reservoir_open(&g_state.reservoir, &g_state.cfg, res_bytes);
    if (rc != 0) {
        fprintf(stderr, "[init] FATAL: reservoir allocation failed\n");
        return -1;
    }

    /* ── Step 3: Start bridge (desktop modes only) ────────────────── */
    if (g_state.cfg.mode != BURNHARD_MODE_LONE_WOLF) {
        fprintf(stderr, "[init] Step 3: Starting bridge heartbeat...\n");
        g_state.bridge = bridge_start(&g_state.cfg,
                                       on_sidecar_state_change,
                                       &g_state);
        if (!g_state.bridge) {
            fprintf(stderr, "[init] Warning: bridge failed to start "
                    "(continuing without sidecar support)\n");
            g_state.cfg.mode = BURNHARD_MODE_DESKTOP_SOLO;
            g_state.cfg.spec_use_sidecar = false;
        }
    } else {
        fprintf(stderr, "[init] Step 3: Skipped (LONE_WOLF mode — no bridge)\n");
    }

    /* ── Step 4: Ready ────────────────────────────────────────────── */
    g_state.initialized = true;

    fprintf(stderr, "\n[init] *** BurnHard online: %s ***\n",
            burnhard_mode_str(g_state.cfg.mode));

    switch (g_state.cfg.mode) {
    case BURNHARD_MODE_FULL_PULSE:
        fprintf(stderr, "[init] S22 = Frontal Lobe (drafting %d tokens)\n",
                g_state.cfg.spec_draft_tokens);
        fprintf(stderr, "[init] Beast Canyon = Muscle (residue crunching)\n");
        break;
    case BURNHARD_MODE_DESKTOP_SOLO:
        fprintf(stderr, "[init] Desktop handling all speculative + residue work\n");
        fprintf(stderr, "[init] Bridge monitoring for sidecar reconnection...\n");
        break;
    case BURNHARD_MODE_LEGACY:
        fprintf(stderr, "[init] No Optane — reservoir on %s\n",
                reservoir_tier_str(g_state.reservoir.tier));
        break;
    case BURNHARD_MODE_LONE_WOLF:
        fprintf(stderr, "[init] Mobile standalone — local GGUF mapping active\n");
        break;
    }

    fprintf(stderr, "\n");
    return 0;
}

void burnhard_shutdown(void) {
    if (!g_state.initialized) return;

    fprintf(stderr, "[init] Shutting down BurnHard...\n");

    if (g_state.bridge) {
        bridge_stop(g_state.bridge);
        g_state.bridge = NULL;
    }

    reservoir_close(&g_state.reservoir);

    g_state.initialized = false;
    fprintf(stderr, "[init] Shutdown complete.\n");
}

/* Accessors for other modules */
const burnhard_config_t *burnhard_get_config(void) {
    return g_state.initialized ? &g_state.cfg : NULL;
}

reservoir_t *burnhard_get_reservoir(void) {
    return g_state.initialized ? &g_state.reservoir : NULL;
}

burnhard_mode_t burnhard_get_mode(void) {
    return g_state.cfg.mode;
}

bool burnhard_sidecar_live(void) {
    return g_state.bridge ? bridge_is_alive(g_state.bridge) : false;
}

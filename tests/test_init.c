/*
 * test_init.c — Smoke test for BurnHard startup sequence
 *
 * Exercises: detect → mode select → reservoir open → bridge start → shutdown
 * Run without any hardware present to verify LEGACY/DESKTOP_SOLO fallbacks.
 */

#include "burnhard_config.h"
#include "burnhard_reservoir.h"
#include "bridge_telemetry.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Forward-declare the init API (defined in burnhard_init.c) */
extern int  burnhard_init(void);
extern void burnhard_shutdown(void);
extern const burnhard_config_t *burnhard_get_config(void);
extern burnhard_mode_t          burnhard_get_mode(void);

int main(void) {
    fprintf(stderr, "=== BurnHard Smoke Test ===\n\n");

    /* ── Test 1: Init succeeds ──────────────────────────────────────── */
    int rc = burnhard_init();
    assert(rc == 0 && "burnhard_init should succeed");

    const burnhard_config_t *cfg = burnhard_get_config();
    assert(cfg != NULL);

    /* ── Test 2: Mode is valid ──────────────────────────────────────── */
    burnhard_mode_t mode = burnhard_get_mode();
    assert(mode >= BURNHARD_MODE_FULL_PULSE && mode <= BURNHARD_MODE_LONE_WOLF);
    fprintf(stderr, "[test] Mode: %s — OK\n", burnhard_mode_str(mode));

    /* ── Test 3: mode_str round-trips ───────────────────────────────── */
    assert(burnhard_mode_str(BURNHARD_MODE_FULL_PULSE) != NULL);
    assert(burnhard_mode_str(BURNHARD_MODE_DESKTOP_SOLO) != NULL);
    assert(burnhard_mode_str(BURNHARD_MODE_LEGACY) != NULL);
    assert(burnhard_mode_str(BURNHARD_MODE_LONE_WOLF) != NULL);
    fprintf(stderr, "[test] mode_str — OK\n");

    /* ── Test 4: RAM is detected ────────────────────────────────────── */
    assert(cfg->caps.system_ram_mb > 0);
    fprintf(stderr, "[test] RAM: %llu MB — OK\n",
            (unsigned long long)cfg->caps.system_ram_mb);

    /* ── Test 5: Double init is safe ────────────────────────────────── */
    rc = burnhard_init();
    assert(rc == 0 && "double init should be no-op");
    fprintf(stderr, "[test] Double init — OK\n");

    /* ── Test 6: Standalone reservoir test ───────────────────────────── */
    {
        reservoir_t res;
        burnhard_config_t test_cfg = *cfg;
        test_cfg.caps.has_optane = false;
        test_cfg.nvme_cache_path = NULL; /* force RAM tier */

        rc = reservoir_open(&res, &test_cfg, 16 * 1024 * 1024); /* 16 MB */
        assert(rc == 0);
        assert(res.tier == RESERVOIR_TIER_RAM);
        assert(res.base != NULL);
        assert(res.size == 16 * 1024 * 1024);

        /* Write + read back */
        ((char *)res.base)[0] = 0x42;
        assert(((char *)res.base)[0] == 0x42);

        reservoir_close(&res);
        fprintf(stderr, "[test] Standalone reservoir (RAM tier, 16 MB) — OK\n");
    }

    /* ── Test 7: Shutdown ───────────────────────────────────────────── */
    burnhard_shutdown();
    fprintf(stderr, "[test] Shutdown — OK\n");

    /* Double shutdown is safe */
    burnhard_shutdown();
    fprintf(stderr, "[test] Double shutdown — OK\n");

    fprintf(stderr, "\n=== All smoke tests passed ===\n");
    return 0;
}

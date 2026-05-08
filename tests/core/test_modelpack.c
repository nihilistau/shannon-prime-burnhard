// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// Model-pack registry validation.
// Scaffolding tests — verify the resolver matches the right preset,
// that apply_preset writes the expected fields, and that describe()
// yields a non-empty human-readable string.

#include "../core/shannon_prime.h"
#include "../core/shannon_prime_modelpack.h"

#include <stdio.h>
#include <string.h>

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int tests_run = 0, tests_passed = 0;
#define CHECK(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [%s] %s\n", PASS, msg); } \
    else      { printf("  [%s] %s\n", FAIL, msg); } \
} while (0)

int main(void) {
    printf("Shannon-Prime Model-Pack Validation\n");
    printf("====================================\n");

    // ── Registry invariants ────────────────────────────────────────
    printf("\n== Registry ==\n");
    int n = sp_model_preset_count();
    CHECK(n >= 4, "At least 4 presets registered (llama-3, qwen3, qwen3-moe, gemma3)");

    for (int i = 0; i < n; ++i) {
        const sp_model_preset_t *p = sp_model_preset_at(i);
        CHECK(p != NULL,            "preset_at(i) non-null");
        CHECK(p->name        != NULL, "name non-null");
        CHECK(p->arch_pattern != NULL, "arch_pattern non-null");
        CHECK(p->k_n_bands >= 1 && p->k_n_bands <= SP_MAX_BANDS,
              "k_n_bands in [1, SP_MAX_BANDS]");
        CHECK(p->v_n_bands >= 1 && p->v_n_bands <= SP_MAX_BANDS,
              "v_n_bands in [1, SP_MAX_BANDS]");
        CHECK(p->residual_bits >= 1 && p->residual_bits <= 4,
              "residual_bits in [1,4]");
    }

    // ── Resolver: architecture name matching ───────────────────────
    printf("\n== Resolver matches ==\n");

    const sp_model_preset_t *llama  = sp_model_preset_resolve("llama",       128, 32, 8);
    const sp_model_preset_t *qwen3  = sp_model_preset_resolve("qwen3",       128, 36, 8);
    const sp_model_preset_t *q3moe  = sp_model_preset_resolve("qwen3moe",    128, 48, 4);
    const sp_model_preset_t *gemma3 = sp_model_preset_resolve("gemma3",      256, 42, 8);
    const sp_model_preset_t *none   = sp_model_preset_resolve("exotic-arch", 128, 32, 8);

    CHECK(llama  != NULL && strcmp(llama->name,  "llama-3")    == 0,
          "llama arch -> llama-3 preset");
    CHECK(qwen3  != NULL && strcmp(qwen3->name,  "qwen3")      == 0,
          "qwen3 arch -> qwen3 preset");
    CHECK(q3moe  != NULL && strcmp(q3moe->name,  "qwen3-moe")  == 0,
          "qwen3moe arch -> qwen3-moe preset (MoE, more specific than qwen3)");
    CHECK(gemma3 != NULL && strcmp(gemma3->name, "gemma3")     == 0,
          "gemma3 arch -> gemma3 preset");
    CHECK(none   == NULL,
          "unknown arch -> NULL (caller keeps shipping defaults)");

    // ── Null guards ────────────────────────────────────────────────
    printf("\n== Null guards ==\n");
    CHECK(sp_model_preset_resolve(NULL, 128, 32, 8) == NULL,
          "NULL arch_name -> NULL");
    CHECK(sp_model_preset_resolve("", 128, 32, 8) == NULL,
          "empty arch_name -> NULL");
    CHECK(sp_model_preset_at(-1)              == NULL, "index -1 -> NULL");
    CHECK(sp_model_preset_at(n + 100)         == NULL, "out-of-range -> NULL");

    // ── apply_preset overlays tunables, leaves dims alone ──────────
    printf("\n== apply_preset ==\n");
    sp_config_t cfg;
    sp_config_init(&cfg, 128, 32, 8);
    int hd_before = cfg.head_dim;

    int rc = sp_config_apply_preset(&cfg, qwen3);
    CHECK(rc == 0,                       "apply_preset returns 0");
    CHECK(cfg.head_dim   == hd_before,   "head_dim untouched");
    CHECK(cfg.k_n_bands  == qwen3->k_n_bands,
          "k_n_bands copied from preset");
    for (int i = 0; i < cfg.k_n_bands; ++i) {
        char msg[64];
        snprintf(msg, sizeof(msg), "k_band_bits[%d] = %d",
                 i, qwen3->k_band_bits[i]);
        CHECK(cfg.k_band_bits[i] == qwen3->k_band_bits[i], msg);
    }

    CHECK(sp_config_apply_preset(NULL,   qwen3) == -1, "NULL cfg -> -1");
    CHECK(sp_config_apply_preset(&cfg,   NULL)  == -1, "NULL preset -> -1");

    // ── describe() produces a non-empty one-liner ──────────────────
    printf("\n== describe ==\n");
    char buf[256];
    int len = sp_model_preset_describe(qwen3, buf, sizeof(buf));
    CHECK(len > 0, "describe returns positive length");
    CHECK(strstr(buf, "qwen3")      != NULL, "description contains name");
    CHECK(strstr(buf, "K=")         != NULL, "description has K= field");
    CHECK(strstr(buf, "V=")         != NULL, "description has V= field");
    CHECK(strstr(buf, "PROVISIONAL") != NULL || strstr(buf, "CALIBRATED") != NULL,
          "description has a calibration status tag");

    // Print all presets so CI logs give reviewer context
    printf("\n== Registered presets ==\n");
    for (int i = 0; i < n; ++i) {
        const sp_model_preset_t *p = sp_model_preset_at(i);
        sp_model_preset_describe(p, buf, sizeof(buf));
        printf("  %s\n", buf);
    }

    printf("\n====================================\n");
    printf("Results: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

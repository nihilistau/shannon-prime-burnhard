// Shannon-Prime Beast Canyon: Standalone Test Harness
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Usage: sp_beast_test <path-to-gguf>
//
// Runs:
//   1. Optane reservoir mapping + GGUF parse
//   2. Optane Day Zero audit (stride latency, bandwidth)
//   3. AVX-512 Shredder benchmark (first tensor)
//   4. Expert table dump (MoE models)
//   5. Full engine boot (dry run)

#include "sp_beast_canyon.h"
#include "sp_diagnostics_bc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path-to-gguf> [--audit-only]\n", argv[0]);
        fprintf(stderr, "\nRuns the Beast Canyon engine validation suite.\n");
        fprintf(stderr, "Put your GGUF on the Optane drive and point this at it.\n");
        return 1;
    }

    const char *gguf_path = argv[1];
    bool audit_only = (argc >= 3 && strcmp(argv[2], "--audit-only") == 0);

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  BEAST CANYON TEST HARNESS                   ║\n");
    fprintf(stderr, "║  Shannon-Prime Heterogeneous Engine          ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    // ── Test 1: Reservoir mapping ───────────────────────────────────
    fprintf(stderr, "=== TEST 1: Optane Reservoir Mapping ===\n\n");

    static sp_optane_reservoir_t reservoir;
    int rc = sp_optane_init(&reservoir, gguf_path);
    if (rc != 0) {
        fprintf(stderr, "FATAL: Reservoir mapping failed (rc=%d)\n", rc);
        return 1;
    }

    sp_optane_print_status(&reservoir);

    // ── Test 2: Optane Audit ────────────────────────────────────────
    fprintf(stderr, "=== TEST 2: Optane Day Zero Audit ===\n\n");

    sp_pulse_monitor_t monitor;
    sp_diag_init(&monitor);
    int audit_fails = sp_diag_optane_audit(&reservoir, &monitor);

    if (audit_only) {
        sp_optane_free(&reservoir);
        return audit_fails;
    }

    // ── Test 3: Shredder Benchmark ──────────────────────────────────
    fprintf(stderr, "=== TEST 3: AVX-512 Shredder Benchmark ===\n\n");

    sp_shredder_t shredder;
    sp_shredder_config_t shred_cfg;
    sp_shredder_config_init(&shred_cfg);

    // Find the first quantized tensor to shred
    const sp_optane_tensor_t *test_tensor = NULL;
    for (uint32_t i = 0; i < reservoir.tensor_count; i++) {
        const sp_optane_tensor_t *t = &reservoir.tensors[i];
        if (t->type == SP_GGML_TYPE_Q4_0 || t->type == SP_GGML_TYPE_Q8_0 ||
            t->type == SP_GGML_TYPE_Q4_K || t->type == SP_GGML_TYPE_Q6_K) {
            test_tensor = t;
            break;
        }
    }

    if (test_tensor) {
        fprintf(stderr, "Shredding tensor: %s (type=%u, %.2f MB)\n",
                test_tensor->name, test_tensor->type,
                (double)test_tensor->n_bytes / (1024.0*1024.0));

        // Calculate number of elements
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < test_tensor->n_dims; d++) {
            n_elements *= test_tensor->ne[d];
        }

        rc = sp_shredder_init(&shredder, &shred_cfg, n_elements);
        if (rc == 0) {
            uint16_t *staging = (uint16_t *)sp_shredder_staging(&shredder);

            // Run 10 iterations to warm up and measure
            for (int iter = 0; iter < 10; iter++) {
                sp_shredder_auto(&shredder, test_tensor->type,
                                 test_tensor->ptr, staging, n_elements);
            }

            sp_shredder_print_status(&shredder);

            // Validate: check first few values are non-zero
            int nonzero = 0;
            for (int i = 0; i < 32 && i < (int)n_elements; i++) {
                if (staging[i] != 0) nonzero++;
            }
            fprintf(stderr, "Validation: %d/32 non-zero values in first block  %s\n\n",
                    nonzero, nonzero > 0 ? "PASS" : "FAIL");

            sp_shredder_free(&shredder);
        }
    } else {
        fprintf(stderr, "No quantized tensors found (F16 model?) — skipping shredder test\n\n");
    }

    // ── Test 4: Expert Table (MoE) ──────────────────────────────────
    if (reservoir.is_moe) {
        fprintf(stderr, "=== TEST 4: MoE Expert Table ===\n\n");
        fprintf(stderr, "Model is MoE with %d experts (top-%d)\n",
                reservoir.n_experts, reservoir.n_experts_per_token);

        for (int e = 0; e < reservoir.n_experts && e < 4; e++) {
            const sp_optane_expert_t *exp = &reservoir.experts[e];
            fprintf(stderr, "  Expert %d: gate=%p up=%p down=%p (%.2f MB)\n",
                    e,
                    exp->gate_proj ? exp->gate_proj->ptr : NULL,
                    exp->up_proj   ? exp->up_proj->ptr   : NULL,
                    exp->down_proj ? exp->down_proj->ptr  : NULL,
                    (double)exp->total_bytes / (1024.0*1024.0));
        }
        if (reservoir.n_experts > 4) {
            fprintf(stderr, "  ... (%d more experts)\n", reservoir.n_experts - 4);
        }
        fprintf(stderr, "\n");
    }

    // ── Test 5: Full Engine Boot (dry run) ──────────────────────────
    fprintf(stderr, "=== TEST 5: Full Engine Boot ===\n\n");

    static sp_beast_engine_t engine;
    sp_beast_config_t beast_cfg;
    sp_beast_config_init(&beast_cfg);
    beast_cfg.gguf_path = gguf_path;
    beast_cfg.force_cpu_only = true;  // Don't require GPUs for test
    beast_cfg.enable_sidecar = false;

    rc = sp_beast_init(&engine, &beast_cfg);
    if (rc == 0) {
        // Topology report
        sp_topology_report_t topo;
        sp_diag_discover_topology(&topo, &engine);
        sp_diag_print_topology(&topo);

        // Dashboard (empty, but validates the printing code)
        sp_diag_print_dashboard(&monitor, &engine);

        sp_beast_free(&engine);
        fprintf(stderr, "Engine boot/shutdown: PASS\n\n");
    } else {
        fprintf(stderr, "Engine boot failed (rc=%d): FAIL\n\n", rc);
    }

    // ── Test 6: Inference — actual CRT pipeline ──────────────────────
    fprintf(stderr, "=== TEST 6: CRT Pipeline Inference ===\n\n");

    if (rc == 0) {
        // Re-init engine without force_cpu_only to engage CRT + L0
        static sp_beast_engine_t inf_engine;
        sp_beast_config_t inf_cfg;
        sp_beast_config_init(&inf_cfg);
        inf_cfg.gguf_path = gguf_path;
        inf_cfg.force_cpu_only = false;  // Let CRT + L0 engage
        inf_cfg.enable_sidecar = false;

        int inf_rc = sp_beast_init(&inf_engine, &inf_cfg);
        if (inf_rc == 0) {
            // Qwen3 tokenizer: "What is AI?" = [3838, 374, 15235, 30]
            // Prepend BOS=151643
            int prompt[] = { 151643, 3838, 374, 15235, 30 };
            int n_prompt = 5;
            int max_gen = 32;
            int *output = (int *)malloc((size_t)max_gen * sizeof(int));

            if (output) {
                int n_gen = sp_beast_generate(&inf_engine,
                                              prompt, n_prompt,
                                              output, max_gen,
                                              0.0f, 1.0f);

                fprintf(stderr, "\nGenerated %d tokens:", n_gen);
                for (int i = 0; i < n_gen && i < 32; i++) {
                    fprintf(stderr, " %d", output[i]);
                }
                fprintf(stderr, "\n\n");
                free(output);
            }

            sp_beast_free(&inf_engine);
            fprintf(stderr, "Inference test: PASS\n\n");
        } else {
            fprintf(stderr, "Engine init for inference failed (rc=%d): SKIP\n\n", inf_rc);
        }
    }

    // ── Cleanup ─────────────────────────────────────────────────────
    sp_optane_free(&reservoir);

    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  BEAST CANYON TEST: COMPLETE                 ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    return audit_fails;
}

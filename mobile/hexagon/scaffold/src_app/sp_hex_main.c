// Shannon-Prime VHT2 - Hexagon DSP FastRPC scaffold (ARM-side main).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Forked from the Hexagon SDK 5.5.6.0 S22U sample. Qualcomm copyright on the
// scaffolding pattern; SP-specific code is AGPLv3.

#include "sp_hex_ext.h"
#include "rpcmem.h"
#include "remote.h"
#include "dsp_capabilities_utils.h"
#include "os_defines.h"

#include <stdlib.h>
#include <stdio.h>

static void print_usage(void) {
    printf(
        "Usage:\n"
        "    sp_hex [-d domain] [-U unsigned_PD] [-n head_dim]\n\n"
        "Options:\n"
        "  -d domain    : DSP domain (0=ADSP, 3=CDSP). Default: 3 (CDSP).\n"
        "  -U unsigned  : 1 = unsigned PD (default), 0 = signed PD.\n"
        "  -n head_dim  : VHT2 vector length (power of 2, >=8). Default: 128.\n"
        "\n"
        "Smoke test: feeds a deterministic fp32 vector through VHT2(VHT2(x))\n"
        "on the cDSP and reports max-abs error. Expected: ~0 to fp32 epsilon.\n"
    );
}

int main(int argc, char *argv[]) {
    int nErr = 0;
    int head_dim = 128;
    int domain = 3;
    int unsignedPDFlag = 1;
    bool isUnsignedPD_Enabled = false;
    int option = 0;

    while ((option = getopt(argc, argv, "d:U:n:h")) != -1) {
        switch (option) {
            case 'd': domain = atoi(optarg); break;
            case 'U': unsignedPDFlag = atoi(optarg); break;
            case 'n': head_dim = atoi(optarg); break;
            case 'h':
            default:
                print_usage();
                if (option == 'h') return 0;
        }
    }

    if (unsignedPDFlag == 1) {
        if (domain == CDSP_DOMAIN_ID || domain == CDSP1_DOMAIN_ID) {
            isUnsignedPD_Enabled = true;
        } else {
            printf("Overriding user request for unsigned PD. Only signed "
                   "offload is allowed on domain %d.\n", domain);
            unsignedPDFlag = 0;
        }
    }

    printf("\n[sp_hex] Shannon-Prime Hexagon DSP scaffold smoke test\n");
    printf("[sp_hex] Domain: %d  PD: %s  head_dim: %d\n",
           domain, unsignedPDFlag == 1 ? "unsigned" : "signed", head_dim);

    nErr = sp_hex_process(domain, head_dim, isUnsignedPD_Enabled);
    if (nErr) {
        printf("ERROR 0x%x: sp_hex smoke test failed\n", nErr);
        return nErr;
    }
    printf("[sp_hex] Direct IDL path: Success\n");

    int eErr = sp_hex_engine_smoke(head_dim);
    if (eErr) {
        printf("ERROR: engine API smoke test failed\n");
        return eErr;
    }

    int bErr = sp_hex_run_bench_sweep();
    if (bErr) {
        printf("ERROR: bench sweep failed\n");
        return bErr;
    }

    int dErr = sp_hex_disk_tier_proof(head_dim);
    if (dErr) {
        printf("ERROR: disk-tier proof failed\n");
        return dErr;
    }

    int vErr = sp_hex_compress_decompress_validate(head_dim);
    if (vErr) {
        printf("ERROR: per-element compress/decompress validate failed\n");
        return vErr;
    }

    // Path A.2 prototype CPU benchmark: fused decompress-matmul vs vanilla.
    // Workload sized to match Dolphin 1B at n_ctx=4096 (per-layer-head shape).
    // Override via env vars: SP_HEX_BENCH_NKV / SP_HEX_BENCH_HD / SP_HEX_BENCH_NQ.
    int bench_nkv = 4096;
    int bench_hd  = head_dim;
    int bench_nq  = 8;
    const char *e = NULL;
    if ((e = getenv("SP_HEX_BENCH_NKV")) && *e) bench_nkv = atoi(e);
    if ((e = getenv("SP_HEX_BENCH_HD"))  && *e) bench_hd  = atoi(e);
    if ((e = getenv("SP_HEX_BENCH_NQ"))  && *e) bench_nq  = atoi(e);
    int kqErr = sp_hex_kq_matmul_bench(bench_nkv, bench_hd, bench_nq);
    if (kqErr) {
        printf("ERROR: kq_matmul_bench failed (err=%d)\n", kqErr);
        // non-fatal — print and continue
    }
    printf("\n[sp_hex] All paths green\n\n");
    return 0;
}

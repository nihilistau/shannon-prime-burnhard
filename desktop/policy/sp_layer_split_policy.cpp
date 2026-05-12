// Shannon-Prime BurnHard — Layer Split & MoE Policy Orchestrator
// Target Model: Qwen3.6-35B-A3B-GGUF (40 Layers Total)

#include "sp_beast_canyon.h"
#include "sp_moe_curriculum.h"
#include <stdio.h>

// Hard constraints per user hardware target
#define RTX_2060_MAX_LAYERS 20
#define MODEL_TOTAL_LAYERS 40 

extern "C" void sp_beast_canyon_apply_split_policy(sp_beast_context_t* ctx) {
    // 1. Layer routing: Lock RTX 2060 12GB to exactly 20 layers.
    // The remaining 20 layers flow to CPU (AVX-512) and iGPU (UHD 750).
    ctx->layer_split.gpu_0_layers = RTX_2060_MAX_LAYERS;
    ctx->layer_split.host_layers = MODEL_TOTAL_LAYERS - RTX_2060_MAX_LAYERS;
    
    fprintf(stderr, "[burnhard] Layer split enforced: RTX 2060 = %d layers, Host/iGPU/Optane = %d layers\n", 
            ctx->layer_split.gpu_0_layers, ctx->layer_split.host_layers);

    // 2. Variable MoE Expert splits (CRT logic)
    // CRT arithmetic cleanly decouples the expert shards without all-reduce copies.
    // High-variance experts run on RTX; residue experts map to UHD 750 SVM memory.
    ctx->moe_policy.enable_crt_dispatch = true;
    ctx->moe_policy.expert_split_mode = SP_EXPERT_SPLIT_VARIABLE;
    
    fprintf(stderr, "[burnhard] CRT MoE variable expert splitting ENABLED. Routing via 24MB L0 SVM.\n");

    // 3. Mobile Sidecar via ADB (Samsung Galaxy S22 Ultra)
    // Re-engage speculative oracle draft model over the USB-C bridge.
    if (ctx->sidecar_available) {
        ctx->sidecar_mode = SP_SIDECAR_DRAFT_ORACLE;
        fprintf(stderr, "[burnhard] S22 Ultra heartbeat detected: FULL_PULSE mode active.\n");
    } else {
        ctx->sidecar_mode = SP_SIDECAR_NONE;
        fprintf(stderr, "[burnhard] S22 Ultra missing: Falling back to DESKTOP_SOLO mode.\n");
    }
}
# Phase 2.3 Design — sp_qnn shim implementation map

## Status (2026-05-01)

**Scaffold complete. Cross-compiles via NDK aarch64-clang against
QAIRT 2.45.40.260406 headers, links cleanly, runs on our S22 Ultra.**

Validated end-to-end on R5CT22445JA:
- `dlopen(libQnnHtp.so)` + `dlopen(libQnnSystem.so)` from `/data/local/tmp/sp_qnn/`
- `QnnInterface_getProviders` + `QnnSystemInterface_getProviders`
- Interface-version pick (matches major, takes highest minor)
- File read of our 32 MB `v69_attn_qwen3_4b.bin`
- `QnnSystemContext_create` + `QnnSystemContext_getBinaryInfo`
- Stubbed `execute` returns the documented `SP_QNN_ERR_EXECUTE`

Next-session fill-in: ~100-150 lines of QNN API calls, mapped below.

## API call sequence

### sp_qnn_load_binary — TODO sections (sp_qnn.c lines ~250-280)

The scaffold has done steps 1-2 (file read, binary-info extract) and
left steps 3-8 to fill in. Each step maps to one QNN API call from
the function table loaded into `g_lib.qnn`:

```
(3) Create log handle  (optional, but most backends want one):
    QnnLog_Callback_t cb = NULL;  // or our own stderr logger
    g_lib.qnn.logCreate(cb, QNN_LOG_LEVEL_WARN, &h->log);

(4) Create backend:
    g_lib.qnn.backendCreate(h->log, NULL /*config*/, &h->backend);

(5) Create device (HTP picks itself up from libQnnHtp.so symbols):
    g_lib.qnn.deviceCreate(h->log, NULL, &h->device);

(6) Create context FROM the binary we already loaded:
    g_lib.qnn.contextCreateFromBinary(h->backend, h->device,
                                       NULL /*config*/,
                                       h->bin_data, h->bin_size,
                                       &h->context, NULL /*profile*/);

(7) Retrieve the named graph (or the first one in binary_info):
    const char *gname = graph_name;
    if (!gname) {
        // Pull from binary_info->contextBinaryInfoVN.graphs[0].graphName
        // (versioned; see QnnSystemContext.h struct definitions)
    }
    g_lib.qnn.graphRetrieve(h->context, gname, &h->graph);

(8) Cache I/O tensor info from binary_info:
    // Walk graphs[0].inputTensors[0..N], graphs[0].outputTensors[0..M]
    // For each, populate h->inputs[i] = { name, rank, dims, dtype, ... }
    // The Qnn_Tensor_t structures in the binary info have everything
    // we need; just copy the relevant fields into our flat struct.
```

Key reference: `QnnSampleApp.cpp` lines ~700-900 (createFromBinary)
and lines ~1100-1300 (executeGraphs) on disk at:
`C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\examples\QNN\SampleApp\SampleApp\src\`

### sp_qnn_execute — TODO sections (sp_qnn.c lines ~310-330)

```
(1) Build Qnn_Tensor_t array per call (reusable across calls if shape
    doesn't change). For each input tensor:

    Qnn_Tensor_t in_t = QNN_TENSOR_INIT;  // macro from QnnTensor.h
    in_t.version = QNN_TENSOR_VERSION_1;
    in_t.v1.id          = h->inputs[i].id;             // from binary_info
    in_t.v1.name        = h->inputs[i].name;
    in_t.v1.type        = QNN_TENSOR_TYPE_APP_WRITE;
    in_t.v1.dataType    = (Qnn_DataType_t)h->inputs[i].dtype;
    in_t.v1.rank        = h->inputs[i].rank;
    in_t.v1.dimensions  = (uint32_t *)h->inputs[i].dims;
    in_t.v1.memType     = QNN_TENSORMEMTYPE_RAW;
    in_t.v1.clientBuf.data     = (void *)inputs[i];
    in_t.v1.clientBuf.dataSize = (uint32_t)input_sizes[i];

    // Same for outputs but with type = QNN_TENSOR_TYPE_APP_READ.

(2) Time the call:
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

(3) Execute:
    g_lib.qnn.graphExecute(h->graph,
                            in_tensors, n_in,
                            out_tensors, n_out,
                            NULL /*profile*/,
                            NULL /*signal*/);

(4) Capture timing:
    gettimeofday(&t1, NULL);
    if (out_exec_us) *out_exec_us = ...
```

### sp_qnn_destroy — minor TODOs

Currently frees only our own buffers. Add:
```
g_lib.qnn.contextFree(h->context, NULL);
g_lib.qnn.deviceFree(h->device);
g_lib.qnn.backendFree(h->backend);
g_lib.qnn.logFree(h->log);
g_lib.qsys.systemContextFree(h->sysCtx);  // requires storing sysCtx in handle
```

Order matters — backend last because device + context depend on it.

## Validation plan once filled in

```bash
# Push the new binary
adb push test_sp_qnn /data/local/tmp/sp_qnn/

# Run against our existing AI-Hub-validated binary
adb shell "cd /data/local/tmp/sp_qnn; LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./test_sp_qnn v69_attn_qwen3_4b.bin"

# Expected output (when execute() is filled in):
#   sp_qnn_init OK
#   load_binary OK (1 input, 1 output)
#     input  x:        rank=3, dims=[1,64,2048], fp32, 524288 bytes
#     output output_0: rank=3, dims=[1,64,2048], fp32, 524288 bytes
#   bench: 50 iters
#     min: ~1556 us  (matches qnn-net-run Phase 2.2 measurement)
#     avg: ~2496 us
#     max: ~4229 us
```

The min should match Phase 2.2's 1.5 ms steady-state to confirm we're
not introducing overhead vs the reference `qnn-net-run`.

## Integration into shannon-prime-llama (Phase 2.3 stage 2)

Once `libsp_qnn.so` works standalone, the integration into
shannon-prime-llama is:

1. Add `lib/shannon-prime/backends/sp_qnn_kernel.{h,c}` — thin C
   wrapper that initializes one sp_qnn_handle per attention layer
   at model load.

2. Add a custom ggml op (call it `GGML_OP_SP_QNN_KQ`) that dispatches
   to `sp_qnn_execute` instead of running the matmul on CPU.

3. In `shannon-prime-llama/src/llama-graph.cpp`, intercept the K@Q^T
   matmul construction and substitute the new op when env var
   `SP_USE_QNN_KQ=1` is set.

4. At inference time, the shannon-prime-hexagon FastRPC bridge
   already produces decompressed K rows; wrap them in rpcmem buffers
   and pass directly to sp_qnn_execute. No copy needed.

The architectural fit is clean because:
- shannon-prime-hexagon ships rpcmem-backed buffers
- QNN graphs accept rpcmem pointers directly (`QNN_TENSORMEMTYPE_MEMHANDLE`)
- The custom-op pattern in shannon-prime-llama is already
  established for FUSED_KQ (commit ab4c65f), so no new infrastructure

Concrete entry: extend the existing `kcap` / FUSED_KQ pathway with
a third variant `SP_QNN_KQ` gated on env var. CPU path, Hexagon-HVX
path, QNN-HTP path — three competing kernels under the same op.

## Risks to surface as we fill in

- **Tensor ID vs Name binding**: QNN may require the tensor `id`
  field to match the IDs encoded in the context binary. The
  `binary_info` has these IDs; our copy may need to preserve them.
- **Memory lifetime**: `binary_info` pointers are owned by the
  `sysCtx` handle. We need to either copy the strings/dims out, or
  hold sysCtx open for the lifetime of `h`. Currently leak — needs
  fix in destroy().
- **Tensor `type` (APP_WRITE/READ)**: must match what was set at
  compile time. AI Hub sets graph inputs as APP_WRITE and outputs
  as APP_READ; that's what we'll use.
- **Profile handle**: optional but helpful for in-process latency
  beyond wall-clock. Mirror `qnn-net-run --profiling_level basic`
  behavior in a Phase 2.3.1 follow-up.

## File inventory

```
backends/qnn_aihub/sp_qnn_runner/
  sp_qnn.h              public API (138 LOC)
  sp_qnn.c              scaffold + dlopen + binary-info extract (340 LOC)
                        (TODOs marked, ~150 LOC to add)
  test_sp_qnn.c         smoke driver (60 LOC)
  build.cmd             NDK aarch64 cross-compile
  PHASE_2_3_DESIGN.md   this document
```

Also produced: `libsp_qnn.so` (10 KB) + `test_sp_qnn` (11 KB) — both
gitignored as build artifacts.

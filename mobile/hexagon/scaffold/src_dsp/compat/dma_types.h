// Compat shim — same role as dma_def.h, just a different SDK-side filename
// that ubwcdma_utils.h and the upstream dma_raw_blur_rw_async/dsp/_run.cpp
// both #include. mini_hexagon_dma.h declares everything either of them needs.
//
// AGPLv3 — see mini_hexagon_dma.h for the upstream Halide attribution.
#ifndef SP_COMPAT_DMA_TYPES_H
#define SP_COMPAT_DMA_TYPES_H
#include <stdint.h>
#include <stdbool.h>
#include "mini_hexagon_dma.h"
#endif

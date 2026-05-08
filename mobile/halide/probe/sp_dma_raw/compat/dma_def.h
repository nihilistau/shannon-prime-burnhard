// Compat shim — reroutes the Hexagon SDK Compute add-on header `dma_def.h`
// to the Halide-vendored mini_hexagon_dma.h, which declares the same enums
// (t_eDmaFmt with eDmaFmt_RawData=0 etc.) as standalone open-source code.
//
// Used because the Compute add-on isn't installed; the addon would normally
// provide this header at:
//   $(HEXAGON_SDK_ROOT)/addons/compute/libs/ubwcdma/inc/dma_def.h
// We sit on the build's -I path BEFORE the addon path so this wins.
//
// AGPLv3 (Shannon-Prime) — Halide is MIT-licensed, attribution in mini_hexagon_dma.h.
#ifndef SP_COMPAT_DMA_DEF_H
#define SP_COMPAT_DMA_DEF_H
#include <stdint.h>
#include <stdbool.h>
#include "mini_hexagon_dma.h"
#endif

// Compat shim — reroutes the SDK's `dmaWrapper.h` to mini_hexagon_dma.h
// PLUS adds the one prototype the mini header is missing:
// `nDmaWrapper_GetFramesize`, used by ubwcdma_utils.h::getFrameSize.
//
// Signature derived from how ubwcdma_utils.h calls it:
//    int sz = nDmaWrapper_GetFramesize(frm, &pStFrameProp, tf);
// where `frm` is t_eDmaFmt, &pStFrameProp is t_StDmaWrapper_FrameProp*, and
// `tf` is the bool "is UBWC" flag. Returns int (frame size in bytes).
//
// At link time this resolves against ubwcdma_dynlib.so. At runtime on the
// phone the cDSP loader resolves to /vendor/dsp/cdsp/ubwcdma_dynlib.so.
//
// AGPLv3 — see mini_hexagon_dma.h for upstream Halide attribution.
#ifndef SP_COMPAT_DMAWRAPPER_H
#define SP_COMPAT_DMAWRAPPER_H
#include <stdint.h>
#include <stdbool.h>
#include "mini_hexagon_dma.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Get the total frame size in bytes for the given format/dimensions.
/// Implemented in ubwcdma_dynlib.so (cDSP-resident); we link against the
/// staged copy on the host and the device loader resolves to the on-phone
/// /vendor/dsp/cdsp/ubwcdma_dynlib.so at runtime.
extern int32 nDmaWrapper_GetFramesize(t_eDmaFmt eFmt,
                                      t_StDmaWrapper_FrameProp *pStFrameProp,
                                      bool bIsUbwc);

#ifdef __cplusplus
}
#endif
#endif

// Vendored from halide/Halide @ master:
//   https://github.com/halide/Halide/blob/master/src/runtime/mini_hexagon_dma.h
// Halide is MIT-licensed; this file is reproduced unchanged with attribution.
//
// We use this to avoid depending on the Qualcomm Hexagon SDK Compute add-on,
// which isn't installed on this machine. The Halide team wrote this exactly
// because they need the same Hexagon DMA wrapper types/enums/prototypes
// without taking a hard dependency on the closed-source SDK headers
// (dma_def.h, dma_types.h, dmaWrapper.h).
//
// This file defines: t_eDmaFmt, t_eDmaWrapper_TransationType,
// t_StDmaWrapper_Roi, t_StDmaWrapper_FrameProp, t_StDmaWrapper_RoiAlignInfo,
// t_StDmaWrapper_DmaTransferSetup, t_DmaWrapper_DmaEngineHandle, and the
// nDmaWrapper_* function prototypes.
//
// What it DOESN'T declare (but ubwcdma_utils.h needs):
//   nDmaWrapper_GetFramesize(t_eDmaFmt, t_StDmaWrapper_FrameProp*, bool)
// We add that prototype in dmaWrapper.h (our compat shim) so the existing
// ubwcdma_utils.h compiles unchanged.
//
// At link time the symbols resolve against ubwcdma_dynlib.so (vendored copy
// at C:\Qualcomm\dsp\cdsp\ubwcdma_dynlib.so or pulled from device).
// At runtime on the phone, the cDSP loader resolves to the real
// /vendor/dsp/cdsp/ubwcdma_dynlib.so via standard library search.

// This header declares the Hexagon DMA API, without depending on the Hexagon SDK.

#ifndef MINI_HEXAGON_DMA_H
#define MINI_HEXAGON_DMA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t uint16;
typedef uint32_t uint32;
typedef int32_t int32;
typedef unsigned long addr_t;

typedef unsigned int qurt_size_t;
typedef unsigned int qurt_mem_pool_t;
#define HALIDE_HEXAGON_ENUM enum __attribute__((aligned(4)))

__inline static int align(int x, int a) {
    return ((x + a - 1) & (~(a - 1)));
}

// NOTE: Halide's upstream mini_hexagon_dma.h defines `QURT_EOK = 0` here
// as a fallback for runtimes that don't pull in the SDK qurt headers.
// We *do* include the SDK qurt headers (via sysmon_cachelock.h, qurt_error.h
// in dsp/sp_dma_raw_run.cpp), so QURT_EOK is already a macro. Keeping the
// upstream line creates a name collision. Removed.

/**
 * Power Corner vote
 */
#define PW_MIN_SVS 0
#define PW_SVS2 1
#define PW_SVS 2
#define PW_SVS_L1 3
#define PW_NORMAL 4
#define PW_NORMAL_L1 5
#define PW_TURBO 6

/**
 * Format IDs
 */
typedef HALIDE_HEXAGON_ENUM{
    eDmaFmt_RawData,
    eDmaFmt_NV12,
    eDmaFmt_NV12_Y,
    eDmaFmt_NV12_UV,
    eDmaFmt_P010,
    eDmaFmt_P010_Y,
    eDmaFmt_P010_UV,
    eDmaFmt_TP10,
    eDmaFmt_TP10_Y,
    eDmaFmt_TP10_UV,
    eDmaFmt_NV124R,
    eDmaFmt_NV124R_Y,
    eDmaFmt_NV124R_UV,
    eDmaFmt_Invalid,
    eDmaFmt_MAX,
} t_eDmaFmt;

/**
 * DMA status (forward decl, currently unused)
 */
typedef void *t_stDmaWrapperDmaStats;

/**
 * Transfer type
 */
typedef HALIDE_HEXAGON_ENUM eDmaWrapper_TransationType{
    /// DDR to L2 transfer
    eDmaWrapper_DdrToL2,
    /// L2 to DDR transfer
    eDmaWrapper_L2ToDdr,
} t_eDmaWrapper_TransationType;

/**
 * Roi Properties
 */
typedef struct stDmaWrapper_Roi {
    uint16 u16X;
    uint16 u16Y;
    uint16 u16W;
    uint16 u16H;
} t_StDmaWrapper_Roi;

/**
 * Frame Properties
 */
typedef struct stDmaWrapper_FrameProp {
    /// Starting physical address to buffer
    addr_t aAddr;
    /// Frame height in pixels
    uint16 u16H;
    /// Frame width in pixels
    uint16 u16W;
    /// Frame stride in pixels
    uint16 u16Stride;
} t_StDmaWrapper_FrameProp;

/**
 * Roi alignment
 */
typedef struct stDmaWrapper_RoiAlignInfo {
    uint16 u16W;
    uint16 u16H;
} t_StDmaWrapper_RoiAlignInfo;

/**
 * DmaTransferSetup Properties
 */
typedef struct stDmaWrapper_DmaTransferSetup {
    uint16 u16FrameW;
    uint16 u16FrameH;
    uint16 u16FrameStride;
    uint16 u16RoiX;
    uint16 u16RoiY;
    uint16 u16RoiW;
    uint16 u16RoiH;
    uint16 u16RoiStride;
    void *pDescBuf;
    void *pTcmDataBuf;
    void *pFrameBuf;
    uint16 bIsFmtUbwc;
    uint16 bUse16BitPaddingInL2;
    t_eDmaFmt eFmt;
    t_eDmaWrapper_TransationType eTransferType;
} t_StDmaWrapper_DmaTransferSetup;

// NOTE: Upstream mini_hexagon_dma.h declares HAP_cache_lock/unlock as
// fallbacks when the SDK isn't present, but uses an older signature
// (void** for paddr_ptr). The SDK's sysmon_cachelock.h that our DSP
// code already includes uses (unsigned long long *) instead — that's
// the load-bearing signature on V66+. We pull from the SDK header,
// so the redeclarations are removed here to avoid a conflicting-types
// error.

/**
 * Handle for wrapper DMA engine
 */
typedef void *t_DmaWrapper_DmaEngineHandle;

extern t_DmaWrapper_DmaEngineHandle hDmaWrapper_AllocDma(void);
extern int32 nDmaWrapper_FreeDma(t_DmaWrapper_DmaEngineHandle hDmaHandle);
extern int32 nDmaWrapper_Move(t_DmaWrapper_DmaEngineHandle hDmaHandle);
extern int32 nDmaWrapper_Wait(t_DmaWrapper_DmaEngineHandle hDmaHandle);
extern int32 nDmaWrapper_FinishFrame(t_DmaWrapper_DmaEngineHandle hDmaHandle);
extern int32 nDmaWrapper_GetRecommendedWalkSize(t_eDmaFmt eFmtId, bool bIsUbwc,
                                                t_StDmaWrapper_RoiAlignInfo *pStWalkSize);
extern int32 nDmaWrapper_GetDescbuffsize(t_eDmaFmt *aeFmtId, uint16 nsize);
extern int32 nDmaWrapper_GetRecommendedIntermBufStride(t_eDmaFmt eFmtId,
                                                       t_StDmaWrapper_RoiAlignInfo *pStRoiSize,
                                                       bool bIsUbwc);
extern int32 nDmaWrapper_GetRecommendedIntermBufSize(t_eDmaFmt eFmtId, bool bUse16BitPaddingInL2,
                                                     t_StDmaWrapper_RoiAlignInfo *pStRoiSize,
                                                     bool bIsUbwc, uint16 u16IntermBufStride);
extern int32 nDmaWrapper_DmaTransferSetup(t_DmaWrapper_DmaEngineHandle hDmaHandle,
                                          t_StDmaWrapper_DmaTransferSetup *stpDmaTransferParm);
extern int32 nDmaWrapper_PowerVoting(uint32 cornercase);

#ifdef __cplusplus
}
#endif

#endif

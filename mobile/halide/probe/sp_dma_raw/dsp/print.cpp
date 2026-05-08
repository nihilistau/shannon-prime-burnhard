// Mode D Stage 1 probe — Halide runtime print/error glue for the cDSP.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3.
//
// The Halide-generated kernel calls halide_print on debug paths and
// halide_error on failure. These are weak symbols supplied by the Halide
// runtime at link time; we override them on the cDSP so the messages land
// in the FARF log instead of stdout (which doesn't exist in the FastRPC
// PD environment).
//
// Mirrors the dma_raw_blur_rw_async/dsp/print.cpp pattern in the Halide
// SDK example tree — same shape, same FARF level wiring. Required by both
// the Makefile.rules path ($(BIN)/print.o prerequisite of lib%_skel.so)
// and the test-*.cmd Windows build path.

#include "HAP_farf.h"

#undef FARF_LOW
#define FARF_LOW 1
#undef FARF_HIGH
#define FARF_HIGH 1

extern "C" {

void halide_print(void *user_context, const char *msg) {
    (void)user_context;
    FARF(HIGH, "halide_print %s\n", msg);
}

void halide_error(void *user_context, const char *msg) {
    FARF(LOW, "halide_error\n");
    halide_print(user_context, msg);
}

}

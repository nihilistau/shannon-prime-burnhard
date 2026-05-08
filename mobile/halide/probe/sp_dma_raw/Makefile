# Mode D Stage 1 probe Makefile.
# Patterned after C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\Examples\standalone\
#   device\apps\dma_raw_blur_rw_async\Makefile
#
# This is a hexagon-standalone build that uses the Hexagon SDK's apps
# Makefile.common + Makefile.rules infrastructure. The same env vars
# (HALIDE_ROOT, HEXAGON_SDK_ROOT) the parent dma_raw_blur_rw_async
# expects must be in scope.
#
# Validated entry path on Windows (Hexagon SDK setup_sdk_env.cmd then
# this Makefile invoked via mingw32-make from C:\Qualcomm\Hexagon_SDK\...
# environment).

# Path canonicalization. On Windows, mingw32-make's `$(shell pwd)` returns
# `C:\...` (backslashes via gow's pwd.EXE) which then mixes with the literal
# forward-slash path components in Makefile.common's expansions. GNU Make
# pattern-rule matching is string-based — mixed slashes break it (the
# symptom is "No rule to make target ...lib<TEST>_skel.so" even though the
# rule exists in Makefile.rules). Force forward slashes everywhere.
#
# NOTE: even with this fix, Make-driven builds on Windows still hit two
# more blockers: (1) Makefile.common's `GENERATOR_DEPS` references
# `libHalide.a` which doesn't exist on the Windows Halide install (it's
# `Halide.lib`); (2) the Compute add-on (`addons/compute/libs/ubwcdma`)
# is required for both compile (`#include "dmaWrapper.h"`) and link
# (`ubwcdma_dynlib.so`). The Windows-native build path that works is
# `test-sp_dma_raw.cmd` in this directory.
ifeq ($(OS),Windows_NT)
  CURR_DIR := $(subst \,/,$(shell cmd /c cd))
else
  CURR_DIR := $(shell pwd)
endif
DEVICE_STANDALONE_TEST_TOP_DIR = ${CURR_DIR}/../..
TESTNAME = $(shell basename ${CURR_DIR})

# RAW frame parameters for the probe:
#   width  = packed_band_stride = 128 bytes
#   height = N_BANDS            = 4
#   stride = same as width
#   format = 0 (RAW)
#   is_input_ubwc = 0, is_output_ubwc = 0
# (No on-disk input / validation file — the host generates and verifies
# in-process.)
TEST_ARGS=100

FEATURES ?= hexagon_dma-hvx_128
Q6_VERSION = 66
ifeq ($(shell expr $$Q6_VERSION + 0 \> 66), 1)
  Q6_VERSION = $(shell echo $$Q6_VERSION)
endif

# Pick this up from the parent SDK examples tree if present.
include $(DEVICE_STANDALONE_TEST_TOP_DIR)/apps/Makefile.common

include $(DEVICE_STANDALONE_TEST_TOP_DIR)/apps/Makefile.rules

# Shannon-Prime Halide kernels — build hooks.
# Patterned after Qualcomm Halide examples at:
#   C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\Examples\offload\apps\hexagon_benchmarks\Makefile
#
# VALIDATED 2026-05-01: end-to-end build succeeds on Windows with
# Visual Studio 2019 BuildTools + HALIDE_Tools 2.4.07. Generator
# binary built via cl.exe + Halide.lib, generator run produces V69
# HVX object (sp_band_dequantize.o, 87 KB) plus C header. Same
# .stmt.html shows prefetch hint emitted as scheduled.
#
# Two-stage build:
#   1. Compile sp_band_dequantize_generator.cpp + Halide host libs
#      → sp_band_dequantize.generator (host executable)
#   2. Run the generator with a Hexagon target
#      → sp_band_dequantize.o + sp_band_dequantize.h
#
# The .o gets linked into libsp_hex_skel.so (via the Hexagon SDK build
# rules in shannon-prime/backends/hexagon/scaffold/CMakeLists.txt) as
# a replacement for sp_band_dequantize_scalar's hot loop.
#
# WINDOWS BUILD (validated path):
#   set VCBAT="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
#   call %VCBAT%
#   cl /nologo /EHsc /std:c++17 /O2 ^
#       /I C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\include ^
#       sp_band_dequantize_generator.cpp ^
#       C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\tools\GenGen.cpp ^
#       /link /LIBPATH:C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\lib Halide.lib ^
#       /OUT:sp_band_dequantize.generator.exe
#   set PATH=C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\bin;%PATH%
#   sp_band_dequantize.generator.exe -g sp_band_dequantize -o build ^
#       -e o,h,html target=hexagon-32-noos-hvx_128
#
# LINUX BUILD (untested, original Makefile rules below for reference).
#
# Variables expected in the env (set by Qualcomm setup-env.cmd or the
# hexagon SDK setup_sdk_env):
#   HALIDE_ROOT      : C:\Qualcomm\HALIDE_Tools\2.4.07\Halide
#   HEXAGON_SDK_ROOT : C:\Qualcomm\Hexagon_SDK\5.5.6.0
#   HEXAGON_TOOLS_ROOT: HEXAGON_SDK_ROOT\tools\HEXAGON_Tools\<ver>\Tools
#
# Targets:
#   make            — build the .o for hvx_128 (V69)
#   make generator  — just build the generator binary
#   make clean      — remove build artifacts
#
# Then in the Hexagon scaffold's CMakeLists.txt, link
# build/sp_band_dequantize.o into the sp_hex_skel target.

HALIDE_ROOT ?= /c/Qualcomm/HALIDE_Tools/2.4.07/Halide
TARGET ?= hexagon-32-noos-hvx_128
BUILD ?= build
GEN_SRC = sp_band_dequantize_generator.cpp
GEN_BIN = $(BUILD)/sp_band_dequantize.generator
KERNEL_OUT = $(BUILD)/sp_band_dequantize

CXX ?= g++
CXXFLAGS = -std=c++17 -O2 -fPIC -I$(HALIDE_ROOT)/include
LDFLAGS = -L$(HALIDE_ROOT)/lib -lHalide -ldl -lpthread -lz

# A generator binary "registers" the sp_band_dequantize generator class
# at static-init time and exposes it via Halide's GeneratorRegistry.
# The Halide tools/GenGen.cpp main is what handles the CLI args (-g,
# -o, -e, target=...) — we link against it and it provides main().
$(GEN_BIN): $(GEN_SRC)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(HALIDE_ROOT)/tools/GenGen.cpp -o $@ $(LDFLAGS)

# Run the generator with the V69 HVX target. Output:
#   sp_band_dequantize.o    : Hexagon native object
#   sp_band_dequantize.h    : C header with the function signature
#   sp_band_dequantize.html : auto-schedule / loop-nest visualization
$(KERNEL_OUT).o: $(GEN_BIN)
	$(GEN_BIN) -g sp_band_dequantize -o $(BUILD) -e o,h,html target=$(TARGET)

generator: $(GEN_BIN)
	@echo "Generator built: $(GEN_BIN)"

kernel: $(KERNEL_OUT).o
	@echo "Kernel built: $(KERNEL_OUT).o"

clean:
	rm -rf $(BUILD)

.PHONY: generator kernel clean

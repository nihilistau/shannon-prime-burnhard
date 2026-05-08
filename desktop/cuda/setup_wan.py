"""
Setup script for the Shannon-Prime VHT2 CUDA extension (Wan video generation).

Builds `shannon_prime_cuda_wan` — a Python module exposing GPU-accelerated
VHT2 butterfly operations for cross-attention KV compression in Wan DiT.

Usage:
    cd backends/cuda
    python setup_wan.py build_ext --inplace

After build, the .so/.pyd lands in backends/cuda/ and is importable as:
    import shannon_prime_cuda_wan as sp_cuda

GPU requirements: CUDA 11+, sm_75+ (RTX 20xx / Turing minimum).
PyTorch requirement: 2.0+ with matching CUDA toolkit.

Architecture flags:
  sm_75 = RTX 2060 / 2070 / 2080  (Turing)
  sm_80 = A100 / RTX 3090          (Ampere)
  sm_86 = RTX 3080 / 3090          (Ampere consumer)
  sm_89 = RTX 4090                  (Ada Lovelace)
  sm_90 = H100                      (Hopper)

Add multiple --generate-code flags for multi-arch fat binary.
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))

# Core library include path — must be defined before sources list.
# Use realpath() to resolve the absolute path before any virtual-drive or
# subst shenanigans. Relative paths like ../../core break when building from
# a subst'd drive letter (X:\) because Windows doesn't traverse up through it.
_CORE_INC = os.path.realpath(os.path.join(HERE, "..", "..", "core"))

# ── Source files ───────────────────────────────────────────────────────────────

sources = [
    os.path.join(HERE, "vht2_wan_ext.cpp"),        # PyTorch binding
    os.path.join(HERE, "shannon_prime_cuda.cu"),   # VHT2 + Möbius + band quant kernels
    os.path.join(HERE, "shannon_prime_sqfree.cu"), # Vilenkin non-p2 stages
    # Core C library — provides sp_mobius_mask_init, sp_band_config_init,
    # sp_sqfree_pad_dim, sp_knight_mask_* etc. that the CUDA files call.
    os.path.join(_CORE_INC, "shannon_prime.c"),
    os.path.join(_CORE_INC, "shannon_prime_sqfree.c"),
]

# ── Compiler flags ─────────────────────────────────────────────────────────────

_CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    # Turing (RTX 20xx, the dev machine): sm_75
    "-gencode=arch=compute_75,code=sm_75",
    # Ampere: sm_80 + sm_86 — comment out if build times matter
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    # Ada Lovelace: sm_89
    "-gencode=arch=compute_89,code=sm_89",
    # PTX for forward compat
    "-gencode=arch=compute_89,code=compute_89",
]

_CXX_FLAGS = [
    "-O3",
    "-std=c++17",
]

# ── Extension definition ───────────────────────────────────────────────────────

ext = CUDAExtension(
    name="shannon_prime_cuda_wan",
    sources=sources,
    include_dirs=[HERE, _CORE_INC],
    extra_compile_args={
        "cxx":  _CXX_FLAGS,
        "nvcc": _CUDA_FLAGS,
    },
    # -lcuda is implicit on Windows via CUDA toolkit; omit to avoid linker errors
    extra_link_args=[],
)

# ── Build ──────────────────────────────────────────────────────────────────────

setup(
    name="shannon_prime_cuda_wan",
    version="1.0.0",
    description="Shannon-Prime VHT2 CUDA kernel for Wan video generation",
    author="Ray Daniels",
    author_email="raydaniels@gmail.com",
    license="AGPLv3",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)

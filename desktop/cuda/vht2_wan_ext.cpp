// Shannon-Prime VHT2: PyTorch Extension Binding for Wan Video Generation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Provides Python-callable wrappers around the CUDA VHT2 kernels,
// specifically tuned for the Wan DiT architecture:
//   head_dim = 128  (= 2^7, pure Hadamard butterfly — 7 stages)
//   layout   = [batch * tokens, head_dim]  (2D flat, vectors contiguous)
//
// The key operations exposed:
//   vht2_forward      — in-place butterfly (forward = inverse, self-inverse)
//   vht2_compress     — butterfly → apply skeleton mask
//   vht2_decompress   — apply skeleton mask → butterfly (same as compress)
//   vht2_roundtrip    — forward → mask → forward (fused compress+decompress)
//
// All operations accept and return float32 CUDA tensors. The skeleton mask
// is a bool/uint8 tensor of shape [head_dim] marking squarefree positions.

#include <torch/extension.h>
#include <cuda_runtime.h>

// ── Forward declarations of CUDA kernel launchers (in shannon_prime_cuda.cu) ──
// stream=nullptr → default CUDA stream (same stream PyTorch uses by default).
// Avoids pulling in ATen CUDA stream headers which add build complexity.
extern "C" void sp_cuda_vht2_forward(float *d_data, int n, int n_vecs,
                                      void *stream);

// ── Helpers ──────────────────────────────────────────────────────────────────

static inline void check_wan_tensor(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(),
        name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32,
        name, " must be float32 (got ", t.scalar_type(), ")");
    TORCH_CHECK(t.is_contiguous(),
        name, " must be contiguous");
    TORCH_CHECK(t.size(-1) == 128,
        name, " last dim must be 128 (Wan head_dim), got ", t.size(-1));
}

// Default CUDA stream — safe to use since all tensors are already on CUDA and
// PyTorch serializes kernel launches on the default stream by default.
static constexpr void *kDefaultStream = nullptr;

// ── Public API ────────────────────────────────────────────────────────────────

// vht2_forward: in-place Hadamard butterfly on [N, 128] float32 CUDA tensor.
// Self-inverse: applying twice recovers the input (up to floating-point noise).
// Returns a new contiguous tensor (does not modify input in-place).
torch::Tensor vht2_forward(const torch::Tensor &input) {
    check_wan_tensor(input, "input");
    auto out   = input.clone();
    int  n_vec = (int)(out.numel() / 128);
    sp_cuda_vht2_forward(out.data_ptr<float>(), 128, n_vec,
                          kDefaultStream);
    return out;
}

// vht2_inplace: modify tensor in-place. Saves one allocation vs vht2_forward.
// Caller is responsible for ensuring the tensor has no aliased views.
void vht2_inplace(torch::Tensor &t) {
    check_wan_tensor(t, "t");
    int n_vec = (int)(t.numel() / 128);
    sp_cuda_vht2_forward(t.data_ptr<float>(), 128, n_vec,
                          kDefaultStream);
}

// vht2_compress: transform + zero non-skeleton positions.
//   input:     [N, 128] float32 CUDA
//   skel_mask: [128]    bool or uint8 CUDA (1 = keep, 0 = zero)
//   Returns:   [N, 128] compressed coefficients (non-skeleton = 0)
torch::Tensor vht2_compress(const torch::Tensor &input,
                             const torch::Tensor &skel_mask) {
    check_wan_tensor(input, "input");
    TORCH_CHECK(skel_mask.is_cuda(), "skel_mask must be CUDA");
    TORCH_CHECK(skel_mask.numel() == 128,
        "skel_mask must have 128 elements (Wan head_dim)");

    auto out  = input.clone();
    int n_vec = (int)(out.numel() / 128);

    // Forward butterfly
    sp_cuda_vht2_forward(out.data_ptr<float>(), 128, n_vec,
                          kDefaultStream);

    // Apply skeleton mask: broadcast [128] mask over [N, 128] output.
    // mul_ with float mask (0.0 / 1.0) zeros non-skeleton positions.
    auto mask_f = skel_mask.to(torch::kFloat32).view({1, 128});
    out.mul_(mask_f);

    return out;
}

// vht2_decompress: apply skeleton mask + inverse butterfly (= forward butterfly,
// since VHT2 is self-inverse). Equivalent to vht2_compress but communicates
// intent clearly — the mask on compressed coefficients, then butterfly out.
torch::Tensor vht2_decompress(const torch::Tensor &coeffs,
                               const torch::Tensor &skel_mask) {
    check_wan_tensor(coeffs, "coeffs");
    TORCH_CHECK(skel_mask.is_cuda(), "skel_mask must be CUDA");
    TORCH_CHECK(skel_mask.numel() == 128, "skel_mask must have 128 elements");

    // Re-apply mask (ensures non-skeleton positions are truly zero,
    // handles any numerical noise from previous compress step)
    auto mask_f = skel_mask.to(torch::kFloat32).view({1, 128});
    auto out    = coeffs.mul(mask_f);   // new tensor, masked
    int n_vec   = (int)(out.numel() / 128);

    // Inverse butterfly (= forward, self-inverse)
    sp_cuda_vht2_forward(out.data_ptr<float>(), 128, n_vec,
                          kDefaultStream);
    return out;
}

// vht2_roundtrip: fused compress→decompress in one function call.
//   forward butterfly → mask → forward butterfly (reconstruction)
//   Equivalent to vht2_decompress(vht2_compress(input, mask), mask)
//   but avoids the intermediate tensor allocation.
torch::Tensor vht2_roundtrip(const torch::Tensor &input,
                              const torch::Tensor &skel_mask) {
    check_wan_tensor(input, "input");
    TORCH_CHECK(skel_mask.is_cuda(), "skel_mask must be CUDA");
    TORCH_CHECK(skel_mask.numel() == 128, "skel_mask must have 128 elements");

    auto out    = input.clone();
    int n_vec   = (int)(out.numel() / 128);
    auto stream = kDefaultStream;
    auto mask_f = skel_mask.to(torch::kFloat32).view({1, 128});

    // Forward (compress)
    sp_cuda_vht2_forward(out.data_ptr<float>(), 128, n_vec, stream);
    out.mul_(mask_f);

    // Inverse (decompress) — self-inverse, no separate kernel needed
    sp_cuda_vht2_forward(out.data_ptr<float>(), 128, n_vec, stream);

    return out;
}

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Shannon-Prime VHT2 CUDA extension for Wan video generation. "
              "Provides GPU-accelerated Hadamard butterfly (head_dim=128) with "
              "squarefree skeleton masking for cross-attention KV compression.";

    m.def("forward",     &vht2_forward,    "VHT2 butterfly forward (returns new tensor)",
          py::arg("input"));
    m.def("inplace",     &vht2_inplace,    "VHT2 butterfly in-place (modifies tensor)",
          py::arg("t"));
    m.def("compress",    &vht2_compress,   "VHT2 forward + skeleton mask",
          py::arg("input"), py::arg("skel_mask"));
    m.def("decompress",  &vht2_decompress, "Skeleton mask + VHT2 inverse",
          py::arg("coeffs"), py::arg("skel_mask"));
    m.def("roundtrip",   &vht2_roundtrip,  "Fused compress+decompress (VHT2→mask→VHT2)",
          py::arg("input"), py::arg("skel_mask"));
}

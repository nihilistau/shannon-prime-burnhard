// sp_beast_gpu_q4k.cu — Q4_K-resident matvec kernel for RTX dispatch.
//
// Stages weights as raw Q4_K (144 bytes / 256 elements) in VRAM and runs
// a hand-rolled CUDA kernel that dequants inline + dot-products against
// an fp32 input vector. This avoids the cuBLAS gemv-on-fp16 path that
// regressed 12x — by keeping weights compressed (~half the VRAM bytes
// of fp16) and fusing dequant+FMA, we hit close to peak VRAM bandwidth.
//
// Per super-block layout (matches beast_dequant_q4_K in sp_beast_canyon.c):
//   bytes  0..1   : fp16 d_super  (per super-block scale)
//   bytes  2..3   : fp16 dmin     (per super-block min)
//   bytes  4..15  : 12 bytes of packed 6-bit sub-block scales/mins
//   bytes 16..143 : 128 bytes of 4-bit packed quants (256 nibbles)
//
// Sub-block sb in [0,8): scale/min via the get_scale_min_k4 unpack.
// Element ie in [0,256) of the super-block:
//   sb = ie / 32, off = ie % 32
//   byte = qs[(sb/2) * 32 + off]
//   nibble = (sb & 1) ? (byte >> 4) : (byte & 0x0F)
//   w = d_super * sc[sb] * nibble - dmin * mn[sb]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {

// Warp-per-row Q4_K matvec.
//
// Block = 256 threads = 8 warps -> 8 rows per block. Each warp's 32
// threads cooperatively process one row of 8 super-blocks (n_in=2048).
// Per super-block, the 32 threads each handle 8 of the 256 elements
// — so threads read 8 consecutive nibble bytes (coalesced 32×1 byte
// reads inside a warp), accumulate locally, then warp-reduce at the
// end via __shfl_xor_sync.
//
// Memory access:
//   * qs base for super-block is at qpair_bytes = qs + (sb/2) * 32.
//     Each thread reads bytes [thread_idx * (32/8) ... +1] etc — wait,
//     simpler: each thread handles 8 elements of a sub-block (32 elems
//     across 32 threads gives 1 elem/thread/sub-block, but we want
//     coalesced byte reads, so we walk sub-blocks and split elements).
//
// The implementation below: per super-block, the 32 threads each
// handle 8 elements (one per sub-block). Thread t handles element t
// within EACH of the 8 sub-blocks. That gives one byte read per
// sub-block per thread. Across the warp, the 32 threads of sub-block
// sb read bytes qpair_bytes[t] for t in [0,32) — perfectly coalesced.
__global__ void sp_q4k_matvec_kernel(
    const uint8_t * __restrict__ W,   // [n_out * (n_in/256) * 144] bytes
    const float   * __restrict__ x,   // [n_in]
    float         * __restrict__ y,   // [n_out]
    int n_out, int n_in)
{
    const int warp_id_in_block = threadIdx.x >> 5;       // 0..7
    const int lane             = threadIdx.x & 31;       // 0..31
    const int row              = blockIdx.x * (blockDim.x >> 5) + warp_id_in_block;
    if (row >= n_out) return;

    const int n_blocks = n_in / 256;
    const size_t row_bytes = (size_t)n_blocks * 144;
    const uint8_t *p = W + (size_t)row * row_bytes;

    float sum = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const uint8_t *block = p + (size_t)b * 144;

        // Super-block fp16 scales (each thread reads them — same
        // 4 bytes broadcast across the warp; cache-friendly).
        __half d_h, dm_h;
        ((uint8_t *)&d_h )[0] = block[0]; ((uint8_t *)&d_h )[1] = block[1];
        ((uint8_t *)&dm_h)[0] = block[2]; ((uint8_t *)&dm_h)[1] = block[3];
        const float d_super    = __half2float(d_h);
        const float dmin_super = __half2float(dm_h);

        const uint8_t *scales = block + 4;
        const uint8_t *qs     = block + 16;
        const float   *xc     = x + (size_t)b * 256;

        // For each of 8 sub-blocks: lane t handles element [sb*32 + t].
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            uint8_t sc, mn;
            if (sb < 4) {
                sc = scales[sb]     & 63;
                mn = scales[sb + 4] & 63;
            } else {
                sc = (scales[sb + 4] & 0x0F) | ((scales[sb - 4] >> 6) << 4);
                mn = (scales[sb + 4] >> 4)   | ((scales[sb]     >> 6) << 4);
            }
            const float ds = d_super    * (float)sc;
            const float ms = dmin_super * (float)mn;
            const int qpair = sb >> 1;
            const int high_nibble = sb & 1;
            const uint8_t *qpair_bytes = qs + qpair * 32;
            // Coalesced: lanes 0..31 read consecutive bytes.
            const uint8_t qbyte = qpair_bytes[lane];
            const int nibble = high_nibble ? (qbyte >> 4) : (qbyte & 0x0F);
            const float w = ds * (float)nibble - ms;
            sum = fmaf(w, xc[sb * 32 + lane], sum);
        }
    }

    // Warp reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, off);
    }
    if (lane == 0) y[row] = sum;
}

// Host launcher. dW_raw is a device pointer to the row-major Q4_K bytes
// (n_out rows, each row having n_in/256 super-blocks of 144 bytes).
// d_x and d_y are device fp32 pointers (caller manages H2D / D2H).
cudaError_t sp_beast_gpu_q4k_matvec_launch(
    const void *dW_raw,
    const float *d_x, float *d_y,
    int n_out, int n_in,
    cudaStream_t stream)
{
    if (n_in % 256 != 0) return cudaErrorInvalidValue;
    const int threads_per_block = 256;
    const int rows_per_block    = threads_per_block / 32;  // 8 warps
    const int blocks            = (n_out + rows_per_block - 1) / rows_per_block;
    sp_q4k_matvec_kernel<<<blocks, threads_per_block, 0, stream>>>(
        (const uint8_t *)dW_raw, d_x, d_y, n_out, n_in);
    return cudaGetLastError();
}

// ============================================================================
// RMS norm + elementwise kernels — feed the matvecs without round-tripping
// ============================================================================

// y[i] = x[i] * inv_rms * weight[i],  inv_rms = 1 / sqrt(mean(x^2) + eps)
// One block per RMS norm call (n_embd up to ~4K — fits one block easily).
// Block reduces x[i]^2 in shared memory, then writes y.
__global__ void sp_rms_norm_kernel(const float *__restrict__ x,
                                    const float *__restrict__ weight,
                                    float       *__restrict__ y,
                                    int n, float eps)
{
    const int tid = threadIdx.x;
    const int T   = blockDim.x;

    __shared__ float ss_buf[32];   // one slot per warp; up to 1024 threads

    // Partial sum-of-squares.
    float partial = 0.0f;
    for (int i = tid; i < n; i += T) {
        const float xi = x[i];
        partial += xi * xi;
    }
    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
    }
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;
    if (lane == 0) ss_buf[warp_id] = partial;
    __syncthreads();

    // Final reduce by first warp.
    float ss = 0.0f;
    if (tid < 32) {
        const int n_warps = (T + 31) >> 5;
        ss = (tid < n_warps) ? ss_buf[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ss += __shfl_xor_sync(0xFFFFFFFF, ss, off);
        }
        if (tid == 0) ss_buf[0] = ss;
    }
    __syncthreads();
    const float inv = rsqrtf(ss_buf[0] / (float)n + eps);

    // Scale and write.
    for (int i = tid; i < n; i += T) {
        y[i] = x[i] * inv * weight[i];
    }
}

cudaError_t sp_beast_gpu_rms_norm_launch(
    const float *d_x, const float *d_w, float *d_y,
    int n, float eps, cudaStream_t stream)
{
    const int threads = (n >= 1024) ? 1024 : 256;
    sp_rms_norm_kernel<<<1, threads, 0, stream>>>(d_x, d_w, d_y, n, eps);
    return cudaGetLastError();
}

// y[i] += a[i]  — residual add.
__global__ void sp_add_kernel(float *__restrict__ y,
                               const float *__restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}

cudaError_t sp_beast_gpu_add_launch(float *d_y, const float *d_a, int n,
                                      cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    sp_add_kernel<<<blocks, threads, 0, stream>>>(d_y, d_a, n);
    return cudaGetLastError();
}

} // extern "C"

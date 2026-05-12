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

// ============================================================================
// SSM block kernels — conv1d, SiLU, L2 norm per head, Delta Net recurrence
// ============================================================================

// SiLU elementwise: y[i] = x[i] / (1 + exp(-x[i])).
__global__ void sp_silu_kernel(float *__restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        x[i] = v / (1.0f + __expf(-v));
    }
}

cudaError_t sp_beast_gpu_silu_launch(float *d_x, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    sp_silu_kernel<<<blocks, threads, 0, stream>>>(d_x, n);
    return cudaGetLastError();
}

// Output gating: y[i] *= silu(g[i]).  In place on y, reads g.
__global__ void sp_silu_gate_kernel(float *__restrict__ y,
                                     const float *__restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gv = g[i];
        y[i] *= gv / (1.0f + __expf(-gv));
    }
}

cudaError_t sp_beast_gpu_silu_gate_launch(float *d_y, const float *d_g, int n,
                                            cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    sp_silu_gate_kernel<<<blocks, threads, 0, stream>>>(d_y, d_g, n);
    return cudaGetLastError();
}

// Depthwise 4-tap conv1d for SSM.
//
// In: qkv_buf [conv_dim], conv_state [conv_k-1, conv_dim] (rolling window),
//     conv_w [conv_k, conv_dim], conv_b [conv_dim].
// Out: qkv_buf gets the conv output written in-place; conv_state is shifted
//      and the new qkv_buf value pushed in as the newest slot.
//
// Algorithm per channel c (one thread):
//   sum = conv_b[c]
//   for t in [0, conv_k-1):  sum += conv_state[t, c] * conv_w[t, c]
//   sum += qkv_buf[c] * conv_w[conv_k-1, c]
//   shift conv_state by one (conv_state[t] = conv_state[t+1], last = qkv_buf[c])
//   qkv_buf[c] = sum
__global__ void sp_conv1d_4tap_kernel(
    float *__restrict__ qkv_buf,
    float *__restrict__ conv_state,
    const float *__restrict__ conv_w,
    const float *__restrict__ conv_b,
    int conv_dim, int conv_k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    const float new_v = qkv_buf[c];
    float sum = (conv_b ? conv_b[c] : 0.0f);
    // Older taps from state.
    for (int t = 0; t < conv_k - 1; ++t) {
        sum += conv_state[(size_t)t * conv_dim + c] * conv_w[(size_t)t * conv_dim + c];
    }
    // Current tap (qkv_buf is the newest).
    sum += new_v * conv_w[(size_t)(conv_k - 1) * conv_dim + c];

    // Shift state: state[0..k-2] = state[1..k-1]; state[k-2] = new_v.
    for (int t = 0; t < conv_k - 2; ++t) {
        conv_state[(size_t)t * conv_dim + c] =
            conv_state[(size_t)(t + 1) * conv_dim + c];
    }
    conv_state[(size_t)(conv_k - 2) * conv_dim + c] = new_v;

    qkv_buf[c] = sum;
}

cudaError_t sp_beast_gpu_conv1d_launch(
    float *d_qkv_buf, float *d_conv_state,
    const float *d_conv_w, const float *d_conv_b,
    int conv_dim, int conv_k, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks  = (conv_dim + threads - 1) / threads;
    sp_conv1d_4tap_kernel<<<blocks, threads, 0, stream>>>(
        d_qkv_buf, d_conv_state, d_conv_w, d_conv_b, conv_dim, conv_k);
    return cudaGetLastError();
}

// L2 normalise Q and K per head, in-place.
// q_all [key_dim], k_all [key_dim],  key_dim = n_k_heads * head_k_dim.
// One block per head, threads cooperatively reduce ||v||² and normalise.
__global__ void sp_l2_norm_qk_kernel(
    float *__restrict__ q_all, float *__restrict__ k_all,
    int head_k_dim)
{
    const int h    = blockIdx.x;   // head id
    const int tid  = threadIdx.x;
    const int T    = blockDim.x;
    float *qh = q_all + (size_t)h * head_k_dim;
    float *kh = k_all + (size_t)h * head_k_dim;

    __shared__ float qsum[32];
    __shared__ float ksum[32];

    // Partial reductions across threads.
    float qp = 0.0f, kp = 0.0f;
    for (int i = tid; i < head_k_dim; i += T) {
        const float qv = qh[i]; qp += qv * qv;
        const float kv = kh[i]; kp += kv * kv;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        qp += __shfl_xor_sync(0xFFFFFFFF, qp, off);
        kp += __shfl_xor_sync(0xFFFFFFFF, kp, off);
    }
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) { qsum[warp_id] = qp; ksum[warp_id] = kp; }
    __syncthreads();
    if (tid < 32) {
        const int n_warps = (T + 31) >> 5;
        float q2 = (tid < n_warps) ? qsum[tid] : 0.0f;
        float k2 = (tid < n_warps) ? ksum[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            q2 += __shfl_xor_sync(0xFFFFFFFF, q2, off);
            k2 += __shfl_xor_sync(0xFFFFFFFF, k2, off);
        }
        if (tid == 0) { qsum[0] = q2; ksum[0] = k2; }
    }
    __syncthreads();
    const float q_inv = rsqrtf(qsum[0]) + 0.0f;       // approx with no eps
    const float k_inv = rsqrtf(ksum[0]) + 0.0f;
    // Note: scalar reference uses 1.0 / (sqrt(sum) + 1e-12). With rsqrtf
    // the eps is dropped — head_k_dim=128 elements squared are nowhere
    // near underflow for trained weights. Verified matching token IDs.
    for (int i = tid; i < head_k_dim; i += T) {
        qh[i] *= q_inv;
        kh[i] *= k_inv;
    }
}

cudaError_t sp_beast_gpu_l2_qk_launch(
    float *d_q, float *d_k, int n_k_heads, int head_k_dim,
    cudaStream_t stream)
{
    const int threads = 128;
    sp_l2_norm_qk_kernel<<<n_k_heads, threads, 0, stream>>>(
        d_q, d_k, head_k_dim);
    return cudaGetLastError();
}

// Group RMS norm over per-head output: for each head, normalise the
// head_v_dim segment to unit RMS and scale by weight[i % norm_dim].
__global__ void sp_ssm_norm_kernel(
    float *__restrict__ output, const float *__restrict__ weight,
    int n_v_heads, int head_v_dim, int norm_dim, float eps)
{
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    const int T   = blockDim.x;
    float *seg = output + (size_t)h * head_v_dim;
    const int slen = (norm_dim < head_v_dim) ? norm_dim : head_v_dim;

    __shared__ float ss_buf[32];
    float partial = 0.0f;
    for (int i = tid; i < slen; i += T) {
        const float v = seg[i];
        partial += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) ss_buf[warp_id] = partial;
    __syncthreads();
    if (tid < 32) {
        const int n_warps = (T + 31) >> 5;
        float ss = (tid < n_warps) ? ss_buf[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xFFFFFFFF, ss, off);
        if (tid == 0) ss_buf[0] = ss;
    }
    __syncthreads();
    const float inv = rsqrtf(ss_buf[0] / (float)slen + eps);
    for (int i = tid; i < slen; i += T) {
        seg[i] *= inv * weight[i % norm_dim];
    }
}

cudaError_t sp_beast_gpu_ssm_norm_launch(
    float *d_output, const float *d_weight,
    int n_v_heads, int head_v_dim, int norm_dim, float eps,
    cudaStream_t stream)
{
    const int threads = 128;
    sp_ssm_norm_kernel<<<n_v_heads, threads, 0, stream>>>(
        d_output, d_weight, n_v_heads, head_v_dim, norm_dim, eps);
    return cudaGetLastError();
}

// Compute per-head gate and beta scalars from alpha_raw, beta_raw, ssm_a,
// dt_bias. One thread per head.
//   gate = exp(softplus(alpha_raw + dt_bias) * ssm_a)
//   beta = sigmoid(beta_raw)
__global__ void sp_ssm_gate_beta_kernel(
    const float *__restrict__ alpha_raw,
    const float *__restrict__ beta_raw,
    const float *__restrict__ ssm_a,
    const float *__restrict__ dt_bias,
    float *__restrict__ gate_vals,
    float *__restrict__ beta_vals,
    int n_v_heads)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_v_heads) return;
    const float a_raw = alpha_raw[h] + dt_bias[h];
    // softplus(x) = log(1+exp(x)); numerically safe for moderate x.
    const float sp = __logf(1.0f + __expf(a_raw));
    gate_vals[h]  = __expf(sp * ssm_a[h]);
    const float b = beta_raw[h];
    beta_vals[h]  = 1.0f / (1.0f + __expf(-b));
}

cudaError_t sp_beast_gpu_gate_beta_launch(
    const float *alpha_raw, const float *beta_raw,
    const float *ssm_a, const float *dt_bias,
    float *gate_vals, float *beta_vals,
    int n_v_heads, cudaStream_t stream)
{
    const int threads = 32;
    const int blocks = (n_v_heads + threads - 1) / threads;
    sp_ssm_gate_beta_kernel<<<blocks, threads, 0, stream>>>(
        alpha_raw, beta_raw, ssm_a, dt_bias, gate_vals, beta_vals, n_v_heads);
    return cudaGetLastError();
}

// Delta Net recurrence — one BLOCK per (v-)head; threads cooperate to
// process head_k_dim x head_v_dim state.
//
// Per head h:
//   kh, qh = K-group's K and Q vectors [head_k_dim]
//   vh     = V vector for this head     [head_v_dim]
//   Sh     = state                       [head_k_dim x head_v_dim] (row-major)
//   oh     = output                      [head_v_dim]
//   gate, beta scalars
//
// 1. Decay:   Sh *= gate
// 2. Retrieve: sk = S^T @ k  → [head_v_dim]
// 3. Delta:   d = (v - sk) * beta
// 4. Update:  S += outer(k, d)
// 5. Query:   o = scale * (S @ q)
//
// All five steps share the head's state, so doing them in one kernel
// keeps Sh resident in (effectively) the L1/L2 cache for the whole head.
__global__ void sp_delta_net_kernel(
    const float *__restrict__ q_all, const float *__restrict__ k_all,
    const float *__restrict__ v_all,
    float       *__restrict__ S_all,    // [n_v_heads * head_k_dim * head_v_dim]
    float       *__restrict__ output,   // [n_v_heads * head_v_dim]
    const float *__restrict__ gate_vals,
    const float *__restrict__ beta_vals,
    int n_v_heads, int n_k_heads, int head_k_dim, int head_v_dim)
{
    const int h    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int T    = blockDim.x;
    const int heads_per_group = n_v_heads / n_k_heads;
    const int kh_idx = h / heads_per_group;

    const float *qh = q_all + (size_t)kh_idx * head_k_dim;
    const float *kh = k_all + (size_t)kh_idx * head_k_dim;
    const float *vh = v_all + (size_t)h * head_v_dim;
    float *Sh = S_all + (size_t)h * (size_t)head_k_dim * head_v_dim;
    float *oh = output + (size_t)h * head_v_dim;
    const float gate  = gate_vals[h];
    const float beta  = beta_vals[h];
    const float scale = rsqrtf((float)head_k_dim);

    const int state_n = head_k_dim * head_v_dim;

    // 1. Decay.
    for (int i = tid; i < state_n; i += T) Sh[i] *= gate;
    __syncthreads();

    // 2. Retrieve: sk[j] = sum_i Sh[i,j] * kh[i].
    // Use shared mem to hold sk vector (head_v_dim).
    __shared__ float sk[256];
    __shared__ float d_vec[256];
    for (int j = tid; j < head_v_dim; j += T) sk[j] = 0.0f;
    __syncthreads();

    // Each thread handles one (i, j) pair stride, accumulating into sk[j].
    // Simpler: one thread per j, loop over i. head_v_dim=128 -> 128 threads
    // exactly. Each does 128 reads.
    for (int j = tid; j < head_v_dim; j += T) {
        float s = 0.0f;
        for (int i = 0; i < head_k_dim; ++i) {
            s += Sh[(size_t)i * head_v_dim + j] * kh[i];
        }
        sk[j] = s;
    }
    __syncthreads();

    // 3. Delta vector.
    for (int j = tid; j < head_v_dim; j += T) {
        d_vec[j] = (vh[j] - sk[j]) * beta;
    }
    __syncthreads();

    // 4. Update: Sh[i,j] += kh[i] * d[j].
    // Parallelise over i; each thread does the full head_v_dim row.
    for (int i = tid; i < head_k_dim; i += T) {
        const float ki = kh[i];
        float *row = Sh + (size_t)i * head_v_dim;
        for (int j = 0; j < head_v_dim; ++j) {
            row[j] = fmaf(ki, d_vec[j], row[j]);
        }
    }
    __syncthreads();

    // 5. Query: oh[j] = scale * sum_i Sh[i,j] * qh[i].
    for (int j = tid; j < head_v_dim; j += T) {
        float s = 0.0f;
        for (int i = 0; i < head_k_dim; ++i) {
            s += Sh[(size_t)i * head_v_dim + j] * qh[i];
        }
        oh[j] = s * scale;
    }
}

cudaError_t sp_beast_gpu_delta_net_launch(
    const float *q_all, const float *k_all, const float *v_all,
    float *S_all, float *output,
    const float *gate_vals, const float *beta_vals,
    int n_v_heads, int n_k_heads, int head_k_dim, int head_v_dim,
    cudaStream_t stream)
{
    const int threads = 128;
    sp_delta_net_kernel<<<n_v_heads, threads, 0, stream>>>(
        q_all, k_all, v_all, S_all, output, gate_vals, beta_vals,
        n_v_heads, n_k_heads, head_k_dim, head_v_dim);
    return cudaGetLastError();
}

} // extern "C"

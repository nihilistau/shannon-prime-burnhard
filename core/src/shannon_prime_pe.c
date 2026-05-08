// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// PrimePE — Lattice-Aligned RoPE Frequency Injection
//
// Pure C implementation of the frequency generation from sp_inject_freqs.py.
// Computes blended lattice + geometric freq_factors at model load time.
// Zero per-token cost — output is a constant tensor.
//
// Paper results (Position Is Arithmetic v8):
//   Q8_0:  −0.82% PPL at α=0.22
//   Q6_K:  −0.66% PPL at α=0.17
//   Q4_K_M: −0.61% PPL at α=0.17

#include "shannon_prime.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Prime sieve (small — up to 10000 is more than enough for any head_dim)
// ============================================================================

#define SP_PE_SIEVE_MAX 10000

static int g_sieve_done = 0;
static int g_primes[1300];     // π(10000) = 1229
static int g_n_primes = 0;

static void sp_pe_sieve(void) {
    if (g_sieve_done) return;

    // Sieve of Eratosthenes
    char *is_prime = (char *)calloc(SP_PE_SIEVE_MAX + 1, 1);
    if (!is_prime) return;
    memset(is_prime, 1, SP_PE_SIEVE_MAX + 1);
    is_prime[0] = is_prime[1] = 0;

    for (int i = 2; i * i <= SP_PE_SIEVE_MAX; ++i) {
        if (is_prime[i]) {
            for (int j = i * i; j <= SP_PE_SIEVE_MAX; j += i)
                is_prime[j] = 0;
        }
    }

    g_n_primes = 0;
    for (int i = 2; i <= SP_PE_SIEVE_MAX && g_n_primes < 1300; ++i) {
        if (is_prime[i])
            g_primes[g_n_primes++] = i;
    }

    free(is_prime);
    g_sieve_done = 1;
}

// ============================================================================
// Tiered frequency generation
// ============================================================================
//
// Three tiers (from paper §3.1):
//   Local (25%):  range 2..101    — word/syntax scale
//   Mid   (33%):  range 101..1009 — clause/paragraph scale
//   Long  (42%):  range 1009..8209 — section/document scale
//
// Uses composite numbers (coordinates in the lattice) by default.
// Paper showed identical performance: prime 129.2 PPL vs composite 129.4 PPL.

static void pick_evenly(const int *pool, int pool_n, int *out, int n) {
    if (n <= 0) return;
    if (pool_n <= n) {
        memcpy(out, pool, pool_n * sizeof(int));
        // Pad with last value
        for (int i = pool_n; i < n; ++i)
            out[i] = pool[pool_n - 1];
        return;
    }
    float step = (float)pool_n / n;
    for (int i = 0; i < n; ++i)
        out[i] = pool[(int)(i * step)];
}

static int *generate_tiered_frequencies(int n_freqs) {
    sp_pe_sieve();

    int *out = (int *)malloc(n_freqs * sizeof(int));
    if (!out) return NULL;

    // Tier allocation
    int n_local = n_freqs / 4;
    if (n_local < 1) n_local = 1;
    int n_mid = (n_freqs * 33) / 100;
    if (n_mid < 1) n_mid = 1;
    int n_long = n_freqs - n_local - n_mid;
    if (n_long < 1) { n_long = 1; n_mid = n_freqs - n_local - n_long; }

    // Build composite pools (non-prime integers in each range)
    // Using composites by default — paper showed identical performance
    int local_pool[200], local_n = 0;
    int mid_pool[1000],  mid_n = 0;
    int long_pool[8000], long_n = 0;

    // Build prime set for fast lookup
    char is_prime_small[SP_PE_SIEVE_MAX + 1];
    memset(is_prime_small, 0, sizeof(is_prime_small));
    for (int i = 0; i < g_n_primes; ++i)
        is_prime_small[g_primes[i]] = 1;

    for (int n = 4; n <= 8209; ++n) {
        if (is_prime_small[n]) continue;  // composites only
        if (n >= 2 && n <= 101 && local_n < 200)
            local_pool[local_n++] = n;
        if (n > 101 && n <= 1009 && mid_n < 1000)
            mid_pool[mid_n++] = n;
        if (n > 1009 && n <= 8209 && long_n < 8000)
            long_pool[long_n++] = n;
    }

    pick_evenly(local_pool, local_n, out, n_local);
    pick_evenly(mid_pool, mid_n, out + n_local, n_mid);
    pick_evenly(long_pool, long_n, out + n_local + n_mid, n_long);

    return out;
}

// ============================================================================
// Public API
// ============================================================================

float *sp_prime_pe_freq_factors(int n_freqs, float freq_base, float alpha) {
    if (n_freqs <= 0 || alpha < 0.0f) return NULL;

    // alpha=0 → identity (all 1.0)
    if (alpha == 0.0f || alpha < 1e-6f) {
        float *factors = (float *)malloc(n_freqs * sizeof(float));
        if (!factors) return NULL;
        for (int i = 0; i < n_freqs; ++i)
            factors[i] = 1.0f;
        return factors;
    }

    // Generate geometric frequencies: θ_j = base^(-2j/d)
    int d = n_freqs * 2;
    float *geometric = (float *)malloc(n_freqs * sizeof(float));
    if (!geometric) return NULL;
    for (int j = 0; j < n_freqs; ++j)
        geometric[j] = powf(freq_base, -2.0f * j / d);

    // Generate lattice frequencies
    int *lattice_int = generate_tiered_frequencies(n_freqs);
    if (!lattice_int) { free(geometric); return NULL; }

    float *lattice = (float *)malloc(n_freqs * sizeof(float));
    if (!lattice) { free(geometric); free(lattice_int); return NULL; }
    for (int i = 0; i < n_freqs; ++i)
        lattice[i] = (float)lattice_int[i];
    free(lattice_int);

    // Normalize lattice to geometric scale
    float geo_min = geometric[0], geo_max = geometric[0];
    float lat_min = lattice[0],   lat_max = lattice[0];
    for (int i = 1; i < n_freqs; ++i) {
        if (geometric[i] < geo_min) geo_min = geometric[i];
        if (geometric[i] > geo_max) geo_max = geometric[i];
        if (lattice[i] < lat_min) lat_min = lattice[i];
        if (lattice[i] > lat_max) lat_max = lattice[i];
    }

    float lat_range = lat_max - lat_min;
    float geo_range = geo_max - geo_min;
    if (lat_range > 1e-12f) {
        for (int i = 0; i < n_freqs; ++i)
            lattice[i] = geo_min + (lattice[i] - lat_min) / lat_range * geo_range;
    }

    // Blend: blended = (1-α)*geometric + α*lattice_norm
    float *factors = (float *)malloc(n_freqs * sizeof(float));
    if (!factors) { free(geometric); free(lattice); return NULL; }

    for (int i = 0; i < n_freqs; ++i) {
        float blended = (1.0f - alpha) * geometric[i] + alpha * lattice[i];
        // freq_factor = blended / geometric (multiplier on the base freq)
        factors[i] = (geometric[i] > 1e-12f) ? blended / geometric[i] : 1.0f;
    }

    free(geometric);
    free(lattice);
    return factors;
}

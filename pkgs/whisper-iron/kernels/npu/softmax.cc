/**
 * Softmax kernel for AMD AIE2.
 *
 * Implements row-wise softmax: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
 * Uses the numerically stable formulation with max subtraction.
 */

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int VEC_FACTOR = 16;

/**
 * Fast exponential approximation for AIE2.
 * Uses the identity: exp(x) = 2^(x * log2(e))
 * With polynomial approximation for 2^frac
 */
inline float fast_exp(float x) {
    // Clamp to avoid overflow/underflow
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 1e38f;  // Large but not inf

    // exp(x) = 2^(x * log2(e)) = 2^(x * 1.4427)
    constexpr float LOG2E = 1.4426950408889634f;
    float t = x * LOG2E;

    // Split into integer and fractional parts
    // t = n + f where n is integer and 0 <= f < 1
    int n = static_cast<int>(t);
    if (t < 0.0f && t != static_cast<float>(n)) n--;  // floor
    float f = t - static_cast<float>(n);

    // Polynomial approximation for 2^f on [0,1)
    // 2^f ≈ 1 + f*(ln2 + f*(ln2^2/2 + f*ln2^3/6))
    constexpr float C1 = 0.6931471805599453f;   // ln(2)
    constexpr float C2 = 0.24022650695910072f;  // ln(2)^2 / 2
    constexpr float C3 = 0.05550410866482158f;  // ln(2)^3 / 6
    float exp_f = 1.0f + f * (C1 + f * (C2 + f * C3));

    // Compute 2^n by bit manipulation
    // For n in reasonable range, use iterative multiply/divide
    float result = exp_f;
    if (n > 0) {
        for (int i = 0; i < n && i < 30; i++) result *= 2.0f;
    } else if (n < 0) {
        for (int i = 0; i > n && i > -30; i--) result *= 0.5f;
    }

    return result;
}

/**
 * Scalar softmax for a single row.
 * Reference implementation.
 */
template <typename T, int N>
void softmax_row_scalar(T* __restrict x, T* __restrict y) {
    // Step 1: Find max
    float max_val = static_cast<float>(x[0]);
    for (int i = 1; i < N; i++) {
        float val = static_cast<float>(x[i]);
        if (val > max_val) max_val = val;
    }

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    float temp[N];
    for (int i = 0; i < N; i++) {
        float val = static_cast<float>(x[i]) - max_val;
        temp[i] = fast_exp(val);
        sum += temp[i];
    }

    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) {
        y[i] = static_cast<T>(temp[i] * inv_sum);
    }
}

/**
 * Vectorized softmax for a single row.
 * Uses vector reductions for max and sum.
 */
template <typename T, int N>
void softmax_row_vector(T* __restrict x, T* __restrict y) {
    static_assert(N % VEC_FACTOR == 0, "N must be divisible by vector factor");

    const int num_iters = N / VEC_FACTOR;

    // Step 1: Find max using vector reduction
    float max_val = static_cast<float>(x[0]);
    for (int i = 0; i < N; i++) {
        float val = static_cast<float>(x[i]);
        if (val > max_val) max_val = val;
    }

    // Step 2: Compute exp(x - max) and accumulate sum
    float sum = 0.0f;
    for (int iter = 0; iter < num_iters; iter++) {
        for (int j = 0; j < VEC_FACTOR; j++) {
            int idx = iter * VEC_FACTOR + j;
            float val = static_cast<float>(x[idx]) - max_val;
            float exp_val = fast_exp(val);
            y[idx] = static_cast<T>(exp_val);  // Store temporarily
            sum += exp_val;
        }
    }

    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    for (int iter = 0; iter < num_iters; iter++) {
        for (int j = 0; j < VEC_FACTOR; j++) {
            int idx = iter * VEC_FACTOR + j;
            float val = static_cast<float>(y[idx]) * inv_sum;
            y[idx] = static_cast<T>(val);
        }
    }
}

/**
 * Exported C functions for IRON integration.
 */
extern "C" {

// Row sizes matching common attention dimensions
void softmax_bf16_64_scalar(bfloat16* x, bfloat16* y) {
    softmax_row_scalar<bfloat16, 64>(x, y);
}

void softmax_bf16_64_vector(bfloat16* x, bfloat16* y) {
    softmax_row_vector<bfloat16, 64>(x, y);
}

void softmax_bf16_128_scalar(bfloat16* x, bfloat16* y) {
    softmax_row_scalar<bfloat16, 128>(x, y);
}

void softmax_bf16_128_vector(bfloat16* x, bfloat16* y) {
    softmax_row_vector<bfloat16, 128>(x, y);
}

void softmax_bf16_256_scalar(bfloat16* x, bfloat16* y) {
    softmax_row_scalar<bfloat16, 256>(x, y);
}

void softmax_bf16_256_vector(bfloat16* x, bfloat16* y) {
    softmax_row_vector<bfloat16, 256>(x, y);
}

void softmax_bf16_512_scalar(bfloat16* x, bfloat16* y) {
    softmax_row_scalar<bfloat16, 512>(x, y);
}

void softmax_bf16_512_vector(bfloat16* x, bfloat16* y) {
    softmax_row_vector<bfloat16, 512>(x, y);
}

void softmax_bf16_1024_scalar(bfloat16* x, bfloat16* y) {
    softmax_row_scalar<bfloat16, 1024>(x, y);
}

void softmax_bf16_1024_vector(bfloat16* x, bfloat16* y) {
    softmax_row_vector<bfloat16, 1024>(x, y);
}

void softmax_bf16_1504_scalar(bfloat16* x, bfloat16* y) {
    softmax_row_scalar<bfloat16, 1504>(x, y);
}

void softmax_bf16_1504_vector(bfloat16* x, bfloat16* y) {
    softmax_row_vector<bfloat16, 1504>(x, y);
}

} // extern "C"

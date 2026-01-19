/**
 * Layer Normalization kernel for AMD AIE2.
 *
 * Implements: y = (x - mean) / sqrt(var + eps) * gamma + beta
 * Where mean and var are computed over the last dimension.
 */

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int VEC_FACTOR = 16;
constexpr float EPSILON = 1e-5f;

/**
 * Fast inverse square root for AIE2.
 * Uses Newton-Raphson iteration.
 * Returns 1/sqrt(x)
 */
inline float fast_inv_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;

    // Initial estimate using bit manipulation trick (Quake III style)
    // This gives ~2% accuracy
    union {
        float f;
        int32_t i;
    } conv;
    conv.f = x;
    conv.i = 0x5f3759df - (conv.i >> 1);
    float y = conv.f;

    // Newton-Raphson iterations: y = y * (1.5 - 0.5*x*y*y)
    // Each iteration roughly doubles the precision
    y = y * (1.5f - 0.5f * x * y * y);  // ~0.2% error
    y = y * (1.5f - 0.5f * x * y * y);  // ~0.0001% error

    return y;
}

/**
 * Scalar layer normalization.
 * Reference implementation.
 */
template <typename T, int N>
void layernorm_scalar(T* __restrict x, T* __restrict gamma, T* __restrict beta, T* __restrict y) {
    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += static_cast<float>(x[i]);
    }
    float mean = sum / static_cast<float>(N);

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = static_cast<float>(x[i]) - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / static_cast<float>(N);

    // Step 3: Normalize and apply affine transform
    float inv_std = fast_inv_sqrt(var + EPSILON);
    for (int i = 0; i < N; i++) {
        float x_norm = (static_cast<float>(x[i]) - mean) * inv_std;
        float g = static_cast<float>(gamma[i]);
        float b = static_cast<float>(beta[i]);
        y[i] = static_cast<T>(x_norm * g + b);
    }
}

/**
 * Vectorized layer normalization.
 * Uses explicit loops for type conversion.
 */
template <typename T, int N>
void layernorm_vector(T* __restrict x, T* __restrict gamma, T* __restrict beta, T* __restrict y) {
    static_assert(N % VEC_FACTOR == 0, "N must be divisible by vector factor");

    const int num_iters = N / VEC_FACTOR;
    const float inv_n = 1.0f / static_cast<float>(N);

    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += static_cast<float>(x[i]);
    }
    float mean = sum * inv_n;

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = static_cast<float>(x[i]) - mean;
        var_sum += diff * diff;
    }
    float var = var_sum * inv_n;
    float inv_std = fast_inv_sqrt(var + EPSILON);

    // Step 3: Normalize and apply affine transform
    for (int iter = 0; iter < num_iters; iter++) {
        for (int j = 0; j < VEC_FACTOR; j++) {
            int idx = iter * VEC_FACTOR + j;
            float x_norm = (static_cast<float>(x[idx]) - mean) * inv_std;
            float g = static_cast<float>(gamma[idx]);
            float b = static_cast<float>(beta[idx]);
            y[idx] = static_cast<T>(x_norm * g + b);
        }
    }
}

/**
 * Exported C functions for IRON integration.
 */
extern "C" {

void layernorm_bf16_64_scalar(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_scalar<bfloat16, 64>(x, gamma, beta, y);
}

void layernorm_bf16_64_vector(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_vector<bfloat16, 64>(x, gamma, beta, y);
}

void layernorm_bf16_128_scalar(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_scalar<bfloat16, 128>(x, gamma, beta, y);
}

void layernorm_bf16_128_vector(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_vector<bfloat16, 128>(x, gamma, beta, y);
}

void layernorm_bf16_256_scalar(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_scalar<bfloat16, 256>(x, gamma, beta, y);
}

void layernorm_bf16_256_vector(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_vector<bfloat16, 256>(x, gamma, beta, y);
}

void layernorm_bf16_384_scalar(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_scalar<bfloat16, 384>(x, gamma, beta, y);
}

void layernorm_bf16_384_vector(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_vector<bfloat16, 384>(x, gamma, beta, y);
}

void layernorm_bf16_512_scalar(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_scalar<bfloat16, 512>(x, gamma, beta, y);
}

void layernorm_bf16_512_vector(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_vector<bfloat16, 512>(x, gamma, beta, y);
}

void layernorm_bf16_768_scalar(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_scalar<bfloat16, 768>(x, gamma, beta, y);
}

void layernorm_bf16_768_vector(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_vector<bfloat16, 768>(x, gamma, beta, y);
}

void layernorm_bf16_1024_scalar(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_scalar<bfloat16, 1024>(x, gamma, beta, y);
}

void layernorm_bf16_1024_vector(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_vector<bfloat16, 1024>(x, gamma, beta, y);
}

void layernorm_bf16_1536_scalar(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_scalar<bfloat16, 1536>(x, gamma, beta, y);
}

void layernorm_bf16_1536_vector(bfloat16* x, bfloat16* gamma, bfloat16* beta, bfloat16* y) {
    layernorm_vector<bfloat16, 1536>(x, gamma, beta, y);
}

} // extern "C"

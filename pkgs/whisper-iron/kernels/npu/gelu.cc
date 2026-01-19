/**
 * GELU activation kernel for AMD AIE2.
 *
 * Implements GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * This is the "tanh approximation" used by PyTorch and most frameworks.
 */

#include <aie_api/aie.hpp>
#include <stdint.h>

// Vector factor for AIE2
constexpr int VEC_FACTOR = 16;

// Constants for GELU approximation
constexpr float SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/pi)
constexpr float GELU_COEF = 0.044715f;

/**
 * Scalar GELU for bfloat16.
 * Reference implementation.
 */
template <typename T, int N>
void gelu_scalar(T* __restrict x, T* __restrict y) {
    for (int i = 0; i < N; i++) {
        float val = static_cast<float>(x[i]);

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = val * val * val;
        float inner = SQRT_2_OVER_PI * (val + GELU_COEF * x3);

        // tanh approximation: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
        // Good for |x| < 3, saturates correctly for larger values
        float inner2 = inner * inner;
        float tanh_approx;
        if (inner > 3.0f) {
            tanh_approx = 1.0f;
        } else if (inner < -3.0f) {
            tanh_approx = -1.0f;
        } else {
            tanh_approx = inner * (27.0f + inner2) / (27.0f + 9.0f * inner2);
        }

        float result = 0.5f * val * (1.0f + tanh_approx);
        y[i] = static_cast<T>(result);
    }
}

/**
 * Vectorized GELU using float vectors.
 * Converts bf16->f32, computes, converts back.
 */
template <typename T, int N>
void gelu_vector(T* __restrict x, T* __restrict y) {
    static_assert(N % VEC_FACTOR == 0, "N must be divisible by vector factor");

    const int num_iters = N / VEC_FACTOR;

    for (int iter = 0; iter < num_iters; iter++) {
        // Load and convert to float
        aie::vector<float, VEC_FACTOR> vx;
        for (int j = 0; j < VEC_FACTOR; j++) {
            vx[j] = static_cast<float>(x[iter * VEC_FACTOR + j]);
        }

        // Compute GELU element-wise in float
        aie::vector<float, VEC_FACTOR> vy;
        for (int j = 0; j < VEC_FACTOR; j++) {
            float val = vx[j];
            float x3 = val * val * val;
            float inner = SQRT_2_OVER_PI * (val + GELU_COEF * x3);
            float inner2 = inner * inner;

            float tanh_approx;
            if (inner > 3.0f) {
                tanh_approx = 1.0f;
            } else if (inner < -3.0f) {
                tanh_approx = -1.0f;
            } else {
                tanh_approx = inner * (27.0f + inner2) / (27.0f + 9.0f * inner2);
            }

            vy[j] = 0.5f * val * (1.0f + tanh_approx);
        }

        // Convert back and store
        for (int j = 0; j < VEC_FACTOR; j++) {
            y[iter * VEC_FACTOR + j] = static_cast<T>(vy[j]);
        }
    }
}

/**
 * Exported C functions for IRON integration.
 */
extern "C" {

// 1024 element versions (2KB per array for bf16)
void gelu_bf16_scalar(bfloat16* x, bfloat16* y) {
    gelu_scalar<bfloat16, 1024>(x, y);
}

void gelu_bf16_vector(bfloat16* x, bfloat16* y) {
    gelu_vector<bfloat16, 1024>(x, y);
}

// 4096 element versions (8KB per array for bf16)
void gelu_bf16_4k_scalar(bfloat16* x, bfloat16* y) {
    gelu_scalar<bfloat16, 4096>(x, y);
}

void gelu_bf16_4k_vector(bfloat16* x, bfloat16* y) {
    gelu_vector<bfloat16, 4096>(x, y);
}

} // extern "C"

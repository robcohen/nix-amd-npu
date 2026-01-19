/**
 * Vector addition kernel for AMD AIE2.
 *
 * Performs element-wise addition: c = a + b
 * Supports both scalar and vectorized implementations.
 */

#include <aie_api/aie.hpp>
#include <stdint.h>

// Vector factor for AIE2 - process 16 elements at a time
constexpr int VEC_FACTOR = 16;

/**
 * Scalar element-wise addition for bfloat16.
 * Simple reference implementation.
 */
template <typename T, int N>
void eltwise_add_scalar(T* __restrict a, T* __restrict b, T* __restrict c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Vectorized element-wise addition for bfloat16.
 * Uses AIE vector intrinsics for 16x speedup.
 */
template <typename T, int N>
void eltwise_add_vector(T* __restrict a, T* __restrict b, T* __restrict c) {
    static_assert(N % VEC_FACTOR == 0, "N must be divisible by vector factor");

    const int num_iters = N / VEC_FACTOR;

    T* __restrict ptr_a = a;
    T* __restrict ptr_b = b;
    T* __restrict ptr_c = c;

    for (int i = 0; i < num_iters; i++) {
        // Load 16 elements from each input
        aie::vector<T, VEC_FACTOR> va = aie::load_v<VEC_FACTOR>(ptr_a);
        aie::vector<T, VEC_FACTOR> vb = aie::load_v<VEC_FACTOR>(ptr_b);

        // Vector addition
        aie::vector<T, VEC_FACTOR> vc = aie::add(va, vb);

        // Store result
        aie::store_v(ptr_c, vc);

        ptr_a += VEC_FACTOR;
        ptr_b += VEC_FACTOR;
        ptr_c += VEC_FACTOR;
    }
}

/**
 * Exported C functions for IRON integration.
 * Process 1024 bfloat16 elements (2KB per array).
 */
extern "C" {

void eltwise_add_bf16_scalar(bfloat16* a, bfloat16* b, bfloat16* c) {
    eltwise_add_scalar<bfloat16, 1024>(a, b, c);
}

void eltwise_add_bf16_vector(bfloat16* a, bfloat16* b, bfloat16* c) {
    eltwise_add_vector<bfloat16, 1024>(a, b, c);
}

// Larger tile versions (4096 elements = 8KB per array)
void eltwise_add_bf16_4k_scalar(bfloat16* a, bfloat16* b, bfloat16* c) {
    eltwise_add_scalar<bfloat16, 4096>(a, b, c);
}

void eltwise_add_bf16_4k_vector(bfloat16* a, bfloat16* b, bfloat16* c) {
    eltwise_add_vector<bfloat16, 4096>(a, b, c);
}

// Float32 versions
void eltwise_add_f32_scalar(float* a, float* b, float* c) {
    eltwise_add_scalar<float, 1024>(a, b, c);
}

void eltwise_add_f32_vector(float* a, float* b, float* c) {
    eltwise_add_vector<float, 1024>(a, b, c);
}

} // extern "C"

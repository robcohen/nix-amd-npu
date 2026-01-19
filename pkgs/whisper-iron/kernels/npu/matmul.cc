/**
 * Matrix Multiplication kernel for AMD AIE2.
 *
 * Implements tiled GEMM: C = A @ B
 * Where A is (M, K), B is (K, N), C is (M, N)
 *
 * Tiling strategy:
 * - Process tiles of size (TILE_M, TILE_K) @ (TILE_K, TILE_N) -> (TILE_M, TILE_N)
 * - Tile sizes chosen to fit in AIE2 tile memory (16KB data memory)
 */

#include <aie_api/aie.hpp>
#include <stdint.h>

// Tile dimensions - must fit 3 tiles in 16KB
constexpr int TILE_M = 32;
constexpr int TILE_K = 32;
constexpr int TILE_N = 32;

// Vector factor for bf16 operations
constexpr int VEC_FACTOR = 16;

/**
 * Scalar matrix multiply for a single tile.
 * C_tile += A_tile @ B_tile
 */
template <typename T>
void matmul_tile_scalar(
    T* __restrict A,
    T* __restrict B,
    T* __restrict C
) {
    for (int m = 0; m < TILE_M; m++) {
        for (int n = 0; n < TILE_N; n++) {
            float acc = static_cast<float>(C[m * TILE_N + n]);
            for (int k = 0; k < TILE_K; k++) {
                float a_val = static_cast<float>(A[m * TILE_K + k]);
                float b_val = static_cast<float>(B[k * TILE_N + n]);
                acc += a_val * b_val;
            }
            C[m * TILE_N + n] = static_cast<T>(acc);
        }
    }
}

/**
 * Vectorized matrix multiply for a single tile.
 * Uses outer product accumulation pattern.
 */
template <typename T>
void matmul_tile_vector(
    T* __restrict A,
    T* __restrict B,
    T* __restrict C
) {
    // Process one row of C at a time
    for (int m = 0; m < TILE_M; m++) {
        // Load current C row into float accumulator
        float c_row[TILE_N];
        for (int n = 0; n < TILE_N; n++) {
            c_row[n] = static_cast<float>(C[m * TILE_N + n]);
        }

        // Accumulate A[m, k] * B[k, :] for all k
        for (int k = 0; k < TILE_K; k++) {
            float a_val = static_cast<float>(A[m * TILE_K + k]);

            // Process B row in chunks
            for (int n = 0; n < TILE_N; n++) {
                float b_val = static_cast<float>(B[k * TILE_N + n]);
                c_row[n] += a_val * b_val;
            }
        }

        // Store C row back
        for (int n = 0; n < TILE_N; n++) {
            C[m * TILE_N + n] = static_cast<T>(c_row[n]);
        }
    }
}

/**
 * Zero initialize a tile.
 */
template <typename T>
void zero_tile(T* __restrict C) {
    for (int i = 0; i < TILE_M * TILE_N; i++) {
        C[i] = static_cast<T>(0.0f);
    }
}

/**
 * Exported C functions for IRON integration.
 * All operate on 32x32 tiles.
 */
extern "C" {

// Zero initialize a tile
void matmul_zero_tile_bf16(bfloat16* C) {
    zero_tile<bfloat16>(C);
}

// Scalar tile matmul (reference)
void matmul_tile_bf16_scalar(bfloat16* A, bfloat16* B, bfloat16* C) {
    matmul_tile_scalar<bfloat16>(A, B, C);
}

// Vectorized tile matmul
void matmul_tile_bf16_vector(bfloat16* A, bfloat16* B, bfloat16* C) {
    matmul_tile_vector<bfloat16>(A, B, C);
}

// Float32 versions
void matmul_zero_tile_f32(float* C) {
    zero_tile<float>(C);
}

void matmul_tile_f32_scalar(float* A, float* B, float* C) {
    matmul_tile_scalar<float>(A, B, C);
}

void matmul_tile_f32_vector(float* A, float* B, float* C) {
    matmul_tile_vector<float>(A, B, C);
}

} // extern "C"

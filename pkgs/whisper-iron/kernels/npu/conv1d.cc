/**
 * 1D Convolution kernel for AMD AIE2.
 *
 * Implements Conv1D for Whisper audio encoder frontend:
 *   y = conv1d(x, weight, bias, stride, padding)
 *
 * Whisper uses two conv layers:
 *   1. Conv1d(80, 384, kernel=3, stride=1, padding=1)  - mel -> features
 *   2. Conv1d(384, 384, kernel=3, stride=2, padding=1) - downsample
 *
 * We implement these as specialized kernels since the dimensions are fixed.
 */

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int VEC_FACTOR = 16;

/**
 * Generic 1D convolution with kernel size 3.
 * Processes one output channel at a time.
 *
 * @param input Input of shape (in_channels, length)
 * @param weight Weight of shape (in_channels, 3) for one output channel
 * @param bias Scalar bias for this output channel
 * @param output Output of shape (out_length,) for one output channel
 * @param in_channels Number of input channels
 * @param length Input sequence length
 * @param stride Convolution stride
 */
template <typename T>
void conv1d_k3_scalar(
    T* __restrict input,
    T* __restrict weight,
    T bias,
    T* __restrict output,
    int in_channels,
    int length,
    int stride
) {
    // Output length = (length + 2*padding - kernel_size) / stride + 1
    // With padding=1, kernel=3: out_len = (length + 2 - 3) / stride + 1 = (length - 1) / stride + 1
    int out_length = (length - 1) / stride + 1;

    for (int o = 0; o < out_length; o++) {
        float acc = static_cast<float>(bias);
        int in_start = o * stride - 1;  // -1 for padding=1

        for (int c = 0; c < in_channels; c++) {
            for (int k = 0; k < 3; k++) {
                int in_pos = in_start + k;
                float in_val = 0.0f;

                // Handle padding (zeros outside bounds)
                if (in_pos >= 0 && in_pos < length) {
                    in_val = static_cast<float>(input[c * length + in_pos]);
                }

                float w_val = static_cast<float>(weight[c * 3 + k]);
                acc += in_val * w_val;
            }
        }

        output[o] = static_cast<T>(acc);
    }
}

/**
 * Vectorized conv1d for kernel size 3.
 * Vectorizes across input channels.
 */
template <typename T>
void conv1d_k3_vector(
    T* __restrict input,
    T* __restrict weight,
    T bias,
    T* __restrict output,
    int in_channels,
    int length,
    int stride
) {
    int out_length = (length - 1) / stride + 1;
    int channel_iters = in_channels / VEC_FACTOR;

    for (int o = 0; o < out_length; o++) {
        float acc = static_cast<float>(bias);
        int in_start = o * stride - 1;

        // Process channels in groups of VEC_FACTOR
        for (int ci = 0; ci < channel_iters; ci++) {
            int c_base = ci * VEC_FACTOR;

            for (int k = 0; k < 3; k++) {
                int in_pos = in_start + k;

                aie::vector<float, VEC_FACTOR> in_vec;
                aie::vector<float, VEC_FACTOR> w_vec;

                // Load input vector (with padding handling)
                if (in_pos >= 0 && in_pos < length) {
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        in_vec[v] = static_cast<float>(input[(c_base + v) * length + in_pos]);
                    }
                } else {
                    in_vec = aie::zeros<float, VEC_FACTOR>();
                }

                // Load weight vector
                for (int v = 0; v < VEC_FACTOR; v++) {
                    w_vec[v] = static_cast<float>(weight[(c_base + v) * 3 + k]);
                }

                // Multiply and accumulate
                aie::vector<float, VEC_FACTOR> prod = aie::mul(in_vec, w_vec);
                acc += aie::reduce_add(prod);
            }
        }

        // Handle remaining channels
        int remaining_start = channel_iters * VEC_FACTOR;
        for (int c = remaining_start; c < in_channels; c++) {
            for (int k = 0; k < 3; k++) {
                int in_pos = in_start + k;
                float in_val = 0.0f;
                if (in_pos >= 0 && in_pos < length) {
                    in_val = static_cast<float>(input[c * length + in_pos]);
                }
                float w_val = static_cast<float>(weight[c * 3 + k]);
                acc += in_val * w_val;
            }
        }

        output[o] = static_cast<T>(acc);
    }
}

/**
 * Specialized conv1d for Whisper layer 1: (80, 384, k=3, s=1, p=1)
 * Processes one output position, all 384 output channels.
 *
 * For efficiency, we process multiple output channels in parallel.
 */
template <typename T>
void conv1d_whisper_layer1_tile(
    T* __restrict input,      // (80, tile_len) - input tile
    T* __restrict weight,     // (384, 80, 3) - all weights
    T* __restrict bias,       // (384,) - all biases
    T* __restrict output,     // (384, tile_len) - output tile
    int tile_len              // Length of this tile
) {
    constexpr int IN_CH = 80;
    constexpr int OUT_CH = 384;
    constexpr int KERNEL = 3;

    // For each output position
    for (int o = 0; o < tile_len; o++) {
        int in_start = o - 1;  // stride=1, padding=1

        // For each output channel
        for (int oc = 0; oc < OUT_CH; oc++) {
            float acc = static_cast<float>(bias[oc]);

            // Convolve over input channels and kernel
            for (int ic = 0; ic < IN_CH; ic++) {
                for (int k = 0; k < KERNEL; k++) {
                    int in_pos = in_start + k;
                    float in_val = 0.0f;

                    if (in_pos >= 0 && in_pos < tile_len) {
                        in_val = static_cast<float>(input[ic * tile_len + in_pos]);
                    }

                    // Weight layout: (out_ch, in_ch, kernel)
                    float w_val = static_cast<float>(weight[oc * IN_CH * KERNEL + ic * KERNEL + k]);
                    acc += in_val * w_val;
                }
            }

            output[oc * tile_len + o] = static_cast<T>(acc);
        }
    }
}

/**
 * Specialized conv1d for Whisper layer 2: (384, 384, k=3, s=2, p=1)
 * Downsamples by factor of 2.
 */
template <typename T>
void conv1d_whisper_layer2_tile(
    T* __restrict input,      // (384, in_tile_len)
    T* __restrict weight,     // (384, 384, 3)
    T* __restrict bias,       // (384,)
    T* __restrict output,     // (384, out_tile_len)
    int in_tile_len,
    int out_tile_len
) {
    constexpr int CHANNELS = 384;
    constexpr int KERNEL = 3;
    constexpr int STRIDE = 2;

    for (int o = 0; o < out_tile_len; o++) {
        int in_start = o * STRIDE - 1;

        for (int oc = 0; oc < CHANNELS; oc++) {
            float acc = static_cast<float>(bias[oc]);

            for (int ic = 0; ic < CHANNELS; ic++) {
                for (int k = 0; k < KERNEL; k++) {
                    int in_pos = in_start + k;
                    float in_val = 0.0f;

                    if (in_pos >= 0 && in_pos < in_tile_len) {
                        in_val = static_cast<float>(input[ic * in_tile_len + in_pos]);
                    }

                    float w_val = static_cast<float>(weight[oc * CHANNELS * KERNEL + ic * KERNEL + k]);
                    acc += in_val * w_val;
                }
            }

            output[oc * out_tile_len + o] = static_cast<T>(acc);
        }
    }
}

/**
 * Exported C functions for IRON integration.
 */
extern "C" {

// Generic conv1d with kernel=3
void conv1d_k3_bf16_scalar(
    bfloat16* input, bfloat16* weight, bfloat16 bias, bfloat16* output,
    int in_channels, int length, int stride
) {
    conv1d_k3_scalar<bfloat16>(input, weight, bias, output, in_channels, length, stride);
}

void conv1d_k3_bf16_vector(
    bfloat16* input, bfloat16* weight, bfloat16 bias, bfloat16* output,
    int in_channels, int length, int stride
) {
    conv1d_k3_vector<bfloat16>(input, weight, bias, output, in_channels, length, stride);
}

// Whisper-specific layer 1: (80, 384, k=3, s=1)
// Tile size: 64 time steps
void conv1d_whisper_l1_tile64_bf16(
    bfloat16* input, bfloat16* weight, bfloat16* bias, bfloat16* output
) {
    conv1d_whisper_layer1_tile<bfloat16>(input, weight, bias, output, 64);
}

// Whisper-specific layer 2: (384, 384, k=3, s=2)
// Input tile: 64, Output tile: 32 (due to stride 2)
void conv1d_whisper_l2_tile64_bf16(
    bfloat16* input, bfloat16* weight, bfloat16* bias, bfloat16* output
) {
    conv1d_whisper_layer2_tile<bfloat16>(input, weight, bias, output, 64, 32);
}

} // extern "C"

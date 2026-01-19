"""
Matrix multiplication kernel for AMD NPU.

Uses MLIR-AIE/IRON when available, falls back to numpy.

NPU Acceleration:
- Uses tiled 32x32 GEMM for large matrices
- Falls back to CPU for small matrices (overhead not worth it)
- Minimum size threshold: 64x64 matrices
"""

import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path

# Try to import NPU runtime (legacy MLIR dialect)
try:
    from .npu_runtime import (
        NPURuntime, get_runtime, npu_available,
        generate_matmul_mlir, MLIR_AIE_AVAILABLE
    )
except ImportError:
    MLIR_AIE_AVAILABLE = False

    def npu_available():
        return False

# Try to import IRON-based NPU kernels (modern approach)
try:
    from .npu import IRON_AVAILABLE
except ImportError:
    IRON_AVAILABLE = False

# BF16 dtype - numpy doesn't have native bf16, use float32 as proxy
# The actual NPU will use bf16
BF16_DTYPE = np.float32

# Minimum matrix size for NPU acceleration
NPU_MIN_SIZE = 64


class MatMulBF16:
    """
    BF16 Matrix multiplication: C = A @ B

    Supports tiled execution for large matrices that don't fit in AIE memory.
    When NPU is available, compiles and caches MLIR kernels for each matrix size.
    """

    # Cache of compiled kernels by (M, K, N) tuple
    _kernel_cache: Dict[Tuple[int, int, int], str] = {}

    def __init__(self, M: int, K: int, N: int, use_npu: bool = True):
        """
        Initialize matmul kernel.

        Args:
            M: Rows of A and C
            K: Cols of A, rows of B
            N: Cols of B and C
            use_npu: Whether to use NPU (if available)
        """
        self.M = M
        self.K = K
        self.N = N
        self.use_npu = use_npu and npu_available()

        # AIE tile memory is limited (~16KB per tile)
        # For bf16: 16KB / 2 bytes = 8192 elements
        # We use 64x64 tiles = 4096 elements per matrix
        self.tile_size = 64

        # Round up to tile boundaries
        self.M_padded = ((M + self.tile_size - 1) // self.tile_size) * self.tile_size
        self.K_padded = ((K + self.tile_size - 1) // self.tile_size) * self.tile_size
        self.N_padded = ((N + self.tile_size - 1) // self.tile_size) * self.tile_size

        if self.use_npu:
            self._init_npu()

    def _init_npu(self):
        """Initialize NPU kernel."""
        key = (self.M_padded, self.K_padded, self.N_padded)

        if key not in MatMulBF16._kernel_cache:
            runtime = get_runtime()

            # Generate and compile MLIR
            mlir_source = generate_matmul_mlir(
                self.M_padded,
                self.K_padded,
                self.N_padded,
                dtype="bf16",
            )

            kernel_name = f"matmul_{self.M_padded}x{self.K_padded}x{self.N_padded}"
            xclbin_path, insts_path = runtime.compile_kernel(kernel_name, mlir_source)

            # Load kernel
            runtime.load_kernel(
                kernel_name,
                xclbin_path,
                insts_path,
                input_shapes=[(self.M_padded, self.K_padded), (self.K_padded, self.N_padded)],
                input_dtypes=[np.float16, np.float16],  # bf16 represented as float16
                output_shape=(self.M_padded, self.N_padded),
                output_dtype=np.float16,
            )

            MatMulBF16._kernel_cache[key] = kernel_name

        self.kernel_name = MatMulBF16._kernel_cache[key]

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Execute matrix multiplication.

        Args:
            A: Input matrix of shape (M, K)
            B: Input matrix of shape (K, N)

        Returns:
            C: Output matrix of shape (M, N)
        """
        assert A.shape == (self.M, self.K), f"A shape mismatch: {A.shape} vs ({self.M}, {self.K})"
        assert B.shape == (self.K, self.N), f"B shape mismatch: {B.shape} vs ({self.K}, {self.N})"

        if self.use_npu:
            return self._execute_npu(A, B)
        else:
            return self._execute_cpu(A, B)

    def _execute_cpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """CPU fallback using numpy."""
        return (A.astype(BF16_DTYPE) @ B.astype(BF16_DTYPE)).astype(BF16_DTYPE)

    def _execute_npu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Execute on NPU.

        Handles padding for matrices that don't align to tile boundaries.
        """
        runtime = get_runtime()

        # Pad inputs if needed
        if self.M != self.M_padded or self.K != self.K_padded:
            A_padded = np.zeros((self.M_padded, self.K_padded), dtype=np.float16)
            A_padded[:self.M, :self.K] = A.astype(np.float16)
        else:
            A_padded = A.astype(np.float16)

        if self.K != self.K_padded or self.N != self.N_padded:
            B_padded = np.zeros((self.K_padded, self.N_padded), dtype=np.float16)
            B_padded[:self.K, :self.N] = B.astype(np.float16)
        else:
            B_padded = B.astype(np.float16)

        # Execute on NPU
        C_padded = runtime.execute(self.kernel_name, [A_padded, B_padded])

        # Extract result (remove padding)
        C = C_padded[:self.M, :self.N].astype(BF16_DTYPE)

        return C


def matmul_bf16(A: np.ndarray, B: np.ndarray, use_npu: bool = True) -> np.ndarray:
    """
    Matrix multiplication with NPU acceleration.

    Args:
        A: Input matrix of shape (M, K)
        B: Input matrix of shape (K, N)
        use_npu: Whether to attempt NPU acceleration

    Returns:
        C: Output matrix of shape (M, N)
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    # Check if NPU acceleration is beneficial
    can_use_npu = (
        use_npu and
        (IRON_AVAILABLE or npu_available()) and
        M >= NPU_MIN_SIZE and
        K >= NPU_MIN_SIZE and
        N >= NPU_MIN_SIZE and
        A.flags['C_CONTIGUOUS'] and
        B.flags['C_CONTIGUOUS']
    )

    if can_use_npu:
        try:
            return _matmul_bf16_npu(A, B)
        except (RuntimeError, ImportError, OSError) as e:
            # Log warning on first failure, then silently fall back
            import warnings
            warnings.warn(
                f"NPU execution failed, falling back to CPU: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )
        except Exception as e:
            # Unexpected error - log with more detail for debugging
            import warnings
            warnings.warn(
                f"Unexpected NPU error (please report this bug): {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )

    # CPU fallback
    return (A.astype(BF16_DTYPE) @ B.astype(BF16_DTYPE)).astype(BF16_DTYPE)


def _matmul_bf16_npu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """NPU implementation of matrix multiplication."""
    if IRON_AVAILABLE:
        # Use modern IRON-based kernel
        from .npu.matmul_kernel import matmul_iron
        return matmul_iron(A, B)
    elif npu_available():
        # Use legacy MLIR dialect approach
        M, K = A.shape
        _, N = B.shape
        kernel = MatMulBF16(M, K, N)
        return kernel(A, B)
    else:
        raise RuntimeError("No NPU backend available")


def linear_bf16(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Linear layer: y = x @ W^T + b

    Args:
        x: Input of shape (..., in_features)
        weight: Weight matrix of shape (out_features, in_features)
        bias: Optional bias of shape (out_features,)

    Returns:
        y: Output of shape (..., out_features)
    """
    # Reshape x to 2D for matmul
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])

    # y = x @ W^T
    y = matmul_bf16(x_2d, weight.T.copy())  # .copy() ensures contiguous array

    # Add bias if present
    if bias is not None:
        y = y + bias.astype(BF16_DTYPE)

    # Reshape back
    out_shape = orig_shape[:-1] + (weight.shape[0],)
    return y.reshape(out_shape)


# ============================================================================
# Batched operations for transformer efficiency
# ============================================================================

def batched_matmul_bf16(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Batched matrix multiplication: C[b] = A[b] @ B[b]

    Args:
        A: Input of shape (batch, M, K)
        B: Input of shape (batch, K, N)

    Returns:
        C: Output of shape (batch, M, N)
    """
    batch_size = A.shape[0]
    M, K = A.shape[1], A.shape[2]
    N = B.shape[2]

    # For now, loop over batch (can be optimized later)
    C = np.zeros((batch_size, M, N), dtype=BF16_DTYPE)
    for b in range(batch_size):
        C[b] = matmul_bf16(A[b], B[b])

    return C


def attention_scores_bf16(Q: np.ndarray, K: np.ndarray, scale: float) -> np.ndarray:
    """
    Compute scaled attention scores: scores = Q @ K^T / scale

    Args:
        Q: Query of shape (batch, n_heads, seq_len, head_dim)
        K: Key of shape (batch, n_heads, seq_len, head_dim)
        scale: Scaling factor (typically sqrt(head_dim))

    Returns:
        scores: Attention scores of shape (batch, n_heads, seq_len, seq_len)
    """
    batch, n_heads, seq_len, head_dim = Q.shape

    # Reshape for batched matmul: (batch * n_heads, seq_len, head_dim)
    Q_flat = Q.reshape(batch * n_heads, seq_len, head_dim)
    K_flat = K.reshape(batch * n_heads, seq_len, head_dim)

    # Compute Q @ K^T for each (batch, head)
    scores_flat = np.zeros((batch * n_heads, seq_len, seq_len), dtype=BF16_DTYPE)
    for i in range(batch * n_heads):
        scores_flat[i] = matmul_bf16(Q_flat[i], K_flat[i].T.copy())

    # Scale and reshape
    scores = scores_flat.reshape(batch, n_heads, seq_len, seq_len)
    scores = scores / scale

    return scores

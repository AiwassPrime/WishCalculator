import logging

import cupy as cp
import numpy as np
import time


def set_logger():
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    set_logger()
    if cp.cuda.runtime.getDeviceCount() == 0:
        print("No CUDA-compatible devices found. GPU acceleration is not possible.")
    else:
        # Create random matrices on CPU
        size = 10000  # Size of the matrices
        matrix_a_cpu = np.random.rand(size, size).astype(np.float32)
        matrix_b_cpu = np.random.rand(size, size).astype(np.float32)

        # Transfer matrices to GPU memory
        matrix_a_gpu = cp.asarray(matrix_a_cpu)
        matrix_b_gpu = cp.asarray(matrix_b_cpu)

        # Perform matrix multiplication on GPU
        start_time = time.time()
        result_gpu = cp.dot(matrix_a_gpu, matrix_b_gpu)
        gpu_time = time.time() - start_time

        # Perform matrix multiplication on CPU for comparison
        start_time = time.time()
        result_cpu = np.dot(matrix_a_cpu, matrix_b_cpu)
        cpu_time = time.time() - start_time

        # Transfer the result back to CPU memory (optional)
        result_gpu_cpu = cp.asnumpy(result_gpu)

        print(f"Matrix multiplication on GPU took {gpu_time:.6f} seconds.")
        print(f"Matrix multiplication on CPU took {cpu_time:.6f} seconds.")

        # Verify if the results are the same
        print("Results are the same:", np.allclose(result_cpu, result_gpu_cpu))

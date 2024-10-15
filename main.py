import numpy as np
import time
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

N = 1000
if N <= 1000:
    input_raw = np.fromfile("input.dat", dtype=np.float64)

    in_mat = input_raw.reshape(3, N).T

    start_time = time.time()
    output = cdist(in_mat, in_mat, "euclidean")  # distance_matrix(a, b)
    end_time = time.time()
    print(f"use: {end_time - start_time}")

    output2 = output * output
    cuda_out2 = np.fromfile("output.dat", dtype=np.float64).reshape(N, N)

    print(f"python: {np.mean(output2)}")
    print(f"cuda: {np.mean(cuda_out2)}")
else:
    print("N is too large, use random func.")
    in_mat = np.random.rand(N, 3).astype(np.float64)
    start_time = time.time()
    output = cdist(in_mat, in_mat, "euclidean")  # distance_matrix(a, b)
    end_time = time.time()
    print(f"use: {end_time - start_time}")

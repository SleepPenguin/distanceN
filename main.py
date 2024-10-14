import numpy as np
import time
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

N = 1000
input_raw = np.fromfile("input.dat", dtype=np.float64)

input = input_raw.reshape(3, N).T

start_time = time.time()
output = cdist(input, input, "euclidean")  # distance_matrix(a, b)
end_time = time.time()
print(end_time - start_time)

output2 = output * output
cuda_out2 = np.fromfile("output.dat", dtype=np.float64).reshape(N, N)

print(f"python: {np.mean(output2)}")
print(f"cuda: {np.mean(cuda_out2)}")

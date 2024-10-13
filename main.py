import numpy as np
import time
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

a = np.random.rand(50000, 3)
start_time = time.time()
c = cdist(a, a) # distance_matrix(a, b)
end_time = time.time()
print(end_time - start_time)

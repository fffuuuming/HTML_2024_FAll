import numpy as np
from scipy.sparse import csr_array
A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
# v = csr_array([1, 0, -1])
# print(A.dot(v.T))
print(A)
print(A.toarray())
print(A.toarray().flatten())
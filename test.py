import numpy as np
from scipy.optimize import linear_sum_assignment as linear_sum_assignment

const_matrix = np.array([[15, 40, 45], [20, 60, 35], [20, 40, 25]])
matches1 = linear_sum_assignment(const_matrix)
print("matches1_type=", type(matches1))
print("matches1=", matches1)
print("matches1_T=", np.array(matches1).T)

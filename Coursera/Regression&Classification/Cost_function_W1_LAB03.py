import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])       #(price in 1000s of dollars)

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = x[i]*w + b
        cost = (f_wb-y[i])**2
        cost_sum+=cost
    total_cost = (1/(2*m))*cost_sum
    return total_cost


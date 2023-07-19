import numpy as np
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common_1 import sigmoid
np.set_printoptions(precision=8)

def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    m,n = X.shape
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost+=(f_wb_i-y[i])**2
    cost/=(2*m)

    reg_cost = 0.
    for i in range(n):
        reg_cost+=(w[j]**2)
    reg_cost=(lambda_*reg_cost)/(2*m)
    total_cost = reg_cost + cost
    return total_cost

def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    m, n = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost+= y[i]*np.log(f_wb_i) + (1-y[i])*np.log(1-f_wb_i)
    cost=-cost/m

    reg_cost = 0
    for j in range(n):
        reg_cost+=(w[j]**2)
    reg_cost = (lambda_*reg_cost)/(2*m)
    total_cost = reg_cost + cost
    return total_cost

def compute_gradient_linear_reg(X, y, w, b, lamda_):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /=m
    for j in range(n):
        dj_dw[j]+= (lamda_*w[j])/m
    return dj_dw, dj_db

def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = (f_wb_i-y[i])
        for j in range(n):
            dj_dw[j]+=err_i*X[i,j]
        dj_db+=err_i
    dj_db/=m
    dj_dw/=m
    for i in range(n):
        dj_dw[i]+= (lambda_*w[i])/m
    return dj_dw, dj_db
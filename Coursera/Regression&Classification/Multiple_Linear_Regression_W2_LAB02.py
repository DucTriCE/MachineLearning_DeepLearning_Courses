import copy, math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict_single_loop(x, w, b):
    m = x.shape[0]
    p=0
    for i in range(m):
        p_i = x[i]*w[i]
        p+=p_i
    p+=b
    return p

# get a row from our training data
x_vec = x_train[0,:]
print(f"x_vec value: {x_vec}")

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"prediction: {f_wb}")

#using dot
def predict(x, w ,b):
    p = np.dot(x, w) + b
    return p

# get a row from our training data
x_vec = x_train[0,:]
print(f"x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec, w_init, b_init)
print(f"prediction: {f_wb}")

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost = (f_wb_i-y[i])**2
        cost_sum+=cost
    total_cost = cost_sum/(2*m)
    return total_cost

cost = compute_cost(x_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

def compute_gradient(x, y, w, b):
    m, n = x.shape #Number of examples & number of features
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        err = (np.dot(x[i], w) + b - y[i])
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*x[i, j]
        dj_db+=err
    dj_dw/=m
    dj_db/=m
    return dj_dw, dj_db

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, w_init, b_init)
print(f'dj_dw at initial w,b: {tmp_dj_dw}')
print(f'dj_db at initial w,b: \n {tmp_dj_db}')

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in) #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w=w-alpha*dj_dw
        b=b-alpha*dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history

initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()
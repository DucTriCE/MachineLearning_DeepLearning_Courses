import math, copy
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = x[i]*w + b
        cost = (f_wb-y[i])**2
        cost_sum+=cost
    total_cost=(1/(2*m))*cost_sum
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = dj_db = 0
    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw_i = (f_wb-y[i])*x[i]
        dj_db_i = (f_wb-y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw/=m
    dj_db/=m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        # Save cost J at each iteration
        if i<100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history   #return w and J,w history for graphing

#initialize parameters
w_init = 0
b_init = 0
#gradient descent setting
tmp_alpha = 1.0e-2
iterations = 10000
#run gradient descent
w_final, b_final, J_hist , p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout= True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()

#predict
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")


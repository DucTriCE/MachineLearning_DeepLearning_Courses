import numpy as np
import matplotlib.pyplot as plt

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

m = x_train.shape[0] # m = len(x_train)

#Plot the datapoint
plt.scatter(x_train, y_train, marker='x', c='r')
#Set the title
plt.title("Housing Prices")
#Set the y_axis label
plt.ylabel("Price (in 1000s of dollars")
#Set the x_axis label
plt.xlabel('Size in (1000 sqft)')
plt.show()

w = 200
b = 100
tmp_f_wb = compute_model_output(x_train, w, b)

#plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
#plot the datapoints
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual value')
#set the title
plt.title("Housing Prices")
#Set the y_axis label
plt.ylabel("Price (in 1000s of dollars")
#Set the x_axis label
plt.xlabel('Size in (1000 sqft)')
#Put a legend on the axes ( chu thich )
plt.legend()
plt.show()

#Prediction
x_i = 1.2
cost_1200_sqft = w*x_i+b
print(cost_1200_sqft)

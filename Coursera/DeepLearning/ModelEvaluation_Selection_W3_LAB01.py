# for array computations and loading data
import numpy as np

# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import __version__
# for building and training neural networks
import tensorflow as tf

# custom functions
import utils

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)
# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

data = np.loadtxt('./data_w3_ex1.csv', delimiter=',')
x = data[:, 0]
y = data[:, 1]

#Convert into 2D array
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

# utils.plot_dataset(x=x, y=y, title="input vs. target")
# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

# print(f"the shape of the training set (input) is: {x_train.shape}")
# print(f"the shape of the training set (target) is: {y_train.shape}\n")
# print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
# print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
# print(f"the shape of the test set (input) is: {x_test.shape}")
# print(f"the shape of the test set (target) is: {y_test.shape}")

degree = 1
poly = PolynomialFeatures(degree, include_bias=False)
x_train_mapped = poly.fit_transform(x_train)
x_cv_mapped = poly.transform(x_cv)
x_test_mapped = poly.transform(x_test)

scaler = StandardScaler()
x_train_mapped_scaled = scaler.fit_transform(x_train_mapped)
x_cv_mapped_scaled = scaler.transform(x_cv_mapped)
x_test_mapped_scaled = scaler.transform(x_test_mapped)

#TRAIN MODEL
nn_train_mses = []
nn_cv_mses = []

#Build different models with different layers and units in each
nn_models = utils.build_models()

for model in nn_models:
    #Setup loss and optimizer
    model.compile(
        loss='mse',
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
    )

    print(f"Training {model.name}...")

    #Train the model
    model.fit(x_train_mapped_scaled, y_train, epochs=300, verbose=0)

    print("Done!\n")
    #Record the training MSEs
    yhat = model.predict(x_train_mapped_scaled)
    print(yhat)
    print(y_train)
    print("\nVALIDATION")
    train_mse = mean_squared_error(y_train, yhat)/2
    nn_train_mses.append(train_mse)

    #Record validation MSEs
    yhat = model.predict(x_cv_mapped_scaled)
    print(yhat)
    print(y_cv)
    cv_mse = mean_squared_error(y_cv, yhat)/2
    nn_cv_mses.append(cv_mse)

print("RESULT: ")
for model_num in range(len(nn_train_mses)):
    print(
        f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
        f"CV MSE: {nn_cv_mses[model_num]:.2f}"
    )

model_num = 3
# Compute the test MSE
yhat = nn_models[model_num-1].predict(x_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Selected Model: {model_num}")
print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")

print(__version__, tf.__version__)
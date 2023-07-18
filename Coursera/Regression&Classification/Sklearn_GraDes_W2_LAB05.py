import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
# print(X_train, X_norm)
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

y_predict_sgd = sgdr.predict(X_norm)
y_predict = np.dot(X_norm, w_norm) + b_norm
print(y_predict_sgd, y_predict, y_train,sep='\n')

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey= True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_predict,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
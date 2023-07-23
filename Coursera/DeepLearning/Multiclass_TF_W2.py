import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
np.set_printoptions(precision=2)
from lab_utils_multiclass_TF import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)
plt_mc(X_train,y_train,classes, centers, std=std)

# show classes in data set
print(f"unique classes {np.unique(y_train)}")
# show how classes are represented
print(f"class representation {y_train[:10]}")
# show shapes of our dataset
print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")

tf.random.set_seed(1234)
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(2, activation='relu', name='L1'),
        tf.keras.layers.Dense(4, activation='linear', name='L2')
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)
model.fit(X_train, y_train, epochs=200)
plt_cat_mc(X_train, y_train, model, classes)

l1 = model.get_layer("L1")
W1, b1 = l1.get_weights()

plt_layer_relu(X_train, y_train.reshape(-1, ), W1, b1, classes)
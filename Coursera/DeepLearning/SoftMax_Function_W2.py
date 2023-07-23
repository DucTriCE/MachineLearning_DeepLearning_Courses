import numpy as np
import matplotlib.pyplot as plt
plt.style.use('deeplearning.mplstyle')
import tensorflow as tf
from sklearn.datasets import make_blobs
from IPython.display import display, Markdown, Latex
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)
#Obvious model
'''
model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)
'''

#Preferred model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(4, activation='linear')
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(0.001),
)

model.fit(X_train, y_train, epochs=10)
#Output not probabilities, must be sent through softmax
p = model.predict(X_train)
print(f"two example output vectors:\n {p[:2]}")
sm = tf.nn.sigmoid(p).numpy()
print(np.max(sm), np.min(sm))
#To select the most likely category, the softmax is not required. One can find the index of the largest output using np.argmax
for i in range(5):
    print( f"{p[i]}, category: {np.argmax(p[i])}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('deeplearning.mplstyle')
import tensorflow as tf
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from autils import plt_act_trio
from lab_utils_relu import *
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt_act_trio()
_ = plt_relu_ex()
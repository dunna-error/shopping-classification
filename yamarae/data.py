import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import pickle
import traceback
from collections import Counter
from multiprocessing import Pool

import tqdm
import fire
import h5py
import numpy as np
# import mmh3
import six
from keras.utils.np_utils import to_categorical
from six.moves import cPickle

from misc import get_logger, Option
opt = Option('./config.json')

print("hello")
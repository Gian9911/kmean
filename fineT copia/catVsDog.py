import glob
import random
import shutil

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os.path
from keras.models import load_model
os.chdir('data/train')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')
for c in random.sample(glob.glob('cat*'), 500):
    shutil.move(c,'data/train/train/cat')
for c in random.sample(glob.glob('dog*'), 500):
    shutil.move(c, 'data/train/train/dog')
for c in random.sample(glob.glob('cat*'),100):
    shutil.move(c,'valid/train/cat')
for c in random.sample(glob.glob('dog*'),100):
    shutil.move(c,'train/valid/dog')
for c in random.sample(glob.glob('cat*'),50):
    shutil.move(c,'test/cat')
for c in random.sample(glob.glob('dog*'),50):
    shutil.move(c,'test/dog')
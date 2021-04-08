########################################################################################
#                     Script author: Michał Tomaszewski                                #
########################################################################################

import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
seed(1)
import random as rn
rn.seed(12345)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np 
import seaborn as sns
import sklearn.metrics as sklm
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.applications.vgg16 import preprocess_input as vgg16_preprocessing

from glaucpack import run_tests, smooth_curve, visualize_training



# READ MODEL FILE
glaucoma_model = load_model('vgg16_modelE_1.hdf5')

# SETUP TEST DATAGEN
test_directory = r''


test_datagen = ImageDataGenerator(preprocessing_function = vgg16_preprocessing)
test_generator = test_datagen.flow_from_directory(test_directory, target_size = (256, 256), shuffle = False, batch_size = 1, class_mode='binary')

# RUN TESTS
run_tests(glaucoma_model, test_generator)

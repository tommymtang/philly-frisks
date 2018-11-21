import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow import feature_column
from keras.utils import to_categorical

# this module gives helper functions for encoding categorical, bucketizable
# etc. data as input for tensorflow model

def map_categorical_to_onehot(cat_data):
    # data is assumed to be a pandas series, ie
    data_to_integer_object = pd.factorize(cat_data.copy())
    data_to_integer = data_to_integer_object[0]
    return to_categorical(data_to_integer)

def bucketize_datetimeoccur(datetimeoccur)

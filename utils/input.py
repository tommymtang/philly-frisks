import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow import feature_column
from keras.utils import to_categorical
import datetime as dt
import time
# this module gives helper functions for encoding categorical, bucketizable
# etc. data as input for tensorflow model


# for bucketizing (categorizing) datetime data, age data

def bucketize_timeoccur(time_occur, num_buckets = 8):
    num_hours_bucket = 24 / num_buckets
    bucket_iterator = pd.Timedelta(hours=num_hours_bucket)
    for ibucket in range(num_buckets):
        ibucket_time = (dt.datetime.min + ibucket * bucket_iterator).time()
        if time_occur < ibucket_time:
            return ibucket
    return num_buckets

def bucketize_timeoccur_series(time_occur_series, num_buckets = 8):
    # works on a series (pandas dataframe column) that is the time of a datetime object
    return time_occur_series.apply(bucketize_timeoccur, args = (num_buckets,)) # now compatible as input to map_categorical_to_onehot

import pandas as pd
import numpy as np
import math
import scipy as sp
from scipy.ndimage import gaussian_filter

def add_success_column(df):
    frisk_colname = 'individual_frisked'
    contraband_colname = 'individual_contraband'
    arrested_colname = 'individual_arrested'
    success = df[frisk_colname] * (df[contraband_colname] + df[arrested_colname])
    success[success > 0] = 1
    df.loc[:,'success'] = success
    return df

def get_local_hit_rate_input(df):
    # returns numpy array of input only relevant for computing local hit rate
    colnames = ['point_x', 'point_y', 'success']
    return df.loc[:, colnames].values

def get_map_limits(local_hit_rate_input, dilution=1000):
    min_x = math.floor(min(local_hit_rate_input[:,0]*dilution))
    max_x = math.ceil(max(local_hit_rate_input[:,0]*dilution))
    min_y = math.floor(min(local_hit_rate_input[:,1]*dilution))
    max_y = math.ceil(max(local_hit_rate_input[:,1]*dilution))
    return {'min_x' : min_x,
            'max_x' : max_x,
            'min_y' : min_y,
            'max_y' : max_y}

def get_success_map(local_hit_rate_input, dilution=1000):
    row_column_id = 1
    col_column_id = 0
    success_values_column_index = 2

    map_limits = get_map_limits(local_hit_rate_input)
    success_map = initialize_map(map_limits)

    points = get_map_coordinates(local_hit_rate_input)
    row_indices = points[:, row_column_id]
    col_indices = points[:, col_column_id]

    success_values = local_hit_rate_input[:, success_values_column_index].copy()
    assert(col_indices.shape == success_values.shape)
    assert(row_indices.shape == success_values.shape)
    success_map[row_indices, col_indices] = success_values
    return success_map

def get_indicator_map(local_hit_rate_input, dilution=1000):
    row_column_id = 1
    col_column_id = 0

    map_limits = get_map_limits(local_hit_rate_input)
    indicator_map = initialize_map(map_limits)

    points = get_map_coordinates(local_hit_rate_input)
    row_indices = points[:, row_column_id].copy()
    col_indices = points[:, col_column_id].copy()

    indicator_map[row_indices, col_indices] = 1
    return indicator_map

def get_local_hit_rate_map(local_hit_rate_input, sig = 8):
    success_map = get_success_map(local_hit_rate_input)
    indicator_map = get_indicator_map(local_hit_rate_input)
    blurred_success = np.nan_to_num(gaussian_filter(success_map, sigma=sig))
    blurred_indicator = np.nan_to_num(gaussian_filter(indicator_map, sigma=sig))
    return np.divide(blurred_success, blurred_indicator)

def initialize_map(map_limits):
    offset = 1
    max_x = math.ceil(map_limits['max_x'])
    min_x = math.floor(map_limits['min_x'])
    max_y = math.ceil(map_limits['max_y'])
    min_y = math.floor(map_limits['min_y'])
    height = max_y - min_y + offset
    width = max_x - min_x + offset
    return np.zeros([height, width])

def get_local_hit_rates(local_hit_rate_map, points):
    x_col_id = 0
    y_col_id = 1
    col_indices = points[:,x_col_id].copy()
    row_indices = points[:,y_col_id].copy()
    return local_hit_rate_map[row_indices, col_indices]

def get_map_coordinates(local_hit_rate_input, dilution=1000):
    point_indices = [0,1]
    point_x_column_index = 0
    point_y_column_index = 1

    points = local_hit_rate_input[:,point_indices].copy()
    points = np.round(points*dilution)
    map_limits = get_map_limits(local_hit_rate_input)

    points[:,point_x_column_index] -= map_limits['min_x']
    points[:,point_y_column_index] -= map_limits['min_y']

    return points.astype(int)

def add_local_hit_rate(df):
    assert not df.isnull().values.any()
    lhr_input = get_local_hit_rate_input(df)
    lhr_map = np.nan_to_num(get_local_hit_rate_map(lhr_input))
    points = get_map_coordinates(lhr_input)
    lhr = get_local_hit_rates(lhr_map, points)
    df['local_hit_rate'] = lhr
    return df

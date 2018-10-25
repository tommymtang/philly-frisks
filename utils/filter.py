import pandas as pd
import numpy as np

stopcode_colname = 'stopcode'
individual_frisk_colname = 'individual_frisked'
illegal_vals = {'lng' : np.nan}

def get_pedestrian_stops(df):
    ped_code = 2701
    return filter(df, stopcode_colname, ped_code)

def get_vehicle_stops(df):
    vehicle_code = 2702
    return filter(df, stopcode_colname, vehicle_code)

def get_individual_frisk_stops(df):
    was_frisked_code = 1
    return filter(df, individual_frisk_colname, was_frisked_code)

def remove_lng_illegal_vals(df):
    lng_colname = 'lng'
    return remove_rows_with_any_nan(df, lng_colname)

def remove_lat_illegal_vals(df):
    lat_colname = 'lat'
    return remove_rows_with_any_nan(df, lat_colname)

def remove_x_illegal_vals(df):
    x_colname = 'point_x'
    return remove_rows_with_any_nan(df, x_colname)

def remove_y_illegal_vals(df):
    y_colname = 'point_y'
    return remove_rows_with_any_nan(df, y_colname)

def remove_district_illegal_vals(df):
    district_colname = 'districtoccur'
    return remove_rows_with_any_nan(df, district_colname)

def remove_psa_illegal_vals(df):
    psa_colname = 'psa'
    return remove_rows_with_any_nan(df, psa_colname)

def remove_gender_illegal_vals(df):
    gender_colname = 'gender'
    return remove_rows_with_any_nan(df, gender_colname)

def remove_age_illegal_vals(df):
    age_colname = 'age'
    return remove_rows_with_any_nan(df, age_colname)

def remove_rows_with_any_nan(df, colnames):
    if not type(colnames) is list:
        colnames = [colnames]
    return df.dropna(subset=colnames, how = 'any')

def all_filters_for_input(df):
    df_frisks = get_individual_frisk_stops(df) # filter out only frisk stops
    df_frisks_ped = get_pedestrian_stops(df_frisks) # filter out only pedestrian frisk stops
    remove_nan_colnames = ['lng', 'lat', 'point_x', 'point_y', 'districtoccur',\
                                'psa', 'gender', 'age']
    return remove_rows_with_any_nan(df_frisks_ped, remove_nan_colnames)


def filter(df, col_name, filter_val):
    return df.loc[df[col_name] == filter_val]

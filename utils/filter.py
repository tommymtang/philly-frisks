import pandas as pd

stopcode_colname = 'stopcode'

def filter_pedestrians(df):
    ped_code = 2701
    return filter(df, stopcode_colname, ped_code)

def filter_vehicles(df):
    vehicle_code = 2702
    return filter(df, stopcode_colname, vehicle_code)


def filter(df, col_name, filter_val):
    return df.loc[df[col_name] == filter_val]

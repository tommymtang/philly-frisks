import unittest
import sys
import pandas as pd
import numpy as np
import math
sys.path.insert(0, '../utils')
import filter as ft
import preprocess as ppr

# Todo: comment before test functions
#       look up javadocs tools

path_to_data = "../philly_ped_vehicle.csv"
nrows_to_read = 300000000
df = pd.read_csv(path_to_data, nrows = nrows_to_read)

class TestFilterMethods(unittest.TestCase):

    ped_code = 2701
    vehicle_code = 2702
    stopcode_colname = 'stopcode'
    individual_frisk_colname = 'individual_frisked'
    was_frisked_code = 1;
    not_frisked_code = 0;

    def test_get_pedestrian_stops_selects_only_ped_stops(self):
        assert(self.ped_code in df[self.stopcode_colname].values)
        assert(self.vehicle_code in df[self.stopcode_colname].values)
        ped_subset = ft.get_pedestrian_stops(df)
        assert(self.ped_code in ped_subset[self.stopcode_colname].values)
        assert(self.vehicle_code not in ped_subset[self.stopcode_colname].values)

    def test_get_vehicle_stops_selects_only_vehicle_stops(self):
        assert(self.ped_code in df[self.stopcode_colname].values)
        assert(self.vehicle_code in df[self.stopcode_colname].values)
        vehicle_subset = ft.get_vehicle_stops(df)
        assert(self.ped_code not in vehicle_subset[self.stopcode_colname].values)
        assert(self.vehicle_code in vehicle_subset[self.stopcode_colname].values)

    def test_get_individual_frisk_stops_selects_only_individual_frisk_stops(self):
        assert(self.was_frisked_code in df[self.individual_frisk_colname].values)
        assert(self.not_frisked_code in df[self.individual_frisk_colname].values)
        individual_frisk_subset = ft.get_individual_frisk_stops(df)
        assert(self.was_frisked_code in individual_frisk_subset[self.individual_frisk_colname].values)
        assert(self.not_frisked_code not in individual_frisk_subset[self.individual_frisk_colname].values)

    def test_remove_lng_illegal_vals_removes_all_illegal_vals(self):
        assert(df['lng'].isna().values.any())
        subset_with_illegal_vals_removed = ft.remove_lng_illegal_vals(df)
        assert(not subset_with_illegal_vals_removed['lng'].isna().values.any())

    def test_remove_lat_illegal_vals_removes_all_illegal_vals(self):
        assert(df['lat'].isna().values.any())
        subset_with_illegal_vals_removed = ft.remove_lat_illegal_vals(df)
        assert(not subset_with_illegal_vals_removed['lat'].isna().values.any())

    def test_remove_x_illegal_vals_removes_all_illegal_vals(self):
        assert(df['point_x'].isna().values.any())
        subset_with_illegal_vals_removed = ft.remove_x_illegal_vals(df)
        assert(not subset_with_illegal_vals_removed['point_x'].isna().values.any())

    def test_remove_y_illegal_vals_removes_all_illegal_vals(self):
        assert(df['point_y'].isna().values.any())
        subset_with_illegal_vals_removed = ft.remove_y_illegal_vals(df)
        assert(not subset_with_illegal_vals_removed['point_y'].isna().values.any())

    def test_remove_district_illegal_vals_removes_all_illegal_vals(self):
        assert(df['districtoccur'].isna().values.any())
        subset_with_illegal_vals_removed = ft.remove_district_illegal_vals(df)
        assert(not subset_with_illegal_vals_removed['districtoccur'].isna().values.any())

    def test_remove_psa_illegal_vals_removes_all_illegal_vals(self):
        assert(df['psa'].isna().values.any())
        subset_with_illegal_vals_removed = ft.remove_psa_illegal_vals(df)
        assert(not subset_with_illegal_vals_removed['psa'].isna().values.any())

    def test_remove_gender_illegal_vals_removes_all_illegal_vals(self):
        assert(df['gender'].isna().values.any())
        subset_with_illegal_vals_removed = ft.remove_gender_illegal_vals(df)
        assert(not subset_with_illegal_vals_removed['gender'].isna().values.any())

    def test_remove_age_illegal_vals_removes_all_illegal_vals(self):
        assert(df['age'].isna().values.any())
        subset_with_illegal_vals_removed = ft.remove_age_illegal_vals(df)
        assert(not subset_with_illegal_vals_removed['age'].isna().values.any())

df_no_nans = ft.all_filters_for_input(df)
df_test_local_hit_rate = ppr.add_success_column(df_no_nans)

class TestPreprocessMethods(unittest.TestCase):

    success_colname = 'success'
    success_code = 1
    fail_code = 0
    frisk_colname = 'individual_frisked'
    contraband_colname = 'individual_contraband'
    arrested_colname = 'individual_arrested'

    def test_add_success_column_adds_successes_correctly(self):
        df_with_success_added = ppr.add_success_column(df)
        tmp = df[self.frisk_colname] * (df[self.contraband_colname ]+ df[self.arrested_colname])
        tmp[tmp > 0] = 1
        assert(tmp.equals(df_with_success_added[self.success_colname]))

    def test_get_local_hit_rate_input(self):
        local_hit_rate_input = ppr.get_local_hit_rate_input(df)
        assert(local_hit_rate_input.shape[1] == 3)
        assert(max(local_hit_rate_input[:,2]) == 1)

    def test_get_map_limits(self):
        local_hit_rate_input = ppr.get_local_hit_rate_input(df_test_local_hit_rate)
        map_limits = ppr.get_map_limits(local_hit_rate_input, 1)
        min_x = map_limits['min_x']
        max_x = map_limits['max_x']
        min_y = map_limits['min_y']
        max_y = map_limits['max_y']
        assert(min_x <= min(local_hit_rate_input[:,0]))
        assert(max_x >= max(local_hit_rate_input[:,0]))
        assert(min_y <= min(local_hit_rate_input[:,1]))
        assert(max_y >= min(local_hit_rate_input[:,1]))

    def test_local_hit_rate_input_to_success_map_gives_correct_shape(self):
        local_hit_rate_input = ppr.get_local_hit_rate_input(df_test_local_hit_rate)
        success_map = ppr.local_hit_rate_input_to_success_map(local_hit_rate_input)
        dilution = 1000
        point_x = local_hit_rate_input[:,0]*dilution
        point_y = local_hit_rate_input[:,1]*dilution
        max_x = max(point_x)
        min_x = min(point_x)
        max_y = max(point_y)
        min_y = min(point_y)
        width = math.ceil(max_x) - math.floor(min_x) + 1
        height = math.ceil(max_y) - math.floor(min_y) + 1
        print('blah')
        print(width)
        print(height)
        assert(width > 0)
        assert(height > 0)
        assert(success_map.shape == (height, width))

    def test_local_hit_rate_input_to_success_map_gives_correct_number_of_hits(self):

        local_hit_rate_input = ppr.get_local_hit_rate_input(df_test_local_hit_rate)
        success_map = ppr.local_hit_rate_input_to_success_map(local_hit_rate_input)
        success_map_num_hits = np.sum(success_map)
        local_hit_rate_input_num_hits = np.sum(local_hit_rate_input[:,2])
        print(local_hit_rate_input_num_hits)
        print(success_map_num_hits)

    def test_local_hit_rates_from_map_gives_correct_shape(self):
        local_hit_rate_input = ppr.get_local_hit_rate_input(df_test_local_hit_rate)
        local_hit_rate_map = ppr.get_local_hit_rate_map(local_hit_rate_input)
        points = ppr.local_hit_rate_input_to_map_coordinates(local_hit_rate_input)
        lhr_column = ppr.local_hit_rates_from_map(local_hit_rate_map, points)
        assert (lhr_column.shape[0] == local_hit_rate_input.shape[0])



if __name__ == '__main__':
    unittest.main()

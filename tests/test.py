import unittest
import sys
import pandas as pd
sys.path.insert(0, '../utils')
import filter as ft

path_to_data = "../philly_ped_vehicle.csv"
nrows_to_read = 3000000
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


if __name__ == '__main__':
    unittest.main()

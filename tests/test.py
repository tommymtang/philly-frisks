import unittest
import sys
import pandas as pd
sys.path.insert(0, '../utils')
import filter

path_to_data = "../philly_ped_vehicle.csv"
nrows_to_read = 2000
df = pd.read_csv(path_to_data, nrows = nrows_to_read)

class TestFilterMethods(unittest.TestCase):

    ped_code = 2701
    vehicle_code = 2702
    stopcode_colname = 'stopcode'

    def test_filter_pedestrians_selects_only_peds(self):
        assert(self.ped_code in df[self.stopcode_colname].values)
        assert(self.vehicle_code in df[self.stopcode_colname].values)
        ped_subset = filter.filter_pedestrians(df)
        assert(self.ped_code in ped_subset[self.stopcode_colname].values)
        assert(self.vehicle_code not in ped_subset[self.stopcode_colname].values)

    def test_filter_vehicles_selects_only_vehicles(self):
        assert(self.ped_code in df[self.stopcode_colname].values)
        assert(self.vehicle_code in df[self.stopcode_colname].values)
        vehicle_subset = filter.filter_vehicles(df)
        assert(self.ped_code not in vehicle_subset[self.stopcode_colname].values)
        assert(self.vehicle_code in vehicle_subset[self.stopcode_colname].values)


if __name__ == '__main__':
    unittest.main()

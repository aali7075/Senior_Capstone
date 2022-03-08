import unittest
import sys
import numpy as np
import collections

from simulation import *
from field_constants import *

class SimulationTestCase(unittest.TestCase):
    def setUp(self):
        self.w1={
        'center': (0, 0, 0), # 3 points, x, y, z (float)
        'shape': (2, 2), # 2 points, rows columns (int)
        'a1': .5, # float
        'b1': .5, # float
        'coil_spacing': .5, # float
        'rotation_axis': None, # string, can be one of None, 'x', 'y', 'z' (could replace with int)
        'theta': 0 # float, radians
        }
        self.w2 = {
            'center': (0, 0, 0), # 3 points, x, y, z (float)
            'shape': (2, 2), # 2 points, rows columns (int)
            'a1': .5, # float
            'b1': .5, # float
            'coil_spacing': .5, # float
            'rotation_axis': None, # string, can be one of None, 'x', 'y', 'z' (could replace with int)
            'theta': 0 # float, radians
        }
        self.p = (1,0,0)
    def test_get_full_b(self):
        arr = np.array(get_full_b(self.w1,self.w2,self.p))
        arr= arr.flatten()
        
        true_arr = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,5.944088,-4.72257866, 9.57242168, -1.09424499,  5.944088,   -4.72257866,  9.57242168, -1.09424499])

         # np.testing.assert_almost_equals returns None if it works
        self.assertEqual(None, np.testing.assert_almost_equal(arr, true_arr, 7))
    def test_field_x(self):
        v1 = field_x(1, 1, 1, .5, .5, 0,1)
        v2 = field_x(1, 1, 1, 1, 1.5, 0,1)
        v3 = field_x(0, 1, 2, 1, .5, 0,1)
        self.assertAlmostEqual(v1,-0.9683241128916278, 7)
        self.assertAlmostEqual(v2,-1.8133092841619112, 7)
        self.assertAlmostEqual(v3,-0.9797568761862816, 7)


if __name__ == '__main__':
    unittest.main()

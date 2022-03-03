import unittest
import sys

from simulation import get_full_b

class SimulationTestCase(unittest.TestCase):
    def test_get_full_b(self):
        w1 = {
        'center': (0, 0, 0), # 3 points, x, y, z (float)
        'shape': (2, 2), # 2 points, rows columns (int)
        'a1': .5, # float
        'b1': .5, # float
        'coil_spacing': .5, # float
        'rotation_axis': None, # string, can be one of None, 'x', 'y', 'z' (could replace with int)
        'theta': 0 # float, radians
    }

        w2 = {
            'center': (0, 0, 0), # 3 points, x, y, z (float)
            'shape': (2, 2), # 2 points, rows columns (int)
            'a1': .5, # float
            'b1': .5, # float
            'coil_spacing': .5, # float
            'rotation_axis': None, # string, can be one of None, 'x', 'y', 'z' (could replace with int)
            'theta': 0 # float, radians
        }
        arr = get_full_b(w1,w2,(1,0,0))
        true_arr = [[ 0, 0 ,0, 0, 0, 0, 0, 0], [ 0, 0 ,0, 0, 0, 0, 0, 0],
        [ 5.944088, -4.72257866, 9.57242168, -1.09424499, 5.944088, -4.72257866, 9.57242168, -1.09424499]]
        self.assertEqual(arr, true_arr)

if __name__ == '__main__':
    unittest.main()

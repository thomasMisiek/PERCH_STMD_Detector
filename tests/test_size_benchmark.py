import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark_sizes import inject_size
from stmd_model import apparent_diameter_px

class TestSizeControlledBenchmark(unittest.TestCase):
    def test_projected_size_changes_with_distance_not_trajectory(self):
        self.assertAlmostEqual(apparent_diameter_px(7,1),apparent_diameter_px(7,2)*2,delta=.001)
        self.assertGreater(apparent_diameter_px(20,1),apparent_diameter_px(7,1))
    def test_only_local_image_region_is_changed(self):
        src=np.full((2,320,320,3),180,dtype=np.uint8)
        trajectory=np.array([[100,120],[102,120]])
        for diameter_mm in (7,20):
            dst=inject_size(src,trajectory,1,diameter_mm)
            self.assertTrue(np.array_equal(dst[:,0:80],src[:,0:80]))
            self.assertLess(dst[0,120,100,0],180)
            self.assertTrue(np.array_equal(src,np.full_like(src,180)))

if __name__=='__main__': unittest.main()

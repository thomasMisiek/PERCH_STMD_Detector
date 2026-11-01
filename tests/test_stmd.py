import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import stmd_model as s
import benchmark as b

class STMDTests(unittest.TestCase):
    def test_projected_target_shrinks_with_distance(self):
        values=[s.apparent_diameter_px(6,d) for d in (.5,1,2)]
        self.assertTrue(values[0]>values[1]>values[2]>0)
        self.assertAlmostEqual(values[2],.655,places=2)

    def test_subpixel_target_changes_at_least_one_output_pixel(self):
        frame=np.full((320,320,3),200,dtype=np.uint8)
        s.stamp(frame,(165,68),s.apparent_diameter_px(6,2))
        self.assertLess(int(frame[68,165,0]),200)
        self.assertGreater(int(frame[68,165,0]),0)  # fractional coverage, not binary dark pixel

    def test_detector_blank_initialization(self):
        d=s.Detector()
        frame=np.full((320,320,3),150,dtype=np.uint8)
        first=d.step(frame)
        self.assertEqual(first['peak'],0)
        second=d.step(frame)
        self.assertTrue(np.isfinite(second['peak']))

    def test_control_and_target_have_same_sensor_format(self):
        frame=np.full((2,320,320,3),210,dtype=np.uint8)
        with_dot,ground_truth=s.make_frames(frame,2.0)
        self.assertEqual(with_dot.shape,frame.shape)
        self.assertEqual(ground_truth.shape,(2,2))
        self.assertTrue(np.any(with_dot!=frame))

    def test_benchmark_window(self):
        self.assertEqual(b.N,500)
        self.assertEqual(b.FPS,25.0)
        self.assertEqual(b.SIDE,320)

if __name__=='__main__':unittest.main()

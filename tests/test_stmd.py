"""Simple consistency checks; not evidence of real-world detection accuracy."""
import unittest
import numpy as np
import multi_hypothesis as m
from stmd_model import Detector, apparent_diameter_px

class TestDemo(unittest.TestCase):
    def test_projection_scales_with_distance(self):
        self.assertGreater(apparent_diameter_px(6, 0.5), apparent_diameter_px(6, 1.0))
        self.assertGreater(apparent_diameter_px(6, 1.0), apparent_diameter_px(6, 2.0))

    def test_trajectory_stays_in_frame(self):
        p=m.fast_trajectory(250)
        self.assertEqual(p.shape, (250,2))
        self.assertTrue(np.all((p >= 12) & (p < 308)))

    def test_no_prior_initialization(self):
        tracker=m.MultiHypothesis(np.ones(4))
        self.assertEqual(len(tracker.tracks), 0)
        r, tracks, _ = tracker.step([np.zeros((320,320), np.float32) for _ in range(4)])
        self.assertFalse(r['detected'])
        self.assertEqual(len(tracks), 0)

    def test_first_frame_neural_memory_empty(self):
        detector=Detector()
        response=detector.step(np.full((320,320,3), 127, np.uint8))
        self.assertEqual(response['peak'], 0)

if __name__ == '__main__':
    unittest.main()

import unittest, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from spatial_trace import SpatialTrace,infer
from evaluate import angular_error_deg
class TestSpatialTrace(unittest.TestCase):
    def test_angular_error(self):
        xy=np.array([[160,160],[160,160]],np.float32)
        gt=np.array([[160,160],[180,160]],np.float32)
        d=angular_error_deg(xy,gt)
        self.assertLess(d[0],.01)
        self.assertGreater(d[1],5.)
    def test_no_future_leak(self):
        m=SpatialTrace(mode='annulus',decay=.92,neighborhood=11.,inhibition=1.3,gain=70)
        cur=np.array([[100,100,2,0],[200,200,1,1]],np.float32)
        scores,_=m.step(cur)
        self.assertAlmostEqual(float(scores[0]),2.,places=3)
        self.assertAlmostEqual(float(scores[1]),1.,places=3)
    def test_one_unified_map(self):
        m=SpatialTrace();p=np.array([[100,100,2,0]],np.float32)
        m.step(p)
        self.assertEqual(m.trace.shape,(320,320))
        self.assertGreater(m.trace.sum(),0)
    def test_decay(self):
        m=SpatialTrace(decay=.8)
        m.step(np.array([[100,100,2,0]],np.float32));v=float(m.trace.sum())
        m.step(np.zeros((0,4),np.float32))
        # A null-candidate frame does not add memory; see implementation caveat
        self.assertLessEqual(float(m.trace.sum()),v+1e-6)
    def test_infer_no_gt(self):
        a=np.zeros((4,40,4),np.float32);c=np.full(4,2)
        a[:,:2]=np.array([[100,120,2,0],[150,200,.9,1]],np.float32)
        xy,s,raw=infer(a,c,{'mode':'raw','gain':0})
        self.assertTrue(np.all(xy[:,0]==100))
if __name__=='__main__':unittest.main()

# Independent architecture consistency checks, including the 14-map banks.
class TestVelocityBankArchitecture(unittest.TestCase):
    def test_three_banks_four_sensory_pathways(self):
        from comparison.multi_hypothesis import CONFIGS, get_detectors
        from comparison.multi_stmd_model import V
        self.assertEqual(len(V),14)
        self.assertEqual(len(CONFIGS),3)
        self.assertEqual(len(get_detectors()),4)

    def test_velocity_maps_are_not_object_tracks(self):
        from comparison.multi_stmd_model import Detector, V
        d=Detector()
        self.assertEqual(d.A.shape,(len(V),320,320))
        self.assertFalse(hasattr(d,'tracks'))

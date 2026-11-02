import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'src'))
from predictive_field import PredictiveField, infer
from motion_bank_benchmark_fast import velocity_set

class TestPredictiveField(unittest.TestCase):
    def test_24_velocity_channels(self):
        v=velocity_set('grid24')
        self.assertEqual(v.shape,(24,2))
        self.assertTrue(np.all(np.any(v!=0,axis=1)))
    def test_memory_stays_bounded_and_causal(self):
        fld=PredictiveField()
        first=np.array([[120,120,2.5,0]],np.float32)
        d0=fld.step(first)
        self.assertLess(float(d0['prior'].max()),1e-6)
        self.assertGreater(float(d0['memory'].max()),.99)
        d1=fld.step(np.array([[121,120,2.5,0]],np.float32))
        self.assertAlmostEqual(float(d1['prior'].max()),1,places=3)
        self.assertTrue(np.all((d1['memory']>=0)&(d1['memory']<=1)))
    def test_prediction_uses_only_previous_estimates(self):
        fld=PredictiveField()
        for x in (100,102,104):fld.step(np.array([[x,100,2.,0]],np.float32))
        self.assertGreater(float(fld.velocity[0]),0)
        p=fld.step(np.zeros((0,4),np.float32))
        self.assertGreater(float(p['prior'][100,105]),0)
        self.assertTrue(np.isnan(p['x']))
    def test_infer_shapes(self):
        arr=np.zeros((4,3,4),np.float32)
        arr[:,0]=[130,160,2.0,0]
        xy,score,v=infer(arr,np.ones(4,dtype=int))
        self.assertEqual(xy.shape,(4,2))
        self.assertTrue(np.all(np.isfinite(xy)))

if __name__=='__main__':unittest.main()

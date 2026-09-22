import json,unittest
from pathlib import Path
import numpy as np
import torch
from models.predict import FEATURES
from models.predict_improved import ImprovedPredictor

class ImprovedModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.sample=json.loads(Path('example.json').read_text())
        self.expected=json.loads(Path('improved_example_output.json').read_text())
    def test_reproduces_actual_exported_examples(self):
        for name in ['vortex','physical','portia']:
            result=ImprovedPredictor(name).predict(self.sample)
            self.assertAlmostEqual(result['any_abnormality_probability'],self.expected[name]['any_abnormality_probability'],places=6)
            self.assertEqual(result['fold'],1)
            self.assertNotIn('cutoff',result)
    def test_row_order_and_missingness(self):
        row=np.array([self.sample.get(k,np.nan) for k in FEATURES]);other=row.copy();other[1]=30;other[18]=np.nan
        for name in ['vortex','physical','portia']:
            m=ImprovedPredictor(name);p=m.joint(np.stack([row,other]))
            self.assertTrue(np.allclose(p.sum(1),1,atol=1e-6))
            self.assertTrue(np.allclose(p[0],m.joint(row)[0],atol=1e-6))
            self.assertTrue(np.allclose(p,m.joint(np.stack([other,row]))[::-1],atol=1e-6))
            with self.assertRaises(ValueError):m.predict(self.sample|{'HE_glu':110})
            with self.assertRaises(ValueError):m.predict(self.sample|{'age':60})

if __name__=='__main__':unittest.main()

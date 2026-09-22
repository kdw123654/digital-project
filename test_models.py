"""Small portable smoke tests; full cohort parity was verified before export."""
import json,unittest
from pathlib import Path
import numpy as np
from models.predict import Predictor,FEATURES

class ModelTests(unittest.TestCase):
    def setUp(self):self.state=json.loads(Path('example.json').read_text())
    def test_probabilities_and_example(self):
        expected=json.loads(Path('example_output.json').read_text())
        for name in ['nn','lr','sparse']:
            model=Predictor(name);raw=[self.state.get(k,np.nan) for k in FEATURES];p=model.joint(raw)
            self.assertEqual(p.shape,(1,16));self.assertTrue(np.isfinite(p).all())
            self.assertAlmostEqual(float(p.sum()),1.,places=6)
            self.assertAlmostEqual(model.predict(self.state)['any_abnormality_probability'],expected[name]['any_abnormality_probability'],places=6)
    def test_reject_wrong_labels_and_age(self):
        for data in [self.state|{'HE_glu':110},self.state|{'age':70},self.state|{'sex':9},self.state|{'living_alone':9}]:
            with self.assertRaises(ValueError):Predictor('nn').predict(data)
    def test_missing_and_row_isolation(self):
        raw=np.array([self.state.get(k,np.nan) for k in FEATURES]);other=raw.copy();other[1]=30
        for name in ['nn','lr','sparse']:
            model=Predictor(name)
            self.assertTrue(np.allclose(model.joint(raw)[0],model.joint(np.stack([raw,other]))[0],atol=1e-6))
            missing=self.state.copy();missing.pop('living_alone')
            self.assertIn('living_alone',model.predict(missing)['missing_inputs'])

if __name__=='__main__':unittest.main()

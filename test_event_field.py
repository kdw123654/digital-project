import json,unittest
from pathlib import Path
import numpy as np,torch
from models.predict_event_field import EventFieldPredictor
from models.predict import FEATURES

class EventFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.model=EventFieldPredictor();cls.person=json.loads(Path('example.json').read_text())
    def test_frozen_example_and_no_external_experts(self):
        actual=self.model.predict(self.person);expected=json.loads(Path('event_field_example_output.json').read_text())
        for k,p in expected['component_probabilities'].items():self.assertAlmostEqual(actual['component_probabilities'][k],p,12)
        self.assertTrue(actual['single_shared_field']);self.assertEqual(actual['external_model_calls'],0)
        self.assertEqual(actual,self.model.predict(self.person))
    def test_row_isolation_and_missing_inputs(self):
        x=np.array([[self.person[k] for k in FEATURES]]*3,float);x[1,2:4]=np.nan;x[2,0]=37
        p=self.model.joint(x);np.testing.assert_allclose(self.model.joint(x[::-1])[::-1],p,atol=1e-6,rtol=1e-5)
        self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p.sum(1),1)
        with self.assertRaises(ValueError):self.model.predict(self.person|{'HE_glu':110})
    def test_core_has_one_state_and_control_classes_are_separate(self):
        from models.event_field_baselines import MatchedResidualMLP,HistoryGRU
        cell=self.model.model;state=cell.initial_state(2)
        self.assertEqual(state[0].shape,(2,16));self.assertEqual(state[2].shape,(2,16,16))
        self.assertFalse(any(isinstance(m,(MatchedResidualMLP,HistoryGRU)) for m in cell.modules()))

if __name__=='__main__':unittest.main()

import json,unittest
from pathlib import Path
import numpy as np,torch
from models.predict_clinical import ClinicalPredictor
from models.clinical_multitask import TARGETS
from models.predict import FEATURES

class ClinicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.person=json.loads(Path('example.json').read_text());cls.model=ClinicalPredictor()
    def test_joint_outputs_match_export_and_use_one_model_pass(self):
        calls=[];handle=self.model.model.register_forward_hook(lambda *args:calls.append(1))
        try:actual=self.model.predict(self.person)
        finally:handle.remove()
        self.assertEqual(len(calls),1);expected=json.loads(Path('clinical_example_output.json').read_text())
        for key in ['component_probabilities','estimated_measurements']:
            for name,value in expected[key].items():self.assertAlmostEqual(actual[key][name],value,10)
        self.assertTrue(actual['single_shared_field']);self.assertEqual(actual['external_model_calls'],0)
    def test_target_values_cannot_enter_the_input(self):
        for name in TARGETS:
            with self.assertRaises(ValueError):self.model.predict(self.person|{name:100})
    def test_all_modes_and_baselines_have_the_correct_output_contract(self):
        for name in self.model.metadata['models']:
            result=ClinicalPredictor(name).predict(self.person)
            self.assertEqual('estimated_measurements' in result,not (name.endswith('classification') or name=='LR'))
            p=list(result['component_probabilities'].values());self.assertTrue(all(0<=a<=1 for a in p));self.assertGreaterEqual(result['any_abnormality_probability']+1e-12,max(p))
    def test_people_are_isolated_and_missing_covariates_supported(self):
        raw=np.array([[self.person[k] for k in FEATURES]]*3,float);raw[1,2:4]=np.nan;raw[2,0]=37
        p,v=self.model.infer(raw);q,u=self.model.infer(raw[::-1]);np.testing.assert_allclose(p,q[::-1],atol=1e-6);np.testing.assert_allclose(v,u[::-1],atol=1e-3)

if __name__=='__main__':unittest.main()

"""Portable prediction and archive boundary checks on synthetic inputs only."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from models.predict_hybrid import HybridPredictor,HERE
from models.model_archive import load_archive

class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scope=threadpool_limits(limits=1);cls.scope.__enter__()
        cls.person=json.loads(Path('example.json').read_text(encoding='utf-8'))
    @classmethod
    def tearDownClass(cls):cls.scope.__exit__(None,None,None)

    def test_saved_outputs_and_missing_inputs(self):
        expected=json.loads(Path('hybrid_example_output.json').read_text(encoding='utf-8'))
        for fold,category,result in zip([1,2,1],['three_selected','three_selected','LR_selected'],expected):
            model=HybridPredictor(fold,category);actual=model.predict(self.person)
            for key,value in result['component_probabilities'].items():self.assertAlmostEqual(actual['component_probabilities'][key],value,12)
            changed=model.predict(self.person|{'HE_wc':None,'WHtR':None,'walking_days':None})
            self.assertEqual(actual,model.predict(self.person))
            self.assertTrue(all(0<=v<=1 for v in changed['component_probabilities'].values()))
            self.assertEqual(actual['external_model_calls'],0)
        with self.assertRaises(ValueError):model.predict(self.person|{'HE_glu':120})
        with self.assertRaises(ValueError):model.predict(self.person|{'sex':3})

    def test_state_model_is_row_independent(self):
        from models.predict import FEATURES
        model=HybridPredictor(2);raw=np.array([[self.person[k] for k in FEATURES]]*3,float)
        raw[1,2:4]=np.nan;raw[2,0]=37
        original=model.joint(raw)
        # Float32 wave GEMM can round differently when batch shape/order changes.
        # Repeated predictions for the identical batch are checked exactly above.
        np.testing.assert_allclose(model.joint(raw[::-1])[::-1],original,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(np.concatenate([model.joint(row) for row in raw]),original,rtol=1e-6,atol=1e-7)

    def test_archive_has_no_pickle_and_rejects_unknown_classes(self):
        for path in HERE.rglob('*.npz'):
            with np.load(path,allow_pickle=False) as arrays:
                self.assertTrue(all(not arrays[k].dtype.hasobject for k in arrays))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);np.savez(folder/'weights.npz')
            (folder/'model.json').write_text(json.dumps({'format':'allowlisted-model-npz-v1','value':{'type':'model','class':'Unknown','attributes':{}}}))
            with self.assertRaises(ValueError):load_archive(folder)

if __name__=='__main__':unittest.main()

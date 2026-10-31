"""Checks for publication configuration selection and portable commands."""
import unittest
from pathlib import Path
import tempfile
import reproduce
from reproduction_paths import qwen_reference_cache

class ReproductionTests(unittest.TestCase):
    def test_configurations_are_within_paper_scope(self):
        rows=reproduce.read_json('configs/training/index.json');names=[r['name']for r in rows]
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(sum('loss_ablation' in n for n in names),18)
        self.assertFalse(any('congress' in n or n.startswith('ira') or 'intercept_test' in n for n in names))
        for row in rows:
            if row['name'].startswith('agnews'):self.assertEqual(row['rows'],127600)
    def test_ablation_zero_weight_is_preserved(self):
        c,cfg=reproduce.training_command('agnews_qwen_loss_ablation_no_activity_seed1234')
        self.assertEqual(float(c[c.index('--lambda_tweet_sparse')+1]),0)
        self.assertNotIn('--require_cached_embeddings',c);self.assertNotIn('--resume',c)
    def test_device_override_preserves_scientific_parameters(self):
        _,base=reproduce.training_command('agnews_qwen_all_fixed_balance')
        _,changed=reproduce.training_command('agnews_qwen_all_fixed_balance',device='cpu',precision='fp32')
        self.assertEqual({k:v for k,v in base.items()if k not in ['device','precision']},{k:v for k,v in changed.items()if k not in ['device','precision']})
    def test_ambiguous_cache_is_never_guessed(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);d=root/'runs/agnews_embedding_comparison/_information_reference_cache/qwen';d.mkdir(parents=True)
            with self.assertRaises(ValueError):qwen_reference_cache(root)
            (d/'information_reference_codes_one.npz').touch()
            self.assertEqual(qwen_reference_cache(root).name,'information_reference_codes_one.npz')
            (d/'information_reference_codes_two.npz').touch()
            with self.assertRaises(ValueError):qwen_reference_cache(root)
if __name__=='__main__':unittest.main()

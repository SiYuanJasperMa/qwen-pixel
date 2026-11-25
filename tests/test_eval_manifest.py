import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pixel.eval_manifest import output_budgets,prepare_manifest


class EvalManifestTests(unittest.TestCase):
    def test_long_program_can_retry_without_changing_its_input_history(self):
        self.assertEqual(output_budgets(1500,1200),[1200,2400,4800,6144])
        self.assertEqual(output_budgets(6000,1200),[1200,2192])
        self.assertEqual(output_budgets(8192,1200),[])

    def test_cached_results_cannot_masquerade_as_a_different_run(self):
        with tempfile.TemporaryDirectory()as folder:
            first=prepare_manifest(folder,{'adapter':'A','budget':1200})
            self.assertEqual(first,prepare_manifest(folder,{'adapter':'A','budget':1200}))
            with self.assertRaisesRegex(ValueError,'different model'):
                prepare_manifest(folder,{'adapter':'B','budget':1200})
            with self.assertRaisesRegex(ValueError,'different model'):
                prepare_manifest(folder,{'adapter':'A','budget':1800})

    def test_unverified_legacy_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory()as folder:
            case=Path(folder)/'sample';case.mkdir();(case/'evaluation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Legacy cached'):
                prepare_manifest(folder,{'adapter':'A'})


if __name__=='__main__':unittest.main()

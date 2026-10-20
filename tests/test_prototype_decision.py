import unittest
from monitoring.prototype_decision import decide

class TestPrototypeDecision(unittest.TestCase):
    def test_block(self): self.assertEqual(decide(approved_target=False,ml_score=0.1)['decision'],'BLOCK')
    def test_allow(self): self.assertEqual(decide(approved_target=True,ml_score=0.1)['decision'],'ALLOW')
    def test_review(self): self.assertEqual(decide(approved_target=True,ml_score=0.5)['decision'],'REVIEW')
    def test_gate(self): self.assertEqual(decide(approved_target=True,ml_score=0.9)['decision'],'GATE')
    def test_missing_score(self): self.assertEqual(decide(approved_target=True,ml_score=None)['decision'],'REVIEW')
    def test_bad_flag(self):
        with self.assertRaises(ValueError):decide(approved_target='false',ml_score=0.1)

if __name__=='__main__':unittest.main()

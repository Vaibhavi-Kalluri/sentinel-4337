import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_synthetic_dataset import (
    METADATA_COLUMNS,
    MODEL_FEATURES,
    SCENARIOS,
    generate_dataset,
)
from split_synthetic_dataset import create_splits


class TestSyntheticDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_dataset(
            samples_per_scenario=60,
            seed=4337,
            batches_per_scenario=6,
        )

    def test_expected_row_count(self):
        self.assertEqual(len(self.df), 60 * len(SCENARIOS))

    def test_expected_scenarios_and_counts(self):
        self.assertEqual(set(self.df["scenario_type"]), set(SCENARIOS))
        self.assertTrue((self.df["scenario_type"].value_counts() == 60).all())

    def test_feature_schema(self):
        self.assertEqual(
            list(self.df.columns),
            MODEL_FEATURES + METADATA_COLUMNS,
        )

    def test_metadata_not_model_features(self):
        self.assertTrue(set(METADATA_COLUMNS).isdisjoint(MODEL_FEATURES))

    def test_label_semantics_are_explicit(self):
        self.assertEqual(set(self.df["label"]), {0, 1})
        self.assertTrue(
            (self.df["label_source"] == "synthetic_simulation").all()
        )

    def test_reproducibility(self):
        second = generate_dataset(
            samples_per_scenario=60,
            seed=4337,
            batches_per_scenario=6,
        )
        self.assertTrue(self.df.equals(second))

    def test_valid_counts_and_binary_features(self):
        count_features = [
            "trace_count", "unique_destinations", "approve_trace_count",
            "transfer_from_trace_count", "value_transfer_trace_count",
            "errored_trace_count",
        ]
        for col in count_features:
            self.assertTrue((self.df[col] >= 0).all())
            self.assertTrue(
                np.equal(self.df[col], np.floor(self.df[col])).all()
            )

        for col in [
            "has_approval", "has_transfer_from", "has_value_transfer",
            "has_trace_errors", "has_paymaster",
        ]:
            self.assertTrue(set(self.df[col].unique()).issubset({0, 1}))

    def test_log_features_match_counts(self):
        for col in [
            "trace_count", "unique_destinations", "approve_trace_count",
            "transfer_from_trace_count", "value_transfer_trace_count",
            "errored_trace_count",
        ]:
            np.testing.assert_allclose(
                self.df[f"log1p_{col}"],
                np.log1p(self.df[col].astype(float)),
            )

    def test_minimum_validation(self):
        with self.assertRaises(ValueError):
            generate_dataset(samples_per_scenario=9, seed=1)

    def test_batches_are_scenario_specific(self):
        self.assertTrue(
            (self.df.groupby("generation_batch")["scenario_type"].nunique() == 1).all()
        )

    def test_grouped_splits_have_no_batch_leakage(self):
        splits = create_splits(self.df, seed=4337)
        batch_sets = {
            name: set(part["generation_batch"])
            for name, part in splits.items()
        }
        names = list(batch_sets)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                self.assertTrue(batch_sets[left].isdisjoint(batch_sets[right]))

        for part in splits.values():
            self.assertEqual(set(part["scenario_type"]), set(SCENARIOS))
            self.assertEqual(set(part["label"]), {0, 1})


if __name__ == "__main__":
    unittest.main()

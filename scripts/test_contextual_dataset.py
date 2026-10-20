import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_contextual_dataset import (
    CONTEXT_FEATURES,
    MODEL_COLUMNS,
    generate_contextual_dataset,
)
from split_contextual_dataset import split_by_session


class TestContextualDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_contextual_dataset(
            samples_per_scenario=60,
            batches_per_scenario=6,
            sessions=6,
            seed=4337,
        )

    def test_expected_rows_and_sessions(self):
        self.assertEqual(len(self.df), 60 * 6)
        self.assertEqual(self.df["session_id"].nunique(), 6)

    def test_every_session_contains_all_scenarios(self):
        expected = set(self.df["scenario_type"].unique())
        for _, group in self.df.groupby("session_id"):
            self.assertEqual(set(group["scenario_type"]), expected)

    def test_context_features_exist_and_are_finite(self):
        self.assertTrue(set(CONTEXT_FEATURES).issubset(self.df.columns))
        self.assertTrue(
            self.df[CONTEXT_FEATURES].notna().all().all()
        )
        self.assertTrue(
            self.df[CONTEXT_FEATURES].apply(
                lambda column: column.map(
                    lambda value: pd.notna(value) and abs(float(value)) < float("inf")
                )
            ).all().all()
        )

    def test_binary_context_features(self):
        for column in ["dest_seen_before", "selector_seen_before"]:
            self.assertTrue(set(self.df[column].unique()).issubset({0, 1}))

    def test_nonnegative_recent_count(self):
        self.assertTrue((self.df["recent_operation_count"] >= 0).all())

    def test_history_is_reproducible(self):
        other = generate_contextual_dataset(
            samples_per_scenario=60,
            batches_per_scenario=6,
            sessions=6,
            seed=4337,
        )
        pd.testing.assert_frame_equal(self.df, other)

    def test_model_columns_exclude_metadata(self):
        self.assertTrue(set(MODEL_COLUMNS).isdisjoint({
            "label", "scenario_type", "session_id", "wallet_id",
            "destination_id", "selector_id", "timestamp_seconds",
            "operation_index", "value_eth", "generation_batch",
            "scenario_id", "label_source",
        }))

    def test_session_splits_do_not_leak(self):
        splits = split_by_session(self.df, seed=4337)
        session_sets = {
            name: set(part["session_id"])
            for name, part in splits.items()
        }
        self.assertTrue(session_sets["train"].isdisjoint(session_sets["validation"]))
        self.assertTrue(session_sets["train"].isdisjoint(session_sets["test"]))
        self.assertTrue(session_sets["validation"].isdisjoint(session_sets["test"]))

        expected = set(self.df["scenario_type"].unique())
        for part in splits.values():
            self.assertEqual(set(part["scenario_type"]), expected)
            self.assertEqual(set(part["label"].unique()), {0, 1})

    def test_split_is_reproducible(self):
        first = split_by_session(self.df, seed=4337)
        second = split_by_session(self.df, seed=4337)
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])


if __name__ == "__main__":
    unittest.main()

"""Reference gates and coordinate conversion; all examples use temporary synthetic data."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.compare_vcg_annotations import (
    compare_boundaries, compare_qrs_references, main, match_qrs_events,
)


class TestBoundaryReferences(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "artifacts").mkdir()
        self.predictions = pd.DataFrame([
            {"record_id": "patient1/record1", "patient_id": "patient1", "label_name": "HC", "split": "train",
             "fs": 1000.0, "qrs_on_ms": -30.0, "qrs_off_ms": 50.0, "t_on_ms": 180.0, "t_off_ms": 330.0},
            {"record_id": "patient2/record2", "patient_id": "patient2", "label_name": "MI", "split": "val",
             "fs": 500.0, "qrs_on_ms": -40.0, "qrs_off_ms": 80.0, "t_on_ms": np.nan, "t_off_ms": np.nan},
        ])
        self.reference = {
            "record_id": "patient1/record1", "reviewer": "Independent reviewer", "review_status": "confirmed",
            "independent_reference": True, "time_reference": "median_relative_ms",
            "qrs_on_ms": -32.0, "qrs_off_ms": 54.0, "t_on_ms": 190.0, "t_off_ms": 350.0,
        }
        self.artifact = {"record_id": "patient1/record1", "fs": 1000.0,
                         "anchors": [1000, 2000, 3000], "used_samples": [2000], "median_anchor_sample": 280}
        self.write_artifact(self.artifact)

    def tearDown(self):
        self.temp.cleanup()

    def write_artifact(self, artifact):
        path = self.root / "artifacts" / "patient1_record1.json"
        path.write_text(json.dumps(artifact))

    def compare(self, references=None, predictions=None):
        return compare_boundaries(self.predictions if predictions is None else predictions,
                                  pd.DataFrame([self.reference]) if references is None else references, self.root)

    def test_independent_confirmed_boundary_reference_and_signed_error(self):
        comparison, summary, audit = self.compare()
        self.assertTrue(audit.accepted.all())
        self.assertEqual(len(comparison), 4)
        np.testing.assert_allclose(comparison.signed_error_ms, [2, -4, -10, -20])
        self.assertEqual(comparison.comparison_scope.unique().tolist(), ["median_estimate_vs_expert_same_median"])
        self.assertTrue(comparison.within_20_ms.all())
        self.assertFalse(comparison.iloc[-1].within_10_ms)
        self.assertEqual(summary.valid_references.sum(), 4)

    def test_predicted_columns_never_become_references(self):
        ref = {key: value for key, value in self.reference.items() if not key.endswith("_ms")}
        ref.update(pred_qrs_on_ms=-30, pred_qrs_off_ms=50, pred_t_on_ms=180, pred_t_off_ms=330)
        comparison, summary, audit = self.compare(pd.DataFrame([ref]))
        self.assertTrue(comparison.empty)
        self.assertTrue(summary.empty)
        self.assertEqual(audit.iloc[0].reason, "no_finite_boundary_reference")

    def test_strict_acceptance_gates(self):
        variants = [
            ("review_status", "pending", "review_status_not_confirmed"),
            ("reviewer", " ", "reviewer_missing"),
            ("independent_reference", False, "independent_reference_not_explicitly_true"),
            ("independent_reference", "not sure", "independent_reference_not_explicitly_true"),
            ("usable_reference", False, "usable_reference_not_true"),
            ("time_reference", "", "time_reference_missing_or_unsupported"),
            ("time_reference", "raw_samples", "time_reference_missing_or_unsupported"),
        ]
        for key, value, reason in variants:
            with self.subTest(key=key, value=value):
                ref = {**self.reference, key: value}
                comparison, _, audit = self.compare(pd.DataFrame([ref]))
                self.assertTrue(comparison.empty)
                self.assertEqual(audit.iloc[0].reason, reason)

    def test_raw_sample_conversion_checks_actual_retained_anchor(self):
        ref = {**self.reference, "time_reference": "raw_recording_ms", "reference_anchor_sample": 2000,
               "qrs_on_ms": 1968, "qrs_off_ms": 2054, "t_on_ms": 2190, "t_off_ms": 2350}
        comparison, _, audit = self.compare(pd.DataFrame([ref]))
        self.assertTrue(audit.accepted.all())
        np.testing.assert_allclose(comparison.reference_ms, [-32, 54, 190, 350])
        self.assertEqual(comparison.comparison_scope.unique().tolist(), ["median_estimate_vs_selected_observed_beat"])
        for anchor, reason in [(1000, "reference_anchor_not_in_retained_beats"),
                               (2000.5, "raw_reference_requires_integer_anchor_sample"),
                               (np.nan, "raw_reference_requires_integer_anchor_sample")]:
            _, _, audit = self.compare(pd.DataFrame([{**ref, "reference_anchor_sample": anchor}]))
            self.assertEqual(audit.iloc[0].reason, reason)

    def test_raw_conversion_uses_sampling_frequency_not_one_sample_per_ms(self):
        pred = self.predictions.iloc[[0]].copy()
        pred.loc[:, "fs"] = 500
        self.write_artifact({**self.artifact, "fs": 500})
        ref = {**self.reference, "time_reference": "raw_recording_ms", "reference_anchor_sample": 2000,
               "qrs_on_ms": 3968, "qrs_off_ms": 4054, "t_on_ms": 4190, "t_off_ms": 4350}
        comparison, _, _ = self.compare(pd.DataFrame([ref]), pred)
        np.testing.assert_allclose(comparison.reference_ms, [-32, 54, 190, 350])

    def test_sampling_frequency_mismatch_rejected(self):
        self.write_artifact({**self.artifact, "fs": 500})
        ref = {**self.reference, "time_reference": "raw_recording_ms", "reference_anchor_sample": 2000}
        with self.assertRaisesRegex(ValueError, "frequency differs"):
            self.compare(pd.DataFrame([ref]))

    def test_saved_median_window_enforced_for_median_and_raw_references(self):
        self.write_artifact({**self.artifact, "median_time_start_ms": -280, "median_time_end_ms": 650})
        comparison, _, audit = self.compare()
        self.assertTrue(audit.accepted.all())
        self.assertTrue((comparison.window_validation == "validated_with_saved_median_window").all())
        for reference in [
            {**self.reference, "t_off_ms": 700},
            {**self.reference, "qrs_on_ms": -300},
            {**self.reference, "time_reference": "raw_recording_ms", "reference_anchor_sample": 2000,
             "qrs_on_ms": 1968, "qrs_off_ms": 2054, "t_on_ms": 2190, "t_off_ms": 2700},
        ]:
            with self.subTest(reference=reference):
                comparison, _, audit = self.compare(pd.DataFrame([reference]))
                self.assertTrue(comparison.empty)
                self.assertEqual(audit.iloc[0].reason, "reference_boundary_outside_selected_beat_window")

    def test_npz_median_window_fallback_and_explicit_missing_window(self):
        comparison, _, _ = self.compare()
        self.assertTrue((comparison.window_validation == "no_window_data").all())
        np.savez(self.root / "artifacts" / "patient1_record1.npz", time_ms=np.arange(-280, 651))
        comparison, _, audit = self.compare()
        self.assertTrue(audit.accepted.all())
        self.assertTrue((comparison.window_validation == "validated_with_saved_median_window").all())
        comparison, _, audit = self.compare(pd.DataFrame([{**self.reference, "t_off_ms": 700}]))
        self.assertTrue(comparison.empty)
        self.assertEqual(audit.iloc[0].reason, "reference_boundary_outside_selected_beat_window")

    def test_missing_predictions_count_in_coverage_not_mae(self):
        ref = {**self.reference, "record_id": "patient2/record2"}
        comparison, summary, audit = self.compare(pd.DataFrame([ref]))
        self.assertTrue(audit.accepted.all())
        self.assertEqual(comparison.matched.sum(), 2)
        self.assertEqual(comparison.missing_prediction.sum(), 2)
        missing = summary[summary.boundary.isin(["t_on", "t_off"])]
        self.assertEqual(missing.valid_references.sum(), 2)
        self.assertTrue((missing.prediction_coverage_pct == 0).all())
        self.assertTrue(missing.mae_ms.isna().all())
        self.assertTrue((missing.within_60_ms_pct_of_references == 0).all())

    def test_partial_reference_only_counts_finite_boundaries(self):
        ref = {**self.reference, "qrs_on_ms": np.nan, "qrs_off_ms": np.nan, "t_off_ms": np.inf}
        comparison, summary, _ = self.compare(pd.DataFrame([ref]))
        self.assertEqual(comparison.boundary.tolist(), ["t_on"])
        self.assertEqual(summary.valid_references.sum(), 1)

    def test_duplicate_keys_raise(self):
        with self.assertRaisesRegex(ValueError, "Prediction record_id"):
            self.compare(predictions=pd.concat([self.predictions, self.predictions.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "natural reference"):
            self.compare(pd.DataFrame([self.reference, self.reference]))
        refs = pd.DataFrame([{**self.reference, "reference_id": "id"},
                             {**self.reference, "reference_id": "id", "reviewer": "Other reviewer"}])
        with self.assertRaisesRegex(ValueError, "reference_id"):
            self.compare(refs)

    def test_unknown_record_and_invalid_reference_order_are_audited(self):
        for ref, reason in [({**self.reference, "record_id": "absent/record"}, "record_not_in_prediction_manifest"),
                            ({**self.reference, "qrs_on_ms": 100}, "reference_boundary_order_invalid")]:
            comparison, _, audit = self.compare(pd.DataFrame([ref]))
            self.assertTrue(comparison.empty)
            self.assertEqual(audit.iloc[0].reason, reason)

    def test_cli_writes_header_only_for_pending_template(self):
        (self.root / "features").mkdir()
        self.predictions.to_csv(self.root / "features" / "all_features.csv", index=False)
        pending = self.root / "pending.csv"
        pd.DataFrame([{**self.reference, "review_status": "pending"}]).to_csv(pending, index=False)
        self.assertEqual(main(["--output-dir", str(self.root), "--annotations", str(pending)]), 0)
        self.assertTrue(pd.read_csv(self.root / "evaluation" / "boundary_reference_comparison.csv").empty)


class TestQRSReferences(unittest.TestCase):
    def test_one_to_one_matching_and_declared_tolerance(self):
        metrics = match_qrs_events([1.0, 1.02, 2.0, 3.0], [1.01, 2.03, 4.0], 50)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 2, 1))
        self.assertAlmostEqual(metrics["timing_mae_ms"], 20)
        self.assertAlmostEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["recall"], 2/3)
        self.assertEqual(len(set(p[1] for p in metrics["pairs"])), metrics["tp"])

    def test_no_predictions_preserves_reference_coverage(self):
        metrics = match_qrs_events([], [1, 2], 50)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 0, 2))
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)
        self.assertTrue(np.isnan(metrics["timing_mae_ms"]))

    def test_event_gates_and_artifact_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artifacts").mkdir()
            (root / "artifacts" / "p_r.json").write_text(json.dumps({"record_id": "p/r", "fs": 500,
                                                                    "anchors": [500, 1000, 1500]}))
            predictions = pd.DataFrame([{"record_id": "p/r", "patient_id": "p", "label_name": "MI", "fs": 500,
                                         "seconds_loaded": 4}])
            base = {"record_id": "p/r", "review_status": "confirmed", "reviewer": "Expert",
                    "independent_reference": True, "complete_event_reference": True,
                    "coverage_start_ms": 0, "coverage_end_ms": 4000}
            references = pd.DataFrame([{**base, "sample": 505}, {**base, "sample": 1005}])
            details, summary, audit = compare_qrs_references(predictions, references, root, 50)
            self.assertEqual(audit.accepted.sum(), 2)
            self.assertEqual((details.iloc[0].tp, details.iloc[0].fp, details.iloc[0].fn), (2, 1, 0))
            self.assertAlmostEqual(summary.iloc[0].timing_mae_ms, 10)
            with self.assertRaisesRegex(ValueError, "natural reference"):
                compare_qrs_references(predictions, pd.concat([references, references.iloc[[0]]]), root)

    def test_event_reference_requires_complete_confirmed_set_and_explicit_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artifacts").mkdir()
            (root / "artifacts" / "p_r.json").write_text(json.dumps({"record_id": "p/r", "fs": 1000,
                                                                    "anchors": [1000, 2000, 3000]}))
            predictions = pd.DataFrame([{"record_id": "p/r", "patient_id": "p", "label_name": "MI", "fs": 1000,
                                         "seconds_loaded": 4}])
            base = {"record_id": "p/r", "review_status": "confirmed", "reviewer": "Expert",
                    "independent_reference": True, "complete_event_reference": True,
                    "coverage_start_ms": 1500, "coverage_end_ms": 2500, "sample": 2000}
            details, _, audit = compare_qrs_references(predictions, pd.DataFrame([base]), root)
            self.assertTrue(audit.accepted.all())
            self.assertEqual(details.iloc[0].n_predicted, 1)
            self.assertEqual(details.iloc[0].tp, 1)
            for changes, reason in [({"complete_event_reference": False}, "complete_event_reference_not_explicitly_true"),
                                    ({"coverage_start_ms": np.nan}, "event_reference_requires_explicit_valid_coverage_interval"),
                                    ({"coverage_end_ms": 5000}, "event_reference_coverage_exceeds_loaded_recording")]:
                details, _, audit = compare_qrs_references(predictions, pd.DataFrame([{**base, **changes}]), root)
                self.assertTrue(details.empty)
                self.assertEqual(audit.iloc[0].reason, reason)
            mixed = pd.DataFrame([base, {**base, "sample": 2100, "review_status": "pending"}])
            details, _, audit = compare_qrs_references(predictions, mixed, root)
            self.assertTrue(details.empty)
            self.assertFalse(audit.accepted.any())
            self.assertEqual(audit.iloc[0].reason, "event_reference_set_contains_unconfirmed_or_invalid_rows")

    def test_invalid_tolerance_or_nonfinite_timestamps_rejected(self):
        for tolerance in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                match_qrs_events([1], [1], tolerance)
        with self.assertRaises(ValueError):
            match_qrs_events([np.nan], [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)

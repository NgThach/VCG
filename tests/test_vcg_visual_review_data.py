"""Saved V1 visual exports must preserve coordinates, missingness and cuts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import vcg_visual_review_data as export
from vcg_preprocessing import Config


class VisualReviewDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art = ROOT / "outputs/vcg_full_evaluation/artifacts"
        cls.frame = pd.concat([pd.read_csv(ROOT / f"outputs/vcg_12_features_train_valid_test/{s}_features.csv")
                               for s in ("train", "valid", "test")], ignore_index=True)
        cls.cfg = Config(**json.loads((ROOT / "outputs/vcg_12_features_train_valid_test/config.json").read_text())["config"])

    def row(self, rid="patient051/s0179lre"):
        return self.frame[self.frame.record_id.eq(rid)].iloc[0].to_dict()

    def record(self, rid="patient051/s0179lre"):
        return export.build_visual_record(self.row(rid), self.art, self.cfg)

    def test_sample_anchor_conversion_uses_each_sampling_frequency(self):
        self.assertEqual(export.sample_time_ms(220, 500, 200), 40.)
        self.assertEqual(export.sample_time_ms(190, 250, 200), -40.)
        self.assertEqual(export.sample_time_ms(200, 2000, 200), 0.)
        self.assertIsNone(export.sample_time_ms(None, 500, 200))
        with self.assertRaises(ValueError):
            export.sample_time_ms(220, 0, 200)

    def test_saved_v1_arrays_boundaries_and_exact_parity(self):
        payload, parity, issues = self.record()
        self.assertTrue(payload["available"])
        self.assertEqual(issues, [])
        self.assertTrue(all(item["matches_exactly"] for item in parity))
        with np.load(self.art / "patient051_s0179lre.npz") as saved:
            np.testing.assert_array_equal(payload["xyz"], saved["median_xyz"])
            np.testing.assert_array_equal(payload["time_ms"], saved["time_ms"])
            np.testing.assert_array_equal(payload["speed"], saved["speed"])
        self.assertEqual(payload["boundary_times_ms"]["t_on"], 246.)
        self.assertEqual(payload["boundary_times_ms"]["t_off"], 263.)
        self.assertEqual(payload["provenance"]["array_sources"]["speed"], "saved_v1_npz")

    def test_loop_cuts_are_inclusive_independent_and_exclude_st(self):
        payload, _, _ = self.record()
        xyz = np.asarray(payload["xyz"])
        a, b = payload["qrs_loop_indices"]
        c, d = payload["t_loop_indices"]
        self.assertLess(b, c)
        self.assertEqual(len(xyz[a:b+1]), b-a+1)
        self.assertEqual(len(xyz[c:d+1]), d-c+1)
        self.assertEqual(c, payload["boundaries"]["t_on"])
        self.assertEqual(d, payload["boundaries"]["t_off"])
        self.assertNotEqual(c, b+1)

    def test_missing_median_keeps_record_and_does_not_invent_times(self):
        payload, parity, _ = self.record("patient033/s0157lre")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["record_id"], "patient033/s0157lre")
        self.assertEqual(payload["xyz"], [])
        self.assertIsNone(payload["qrs_loop_indices"])
        self.assertIsNone(payload["t_loop_indices"])
        self.assertEqual(payload["boundary_times_ms"], {})
        self.assertIn("Không đủ 3 beat", payload["reason"])
        self.assertEqual(parity, [])

    def test_missing_t_peak_still_exports_real_xyz_and_qrs(self):
        payload, parity, _ = self.record("patient067/s0227lre")
        self.assertTrue(payload["available"])
        self.assertGreater(len(payload["xyz"]), 100)
        self.assertIsNotNone(payload["qrs_loop_indices"])
        self.assertIsNone(payload["t_loop_indices"])
        self.assertIsNone(payload["boundary_times_ms"]["t_peak"])
        self.assertEqual(payload["meta"]["segmentation_status"], "t_peak_missing")
        self.assertTrue(all(item["matches_exactly"] for item in parity))

    def test_recomputed_boundary_difference_is_reported_never_substituted(self):
        original = export.delineate_loops
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            result["t_on"] += 5
            return result
        with patch.object(export, "delineate_loops", changed):
            payload, parity, _ = self.record()
        self.assertEqual(payload["boundary_times_ms"]["t_on"], 246.)
        self.assertEqual(payload["boundaries"]["t_on"], 502)
        self.assertFalse(next(x for x in parity if x["field"] == "t_on")["matches_exactly"])
        self.assertEqual(payload["parity"]["n_differences"], 1)

    def test_csv_mismatch_remains_visible_and_is_not_fixed(self):
        row = self.row()
        row["t_on_ms"] = 251.
        payload, _, issues = export.build_visual_record(row, self.art, self.cfg)
        self.assertEqual(payload["boundary_times_ms"]["t_on"], 246.)
        self.assertTrue(any("prediction mismatch" in i["issue"] for i in issues))
        self.assertIn("integrity_warning", payload["meta"]["segmentation_status"])

    def test_resume_offline_js_and_existing_review_file_preserved(self):
        frame = pd.DataFrame([self.row(), self.row("patient033/s0157lre")])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            review = out / "manual_visual_review.csv"
            review.write_text("record_id,notes\npatient051/s0179lre,keep this\n")
            baseline = review.read_bytes()
            one = export.prepare_visual_review_data(frame, self.art, out, self.cfg)
            two = export.prepare_visual_review_data(frame, self.art, out, self.cfg)
            self.assertEqual(two["validation"]["cache_hits"], 2)
            self.assertEqual(two["validation"]["n_records"], 2)
            self.assertEqual(two["validation"]["n_missing_medians"], 1)
            self.assertTrue(two["validation"]["all_ids_preserved"])
            for row in two["manifest"].to_dict("records"):
                raw = (out / row["payload_path"]).read_bytes()
                self.assertTrue(raw.startswith(b"window.VCGVisualReview.receiveRecord("))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["payload_sha256"])
                self.assertNotIn(b"NaN", raw)
            self.assertEqual(review.read_bytes(), baseline)
            self.assertEqual(one["validation"]["n_parity_differences"], 0)

    def test_load_failure_is_a_manifest_row_and_js_placeholder(self):
        frame = pd.DataFrame([self.row()])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = export.prepare_visual_review_data(frame, out / "absent", out / "result", self.cfg)
            self.assertEqual(len(result["manifest"]), 1)
            item = result["manifest"].iloc[0]
            self.assertEqual(item.segmentation_status, "load_or_validation_error")
            self.assertFalse(item.has_median)
            self.assertTrue((out / "result" / item.payload_path).is_file())
            self.assertEqual(result["validation"]["n_integrity_issues"], 1)


if __name__ == "__main__":
    unittest.main()

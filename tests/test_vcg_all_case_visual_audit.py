"""Exhaustive diagnostic displays must retain missingness and count unique cases."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.figure
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vcg_all_case_visual_audit import (
    TOLERANCES, build_disagreement_tables, create_all_case_contact_sheets,
)


class AllCaseVisualAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.qc = self.out / "qc"
        self.qc.mkdir()

    @staticmethod
    def _frame():
        # Tangent minus threshold equals the listed gap; the first two records
        # share one patient so record and patient denominators differ.
        specifications = [
            ("patient001/a", "HC", "train", 10., 20., True),
            ("patient001/b", "HC", "train", 11., 21., True),
            ("patient002/c", "MI", "valid", 60., 100., True),
            ("patient003/d", "MI", "test", 61., 101., True),
            ("patient004/e", "MI", "test", np.nan, 0., True),
            ("patient005/f", "HC", "valid", np.nan, np.nan, False),
            ("patient006/g", "MI", "train", 0., np.nan, True),
            ("patient007/h", "HC", "valid", 10., 10., True),
        ]
        rows = []
        for rid, label, split, on_gap, off_gap, success in specifications:
            rows.append({
                "record_id": rid, "patient_id": rid.split("/")[0], "label_name": label,
                "split": split, "processing_success": success,
                "t_tangent_on_ms": 150. + on_gap if np.isfinite(on_gap) else 175. if success else np.nan,
                "t_threshold_on_ms": 150. if np.isfinite(on_gap) else np.nan,
                "t_tangent_off_ms": 300. + off_gap if np.isfinite(off_gap) else np.nan,
                "t_threshold_off_ms": 300. if np.isfinite(off_gap) else np.nan,
                "t_on_ms": 160. if success else np.nan, "t_off_ms": 350. if success else np.nan,
                "t_duration_ms": 190. if success else np.nan,
                "qc_flags": "st_level_requires_review" if rid.endswith("/c") else "",
                "feature_completeness": 12 if success else 0,
            })
        return pd.DataFrame(rows)

    def _write_qc(self, frame):
        pd.DataFrame({"record_id": frame.record_id, "n_significant_lobes": np.ones(len(frame))}).to_csv(
            self.qc / "t_boundary_audit.csv", index=False)
        pd.DataFrame({"record_id": frame.record_id,
                      "secondary_activity_count": np.zeros(len(frame)),
                      "screening_signals": [""] * len(frame)}).to_csv(
            self.qc / "qrs_boundary_audit.csv", index=False)

    def _build(self, frame=None):
        frame = self._frame() if frame is None else frame
        self._write_qc(frame)
        return build_disagreement_tables(frame, self.out, self.qc)

    @staticmethod
    def _summary(tables, boundary, tolerance, scope="all", label="all"):
        summary = tables["summary"]
        match = summary.loc[summary.scope.eq(scope) & summary.label_name.eq(label)
                            & summary.boundary.eq(boundary) & summary.tolerance_ms.eq(tolerance)]
        assert len(match) == 1
        return match.iloc[0]

    def test_strict_thresholds_and_unique_union_are_not_summed(self):
        tables = self._build()
        expected = {
            "t_on": [5, 3, 2, 2, 1, 0],
            "t_off": [5, 4, 3, 2, 2, 1],
            "any_boundary": [5, 4, 3, 2, 2, 1],
        }
        for boundary, counts in expected.items():
            for tolerance, count in zip(TOLERANCES, counts):
                with self.subTest(boundary=boundary, tolerance=tolerance):
                    self.assertEqual(self._summary(tables, boundary, tolerance).n_cases, count)
        records = tables["records"].set_index("record_id")
        # Exactly 10 is excluded, while 11 is included: comparisons are >, not >=.
        self.assertFalse(bool(records.loc["patient007/h", "any_gap_gt10"]))
        self.assertFalse(records.loc["patient001/a", "t_on_gap_ms"] > 10)
        self.assertTrue(bool(records.loc["patient001/b", "any_gap_gt10"]))
        union = self._summary(tables, "any_boundary", 10)
        self.assertEqual(union.n_cases, 4)
        self.assertEqual(union.n_patients, 3)
        self.assertLess(union.n_cases, self._summary(tables, "t_on", 10).n_cases
                        + self._summary(tables, "t_off", 10).n_cases)
        exported = pd.read_csv(self.out / "all_cases" / "cases_gap_gt10ms.csv")
        self.assertEqual(len(exported), 4)
        self.assertTrue(exported.record_id.is_unique)

    def test_missing_candidates_remain_nan_and_have_explicit_denominators(self):
        tables = self._build()
        records = tables["records"].set_index("record_id")
        self.assertTrue(pd.isna(records.loc["patient004/e", "t_on_gap_ms"]))
        self.assertEqual(records.loc["patient004/e", "t_on_pair_status"], "threshold_missing")
        self.assertEqual(records.loc["patient005/f", "method_pair_status"], "neither")
        self.assertTrue(pd.isna(records.loc["patient005/f", "max_method_gap_ms"]))
        self.assertFalse(bool(records.loc["patient005/f", "observed_equal_only_on_available_pairs"]))
        self.assertTrue(bool(records.loc["patient004/e", "observed_equal_only_on_available_pairs"]))
        self.assertTrue(bool(records.loc["patient006/g", "observed_equal_only_on_available_pairs"]))
        for boundary, denominator, missing in [("t_on", 6, 2), ("t_off", 6, 2), ("any_boundary", 7, 1)]:
            row = self._summary(tables, boundary, 0)
            self.assertEqual(row.n_total, 8)
            self.assertEqual(row.n_denominator, denominator)
            self.assertEqual(row.n_missing_pair, missing)
            self.assertAlmostEqual(row.pct_of_available_pairs, 100 * 5 / denominator)
        self.assertTrue(records.reference_status.eq("GROUND_TRUTH_UNAVAILABLE").all())
        config = json.loads((self.out / "all_cases" / "visual_audit_config.json").read_text())
        self.assertTrue(config["unavailable_pairs_not_zero"])
        self.assertIn("strictly greater", config["threshold_comparison"])

    def test_subgroups_have_their_own_available_pair_denominators(self):
        tables = self._build()
        expectations = [
            ("all", "HC", "t_on", 4, 3, 1, 1),
            ("all", "MI", "t_on", 4, 3, 1, 2),
            ("train_valid", "MI", "t_on", 2, 2, 0, 1),
            ("test", "MI", "t_on", 2, 1, 1, 1),
            ("test", "MI", "t_off", 2, 2, 0, 1),
            ("valid", "HC", "any_boundary", 2, 1, 1, 0),
        ]
        for scope, label, boundary, total, denominator, missing, changed in expectations:
            with self.subTest(scope=scope, label=label, boundary=boundary):
                row = self._summary(tables, boundary, 10, scope, label)
                self.assertEqual((row.n_total, row.n_denominator, row.n_missing_pair, row.n_cases),
                                 (total, denominator, missing, changed))
                self.assertAlmostEqual(row.pct_of_available_pairs, 100 * changed / denominator)
        absent_group = self._summary(tables, "t_on", 10, "test", "HC")
        self.assertEqual(absent_group.n_total, 0)
        self.assertEqual(absent_group.n_denominator, 0)
        self.assertTrue(pd.isna(absent_group.pct_of_available_pairs))

    def test_duplicate_record_ids_are_rejected(self):
        frame = self._frame()
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate record_id"):
            build_disagreement_tables(duplicate, self.out, self.qc)

    def test_contact_sheets_render_every_record_including_missing_median(self):
        artifacts = self.out / "artifacts"
        artifacts.mkdir()
        time = np.arange(-100., 501.)
        xyz = np.column_stack([np.exp(-(time / 30.) ** 2) * v for v in [1., .4, .2]])
        xyz += np.exp(-((time[:, None] - 230.) / 60.) ** 2) * np.asarray([.2, .1, .04])
        rows = []
        missing_id = "patient105/synthetic"
        for j in range(13):
            rid = f"patient{100+j:03d}/synthetic"
            safe = rid.replace("/", "_")
            artifact = {"record_id": rid, "boundaries": {"qrs_off": 145}}
            (artifacts / (safe + ".json")).write_text(json.dumps(artifact))
            if rid == missing_id:
                np.savez_compressed(artifacts / (safe + ".npz"), anchors=np.array([10, 100]))
            else:
                np.savez_compressed(artifacts / (safe + ".npz"), median_xyz=xyz,
                                    time_ms=time, t_magnitude=np.linalg.norm(xyz, axis=1))
            rows.append({"record_id": rid, "display_rank": j+1, "label_name": "HC" if j % 2 else "MI",
                         "split": ("train", "valid", "test")[j % 3], "t_on_ms": 160., "t_off_ms": 340.,
                         "t_tangent_on_ms": 160., "t_tangent_off_ms": 340.,
                         "t_threshold_on_ms": 150., "t_threshold_off_ms": 360.,
                         "t_on_gap_ms": 10. if rid != missing_id else np.nan,
                         "t_off_gap_ms": 20. if rid != missing_id else np.nan})
        records = pd.DataFrame(rows)
        rendered_titles, missing_titles = [], []
        original_save = matplotlib.figure.Figure.savefig
        def save_and_capture(figure, *args, **kwargs):
            for axis in figure.axes:
                if axis.get_title().startswith("#"):
                    rendered_titles.append(axis.get_title())
                    if any("KHÔNG CÓ MEDIAN V1" in text.get_text() for text in axis.texts):
                        missing_titles.append(axis.get_title())
            return original_save(figure, *args, **kwargs)
        with patch.object(matplotlib.figure.Figure, "savefig", new=save_and_capture):
            manifest = create_all_case_contact_sheets(records, artifacts, self.out, page_size=12)
        self.assertEqual(len(manifest), 13)
        self.assertTrue(manifest.record_id.is_unique)
        self.assertEqual(set(manifest.record_id), set(records.record_id))
        self.assertEqual(manifest.groupby("page").size().to_dict(), {1: 12, 2: 1})
        self.assertEqual(len(rendered_titles), 13)
        for rid in records.record_id:
            self.assertEqual(sum(rid in title for title in rendered_titles), 1)
        self.assertEqual(len(missing_titles), 1)
        self.assertIn(missing_id, missing_titles[0])
        for relative in manifest.plot_path.unique():
            image = self.out / relative
            self.assertTrue(image.is_file())
            self.assertGreater(image.stat().st_size, 10000)
        self.assertTrue((self.out / "all_cases" / "contact_sheet_manifest.csv").is_file())

    def test_saved_real_cohort_counts_when_available(self):
        source = ROOT / "outputs" / "vcg_12_features_train_valid_test"
        qc = ROOT / "outputs" / "vcg_qc_audit_v1"
        paths = [source / f"{split}_features.csv" for split in ["train", "valid", "test"]]
        if not all(path.is_file() for path in paths) or not (qc / "t_boundary_audit.csv").is_file():
            self.skipTest("Saved PTB cohort not installed; synthetic count tests remain mandatory")
        frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        tables = build_disagreement_tables(frame, self.out, qc)
        self.assertEqual(len(tables["records"]), 448)
        expected = [("t_on", 60, 30, 14), ("t_off", 60, 438, 32),
                    ("t_off", 100, 438, 16), ("any_boundary", 60, 438, 46),
                    ("any_boundary", 100, 438, 20)]
        for boundary, tolerance, denominator, count in expected:
            with self.subTest(boundary=boundary, tolerance=tolerance):
                row = self._summary(tables, boundary, tolerance)
                self.assertEqual(row.n_denominator, denominator)
                self.assertEqual(row.n_cases, count)


if __name__ == "__main__":
    unittest.main()

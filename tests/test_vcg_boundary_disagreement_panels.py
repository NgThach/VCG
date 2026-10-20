"""All-case plots preserve missing candidates and compare temporal cuts only."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.vcg_boundary_disagreement_panels import (
    candidate_positions, interval_segments, render_all_case_panels,
    validate_case_panels, validate_saved_median, v1_loop_points,
)


class AllCasePanelTests(unittest.TestCase):
    def test_candidates_keep_zero_and_missing_distinct(self):
        time = np.arange(-10., 11.)
        positions = candidate_positions({"t_on": 0, "t_off": 20, "t_threshold_on": None,
                                          "t_threshold_off": 20, "t_tangent_off": 20}, time)
        self.assertEqual(positions["t_on"], {"index": 0, "ms": -10.})
        self.assertIsNone(positions["t_threshold_on"]["ms"])
        self.assertEqual(positions["t_threshold_off"]["ms"] - positions["t_tangent_off"]["ms"], 0)
        for invalid in (-1, 21, .5, np.nan):
            self.assertIsNone(candidate_positions({"t_on": invalid}, time)["t_on"]["index"])

    def test_identical_intervals_have_zero_disagreement(self):
        segments = interval_segments(12, (2, 9), (2, 9))
        self.assertEqual(segments["common"].sum(), 7)
        self.assertEqual(segments["v1_only"].sum(), 0)
        self.assertEqual(segments["alternative_only"].sum(), 0)

    def test_segment_membership_preserves_endpoints(self):
        segments = interval_segments(12, (2, 8), (4, 10))
        self.assertEqual(segments["common"].sum(), 4)
        self.assertEqual(segments["v1_only"].sum(), 2)
        self.assertEqual(segments["alternative_only"].sum(), 2)
        with self.assertRaises(ValueError):
            interval_segments(12, (8, 2), (4, 10))

    @staticmethod
    def artifact(root, rid, median=True):
        time = np.arange(-100., 401.)
        xyz = np.column_stack((np.exp(-(time / 25) ** 2),
                               .4 * np.exp(-((time - 5) / 25) ** 2),
                               .2 * np.exp(-((time + 7) / 20) ** 2)))
        xyz += np.asarray((.2, .12, .05))[None, :] * np.exp(-((time[:, None] - 230) / 55) ** 2)
        speed = np.linalg.norm(np.gradient(xyz, .001, axis=0), axis=1)
        mag = np.linalg.norm(xyz, axis=1)
        artifact = {"record_id": rid, "fs": 1000, "boundaries": {
            "qrs_on": 75, "qrs_off": 140, "q_threshold": 1.,
            "t_on": 270, "t_peak": 330, "t_off": 400, "t_rise_index": 300,
            "t_fall_index": 360, "t_search": [200, 470], "st_level": .01,
            "t_end_baseline": .001, "t_method": "synthetic unit test",
            "t_tangent_on": 270, "t_tangent_off": 400,
            "t_threshold_on": None, "t_threshold_off": 420}}
        safe = rid.replace("/", "_")
        (root / (safe + ".json")).write_text(json.dumps(artifact))
        arrays = dict(median_xyz=xyz, time_ms=time, speed=speed, smooth_speed=speed,
                      magnitude=mag, t_magnitude=mag, t_slope=np.gradient(mag, .001)) if median else {}
        np.savez_compressed(root / (safe + ".npz"), **arrays)

    def test_serial_renders_all_records_and_caches_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            artifacts, out = temp / "artifacts", temp / "all_cases"
            artifacts.mkdir()
            metadata = []
            for index, median in enumerate((True, False)):
                rid = f"synthetic{index}/test"
                self.artifact(artifacts, rid, median)
                metadata.append(dict(record_id=rid, patient_id=f"synthetic{index}",
                                     label_name="HC", split="train"))
            frame = pd.DataFrame(metadata)
            result = render_all_case_panels(frame, artifacts, out, workers=1)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.has_median.sum(), 1)
            self.assertEqual(set(result.render_status), {"rendered", "missing_median_notice"})
            second = render_all_case_panels(frame, artifacts, out, workers=1)
            self.assertTrue(second.render_status.eq("cached").all())
            self.assertEqual(result.sha256.tolist(), second.sha256.tolist())
            self.assertEqual(validate_case_panels(second, out, frame.record_id.tolist())["n_records"], 2)
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_case_panels(second, out, [frame.record_id.iloc[0]])
            (out / second.plot_path.iloc[0]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "modified"):
                validate_case_panels(second, out)

    def test_duplicate_ids_rejected_before_rendering(self):
        frame = pd.DataFrame([dict(record_id="a/a", patient_id="a", label_name="HC", split="train")] * 2)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                render_all_case_panels(frame, Path(temp), Path(temp) / "out", workers=1)

    def test_v1_loops_use_independent_inclusive_sample_cuts(self):
        xyz = np.arange(60).reshape(20, 3)
        boundaries = dict(qrs_on=2, qrs_off=5, t_on=12, t_off=17,
                          t_threshold_on=8, t_threshold_off=19)
        np.testing.assert_array_equal(v1_loop_points(xyz, boundaries, "qrs"), xyz[2:6])
        np.testing.assert_array_equal(v1_loop_points(xyz, boundaries, "t"), xyz[12:18])
        self.assertEqual(len(v1_loop_points(xyz, boundaries, "qrs")), 4)
        boundaries["t_on"] = None
        self.assertIsNone(v1_loop_points(xyz, boundaries, "t"))
        np.testing.assert_array_equal(v1_loop_points(xyz, boundaries, "qrs"), xyz[2:6])

    def test_saved_timebase_uses_record_fs_and_anchor(self):
        xyz = np.zeros((101, 3))
        t = (np.arange(101) - 35) * 1000 / 500
        artifact = dict(fs=500, median_anchor_sample=35)
        saved_t, _, fs = validate_saved_median(artifact, dict(time_ms=t, median_xyz=xyz))
        self.assertEqual(fs, 500)
        self.assertEqual(saved_t[35], 0)
        self.assertEqual(saved_t[40], 10)
        with self.assertRaisesRegex(ValueError, "anchor"):
            validate_saved_median(dict(fs=500, median_anchor_sample=36), dict(time_ms=t, median_xyz=xyz))
        with self.assertRaisesRegex(ValueError, "sampling|spacing"):
            validate_saved_median(dict(fs=1000), dict(time_ms=t, median_xyz=xyz))

    def test_failed_load_retains_record_with_reason_and_placeholder(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            artifacts = temp / "artifacts"
            artifacts.mkdir()
            frame = pd.DataFrame([dict(record_id="missing/source", patient_id="missing",
                                       label_name="MI", split="valid")])
            result = render_all_case_panels(frame, artifacts, temp / "out", workers=1)
            self.assertEqual(result.record_id.tolist(), ["missing/source"])
            self.assertEqual(result.render_status.iloc[0], "render_error")
            self.assertIn("FileNotFoundError", result.render_error.iloc[0])
            self.assertEqual(validate_case_panels(result, temp / "out")["n_records"], 1)


if __name__ == "__main__":
    unittest.main()

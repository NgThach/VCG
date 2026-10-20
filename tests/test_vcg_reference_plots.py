"""Reference figures must never turn missing annotation into zero error."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.vcg_reference_plots import (
    _eligible, _index_interval, _cut_runs, create_reference_plots,
    create_diagnostic_case_gallery,
)


class ReferencePlotTests(unittest.TestCase):
    def test_finite_predictions_without_acceptance_are_not_accuracy_pairs(self):
        frame = pd.DataFrame([{"predicted_ms": 12., "reference_ms": 12.}])
        self.assertTrue(_eligible(frame, ("predicted_ms", "reference_ms")).empty)
        frame["evaluation_eligible"] = False
        self.assertTrue(_eligible(frame, ("predicted_ms", "reference_ms")).empty)
        frame["evaluation_eligible"] = True
        self.assertEqual(len(_eligible(frame, ("predicted_ms", "reference_ms"))), 1)

    def test_unavailable_references_create_status_and_no_accuracy_images(self):
        tables = {
            "boundary_error_per_record": pd.DataFrame([{
                "evaluation_eligible": False, "predicted_ms": 12., "reference_ms": np.nan,
                "error_ms": np.nan, "absolute_error_ms": np.nan}]),
            "loop_overlap_per_record": pd.DataFrame(),
            "feature_reference_comparison": pd.DataFrame(),
        }
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = create_reference_plots(tables, out / "unused_artifacts", out)
            self.assertEqual(result["n_evaluated_pairs"], 0)
            self.assertEqual(len(result["manifest"]), 10)
            self.assertTrue(result["manifest"].status.eq("unavailable").all())
            self.assertFalse(list(out.rglob("*.png")))
            self.assertTrue((out / "plots_status.csv").is_file())
            self.assertIsNone(result["gallery"])
            self.assertIsNone(result["all_gallery"])
            self.assertIsNone(result["boundary_gallery"])
            self.assertEqual(result["n_visualized_loop_pairs"], 0)
            self.assertEqual(result["n_visualized_boundary_pairs"], 0)

    def test_overlay_time_contract_rejects_unsnapped_reference(self):
        time = np.arange(-5, 6, dtype=float)
        row = pd.Series({"reference_start_ms": -2., "reference_end_ms": 3.,
                         "reference_start_index": 3, "reference_end_index": 8})
        self.assertEqual(_index_interval(row, "reference", time), (3, 8))
        row["reference_start_ms"] = -2.2
        with self.assertRaisesRegex(ValueError, "represented sample times"):
            _index_interval(row, "reference", time)
        # Runs are point endpoints for continuously classified adjacent segments.
        self.assertEqual(_cut_runs(np.array([False, True, True, False, True])), [(1, 3), (4, 5)])

    @staticmethod
    def _artifact(root, record_id, shift=0):
        time = np.arange(-100., 401.)
        xyz = np.column_stack((np.exp(-(time / 25) ** 2),
                               .4 * np.exp(-((time - 5) / 25) ** 2),
                               .2 * np.exp(-((time + 7) / 20) ** 2)))
        xyz += np.asarray((.2, .12, .05))[None, :] * np.exp(-((time[:, None] - 230 - shift) / 55) ** 2)
        speed = np.linalg.norm(np.gradient(xyz, .001, axis=0), axis=1)
        mag = np.linalg.norm(xyz, axis=1)
        artifact = {"record_id": record_id, "fs": 1000, "median_anchor_sample": 100,
                    "pq_indices": [20, 60], "boundaries": {
                        "qrs_on": 75, "qrs_off": 140, "qrs_speed_peak": 85, "q_threshold": 1.,
                        "t_on": 270, "t_peak": 330, "t_off": 400,
                        "t_rise_index": 300, "t_fall_index": 360, "t_search": [200, 470],
                        "st_level": .01, "t_end_baseline": .001, "t_method": "synthetic_test_only"}}
        safe = record_id.replace("/", "_")
        (root / (safe + ".json")).write_text(json.dumps(artifact))
        np.savez_compressed(root / (safe + ".npz"), median_xyz=xyz, time_ms=time,
                            speed=speed, smooth_speed=speed, magnitude=mag,
                            t_magnitude=mag, t_slope=np.gradient(mag, .001))

    def test_verified_synthetic_pairs_generate_figures_and_same_signal_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            artifacts = out / "artifacts"
            artifacts.mkdir()
            boundaries, loops, features = [], [], []
            for j, label in enumerate(("HC", "MI")):
                rid = f"patient{j:03d}/synthetic"
                self._artifact(artifacts, rid, j)
                common = {"record_id": rid, "patient_id": f"patient{j:03d}", "label_name": label,
                          "split": "train", "evaluation_eligible": True, "reference_source": "synthetic unit test", "reviewer": "test"}
                for name, prediction, reference in (("qrs_on", -25., -30.), ("qrs_off", 40., 45.),
                                                     ("t_on", 170., 160.), ("t_off", 300., 310.)):
                    boundaries.append({**common, "boundary": name, "predicted_ms": prediction,
                                       "reference_ms": reference, "error_ms": prediction-reference,
                                       "absolute_error_ms": abs(prediction-reference)})
                for name, pred, ref in (("QRS", (-25., 40.), (-30., 45.)), ("T", (170., 300.), (160., 310.))):
                    missed = pred[0] - ref[0] + ref[1] - pred[1]
                    loops.append({**common, "loop": name, "predicted_start_ms": pred[0], "predicted_end_ms": pred[1],
                                  "reference_start_ms": ref[0], "reference_end_ms": ref[1],
                                  "predicted_duration_ms": pred[1]-pred[0], "reference_duration_ms": ref[1]-ref[0],
                                  "duration_error_ms": -missed, "temporal_iou": (pred[1]-pred[0]) / (ref[1]-ref[0]),
                                  "missed_ms": missed, "excess_ms": 0.})
                features.append({**common, "feature": "QRS_Arc", "predicted": 2., "reference": 2.1,
                                 "absolute_feature_error": .1, "relative_feature_error_pct": .1/2.1*100})
            result = create_reference_plots({"boundary_error_per_record": pd.DataFrame(boundaries),
                                             "loop_overlap_per_record": pd.DataFrame(loops),
                                             "feature_reference_comparison": pd.DataFrame(features)}, artifacts, out)
            self.assertEqual(result["n_evaluated_pairs"], 8)
            self.assertTrue(result["manifest"].status.eq("created").all())
            self.assertTrue(result["gallery"].is_file())
            self.assertTrue(result["all_gallery"].is_file())
            self.assertTrue(result["boundary_gallery"].is_file())
            self.assertEqual(result["n_visualized_loop_pairs"], 4)
            self.assertEqual(result["n_visualized_boundary_pairs"], 8)
            self.assertEqual(result["n_visualized_boundary_records"], 2)
            self.assertEqual(len(list((out / "plots" / "verified_reference" / "loop_overlays").glob("*.png"))), 4)
            # Identical accepted input does not create an archive on rerun.
            repeat = create_reference_plots({"boundary_error_per_record": pd.DataFrame(boundaries),
                                             "loop_overlap_per_record": pd.DataFrame(loops),
                                             "feature_reference_comparison": pd.DataFrame(features)}, artifacts, out)
            self.assertEqual(repeat['input_state']['archived_previous_run'], '')
            # A withdrawn/rejected reference cannot leave its accuracy images
            # or highest/lowest gallery in the current output namespace.
            withdrawn = create_reference_plots({}, artifacts, out)
            self.assertTrue(withdrawn['manifest'].status.eq('unavailable').all())
            self.assertFalse(list((out/'plots'/'verified_reference').rglob('*.png')))
            self.assertFalse((out/'reference_loop_gallery.html').exists())
            self.assertFalse((out/'reference_loop_gallery_manifest.csv').exists())
            self.assertFalse((out/'all_reference_loop_gallery.html').exists())
            self.assertFalse((out/'all_reference_loop_gallery_manifest.csv').exists())
            self.assertFalse((out/'all_reference_boundary_gallery.html').exists())
            self.assertFalse((out/'all_reference_boundary_gallery_manifest.csv').exists())
            history = out/withdrawn['input_state']['archived_previous_run']
            self.assertTrue((history/'reference_loop_gallery.html').is_file())
            self.assertTrue((history/'all_reference_loop_gallery.html').is_file())
            self.assertTrue((history/'all_reference_boundary_gallery.html').is_file())
            self.assertTrue(list((history/'verified_reference').rglob('*.png')))

    def test_all_loop_pairs_are_visualized_beyond_highest_lowest_six(self):
        """Middle-ranked cases must be in the complete gallery, once each."""
        rows = []
        for j in range(8):
            common = {"record_id": f"patient{j:03d}/synthetic", "patient_id": f"patient{j:03d}",
                      "label_name": "HC" if j % 2 else "MI", "split": "valid" if j % 2 else "train",
                      "evaluation_eligible": True, "reference_source": "synthetic unit test", "reviewer": "test"}
            for loop, start, end in (("QRS", -25., 40.), ("T", 170., 300.)):
                shift = float(j)
                rows.append({**common, "loop": loop, "predicted_start_ms": start,
                             "predicted_end_ms": end, "reference_start_ms": start - shift,
                             "reference_end_ms": end + shift, "predicted_duration_ms": end-start,
                             "reference_duration_ms": end-start+2*shift, "duration_error_ms": -2*shift,
                             "temporal_iou": (end-start)/(end-start+2*shift), "missed_ms": 2*shift,
                             "excess_ms": 0.})
        rendered = []
        def overlay(row, artifact_root, path):
            rendered.append((row.record_id, row.loop))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic coverage placeholder; actual drawing tested separately")
        with tempfile.TemporaryDirectory() as temp, patch("scripts.vcg_reference_plots._overlay_loop", side_effect=overlay):
            out = Path(temp)
            result = create_reference_plots({"loop_overlap_per_record": pd.DataFrame(rows)}, out / "not_read", out)
            expected = {(row["record_id"], row["loop"]) for row in rows}
            self.assertEqual(set(rendered), expected)
            self.assertEqual(len(rendered), len(expected))
            manifest = pd.read_csv(out / "all_reference_loop_gallery_manifest.csv")
            self.assertEqual(set(zip(manifest.record_id, manifest.loop)), expected)
            self.assertEqual(len(manifest), 16)
            self.assertFalse(manifest.duplicated(["record_id", "loop"]).any())
            self.assertEqual(result["n_visualized_loop_pairs"], 16)
            self.assertEqual(int(manifest.n_shifted_pairs.sum()), 14)
            page = result["all_gallery"].read_text()
            self.assertEqual(page.count('<article '), 16)
            self.assertIn("16 eligible recording/loop pairs", page)
            self.assertIn('id="loop"', page)
            self.assertIn('id="error"', page)
            self.assertIn('id="iou"', page)
            self.assertIn('id="sort"', page)
            self.assertIn('data-label="HC"', page)
            self.assertIn('data-split="valid"', page)

    def test_single_accepted_boundary_is_visualized_without_inventing_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            artifacts = out / "artifacts"
            artifacts.mkdir()
            rid = "patient000/partial"
            self._artifact(artifacts, rid)
            pair = {"record_id": rid, "patient_id": "patient000", "label_name": "MI", "split": "valid",
                    "evaluation_eligible": True, "reference_source": "synthetic isolated QRS-on", "reviewer": "test",
                    "boundary": "qrs_on", "predicted_ms": -25., "reference_ms": -35.,
                    "error_ms": 10., "absolute_error_ms": 10.}
            result = create_reference_plots({"boundary_error_per_record": pd.DataFrame([pair])}, artifacts, out)
            self.assertEqual(result["n_visualized_boundary_pairs"], 1)
            self.assertEqual(result["n_visualized_boundary_records"], 1)
            self.assertIsNone(result["all_gallery"])
            self.assertEqual(result["n_visualized_loop_pairs"], 0)
            manifest = pd.read_csv(out / "all_reference_boundary_gallery_manifest.csv")
            self.assertEqual(list(manifest.boundary), ["qrs_on"])
            self.assertTrue((out / manifest.plot_path.iloc[0]).is_file())
            self.assertEqual(result["boundary_gallery"].read_text().count('<article '), 1)
            self.assertIn("1 eligible boundary pairs", result["boundary_gallery"].read_text())
            status = result["manifest"].set_index("plot_family")
            self.assertEqual(status.loc["all_reference_boundary_gallery", "status"], "created")
            self.assertEqual(status.loc["all_reference_loop_gallery", "status"], "unavailable")

    def test_diagnostic_gallery_does_not_create_reference_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            artifacts = out / "artifacts"
            artifacts.mkdir()
            rid = "patient000/synthetic"
            self._artifact(artifacts, rid)
            frame = pd.DataFrame([{"record_id": rid, "patient_id": "patient000", "label_name": "HC", "split": "valid"}])
            priority = pd.DataFrame([{"record_id": rid, "priority_score": 3, "priority_reasons": "method_gap;no_flags"}])
            result = create_diagnostic_case_gallery(frame, priority, artifacts, out, max_cases=1)
            self.assertEqual(result["n_cases"], 1)
            self.assertTrue(result["manifest"].evaluation_mode.eq("diagnostic_only").all())
            self.assertIn("not independent ground truth", result["manifest"].interpretation.iloc[0])
            self.assertTrue(result["gallery"].is_file())
            self.assertFalse((out / "reference_annotations.csv").exists())
            # CSV round trips represent empty priority reasons as NaN.
            priority["priority_reasons"] = np.nan
            second = create_diagnostic_case_gallery(frame, priority, artifacts, out, max_cases=1)
            self.assertEqual(second["n_new_figures"], 0)


if __name__ == "__main__":
    unittest.main()

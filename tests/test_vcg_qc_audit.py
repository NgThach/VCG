"""Notebook 07 audit contracts: provenance, counts, triage, and reference gates.

Artifact consistency is deliberately not called delineation accuracy. All files
created by these tests are under TemporaryDirectory, never production outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import vcg_qc_audit as audit


@unittest.skipUnless((ROOT / "outputs/vcg_12_features_train_valid_test/config.json").is_file(),
                     "Requires saved Notebook 06 PTB cohort")
class TestFrozenCohortAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = TemporaryDirectory(prefix="vcg07-tests-")
        cls.output = Path(cls.workspace.name)
        cls.frame, cls.manifest, cls.config, cls.input_dir, cls.evaluation_dir = audit.load_inputs()
        cls.original = cls.frame.copy(deep=True)
        cls.artifacts = cls.evaluation_dir / "artifacts"
        cls.qc = audit.qc_tables(cls.frame, cls.output)
        with patch.object(audit, "tqdm", side_effect=lambda rows, **kwargs: rows):
            cls.qrs, cls.twave, cls.joined = audit.boundary_audits(cls.frame, cls.artifacts, cls.output)
        cls.template = audit.create_review_template(cls.joined, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def copy_input(self, target):
        target = Path(target)
        for name in ["train_features.csv", "valid_features.csv", "test_features.csv",
                     "split_manifest.csv", "config.json"]:
            shutil.copy2(self.input_dir / name, target / name)

    def test_exact_population_and_original_metadata_are_preserved(self):
        self.assertEqual(len(self.frame), 448)
        self.assertTrue(self.frame.record_id.is_unique)
        self.assertEqual(self.frame.patient_id.nunique(), 200)
        self.assertEqual(self.frame.label_name.value_counts().to_dict(), {"MI": 368, "HC": 80})
        self.assertEqual(set(self.frame.split), {"train", "valid", "test"})
        self.assertTrue(self.frame.groupby("patient_id").split.nunique().eq(1).all())
        source = pd.concat([pd.read_csv(self.input_dir / f"{split}_features.csv")
                            for split in ["train", "valid", "test"]]).sort_values("record_id").reset_index(drop=True)
        keys = ["record_id", "patient_id", "label", "label_name", "diagnosis", "split"]
        pd.testing.assert_frame_equal(self.frame[keys], source[keys])
        pd.testing.assert_frame_equal(self.frame, self.original)

    def test_rejects_duplicate_population_patient_leakage_and_label_tampering(self):
        for failure in ["duplicate", "leakage", "label", "split", "time_reference"]:
            with self.subTest(failure=failure), TemporaryDirectory(prefix="vcg07-input-") as directory:
                self.copy_input(directory)
                path = Path(directory) / "train_features.csv"
                frame = pd.read_csv(path)
                if failure == "duplicate":
                    frame.loc[1, "record_id"] = frame.loc[0, "record_id"]
                    message = "448 unique"
                elif failure == "leakage":
                    other = self.frame[self.frame.split.eq("valid")].iloc[0]
                    frame.loc[0, "patient_id"] = other.patient_id
                    message = "Patient leakage"
                elif failure == "label":
                    frame.loc[0, "label"] = 1 - frame.loc[0, "label"]
                    message = "label mapping"
                elif failure == "split":
                    frame.loc[0, "split"] = "test"
                    message = "Wrong split"
                else:
                    frame.loc[0, "boundary_time_reference"] = "raw_recording_ms"
                    message = "time reference"
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    audit.load_inputs(directory, self.evaluation_dir)

    def test_qc_denominators_overlap_and_zero_flags_are_counted_correctly(self):
        totals = self.qc["totals"]
        self.assertEqual(totals["no_flags"], 65)
        self.assertEqual(totals["needs_review"], 378)
        self.assertEqual(totals["processing_failed"], 5)
        self.assertEqual(totals["full12"], 441)
        self.assertEqual(totals["missing_features"], 7)
        self.assertEqual(totals["no_flags"] + totals["needs_review"] + totals["processing_failed"], 448)
        matrix = self.qc["matrix"]
        self.assertGreater(int(matrix.sum().sum()), 448 - totals["no_flags"])
        self.assertTrue(np.array_equal(matrix.sum(axis=1).to_numpy(), self.frame.n_flags.to_numpy()))
        for row in self.qc["frequency"].itertuples():
            self.assertEqual(row.HC_denominator, 80)
            self.assertEqual(row.MI_denominator, 368)
            self.assertAlmostEqual(row.HC_pct, 100 * row.HC_count / 80)
            self.assertAlmostEqual(row.MI_pct, 100 * row.MI_count / 368)
            self.assertEqual(row.total, row.HC_count + row.MI_count)
            self.assertEqual(int(self.qc["cooccurrence"].loc[row.qc_flag, row.qc_flag]), row.total)
        co = self.qc["cooccurrence"]
        self.assertTrue(np.array_equal(co.to_numpy(), co.to_numpy().T))
        flags = ["st_level_requires_review", "t_end_methods_disagree"]
        overlap = (matrix[flags].sum(axis=1) == 2).sum()
        self.assertEqual(co.loc[flags[0], flags[1]], overlap)

    def test_st_rule_matches_saved_flags_for_all_resolved_peaks_without_calling_it_accuracy(self):
        resolved = self.twave[self.twave.st_flag_recomputed.notna()]
        self.assertEqual(len(resolved), 442)
        self.assertTrue(resolved.st_flag_recomputed.eq(resolved.st_flag_present).all())
        self.assertTrue(resolved.st_flag_recomputed.eq(resolved.st_trigger_margin_mV.gt(0)).all())
        self.assertEqual(int(resolved.st_flag_recomputed.sum()), 311)
        self.assertTrue(self.twave.evidence_level.eq("unverified").all())
        qrs = self.qrs[self.qrs.threshold_recomputed.notna()]
        np.testing.assert_allclose(qrs.threshold_recomputed, qrs.qrs_threshold_mV_s, rtol=0, atol=1e-10)

    def test_shifted_artifact_timebase_cannot_silently_match_source_boundaries(self):
        row = self.frame[self.frame.processing_success & self.frame.extraction_success].iloc[0].to_dict()
        safe = row["record_id"].replace("/", "_")
        with TemporaryDirectory(prefix="vcg07-shifted-") as directory:
            directory = Path(directory)
            shutil.copy2(self.artifacts / (safe + ".json"), directory / (safe + ".json"))
            with np.load(self.artifacts / (safe + ".npz")) as saved:
                arrays = {key: saved[key] for key in saved.files}
            arrays["time_ms"] = arrays["time_ms"] + 13
            np.savez_compressed(directory / (safe + ".npz"), **arrays)
            with self.assertRaisesRegex(ValueError, "Artifact boundary mismatch"):
                audit.audit_record(row, directory)

    def test_review_template_is_train_valid_stratified_and_contains_no_fabricated_references(self):
        template = self.template
        self.assertTrue(template.record_id.is_unique)
        self.assertEqual(set(template.split), {"train", "valid"})
        self.assertFalse(template.record_id.isin(self.frame.loc[self.frame.split.eq("test"), "record_id"]).any())
        self.assertTrue(template.review_status.eq("unverified").all())
        self.assertFalse(template.independent_reference.any())
        self.assertTrue(template.reviewer.fillna("").eq("").all())
        self.assertTrue(template.review_notes.fillna("").eq("").all())
        self.assertTrue(template.proposed_group_A_B_C_D.fillna("").eq("").all())
        self.assertTrue(template.reference_anchor_sample.isna().all())
        for label in ["HC", "MI"]:
            group = template[template.label_name.eq(label)]
            self.assertGreaterEqual(len(group), 30)
            flagged = group.qc_flags.fillna("").ne("")
            self.assertTrue(flagged.any())
            self.assertTrue((~flagged).any())
        source = self.frame.set_index("record_id").loc[template.record_id]
        for key in ["patient_id", "label", "label_name", "split"]:
            self.assertTrue(np.array_equal(template[key].to_numpy(), source[key].to_numpy()))
        for boundary in ["qrs_on", "qrs_off", "t_on", "t_off"]:
            self.assertTrue(template[boundary + "_review_ms"].isna().all())
            np.testing.assert_allclose(template[boundary + "_pred_ms"], source[boundary + "_ms"], equal_nan=True)
        # Every incomplete train/valid record and every observed development flag
        # must be represented, without using test to select development examples.
        development = self.frame[self.frame.split.isin(["train", "valid"])]
        self.assertTrue(set(development.loc[~development.extraction_success, "record_id"]) <= set(template.record_id))
        for label in ["HC", "MI"]:
            all_flags = {f for text in development.loc[development.label_name.eq(label), "qc_flags"] for f in audit.split_flags(text)}
            selected_flags = {f for text in template.loc[template.label_name.eq(label), "qc_flags"] for f in audit.split_flags(text)}
            self.assertTrue(all_flags <= selected_flags)

    def test_no_reference_means_empty_comparison_and_unavailable_qc_error_rates(self):
        comparison, summary, accepted = audit.review_references(self.frame, self.output, self.evaluation_dir)
        self.assertTrue(comparison.empty)
        self.assertTrue(summary.empty)
        self.assertFalse(accepted.accepted.any())
        rates = pd.read_csv(self.output / "qc_reference_rates.csv")
        self.assertTrue(rates[["false_alarm_rate", "missed_error_rate"]].isna().all().all())
        self.assertTrue(rates[["n_verified_good", "n_verified_error"]].eq(0).all().all())
        verdicts = pd.read_csv(self.output / "reference_record_verdicts.csv")
        self.assertTrue(verdicts.reference_verdict.eq("unverified").all())
        self.assertFalse(verdicts.confirmed_four_boundaries.any())
        status = json.loads((self.output / "reference_status.json").read_text())
        self.assertFalse(status["accuracy_available"])

    def test_screening_does_not_assign_confirmed_A_B_C_or_boundary_correctness(self):
        successful = self.joined[self.joined.processing_success]
        self.assertTrue(successful.audit_group.eq("D_unverified").all())
        self.assertTrue(self.joined.loc[~self.joined.processing_success, "audit_group"].eq("technical_failure").all())
        self.assertTrue(self.joined.reference_verdict.eq("not_assessed").all())
        self.assertTrue(successful.screen_positive.any())
        self.assertTrue((successful.n_flags.eq(0) & successful.screen_positive).any())
        self.assertFalse(self.joined.audit_group.isin(["A", "B", "C", "correct", "incorrect"]).any())
        pd.testing.assert_frame_equal(self.frame, self.original)

    def test_confirmed_reference_errors_require_explicit_tolerance_before_qc_verdict(self):
        # Synthetic expert inputs exercise the acceptance gate only. They are
        # test fixtures, not independent annotations for real PTB evaluation.
        record = self.template[self.template.signal_availability.eq("median_available")].iloc[[0]].copy()
        record["review_status"] = "confirmed"
        record["reviewer"] = "synthetic_test_fixture"
        record["independent_reference"] = True
        for boundary in ["qrs_on", "qrs_off", "t_on", "t_off"]:
            record[boundary + "_review_ms"] = record[boundary + "_pred_ms"]
        with TemporaryDirectory(prefix="vcg07-refs-") as directory:
            directory = Path(directory)
            record.to_csv(directory / "manual_review_template.csv", index=False)
            comparison, summary, accepted = audit.review_references(self.frame, directory, self.evaluation_dir)
            self.assertEqual(len(comparison), 4)
            self.assertTrue(accepted.accepted.all())
            rates = pd.read_csv(directory / "qc_reference_rates.csv")
            self.assertTrue(rates[["false_alarm_rate", "missed_error_rate"]].isna().all().all())
            verdict = pd.read_csv(directory / "reference_record_verdicts.csv")
            self.assertEqual(verdict.loc[verdict.record_id.eq(record.iloc[0].record_id), "reference_verdict"].iloc[0],
                             "confirmed_no_acceptance_criterion")
            audit.review_references(self.frame, directory, self.evaluation_dir, acceptance_ms=20)
            verdict = pd.read_csv(directory / "reference_record_verdicts.csv")
            self.assertEqual(verdict.loc[verdict.record_id.eq(record.iloc[0].record_id), "reference_verdict"].iloc[0],
                             "within_declared_criterion")
            rates = pd.read_csv(directory / "qc_reference_rates.csv")
            self.assertEqual(rates.n_verified_good.sum(), 1)
            for invalid in [0, -10, np.nan, np.inf]:
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "explicitly positive"):
                    audit.review_references(self.frame, directory, self.evaluation_dir, acceptance_ms=invalid)

    def test_multiple_confirmed_reviewers_require_adjudication_not_last_row_wins(self):
        record = self.template[self.template.signal_availability.eq("median_available")].iloc[[0]].copy()
        record["review_status"] = "confirmed"
        record["independent_reference"] = True
        record["reviewer"] = "synthetic_reviewer_one"
        for boundary in ["qrs_on", "qrs_off", "t_on", "t_off"]:
            record[boundary + "_review_ms"] = record[boundary + "_pred_ms"]
        other = record.copy()
        other["reviewer"] = "synthetic_reviewer_two"
        # Distinct references exercise aggregation; both remain in the accepted
        # comparison table rather than silently selecting the last reviewer.
        other["qrs_on_review_ms"] = other["qrs_on_review_ms"] - 1
        with TemporaryDirectory(prefix="vcg07-multiref-") as directory:
            directory = Path(directory)
            pd.concat([record, other], ignore_index=True).to_csv(directory / "manual_review_template.csv", index=False)
            comparison, summary, accepted = audit.review_references(self.frame, directory, self.evaluation_dir, acceptance_ms=20)
            self.assertEqual(len(comparison), 8)
            self.assertTrue(accepted.accepted.all())
            verdicts = pd.read_csv(directory / "reference_record_verdicts.csv")
            self.assertEqual(verdicts.loc[verdicts.record_id.eq(record.iloc[0].record_id), "reference_verdict"].iloc[0],
                             "multiple_references_require_adjudication")
            rates = pd.read_csv(directory / "qc_reference_rates.csv")
            self.assertTrue(rates[["n_verified_good", "n_verified_error"]].eq(0).all().all())
            self.assertTrue(rates[["false_alarm_rate", "missed_error_rate"]].isna().all().all())


if __name__ == "__main__":
    unittest.main()

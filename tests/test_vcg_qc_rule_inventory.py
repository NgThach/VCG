"""Coverage/provenance tests for the read-only Notebook 07 rule inventory."""
import ast
from dataclasses import asdict, replace
from pathlib import Path
import re
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from vcg_preprocessing import Config
from vcg_qc_rule_inventory import qc_rules_inventory, source_audit_findings, COLUMNS


def test_every_literal_flag_and_beat_rejection_has_a_documented_rule():
    """Discover names from source; new flags cannot silently evade inventory."""
    discovered = set()
    for filename in ["vcg_pipeline.py", "vcg_preprocessing.py", "scripts/run_full_vcg_evaluation.py"]:
        tree = ast.parse((ROOT / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "append" and node.args and isinstance(node.args[0], ast.Constant):
                    value = node.args[0].value
                    target = ast.unparse(node.func.value)
                    if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]*", value) and any(
                            token in target for token in ["flags", "warnings", "extra"]):
                        discovered.add(value)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "reason" for t in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and node.value.value:
                    # Restrict to representative_beat's categorical codes, not prose.
                    if re.fullmatch(r"[a-z][a-z0-9_]*", node.value.value):
                        discovered.add(node.value.value)
    # Batch adds these as literal elements rather than append calls.
    discovered.add("processing_failed")
    inventory = qc_rules_inventory()
    assert discovered <= set(inventory.qc_flag), discovered - set(inventory.qc_flag)
    assert inventory.qc_flag.is_unique
    assert list(inventory.columns) == COLUMNS
    assert not inventory.isna().any().any()


def test_provenance_points_to_actual_implemented_flag_or_dynamic_warning():
    frame = qc_rules_inventory()
    for row in frame.itertuples():
        source = (ROOT / row.source_file).read_text().splitlines()
        line = source[row.source_line - 1]
        needle = "preliminary_detector: " if row.qc_flag.startswith("preliminary_detector:") else row.qc_flag
        assert needle in line, (row.qc_flag, line)
    for finding in source_audit_findings():
        assert (ROOT / finding["source_file"]).read_text().splitlines()[finding["source_line"] - 1]
        assert "evidence_scope" in finding


def test_distinguishes_direct_record_flags_from_separate_diagnostics():
    frame = qc_rules_inventory().set_index("qc_flag")
    for flag in ["invalid_values", "isolated_impulse_review", "edge", "rr_outlier",
                 "optional_highpass_may_change_ST_T", "processing_failed",
                 "eval_t_on_methods_gap_gt100ms"]:
        assert not bool(frame.loc[flag, "contributes_to_v1_status"]), flag
    for flag in ["rr_irregular", "pq_not_quiet_review", "t_end_methods_disagree",
                 "st_level_requires_review", "powerline_inconclusive_review", "features_incomplete"]:
        assert bool(frame.loc[flag, "contributes_to_v1_status"]), flag
    assert frame.loc["processing_failed", "scope"] == "batch_failure"
    assert frame.loc["eval_t_on_methods_gap_gt100ms", "scope"] == "batch_evaluation"
    assert "NOT" not in frame.loc["st_level_requires_review", "trigger_condition"]
    assert "st-mbase > max(5*mnoise,0.15*amp)" in frame.loc["st_level_requires_review", "trigger_condition"]


def test_configurable_thresholds_are_reported_without_mutating_v1_config():
    original = Config()
    before = asdict(original)
    changed = replace(original, min_template_correlation=.91, t_fraction=.12,
                      qrs_fraction=.09, baseline_speed_fraction=.13,
                      input_saturation_abs_mV=14.0)
    frame = qc_rules_inventory(changed).set_index("qc_flag")
    assert "0.91" in frame.loc["different_morphology_retained_review", "threshold_v1"]
    assert "0.12" in frame.loc["t_off_no_baseline_return", "threshold_v1"]
    assert "0.09" in frame.loc["qrs_on_not_found", "threshold_v1"]
    assert "0.13" in frame.loc["pq_not_quiet_review", "threshold_v1"]
    assert "14" in frame.loc["near_nominal_adc_rail_review", "threshold_v1"]
    pd.testing.assert_frame_equal(qc_rules_inventory(changed), qc_rules_inventory(asdict(changed)))
    assert asdict(original) == before
    assert Config() == original


def test_all_saved_notebook06_flags_have_rules_when_results_present():
    base = ROOT / "outputs/vcg_12_features_train_valid_test"
    inventory = set(qc_rules_inventory().qc_flag)
    for split in ["train", "valid", "test"]:
        path = base / f"{split}_features.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for column in ["qc_flags", "input_qc_flags", "evaluation_flags", "qrs_qc_flags", "t_qc_flags"]:
            if column in frame:
                observed = {flag.strip() for text in frame[column].dropna().astype(str)
                            for flag in text.split(";") if flag.strip()}
                assert observed <= inventory, (column, observed - inventory)


def load_tests(loader, tests, pattern):
    """Run with stdlib unittest too; the project environment need not have pytest."""
    return unittest.TestSuite(unittest.FunctionTestCase(function)
                              for name, function in sorted(globals().items())
                              if name.startswith("test_") and callable(function))


if __name__ == "__main__":
    unittest.main()

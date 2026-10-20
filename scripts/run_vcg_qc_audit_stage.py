#!/usr/bin/env python3
"""Notebook-07 stages; subprocess entry supports safe multiprocessing on macOS."""
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcg_qc_audit import (ROOT, load_inputs, initialize_audit, qc_tables,
    boundary_audits, failure_analysis, create_review_template, review_references,
    technical_plots, sensitivity_plots, write_json)
from vcg_preprocessing import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['base', 'gallery', 'sensitivity'])
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/vcg_qc_audit_v1')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    frame, manifest, config, input_dir, evaluation_dir = load_inputs()
    artifacts = initialize_audit(frame, input_dir, evaluation_dir, args.output_dir, config)
    cfg = Config(**config['config'])
    if args.stage == 'base':
        from vcg_qc_rule_inventory import qc_rules_inventory, source_audit_findings
        rules = qc_rules_inventory(cfg)
        rules.to_csv(args.output_dir/'qc_rules_inventory.csv', index=False)
        write_json(args.output_dir/'source_audit_findings.json', source_audit_findings())
        qc = qc_tables(frame, args.output_dir)
        unknown = set(qc['frequency'].qc_flag)-set(rules.qc_flag)
        if unknown:
            raise ValueError(f'Observed flags absent from source inventory: {unknown}')
        qrs, twave, joined = boundary_audits(frame, artifacts, args.output_dir, cfg)
        failure_analysis(frame, artifacts, args.output_dir)
        sample = create_review_template(joined, args.output_dir)
        review_references(frame, args.output_dir, evaluation_dir)
        technical_plots(frame, qc, qrs, twave, args.output_dir)
        print('Records:', len(frame), 'review rows:', len(sample))
    elif args.stage == 'gallery':
        from vcg_qc_audit_gallery import create_audit_gallery, validate_audit_gallery
        joined = pd.read_csv(args.output_dir/'audit_record_index.csv')
        result = create_audit_gallery(joined, artifacts, args.output_dir,
            ROOT/'content/ptbdb/physionet.org/files/ptbdb/1.0.0', cfg, args.workers, True)
        validation = validate_audit_gallery(args.output_dir, set(frame.record_id))
        write_json(args.output_dir/'gallery_validation.json', validation)
        print(result)
    elif args.stage == 'sensitivity':
        from vcg_feature_boundary_sensitivity import run_feature_sensitivity, summarize_sensitivity
        result = run_feature_sensitivity(frame, artifacts, args.output_dir, cfg, resume=True)
        full = result['long'][result['long'].original_feature_completeness.eq(12)]
        summary, rank = summarize_sensitivity(full)
        summary.to_csv(args.output_dir/'feature_sensitivity_full12_summary.csv', index=False)
        rank.to_csv(args.output_dir/'feature_sensitivity_full12_rank.csv', index=False)
        sensitivity_plots(result, args.output_dir)
        print({k: result['metadata'][k] for k in ['n_records', 'n_scenarios', 'n_long_rows', 'cache_hits']})


if __name__ == '__main__':
    main()

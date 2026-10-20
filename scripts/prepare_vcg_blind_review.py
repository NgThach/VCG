#!/usr/bin/env python3
"""Export selected median signals without predicted boundaries, QC, or labels.

Use this CSV and its figures for an independent first annotation pass. The
separate manual_review_template.csv retains predictions for later adjudication.
No reference annotation is generated or prefilled by this script.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/vcg_full_evaluation_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare(output):
    output = Path(output)
    sample = pd.read_csv(output/'annotations/manual_review_template.csv')
    directory = output/'annotations/blinded_signals'
    directory.mkdir(exist_ok=True)
    rows = []
    for _, row in sample.iterrows():
        rid = row.record_id
        path = output/'artifacts'/(rid.replace('/', '_')+'.npz')
        review_path = ''
        availability = 'no_median_beat'
        if path.exists():
            with np.load(path) as saved:
                if 'median_xyz' in saved and 'time_ms' in saved:
                    xyz, time = saved['median_xyz'], saved['time_ms']
                    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True,
                                              layout='constrained')
                    for j, color in enumerate(['#2563eb', '#059669', '#d97706']):
                        axes[j].plot(time, xyz[:, j], color=color, lw=1)
                        axes[j].set_ylabel('XYZ'[j]+' (mV)')
                    axes[3].plot(time, np.linalg.norm(xyz, axis=1), color='#0891b2', lw=1)
                    axes[3].set_ylabel('Magnitude (mV)')
                    axes[3].set_xlabel('Time relative to alignment anchor (ms)')
                    for ax in axes:
                        ax.grid(alpha=.2)
                        ax.axhline(0, color='.7', lw=.5)
                    fig.suptitle(rid+' | Median Frank XYZ for independent boundary annotation')
                    target = directory/(rid.replace('/', '_')+'.png')
                    fig.savefig(target, dpi=120)
                    plt.close(fig)
                    review_path = target.relative_to(output).as_posix()
                    availability = 'median_signal_available'
        rows.append(dict(record_id=rid, time_reference='median_relative_ms',
            reference_anchor_sample=np.nan, reviewer='', review_status='pending',
            independent_reference=False, qrs_on_ms=np.nan, qrs_off_ms=np.nan,
            t_on_ms=np.nan, t_off_ms=np.nan, signal_availability=availability,
            review_plot_path=review_path, notes=''))
    path = output/'annotations/manual_review_blinded.csv'
    if path.exists():
        raise FileExistsError(f'{path} exists; expert edits must not be overwritten')
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    print(f'{len(frame)} pending review rows, {int(frame.review_plot_path.ne("").sum())} signals; no reference values filled.')
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/vcg_full_evaluation'))
    prepare(parser.parse_args().output_dir)

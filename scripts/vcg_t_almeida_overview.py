"""Every stored recording: median/T boundaries and the actual V1/V2 T loops.

Rendering only. No delineation, filtering, interpolation or substitute loops.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vcg_t_almeida_plots import load_case, _equal_3d, AXIS_COLORS, V1, V2, ON, OFF


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _time(value):
    return f'{value:g}' if pd.notna(value) else 'NA'


def _reason(row):
    values = []
    for key in ('failure_reason', 't_on_failure_reason', 't_off_failure_reason'):
        value = row.get(key)
        if pd.notna(value) and str(value) and str(value) not in values:
            values.append(str(value))
    return '\n'.join(values)


def _draw_record(fig, grid, position, case):
    row, median, arrays = case['row'], case['median'], case['arrays']
    ax = fig.add_subplot(grid[position, 0])
    title = (f"{row['record_id']} | {row['label_name']} / {row['split']} | {row['status']}\n"
             f"T V1: {_time(row.get('t_on_v1_ms'))} – {_time(row.get('t_off_v1_ms'))} ms"
             f"    V2: {_time(row.get('t_on_v2_ms'))} – {_time(row.get('t_off_v2_ms'))} ms")
    ax.set_title(title, loc='left', fontsize=10, pad=9)
    xyz, time = median['xyz'], median['time_ms']
    if xyz is None:
        ax.axis('off')
        ax.text(.03, .60, 'MEDIAN UNAVAILABLE\n'+
                '\n'.join(textwrap.wrap(str(row.get('failure_reason')), 74)),
                transform=ax.transAxes, va='center', fontsize=9, color='#92392a')
    else:
        for j, axis in enumerate('XYZ'):
            ax.plot(time, xyz[:, j], lw=.95, color=AXIS_COLORS[j], label=axis)
        for boundary in ('qrs_on_ms', 'qrs_off_ms'):
            if pd.notna(row.get(boundary)):
                ax.axvline(row[boundary], color='#94a3b8', ls=':', lw=.8)
        for version, color, style in (('v1', V1, '--'), ('v2', V2, '-')):
            on, off = row.get('t_on_'+version+'_ms'), row.get('t_off_'+version+'_ms')
            if pd.notna(on) and pd.notna(off):
                ax.axvspan(on, off, color=color, alpha=.10)
            for boundary, value in (('on', on), ('off', off)):
                if pd.notna(value):
                    edge_color = color if version == 'v1' else ON if boundary == 'on' else OFF
                    ax.axvline(value, color=edge_color, ls=style, lw=1.15)
        ax.set(xlabel='ms relative to saved QRS anchor', ylabel='XYZ (mV)',
               xlim=(time[0], time[-1]))
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, ncol=3, loc='upper left')

    loops = [arrays[key] for key in ('t_loop_v1', 't_loop_v2') if key in arrays]
    limits = np.vstack(loops) if loops else xyz
    for column, (version, color) in enumerate((('v1', V1), ('v2', V2)), 1):
        ax3 = fig.add_subplot(grid[position, column], projection='3d')
        loop = arrays.get('t_loop_'+version)
        if loop is None:
            ax3.axis('off')
            if xyz is None:
                message = 'Median unavailable\nNo T loop'
            elif version == 'v2':
                message = 'No valid V2 T loop\n'+_reason(row)
            else:
                message = 'No valid V1 T loop\nMissing T boundary'
            wrapped = '\n'.join('\n'.join(textwrap.wrap(line, 42)) for line in message.splitlines())
            ax3.text2D(.02, .87, wrapped, transform=ax3.transAxes,
                       va='top', fontsize=8, color='#92392a')
        else:
            on, off = row['t_on_'+version+'_sample'], row['t_off_'+version+'_sample']
            if xyz is None or pd.isna(on) or pd.isna(off):
                raise ValueError('Loop exists without median/boundaries: '+row['record_id'])
            if not np.array_equal(loop, xyz[int(on):int(off)+1]):
                raise ValueError('Saved loop differs from original median cut: '+row['record_id'])
            ax3.plot(*loop.T, color=color, lw=1.5)
            ax3.scatter(*loop[0], color=color, s=25, marker='o')
            ax3.scatter(*loop[-1], color=OFF, s=30, marker='x')
            start = max(0, len(loop)//2-1)
            end = min(len(loop)-1, start+max(1, len(loop)//10))
            ax3.quiver(*loop[start], *(loop[end]-loop[start]), color=color, arrow_length_ratio=.3)
            _equal_3d(ax3, limits)
            ax3.set(xlabel='X (mV)', ylabel='Y (mV)', zlabel='Z (mV)')
            ax3.tick_params(labelsize=7, pad=0)
            for label in (ax3.xaxis.label, ax3.yaxis.label, ax3.zaxis.label):
                label.set_fontsize(8)
        ax3.set_title(version.upper()+' T loop', fontsize=10, color=color, pad=5)
    return {
        'record_id': row['record_id'], 'label_name': row['label_name'], 'split': row['split'],
        'status': row['status'], 'median_present': xyz is not None,
        'has_t_loop_v1': 't_loop_v1' in arrays, 'has_t_loop_v2': 't_loop_v2' in arrays,
        'failure_reason': _reason(row),
    }


def create_t_loop_contact_sheets(results, experiment_dir, output_dir=None, page_size=6):
    """Display every ID exactly once, keeping missing medians and partial T marks."""
    if not isinstance(page_size, int) or not 1 <= page_size <= 6:
        raise ValueError('page_size must be an integer from 1 to 6')
    if not results.record_id.is_unique or results.empty:
        raise ValueError('Expected nonempty unique recording IDs')
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir or experiment_dir/'notebook08_extension/all_record_t_loops')
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = results.sort_values('record_id').reset_index(drop=True)
    manifest = []
    pages = int(np.ceil(len(ordered)/page_size))
    source_hash = _sha(__file__)+_sha(Path(__file__).with_name('vcg_t_almeida_plots.py'))
    for start in range(0, len(ordered), page_size):
        group = ordered.iloc[start:start+page_size]
        number = start//page_size+1
        path = output_dir/f'page_{number:03d}.png'
        sidecar = path.with_suffix('.json')
        h = hashlib.sha256((source_hash+str(page_size)+str(len(ordered))+
                           group.to_json(orient='records', double_precision=12)).encode())
        for record in group.itertuples(index=False):
            safe = record.record_id.replace('/', '_')
            for source in (experiment_dir/'diagnostics'/(safe+'.json'),
                           experiment_dir/'artifacts'/(safe+'.npz'),
                           Path(record.v1_median_npz_path), Path(record.v1_median_json_path)):
                h.update(source.read_bytes())
        fingerprint = h.hexdigest()
        cache = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        reusable = (path.is_file() and cache.get('fingerprint') == fingerprint
                    and cache.get('png_sha256') == _sha(path))
        if reusable:
            record_rows = cache['records']
        else:
            figure_height = 3.65*len(group)+1.4
            fig = plt.figure(figsize=(19, figure_height))
            grid = fig.add_gridspec(len(group), 3, width_ratios=(1.7, 1, 1),
                                    left=.05, right=.975, bottom=1.15/figure_height, top=.945,
                                    wspace=.16, hspace=.52)
            record_rows = []
            try:
                for position, record in enumerate(group.itertuples(index=False)):
                    case = load_case(experiment_dir, record.record_id)
                    record_rows.append(_draw_record(fig, grid, position, case))
                fig.suptitle(
                    f'All recordings: T boundaries and V1/V2 T loops | page {number}/{pages}'
                    f' | records {start+1}–{start+len(group)}/{len(ordered)}',
                    fontsize=15, y=.985)
                fig.text(.05, .22/figure_height,
                         'XYZ: blue / orange / purple. QRS: grey dotted. V1 T: grey dashed; '
                         'V2 T-on/off: cyan / red. 3D: circle=start, x=end, arrow=time.\n'
                         'Same PQ origin and 3D scale within each recording. NA=unresolved. '
                         'Predictions only; no independent reference accuracy claim.',
                         fontsize=10, color='#475569')
                fig.savefig(path, dpi=110, facecolor='white')
            finally:
                plt.close(fig)
            sidecar.write_text(json.dumps({
                'fingerprint': fingerprint, 'png_sha256': _sha(path), 'records': record_rows,
            }, ensure_ascii=False, indent=2)+'\n')
        for record in record_rows:
            safe = record['record_id'].replace('/', '_')
            manifest.append({
                **record, 'page': number, 'plot_path': path.name,
                'full_xyz_figure': '../../figures/'+safe+'_comparison.png',
                'full_loop_figure': '../../figures/'+safe+'_loops.png',
            })
        print(f'T-loop overview: page {number}/{pages} | {start+len(group)}/{len(ordered)} IDs'
              + (' (cache)' if reusable else ''), flush=True)
    manifest = pd.DataFrame(manifest)
    if (len(manifest) != len(ordered) or not manifest.record_id.is_unique
            or set(manifest.record_id) != set(ordered.record_id)):
        raise ValueError('All-record T-loop overview lost or duplicated IDs')
    manifest.to_csv(output_dir/'manifest.csv', index=False)
    summary = {
        'recordings_displayed': len(manifest), 'pages': pages, 'page_size': page_size,
        'medians_displayed': int(manifest.median_present.sum()),
        'v1_t_loops_displayed': int(manifest.has_t_loop_v1.sum()),
        'v2_t_loops_displayed': int(manifest.has_t_loop_v2.sum()),
        'ids_without_v2_loop_displayed': int((~manifest.has_t_loop_v2).sum()),
        'missing_medians_displayed': int((~manifest.median_present).sum()),
        'every_id_exactly_once': True, 'no_substitute_loops': True,
        'source_csv_sha256': _sha(experiment_dir/'v2_t_boundaries.csv'),
    }
    (output_dir/'coverage.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    sections = []
    for number, records in manifest.groupby('page', sort=True):
        links = ' · '.join(
            html.escape(r.record_id)+' [<a href="'+html.escape(r.full_xyz_figure)+'">XYZ</a> / '
            '<a href="'+html.escape(r.full_loop_figure)+'">T loops 3D + projections</a>]'
            for r in records.itertuples(index=False))
        sections.append(f'<section><h2>Page {number}/{pages}</h2><p>{links}</p>'
                        f'<img loading="lazy" src="{records.plot_path.iloc[0]}" '
                        f'alt="T loops page {number}"></section>')
    (output_dir/'index.html').write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>All-record T loops</title>'
        '<style>body{font:15px system-ui;margin:25px;color:#173143}img{width:100%;height:auto}'
        'section{margin:35px 0;border-top:1px solid #cbd5e1}a{color:#0f766e}</style></head><body>'
        f'<h1>T loops: all {len(manifest)} recording IDs</h1>'
        f'<p>{summary["v2_t_loops_displayed"]} V2 loops; '
        f'{summary["ids_without_v2_loop_displayed"]} IDs without V2 loop remain visible. '
        'No independent reference accuracy claim.</p>'+''.join(sections)+'</body></html>')
    return manifest, summary

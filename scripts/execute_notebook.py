#!/usr/bin/env python3
"""Execute a notebook with this Python, save outputs separately, stop on errors."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def execute(path, output=None):
    import nbformat
    from nbclient import NotebookClient

    path = Path(path).resolve()
    output = Path(output or ROOT / ('executed_' + path.name))
    log_dir = ROOT / 'outputs/reproducibility'
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    notebook = nbformat.read(path, as_version=4)
    log_path = log_dir / (path.stem + '.log')
    log_path.write_text('', encoding='utf-8')

    def starting(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            message = f'{path.name}: cell {cell_index + 1}/{len(notebook.cells)}'
            print(message, flush=True)
            with log_path.open('a', encoding='utf-8') as stream:
                stream.write(message + '\n')

    with tempfile.TemporaryDirectory(prefix='vcg-kernel-') as temporary:
        runtime = Path(temporary)
        spec = runtime / 'kernels/vcg-local'
        spec.mkdir(parents=True)
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'VCG current Python', 'language': 'python',
        }), encoding='utf-8')
        old_path = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = str(runtime) + (os.pathsep + old_path if old_path else '')
        try:
            client = NotebookClient(notebook, timeout=-1, kernel_name='vcg-local',
                allow_errors=False, resources={'metadata': {'path': str(ROOT)}},
                on_cell_start=starting, store_widget_state=False)
            try:
                client.execute()
            finally:
                nbformat.write(notebook, output)
        finally:
            if old_path is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = old_path
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    report = {'notebook': path.name, 'python': sys.executable,
              'code_cells': len(cells),
              'executed_code_cells': sum(cell.execution_count is not None for cell in cells),
              'execution_errors': sum(out.output_type == 'error' for cell in cells
                                      for out in cell.get('outputs', [])),
              'embedded_png_outputs': sum('image/png' in out.get('data', {}) for cell in cells
                                           for out in cell.get('outputs', [])),
              'seconds': round(time.monotonic() - started, 2)}
    (log_dir / (path.stem + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    execute(args.notebook, args.output)


if __name__ == '__main__':
    main()

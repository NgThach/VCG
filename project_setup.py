"""Download verified PTB files and build notebook prerequisites in this clone."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'content/ptbdb/physionet.org/files/ptbdb/1.0.0'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def check_dataset(deep=False):
    manifest = json.loads((ROOT / 'data/ptbdb-files.json').read_text(encoding="utf-8"))
    missing, corrupt = [], []
    for name, expected in manifest.items():
        path = DATA_ROOT / name
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != expected['bytes'] or (deep and digest(path) != expected['sha256']):
            corrupt.append(name)
    return {'expected_files': len(manifest), 'missing': missing, 'corrupt': corrupt,
            'verified_sha256': deep, 'complete': not missing and not corrupt}


def download(url, destination, expected):
    partial = destination.with_suffix(destination.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'User-Agent': 'VCG-reproducible-setup',
        **({'Range': f'bytes={offset}-'} if offset else {})})
    print(f'Tải PTB từ {url}', flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        if not append:
            offset = 0
        total = expected['bytes']
        last_log = time.monotonic()
        with partial.open('ab' if append else 'wb') as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(chunk)
                offset += len(chunk)
                if time.monotonic() - last_log > 10:
                    print(f'  {offset / 1e6:.0f}/{total / 1e6:.0f} MB', flush=True)
                    last_log = time.monotonic()
    if partial.stat().st_size != expected['bytes'] or digest(partial) != expected['sha256']:
        partial.unlink()
        raise ValueError('Gói tải về không khớp SHA-256; hãy chạy lại để tải bản đầy đủ.')
    partial.replace(destination)


def download_bundle(resource, cache, archive):
    """Small release assets avoid a single very large HTTP upload/download."""
    def get_part(part):
        path = cache / part['filename']
        if not path.exists() or path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
            for attempt in range(3):
                try:
                    download(part['url'], path, part)
                    break
                except (OSError, ValueError) as error:
                    if attempt == 2:
                        raise
                    print(f"Thử lại {part['filename']}: {error}", flush=True)
                    time.sleep(1)
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        pieces = list(pool.map(get_part, resource['parts']))
    temporary = archive.with_suffix(archive.suffix + '.joining')
    with temporary.open('wb') as target:
        for path in pieces:
            with path.open('rb') as source:
                shutil.copyfileobj(source, target)
    if temporary.stat().st_size != resource['bytes'] or digest(temporary) != resource['sha256']:
        raise ValueError('Các phần ghép lại không khớp SHA-256 của dataset')
    temporary.replace(archive)
    for path in pieces:
        path.unlink()


def ensure_dataset():
    state = check_dataset()
    if state['corrupt']:
        raise ValueError('Dữ liệu hiện có bị thay đổi; chuyển các file này ra ngoài trước khi tải lại: '
                         + ', '.join(state['corrupt'][:8]))
    if state['complete']:
        return state
    resource = json.loads((ROOT / 'data/resources.json').read_text(encoding="utf-8"))['ptbdb']
    cache = ROOT / 'content/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / resource['filename']
    if not archive.exists() or archive.stat().st_size != resource['bytes'] or digest(archive) != resource['sha256']:
        download_bundle(resource, cache, archive)
    manifest = json.loads((ROOT / 'data/ptbdb-files.json').read_text(encoding="utf-8"))
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    # Explicit members only: reject links, traversal, duplicates and unlisted files.
    with tarfile.open(archive, 'r:gz') as bundle:
        seen = set()
        for member in bundle:
            if (not member.isfile() or member.name not in manifest or member.name in seen
                    or member.size != manifest[member.name]['bytes']):
                raise ValueError('Archive member không hợp lệ: ' + member.name)
            seen.add(member.name)
            path = DATA_ROOT / member.name
            if not path.resolve().is_relative_to(DATA_ROOT.resolve()):
                raise ValueError('Archive path thoát khỏi thư mục dataset')
            if path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.partial')
            with bundle.extractfile(member) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            if digest(temporary) != manifest[member.name]['sha256']:
                raise ValueError('File không khớp checksum: ' + member.name)
            temporary.replace(path)
        if seen != set(manifest):
            raise ValueError('Archive thiếu file trong manifest')
    result = check_dataset(deep=True)
    if not result['complete']:
        raise ValueError('Dataset chưa đầy đủ: ' + json.dumps(result))
    print(f"PTB: đủ {result['expected_files']} file; SHA-256 khớp manifest.", flush=True)
    return result


def run(*arguments):
    print('Chạy:', ' '.join(str(arg) for arg in arguments), flush=True)
    # Notebook prerequisites run in child kernels; keep their JSON/text IO UTF-8
    # on Windows too, including older notebooks that use Path.read_text().
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True,
                   env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})


def execute(number):
    notebook = next(ROOT.glob(f'{number:02d}_*.ipynb'))
    run(ROOT / 'scripts/execute_notebook.py', notebook)


def full_v1_ready():
    folder = ROOT / 'outputs/vcg_full_evaluation'
    completion = folder / 'evaluation/completion.json'
    if not completion.is_file():
        return False
    state = json.loads(completion.read_text(encoding="utf-8"))
    ids = [row.split(',')[0] for row in (ROOT / 'data/split_manifest.csv').read_text(encoding="utf-8").splitlines()[1:]]
    return (state.get('full_dataset_complete') is True and state.get('completed_records') == 448
            and all((folder / 'artifacts' / (rid.replace('/', '_') + ext)).is_file()
                    for rid in ids for ext in ('.json', '.npz')))


def cached_v1_inputs_ready():
    """Validate distributed median inputs without claiming a completed evaluation.

    A source notebook can gain plots/text while the numerical V1 modules stay
    identical. Do not resume or rewrite that historical evaluation in this case.
    Notebook 06 still extracts features from raw; V2 independently checks their
    parity against these medians before using them.
    """
    folder = ROOT / 'outputs/vcg_full_evaluation'
    config_path = folder / 'config.json'
    manifest_path = folder / 'split_manifest.csv'
    if not config_path.is_file() or not manifest_path.is_file():
        return False
    import pandas as pd
    from dataclasses import asdict
    from vcg_preprocessing import Config
    from scripts.vcg_reference_metrics import load_median

    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
        # All numerical preprocessing/detection/delineation/feature code lives
        # in these two modules. Never ignore a mismatch in either one.
        hashes = config.get('baseline_hashes', {})
        if any(hashes.get(name) != digest(ROOT / name)
               for name in ('vcg_pipeline.py', 'vcg_preprocessing.py')):
            return False
        if config.get('config') != json.loads(json.dumps(asdict(Config()))):
            return False
        columns = ['record_id', 'patient_id', 'split', 'label_name']
        expected = pd.read_csv(ROOT / 'data/split_manifest.csv')[columns]
        saved = pd.read_csv(manifest_path)[columns]
        for frame in (expected, saved):
            frame['split'] = frame['split'].replace({'val': 'valid', 'validation': 'valid'})
        if (expected.empty or not expected.record_id.is_unique or not saved.record_id.is_unique
                or not expected.sort_values('record_id').reset_index(drop=True).equals(
                    saved.sort_values('record_id').reset_index(drop=True))):
            return False
        for rid in expected.record_id:
            # Verifies record identity, finite XYZ, fs, anchor and ms time grid;
            # recorded extraction failures may have xyz=None, never a fake beat.
            load_median(folder / 'artifacts', rid)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def ensure_v1():
    if not full_v1_ready() and cached_v1_inputs_ready():
        print('V1: artifact đầu vào đã được xác minh; không resume báo cáo cũ thiếu completion.', flush=True)
        return
    if not full_v1_ready():
        folder = ROOT / 'outputs/vcg_full_evaluation'
        arguments = [ROOT / 'scripts/run_full_vcg_evaluation.py', '--workers',
                     os.environ.get('VCG_WORKERS', '4'), '--no-plots',
                     '--split-manifest', ROOT / 'data/split_manifest.csv']
        if (folder / 'config.json').exists():
            arguments.append('--resume')
        run(*arguments)
    if not full_v1_ready():
        raise RuntimeError('V1 chưa tạo đủ artifact cho 448 ID')


def ensure_features():
    path = ROOT / 'outputs/vcg_12_features_train_valid_test/completion.json'
    if (not path.exists() or json.loads(path.read_text(encoding="utf-8")).get('completed_records') != 448
            or not all((path.parent / name).is_file() for name in
                       ['train_features.csv', 'valid_features.csv', 'test_features.csv', 'config.json', 'split_manifest.csv'])):
        execute(6)
    # Use the existing strict source/identity checks; never accept stale source hashes.
    sys.path.insert(0, str(ROOT / 'scripts'))
    from vcg_qc_audit import load_inputs
    load_inputs()


def prepare_notebook(number):
    number = int(number)
    if number not in range(1, 10):
        raise ValueError('Notebook phải từ 1 đến 9')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'vcg-matplotlib'))
    ensure_dataset()
    if number >= 6:
        ensure_v1()
    if number >= 7:
        ensure_features()
    if number == 8:
        qc = ROOT / 'outputs/vcg_qc_audit_v1'
        if not all((qc / name).is_file() for name in
                   ['manual_review_blinded.csv', 'feature_sensitivity.csv', 'qc_audit_report.md']):
            execute(7)
        experiment = ROOT / 'outputs/vcg_t_almeida_v2_experiment'
        summary = experiment / 'execution_summary.json'
        if (not summary.is_file() or json.loads(summary.read_text(encoding="utf-8")).get('attempted') != 448
                or not (experiment / 'gallery/manifest.csv').is_file()
                or not (experiment / 'figures/patient104_s0306lre_loops.png').is_file()):
            run(ROOT / 'scripts/run_vcg_t_almeida_experiment.py')
    print(f'Đầu vào Notebook {number:02d} đã sẵn sàng.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=int, choices=range(1, 10),
                        help='Tạo các kết quả đầu vào còn thiếu cho notebook này')
    parser.add_argument('--verify', action='store_true', help='Đọc và kiểm SHA-256 toàn bộ dataset')
    args = parser.parse_args()
    if args.notebook:
        prepare_notebook(args.notebook)
    else:
        ensure_dataset()
    if args.verify:
        state = check_dataset(deep=True)
        print(json.dumps(state, indent=2))
        if not state['complete']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

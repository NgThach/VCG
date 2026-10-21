"""Hash numerical files byte-for-byte and notebook source without saved output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_digest(path):
    path = Path(path)
    if path.suffix == '.ipynb':
        notebook = json.loads(path.read_text(encoding='utf-8'))
        cells = [{'cell_type': cell['cell_type'],
                  'source': ''.join(cell.get('source', []))}
                 for cell in notebook['cells']]
        content = json.dumps(cells, ensure_ascii=False, sort_keys=True).encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

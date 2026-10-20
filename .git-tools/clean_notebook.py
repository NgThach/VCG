"""Git clean filter: commit notebook source while keeping local figures intact."""
import json
import sys


notebook = json.load(sys.stdin)
for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
        cell.get('metadata', {}).pop('execution', None)
        cell.get('metadata', {}).pop('execution_scope', None)
notebook.get('metadata', {}).pop('widgets', None)
json.dump(notebook, sys.stdout, ensure_ascii=False, indent=1)
sys.stdout.write('\n')

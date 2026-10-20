#!/usr/bin/env python3
"""Run the requested Notebook09 tests and persist an actual result report."""
from pathlib import Path
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
out=ROOT/'outputs/vcg_t_almeida_v2_experiment'
(out/'logs').mkdir(parents=True,exist_ok=True)
(out/'reports').mkdir(exist_ok=True)
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_vcg_t_almeida.py')
with (out/'logs/test_run.log').open('w') as stream:
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    skipped=len(result.skipped),successful=result.wasSuccessful(),
    scope='quadratic spline filter mathematics, single-lead, TLS, failure paths, real PTB integrity',
    clinical_validation=False)
(out/'reports/test_results.json').write_text(json.dumps(report,indent=2))
print((out/'logs/test_run.log').read_text())
sys.exit(0 if result.wasSuccessful() else 1)

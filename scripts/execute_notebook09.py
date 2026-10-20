#!/usr/bin/env python3
"""Execute Notebook09 in the current Python environment using a local kernelspec."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vcg_t_almeida_v2_experiment'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/vcg_t_almeida_mpl')
runtime=OUT/'runtime_jupyter'
spec=runtime/'kernels/vcg09-ipcv'
spec.mkdir(parents=True,exist_ok=True)
(spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    display_name='ipcv (Notebook09 local)',language='python',env={'MPLCONFIGDIR':'/private/tmp/vcg_t_almeida_mpl'})))
os.environ['JUPYTER_PATH']=str(runtime)+(os.pathsep+os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
os.environ['JUPYTER_RUNTIME_DIR']='/private/tmp/vcg_t_almeida_jupyter_runtime'
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
import nbformat
from nbclient import NotebookClient


def main():
    path=ROOT/'09_VCG_T_Loop_Almeida_Wavelet_TLS_Experiment.ipynb'
    nb=nbformat.read(path,as_version=4)
    log=OUT/'logs/notebook_execution.log'
    def starting(cell,cell_index,**kwargs):
        if cell.cell_type=='code':
            message=f'Executing code cell {cell_index+1}/{len(nb.cells)}'
            print(message,flush=True)
            with log.open('a') as f:f.write(message+'\n')
    start=time.monotonic()
    client=NotebookClient(nb,timeout=1200,kernel_name='vcg09-ipcv',allow_errors=False,
        resources={'metadata':{'path':str(ROOT)}},on_cell_start=starting,store_widget_state=False)
    client.execute()
    nbformat.write(nb,path)
    code=[c for c in nb.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
    images=sum('image/png' in o.get('data',{}) for c in code for o in c.get('outputs',[]))
    report=dict(notebook=path.name,python=sys.executable,code_cells=len(code),
        executed_code_cells=sum(c.execution_count is not None for c in code),execution_errors=len(errors),
        embedded_png_outputs=images,seconds_elapsed=time.monotonic()-start)
    (OUT/'reports/notebook_execution.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

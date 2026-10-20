"""All-record overview of the actual saved V1 XYZ cuts, matching Notebook05."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_v1_contact_sheets(frame, artifact_root, output_dir, page_size=12):
    """Show every record exactly once, with real XYZ and final V1 boundaries.

    A malformed/missing artifact becomes a visible error panel rather than a
    dropped row. No boundary, waveform, or loop is estimated by this renderer.
    """
    if not isinstance(page_size, int) or not 1 <= page_size <= 12:
        raise ValueError('page_size must be between1 and12')
    if not frame.record_id.is_unique:
        raise ValueError('Duplicate record_id')
    root, output_dir = Path(artifact_root), Path(output_dir)
    if (root/'artifacts').is_dir():
        root = root/'artifacts'
    out = output_dir/'contact_sheets'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    ordered = frame.sort_values('record_id').reset_index(drop=True)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    colors = ['#1873aa','#c04c3c','#32834e']
    edge_colors = {'qrs_on':'#176ca5','qrs_off':'#176ca5','t_on':'#c67116','t_off':'#c67116'}
    for start in range(0,len(ordered),page_size):
        page = ordered.iloc[start:start+page_size]
        number = start//page_size+1
        path = out/f'page_{number:03d}.png'
        check_path = path.with_suffix('.json')
        h = hashlib.sha256((source_hash+page.to_json(orient='records',double_precision=12)).encode())
        for rid in page.record_id:
            for ext in ['.json','.npz']:
                p = root/(rid.replace('/','_')+ext)
                h.update(p.read_bytes() if p.is_file() else b'missing')
        fingerprint = h.hexdigest()
        cache = json.loads(check_path.read_text()) if check_path.is_file() else {}
        valid_cache = path.is_file() and cache.get('fingerprint')==fingerprint and cache.get('png_sha256')==hashlib.sha256(path.read_bytes()).hexdigest()
        status = {}
        if not valid_cache:
            fig, axes = plt.subplots(4,3,figsize=(17.5,12.8))
            for ax, (_, record) in zip(axes.flat,page.iterrows()):
                rid = record.record_id
                safe = rid.replace('/','_')
                available = False
                try:
                    artifact = json.loads((root/(safe+'.json')).read_text())
                    if artifact.get('record_id')!=rid:
                        raise ValueError('Artifact record_id mismatch')
                    with np.load(root/(safe+'.npz'),allow_pickle=False) as arrays:
                        if 'median_xyz' in arrays:
                            xyz = np.array(arrays['median_xyz'])
                            time = np.array(arrays['time_ms'])
                            anchor,fs=artifact.get('median_anchor_sample'),float(artifact['fs'])
                            if xyz.shape!=(len(time),3) or not np.isfinite(xyz).all() or anchor is None:
                                raise ValueError('Invalid saved median')
                            if not np.allclose(time,(np.arange(len(time))-anchor)*1000/fs,rtol=0,atol=1e-7):
                                raise ValueError('Saved median time/fs/anchor mismatch')
                            boundaries = artifact.get('boundaries',{})
                            times = {}
                            for edge in edge_colors:
                                ix = boundaries.get(edge)
                                if ix is not None:
                                    if int(ix)!=ix or not 0<=ix<len(time):
                                        raise ValueError('Invalid saved boundary index:'+edge)
                                    times[edge]=float(time[int(ix)])
                                    csv_ms=record.get(edge+'_ms')
                                    if not pd.notna(csv_ms) or not np.isclose(csv_ms,times[edge],rtol=0,atol=1e-7):
                                        raise ValueError('CSV/artifact boundary mismatch:'+edge)
                            for j,color in enumerate(colors):
                                ax.plot(time,xyz[:,j],color=color,lw=.85)
                            for edge,value in times.items():
                                ax.axvline(value,color=edge_colors[edge],lw=.9,ls='--' if edge.endswith('on') else ':')
                            for loop,color in [('qrs','#77aed5'),('t','#efaa62')]:
                                a,z=times.get(loop+'_on'),times.get(loop+'_off')
                                if a is not None and z is not None and a<z:
                                    ax.axvspan(a,z,color=color,alpha=.20)
                            ax.axvline(0,color='.55',ls=':',lw=.6)
                            ax.set(xlim=(time[0],time[-1]),xlabel='ms so với QRS anchor',ylabel='XYZ (mV)')
                            ax.tick_params(labelsize=7.5); ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
                            ax.grid(alpha=.14)
                            available=True; status[rid]='median_displayed'
                        else:
                            reason=artifact.get('error') or artifact.get('failure_stage') or 'V1 không tạo được median beat'
                            ax.text(.5,.5,'KHÔNG CÓ MEDIAN V1\n'+'\n'.join(textwrap.wrap(str(reason),52)),ha='center',va='center',transform=ax.transAxes,fontsize=8,color='#964b30')
                            ax.set_xticks([]); ax.set_yticks([]); status[rid]='median_unavailable'
                except Exception as exc:
                    status[rid]='render_error:'+type(exc).__name__+': '+str(exc)
                    ax.text(.5,.5,'LỖI DỮ LIỆU/RENDER\n'+'\n'.join(textwrap.wrap(str(exc),52)),ha='center',va='center',transform=ax.transAxes,fontsize=8,color='#964b30')
                    ax.set_xticks([]); ax.set_yticks([])
                bounds = ' · '.join(edge.replace('qrs_','QRS-').replace('t_','T-')+'='+
                    (f'{record[edge+"_ms"]:g}' if pd.notna(record.get(edge+'_ms')) else 'thiếu')
                    for edge in ['qrs_on','qrs_off','t_on','t_off'])
                ax.set_title(f"{rid} · {record.label_name} · {record.split}\n{bounds} ms",fontsize=7.6)
            for ax in axes.flat[len(page):]:
                ax.axis('off')
            pages=int(np.ceil(len(ordered)/page_size))
            fig.suptitle(f'Median Frank XYZ và vùng V1 đã cắt — trang {number}/{pages}, ca {start+1}–{start+len(page)}/{len(ordered)}',fontsize=14)
            fig.text(.025,.012,'X: xanh dương; Y: đỏ; Z: xanh lá. Vùng xanh nhạt: QRS; cam nhạt: T. Đây là prediction V1 đã lưu, không phải xác nhận biên đúng. Chi tiết/zoom/manual review trong gallery.',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.95))
            fig.savefig(path,dpi=120,bbox_inches='tight',facecolor='white'); plt.close(fig)
            check_path.write_text(json.dumps({'fingerprint':fingerprint,'png_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'record_status':status},ensure_ascii=False,indent=2)+'\n')
        else:
            status=cache.get('record_status',{})
        rows.extend({'record_id':r.record_id,'page':number,'plot_path':str(path.relative_to(output_dir)),
            'display_status':status.get(r.record_id,'cached')} for r in page.itertuples(index=False))
    manifest=pd.DataFrame(rows)
    manifest.to_csv(output_dir/'contact_sheet_manifest.csv',index=False)
    if len(manifest)!=len(frame) or not manifest.record_id.is_unique or set(manifest.record_id)!=set(frame.record_id):
        raise ValueError('Contact-sheet coverage mismatch')
    return manifest

"""Inspectable figures for Notebook09; all loops are unchanged median XYZ cuts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from vcg_reference_metrics import load_median
from vcg_t_almeida import modulus_extrema

AXIS_COLORS=['#2563eb','#d97706','#7c3aed']
V1='#64748b';V2='#0f766e';ON='#0891b2';OFF='#be123c'


def load_case(out,record_id):
    out=Path(out);safe=record_id.replace('/','_')
    table=pd.read_csv(out/'v2_t_boundaries.csv')
    row=table.set_index('record_id').loc[record_id].to_dict();row['record_id']=record_id
    median=load_median(ROOT/'outputs/vcg_full_evaluation/artifacts',record_id)
    diag=json.loads((out/'diagnostics'/(safe+'.json')).read_text())
    with np.load(out/'artifacts'/(safe+'.npz'),allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    return dict(row=row,median=median,diag=diag,arrays=arrays)


def _new(rows=1,cols=1,size=None):
    return plt.subplots(rows,cols,figsize=size or (13,3*rows),squeeze=False,layout='constrained')


def _missing(case,title):
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    ax.axis('off');ax.text(.03,.72,title,fontsize=17,weight='bold',transform=ax.transAxes)
    missing=case['median']['xyz'] is None
    ax.text(.03,.45,case['row']['record_id']+(' — median unavailable' if missing else ' — requested intermediate unavailable'),fontsize=13,transform=ax.transAxes)
    import textwrap
    ax.text(.03,.20,'\n'.join(textwrap.wrap(str(case['row'].get('failure_reason','')),100)),fontsize=11,transform=ax.transAxes)
    return fig


def _mark(ax,case,old=True,new=True):
    r=case['row']
    for label,color in [('qrs_on_ms','#9ca3af'),('qrs_off_ms','#9ca3af')]:
        if pd.notna(r.get(label)): ax.axvline(r[label],color=color,ls=':',lw=1)
    if old:
        for b in ['t_on','t_off']:
            v=r.get(b+'_v1_ms')
            if pd.notna(v): ax.axvline(v,color=V1,ls='--',lw=1.4,label=b+' V1')
    if new:
        for b,color in [('t_on',ON),('t_off',OFF)]:
            v=r.get(b+'_v2_ms')
            if pd.notna(v): ax.axvline(v,color=color,lw=1.7,label=b+' V2')


def plot_median(case,comparison=False):
    if case['median']['xyz'] is None:return _missing(case,'Median XYZ / V1 vs V2')
    xyz,t=case['median']['xyz'],case['median']['time_ms']
    fig,axs=_new(3,1,(13,8))
    for j,axis in enumerate('XYZ'):
        ax=axs[j,0];ax.plot(t,xyz[:,j],color=AXIS_COLORS[j],lw=1.3,label=axis)
        _mark(ax,case,new=comparison)
        if case['diag'].get('search_window'):
            a,b=case['diag']['search_window'];ax.axvspan(t[a],t[b],color='#f59e0b',alpha=.07,label='T search I')
        if comparison:
            for version,color in [('v1',V1),('v2',V2)]:
                a,b=case['row'].get('t_on_'+version+'_ms'),case['row'].get('t_off_'+version+'_ms')
                if pd.notna(a) and pd.notna(b): ax.axvspan(a,b,color=color,alpha=.12)
        ax.axvline(0,color='#475569',lw=.7);ax.grid(alpha=.16);ax.set_ylabel(axis+' (mV)')
    axs[0,0].legend(ncol=4,fontsize=8,loc='upper right');axs[-1,0].set_xlabel('Time relative to saved QRS anchor (ms)')
    fig.suptitle(case['row']['record_id']+' | '+case['row']['status']+' | unchanged PQ-centered XYZ',fontsize=13)
    return fig


def plot_wavelets(case):
    if case['median']['xyz'] is None:return _missing(case,'Quadratic-spline wavelets')
    t=case['median']['time_ms'];d=case['diag'];arrays=case['arrays']
    fig,axs=_new(3,3,(16,10))
    window=d.get('search_window',[0,len(t)-1])
    for row,axis in enumerate('XYZ'):
        for col,scale in enumerate((3,4,5)):
            ax=axs[row,col];w=arrays['wavelet_scale'+str(scale)][:,row]
            ax.plot(t,w,color=AXIS_COLORS[row],lw=1)
            ax.axvspan(t[window[0]],t[window[1]],color='#f59e0b',alpha=.08)
            sl=d.get('single_leads',[])
            attempt=next((a for a in (sl[row]['attempts'] if sl else []) if a['wavelet_scale']==scale),None)
            if attempt:
                selected=[e for e in attempt['extrema'] if e['selected']]
                rejected=[e for e in attempt['extrema'] if not e['selected']]
                for ext,c,marker,label in [(selected,'#059669','o','Selected slopes'),(rejected,'#dc2626','x','Other extrema')]:
                    if ext: ax.scatter([t[e['sample']] for e in ext],[e['amplitude'] for e in ext],c=c,marker=marker,s=28,label=label,zorder=4)
                for key,ls in [('detection_threshold','--'),('significant_threshold',':')]:
                    if key in attempt:
                        value=attempt[key];ax.axhline(value,color='#6b7280',ls=ls,lw=.7,label=key);ax.axhline(-value,color='#6b7280',ls=ls,lw=.7)
            ax.set_title(f'{axis}, scale 2^{scale}'+(' (ML selected)' if scale==d.get('selected_scale') else ''),fontsize=10)
            ax.grid(alpha=.15)
            if row==2:ax.set_xlabel('ms relative to anchor')
            if col==0:ax.set_ylabel('WT (normalized filter units)')
    axs[0,1].legend(fontsize=6,loc='upper right')
    fig.suptitle(case['row']['record_id']+' | undecimated WT; scale is not a millisecond window',fontsize=13)
    return fig


def plot_single_leads(case):
    if case['median']['xyz'] is None:return _missing(case,'Single-lead initialization')
    t=case['median']['time_ms'];fig,axs=_new(3,1,(13,9))
    for j,s in enumerate(case['diag'].get('single_leads',[])):
        ax=axs[j,0];w=case['arrays']['wavelet_scale'+str(s['wavelet_scale'])][:,j]
        ax.plot(t,w,color=AXIS_COLORS[j],label=s['axis'])
        for k,color in [('first_significant_extremum','#d97706'),('last_significant_extremum','#7c3aed')]:
            p=s.get(k)
            if p is not None: ax.scatter(t[p],w[p],color=color,s=45,label=k)
        for key,color in [('t_on_candidate_sample',ON),('t_off_candidate_sample',OFF),('t_peak_candidate_sample','#111827')]:
            p=s.get(key)
            if p is not None:ax.axvline(t[p],color=color,ls='--',label=key)
        ax.set_title(f"{s['axis']} / scale{s['wavelet_scale']} / {s['morphology_type']} / {s['detection_status']}\n{s['failure_reason']}",fontsize=9)
        ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.15);ax.set_ylabel('WT')
    axs[-1,0].set_xlabel('ms relative to anchor');return fig


def _equal_3d(ax,points):
    a=np.asarray(points)
    if not len(a):return
    low,high=a.min(axis=0),a.max(axis=0);center=(high+low)/2
    radius=max(float(np.max(high-low))/2,1e-4)*1.08
    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
    ax.set_box_aspect((1,1,1));ax.view_init(24,-55)


def plot_wt_loop(case):
    if case['median']['xyz'] is None:return _missing(case,'Wavelet Transform Loop')
    fig=plt.figure(figsize=(10,8),layout='constrained');ax=fig.add_subplot(projection='3d')
    d=case['diag'];scale=d.get('selected_scale',4);w=case['arrays']['wavelet_scale'+str(scale)]
    a,b=d.get('search_window',[0,len(w)-1]);segment=w[a:b+1]
    ax.plot(*segment.T,color='#94a3b8',lw=1,label='WT loop in T search')
    for boundary,color in [('t_on',ON),('t_off',OFF)]:
        branch=d.get('branches',{}).get(boundary,{})
        if branch.get('chosen_iteration'):
            entry=branch['history'][branch['chosen_iteration']-1];start,end=entry['window_start'],entry['window_end']
            points=w[start:end+1];ax.plot(*points.T,color=color,lw=2,label=boundary+' TLS window')
    _equal_3d(ax,segment);ax.set(xlabel='Wx',ylabel='Wy',zlabel='Wz',title=f"{case['row']['record_id']} | WT loop at scale 2^{scale}")
    ax.legend(fontsize=8);return fig


def _entry(case,boundary,iteration=None):
    branch=case['diag'].get('branches',{}).get(boundary,{})
    if not branch.get('history'):return None
    it=iteration if iteration is not None else branch.get('chosen_iteration') or len(branch['history'])
    return branch['history'][max(0,min(len(branch['history'])-1,int(it)-1))]


def plot_tls_fit(case,boundary='t_on',iteration=None):
    if case['median']['xyz'] is None:return _missing(case,'TLS fit')
    entry=_entry(case,boundary,iteration)
    if not entry or not entry.get('direction'):
        return _missing(case,'TLS unavailable: '+case['diag'].get('failure_reason',''))
    scale=case['diag']['selected_scale'];w=case['arrays']['wavelet_scale'+str(scale)]
    points=w[entry['window_start']:entry['window_end']+1]
    fig=plt.figure(figsize=(10,8),layout='constrained');ax=fig.add_subplot(projection='3d')
    center=np.asarray(entry['center']);u=np.asarray(entry['direction']);scores=(points-center)@u
    line=center+np.array([scores.min(),scores.max()])[:,None]*u
    ax.scatter(*points.T,c=np.arange(len(points)),cmap='viridis',s=13,label='WT samples in W')
    ax.plot(*line.T,color=OFF,lw=2,label='Affine TLS line')
    ax.quiver(*center,*(u*np.ptp(scores)*.5),color=ON,arrow_length_ratio=.2)
    _equal_3d(ax,points);ax.set(xlabel='Wx',ylabel='Wy',zlabel='Wz')
    ax.set_title(f"{boundary} iteration {entry['iteration']} | u=({u[0]:+.3f}, {u[1]:+.3f}, {u[2]:+.3f})\nenergy fraction={entry['fit_quality']:.3f}; {entry.get('stop_reason','')}",fontsize=11)
    ax.legend(fontsize=8);return fig


def plot_derived(case,boundary='t_on',iteration=None):
    if case['median']['xyz'] is None:return _missing(case,'Derived wavelet')
    entry=_entry(case,boundary,iteration);fig,axs=_new(1,1,(13,5));ax=axs[0,0]
    if not entry or not entry.get('direction'):
        ax.text(.05,.5,'No TLS direction: '+case['diag'].get('failure_reason',''),transform=ax.transAxes);return fig
    scale=case['diag']['selected_scale'];w=case['arrays']['wavelet_scale'+str(scale)];t=case['median']['time_ms'];u=np.asarray(entry['direction'])
    for j,axis in enumerate('XYZ'):ax.plot(t,w[:,j],color=AXIS_COLORS[j],alpha=.55,lw=.8,label='W'+axis)
    derived=case['arrays'][boundary+'_derived_iterations'][entry['derived_index']]
    ax.plot(t,derived,color='#111827',lw=1.6,label='Wderived = W @ u')
    ax.axvspan(t[entry['window_start']],t[entry['window_end']],color=ON,alpha=.12,label='TLS fit window W')
    if entry.get('extremum_sample') is not None:
        p=entry['extremum_sample'];ax.scatter(t[p],derived[p],s=45,color=OFF,zorder=4,label='Relevant slope')
    if entry.get('candidate_sample') is not None:ax.axvline(t[entry['candidate_sample']],color=V2,lw=1.8,label='Boundary candidate')
    threshold=entry.get('boundary_threshold')
    if threshold is not None:ax.axhline(threshold,color=OFF,ls='--',lw=.8);ax.axhline(-threshold,color=OFF,ls='--',lw=.8)
    a,b=case['diag']['search_window'];ax.set_xlim(t[max(0,a-80)],t[min(len(t)-1,b+80)])
    ax.set_title(f"{boundary}, iteration {entry['iteration']}: {u[0]:+.4f} Wx {u[1]:+.4f} Wy {u[2]:+.4f} Wz\n{entry.get('stop_reason','')}",fontsize=11)
    ax.set(xlabel='ms relative to anchor',ylabel='WT');ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=4);return fig


def plot_iterations(case):
    if case['median']['xyz'] is None:return _missing(case,'TLS iteration history')
    fig,axs=_new(2,1,(12,7));t=case['median']['time_ms']
    for boundary,color in [('t_on',ON),('t_off',OFF)]:
        history=case['diag'].get('branches',{}).get(boundary,{}).get('history',[])
        ix=[r['iteration'] for r in history]
        candidate=[t[r['candidate_sample']] if r.get('candidate_sample') is not None else np.nan for r in history]
        amplitude=[r.get('extremum_amplitude',np.nan) for r in history]
        axs[0,0].plot(ix,candidate,'o-',color=color,label=boundary)
        axs[1,0].plot(ix,amplitude,'o-',color=color,label=boundary)
        for r in history:
            if r.get('stop_reason'):axs[0,0].text(.02,.95 if boundary=='t_on' else .86,boundary+': '+r['stop_reason'],transform=axs[0,0].transAxes,fontsize=8,va='top',color=color)
    for ax in axs[:,0]:ax.grid(alpha=.15);ax.legend(fontsize=8);ax.set_xlabel('Iteration (not time)')
    axs[0,0].set_ylabel('Candidate boundary (ms)');axs[1,0].set_ylabel('Relevant |WT| extremum');return fig


def plot_t_loops(case):
    if case['median']['xyz'] is None:return _missing(case,'T loops 3D')
    arrays=case['arrays'];loops=[arrays[k] for k in ('t_loop_v1','t_loop_v2') if k in arrays]
    fig=plt.figure(figsize=(15,10),layout='constrained')
    combined=np.vstack(loops) if loops else case['median']['xyz']
    for i,(key,title,color) in enumerate([('t_loop_v1','V1',V1),('t_loop_v2','V2',V2)]):
        ax=fig.add_subplot(2,3,i+1,projection='3d');loop=arrays.get(key)
        if loop is not None:
            ax.plot(*loop.T,color=color,lw=1.8);ax.scatter(*loop[0],color=color,marker='o',s=45,label='start');ax.scatter(*loop[-1],color=OFF,marker='x',s=55,label='end')
            k=max(0,len(loop)//2-1);delta=loop[min(len(loop)-1,k+max(1,len(loop)//12))]-loop[k]
            ax.quiver(*loop[k],*delta,color=color,arrow_length_ratio=.3)
        else:ax.text2D(.1,.5,'No valid T loop',transform=ax.transAxes)
        _equal_3d(ax,combined);ax.set(xlabel='X (mV)',ylabel='Y (mV)',zlabel='Z (mV)',title=title+' / same PQ origin and scale')
        if loop is not None:ax.legend(fontsize=8)
    info=fig.add_subplot(2,3,3);info.axis('off')
    info.text(0,.85,case['row']['record_id'],fontsize=12,weight='bold')
    info.text(0,.68,case['row']['status'],fontsize=13,color=V2)
    import textwrap
    reason=str(case['row'].get('failure_reason',''))
    if reason!='nan':info.text(0,.48,'\n'.join(textwrap.wrap(reason,37)),fontsize=9,va='top')
    for pos,(a,b) in enumerate([(0,1),(0,2),(1,2)],4):
        ax=fig.add_subplot(2,3,pos)
        for key,color,label in [('t_loop_v1',V1,'V1'),('t_loop_v2',V2,'V2')]:
            loop=arrays.get(key)
            if loop is not None:
                ax.plot(loop[:,a],loop[:,b],color=color,lw=1.5,label=label)
                ax.scatter(loop[0,a],loop[0,b],c=color,s=25);ax.scatter(loop[-1,a],loop[-1,b],c=color,marker='x',s=30)
        ax.set(xlabel='XYZ'[a]+' (mV)',ylabel='XYZ'[b]+' (mV)',title='XYZ'[a]+'XYZ'[b]+' projection')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.16)
        if loops:ax.legend(fontsize=8)
    return fig


def render_cohort_figures(results,out):
    out=Path(out);manifest=[]
    for i,row in enumerate(results.to_dict('records'),1):
        rid=row['record_id'];safe=rid.replace('/','_')
        paths={}
        for kind,func in [('comparison',lambda c:plot_median(c,comparison=True)),('loops',plot_t_loops)]:
            path=out/'figures'/(safe+'_'+kind+'.png');paths[kind]=str(path.relative_to(out))
            if not path.exists():
                case=load_case(out,rid)
                fig=func(case);fig.savefig(path,dpi=120);plt.close(fig)
        manifest.append(dict(record_id=rid,**paths))
        if i%25==0 or i==len(results): print(f'Figures {i}/{len(results)}',flush=True)
    pd.DataFrame(manifest).to_csv(out/'figures/manifest.csv',index=False)
    return manifest

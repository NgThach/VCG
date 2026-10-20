"""Fixed-landmark and end-to-end comparisons; measurements are not ground truth."""
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vcg_preprocessing import (Config, legacy_config, filter_frequency_response,
                               welch_xyz, _line_psd_metrics)
from vcg_pipeline import (process_record, spatial_speed, cut_loop, extract_features,
                         FEATURE_NAMES, qc_row)

COLORS={'A_legacy':'#2563eb','B_fir_auto':'#059669','C_unfiltered':'#ea580c'}
STYLES={'A_legacy':'--','B_fir_auto':'-','C_unfiltered':':'}


def compare_preprocessing_configs(record_path,cfg=Config()):
    configs={'A_legacy':legacy_config(cfg),'B_fir_auto':cfg,
             'C_unfiltered':replace(cfg,morphology_mode='none',notch_mode='off',morphology_highpass_hz=None)}
    results={};errors={}
    for name,config in configs.items():
        try:results[name]=process_record(record_path,config)
        except (ValueError,OSError) as exc:errors[name]=str(exc)
    if 'B_fir_auto' not in results:
        raise ValueError('Không có median beat B làm tham chiếu biên cố định: '+str(errors))
    reference=results['B_fir_auto'];rep=reference['rep'];bounds=reference['boundaries'];fs=reference['raw']['fs']
    # Freeze anchors, used-beat set, each PQ window, final PQ and QRS/T bounds.
    used=rep['beat_qc'][rep['beat_qc']['included']]
    pre=rep['anchor'];post=len(rep['xyz'])-pre-1;pq0,pq1=rep['pq']
    fixed={}
    for name,result in results.items():
        beats=[]
        for row in used.to_dict('records'):
            p=int(row['sample']);beat=result['morph'][p-pre:p+post+1].copy()
            a,z=int(row['pq_on']),int(row['pq_off'])
            beats.append(beat-np.median(beat[a:z],axis=0))
        xyz=np.median(np.stack(beats),axis=0);xyz-=np.median(xyz[pq0:pq1],axis=0)
        speed=spatial_speed(xyz,fs,cfg)
        qrs=cut_loop(xyz,bounds['qrs_on'],bounds['qrs_off'])
        tloop=cut_loop(xyz,bounds['t_on'],bounds['t_off'])
        speeds={label:None if loop is None else speed[bounds[a]:bounds[z]+1]
                for label,loop,a,z in [('QRS',qrs,'qrs_on','qrs_off'),('T',tloop,'t_on','t_off')]}
        fixed[name]={'xyz':xyz,'time_ms':rep['time_ms'],'qrs_loop':qrs,'t_loop':tloop,
                     'speed':speed,'features':extract_features(qrs,tloop,fs,cfg,speeds=speeds)}
    if not np.allclose(fixed['B_fir_auto']['xyz'],rep['xyz'],atol=1e-12,rtol=1e-10):
        raise RuntimeError('Fixed reconstruction does not match reference B; không báo so sánh giả.')
    values=[];changes=[]
    for kind,branches in [('fixed',fixed),('full',results)]:
        for name,result in branches.items():
            for feature,value in result['features'].items():
                baseline=branches.get('C_unfiltered',{}).get('features',{}).get(feature,np.nan)
                change=value-baseline
                values.append({'comparison':kind,'config':name,'feature':feature,'value':value})
                changes.append({'comparison':kind,'config':name,'feature':feature,'value':value,
                                'reference_C':baseline,'absolute_change':abs(change),
                                'relative_change_pct':100*change/abs(baseline) if np.isfinite(baseline) and abs(baseline)>1e-12 else np.nan})
    signal=[]
    for name,result in results.items():
        diag=result['preprocessing'];raw=result['raw']['xyz'];morph=result['morph']
        f,psd=welch_xyz(morph,fs,cfg);line=_line_psd_metrics(f,psd,cfg)
        hf=(f>=200)&(f<=min(450,fs/2))
        for j,axis in enumerate('XYZ'):
            signal.append({'config':name,'axis':axis,'rms_difference_uV':float(np.sqrt(np.mean((morph[:,j]-raw[:,j])**2))*1000),
                           '50_band_density_after':float(line[0][j]) if line is not None else np.nan,
                           'high_frequency_density_after':float(np.mean(psd[hf,j])) if hf.any() else np.nan,
                           'notch_applied':diag['notch_applied'],'powerline_status':diag['powerline_status']})
    return {'results':results,'fixed':fixed,'reference':reference,'errors':errors,
            'feature_values':pd.DataFrame(values),'feature_changes':pd.DataFrame(changes),
            'signal_metrics':pd.DataFrame(signal),
            'full_qc':pd.DataFrame([{'config':name,**qc_row(r)} for name,r in results.items()])}


def powerline_diagnostic_table(diagnostics):
    rows=[]
    for name,info in diagnostics['powerline']['channels'].items():
        rows.append({'axis':name,**info,'notch_on_axis':name in diagnostics['notch_channels']})
    return pd.DataFrame(rows)


def plot_filter_responses(fs,cfg=Config()):
    table=filter_frequency_response(fs,cfg);f=table.frequency_hz
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    axes[0,0].plot(f,10**(table.fir_one_pass_db/20),label='FIR một chiều')
    axes[0,0].plot(f,10**(table.fir_two_pass_db/20),ls='--',label='FIR hai chiều thực dùng')
    axes[0,0].axvspan(0,cfg.fir_preserve_hz,color='#059669',alpha=.10)
    axes[0,0].set(xlim=(0,230),ylim=(-.03,1.08),ylabel='Gain biên độ',title='FIR: dải giữ và vùng chuyển tiếp')
    for key,label in [('fir_one_pass_db','FIR một chiều'),('fir_two_pass_db','FIR hai chiều')]:
        axes[0,1].plot(f,table[key],label=label)
    axes[0,1].axvline(cfg.fir_cutoff_hz,color='.4',ls=':')
    axes[0,1].set(xlim=(130,220),ylim=(-220,2),ylabel='Gain (dB)',title='175 Hz là điểm thiết kế, không là mép giữ nguyên')
    for key,label in [('notch_one_pass_db','Notch một chiều'),('notch_two_pass_db','Notch hai chiều')]:
        axes[1,0].plot(f,10**(table[key]/20),label=label)
        axes[1,1].plot(f,table[key],label=label)
    axes[1,0].set(xlim=(45,55),ylim=(-.03,1.08),ylabel='Gain biên độ',title=f'Notch {cfg.notch_hz:g} Hz; Q={cfg.notch_q:g}')
    axes[1,1].axhline(-3,color='.5',ls=':')
    axes[1,1].set(xlim=(47,53),ylim=(-90,2),ylabel='Gain (dB)',title='Dải suy giảm sau hai chiều rộng hơn một chiều')
    for ax in axes.flat:ax.set_xlabel('Tần số (Hz)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    return fig,table


def plot_preprocessing_diagnostics(raw,morph,fs,diagnostics,cfg=Config()):
    """Yield one figure at a time so pyplot.show does not repeat open figures."""
    before=diagnostics['input_quality'];f=before['psd_frequency_hz'];p=before['psd_density']
    after=diagnostics['psd_after_density'];fa=diagnostics['psd_after_frequency_hz']
    fig,axes=plt.subplots(3,2,figsize=(13,9),layout='constrained')
    for j,name in enumerate('XYZ'):
        for col in range(2):
            ax=axes[j,col]
            ax.semilogy(f,np.maximum(p[:,j],1e-20),color='.45',lw=2,label='Gốc')
            ax.semilogy(fa,np.maximum(after[:,j],1e-20),color='#2563eb',lw=1,ls='--',label='Sau morphology')
            ax.axvline(cfg.notch_hz,color='#ea580c',ls=':',lw=1)
            ax.set(xlim=(0,250) if col==0 else (42,58),ylabel=name+' PSD (mV²/Hz)')
            ax.grid(alpha=.2)
            if j==0:ax.legend(fontsize=9)
            if j==2:ax.set_xlabel('Tần số (Hz)')
    fig.suptitle('Welch PSD trước/sau; cùng grid. Phổ cao không tự là nhiễu.')
    yield fig
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    w=diagnostics['powerline']['windows']
    for ax,name in zip(axes,'XYZ'):
        info=diagnostics['powerline']['channels'][name]
        if not w.empty:
            data=w[w.channel==name];ax.plot(data.center_s,data.score_db,marker='o',label='R_dB từng cửa sổ')
        ax.axhline(cfg.powerline_score_threshold_db,color='#dc2626',ls='--',label='Ngưỡng thử nghiệm')
        ax.set(ylabel=name+' (dB)',title=name+': '+info['status'])
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    axes[-1].set_xlabel('Tâm cửa sổ thời gian (s)')
    fig.suptitle('Power-line persistence: đỉnh PSD là một phần bằng chứng')
    yield fig
    fig,axes=plt.subplots(3,2,figsize=(13,8),sharex=True,layout='constrained')
    n=min(len(raw),int(4*fs));tt=np.arange(n)/fs
    for j,name in enumerate('XYZ'):
        axes[j,0].plot(tt,raw[:n,j],color='.45',lw=2,label='Gốc')
        axes[j,0].plot(tt,morph[:n,j],color='#2563eb',lw=1,ls='--',label='Sau morphology')
        axes[j,1].plot(tt,(morph[:n,j]-raw[:n,j])*1000,color='#7c3aed')
        axes[j,0].set_ylabel(name+' (mV)');axes[j,1].set_ylabel(name+' sai khác (µV)')
        for col in range(2):axes[j,col].grid(alpha=.2)
        if j==0:axes[j,0].legend(fontsize=9);axes[j,1].set_title('Sai khác không tự là nhiễu đã loại')
        if j==2:
            for col in range(2):axes[j,col].set_xlabel('Thời gian recording (s)')
    fig.suptitle('XYZ trước/sau và sai khác trên cùng khoảng thời gian')
    yield fig
    runs=diagnostics['powerline']['quiet_intervals']
    fig,axes=plt.subplots(3,1,figsize=(12,6),sharex=True,layout='constrained')
    if runs:
        lo,hi=runs[len(runs)//2];a=max(0,lo-int(.1*fs));z=min(len(raw),hi+int(.1*fs))
        time=np.arange(a,z)/fs
        for j,ax in enumerate(axes):
            ax.plot(time,raw[a:z,j],color='.35')
            ax.axvspan(lo/fs,hi/fs,color='#059669',alpha=.15,label='Đoạn ít hoạt động ứng viên')
            ax.set_ylabel('XYZ'[j]+' (mV)');ax.grid(alpha=.2)
        axes[0].legend(fontsize=9)
    else:
        for ax in axes:ax.text(.5,.5,'Chưa có vùng ít hoạt động đủ dài; không ép kết luận.',ha='center',transform=ax.transAxes)
    axes[-1].set_xlabel('Thời gian recording (s)')
    fig.suptitle('Kiểm tra vùng ít hoạt động; không mặc định mọi đoạn giữa QRS là yên')
    yield fig


def plot_median_config_comparison(comparison,kind='fixed'):
    reference=comparison['reference'];bounds=reference['boundaries']
    fig,axes=plt.subplots(4,1,figsize=(13,10),sharex=True,layout='constrained')
    for name,item in (comparison['fixed'] if kind=='fixed' else comparison['results']).items():
        xyz=item['xyz'] if kind=='fixed' else item['rep']['xyz']
        time=item['time_ms'] if kind=='fixed' else item['rep']['time_ms']
        speed=item['speed'] if kind=='fixed' else item['boundaries']['speed']
        for j in range(3):axes[j].plot(time,xyz[:,j],color=COLORS[name],ls=STYLES[name],lw=1.2,label=name)
        axes[3].plot(time,speed,color=COLORS[name],ls=STYLES[name],lw=1.2,label=name)
    time=reference['rep']['time_ms']
    for ax in axes:
        for a,z,color in [('qrs_on','qrs_off','#059669'),('t_on','t_off','#ea580c')]:
            if bounds[a] is not None and bounds[z] is not None:ax.axvspan(time[bounds[a]],time[bounds[z]],color=color,alpha=.08)
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3)
    for j in range(3):axes[j].set_ylabel('XYZ'[j]+' (mV)')
    axes[3].set_ylabel('Tốc độ (mV/s)');axes[-1].set_xlabel('Thời gian so với mốc căn nhịp (ms)')
    fig.suptitle('Biên/beat/PQ cố định của B' if kind=='fixed' else 'Chạy lại mỗi pipeline: mốc và baseline có thể khác')
    return fig


def plot_3d_config_comparison(comparison,kind='fixed'):
    branches=comparison['fixed'] if kind=='fixed' else comparison['results']
    fig=plt.figure(figsize=(13,6));fig.subplots_adjust(left=.04,right=.93,wspace=.2,top=.85,bottom=.12)
    for position,key,title in [(1,'qrs_loop','QRS'),(2,'t_loop','T')]:
        ax=fig.add_subplot(1,2,position,projection='3d');points=[]
        for name,item in branches.items():
            loop=item[key]
            if loop is None:continue
            points.append(loop);ax.plot(*loop.T,color=COLORS[name],ls=STYLES[name],lw=1.5,label=name)
        if points:
            values=np.vstack(points);center=(values.max(axis=0)+values.min(axis=0))/2
            radius=max(float(np.ptp(values,axis=0).max()/2),1e-3)*1.15
            ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_box_aspect((1,1,1));ax.legend(fontsize=8)
        ax.set(xlabel='X (mV)',ylabel='Y (mV)',title=title)
        ax.text2D(1.08,.6,'Z (mV)',transform=ax.transAxes,rotation=90,clip_on=False)
    fig.suptitle('So sánh loops 3D — '+kind+'; mỗi ô dùng chung thang cho A/B/C')
    return fig

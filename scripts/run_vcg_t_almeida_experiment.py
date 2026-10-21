#!/usr/bin/env python3
"""Read-only V1 -> independent T-WT/TLS experiment, checkpoints and reporting."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("MPLCONFIGDIR", str(__import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'vcg_t_almeida_mpl'))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from vcg_preprocessing import Config as V1Config
from vcg_pipeline import FEATURE_NAMES, cut_loop, extract_features, spatial_speed
from vcg_t_almeida import AlmeidaConfig, METHOD_VERSION, METHOD_CLASS, config_id, delineate_t, json_safe
from vcg_reference_metrics import load_median, validate_annotations

DEFAULT_OUT = ROOT / "outputs/vcg_t_almeida_v2_experiment"
FEATURE_INPUT = ROOT / "outputs/vcg_12_features_train_valid_test"
ARTIFACT_INPUT = ROOT / "outputs/vcg_full_evaluation/artifacts"


def sha(path):
    from project_integrity import file_digest
    return file_digest(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement of this experiment's files only.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, indent=2)+"\n")
    tmp.replace(path)


def safe_id(record_id):
    return record_id.replace("/", "_")


def load_v1_inputs(input_dir=FEATURE_INPUT):
    input_dir = Path(input_dir)
    frames = [pd.read_csv(input_dir / (s+"_features.csv")) for s in ("train", "valid", "test")]
    frame = pd.concat(frames, ignore_index=True).sort_values("record_id").reset_index(drop=True)
    manifest = pd.read_csv(input_dir/"split_manifest.csv")
    if not frame.record_id.is_unique or not manifest.record_id.is_unique or set(frame.record_id) != set(manifest.record_id):
        raise ValueError("V1 table/manifest record identity mismatch")
    m = manifest.set_index("record_id").loc[frame.record_id]
    for key in ("patient_id", "split", "label_name"):
        if not np.array_equal(frame[key].to_numpy(), m[key].to_numpy()):
            raise ValueError("V1 manifest mismatch: "+key)
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("Patient leakage in V1 manifest")
    source_cfg = json.loads((input_dir/"config.json").read_text())
    for name, digest in source_cfg["source_sha256"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("V1 source differs from frozen run: "+name)
    return frame, manifest, V1Config(**source_cfg["config"])


def protected_files():
    files = [p for p in ROOT.glob("*.ipynb") if p.name[:2] in ("05", "06", "07", "08")]
    for name in ("vcg_pipeline.py", "vcg_preprocessing.py", "vcg_preprocessing_evaluation.py",
                 "vcg_12_features.csv", "vcg_processing_qc.csv", "vcg_processing_failures.csv"):
        files.append(ROOT/name)
    for name in ("outputs/vcg_12_features_train_valid_test", "outputs/vcg_full_evaluation",
                 "outputs/vcg_qc_audit_v1", "outputs/vcg_boundary_reference_eval_v1"):
        files.extend(p for p in (ROOT/name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(set(files))


def freeze_v1(out):
    path = Path(out)/"protected_v1_hashes.json"
    if path.exists():
        hashes = json.loads(path.read_text())
        verify_v1(hashes)
    else:
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected_files()}
        write_json(path, hashes)
    return hashes


def verify_v1(hashes):
    changed = [name for name,digest in hashes.items() if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    if changed:
        raise RuntimeError("Protected V1 changed: "+", ".join(changed[:8]))
    return dict(protected_files=len(hashes), changed_files=0, v1_unchanged=True)


def initialize(out=DEFAULT_OUT, cfg=AlmeidaConfig()):
    out = Path(out).resolve()
    # Reject destinations that could overwrite any previous pipeline's files.
    protected_roots = [FEATURE_INPUT.resolve(), ARTIFACT_INPUT.parent.resolve(),
        (ROOT/'outputs/vcg_qc_audit_v1').resolve(),
        (ROOT/'outputs/vcg_boundary_reference_eval_v1').resolve(),
        (ROOT/'content').resolve(),(ROOT/'backups').resolve()]
    if out == ROOT or any(out == p or p in out.parents or out in p.parents for p in protected_roots):
        raise ValueError("Output destination overlaps protected V1")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("artifacts", "checkpoints", "diagnostics", "figures", "gallery", "reports", "logs"):
        (out/name).mkdir(exist_ok=True)
    hashes = freeze_v1(out)
    frame, manifest, v1cfg = load_v1_inputs()
    manifest_copy = out/"split_manifest.csv"
    original = (FEATURE_INPUT/"split_manifest.csv").read_bytes()
    if manifest_copy.exists() and manifest_copy.read_bytes() != original:
        raise ValueError("Experiment manifest changed; no implicit split replacement")
    manifest_copy.write_bytes(original)
    engine_hash = sha(ROOT/"vcg_t_almeida.py")
    fingerprint = hashlib.sha256((engine_hash+config_id(cfg)+sha(FEATURE_INPUT/"split_manifest.csv")).encode()).hexdigest()
    existing = out/"config.json"
    if existing.exists() and json.loads(existing.read_text()).get("algorithm_fingerprint") != fingerprint:
        raise ValueError("Frozen algorithm/config differs; use a separate output directory")
    config = dict(method=METHOD_VERSION, method_class=METHOD_CLASS, config=asdict(cfg),
        config_id=config_id(cfg), algorithm_fingerprint=fingerprint,
        locked_before_test=True, selection_or_tuning_on_test=False,
        generated_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),
        v1_feature_root=str(FEATURE_INPUT), v1_artifact_root=str(ARTIFACT_INPUT),
        source_v1_config=asdict(v1cfg), time_reference="median_relative_ms",
        sample_indices="zero-based median-local; never original recording indices",
        channels=["vx", "vy", "vz"], units="mV", loop_slice="xyz[on:off+1]",
        median_preprocessing="V1 FIR 401 taps/cutoff175 at1000Hz; conditional50Hz notch; per-beat and median PQ centering",
        wavelet_signal="unchanged V1 median; no extra signal filter/resample/recentering",
        numerical_sources={"engine":engine_hash,"v1_pipeline":sha(ROOT/"vcg_pipeline.py")},
        split_manifest_sha256=sha(manifest_copy), reference_policy="accepted independent compatible annotations only",
        versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","matplotlib","wfdb","nbformat","nbclient")},
        notebook08_available=[p.name for p in ROOT.glob("08*.ipynb")],
        author_code_commit="c8e3de47c54a9214138143676d2aa546b0540dd2",
        scientific_limits=["not a bit-exact author-code port", "median RR replaces recursive RR",
            "RMS uses available post-anchor median instead of full RR", "polyphase kernel resampling is a project interpolation",
            "no compatible independent reference has yet been accepted; disagreement is not accuracy"])
    write_json(existing,config)
    return frame,manifest,v1cfg,hashes,config


def feature_values(median, row, v2, cfg):
    """Keep QRS values and calculate T with the SAME full-median SG speed."""
    original = {k:float(row[k]) if pd.notna(row[k]) else np.nan for k in FEATURE_NAMES}
    values = dict(original)
    for key in FEATURE_NAMES[7:]: values[key] = np.nan
    if median["xyz"] is None:
        return values, dict(status="no_median"), {}
    xyz, fs, b = median["xyz"], median["fs"], median["boundaries"]
    with np.load(median["npz_path"],allow_pickle=False) as saved:
        speed = saved["speed"].copy() if "speed" in saved else spatial_speed(xyz,fs,cfg)
    recomputed_speed = spatial_speed(xyz,fs,cfg)
    if not np.allclose(speed,recomputed_speed,rtol=0,atol=1e-10):
        raise ValueError("Saved V1 full-median spatial speed not reproducible")
    qrs = cut_loop(xyz,b.get("qrs_on"),b.get("qrs_off"))
    old_t = cut_loop(xyz,b.get("t_on"),b.get("t_off"))
    speeds = {"QRS":None if qrs is None else speed[b['qrs_on']:b['qrs_off']+1],
              "T":None if old_t is None else speed[b['t_on']:b['t_off']+1]}
    baseline = extract_features(qrs,old_t,fs,cfg,speeds=speeds)
    for k in FEATURE_NAMES:
        if not np.isclose(original[k],baseline[k],rtol=1e-10,atol=1e-12,equal_nan=True):
            raise ValueError("V1 feature parity failed: "+k)
    loops = {}
    if qrs is not None: loops["qrs_loop_v1_unchanged"] = qrs
    if old_t is not None: loops["t_loop_v1"] = old_t
    if v2.get("t_on_sample") is not None and v2.get("t_off_sample") is not None:
        new_t = cut_loop(xyz,v2["t_on_sample"],v2["t_off_sample"])
        if new_t is not None:
            loops["t_loop_v2"] = new_t
            speeds["T"] = speed[v2["t_on_sample"]:v2["t_off_sample"]+1]
            fresh = extract_features(qrs,new_t,fs,cfg,speeds=speeds)
            for k in FEATURE_NAMES[7:]: values[k] = fresh[k]
    return values,dict(status="verified", max_full_speed_difference=float(np.max(abs(speed-recomputed_speed))),
                        feature_checks=len(FEATURE_NAMES)),loops


def _diagnostic(v2):
    diag = {k:v for k,v in v2.items() if k != "wavelets"}
    diag["branches"] = {k:{a:b for a,b in v.items() if a != "derived_history"}
                        for k,v in v2.get("branches",{}).items()}
    return diag


def process_one(row, out, cfg, v1cfg, fingerprint, resume=True):
    out = Path(out)
    rid, safe = row["record_id"], safe_id(row["record_id"])
    median = load_median(ARTIFACT_INPUT,rid)
    row_hash = hashlib.sha256(json.dumps(json_safe(row),sort_keys=True,default=str).encode()).hexdigest()
    identity = dict(algorithm_fingerprint=fingerprint, median_npz_sha256=median["signal_sha256"],
        median_json_sha256=median["artifact_sha256"], source_row_sha256=row_hash)
    checkpoint = out/"checkpoints"/(safe+".json")
    diag_path = out/"diagnostics"/(safe+".json")
    array_path = out/"artifacts"/(safe+".npz")
    if resume and checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior.get("identity") != identity:
            raise ValueError("Checkpoint identity mismatch: "+rid)
        if diag_path.is_file() and array_path.is_file() and sha(diag_path)==prior.get("diagnostics_sha256") and sha(array_path)==prior.get("arrays_sha256"):
            return prior["result"],json.loads(diag_path.read_text()),True
    error, trace = "", ""
    try:
        if median["xyz"] is None:
            v2 = dict(status="unresolved", failure_reason="MISSING_V1_MEDIAN: "+median["artifact"].get("error",""),
                branches={},single_leads=[],review_flags=[],t_on_sample=None,t_off_sample=None,
                t_on_ms=None,t_off_ms=None,t_duration_ms=None)
        else:
            b = median["boundaries"]
            # Deliberately pass ONLY QRS marks. V1 T boundaries are not inputs.
            v2 = delineate_t(median["xyz"],median["fs"],median["anchor"],b["qrs_on"],b["qrs_off"],
                             float(np.median(median["artifact"]["rr_seconds"])),cfg)
        features,parity,loops = feature_values(median,row,v2,v1cfg)
    except Exception as exc:
        error = type(exc).__name()+": "+str(exc)
        trace = traceback.format_exc()
        v2 = dict(status="unresolved",failure_reason="EXPERIMENT_EXCEPTION: "+error,branches={},
                  single_leads=[],review_flags=[],t_on_sample=None,t_off_sample=None,
                  t_on_ms=None,t_off_ms=None,t_duration_ms=None)
        features = {k:(row[k] if k.startswith("QRS_") else np.nan) for k in FEATURE_NAMES}
        parity,loops = dict(status="exception"),{}
    # Access V1 T values for the first time AFTER independent numerical V2.
    result = {k:row[k] for k in ("record_id","patient_id","split","label_name","label")}
    b = median["boundaries"] or {}
    result.update(method=METHOD_VERSION,method_class=METHOD_CLASS,config_id=config_id(cfg),
        algorithm_fingerprint=fingerprint,status=v2["status"],failure_reason=v2["failure_reason"],
        review_flags="; ".join(v2["review_flags"]),median_present=median["xyz"] is not None,
        fs=median["fs"],anchor_sample=median["anchor"],time_reference="median_relative_ms",units="mV",
        n_samples=len(median["xyz"]) if median["xyz"] is not None else 0,
        qrs_on_sample=b.get("qrs_on"),qrs_off_sample=b.get("qrs_off"),
        qrs_on_ms=row.get("qrs_on_ms"),qrs_off_ms=row.get("qrs_off_ms"),
        t_on_v1_sample=b.get("t_on"),t_off_v1_sample=b.get("t_off"),
        t_on_v1_ms=row.get("t_on_ms"),t_off_v1_ms=row.get("t_off_ms"),
        t_duration_v1_ms=row.get("t_duration_ms"),
        t_on_v2_sample=v2.get("t_on_sample"),t_off_v2_sample=v2.get("t_off_sample"),
        t_on_v2_ms=v2.get("t_on_ms"),t_off_v2_ms=v2.get("t_off_ms"),t_duration_v2_ms=v2.get("t_duration_ms"),
        selected_scale=v2.get("selected_scale"),v1_qc_status=row.get("qc_status"),v1_qc_flags=row.get("qc_flags"),
        v1_median_npz_sha256=median["signal_sha256"],v1_median_json_sha256=median["artifact_sha256"],
        v1_median_npz_path=median["npz_path"],v1_median_json_path=median["json_path"],
        v1_feature_parity=parity["status"],error=error,**features)
    for boundary in ("t_on","t_off"):
        branch=v2.get("branches",{}).get(boundary,{})
        result[boundary+"_stop_reason"]=branch.get("stop_reason","")
        result[boundary+"_iterations"]=len(branch.get("history",[]))
        result[boundary+"_converged"]=branch.get("converged",False)
    arrays = {**loops}
    for scale,coeff in v2.get("wavelets",{}).items(): arrays["wavelet_scale"+str(scale)] = coeff
    for boundary,branch in v2.get("branches",{}).items():
        if len(branch["derived_history"]): arrays[boundary+"_derived_iterations"] = branch["derived_history"]
    # An explicit empty NPZ is valid for missing median; never a fake loop.
    np.savez_compressed(array_path,**arrays)
    diag = _diagnostic(v2)
    diag.update(record_id=rid,patient_id=row["patient_id"],split=row["split"],identity=identity,
                feature_parity=parity,exception_traceback=trace)
    write_json(diag_path,diag)
    write_json(checkpoint,dict(identity=identity,result=result,diagnostics_sha256=sha(diag_path),arrays_sha256=sha(array_path)))
    return json_safe(result),json_safe(diag),False


def build_comparison(frame, results):
    merged=frame[["record_id",*FEATURE_NAMES]].merge(results,on="record_id",validate="one_to_one",suffixes=("_v1","_v2"))
    feature_rows=[]
    for row in merged.to_dict("records"):
        for feature in FEATURE_NAMES:
            old,new=row[feature+"_v1"],row[feature+"_v2"]
            delta=new-old if pd.notna(old) and pd.notna(new) else np.nan
            feature_rows.append({k:row[k] for k in ("record_id","patient_id","split","label_name","status","config_id")}|dict(
                feature=feature,v1_value=old,v2_value=new,signed_difference=delta,
                absolute_difference=abs(delta),relative_difference=delta/abs(old) if pd.notna(old) and abs(old)>1e-9 else np.nan,
                reason="QRS preserved exactly" if feature.startswith("QRS_") else row["failure_reason"]))
    comparison=results.copy()
    for boundary in ("t_on","t_off"):
        comparison[boundary+"_v2_minus_v1_ms"]=pd.to_numeric(comparison[boundary+"_v2_ms"],errors="coerce")-pd.to_numeric(comparison[boundary+"_v1_ms"],errors="coerce")
    return comparison,pd.DataFrame(feature_rows)


def reference_evaluation(frame,comparison,out,reference_csv=None):
    path=Path(reference_csv) if reference_csv else ROOT/"outputs/vcg_boundary_reference_eval_v1/reference_annotations.csv"
    if path.exists():
        annotations=pd.read_csv(path)
        accepted,audit=validate_annotations(annotations,frame,ARTIFACT_INPUT,allow_test=True)
    else:
        accepted=pd.DataFrame(columns=["record_id","boundary","reference_ms"])
        audit=pd.DataFrame(columns=["record_id","rejection_reason"])
    accepted.to_csv(out/"reports/accepted_references.csv",index=False)
    audit.to_csv(out/"reports/reference_acceptance_audit.csv",index=False)
    mapping={(r.record_id,r.boundary):r.reference_ms for r in accepted.itertuples()}
    rows=[]
    for r in comparison.to_dict("records"):
        for boundary in ("t_on","t_off"):
            ref=mapping.get((r['record_id'],boundary),np.nan)
            for version in ("v1","v2"):
                pred=r[boundary+"_"+version+"_ms"]
                paired=pd.notna(pred) and pd.notna(ref)
                rows.append(dict(record_id=r['record_id'],patient_id=r['patient_id'],split=r['split'],
                    label_name=r['label_name'],boundary=boundary,version=version,predicted_ms=pred,
                    reference_ms=ref,error_ms=pred-ref if paired else np.nan,
                    reference_status="ACCEPTED_INDEPENDENT_REFERENCE" if pd.notna(ref) else "GROUND_TRUTH_UNAVAILABLE"))
    per=pd.DataFrame(rows)
    per.to_csv(out/"reports/reference_error_per_record.csv",index=False)
    stats=[]
    for (boundary,version),g in per.groupby(["boundary","version"]):
        e=pd.to_numeric(g.error_ms,errors="coerce").dropna().to_numpy()
        stats.append(dict(boundary=boundary,version=version,n_reference=int(g.reference_ms.notna().sum()),
            n_paired=len(e),mean_signed_error_ms=float(e.mean()) if len(e) else np.nan,
            mae_ms=float(np.mean(abs(e))) if len(e) else np.nan,
            rmse_ms=float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
            sd_error_ms=float(e.std(ddof=1)) if len(e)>1 else np.nan))
    pd.DataFrame(stats).to_csv(out/"reports/reference_error_summary.csv",index=False)
    return dict(source=str(path),accepted_reference_boundaries=len(accepted),
        accepted_t_boundaries=sum(accepted.boundary.isin(["t_on","t_off"])) if len(accepted) else 0,
        ground_truth_available=bool(any(pd.notna(v) for v in mapping.values())),
        accuracy_claim="Unavailable without accepted compatible independent reference")


def write_outputs(frame,results,diagnostics,out,config,hashes,reference_csv=None):
    out=Path(out)
    results=pd.DataFrame(results).sort_values("record_id").reset_index(drop=True)
    diag_by_id={d['record_id']:d for d in diagnostics}
    # Preserve the detailed branch cause alongside the cohort-level status.
    for boundary in ('t_on','t_off'):
        reasons=[]
        for row in results.to_dict('records'):
            if pd.notna(row.get(boundary+'_v2_sample')):
                reasons.append('')
                continue
            diag=diag_by_id[row['record_id']]
            branch=diag.get('branches',{}).get(boundary,{})
            history=branch.get('history',[])
            last=history[-1] if history else {}
            wave_reason=last.get('failure_reason') or last.get('wave',{}).get('failure_reason','')
            reasons.append('; '.join(str(v) for v in [branch.get('stop_reason'),wave_reason] if v)
                           or diag.get('failure_reason',''))
        results[boundary+'_failure_reason']=reasons
    for column in ["qrs_on_ms","qrs_off_ms","t_on_v1_ms","t_off_v1_ms","t_duration_v1_ms",
                   "t_on_v2_sample","t_off_v2_sample","t_on_v2_ms","t_off_v2_ms","t_duration_v2_ms",*FEATURE_NAMES]:
        results[column]=pd.to_numeric(results[column],errors="coerce")
    results.to_csv(out/"v2_t_boundaries.csv",index=False)
    results[results.status.eq('unresolved')].to_csv(out/"diagnostics/failures.csv",index=False)
    compare,features=build_comparison(frame,results)
    compare.to_csv(out/"v1_v2_boundary_comparison.csv",index=False)
    features.to_csv(out/"v1_v2_feature_comparison.csv",index=False)
    sl_rows,iteration_rows,extrema_rows=[],[],[]
    for diag in diagnostics:
        meta={k:diag[k] for k in ("record_id","patient_id","split")}
        for s in diag.get("single_leads",[]):
            sl_rows.append(meta|{k:v for k,v in s.items() if k not in ('attempts','extrema','selected_extrema','peak_sources','t_peak_candidates','boundary_criteria','rms_interval')})
            for attempt in s['attempts']:
                for extremum in attempt['extrema']:
                    extrema_rows.append(meta|dict(axis=s['axis'],wavelet_scale=attempt['wavelet_scale'],**extremum))
        for boundary,branch in diag.get("branches",{}).items():
            for entry in branch['history']:
                direction=entry.get('direction',[None]*3)
                iteration_rows.append(meta|{k:v for k,v in entry.items() if k not in ('wave','direction','center','singular_values')}|dict(
                    ux=direction[0],uy=direction[1],uz=direction[2],
                    singular_values=json.dumps(entry.get('singular_values')),center=json.dumps(entry.get('center'))))
    sl_table=pd.DataFrame(sl_rows) if sl_rows else pd.DataFrame(columns=['record_id','patient_id','split','axis','wavelet_scale','morphology_type','rms'])
    sl_table.to_csv(out/"diagnostics/single_lead_candidates.csv",index=False)
    iteration_table=pd.DataFrame(iteration_rows) if iteration_rows else pd.DataFrame(columns=['record_id','boundary','iteration','status'])
    iteration_table.to_csv(out/"diagnostics/tls_iteration_history.csv",index=False)
    extrema_table=pd.DataFrame(extrema_rows) if extrema_rows else pd.DataFrame(columns=['record_id','axis','wavelet_scale','sample','amplitude'])
    extrema_table.to_csv(out/"diagnostics/wavelet_extrema.csv",index=False)
    bank=next((d['filter_bank'] for d in diagnostics if 'filter_bank' in d),{})
    write_json(out/"diagnostics/wavelet_filter_bank.json",bank)
    ref=reference_evaluation(frame,compare,out,reference_csv)
    by=[]
    for group_column in ('label_name','split'):
        for group,g in results.groupby(group_column):
            by.append(dict(group_by=group_column,group=group,recordings=len(g),medians=int(g.median_present.sum()),
                t_on_available=int(g.t_on_v2_ms.notna().sum()),t_off_available=int(g.t_off_v2_ms.notna().sum()),
                both_t_boundaries=int((g.t_on_v2_ms.notna()&g.t_off_v2_ms.notna()).sum()),
                detected=int(g.status.eq('detected').sum()),needs_review=int(g.status.eq('needs_review').sum()),
                unresolved=int(g.status.eq('unresolved').sum())))
    pd.DataFrame(by).to_csv(out/"reports/coverage_by_label_and_split.csv",index=False)
    differences=[]
    for (label,split),g in compare.groupby(['label_name','split']):
        for boundary in ('t_on','t_off'):
            vals=pd.to_numeric(g[boundary+'_v2_minus_v1_ms'],errors='coerce').dropna()
            differences.append(dict(label_name=label,split=split,boundary=boundary,n_paired=len(vals),
                median_v2_minus_v1_ms=float(vals.median()) if len(vals) else np.nan,
                median_absolute_disagreement_ms=float(vals.abs().median()) if len(vals) else np.nan,
                interpretation='Disagreement only; not reference error'))
    pd.DataFrame(differences).to_csv(out/"reports/boundary_disagreement_summary.csv",index=False)
    feature_stats=[]
    for (label,feature),g in features.groupby(['label_name','feature']):
        x=pd.to_numeric(g.relative_difference,errors='coerce').dropna()
        feature_stats.append(dict(label_name=label,feature=feature,n_paired=int(g.signed_difference.notna().sum()),
            median_absolute_difference=float(g.absolute_difference.median()) if g.absolute_difference.notna().any() else np.nan,
            median_absolute_relative_change_pct=float(x.abs().median()*100) if len(x) else np.nan))
    pd.DataFrame(feature_stats).to_csv(out/"reports/feature_change_summary.csv",index=False)
    integrity=verify_v1(hashes)
    actual_v1=dict(cohort=len(frame),patients=int(frame.patient_id.nunique()),
        medians=int(results.median_present.sum()),qrs_pairs=int(frame[['qrs_on_ms','qrs_off_ms']].notna().all(axis=1).sum()),
        four_boundaries=int(frame[['qrs_on_ms','qrs_off_ms','t_on_ms','t_off_ms']].notna().all(axis=1).sum()))
    summary=dict(method=METHOD_VERSION,method_class=METHOD_CLASS,config_id=config['config_id'],
        algorithm_fingerprint=config['algorithm_fingerprint'],input_counts=actual_v1,
        attempted=len(results),median_available=int(results.median_present.sum()),
        statuses=results.status.value_counts().to_dict(),
        t_on_available=int(results.t_on_v2_ms.notna().sum()),t_off_available=int(results.t_off_v2_ms.notna().sum()),
        t_loops_available=int((results.t_on_v2_ms.notna()&results.t_off_v2_ms.notna()).sum()),
        full_12_features=int(results[FEATURE_NAMES].notna().all(axis=1).sum()),
        v1_feature_parity=results.v1_feature_parity.value_counts().to_dict(),
        experiment_exceptions=int(results.error.fillna('').ne('').sum()),
        failure_reasons=results.loc[results.status.eq('unresolved'),'failure_reason'].value_counts().to_dict(),
        reference=ref,v1_integrity=integrity,split_manifest_byte_identical=(out/'split_manifest.csv').read_bytes()==(FEATURE_INPUT/'split_manifest.csv').read_bytes(),
        test_used_for_tuning=False,config_locked_before_test=True,
        comparison_is_accuracy=False,synthetic_tests_are_clinical_validation=False)
    write_json(out/"execution_summary.json",summary)
    write_report(out,summary,results,features)
    return results,summary


def write_report(out,summary,results,features):
    n=summary['median_available']; detected=summary['t_loops_available']
    median_coverage=f'{100*detected/n:.2f}%' if n else 'NA (no available median)'
    lines=["# Notebook 09 — Thí nghiệm T delineation Wavelet/TLS", "",
      "## Phạm vi và mức tái hiện", "",
      "Tái dựng các công thức quadratic-spline WT và vòng lặp TLS của Almeida (2009), sử dụng single-lead Martínez (2004). Đây là **reconstruction có điều chỉnh được ghi rõ**, không phải bản port bit-exact của wavedet. Xem `../../../METHOD_REPRODUCTION.md`.", "",
      "Nguồn: [Almeida 2009](https://doi.org/10.1109/TBME.2009.2021658), [Martínez 2004](https://doi.org/10.1109/TBME.2003.821031), [mã tham khảo cố định](https://github.com/marianux/ecg-kit/tree/c8e3de47c54a9214138143676d2aa546b0540dd2/common/wavedet).", "",
      "Đúng công thức: h=[1,3,3,1]/8, g=2[1,-1], bank à trous không decimation; epsilon=0.25 RMS, gamma=0.125 max; Ton=0.25 slope, SL Toff=0.4 slope, ML Toff=0.25 slope; chọn scale5 khi ≥2 trục phát hiện ở scale5; hai hướng TLS affine khác nhau; sigma=15.3 ms, cập nhật W và dừng <2 native samples hoặc cực trị không cải thiện.", "",
      "Điều chỉnh: resample hệ số bằng Kaiser polyphase; reflect-padding và cảnh báo biên; median RR thay recursive RR; RMS tính trên phần median có sẵn; cửa sổ dùng QRS V1 chung; thiếu tiêu chí biên trả unresolved thay vì fallback tới mép; SVD giải TLS thay nonlinear optimizer. Chi tiết cực trị theo mã 2011–2012 không được coi là chính xác lịch sử 2004/2009.", "",
      "## Dữ liệu và kết quả chạy thực tế", "",
      f"Đọc {summary['attempted']} IDs. Số median hợp lệ: {n}; số QRS-pair V1: {summary['input_counts']['qrs_pairs']}; số đủ bốn biên V1: {summary['input_counts']['four_boundaries']}. Các số được tính lại từ input, không điền theo mô tả nhiệm vụ.", "",
      f"V2 tạo được {detected} T loops ({median_coverage} trên median; {100*detected/summary['attempted']:.2f}% trên toàn cohort). Trạng thái: {json.dumps(summary['statuses'],ensure_ascii=False)}. Exceptions: {summary['experiment_exceptions']}.", "",
      "Các trạng thái detected/needs_review mô tả điều kiện của thuật toán, không xác nhận biên sinh lý đúng. Xem `coverage_by_label_and_split.csv` cho phân bố HC/MI và split.", "",
      "QC V2 chỉ kiểm tra những điều kiện đã khai báo cho thí nghiệm; QC V1 được giữ ở các cột riêng. Không so trực tiếp số needs_review V1/V2 để khẳng định chất lượng tăng. Kết quả chỉ giải được một biên vẫn giữ biên đó, nhưng không xuất T loop.", "",
      "## Ca smoke được chỉ định trước", ""]
    for rid in ('patient104/s0306lre','patient051/s0179lre','patient034/s0109lre','patient033/s0157lre'):
        g=results[results.record_id.eq(rid)]
        if not len(g): continue
        r=g.iloc[0]
        def fmt(x): return 'missing' if pd.isna(x) else f'{x:g}'
        lines.append(f"- `{rid}`: V1 T=[{fmt(r.t_on_v1_ms)}, {fmt(r.t_off_v1_ms)}] ms; V2=[{fmt(r.t_on_v2_ms)}, {fmt(r.t_off_v2_ms)}] ms; **{r.status}**. {r.failure_reason or ''}")
    lines += ["", "Không chọn ví dụ dựa trên V2 đẹp hơn; bốn ca trên được yêu cầu trước khi chạy. Gallery giữ mọi ID, bao gồm ca thiếu median.", "",
      "![Ca T rõ được chỉ định trước: V1/V2 trên XYZ](../figures/patient104_s0306lre_comparison.png)", "",
      "![Ca17ms V1: V2 còn thiếu T-off, không ép tạo loop](../figures/patient051_s0179lre_comparison.png)", "",
      "## Thiếu kết quả và nguyên nhân", "",
      "Các cột `t_on_failure_reason` và `t_off_failure_reason` trong v2_t_boundaries.csv giữ lý do chi tiết của từng nhánh (extrema, zero crossing, threshold/local minimum hoặc TLS). `failure_reason=missing_t_boundary` chỉ là trạng thái tổng quát của khoảng T.", ""]
    lines += [f"- {reason}: {count} recordings." for reason,count in summary['failure_reasons'].items()]
    lines += ["", "## V1/V2, reference và features", "",
      "`boundary_disagreement_summary.csv` báo chênh V2−V1 theo HC/MI và split. Đây là độ bất đồng; không phải sai số so với chuẩn.", "",
      f"Có {summary['reference']['accepted_reference_boundaries']} reference boundaries được validator chấp nhận, trong đó {summary['reference']['accepted_t_boundaries']} biên T. Các ô MAE/RMSE thiếu reference được giữ NaN; xem reference_error_summary.csv.", "",
      "QRS features giữ nguyên V1. T features gọi đúng extract_features V1 với spatial_speed tính trên toàn median rồi mới cắt, tránh đạo hàm lại ở mép loop. V1 được tái tạo và kiểm tra parity trước khi so sánh. `feature_change_summary.csv` báo mức thay đổi theo feature/nhãn; tăng/giảm không thể hiện tốt/xấu.", ""]
    for label in ('HC','MI'):
        g=features[features.label_name.eq(label)&features.feature.str.startswith('T_')]
        if len(g):
            by=g.groupby('feature').relative_difference.apply(lambda s:s.abs().median()*100).dropna()
            lines.append(f"- {label}: median |relative change| (%): "+'; '.join(f'{k}={v:.2f}' for k,v in by.items())+'.')
    lines += ["", "## Tái lập, bảo vệ V1 và giới hạn", "",
      f"{summary['v1_integrity']['protected_files']} file V1 được hash trước/sau; changed={summary['v1_integrity']['changed_files']}. Manifest byte-identical={summary['split_manifest_byte_identical']}. Config/engine được khóa trước khi chạy test; không huấn luyện lại ML, không chọn ngưỡng theo nhãn/test.", "",
      "Checkpoint kiểm tra engine/config, source row và SHA256 JSON/NPZ median; cache intermediate đủ tái tạo. Chạy lại bằng lệnh trong README của thí nghiệm.", "",
      "Không có fixture numerical từ tác giả hoặc bản chạy MATLAB để chứng minh equivalence; bank modern dfilt MAT không được dùng vì là đối tượng MATLAB opaque, không được giả gọi là coefficients gốc. Median khác beat lâm sàng gốc; border/RMS/RR adaptations có thể thay kết quả. T dẹt, thiếu cặp slopes hoặc đơn điệu có thể unresolved. TLS tối đa hóa hướng biến thiên theo giả định noise đồng nhất, không bảo đảm hướng biên sinh lý tối ưu.", "",
      "Bước tiếp: chú giải độc lập trên đúng median (ưu tiên cả ca có/không cảnh báo), đánh giá Ton/Toff và ảnh hưởng features; kiểm tra đối chiếu tác giả/MATLAB và adaptation sensitivity trên train/valid trước khi quyết định thay V1. **Chưa có căn cứ kết luận Almeida/V2 tốt hơn V1.**", ""]
    # Deterministic descriptive examples from train/valid. Selection is by
    # recorded morphology/QC category, never which V2 picture looks better.
    cases=[]
    for rid in ('patient104/s0306lre','patient051/s0179lre','patient034/s0109lre','patient033/s0157lre'):
        if results.record_id.eq(rid).any():cases.append(dict(record_id=rid,selection='user_preselected',source='task before V2 execution'))
    sl_path=out/'diagnostics/single_lead_candidates.csv'
    if sl_path.exists():
        sl=pd.read_csv(sl_path)
        for shape in ('negative','biphasic_positive_negative','biphasic_negative_positive'):
            group=sl[sl.split.ne('test')&sl.morphology_type.eq(shape)].sort_values(['record_id','axis'])
            if len(group):cases.append(dict(record_id=group.iloc[0].record_id,selection='first_train_valid_SL_'+shape,
                source='SL algorithm morphology, not independently verified; axis='+group.iloc[0].axis))
        low=sl[sl.split.ne('test')&pd.to_numeric(sl.get('rms'),errors='coerce').gt(0)].sort_values(['rms','record_id','axis'])
        if len(low):cases.append(dict(record_id=low.iloc[0].record_id,selection='minimum_available_WT_RMS_proxy',
            source='WT RMS proxy only; not verified flat T morphology'))
    tv=results[results.split.ne('test')]
    for flag in ('st_level_requires_review','t_end_methods_disagree'):
        group=tv[tv.v1_qc_flags.fillna('').str.contains(flag,regex=False)].sort_values('record_id')
        if len(group):cases.append(dict(record_id=group.iloc[0].record_id,selection='first_train_valid_V1_'+flag,
            source='V1 QC flag; not reference error'))
    wide=tv.assign(qrs_duration=tv.qrs_off_ms-tv.qrs_on_ms).dropna(subset=['qrs_duration']).sort_values(['qrs_duration','record_id'],ascending=[False,True])
    if len(wide):cases.append(dict(record_id=wide.iloc[0].record_id,selection='largest_V1_QRS_duration_train_valid',
        source='Duration proxy, not independently classified abnormal QRS'))
    pd.DataFrame(cases).to_csv(out/'diagnostics/representative_cases.csv',index=False)
    lines += ['## Ví dụ bổ sung có quy tắc chọn công khai','',
      'Mỗi nhóm chọn ID đầu theo thứ tự hoặc cực trị proxy đã nêu trên train/valid; không chọn theo V2 tốt hơn. Morphology tự động và QC không được coi là annotation độc lập.']
    lines += [f"- `{c['record_id']}`: {c['selection']} ({c['source']})." for c in cases if c['selection']!='user_preselected']
    lines += ['', '## Kiểm thử và giao diện','']
    for name in ('test_results.json','notebook_execution.json'):
        p=out/'reports'/name
        if p.exists():lines.append(f"- `{name}`: "+json.dumps(json.loads(p.read_text()),ensure_ascii=False)+'.')
    p=out/'gallery/browser_validation.json'
    if p.exists():
        browser=json.loads(p.read_text())
        lines.append(f"- Real headless Chrome mở {browser['records']} payloads, thử {browser['representative_tab_views']} lượt tab đại diện; JS exceptions={len(browser['javascript_exceptions'])}. Ảnh initial/TLS/loops đã được lưu; đây không phải chuyên gia review từng recording.")
    lines += ['', 'Gallery offline: [mở toàn cohort](../gallery/index.html). Test log: `../logs/test_run.log`.','']
    (out/'reports/experiment_report.md').write_text('\n'.join(lines))


def run_experiment(out=DEFAULT_OUT,record_ids=None,resume=True,plots=True,gallery=True,reference_csv=None):
    cfg=AlmeidaConfig()
    frame,manifest,v1cfg,hashes,config=initialize(out,cfg)
    out=Path(out)
    selected=frame if record_ids is None else frame[frame.record_id.isin(record_ids)]
    if record_ids and set(record_ids)-set(selected.record_id):
        raise ValueError('Unknown requested record IDs')
    result_rows,diagnostics=[],[]
    reused=0
    for i,row in enumerate(selected.to_dict('records'),1):
        result,diag,cached=process_one(row,out,cfg,v1cfg,config['algorithm_fingerprint'],resume)
        result_rows.append(result);diagnostics.append(diag);reused+=int(cached)
        if i%50==0 or i==len(selected): print(f'Processed {i}/{len(selected)}; checkpoints reused={reused}',flush=True)
    results,summary=write_outputs(selected,result_rows,diagnostics,out,config,hashes,reference_csv)
    if plots:
        from vcg_t_almeida_plots import render_cohort_figures
        render_cohort_figures(results,out)
    if gallery:
        from vcg_t_almeida_gallery import build_gallery
        summary['gallery']=build_gallery(results,out)
    summary['checkpoint_reused']=reused
    summary['v1_integrity']=verify_v1(hashes)
    write_json(out/'execution_summary.json',summary)
    write_json(out/'logs/last_run.json',dict(summary=summary,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT/'vcg_t_almeida.py',Path(__file__),ROOT/'scripts/vcg_t_almeida_plots.py',ROOT/'scripts/vcg_t_almeida_gallery.py'] if p.exists()}))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--record',action='append',dest='record_ids')
    parser.add_argument('--no-resume',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--no-gallery',action='store_true')
    parser.add_argument('--reference-csv',type=Path)
    args=parser.parse_args(argv)
    summary=run_experiment(args.output,args.record_ids,not args.no_resume,not args.no_plots,not args.no_gallery,args.reference_csv)
    print(json.dumps(json_safe(summary),ensure_ascii=False,indent=2))


if __name__=='__main__': main()

from pathlib import Path
from base64 import b64encode
from io import BytesIO
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "10_VCG_V3_Consensus_T_Delineation.ipynb"

def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())

cells = [
md("""# 10 — VCG V3: consensus T delineation không cần nhãn

Notebook này xây V3 trên V1 frozen:

```
PTB -> preprocessing/QC -> QRS -> median
    -> V1 tangent/threshold + Wavelet/TLS + derivative threshold
    -> consensus + uncertainty -> loops/features
    -> đánh giá không nhãn
```

Không có ground truth T-on/T-off. Vì vậy agreement, stability, coverage và
feature sensitivity chỉ là đánh giá **tính nhất quán nội bộ**, không phải accuracy.
V1 artifacts/features không bị ghi đè."""),
md("""## 1. Phương pháp V3

Mỗi phương pháp trả candidate, status, flags và trace:

- **V1 tangent/threshold**: baseline hiện tại.
- **Wavelet/TLS**: reconstruction Almeida/Martínez có adaptation được ghi trong
  `METHOD_REPRODUCTION.md`.
- **Derivative/threshold**: phép đo độc lập từ magnitude, slope và TP/PQ baseline.

Consensus chỉ nhận boundary khi tối thiểu hai method khác nhau đồng thuận trong
tolerance, đúng thứ tự QRS-off < T-on < T-peak < T-off. Nếu không, trả
`needs_review` hoặc `unresolved`, không tự dựng loop."""),
code(r'''
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if ROOT.name != "VCG" and (ROOT / "VCG").is_dir(): ROOT = ROOT / "VCG"
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from vcg_preprocessing import Config
from vcg_pipeline import (FEATURE_NAMES, load_xyz, preprocess_xyz,
    detect_qrs_anchors, representative_beat, delineate_loops, cut_loop,
    extract_features, spatial_speed, ms_samples)
from vcg_t_almeida import AlmeidaConfig, delineate_t

DATA_ROOT = ROOT / "content/ptbdb/physionet.org/files/ptbdb/1.0.0"
OUTPUT_ROOT = ROOT / "outputs/vcg_v3_consensus"
SPLIT_MANIFEST = ROOT / "data/split_manifest.csv"
V1_CFG, ALMEIDA_CFG = Config(), AlmeidaConfig()
TOLERANCE_MS, MIN_METHODS = 25.0, 2
MIN_T_MS, MAX_T_MS = 45.0, 450.0
RECORD_ID, RUN_COHORT, COHORT_LIMIT = "patient104/s0306lre", False, 20

def meta():
    return dict(method_version="vcg-v3-consensus-0.1",
        method_class="unlabeled_multi_method_consensus",
        tolerance_ms=TOLERANCE_MS, min_methods=MIN_METHODS,
        min_t_ms=MIN_T_MS, max_t_ms=MAX_T_MS,
        evaluation_note="Agreement/stability are not boundary accuracy.",
        v1_config=V1_CFG.__dict__, almeida_config=ALMEIDA_CFG.__dict__)

def finite_int(x):
    return x is not None and np.isfinite(x) and int(x) == x

def ms(x, fs, anchor):
    return None if x is None else (int(x)-int(anchor))*1000/float(fs)

def load_state(rid):
    raw = load_xyz(DATA_ROOT / rid, V1_CFG.seconds)
    fs = raw["fs"]
    morph, detection, prep = preprocess_xyz(raw["xyz"], fs, V1_CFG, return_diagnostics=True)
    detector = detect_qrs_anchors(morph, detection, fs, V1_CFG)
    rep = representative_beat(morph, detector["anchors"], fs, V1_CFG,
                              quality=prep["input_quality"])
    rep["fs"], rep["record_id"] = fs, rid
    return dict(raw=raw, fs=fs, morph=morph, detection=detection,
                preprocessing=prep, detector=detector, rep=rep)

print(json.dumps(meta(), default=str, ensure_ascii=False, indent=2)[:1200])
print("PTB available:", DATA_ROOT.is_dir())
'''),
md("""## 2. Preprocessing và median beat

V3 giữ nguyên preprocessing, QRS detector và representative beat của V1. Chỉ
tầng tách T thay đổi; điều này giúp so sánh không bị lẫn với thay đổi filter."""),
md("""## 2A. Preview số liệu đã có trong repo

Cell này đọc các bảng V1 đang tracked, không giả làm kết quả V3. Nó cho bạn xem
ngay quy mô cohort, split, label và phân phối một vài feature trước khi chạy
pipeline tín hiệu. Đây là đối chiếu với Notebook 06; V3 chỉ thay tầng T boundary,
không được hiểu là đã chạy lại toàn bộ cohort."""),
code(r'''
def show_tracked_baseline_preview():
    if (ROOT / "vcg_12_features.csv").is_file():
        f=pd.read_csv(ROOT / "vcg_12_features.csv")
        print("V1 feature rows:", len(f), "patients:", f.patient_id.nunique())
        display(f.groupby("label", dropna=False).size().rename("records").to_frame())
        display(f[FEATURE_NAMES].describe().T[["count","mean","std","min","max"]])
        fig,ax=plt.subplots(1,2,figsize=(12,4),layout="constrained")
        f.boxplot(column="T_MaxVel",by="label",ax=ax[0],grid=False)
        ax[0].set_title("Tracked V1 T_MaxVel by label"); ax[0].set_xlabel("label"); ax[0].set_ylabel("mV/s")
        f.boxplot(column="T_Area",by="label",ax=ax[1],grid=False)
        ax[1].set_title("Tracked V1 T_Area by label"); ax[1].set_xlabel("label"); ax[1].set_ylabel("mV²")
        fig.suptitle("Baseline preview only — not V3 output")
        preview=ROOT / "outputs/vcg_v3_consensus"; preview.mkdir(parents=True,exist_ok=True)
        fig.savefig(preview / "tracked_v1_feature_preview.png",dpi=160,bbox_inches="tight")
        print("Đã lưu preview:", preview / "tracked_v1_feature_preview.png")
        plt.show()
    else: print("Không có vcg_12_features.csv")
    if (ROOT / "data/split_manifest.csv").is_file():
        m=pd.read_csv(ROOT / "data/split_manifest.csv")
        print("Manifest rows:",len(m),"patients:",m.patient_id.nunique())
        display(m.groupby("split",dropna=False).size().rename("records").to_frame())

show_tracked_baseline_preview()
'''),
code(r'''
state = load_state(RECORD_ID) if DATA_ROOT.is_dir() else None
if state:
    print(RECORD_ID, "anchors=", len(state["detector"]["anchors"]),
          "beats=", state["rep"]["n_used"],
          "flags=", state["rep"].get("quality_flags", []))
else:
    print("Chưa có PTB local; chạy cell này sau khi tải dữ liệu theo README.")
'''),
md("""## 3. Ba candidate T

A dùng V1 tangent/threshold. B dùng Wavelet/TLS độc lập với V1 T marks. C dùng
magnitude + derivative + threshold về baseline. C chỉ là phép đo robustness, không
phải nhãn tham chiếu."""),
code(r'''
def v1_candidate(st):
    if st is None: return dict(method="v1_tangent_threshold", status="unavailable")
    b = delineate_loops(st["rep"], st["fs"], V1_CFG)
    return dict(method="v1_tangent_threshold",
        status="detected" if b.get("t_on") is not None and b.get("t_off") is not None else "unresolved",
        t_on=b.get("t_on"), t_peak=b.get("t_peak"), t_off=b.get("t_off"),
        qrs_on=b.get("qrs_on"), qrs_off=b.get("qrs_off"),
        flags=list(b.get("flags", [])), trace=b)

def wavelet_candidate(st, a):
    if st is None: return dict(method="wavelet_tls", status="unavailable")
    if not finite_int(a.get("qrs_on")) or not finite_int(a.get("qrs_off")):
        return dict(method="wavelet_tls", status="unresolved", reason="missing_qrs")
    try:
        r = delineate_t(st["rep"]["xyz"], st["fs"], st["rep"]["anchor"],
                        a["qrs_on"], a["qrs_off"], st["rep"]["rr_seconds"], ALMEIDA_CFG)
    except Exception as exc:
        return dict(method="wavelet_tls", status="error", reason=f"{type(exc).__name__}: {exc}")
    return dict(method="wavelet_tls", status=r.get("status", "unresolved"),
        t_on=r.get("t_on_sample"), t_peak=None, t_off=r.get("t_off_sample"),
        qrs_on=a["qrs_on"], qrs_off=a["qrs_off"],
        flags=list(r.get("review_flags", [])), trace=r)

def derivative_candidate(st, a):
    if st is None: return dict(method="derivative_threshold", status="unavailable")
    qoff = a.get("qrs_off")
    if not finite_int(qoff): return dict(method="derivative_threshold", status="unresolved", reason="missing_qrs_off")
    from scipy.signal import savgol_filter, find_peaks
    rep, fs, n = st["rep"], st["fs"], len(st["rep"]["xyz"])
    mag = np.linalg.norm(rep["xyz"], axis=1)
    win = max(5, min(n-(1-n%2), int(round(21*fs/1000))|1)); win -= win % 2 == 0
    smooth = savgol_filter(mag, win, 3, mode="interp")
    lo = int(qoff)+ms_samples(20, fs)
    hi = min(n-1, rep["anchor"]+int(min(.60,.70*rep["rr_seconds"])*fs))
    if hi-lo < ms_samples(100, fs): return dict(method="derivative_threshold", status="unresolved", reason="short_t_window")
    base = float(np.median(smooth[rep["pq"][0]:rep["pq"][1]]))
    noise = max(float(1.4826*np.median(np.abs(smooth[rep["pq"][0]:rep["pq"][1]]-base))), 1e-6)
    peaks, props = find_peaks(smooth[lo:hi+1], prominence=max(3*noise,1e-6), distance=max(1,ms_samples(40,fs)))
    peaks += lo
    if not len(peaks): return dict(method="derivative_threshold", status="unresolved", reason="no_t_peak")
    peak = int(peaks[np.argmax(props["prominences"])])
    threshold = base + max(3*noise, .08*max(smooth[peak]-base,0))
    run = max(2, ms_samples(20, fs))
    def cross(start, stop, step):
        count=0
        for i in range(start, stop+step, step):
            count = count+1 if smooth[i] <= threshold else 0
            if count >= run: return i-run+1 if step>0 else i+run-1
        return None
    on, off = cross(peak,lo,-1), cross(peak,hi,1)
    valid = on is not None and off is not None and int(qoff)<on<peak<off<n
    flags = ["t_low_amplitude"] if smooth[peak]-base <= 5*noise else []
    if on is None or off is None: flags.append("threshold_crossing_missing")
    return dict(method="derivative_threshold", status="detected" if valid else "unresolved",
        t_on=on, t_peak=peak, t_off=off, qrs_on=a.get("qrs_on"), qrs_off=int(qoff),
        flags=flags, trace=dict(magnitude=mag, smooth=smooth, search=(lo,hi)))

a = v1_candidate(state)
b = wavelet_candidate(state, a)
c = derivative_candidate(state, a)
display(pd.DataFrame([{k:x.get(k) for k in ("method","status","t_on","t_peak","t_off","flags")} for x in (a,b,c)]))
'''),
md("""## 4. Consensus và uncertainty

`spread_ms` là độ phân tán của các candidate trong cluster. `confidence` không
phải xác suất đúng; nó chỉ phản ánh số method và mức agreement. Boundary không đủ
method không được dùng để dựng T loop."""),
code(r'''
def cluster(values, tol):
    groups=[]
    for method, value in sorted(values, key=lambda x:x[1]):
        for g in groups:
            if abs(value-np.mean([v for _,v in g])) <= tol: g.append((method,value)); break
        else: groups.append([(method,value)])
    return sorted(groups, key=lambda g:(-len({m for m,_ in g}), np.std([v for _,v in g])))

def consensus(st, candidates):
    rep, fs, n = st["rep"], st["fs"], len(st["rep"]["xyz"])
    qoff = next((x.get("qrs_off") for x in candidates if finite_int(x.get("qrs_off"))), None)
    tol = max(1, round(TOLERANCE_MS*fs/1000))
    usable=[x for x in candidates if x.get("status") in {"detected","needs_review"}
            and finite_int(x.get("t_on")) and finite_int(x.get("t_off"))
            and finite_int(qoff) and 0<=qoff<x["t_on"]<x["t_off"]<n
            and (not finite_int(x.get("t_peak")) or x["t_on"] < x["t_peak"] < x["t_off"])]
    out=dict(status="unresolved", flags=[], candidates=candidates, method_count=len(usable), boundary={})
    for name in ("t_on","t_off"):
        groups=cluster([(x["method"],int(x[name])) for x in usable], tol)
        g=groups[0] if groups else []; methods=sorted({m for m,_ in g})
        spread=float(np.std([v for _,v in g])) if g else np.nan
        sample=round(np.median([v for _,v in g])) if g else None
        out["boundary"][name]=dict(sample=sample, ms=ms(sample,fs,rep["anchor"]),
            methods=methods, n_methods=len(methods), spread_ms=spread*1000/fs if g else np.nan,
            members=[(m,int(v)) for m,v in g])
        if len(methods)<MIN_METHODS: out["flags"].append(name+"_insufficient_method_agreement")
    on,off=out["boundary"]["t_on"]["sample"],out["boundary"]["t_off"]["sample"]
    if on is not None and off is not None and qoff<on<off<n:
        out["duration_ms"]=(off-on)*1000/fs
        if MIN_T_MS <= out["duration_ms"] <= MAX_T_MS and not out["flags"]: out["status"]="consensus"
        else: out["status"]="needs_review"; out["flags"].append("duration_or_review_flag")
    else: out["flags"].append("invalid_consensus_order")
    out["flags"]=sorted(set(out["flags"]))
    if out["status"]=="unresolved" and out["flags"]: out["status"]="needs_review"
    return out

result = dict(record_id=RECORD_ID, v1=a, wavelet=b, derivative=c,
              consensus=consensus(state, [a,b,c]) if state else None, meta=meta())
if result["consensus"]:
    cc=result["consensus"]
    display(pd.DataFrame([dict(record_id=RECORD_ID,status=cc["status"],
        t_on_ms=cc["boundary"]["t_on"]["ms"], t_off_ms=cc["boundary"]["t_off"]["ms"],
        duration_ms=cc.get("duration_ms"), t_on_methods=",".join(cc["boundary"]["t_on"]["methods"]),
        t_off_methods=",".join(cc["boundary"]["t_off"]["methods"]), flags=";".join(cc["flags"]))]))
'''),
md("""## 5. Cắt loops và 12 features

Chỉ tạo T loop khi cặp consensus hợp lệ. Feature finite không đồng nghĩa feature
được xác nhận; trạng thái và spread boundary phải đi kèm."""),
code(r'''
def add_features(st, res):
    if not st or not res or not res["consensus"]: return res
    rep, fs, q = st["rep"], st["fs"], res["v1"]
    cb=res["consensus"]["boundary"]
    qloop=cut_loop(rep["xyz"],q.get("qrs_on"),q.get("qrs_off"))
    tloop=cut_loop(rep["xyz"],cb["t_on"]["sample"],cb["t_off"]["sample"])
    speed=spatial_speed(rep["xyz"],fs,V1_CFG)
    speeds={"QRS":None,"T":None}
    if qloop is not None: speeds["QRS"]=speed[q["qrs_on"]:q["qrs_off"]+1]
    if tloop is not None: speeds["T"]=speed[cb["t_on"]["sample"]:cb["t_off"]["sample"]+1]
    res["qrs_loop"],res["t_loop"]=qloop,tloop
    res["features"]=extract_features(qloop,tloop,fs,V1_CFG,speeds=speeds)
    return res

result=add_features(state,result)
if result.get("features"): display(pd.DataFrame([result["features"]]))
'''),
md("""## 6. Visual audit

Các đường candidate chỉ mô tả phương pháp. Đường đỏ là consensus nếu có; không có
đường nào được gọi là expert truth."""),
code(r'''
if state and result["consensus"]:
    rep,fs=state["rep"],state["fs"]; t=rep["time_ms"]; xyz=rep["xyz"]
    mag=np.linalg.norm(xyz,axis=1)
    fig,ax=plt.subplots(2,1,figsize=(14,8),sharex=True,layout="constrained")
    for j,col in enumerate(("#2563eb","#059669","#d97706")): ax[0].plot(t,xyz[:,j],color=col,label="XYZ"[j])
    ax[1].plot(t,mag,color="0.25",label="magnitude")
    cols={"v1_tangent_threshold":"#d97706","wavelet_tls":"#7c3aed","derivative_threshold":"#0891b2"}
    for item in result["consensus"]["candidates"]:
        for name,ls in (("t_on",":"),("t_off","--"),("t_peak","-")):
            if finite_int(item.get(name)): ax[1].axvline(ms(item[name],fs,rep["anchor"]),color=cols[item["method"]],ls=ls,alpha=.7,label=item["method"]+" "+name)
    for name in ("t_on","t_off"):
        s=result["consensus"]["boundary"][name]["sample"]
        if s is not None: ax[0].axvline(t[s],color="#dc2765",lw=2,label="V3 "+name); ax[1].axvline(t[s],color="#dc2765",lw=2)
    for x in ax: x.axvline(0,color="black",ls="--",lw=.8); x.grid(alpha=.2); x.legend(ncol=3,fontsize=8)
    ax[0].set_ylabel("XYZ (mV)"); ax[1].set_ylabel("Magnitude"); ax[1].set_xlabel("ms relative to QRS anchor")
    fig.suptitle(f"{RECORD_ID} | {result['consensus']['status']} | {'; '.join(result['consensus']['flags']) or 'no flags'}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT_ROOT / (RECORD_ID.replace('/', '_') + "_v3_boundary_audit.png")
    fig.savefig(audit_path, dpi=160, bbox_inches="tight")
    print("Đã lưu hình audit:", audit_path)
    plt.show()
'''),
md("""## 6A. Cách đọc hình và so sánh với V1

Candidate là các đường mảnh theo từng phương pháp; đường đỏ dày là consensus.
`spread_ms` là độ phân tán candidate. V1 tương ứng các cell tìm T bằng magnitude,
tangent và threshold trong Notebook 05; V3 chỉ chấp nhận boundary khi có thêm
đồng thuận. PNG được lưu trong `outputs/vcg_v3_consensus/`."""),
code(r'''
def print_candidate_report(st, res):
    if not st or not res or not res.get("consensus"):
        print("Chưa có state/result; hãy chạy các cell trước."); return
    fs, anchor = st["fs"], st["rep"]["anchor"]
    rows=[]
    for item in res["consensus"]["candidates"]:
        rows.append(dict(method=item.get("method"), status=item.get("status"),
            t_on_ms=ms(item.get("t_on"),fs,anchor), t_peak_ms=ms(item.get("t_peak"),fs,anchor),
            t_off_ms=ms(item.get("t_off"),fs,anchor), flags=";".join(item.get("flags") or [])))
    print("BẢNG CANDIDATE — không phải nhãn chuẩn")
    display(pd.DataFrame(rows))
    c=res["consensus"]
    print("CONSENSUS:", c["status"], "flags:", c["flags"])
    for name in ("t_on", "t_off"):
        print(name, "methods=", c["boundary"][name]["methods"], "spread_ms=", c["boundary"][name]["spread_ms"])
    print("spread nhỏ chỉ phản ánh nhất quán nội bộ; không chứng minh boundary đúng sinh lý.")

print_candidate_report(state, result)
'''),
md("""## 6B. So sánh trực tiếp V1 và V3

`delta_v3_minus_v1_ms` chỉ là thay đổi do thuật toán. Không gọi nó là sai số vì
chưa có reference. Nếu V3 thiếu đồng thuận, giá trị để trống và record cần review."""),
code(r'''
def v1_v3_comparison(st, res):
    if not st or not res or not res.get("consensus"): return pd.DataFrame()
    fs, anchor=st["fs"],st["rep"]["anchor"]; rows=[]
    for name in ("t_on","t_off"):
        v1_ms, v3_ms = ms(res["v1"].get(name),fs,anchor), ms(res["consensus"]["boundary"][name]["sample"],fs,anchor)
        rows.append(dict(boundary=name, v1_ms=v1_ms, v3_consensus_ms=v3_ms,
            delta_v3_minus_v1_ms=None if v1_ms is None or v3_ms is None else v3_ms-v1_ms,
            v3_methods=",".join(res["consensus"]["boundary"][name]["methods"]),
            v3_spread_ms=res["consensus"]["boundary"][name]["spread_ms"]))
    return pd.DataFrame(rows)

comparison=v1_v3_comparison(state,result)
print("SO SÁNH V1 → V3 (không phải accuracy):")
display(comparison)
'''),
md("""## 7. Đánh giá không nhãn

Không tính accuracy/MAE khi chưa có annotation. Thay vào đó:

- **coverage**: candidate, consensus, valid T loop;
- **agreement**: số method và `spread_ms` cho T-on/T-off;
- **failure taxonomy**: missing QRS, no peak, no threshold, method disagreement;
- **perturbation stability**: thêm nhiễu nhỏ, so delta boundary;
- **feature sensitivity**: dịch boundary ±10/±20 ms;
- **patient/split integrity**: chỉ mô tả, không tinh chỉnh theo label/test."""),
code(r'''
def perturb_stability(st, seeds=(0,1,2), fraction=.01):
    if not st: return pd.DataFrame()
    base=consensus(st,[v1_candidate(st),wavelet_candidate(st,v1_candidate(st)),derivative_candidate(st,v1_candidate(st))])
    bon,boff=base["boundary"]["t_on"]["sample"],base["boundary"]["t_off"]["sample"]
    sig=float(1.4826*np.median(np.abs(st["rep"]["xyz"]-np.median(st["rep"]["xyz"],axis=0))))
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); clone=dict(st); clone["rep"]=dict(st["rep"])
        clone["rep"]["xyz"]=st["rep"]["xyz"]+rng.normal(0,max(sig*fraction,1e-9),st["rep"]["xyz"].shape)
        aa=v1_candidate(clone); rr=consensus(clone,[aa,wavelet_candidate(clone,aa),derivative_candidate(clone,aa)])
        on,off=rr["boundary"]["t_on"]["sample"],rr["boundary"]["t_off"]["sample"]
        rows.append(dict(seed=seed,status=rr["status"],
            t_on_delta_ms=None if on is None or bon is None else (on-bon)*1000/st["fs"],
            t_off_delta_ms=None if off is None or boff is None else (off-boff)*1000/st["fs"],
            flags=";".join(rr["flags"])))
    return pd.DataFrame(rows)

stability=perturb_stability(state)
display(stability)
'''),
code(r'''
def boundary_feature_sensitivity(st, res, radii=(0,10,20)):
    if not st or not res or res.get("t_loop") is None: return pd.DataFrame()
    rep,fs,q=st["rep"],st["fs"],res["v1"]; cb=res["consensus"]["boundary"]
    qloop=cut_loop(rep["xyz"],q["qrs_on"],q["qrs_off"]); speed=spatial_speed(rep["xyz"],fs,V1_CFG)
    rows=[]
    for radius in radii:
        shifts=[(0,0)] if not radius else [(-radius,-radius),(-radius,radius),(radius,-radius),(radius,radius)]
        for so,sf in shifts:
            on=cb["t_on"]["sample"]+round(so*fs/1000); off=cb["t_off"]["sample"]+round(sf*fs/1000)
            tl=cut_loop(rep["xyz"],on,off)
            if tl is None: continue
            feat=extract_features(qloop,tl,fs,V1_CFG,speeds={"QRS":speed[q["qrs_on"]:q["qrs_off"]+1],"T":speed[on:off+1]})
            rows.append(dict(radius_ms=radius,on_shift_ms=so,off_shift_ms=sf,**feat))
    return pd.DataFrame(rows)

sensitivity=boundary_feature_sensitivity(state,result)
display(sensitivity.head(12))
'''),
md("""## 7A. In trực tiếp stability và feature sensitivity

`delta_ms` đo boundary thay đổi sau nhiễu nhỏ. `median_abs_change`/`max_abs_change`
đo feature thay đổi khi dịch boundary ±10/±20 ms. Đây là robustness và uncertainty
propagation, không phải bằng chứng detector đúng. Nếu bảng rỗng thì record chưa có
consensus T loop hợp lệ."""),
code(r'''
print("STABILITY SAU NHIỄU NHỎ")
if stability.empty:
    print("Không có stability result.")
else:
    display(stability)
    print("Median |Δ T-on|:", stability.t_on_delta_ms.abs().median(), "ms")
    print("Median |Δ T-off|:", stability.t_off_delta_ms.abs().median(), "ms")
print("FEATURE SENSITIVITY")
if sensitivity.empty:
    print("Không có sensitivity result; cần consensus T loop.")
else:
    display(sensitivity.head(24))
    print("radius_ms=10/20 là perturbation giả định, không phải annotation.")
'''),
md("""## 8. Cohort run và output

Mặc định tắt để tránh chạy nhầm toàn bộ PTB. Khi bật `RUN_COHORT=True`, notebook
giữ cả record lỗi, đọc split manifest nhưng chỉ dùng label cho bảng mô tả. Không
tinh chỉnh tolerance/ngưỡng theo test."""),
code(r'''
def run_cohort(limit=COHORT_LIMIT):
    manifest=pd.read_csv(SPLIT_MANIFEST).sort_values("record_id").head(limit)
    rows=[]; errors=[]
    for rid in manifest.record_id:
        try:
            st=load_state(rid); aa=v1_candidate(st)
            rr=dict(record_id=rid,v1=aa,wavelet=wavelet_candidate(st,aa),derivative=derivative_candidate(st,aa))
            rr["consensus"]=consensus(st,[rr["v1"],rr["wavelet"],rr["derivative"]]); rr=add_features(st,rr)
            cc=rr["consensus"]; rows.append(dict(record_id=rid,patient_id=rid.split("/")[0],
                status=cc["status"],t_on_ms=cc["boundary"]["t_on"]["ms"],t_off_ms=cc["boundary"]["t_off"]["ms"],
                duration_ms=cc.get("duration_ms"),t_on_spread_ms=cc["boundary"]["t_on"]["spread_ms"],
                t_off_spread_ms=cc["boundary"]["t_off"]["spread_ms"],flags=";".join(cc["flags"])))
        except Exception as exc:
            errors.append(dict(record_id=rid,error=f"{type(exc).__name__}: {exc}"))
            rows.append(dict(record_id=rid,status="error",flags="exception"))
    return pd.DataFrame(rows),pd.DataFrame(errors),manifest

if RUN_COHORT and DATA_ROOT.is_dir():
    cohort,errors,manifest=run_cohort()
    display(cohort.head()); print("attempted",len(cohort),"errors",len(errors))
else:
    cohort,errors,manifest=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    print("Cohort disabled; inspect demo first.")
'''),
code(r'''
def save_outputs(frame, errors):
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    (OUTPUT_ROOT/"config.json").write_text(json.dumps(meta(),ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    if not frame.empty: frame.to_csv(OUTPUT_ROOT/"cohort_summary.csv",index=False)
    if not errors.empty: errors.to_csv(OUTPUT_ROOT/"errors.csv",index=False)
    (OUTPUT_ROOT/"README.md").write_text(
        "# VCG V3 consensus output\n\nAgreement/stability are not boundary accuracy. "
        "V1 outputs are not overwritten.\n",encoding="utf-8")
    print(OUTPUT_ROOT)

if RUN_COHORT: save_outputs(cohort,errors)
'''),
md("""## 9. Kết luận và tiêu chí thay thế V1

Chưa có nhãn thì V3 chỉ nên dùng để:

- tìm record ổn định;
- phát hiện method disagreement;
- định lượng boundary/feature sensitivity;
- tạo danh sách cho expert review;
- so sánh robustness giữa V1 và V3.

Không nên kết luận V3 tốt hơn V1 dựa trên coverage hoặc agreement đơn thuần.
Chỉ thay V1 sau khi có annotation độc lập và xác nhận rằng consensus/stability
tương quan với boundary quality. Sau đó mới khóa acceptance gate trên validation
và báo cáo test một lần cuối."""),
]

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
               "language_info":{"name":"python","version":"3.11+"}}
nb.cells = cells

# Embed a reproducible, non-V3 baseline preview so the notebook is useful when
# opened before PTB/wfdb is installed. Signal-derived V3 cells remain unexecuted
# until the user runs them with the raw PTB database.
try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    preview_cell = next(c for c in nb.cells if c.cell_type == "code" and "show_tracked_baseline_preview()" in c.source)
    outputs = [{"output_type": "stream", "name": "stdout", "text": "Embedded preview: tracked V1 CSV only; not V3 signal output.\n"}]
    csv = ROOT / "vcg_12_features.csv"
    if csv.is_file():
        frame = pd.read_csv(csv)
        text = (f"V1 feature rows: {len(frame)}; patients: {frame.patient_id.nunique()}\n"
                + frame.groupby("label", dropna=False).size().rename("records").to_string() + "\n")
        outputs[0]["text"] += text
        outputs.append({"output_type": "execute_result", "execution_count": 1,
            "data": {"text/plain": [text]}, "metadata": {}})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        frame.boxplot(column="T_MaxVel", by="label", ax=ax[0], grid=False)
        ax[0].set_title("Tracked V1 T_MaxVel by label")
        frame.boxplot(column="T_Area", by="label", ax=ax[1], grid=False)
        ax[1].set_title("Tracked V1 T_Area by label")
        fig.suptitle("Baseline preview only — not V3 output")
        buf = BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
        outputs.append({"output_type": "display_data",
            "data": {"image/png": b64encode(buf.getvalue()).decode("ascii")}, "metadata": {}})
    preview_cell.execution_count = 1
    preview_cell.outputs = [nbf.from_dict(item) for item in outputs]
except Exception as exc:
    # Notebook generation must still work when optional plotting dependencies are absent.
    preview_cell = next((c for c in nb.cells if c.cell_type == "code" and "show_tracked_baseline_preview()" in c.source), None)
    if preview_cell is not None:
        preview_cell.execution_count = None
        preview_cell.outputs = [nbf.from_dict({"output_type": "stream", "name": "stdout", "text": f"Preview embedding skipped: {exc}\n"})]
nbf.write(nb,TARGET)
print(TARGET)


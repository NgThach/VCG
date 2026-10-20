# PTB metadata, split, and reference audit

Audit date: 2026-10-15 (Asia/Ho_Chi_Minh). This audit reads existing artifacts and primary PhysioNet documentation; it does not run a new delineation or classification evaluation.

## Local inventory and diagnostic labels

The local `content/ptbdb/physionet.org/files/ptbdb/1.0.0/RECORDS` contains 549 unique recording names and exactly matches `ptb_records_overview.csv`. Every listed recording has all three required files: `.hea`, `.dat`, and `.xyz` (549 files of each suffix). There are 290 represented patients. All records have 15 channels sampled at 1000 Hz, including measured Frank `vx`, `vy`, `vz`. Durations range from 32.000 to 120.012 seconds. These are local inventory results, rather than the inconsistent historical counts in introductory prose or the old README.

| Population | Records | Patients |
| --- | ---: | ---: |
| Myocardial infarction (MI) | 368 | 148 |
| Healthy control (HC) | 80 | 52 |
| Other/unknown diagnoses | 101 | 90 |
| All records | 549 | 290 |
| MI/HC evaluation population | 448 | 200 |

The other/unknown group includes 27 records from 22 patients with unavailable diagnoses. None of the 290 patients changes the grouped MI/HC/OTHER diagnosis across their local records. `content/ptbdb_master_dataset.csv` agrees with these diagnostic counts. Its `is_healthy_control=False` means only that the record is not labeled HC: it includes MI, other disorders, and unknown cases, and must not be treated as an MI label.

PhysioNet's current data-description section reports 549 recordings from 290 subjects, 1000 Hz, 12 conventional plus 3 Frank leads, and clinical summaries in headers. Its introductory announcement and old README contain older, inconsistent subject/HC counts. The local manifest and parsed headers are the operational denominator. [PTB Diagnostic ECG Database v1.0.0](https://physionet.org/content/ptbdb/1.0.0/)

## Existing split and two historical extraction failures

`02_VCG_MI_HC_pipeline.ipynb`, code cell 36 (zero-based JSON cell index), defines a patient-stratified 75/25 train/test split. There is no pre-existing PTB validation split or standalone persisted split assignment file in the inspected workspace. The algorithm is:

```python
patient_labels = df_features[["patient_id", "label"]].drop_duplicates()
unique_patients = patient_labels.drop_duplicates("patient_id")
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.25, random_state=42,
    stratify=unique_patients["label"]
)
```

Reproducing this algorithm from the existing `vcg_12_features.csv` **in its original row order** gives exactly the stored notebook outputs: 150 training patients (111 MI, 39 HC) and 50 test patients (37 MI, 13 HC), without patient overlap. The features contain 331 training recordings (268 MI, 63 HC) and 115 test recordings (98 MI, 17 HC).

The historical feature table has 446 rows, rather than the 448 eligible recordings, because of the following explicit failures in `vcg_processing_failures.csv`:

| Recording | Patient | Diagnosis | Logged error |
| --- | --- | --- | --- |
| `patient007/s0029lre` | patient007 | MI | `ValueError: Loop không hợp lệ: (2, 3)` |
| `patient010/s0042lre` | patient010 | MI | `ValueError: Loop không hợp lệ: (2, 3)` |

Both patients retain other recordings in the feature table, so all 200 eligible patients are represented. Both failures belong to the original training patient set. Applying original patient assignments to all 448 eligible recordings yields 333 train and 115 test records. They must remain attempted records and be assigned by patient when a new extraction is run; they are not unknown labels or absent patients.

A new validation partition can be formed without changing the original test set by running `train_test_split(original_train_patients, test_size=0.20, random_state=42, stratify=original_train_patients['label'])`, preserving the returned original training row order. This yields:

| Partition | Patients MI/HC | All eligible records MI/HC |
| --- | ---: | ---: |
| New train | 89 / 31 | 216 / 47 |
| New validation | 22 / 8 | 54 / 16 |
| Preserved test | 37 / 13 | 98 / 17 |

The resulting patient proportions are 60/15/25. These are proposed new train/validation assignments with the old test preserved; they must not be described as a previously existing three-way split. Exact ordered and sorted patient IDs are in `reference_audit.json`.

Notebook 05 also contains an unrelated synthetic detector-window experiment: 45 generated subjects, two recordings per subject, split into 30 validation subjects (60 cases) and 15 holdout subjects (30 cases). Its 810 detail rows are 90 generated cases times 9 windows. Those synthetic assignments are not PTB patient splits. The PTB stability table has 4032 rows, or 448 records times 9 windows, and measures detector agreement rather than reference accuracy.

## Independent delineation references

No independent beat or QRS/T boundary annotation file is present in the local PTB v1.0.0 tree: no `ANNOTATORS`, `.atr`, `.qrs`, `.qt*`, or manual boundary table was found. Headers provide signal calibration and clinical metadata, rather than manually timed waveform boundaries. Root `vcg_processing_qc.csv` and `vcg_v4_results/qc.csv` are algorithm outputs. They must not become reference labels for evaluating the same detector.

The published original PTB record directory likewise lists signal/header files and clinical metadata rather than a waveform-boundary annotation release. This supports the limited conclusion that **the original PTB v1.0.0 package and this workspace do not supply independent four-boundary labels**; it does not claim that no later PTB-related annotation resource exists. [PTB release](https://physionet.org/content/ptbdb/1.0.0/), [example original recording directory](https://physionet.org/content/ptbdb/1.0.0/patient001/)

### PhysioNet/CinC Challenge 2006: same records, different endpoint

The 2006 QT challenge reuses the same 549 PTB recordings, so it is relevant to this inventory. Its measurements concern a representative beat chosen by each observer in **lead II**. Participants supplied absolute recording times for QRS/PQ onset and T end, in milliseconds. The challenge instructed observers to choose the first representative beat, but different observers could choose different beats. [Challenge specification](https://physionet.org/content/challenge-2006/1.0.0/), [measurement submission schema](https://physionet.org/content/challenge-2006/1.0.0/template.txt)

`reference-QT.txt` has four fields: recording name, median QT in milliseconds, mean absolute difference from that median (MAD), and number of contributing observers. Its reference is the median duration from 15 manual/semi-automated division-1 entrants, with variable observer coverage per record. It contains neither absolute reference sample indices nor QRS offset/T onset. [Reference file and field definitions](https://physionet.org/content/challenge-2006/1.0.0/reference-QT.txt)

Consequently, record identity aligns, but the reference endpoint does not directly match a median-beat Frank-XYZ QRS/T segmentation. A comparison of `VCG T_off - QRS_on` against those lead-II typical QT durations can only be reported as **cross-definition duration agreement**, with the lead, beat-selection, and aggregation mismatch stated. It is not four-boundary delineation accuracy, cannot supply beat TP/FP/FN, and cannot independently assess QRS offset or T onset. A valid boundary assessment would need an explicit same-record, same-beat, time-base mapping, a declared compatible boundary definition, and independent boundary annotations.

The challenge also publishes submitted PQ/T-end measurement files (including revisions). Their directory was verified, but individual measurement files were not retrievable through the web tool during this audit. The reference file and submission template were readable. No unverified raw participant measurements were imported or used as labels. [Published measurement directory](https://www.physionet.org/content/challenge-2006/1.0.0/measurements/)

PTB-XL+ fiducial annotations refer to PTB-XL recording IDs and are computed by ECGDeli. They are neither matching original PTB Frank-XYZ recordings nor independent manual boundary labels for these records. They cannot be silently substituted. [PTB-XL+ fiducial-point description](https://physionet.org/content/ptb-xl-plus/1.0.0/)

## Quality-control and accuracy limits

Existing `vcg_v4_results/qc.csv` has 448 attempted records: 109 `auto_candidate`, 335 `needs_review`, and 4 `failed`. Its 444 candidate-feature rows include 364 MI and 80 HC; its 109 QC-pass rows include 79 MI and 30 HC. These are historical processing/QC counts, not a new evaluation. The large and class-dependent selection means a classifier restricted to QC passes can describe only that selected subset unless exclusions, coverage, and sensitivity analyses are reported.

Ordered boundaries, physiological duration ranges, visible plausible loops, successful feature extraction, and agreement between implementations are useful QC. None alone establishes true boundary accuracy. Without compatible independent reference boundaries, boundary MAE/RMSE, sensitivity, precision, F1, or percentage-correct must be marked unavailable. Synthetic-reference detector scores must remain labeled synthetic.

HC/MI classification can still be evaluated against the independent header diagnoses on held-out patients. Freeze test patient IDs before modeling, fit imputation/scaling/feature selection on train only, choose parameters/thresholds on validation only, count every attempted record, and report diagnosis-specific failures/review/coverage. Median-beat and record-level predictions are correlated within a patient; confidence intervals and patient-level aggregation should preserve patient groups.

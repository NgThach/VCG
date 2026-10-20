# PTB VCG — QRS/T loops và thí nghiệm Wavelet/TLS

Dự án xử lý ba kênh Frank **X/Y/Z** của PTB, tạo nhịp đại diện bằng trung vị,
tách QRS và T, trích xuất 12 đặc trưng và so sánh cách tìm biên T cũ (V1) với
Wavelet/TLS (V2). Phạm vi MI/HC hiện có **448 bản ghi / 200 bệnh nhân**.

Repo chứa mã nguồn, 9 notebook, các kiểm thử, tài liệu và danh sách chia tập.
Dữ liệu gốc, cache, checkpoint, hình và gallery được tạo lại trên máy chạy.
Các notebook lưu trong Git gồm mã và Markdown; output được bỏ khi commit để
Notebook08 không vượt giới hạn file GitHub. File notebook có hình trên máy gốc
vẫn giữ nguyên.

## 1. Cài đặt

Dùng **Python 3.11 trở lên**. Các phiên bản trong `requirements.txt` là phiên bản
của môi trường đã dùng cho dự án.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ptb-vcg --display-name "PTB VCG"
```

Chạy các lệnh dưới đây từ thư mục gốc repo. Trên Windows, thay lệnh kích hoạt
môi trường bằng `.venv\Scripts\activate`.

## 2. Tải dữ liệu PTB

Nguồn: [PTB Diagnostic ECG Database 1.0.0](https://physionet.org/content/ptbdb/1.0.0/).
Dữ liệu chiếm khoảng 1.8 GB khi giải nén. Lệnh sau tải vào đúng đường dẫn mà
notebook và pipeline sử dụng, giữ cấu trúc thư mục bệnh nhân:

```bash
python - <<'PY'
from pathlib import Path
import wfdb

destination = Path("content/ptbdb/physionet.org/files/ptbdb/1.0.0")
destination.mkdir(parents=True, exist_ok=True)
wfdb.dl_database("ptbdb", str(destination), keep_subdirs=True)
PY
```

Nếu đã có dữ liệu trên máy, đặt hoặc liên kết thư mục chứa `patient001/`,
`patient002/`, ... vào vị trí trên. Cần cả `.hea`, `.dat` và `.xyz` mà header trỏ tới.

## 3. Tạo dữ liệu V1 và chạy luồng chính

Các bước dưới dùng dữ liệu tải về và tạo thư mục `outputs/` mới. Không cần chạy
lại thí nghiệm phân loại cũ ở Notebook02. `data/split_manifest.csv` giữ đúng
phân chia theo bệnh nhân hiện tại; các CSV `vcg_*` ở root được giữ làm đầu vào
đối chiếu lịch sử và kiểm tra nguồn.

### Bước A — Tạo median, biên QRS/T và kết quả V1

```bash
python scripts/run_full_vcg_evaluation.py --workers 4 --split-manifest data/split_manifest.csv
```

Kết quả: `outputs/vcg_full_evaluation/`, gồm median XYZ và biên đã lưu. Nếu lần
chạy bị ngắt, thêm `--resume` với cùng cấu hình và danh sách chia tập.

### Bước B — Tạo các bảng train/valid/test và kiểm tra QC

Chạy notebook thành **bản có tiền tố `executed_`** để giữ nguyên các file nguồn
được pipeline kiểm tra bằng hash. Bản thực thi nằm tại root, nên các liên kết
gallery tương đối vẫn hoạt động.

```bash
jupyter nbconvert --to notebook --execute 06_PTB_VCG_12_Features_Train_Valid_Test.ipynb --output executed_06 --ExecutePreprocessor.kernel_name=ptb-vcg --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute 07_VCG_QC_vs_Delineation_Audit.ipynb --output executed_07 --ExecutePreprocessor.kernel_name=ptb-vcg --ExecutePreprocessor.timeout=-1
```

Kết quả: `outputs/vcg_12_features_train_valid_test/` và `outputs/vcg_qc_audit_v1/`.
QC là cờ yêu cầu xem lại, không phải nhãn xác nhận biên sai.

### Bước C — Chạy thí nghiệm tìm biên T bằng Wavelet/TLS

```bash
python scripts/run_vcg_t_almeida_experiment.py
```

Kết quả: `outputs/vcg_t_almeida_v2_experiment/`. Mỗi ID có trạng thái, mốc T,
lý do thiếu, các vòng TLS và T loop nếu tìm được đủ hai biên hợp lệ. Các ca
thất bại vẫn có trong danh sách và gallery. Mở giao diện tại:

```text
outputs/vcg_t_almeida_v2_experiment/gallery/index.html
```

### Bước D — Tạo phần trực quan của tất cả bản ghi trong Notebook08

Chạy bước này **sau B và C**, vì Notebook08 đọc kết quả của cả V1 lẫn V2.

```bash
jupyter nbconvert --to notebook --execute 08_QRS_T_Loop_Reference_Error_Analysis.ipynb --output executed_08 --ExecutePreprocessor.kernel_name=ptb-vcg --ExecutePreprocessor.timeout=-1
jupyter lab
```

Mở `executed_08.ipynb`: mục **14C** là gallery V1/V2; **14D–14E** giải thích
từng bước và các ca cần xem lại; **14F** nhúng 75 trang hình đủ 448 ID, gồm cả
ca thiếu biên/median. Bản HTML xem toàn bộ các trang:

```text
outputs/vcg_t_almeida_v2_experiment/notebook08_extension/all_record_t_loops/index.html
```

Notebook09 giải thích Wavelet/TLS và từng thuật ngữ bằng tiếng Việt. Có thể
đọc mã/Markdown trực tiếp, hoặc tạo bản có hình sau bước C:

```bash
jupyter nbconvert --to notebook --execute 09_VCG_T_Loop_Almeida_Wavelet_TLS_Experiment.ipynb --output executed_09 --ExecutePreprocessor.kernel_name=ptb-vcg --ExecutePreprocessor.timeout=-1
```

Khi thay thuật toán, cấu hình hoặc nguồn dữ liệu, dùng thư mục kết quả mới theo
các tùy chọn của runner. Không ghép checkpoint thuộc các phiên bản khác nhau.

## 4. Các file cần đọc

- `vcg_pipeline.py`, `vcg_preprocessing.py`: các bước xử lý và tách biên V1.
- `vcg_t_almeida.py`: phép biến đổi wavelet và hai nhánh tìm T-on/T-off bằng TLS.
- `scripts/`: chạy toàn bộ dữ liệu, báo cáo, giao diện xem và vẽ hình.
- `tests/`: các phép kiểm thử đã có; chạy riêng khi cần kiểm tra mã.
- `METHOD_REPRODUCTION.md`: nguồn công thức, phần điều chỉnh và giới hạn.
- `PREPROCESSING_UPGRADE.md`: thay đổi tiền xử lý so với phiên bản cũ.
- Notebook01–05: khảo sát dữ liệu và các bước phát triển trước luồng hiện tại.

## 5. Giới hạn cần giữ khi diễn giải

Median được lấy theo từng thời điểm của các nhịp đã căn QRS; đặc điểm chỉ có ở
ít nhịp có thể bị làm yếu. Tạo được loop hoặc đủ đặc trưng không xác nhận biên
đúng, cũng chưa chứng minh thông tin phân biệt MI/HC được giữ đầy đủ.

V2 là bản dựng theo bài báo có các điều chỉnh được công bố, chưa xác nhận tương
đương toàn bộ mã tác giả. Hiện chưa có bộ mốc độc lập phù hợp để khẳng định V2
chính xác hơn V1. Đây là mã nghiên cứu, không phải công cụ chẩn đoán.

## 6. Làm việc với Git

Sau khi clone, bật bộ lọc để các lần commit tiếp theo giữ mã và Markdown của
notebook, còn hình/kết quả vẫn nằm trong bản làm việc trên máy:

```bash
git config filter.notebook-clean.clean "python3 .git-tools/clean_notebook.py"
git config filter.notebook-clean.smudge cat
git config filter.notebook-clean.required true
```

Giấy phép và trích dẫn nguồn nằm trong `THIRD_PARTY_NOTICES.md`; bản văn GPL v2
cho thành phần có khai báo GPL-2.0-only nằm trong `LICENSE`.

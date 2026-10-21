# PTB VCG — QRS/T loops và thí nghiệm Wavelet/TLS

Dự án xử lý ba kênh Frank **X/Y/Z** của PTB, tạo nhịp đại diện bằng trung vị,
tách QRS và T, trích xuất 12 đặc trưng và so sánh cách tìm biên T cũ (V1) với
Wavelet/TLS (V2). Phạm vi MI/HC hiện có **448 bản ghi / 200 bệnh nhân**.

Repo chứa mã nguồn, 9 notebook, kiểm thử và danh sách chia tập theo bệnh nhân.
**Dataset đầy đủ đã được đính kèm tại [GitHub Release data-v1](https://github.com/NgThach/VCG/releases/tag/data-v1).**
`git clone` tải mã nguồn; lệnh chuẩn bị bên dưới tải dataset và kiểm checksum.
Các kết quả trung gian được tính lại tại máy chạy, không phụ thuộc cache trên máy tác giả.

## 1. Clone và cài đặt

Đã kiểm tra bằng **Python 3.11**. Chạy từ thư mục gốc repo:

```bash
git clone https://github.com/NgThach/VCG.git
cd VCG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ptb-vcg --display-name "PTB VCG"
```

Windows: kích hoạt bằng `.venv\Scripts\activate`. Trong Jupyter chọn kernel
**PTB VCG** để dùng đúng thư viện vừa cài.

## 2. Tải và kiểm tra dataset

```bash
python project_setup.py --verify
```

Lệnh tải 41 phần nhỏ, tự ghép thành gói **1,35 GB**, kiểm SHA-256 của gói và từng file sau giải nén,
đặt đúng vị trí `content/ptbdb/physionet.org/files/ptbdb/1.0.0/`.
Có **549 `.hea` + 549 `.dat` + 549 `.xyz`**, `RECORDS`, `README`, `CONTROLS`
và `SHA256SUMS.txt` từ PhysioNet. Không cần đăng nhập GitHub khi tải release công khai.
Cần kết nối Internet lần tải đầu và khoảng **8 GB dung lượng trống** cho dữ liệu,
gói tải, kết quả và môi trường Python.

Nguồn: [PTB Diagnostic ECG Database 1.0.0](https://physionet.org/content/ptbdb/1.0.0/).
Giấy phép/trích dẫn ở `THIRD_PARTY_NOTICES.md`.
Nếu đã có dữ liệu, đặt đủ file tại đường dẫn trên rồi chạy `--verify`;
chương trình không ghi đè file có checksum khác.

## 3. Chạy notebook

```bash
jupyter lab
```

Mỗi notebook có cell **Chuẩn bị dữ liệu trên máy này** ngay đầu file.
Chọn **Run All**. Cell này tải dữ liệu nếu thiếu và tạo các đầu vào cần thiết:

- **01–05:** đọc PTB gốc, không cần output từ notebook khác.
- **06:** tạo median/artifact V1 cho đủ 448 ID rồi xuất train/valid/test.
- **07:** tự chạy 06 nếu chưa có bảng features; tạo kiểm tra QC và mẫu review.
- **08:** tự tạo đầu vào của 06–07 và kết quả Wavelet/TLS nếu thiếu; hiển thị đủ 448 ID.
- **09:** tự tạo đầu vào V1 của 06; chạy và giải thích thí nghiệm Wavelet/TLS.

**Lần chạy đầu của 06–09 có thể kéo dài nhiều phút** vì tính trên toàn bộ cohort
và tạo hàng trăm hình. Dòng tiến độ hiển thị bước đang chạy. Giữ nguyên code/cấu hình
để tiếp tục dùng checkpoint sau khi ngắt; khi thay thuật toán, dùng thư mục kết quả mới.

Có thể chuẩn bị trước khi mở riêng Notebook08:

```bash
python project_setup.py --notebook 8 --verify
```

Hoặc chạy không qua giao diện, dùng đúng Python đang kích hoạt và lưu output riêng:

```bash
python scripts/execute_notebook.py 08_QRS_T_Loop_Reference_Error_Analysis.ipynb
python scripts/execute_notebook.py 09_VCG_T_Loop_Almeida_Wavelet_TLS_Experiment.ipynb
```

Kết quả là `executed_08_*.ipynb`, `executed_09_*.ipynb` ở root; log và thống kê
ở `outputs/reproducibility/`. Chạy lần lượt **01 → 09** nếu muốn kiểm tra toàn bộ lịch sử.
Notebook02 ghi lại các CSV legacy ở root; nên chạy trước khi đóng băng kết quả 07–09.

Notebook08 mục **14F** nhúng 75 trang hình bao phủ 448 ID, kể cả thiếu median/biên.
Gallery Wavelet/TLS: `outputs/vcg_t_almeida_v2_experiment/gallery/index.html`.
Notebook09 giải thích các bước và thuật ngữ bằng tiếng Việt.

Chi tiết đầu vào, thứ tự phụ thuộc, kiểm chứng và giới hạn:
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

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

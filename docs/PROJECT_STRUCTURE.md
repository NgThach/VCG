# Cấu trúc dự án sau khi dọn

## Các phần đang dùng

- **Notebook 01–11:** nội dung khảo sát, xử lý, trực quan và các thí nghiệm V3/V4.
- **`data/features/v1/` và `data/features/v2/`:** các bảng VCG features hiện tại,
  mỗi bộ gồm train/valid/test. Đã có sẵn trong Git, không cần tải tín hiệu PTB.
- **`examples/Read_VCG_Features.ipynb`:** chỉ đọc các bảng features và tạo X/y;
  có thể mở riêng trên Colab. Xem `data/features/README.md` để chọn V1/V2.
- **Bốn module ở root:** `vcg_preprocessing.py`, `vcg_pipeline.py`,
  `vcg_preprocessing_evaluation.py`, `vcg_t_almeida.py`.
- **Chuẩn bị dữ liệu trong bản GitHub:** `project_setup.py`, `project_integrity.py`,
  `data/` và `requirements.txt`.
- **`scripts/execute_notebook.py`:** runner chung cho mọi notebook, thay runner riêng cho 09.
- **Ba runner pipeline:** `run_full_vcg_evaluation.py`, `run_vcg_qc_audit_stage.py`,
  `run_vcg_t_almeida_experiment.py`.
- **Các module `scripts/vcg_*.py` còn lại:** notebook và runner import để kiểm tra QC,
  tính metric, chuẩn bị review, vẽ hình và tạo gallery. Chúng là thư viện đang dùng.
- **`scripts/compare_vcg_annotations.py`:** so sánh annotation với dự đoán; được phần
  báo cáo/QC dùng trực tiếp.
- **`scripts/reference_audit.json` và `.md`:** runner V1 đọc và sao lưu để ghi nguồn dữ liệu.
- **`tests/`:** các bộ kiểm thử hiện có.

## Đã xoá 11 script

- `pipeline_core.py`, `test_run.py`, `create_visualizations.py`,
  `render_vcg_comparison.py`: các bản pipeline/vẽ thử độc lập; phần nghiên cứu lịch sử
  đã có trong Notebook02–03, không có notebook/module hiện tại gọi các file này.
- `scripts/create_notebook08.py`, `scripts/extend_notebook08_almeida.py`:
  công cụ tạo/chèn notebook dùng trước đây; Notebook08 đã chứa nội dung hoàn chỉnh.
  Đã gỡ yêu cầu sao lưu generator khỏi bước lưu nguồn của Notebook08.
- `scripts/execute_notebook09.py`: trùng chức năng với runner chung.
- `scripts/check_vcg_t_almeida.py`, `scripts/check_vcg_t_almeida_gallery.mjs`:
  script kiểm tra một lần; Notebook09 và `tests/` vẫn chứa các phép kiểm đang dùng.
- `scripts/prepare_vcg_blind_review.py`, `scripts/finalize_vcg_evaluation.py`:
  tiện ích của luồng báo cáo cũ; luồng hiện tại dùng Notebook07 để tạo QC/review
  và Notebook08 để xem/chú giải/đánh giá.

Các thư mục cache Python và file Finder ở thư mục mã nguồn cũng đã được dọn.
Các bản nguồn trong `outputs/` hoặc backup nghiên cứu là dấu vết của lần chạy cũ;
không được import làm script hoạt động. Bước lưu nguồn của Notebook08 tạo phiên bản
snapshot mới khi nguồn thay đổi, giữ nguyên snapshot cũ đã được đóng băng.

## Rà soát sau khi dọn

Đã rà tĩnh import, lệnh gọi script trong các notebook lịch sử 01–09 và các danh sách file nguồn.
Cả 23 file Python còn trong `scripts/` đều thuộc chuỗi phụ thuộc của notebook/runner.
Không còn tham chiếu mã chạy tới 11 file đã xoá; các file nguồn yêu cầu lưu snapshot
đều tồn tại và toàn bộ Python còn lại phân tích cú pháp được.

Đợt dọn này không chạy lại bộ tests hoặc notebook. Kết quả thực thi trong
`VERIFICATION.md` thuộc lượt kiểm tra trước khi dọn.

## Dọn các đầu ra không còn được đọc — 21/10/2026

Đã xoá 2.530 file, tổng cộng 604.600.436 byte (khoảng 605 MB) tại workspace:

- Hai bảng overview do Notebook01 xuất ở root: `ptb_records_overview.csv` và
  `ptb_patients_overview.csv`.
- Các kết quả xuất độc lập: `output/`, `vcg_preprocessing_results/`,
  `vcg_window_validation/`, `vcg_v4_results/` và các bản sao cũ trong `backups/`.
- Các lần chạy cũ/thử: `outputs/vcg_full_evaluation_runs/`,
  `outputs/vcg_full_evaluation_smoke/` và `outputs/vcg_t_almeida_v2_experiment/smoke/`.
- Runtime Jupyter tạm, ảnh kiểm tra trong `gallery/qa/`, ảnh xuất riêng trong
  `figures/walkthrough/`, các bản notebook tự sao lưu, file cập nhật và ảnh preview
  cũ trong `outputs/vcg_t_almeida_v2_experiment/notebook08_extension/`.

Các file này không được notebook/module hiện tại đọc làm đầu vào và không nằm trong
danh sách checksum được bảo vệ. Chúng đã bị Git bỏ qua từ trước; không có dataset
hay bảng feature nào bị gỡ khỏi GitHub. Notebook có thể xuất lại các bảng/hình khi chạy.

Giữ nguyên `data/features/`, dữ liệu PTB, median/artifacts, các CSV đầu vào, kết quả
V1 đã đóng băng, gallery chính và `notebook08_extension/all_record_t_loops/` đang được
Notebook08 hiển thị. Ba CSV legacy ở root cũng còn được dùng để tái lập split và
kiểm tra nguồn, nên vẫn cần giữ.

Đã đối chiếu SHA-256 trước/sau: 9.406 file nguồn, feature và kết quả còn được dùng
không thay đổi. Lượt dọn này không chạy lại notebook hoặc bộ tests.

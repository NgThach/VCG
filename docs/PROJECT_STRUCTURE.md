# Cấu trúc dự án sau khi dọn

## Các phần đang dùng

- **Notebook 01–09:** nội dung khảo sát, xử lý và trực quan của dự án.
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

Đã rà tĩnh import, lệnh gọi script trong cả 9 notebook và các danh sách file nguồn.
Cả 23 file Python còn trong `scripts/` đều thuộc chuỗi phụ thuộc của notebook/runner.
Không còn tham chiếu mã chạy tới 11 file đã xoá; các file nguồn yêu cầu lưu snapshot
đều tồn tại và toàn bộ Python còn lại phân tích cú pháp được.

Đợt dọn này không chạy lại bộ tests hoặc notebook. Kết quả thực thi trong
`VERIFICATION.md` thuộc lượt kiểm tra trước khi dọn.

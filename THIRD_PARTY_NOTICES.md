# Nguồn và giấy phép

## Phương pháp Wavelet/TLS

`vcg_t_almeida.py` có khai báo `SPDX-License-Identifier: GPL-2.0-only`.
Bản văn GPL phiên bản 2 được cung cấp trong `LICENSE`. Các công thức và mã tham
khảo được ghi rõ trong `METHOD_REPRODUCTION.md`:

- Martínez et al. (2004), *A Wavelet-Based ECG Delineator: Evaluation on Standard Databases*,
  [DOI: 10.1109/TBME.2003.821031](https://doi.org/10.1109/TBME.2003.821031).
- Almeida et al. (2009), *Multilead ECG Delineation Using Spatially Projected Leads From Wavelet Transform Loops*,
  [DOI: 10.1109/TBME.2009.2021658](https://doi.org/10.1109/TBME.2009.2021658).
- ECG-kit / wavedet, mã tham khảo tại commit
  `c8e3de47c54a9214138143676d2aa546b0540dd2`, có tên tác giả
  J. P. Martínez và R. Almeida, giấy phép GPL v2:
  [nguồn](https://github.com/marianux/ecg-kit/tree/c8e3de47c54a9214138143676d2aa546b0540dd2/common/wavedet).

Đây là bản dựng theo công thức và có các điều chỉnh được công bố trong tài liệu
phương pháp; chưa xác nhận tương đương từng đầu ra với chương trình tác giả.

## Dữ liệu PTB

Dữ liệu gốc không được lưu trong Git. Tải từ
[PTB Diagnostic ECG Database 1.0.0](https://physionet.org/content/ptbdb/1.0.0/),
theo điều kiện sử dụng và giấy phép Open Data Commons Attribution License v1.0
được công bố tại nguồn. Các CSV mang tiền tố `vcg_` và danh sách chia tập là
kết quả/danh sách dẫn xuất từ dữ liệu này; không phải mốc QRS/T do chuyên gia xác nhận.

Trích dẫn dữ liệu:

- Bousseljot R., Kreiseler D., Schnabel A. (1995). Nutzung der EKG-Signaldatenbank
  CARDIODAT der PTB über das Internet. *Biomedizinische Technik*, 40, Supplement 1, 317.
- Goldberger A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
  Components of a New Research Resource for Complex Physiologic Signals.
  *Circulation*, 101(23), e215–e220.

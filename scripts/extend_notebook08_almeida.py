"""Add the saved Wavelet/TLS experiment to Notebook08 without rerunning V1.

The extension reads existing Notebook09 results. Existing cells, outputs and
manual reviews are retained. Only new cells are executed to embed their figures.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import textwrap

os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'vcg_t_almeida_mpl'))
os.environ.setdefault('IPYTHONDIR', str(__import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'vcg_qc_audit_ipython'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TARGET = ROOT / '08_QRS_T_Loop_Reference_Error_Analysis.ipynb'
EXPERIMENT = ROOT / 'outputs/vcg_t_almeida_v2_experiment'
AUDIT = EXPERIMENT / 'notebook08_extension'
TAG = 'nb08-almeida-wavelet-tls'
INTRO = '''
**Tách T bằng Wavelet/TLS đã được bổ sung ở [mục 14C–14F](#nb08-wavelet-tls):**
gallery V1/V2 cho đủ 448 ID, các bước WT → single-lead → TLS → T-on/T-off,
hai T loops 3D và các ca chưa giải được. **Mục 14F nhúng hình của tất cả 448 ID**,
mỗi ca có XYZ/biên T và T loops V1/V2. Có thể đổi `ALMEIDA_RECORD_ID` ở mục 14D
để xem chi tiết các bước tính của bản ghi khác.
'''.strip()


def sha(path):
    from project_integrity import file_digest
    return file_digest(path)


def build_extension_cells():
    cells = []

    def add(kind, source):
        factory = nb.v4.new_markdown_cell if kind == 'markdown' else nb.v4.new_code_cell
        cell = factory(textwrap.dedent(source).strip())
        cell.metadata['tags'] = [TAG]
        cells.append(cell)

    def md(source):
        add('markdown', source)

    def code(source):
        add('code', source)

    md('''
    <a id="nb08-wavelet-tls"></a>

    ## 14C. T loops mới: gallery Wavelet/TLS và đối chiếu V1/V2

    Phần này đưa trực tiếp kết quả thí nghiệm Notebook 09 vào Notebook 08.
    Mở gallery bên dưới để tìm ID, chuyển Previous/Next, lọc nhãn/split/trạng thái,
    xem XYZ, WT ở scale 3/4/5, single-lead candidates, từng vòng TLS và T loops 3D.
    Gallery giữ đủ **448 ID**, kể cả những ca thiếu median hoặc thiếu biên T.

    Dữ liệu hiển thị lấy từ kết quả đã lưu của cấu hình Wavelet/TLS hiện tại.
    QRS, median Frank XYZ, gốc PQ và hệ thời gian vẫn theo V1. Các biên T V1 chỉ
    dùng đối chiếu sau khi V2 đã được tính độc lập; phần này không tính lại V1.

    Đây là bản dựng theo công thức Almeida/Martínez có điều chỉnh đã công bố,
    **chưa xác minh tương đương hoàn toàn mã tác giả**. Xem
    [Notebook 09](09_VCG_T_Loop_Almeida_Wavelet_TLS_Experiment.ipynb) và
    [truy vết phương pháp](METHOD_REPRODUCTION.md) để đọc nguồn và giới hạn.
    Chênh V2−V1 là bất đồng phương pháp; chưa có reference độc lập được chấp nhận
    để kết luận cách nào chính xác hơn.
    ''')
    code('''
    from pathlib import Path
    import sys, json
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown, Image, IFrame

    almeida_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                       if (p/'vcg_t_almeida.py').is_file())
    if str(almeida_root/'scripts') not in sys.path:
        sys.path.insert(0, str(almeida_root/'scripts'))
    if str(almeida_root) not in sys.path:
        sys.path.insert(0, str(almeida_root))
    almeida_out = almeida_root/'outputs/vcg_t_almeida_v2_experiment'
    almeida_required = ['v2_t_boundaries.csv', 'split_manifest.csv', 'config.json',
                        'execution_summary.json', 'gallery/index.html', 'gallery/manifest.csv']
    almeida_missing = [p for p in almeida_required if not (almeida_out/p).is_file()]
    if almeida_missing:
        raise FileNotFoundError('Thiếu kết quả Notebook 09: '+', '.join(almeida_missing)+
            '. Chạy scripts/run_vcg_t_almeida_experiment.py trước.')
    almeida_results = pd.read_csv(almeida_out/'v2_t_boundaries.csv')
    almeida_manifest = pd.read_csv(almeida_out/'split_manifest.csv')
    almeida_gallery_manifest = pd.read_csv(almeida_out/'gallery/manifest.csv')
    almeida_config = json.loads((almeida_out/'config.json').read_text())
    almeida_summary = json.loads((almeida_out/'execution_summary.json').read_text())
    almeida_coverage = pd.read_csv(almeida_out/'reports/coverage_by_label_and_split.csv')
    if (not almeida_results.record_id.is_unique or len(almeida_results) != 448
            or set(almeida_results.record_id) != set(almeida_manifest.record_id)
            or set(almeida_results.record_id) != set(almeida_gallery_manifest.record_id)):
        raise ValueError('Kết quả/gallery V2 không khớp đầy đủ manifest 448 ID.')
    if not almeida_results.algorithm_fingerprint.eq(almeida_config['algorithm_fingerprint']).all():
        raise ValueError('Kết quả V2 khác cấu hình đã lưu; không trộn các lần chạy.')
    display(pd.Series({
        'recordings': len(almeida_results),
        'median_available': int(almeida_results.median_present.sum()),
        'T_on_V2_available': int(almeida_results.t_on_v2_ms.notna().sum()),
        'T_off_V2_available': int(almeida_results.t_off_v2_ms.notna().sum()),
        'T_loops_V2': int(almeida_results[['t_on_v2_ms','t_off_v2_ms']].notna().all(axis=1).sum()),
        'unresolved': int(almeida_results.status.eq('unresolved').sum()),
    }, name='Số bản ghi').to_frame())
    display(almeida_coverage)
    display(Markdown(
        '[Mở gallery Wavelet/TLS V1–V2 toàn bộ 448 ID trong tab riêng]'
        '(outputs/vcg_t_almeida_v2_experiment/gallery/index.html). '
        'Các tab: Signals → Wavelets → Single-lead → TLS → Loops → History. '
        'Trong tab TLS chọn nhánh T-on/T-off và iteration; kéo hình 3D để xoay. '
        'Nếu iframe bị chặn trong trình xem notebook, dùng liên kết mở tab riêng.'))
    display(IFrame(src='outputs/vcg_t_almeida_v2_experiment/gallery/index.html',
                   width='100%', height=1120))
    ''')
    md('''
    ## 14D. Xem trực tiếp từng bước tách T bằng Wavelet/TLS

    Đổi `ALMEIDA_RECORD_ID` ở cell dưới, rồi chạy từ đây tới hết mục 14D.
    Mặc định dùng `patient104/s0306lre`, cũng là walkthrough của Notebook 09.
    Muốn chuyển liên tục giữa mọi bản ghi mà không chạy lại cell, dùng gallery 14C.

    **Bước 1 — Median XYZ và khoảng T:** nét đứt xám là biên T V1; hai đường
    T-on/T-off V2 là xanh lam/đỏ. Vùng tô là khoảng cắt thực tế của mỗi cách.
    Trục thời gian là `(sample−anchor)×1000/fs`; chỉ số sample thuộc median,
    không phải recording raw. QRS-on/off vẫn là biên V1.
    ''')
    code('''
    from vcg_t_almeida_plots import (
        load_case as almeida_load_case, plot_median as almeida_plot_median,
        plot_wavelets as almeida_plot_wavelets, plot_single_leads as almeida_plot_single_leads,
        plot_wt_loop as almeida_plot_wt_loop, plot_tls_fit as almeida_plot_tls_fit,
        plot_derived as almeida_plot_derived, plot_iterations as almeida_plot_iterations,
        plot_t_loops as almeida_plot_t_loops)
    import matplotlib.pyplot as almeida_plt
    # Module vẽ batch dùng Agg; bật formatter sau import để nhúng PNG trong notebook.
    from matplotlib_inline.backend_inline import configure_inline_support, set_matplotlib_formats
    configure_inline_support(get_ipython(), 'module://matplotlib_inline.backend_inline')
    set_matplotlib_formats('png')

    ALMEIDA_RECORD_ID = 'patient104/s0306lre'
    almeida_case = almeida_load_case(almeida_out, ALMEIDA_RECORD_ID)
    almeida_diag = almeida_case['diag']
    almeida_columns = ['record_id','label_name','split','status',
        'qrs_on_ms','qrs_off_ms','t_on_v1_ms','t_off_v1_ms','t_duration_v1_ms',
        't_on_v2_ms','t_off_v2_ms','t_duration_v2_ms','failure_reason',
        't_on_failure_reason','t_off_failure_reason']
    display(almeida_results.set_index('record_id', drop=False).loc[
        [ALMEIDA_RECORD_ID], almeida_columns])
    if almeida_case['median']['xyz'] is not None:
        almeida_time = almeida_case['median']['time_ms']
        almeida_window = almeida_diag.get('search_window')
        print('fs:', almeida_case['median']['fs'], '| anchor:', almeida_case['median']['anchor'])
        print('T search samples:', almeida_window,
              '| ms:', almeida_time[almeida_window] if almeida_window else 'unavailable')
        print('ML scale:', almeida_diag.get('selected_scale'),
              '| initial TLS window:', almeida_diag.get('initial_window'))
    almeida_fig = almeida_plot_median(almeida_case, comparison=True)
    display(almeida_fig); almeida_plt.close(almeida_fig)
    ''')
    md(r'''
    ### Bước 2 — Wavelet đa scale và khởi tạo riêng X/Y/Z

    Bank quadratic spline không decimation được áp dụng lên cả ba trục.
    Cực trị WT biểu diễn các sườn; zero crossing hỗ trợ xác định đỉnh T.
    Hình đầu cho scale 3/4/5, cực trị được chọn/bỏ và ngưỡng; hình sau cho
    T-on/T-peak/T-off ứng viên riêng trên X, Y, Z cùng kiểu hình thái và lý do thiếu.
    Các scale là mức của bank thiết kế, không phải cửa sổ 8/16/32 ms.

    Ba loại ngưỡng có vai trò khác nhau: phát hiện `0.25 × RMS(W)`, chọn sườn
    `0.125 × max|W|`, và tìm biên tương đối theo biên độ sườn. Khởi tạo đơn trục
    dùng tỷ lệ onset 0.25, offset 0.4; nhánh multilead offset dùng 0.25.
    ''')
    code('''
    almeida_sl_rows = []
    for almeida_sl in almeida_diag.get('single_leads', []):
        almeida_sl_rows.append({k: almeida_sl.get(k) for k in [
            'axis','wavelet_scale','morphology_type','detection_status',
            't_on_candidate_sample','t_peak_candidate_sample','t_off_candidate_sample',
            'failure_reason']})
    if almeida_sl_rows:
        display(pd.DataFrame(almeida_sl_rows))
    else:
        print('Không có single-lead candidates:', almeida_case['row'].get('failure_reason'))
    for almeida_plot in (almeida_plot_wavelets, almeida_plot_single_leads):
        almeida_fig = almeida_plot(almeida_case)
        display(almeida_fig); almeida_plt.close(almeida_fig)
    ''')
    md(r'''
    ### Bước 3 — WT loop 3D, hai hướng TLS và derived wavelet

    WT loop là `[W_X, W_Y, W_Z]`. TLS fit đường thẳng affine trong cửa sổ W
    để tìm hướng `u`; derived wavelet là `W_d = W @ u`. Centering dùng trong fit,
    còn phép chiếu dùng WT gốc. Đây là TLS trên WT loop, không phải fit XYZ.

    Hai nhánh T-on/T-off lặp độc lập, cập nhật cửa sổ theo biên và sườn liên quan.
    Các hình dưới hiển thị vòng được chọn của từng nhánh, hướng fit, ngưỡng và
    candidate. Hình lịch sử cùng bảng ghi toàn bộ vòng lặp, cửa sổ, trạng thái và
    nguyên nhân dừng; gallery 14C cho phép chọn xem bất kỳ vòng nào.
    Khi nhánh không giải được, hình và bảng giữ nguyên lý do thiếu.
    ''')
    code('''
    almeida_fig = almeida_plot_wt_loop(almeida_case)
    display(almeida_fig); almeida_plt.close(almeida_fig)
    for almeida_boundary in ('t_on', 't_off'):
        almeida_branch = almeida_diag.get('branches', {}).get(almeida_boundary, {})
        display(Markdown('**'+almeida_boundary+'** — chosen iteration: '+
            str(almeida_branch.get('chosen_iteration'))+'; stop: '+
            str(almeida_branch.get('stop_reason', 'unavailable'))))
        almeida_fig = almeida_plot_tls_fit(almeida_case, boundary=almeida_boundary)
        display(almeida_fig); almeida_plt.close(almeida_fig)
        almeida_fig = almeida_plot_derived(almeida_case, boundary=almeida_boundary)
        display(almeida_fig); almeida_plt.close(almeida_fig)
    almeida_history = []
    for almeida_boundary, almeida_branch in almeida_diag.get('branches', {}).items():
        for almeida_entry in almeida_branch.get('history', []):
            almeida_history.append({k: almeida_entry.get(k) for k in [
                'boundary','iteration','window_start','window_end','direction',
                'fit_quality','extremum_sample','extremum_amplitude','boundary_threshold',
                'candidate_sample','accepted','status','stop_reason']})
    display(pd.DataFrame(almeida_history))
    almeida_fig = almeida_plot_iterations(almeida_case)
    display(almeida_fig); almeida_plt.close(almeida_fig)
    ''')
    md('''
    ### Bước 4 — T loops V1/V2 trên chính XYZ của median

    Sau khi có đủ hai biên hợp lệ, loop được cắt bằng `xyz[on:off+1]` trên median
    V1. Hai hình 3D dùng chung gốc PQ và tỉ lệ; ba hình chiếu XY/XZ/YZ giữ tỉ lệ
    bằng nhau. Chấm tròn là đầu, dấu x là cuối; mũi tên chỉ chiều thời gian.
    Nhánh thiếu một biên không tạo loop. QRS vẫn dùng khoảng V1 đã lưu.
    ''')
    code('''
    almeida_fig = almeida_plot_t_loops(almeida_case)
    display(almeida_fig); almeida_plt.close(almeida_fig)
    ''')
    md('''
    ## 14E. Các ca cần xem lại: đủ biên, T ngắn, thiếu biên và thiếu median

    Bốn ID dưới giống các ví dụ V1 trong Notebook 08 và Notebook 09. Bảng giữ
    cả biên từng phần và nguyên nhân thất bại; các hình so sánh bên dưới có cả
    ca unresolved. `NaN` là thiếu kết quả, không phải 0 ms.

    Đặc biệt, `patient051/s0179lre` có T V1 dài 17 ms; V2 mới tìm được T-on,
    chưa có T-off nên **chưa có T loop V2**. `patient034/s0109lre` vẫn chưa giải
    được cặp biên T; `patient033/s0157lre` thiếu median từ bước trước.
    ''')
    code('''
    almeida_example_ids = ['patient104/s0306lre','patient051/s0179lre',
                           'patient034/s0109lre','patient033/s0157lre']
    display(almeida_results.set_index('record_id', drop=False).loc[
        almeida_example_ids, almeida_columns])
    for almeida_rid in almeida_example_ids:
        display(Markdown('**'+almeida_rid+' — XYZ/biên T và T loops V1/V2**'))
        for almeida_kind in ('comparison', 'loops'):
            almeida_png = almeida_out/'figures'/(almeida_rid.replace('/','_')+'_'+almeida_kind+'.png')
            if not almeida_png.is_file():
                raise FileNotFoundError('Thiếu hình đã lưu: '+str(almeida_png))
            display(Image(filename=str(almeida_png), width=1150))
    display(Markdown(
        '**Kết quả cấu hình hiện tại:** '+str(almeida_summary['t_loops_available'])+
        ' T loops V2 / '+str(almeida_summary['median_available'])+' median khả dụng; '+
        str(almeida_summary['statuses'].get('unresolved', 0))+
        ' ID unresolved trên tổng '+str(len(almeida_results))+' ID. '
        'Coverage này không phải tỷ lệ tách đúng. '
        '[Xem đủ 448 ca trong gallery](outputs/vcg_t_almeida_v2_experiment/gallery/index.html) · '
        '[Báo cáo Wavelet/TLS](outputs/vcg_t_almeida_v2_experiment/reports/experiment_report.md).'))
    ''')
    md('''
    <a id="nb08-all-t-loops"></a>

    ## 14F. Tất cả 448 bản ghi: XYZ, biên T và T loops V1/V2 ngay trong notebook

    **Mọi ID trong manifest xuất hiện đúng một lần trong các trang hình dưới đây.**
    Mỗi dòng là một bản ghi: bên trái là median XYZ và các biên T đã lưu; giữa là
    T loop V1; bên phải là T loop Wavelet/TLS V2. Hai loop 3D dùng cùng gốc PQ và
    cùng tỉ lệ trong từng bản ghi, có dấu đầu/cuối và mũi tên thời gian.

    Ca chưa có đủ hai biên T hoặc thiếu median vẫn có dòng riêng ghi rõ nguyên
    nhân. Biên T từng phần vẫn hiện trên XYZ. Không lọc theo QC, nhãn, split hoặc
    độ đẹp của loop. Các trang sắp theo record ID; 6 bản ghi/trang, trang cuối 4.

    Hình được nhúng sẵn để đọc trực tiếp. Dưới mỗi tiêu đề trang có liên kết tới
    hình T loops đầy đủ của từng ID (kèm ba hình chiếu XY/XZ/YZ). Để xem wavelet,
    từng vòng TLS, zoom hoặc xoay 3D của bất kỳ ca nào, dùng gallery 14C.
    ''')
    code('''
    from vcg_t_almeida_overview import create_t_loop_contact_sheets
    almeida_all_loop_out = almeida_out/'notebook08_extension/all_record_t_loops'
    almeida_all_loop_manifest, almeida_all_loop_summary = create_t_loop_contact_sheets(
        almeida_results, almeida_out, almeida_all_loop_out, page_size=6)
    if (len(almeida_all_loop_manifest) != 448
            or not almeida_all_loop_manifest.record_id.is_unique
            or set(almeida_all_loop_manifest.record_id) != set(almeida_manifest.record_id)):
        raise ValueError('Hình tổng quan T loops chưa bao phủ đúng tất cả 448 ID.')
    display(pd.Series(almeida_all_loop_summary, name='Hiển thị toàn bộ T loops').to_frame())
    display(Markdown(
        '[Mở toàn bộ các trang T loops trong tab riêng]'
        '(outputs/vcg_t_almeida_v2_experiment/notebook08_extension/all_record_t_loops/index.html). '
        'Mỗi ID có một dòng trong manifest.csv của thư mục này.'))
    for almeida_page, almeida_page_rows in almeida_all_loop_manifest.groupby('page', sort=True):
        display(Markdown('### Trang '+str(almeida_page)+'/'+
            str(almeida_all_loop_summary['pages'])+' — '+str(len(almeida_page_rows))+' bản ghi'))
        display(Markdown(' · '.join(
            '['+almeida_rid+'](outputs/vcg_t_almeida_v2_experiment/figures/'+
            almeida_rid.replace('/','_')+'_loops.png)'
            for almeida_rid in almeida_page_rows.record_id)))
        display(Image(filename=str(almeida_all_loop_out/almeida_page_rows.plot_path.iloc[0]),
                      width=1700))
    ''')
    return cells


def attach_wavelet_extension(notebook):
    """Idempotently insert the extension immediately after the primary V1 gallery."""
    notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get('tags', [])]
    if INTRO not in notebook.cells[0].source:
        title, remainder = notebook.cells[0].source.split('\n', 1)
        paragraphs = [p for p in remainder.strip().split('\n\n')
                      if not p.startswith('**Tách T bằng Wavelet/TLS đã được bổ sung')]
        notebook.cells[0].source = title+'\n\n'+INTRO+'\n\n'+'\n\n'.join(paragraphs)
    insert = next((i for i, c in enumerate(notebook.cells)
                   if c.cell_type == 'markdown' and c.source.startswith('## 3. Cách dùng Manual')),
                  None)
    if insert is None:
        raise ValueError('Không tìm thấy vị trí sau gallery V1 của Notebook 08.')
    notebook.cells[insert:insert] = build_extension_cells()
    notebook.metadata['notebook08_wavelet_tls'] = {
        'results_source': 'outputs/vcg_t_almeida_v2_experiment',
        'read_saved_results_only': True,
        'sections': ['14C', '14D', '14E', '14F'],
    }
    return notebook


def execute_extension(notebook):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    os.chdir(ROOT)
    shell = InteractiveShell.instance()
    code_cells = [c for c in notebook.cells if c.cell_type == 'code' and TAG in c.metadata.get('tags', [])]
    for i, cell in enumerate(code_cells, 1):
        with capture_output(display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Notebook08 Wavelet/TLS cell {i} failed') from (
                result.error_before_exec or result.error_in_exec)
        cell.execution_count = result.execution_count
        cell.metadata['execution_scope'] = 'new Wavelet/TLS cells only; existing outputs retained'
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nb.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nb.v4.new_output('stream', name='stderr', text=captured.stderr))
        for item in captured.outputs:
            cell.outputs.append(nb.v4.new_output('display_data', data=item.data, metadata=item.metadata))
        print(f'Notebook08 Wavelet/TLS: rendered cell {i}/{len(code_cells)}', flush=True)


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    original_hash = sha(TARGET)
    backup = AUDIT/f'08_before_wavelet_tls_{original_hash[:12]}.ipynb'
    if not backup.exists():
        backup.write_bytes(TARGET.read_bytes())
    hashes_path = EXPERIMENT/'protected_v1_hashes.json'
    prior_hashes = json.loads(hashes_path.read_text())
    unexpected = [name for name, digest in prior_hashes.items()
                  if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    if unexpected:
        raise RuntimeError('Protected sources changed before extension: '+', '.join(unexpected[:8]))
    prior_manifest = AUDIT/f'protected_v1_hashes_before_{original_hash[:12]}.json'
    if not prior_manifest.exists():
        prior_manifest.write_bytes(hashes_path.read_bytes())
    notebook = attach_wavelet_extension(nb.read(TARGET, as_version=4))
    execute_extension(notebook)
    # Rebase only this explicitly requested presentation file. Keep the original
    # manifest and notebook above; every other protected entry must still match.
    unexpected = [name for name, digest in prior_hashes.items()
                  if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    if unexpected:
        raise RuntimeError('Protected sources changed during rendering: '+', '.join(unexpected[:8]))
    nb.write(notebook, TARGET)
    updated_hash = sha(TARGET)
    updated_hashes = dict(prior_hashes)
    updated_hashes[TARGET.name] = updated_hash
    hashes_path.write_text(json.dumps(updated_hashes, indent=2, ensure_ascii=False)+'\n')
    code_cells = [c for c in notebook.cells if c.cell_type == 'code' and TAG in c.metadata.get('tags', [])]
    audit = {
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User requested Wavelet/TLS visualization inside Notebook08',
        'notebook': TARGET.name,
        'before_sha256': original_hash,
        'after_sha256': updated_hash,
        'notebook_backup': str(backup.relative_to(ROOT)),
        'original_protected_manifest': str(prior_manifest.relative_to(ROOT)),
        'protected_manifest_entries_updated': [TARGET.name],
        'other_protected_files_unchanged': len(prior_hashes)-1,
        'existing_code_cells_and_outputs_retained': True,
        'executed_new_code_cells': len(code_cells),
        'embedded_new_pngs': sum('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs),
        'execution_errors': 0,
        'cohort_recomputed': False,
        'tests_run': False,
    }
    all_loop_coverage = AUDIT/'all_record_t_loops/coverage.json'
    if all_loop_coverage.is_file():
        audit['all_record_t_loop_overview'] = json.loads(all_loop_coverage.read_text())
    (AUDIT/f'update_{updated_hash[:12]}.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

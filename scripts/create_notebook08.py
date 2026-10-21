"""Build and execute Notebook08, preserving previous notebooks and V1 outputs."""
from pathlib import Path
import os
import textwrap

os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'vcg_qc_audit_matplotlib'))
os.environ.setdefault('IPYTHONDIR', str(__import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'vcg_qc_audit_ipython'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / '08_QRS_T_Loop_Reference_Error_Analysis.ipynb'


def build_reference_supplement():
    cells = []
    def md(source):
        cells.append(nb.v4.new_markdown_cell(textwrap.dedent(source).strip()))
    def code(source):
        cells.append(nb.v4.new_code_cell(textwrap.dedent(source).strip()))
    def section(title, purpose, source):
        md(f'## {title}\n\n{purpose}')
        code(source)
    md('''
    # 08 — Sai số QRS/T loops theo tham chiếu độc lập

    Notebook đánh giá bốn biên, khoảng cắt QRS/T và 12 features của **V1** trên
    **đúng median Frank XYZ đã lưu**. Dùng trực tiếp `vx`, `vy`, `vz` của PTB;
    không có phép chuyển đổi ECG → VCG. Giữ nguyên V1 và notebooks05/06/07.

    **Mode A — Verified reference evaluation:** chỉ các mốc độc lập, đúng signal,
    reviewer, hệ thời gian và trạng thái chấp nhận mới được tính sai số.

    **Mode B — Diagnostic-only:** không có reference thì xuất
    `GROUND_TRUTH_UNAVAILABLE`; MAE, IoU và sai lệch feature giữ thiếu.
    Tangent/threshold hay một detector thứ hai không tạo ground truth.

    Chọn kernel **ipcv** → **Run All**. Notebook đã được chạy và lưu output.
    Đầu ra mới: `outputs/vcg_boundary_reference_eval_v1/`.

    **Sau khi review:** export CSV từ công cụ ở mục6, thêm đường dẫn vào
    `ANNOTATION_FILES` bên dưới rồi Run All. Không ghi đè file annotation có sẵn.
    Bộ63 ưu tiên review không phải mẫu ngẫu nhiên đại diện toàn448.

    **Bản trực quan đầy đủ:** mục2A đếm và vẽ chênh lệch phương pháp; mục14A có
    hình chi tiết cho **mọi recording**, mục14B nhúng toàn bộ các trang hình tổng quan
    của448 ca ngay trong notebook. Vùng tô màu cho thấy chính xác khoảng giữa hai
    candidate; con số chênh không được gọi là lỗi so với chuyên gia khi chưa có reference.
    ''')
    section('1. Cấu hình và bảo toàn project',
        'Khóa dữ liệu V1, các notebooks05/06/07 và artifacts trước khi đánh giá. '
        'Mặc định chỉ dùng tham chiếu train/valid. Test chỉ được mở cho đánh giá cuối sau khi quy trình đã chốt.', '''
        from pathlib import Path
        import sys, json
        import numpy as np
        import pandas as pd
        from IPython.display import display, Markdown, Image, IFrame

        ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'vcg_pipeline.py').is_file())
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
        from vcg_pipeline import FEATURE_NAMES
        from vcg_boundary_reference_eval import (initialize, scan_reference_sources, priority_cases,
            import_notebook07_reviews, read_annotations, save_results, write_report,
            verify_protected, snapshot_sources, write_json, sha256)
        from vcg_reference_metrics import evaluate_references, load_median
        from vcg_reference_annotation_ui import build_annotation_tool
        from vcg_reference_plots import create_reference_plots, create_diagnostic_case_gallery

        # Thêm CSV đã được reviewer export; không dùng các mốc prediction làm ref.
        ANNOTATION_FILES = []  # Ví dụ: [Path('/duong/dan/reviewer_A.csv'), Path('/duong/dan/reviewer_B.csv')]
        FINAL_TEST_EVALUATION = False
        OUT = ROOT/'outputs/vcg_boundary_reference_eval_v1'
        frame, manifest, CFG, ARTIFACTS, PROTECTED = initialize(OUT, allow_test=FINAL_TEST_EVALUATION)
        pd.set_option('display.max_colwidth', 110)
        pd.set_option('display.max_rows', 40)
        print('Read-only V1; output:', OUT)
        print('Frank channels: vx, vy, vz | Hệ thời gian: median_relative_ms')
        print('Test reference evaluation:', FINAL_TEST_EVALUATION)
        ''')
    section('2. Cohort, split và coverage prediction',
        'Giữ nguyên split theo bệnh nhân của notebook06. Thất bại prediction vẫn nằm trong mẫu số; '
        'trích xuất đủ features không chứng minh biên đúng.', '''
        assert len(frame)==448 and frame.record_id.is_unique
        assert frame.patient_id.nunique()==200
        assert frame.groupby('patient_id').split.nunique().eq(1).all()
        display(frame.groupby(['split','label_name']).agg(
            recordings=('record_id','size'), patients=('patient_id','nunique'),
            complete_12_features=('extraction_success','sum'),
            qrs_predictions=('qrs_boundary_valid','sum'), t_predictions=('t_boundary_valid','sum')))
        display(pd.crosstab(frame.qc_status, frame.feature_completeness))
        display(Markdown('**Phạm vi:** cohort có448 recordings nhưng số cặp đủ để tính sai số phụ thuộc annotation từng biên. Các lỗi tạo median và biên thiếu được báo cáo riêng.'))
        ''')
    section('2A. Biểu đồ thực tế: bao nhiêu ca chênh và chênh bao nhiêu ms?',
        'Đếm trực tiếp các candidate tangent và threshold đang lưu, không chỉ đếm QC. '
        'Hai phương pháp dùng cùng signal nên đây là bất đồng phương pháp. Thiếu candidate được đếm riêng, không thay bằng0. '
        'Mọi mức10/20/40/60/100 ms đều là mức mô tả đã khai báo, không dùng test để chọn ngưỡng.', '''
        from vcg_all_case_visual_audit import (build_disagreement_tables,
            plot_disagreement_summary, create_all_case_contact_sheets)
        visual_tables = build_disagreement_tables(frame, OUT)
        visual_summary = visual_tables['summary']
        all_summary = visual_summary[visual_summary.scope.eq('all') & visual_summary.label_name.eq('all')]
        display(all_summary[['boundary','tolerance_ms','n_cases','n_denominator','n_missing_pair','n_patients']])
        summary_figures = plot_disagreement_summary(visual_tables, OUT)
        for row in summary_figures.itertuples(index=False):
            display(Image(filename=str(OUT/row.plot_path), width=1150))
        union_counts = all_summary[all_summary.boundary.eq('any_boundary')].set_index('tolerance_ms')
        display(Markdown('**Đếm ca duy nhất:** ' + '; '.join(
            f">{threshold} ms: **{int(union_counts.loc[threshold,'n_cases'])} ca**"
            for threshold in [0,10,20,40,60,100]) +
            '. Các mức tích lũy không được cộng với nhau. Mẫu thiếu cặp không được kết luận khớp hay lệch. '
            '**QRS:** chưa có bộ biên QRS thứ hai hoặc reference độc lập để đếm lệch; hình thực tế vẫn hiện cho toàn bộ ca.'))
        ''')
    section('3. Kiểm tra nguồn annotation thực tế',
        'Đọc toàn bộ549 headers PTB để xác nhận channel/fs/units; quét các tệp annotation và CSV trong project. '
        'Nguồn bên ngoài có tài liệu chính thức nhưng không được nhập làm reference nếu sai signal/beat/lead/scope.', '''
        sources, candidates, headers = scan_reference_sources(frame, OUT)
        display(headers.groupby(['fs','frank_channels','frank_units']).size().rename('recordings').to_frame())
        display(pd.read_csv(OUT/'local_file_extension_inventory.csv'))
        display(candidates[['path','type','n_rows','finite_reference_values','status']])
        display(sources[['source_type','annotation_scope','status','reason']])
        display(Markdown('**Kết luận nguồn:** `.xyz` là file waveform Frank được header trỏ tới, không phải annotation. Các file template không có giá trị ref không tạo ground truth. Xem reference_sources.json để truy nguồn tài liệu.'))
        ''')
    section('4. Quy ước tham chiếu và phép đổi thời gian',
        'Ưu tiên annotate ngay trên median V1. Không trừ một R-peak để biến annotation raw một nhịp thành nhãn của median. '
        'Các ví dụ dưới chỉ minh họa tọa độ thật của signal, không tạo reference.', '''
        demo_id = frame.loc[frame.processing_success, 'record_id'].iloc[0]
        demo = load_median(ARTIFACTS, demo_id)
        print('Record minh họa:', demo_id, '| fs:', demo['fs'])
        print('Median signal SHA-256:', demo['signal_sha256'])
        display(Markdown(
            '**Thời gian:** t[i]=(i−median_anchor_sample)×1000/fs. Reference được gắn vào mẫu gần nhất, '
            'tie chọn mẫu sớm; ghi riêng quantization_error_ms. Ngoài cửa sổ bị từ chối, không clamp. '
            '**Khoảng đo:** duration=t_off−t_on theo thời gian liên tục; cắt XYZ/features bằng slice inclusive đúng V1. '
            'Độ dài cắt n_samples không được thay thế duration bằng n_samples/fs.'))
        ''')
    section('5. Truy xuất các trường hợp ưu tiên của notebook07',
        'Tự động lấy bất đồng T-on/T-off, T duration bất thường, nhiều thùy, hoạt động phụ QRS, '
        'thiếu features, ST và ca không QC. Điểm ưu tiên chỉ hỗ trợ review, không phải xác suất sai hay nguy cơ MI.', '''
        priority = priority_cases(frame, OUT)
        reason_counts = priority.priority_reasons.str.get_dummies(';').sum().sort_values(ascending=False)
        display(reason_counts.rename('recordings').to_frame())
        display(priority.loc[priority.eligible_for_development_review,
            ['record_id','label_name','split','priority_reasons','t_duration_ms','t_on_method_gap_ms','t_off_method_gap_ms']].head(18))
        display(Markdown('**Giới hạn:** bất đồng phương pháp không phải actual error. Theo yêu cầu xem toàn bộ, gallery riêng ở mục14A hiển thị cả test để mô tả; không dùng các hình test để điều chỉnh thuật toán hoặc chọn ngưỡng.'))
        ''')
    section('6. Công cụ annotation trực quan, lần đầu ẩn prediction',
        'Tái sử dụng đúng63 ca từ blinded review notebook07. Có thể mở rộng bằng cách thay REVIEW_IDS bằng các ID train/valid khác. '
        'Công cụ hoạt động offline: chọn biên, click trên signal hoặc nhập ms, ghi visibility/quality/reviewer, '
        'lưu rồi mới bật overlay V1. Export CSV để notebook kiểm tra. Reviewer thứ hai dùng danh tính/session riêng.', '''
        selected07 = pd.read_csv(ROOT/'outputs/vcg_qc_audit_v1/manual_review_blinded.csv')
        REVIEW_IDS = list(selected07.record_id)
        assert len(REVIEW_IDS)>=63 and set(REVIEW_IDS).issubset(set(frame[frame.split.ne('test')].record_id))
        review_records = frame.set_index('record_id').loc[REVIEW_IDS].reset_index()
        ui = build_annotation_tool(frame, review_records, ARTIFACTS, OUT, cfg=CFG)
        print('Công cụ:', ui['html_path'])
        print('Review records:', len(REVIEW_IDS), '| Không có median V1:', int((~review_records.processing_success).sum()))
        display(review_records.groupby(['split','label_name']).size().rename('selected_recordings').to_frame())
        display(Markdown('[Mở công cụ annotation trong tab riêng](outputs/vcg_boundary_reference_eval_v1/review/independent_annotation.html). '
            'Trong Jupyter có thể dùng iframe dưới đây. File CSV được lưu/export bằng thao tác của reviewer, chưa có mốc nào được điền tự động.'))
        display(IFrame(src='outputs/vcg_boundary_reference_eval_v1/review/independent_annotation.html', width='100%', height=1000))
        ''')
    section('7. Đọc annotation và giữ nguồn độc lập',
        'File chính được tạo một lần và không ghi đè khi Run All. Thêm CSV export vào ANNOTATION_FILES. '
        'Nếu notebook07 đã có review thật, giữ giá trị gốc nhưng không tự nâng mức chất lượng hoặc trạng thái độc lập.', '''
        annotations = read_annotations(OUT, ANNOTATION_FILES)
        imported07 = import_notebook07_reviews(frame, ARTIFACTS, OUT)
        if not imported07.empty:
            annotations = pd.concat([annotations, imported07], ignore_index=True)
        print('Số hàng annotation/template đã đọc:', len(annotations))
        print('Số hàng có giá trị review thực từ notebook07:', len(imported07))
        display(annotations[['record_id','reviewer','annotation_scope','reference_quality','review_status',
            'qrs_on_ref_ms','qrs_off_ref_ms','t_on_ref_ms','t_off_ref_ms']].head())
        display(Markdown('**Điều kiện:** confirmed + chất lượng good/accepted + độc lập rõ ràng + đúng record/patient/split/fs/hash/median scope. '
            'Một biên uncertain/not_visible/rejected không được dùng tính error. Hai reviewer trùng cần primary/adjudicated rõ ràng; không chọn ngầm.'))
        ''')
    section('8. Xác minh reference và tính toàn bộ bảng đánh giá',
        'Chỉ numeric engine mới quyết định cặp nào hợp lệ. Tất cả448 recordings vẫn hiện trong bảng từng biên, '
        'loop và feature, kể cả khi không có reference hoặc prediction.', '''
        results = evaluate_references(frame, annotations, ARTIFACTS, CFG, allow_test=FINAL_TEST_EVALUATION)
        ledger, status = save_results(results, frame, OUT, allow_test=FINAL_TEST_EVALUATION)
        display(pd.DataFrame([status]))
        display(results['annotation_acceptance_audit'].head(20))
        display(ledger.reference_status.value_counts().rename('recordings').to_frame())
        print('Trạng thái hiện tại:', status['mode'])
        ''')
    section('9. Sai số QRS-on/QRS-off/T-on/T-off và coverage',
        'Signed error = predicted−reference: dương là muộn, âm là sớm. MAE/median/RMSE/P90/P95/bias '
        'và các mức10/20/40/60 ms chỉ có nghĩa khi có cặp hợp lệ. Thiếu prediction không bị loại khỏi coverage.', '''
        boundary = results['boundary_error_per_record']
        summary = results['boundary_error_summary']
        display(summary[summary.group_by.isin(['all','train_valid','label_name','split'])])
        display(boundary[['record_id','boundary','predicted_ms','reference_ms','error_ms',
            'absolute_error_ms','reference_available','prediction_available','reference_status']].head(12))
        if not boundary.error_ms.notna().any():
            display(Markdown('**GROUND_TRUTH_UNAVAILABLE:** chưa thể trả lời bốn mốc lệch bao nhiêu ms; NaN không phải sai số0.'))
        ''')
    section('10. Thống kê theo bệnh nhân và hai reviewer',
        'Tổng hợp trong từng bệnh nhân trước khi so sánh các bệnh nhân. Bất đồng reviewer được tính riêng, '
        'không gọi là lỗi V1 và không tự nhận một reviewer là chân lý tuyệt đối.', '''
        display(results['patient_boundary_summary'].head(20))
        display(results['patient_feature_summary'].head(20))
        display(results['interreviewer_disagreement'].head(20))
        display(Markdown('**Phạm vi suy luận:** các recordings cùng người không độc lập. Mẫu review được ưu tiên không cho ước lượng '
            'sai số toàn448 nếu chưa có kế hoạch lấy mẫu xác suất hoặc annotation đủ rộng. Các bảng record là mô tả có điều kiện trên reference hiện có.'))
        ''')
    section('11. Temporal IoU, cắt thiếu/thừa và duration',
        'QRS/T predicted và reference đều cắt từ chính median XYZ. Giao/hợp dùng cùng quy ước endpoint; '
        'không cộng thêm một mẫu. Cả coverage và thiếu prediction được giữ bên cạnh metric.', '''
        loops = results['loop_overlap_per_record']
        display(results['loop_overlap_summary'])
        display(loops.head(12))
        display(Markdown('**Định nghĩa:** IoU=intersection/union; missed=reference_duration−intersection; '
            'excess=predicted_duration−intersection; reference coverage=intersection/reference_duration. '
            'duration_error=pred_duration−ref_duration. Nếu một khoảng chưa có, metric chưa thể tính.'))
        ''')
    section('12. Ảnh hưởng lên đúng12 features V1',
        'Tên features lấy từ FEATURE_NAMES. Tính spatial speed trên toàn median trước khi slice, '
        'đối chiếu predicted features với CSV notebook06. Relative error dùng abs(reference), không chia khi ref gần0.', '''
        features = results['feature_reference_comparison']
        assert set(features.feature)==set(FEATURE_NAMES)
        display(results['baseline_reproduction_check'].head(15))
        display(results['feature_error_summary'].head(24))
        display(features.head(12))
        display(Markdown('**Công thức:** feature_error=pred−ref; absolute=abs(pred−ref); relative_pct=100×absolute/abs(ref). '
            'Khi abs(ref)<=1e−9 trong đơn vị riêng của feature: relative=NaN và bật near_zero. '
            'Bảng này đo thay đổi do biên; không đo khả năng phân loại bệnh và không phải feature importance.'))
        ''')
    section('13. Đồ thị sai số và gallery có reference',
        'Chỉ vẽ các cặp đã được chấp nhận: phân bố absolute error, boxplot HC/MI, Bland–Altman, '
        'IoU, duration scatter, feature heatmap và overlay XYZ/3D/XY/XZ/YZ. Không có reference thì ghi UNAVAILABLE, không vẽ accuracy giả.', '''
        plotted = create_reference_plots(results, ARTIFACTS, OUT, cfg=CFG)
        display(plotted['manifest'])
        manifest_plot = plotted['manifest']
        # Chỉ hiển thị hình có trạng thái created của chính lần đánh giá hiện tại.
        for plot_row in manifest_plot.itertuples(index=False):
            if plot_row.status=='created' and str(plot_row.plot_path).endswith('.png'):
                display(Image(filename=str(OUT/plot_row.plot_path), width=1050))
        if plotted['gallery'] is not None:
            display(Markdown('[Gallery overlay QRS/T có reference](outputs/vcg_boundary_reference_eval_v1/reference_loop_gallery.html).'))
        if plotted.get('all_gallery') is not None:
            display(Markdown('[Tất cả loop có reference hợp lệ](outputs/vcg_boundary_reference_eval_v1/all_reference_loop_gallery.html).'))
        if plotted.get('boundary_gallery') is not None:
            display(Markdown('[Tất cả biên có reference hợp lệ, kể cả reference từng phần](outputs/vcg_boundary_reference_eval_v1/all_reference_boundary_gallery.html).'))
        if status['n_valid_boundary_references']==0:
            display(Markdown('**Chưa có error plots hoặc gallery tốt/xấu:** không có cặp reference hợp lệ để xếp hạng. Xem plots_status.csv.'))
        ''')
    section('14. Gallery chẩn đoán các trường hợp ưu tiên',
        'Xem V1 chọn QRS threshold hoặc T tangent/threshold tại đâu, cùng ST/contact và hình loop. '
        'Gallery này phục vụ review train/valid; không gọi bất thường sinh lý hay bất đồng phương pháp là lỗi đã xác nhận.', '''
        diagnostic_frame = frame if FINAL_TEST_EVALUATION else frame[frame.split.ne('test')]
        diagnostic_priority = priority[priority.record_id.isin(diagnostic_frame.record_id)]
        diagnostic = create_diagnostic_case_gallery(diagnostic_frame, diagnostic_priority, ARTIFACTS, OUT, cfg=CFG, max_cases=36)
        display(diagnostic['manifest'].head(15))
        print('Gallery diagnostic:', diagnostic['gallery'])
        display(Markdown('[Mở gallery diagnostic](outputs/vcg_boundary_reference_eval_v1/diagnostic_gallery.html). '
            'Các ca có17 ms/không đủ features được giữ rõ trạng thái; trường hợp không tạo median không được tự dựng loop để chấm điểm.'))
        # Các minh họa thực tế đã lưu từ notebook07, chỉ đọc; không sửa artifact cũ.
        for rid in ['patient051/s0179lre','patient034/s0109lre','patient104/s0306lre']:
            old = ROOT/'outputs/vcg_qc_audit_v1/plots'
            row = frame.set_index('record_id').loc[rid]
            p = old/row.label_name/(rid.replace('/','_')+'.png')
            if p.is_file():
                display(Image(filename=str(p), width=1050))
        ''')
    section('14A. Toàn bộ448 ca: hình chi tiết, số ms và vùng chênh trên signal/loop',
        'Mỗi ca có hình riêng gồm XYZ, QRS velocity, T magnitude, candidate/tiếp tuyến, timeline và3D. '
        'Vùng giữa tangent và threshold được tô màu trên tín hiệu và biểu diễn bằng đoạn XYZ tương ứng. '
        'Gallery bao phủ tất cả448 ca, không chỉ một nhóm ví dụ. Có bộ lọc ngưỡng và đếm lại chính các hình đang xem. '
        'Đã xem prediction thì không tự coi lần annotation sau là lần đầu độc lập cho cùng ca.', '''
        import subprocess
        from vcg_all_case_gallery import build_all_case_gallery
        from vcg_boundary_disagreement_panels import validate_case_panels
        # CLI riêng giúp multiprocessing chạy đúng cả khi notebook mở trên macOS.
        rendered = subprocess.run([sys.executable, str(ROOT/'scripts/vcg_boundary_disagreement_panels.py'),
            '--output-dir',str(OUT/'all_cases'),'--workers','4'], capture_output=True,text=True)
        if rendered.returncode:
            raise RuntimeError(rendered.stdout+'\\n'+rendered.stderr)
        panel_manifest = pd.read_csv(OUT/'all_cases/all_case_panel_manifest.csv')
        panel_checks = validate_case_panels(panel_manifest,OUT/'all_cases',frame.record_id.tolist())
        visual_tables = build_disagreement_tables(frame, OUT)
        full_gallery = build_all_case_gallery(visual_tables['records'],visual_tables['summary'],panel_manifest,OUT/'all_cases')
        display(pd.DataFrame([{'recordings':len(panel_manifest),
            'có_median':int(panel_manifest.has_median.sum()),
            'thiếu_median':int((~panel_manifest.has_median).sum())}]))
        display(Markdown('[Mở gallery448 ca — lọc để xem hết từng nhóm chênh](outputs/vcg_boundary_reference_eval_v1/all_cases/all_cases_gallery.html). '
            'Ví dụ chọn ngưỡng >60 ms để xem toàn bộ ca tương ứng, không cần tự dò CSV. '
            'Test chỉ được xem mô tả; chưa thay đổi V1.'))
        display(IFrame(src='outputs/vcg_boundary_reference_eval_v1/all_cases/all_cases_gallery.html',width='100%',height=1100))
        ''')
    section('14B. Xem hết448 tín hiệu ngay trong notebook',
        'Mỗi ô nhỏ là magnitude thật của một recording với đơn vịmV và thời gianms, '
        'có biên V1, hai candidate và khoảng tô chênh. Các trang phủ đủ448 ca đúng một lần; '
        'năm ca không có median hiện rõ nguyên nhân. Bấm gallery14A để xem sáu panel độ phân giải đầy đủ.', '''
        contact_manifest = create_all_case_contact_sheets(visual_tables['records'],ARTIFACTS,OUT,page_size=12)
        assert contact_manifest.record_id.is_unique and set(contact_manifest.record_id)==set(frame.record_id)
        display(Markdown(f"**{len(contact_manifest)} ca / {contact_manifest.page.nunique()} trang**, xếp theo chênh lớn nhất giảm dần. "
            "Δon/Δoff là |tangent−threshold|; thiếu cặp không phải0ms."))
        for page, rows in contact_manifest.groupby('page',sort=True):
            display(Markdown(f'**Trang {page}/{contact_manifest.page.nunique()}**'))
            display(Image(filename=str(OUT/rows.plot_path.iloc[0]),width=1400))
        ''')
    section('15. Báo cáo11 câu hỏi nghiên cứu',
        'Tất cả con số kết luận được sinh từ bảng đã tính. Nếu thiếu reference, trả lời rõ chưa thể biết '
        'sai số thực tế và chỉ ra annotation cần bổ sung để quyết định V2.', '''
        priority_current = pd.read_csv(OUT/'priority_review_cases.csv')
        report_path, conclusions = write_report(results, frame, status, priority_current, OUT)
        for i, item in enumerate(conclusions, 1):
            display(Markdown(f"**{i}. {item['question']}** {item['answer']}"))
        print('Báo cáo:', report_path)
        ''')
    section('16. Kiểm tra cuối và nguồn tái lập',
        'Xác nhận đủ output, không ghi đè baseline và không dùng test để điều chỉnh. '
        'Mọi annotation chưa chấp nhận và lý do loại đều có audit; không ngầm thay missing bằng0.', '''
        assert verify_protected(PROTECTED)
        assert len(boundary)==len(frame)*4
        assert len(loops)==len(frame)*2
        assert len(features)==len(frame)*len(FEATURE_NAMES)
        assert not np.isinf(features.select_dtypes(include='number')).any().any()
        if status['n_valid_boundary_references']==0:
            assert boundary.error_ms.isna().all()
            assert loops.temporal_iou.isna().all()
            assert features.relative_feature_error_pct.isna().all()
        required = ['reference_availability_report.csv','reference_annotations.csv',
            'boundary_error_per_record.csv','boundary_error_summary.csv','loop_overlap_per_record.csv',
            'loop_overlap_summary.csv','feature_reference_comparison.csv','priority_review_cases.csv',
            'reference_evaluation_report.md']
        assert all((OUT/name).is_file() for name in required)
        source_snapshot = snapshot_sources(OUT)
        save_event_path = OUT/'observed_notebook_save_events.json'
        save_events = json.loads(save_event_path.read_text()) if save_event_path.is_file() else []
        checks = {'records':len(frame),'patients':frame.patient_id.nunique(),
            'code_sections':19,'mode':status['mode'],
            'accepted_boundary_references':status['n_valid_boundary_references'],
            'protected_inputs_unchanged_during_final_execution':True,
            'all_previous_inputs_match_initial_creation_snapshot':not bool(save_events),
            'observed_notebook_save_events':len(save_events),
            'baseline_algorithm_modified':False,'test_used_for_tuning':False,
            'reference_annotations_sha256':sha256(OUT/'reference_annotations.csv')}
        assert len(panel_manifest)==448 and len(contact_manifest)==448
        checks.update({'all_case_figures':len(panel_manifest),'all_case_contact_pages':contact_manifest.page.nunique(),
            'method_disagreement_is_reference_error':False})
        write_json(OUT/'notebook_execution_checks.json', checks)
        display(pd.DataFrame([checks]))
        display(Markdown('**Đã hoàn tất đánh giá theo mức bằng chứng hiện có.** '
            '[Báo cáo](outputs/vcg_boundary_reference_eval_v1/reference_evaluation_report.md) · '
            '[Công cụ annotation](outputs/vcg_boundary_reference_eval_v1/review/independent_annotation.html). '
            'Khi bổ sung CSV độc lập, chạy lại notebook để tự chuyển sang Mode A cho các cặp được chấp nhận.'))
        ''')
    return nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'ipcv', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.15'},
    })


def build():
    """Put actual XYZ/V1 inspection first; keep reference evaluation as appendices."""
    legacy = build_reference_supplement()
    blocks = {}
    for i, cell in enumerate(legacy.cells[:-1]):
        if cell.cell_type=='markdown' and legacy.cells[i+1].cell_type=='code':
            blocks[cell.source.splitlines()[0]] = legacy.cells[i+1].source
    def old(number):
        prefix='## '+number+'. '
        return next(source for heading,source in blocks.items() if heading.startswith(prefix))
    cells=[]
    def md(source): cells.append(nb.v4.new_markdown_cell(textwrap.dedent(source).strip()))
    def code(source): cells.append(nb.v4.new_code_cell(textwrap.dedent(source).strip()))
    def section(title, purpose, source):
        md('## '+title+'\n\n'+purpose); code(source)
    md('''
    # 08 — Kiểm tra trực quan QRS/T trên toàn bộ448 recordings

    **Luồng chính:** mở gallery → chọn recording → xem median Frank XYZ và bốn mốc
    V1 đã chọn → zoom QRS/T → xem loops3D → tự đặt mốc → so V1 với mốc thủ công
    → lưu → Previous/Next. Không cần đổi RECORD_ID hoặc chạy lại notebook mỗi lần.

    Dùng trực tiếp `vx/vy/vz` của PTB, median đặt gốc PQ, đúng quy trình notebook05
    Bước6–19. Màu xanh nhạt là khoảng QRS mà V1 đã cắt; cam nhạt là khoảng T.
    Tín hiệu ngoài các vùng tô vẫn hiển thị để kiểm tra phần đầu/cuối có bị bỏ sót.
    Candidate tangent/threshold là lớp tham khảo, **không chứng minh biên đúng**.

    **14A đã nâng cấp:** gallery offline bao phủ đủ448 ca, chuyển từng ca, lọc/tìm,
    zoom, bật/tắt biên và lưu Manual Visual Review. **14B đã nâng cấp:** toàn bộ
    contact sheets là XYZ thật với biên cuối cùng V1, không còn lấy chênh candidate làm nội dung chính.

    Manual Visual Review có thể đã xem prediction: lưu riêng, luôn
    `independent_reference=False`; không ghi vào annotation độc lập hoặc thay prediction.
    Phân tích reference/sai số cũ được giữ ở **phụ lục**, chỉ tính với reference được chấp nhận.

    Kernel **ipcv** → **Run All**. Các hình/payload có cache. Đầu ra trực quan ở
    `outputs/vcg_boundary_reference_eval_v1/all_cases/`; không sửa notebooks05/06/07,
    dữ liệu V1 hoặc split theo bệnh nhân. Test chỉ được xem mô tả, không tinh chỉnh.
    ''')
    section('1. Import, cấu hình và giữ nguyên V1',
        'Giữ đường dẫn annotation độc lập của người dùng. Các tùy chọn bổ sung không thay đổi prediction.',
        old('1')+'''\n
SHOW_METHOD_DIAGNOSTICS = False  # Phụ lục tùy chọn; không phải nội dung xem chính.
from vcg_visual_review_data import prepare_visual_review_data
from vcg_visual_review_ui import build_visual_review_gallery
from vcg_visual_review_overview import create_v1_contact_sheets
from vcg_boundary_disagreement_panels import validate_case_panels
''')
    section('2. Dữ liệu đúng notebook05, median và mốc V1 đúng notebook06',
        'Kiểm tra đầy đủ448ID; thời gian mỗi ca theo fs/anchor riêng. Tính lại intermediate bằng đúng delineate_loops '
        'để kiểm tra nhất quán; mọi khác biệt được ghi, không thay mốc đã lưu.', '''
        REVIEW_OUT = OUT/'all_cases'
        DATA = prepare_visual_review_data(frame,ARTIFACTS,REVIEW_OUT,cfg=CFG,resume=True)
        signal_manifest = DATA['manifest']
        assert len(signal_manifest)==448 and signal_manifest.record_id.is_unique
        data_metrics = {key:DATA['validation'][key] for key in [
            'n_records','n_medians','n_qrs_loops','n_t_loops','n_parity_fields',
            'n_parity_differences','n_integrity_issues','cache_hits']}
        display(pd.DataFrame([data_metrics]))
        display(signal_manifest.groupby(['label_name','split','segmentation_status']).size().rename('recordings').to_frame())
        differences=DATA['parity'][~DATA['parity'].matches_exactly.astype(bool)]
        if len(differences):
            display(differences)
        else:
            print('Toàn bộ intermediate và mốc V1 đối chiếu đều khớp; không thay prediction.')
        display(Markdown('**Nguồn hiển thị:** XYZ đã trừ baseline PQ và biên/sample đã lưu. '
            't[i]=(i−anchor)×1000/fs. QRS và T được cắt riêng bằng slice inclusive; không lấy đoạn ST giữa hai loop. '
            'Các payload lỗi/thiếu vẫn giữ nguyên ID để xuất hiện trong giao diện.'))
        ''')
    section('14A. Gallery tương tác: xem từng recording, zoom và Manual Visual Review',
        'Đây là giao diện chính cho toàn bộ448 ca, không lọc theo QC hoặc số candidate. '
        'A: full median XYZ; B: QRS XYZ+velocity/ngưỡng/search; C: T XYZ+magnitude/ST/TP; '
        'D: hai loop V1 trong hệ PQ, cùng tỉ lệ, kèm hướng thời gian và projections. '
        'Những mốc/loop không tồn tại được báo thiếu.', '''
        import subprocess
        rendered = subprocess.run([sys.executable,str(ROOT/'scripts/vcg_boundary_disagreement_panels.py'),
            '--output-dir',str(REVIEW_OUT),'--workers','4'],capture_output=True,text=True)
        if rendered.returncode:
            raise RuntimeError(rendered.stdout+chr(10)+rendered.stderr)
        panel_manifest = pd.read_csv(REVIEW_OUT/'all_case_panel_manifest.csv')
        panel_checks = validate_case_panels(panel_manifest,REVIEW_OUT,frame.record_id.tolist())
        gallery = build_visual_review_gallery(frame,signal_manifest,panel_manifest,REVIEW_OUT)
        print('Gallery:',gallery['gallery_path'])
        display(pd.DataFrame([panel_checks]))
        display(Markdown('[Mở gallery offline toàn448 ca](outputs/vcg_boundary_reference_eval_v1/all_cases/all_cases_gallery.html). '
            'Dùng Previous/Next hoặc danh sách ID; bộ lọc vẫn giữ các ca thiếu biên/median. '
            'Nút tổng quan chuyển sang danh sách hình nhiều ca. Chế độ chi tiết chỉ tải signal của ca đang chọn.'))
        display(IFrame(src='outputs/vcg_boundary_reference_eval_v1/all_cases/all_cases_gallery.html',width='100%',height=1250))
        ''')
    section('3. Cách dùng Manual Visual Review và đọc chênh lệch',
        'Các mốc thủ công là đánh giá bằng mắt, không tự biến thành ground truth độc lập. '
        'Dữ liệu review nằm trong trình duyệt và CSV do người dùng xuất; Run All không ghi đè đánh dấu.', r'''
        display(Markdown(
            '1. Mở gallery, nhập tên/mã reviewer rồi chọn recording. Có thể tắt bốn mốc V1 để quan sát signal; '
            'việc đã từng thấy prediction vẫn được ghi lại.\n\n'
            '2. Zoom vùng QRS/T, chọn biên đang đặt rồi click trên signal hoặc nhập thời gian ms. '
            'Bật overlay để so mốc thủ công với V1; không thay mốc đã lưu của pipeline.\n\n'
            '3. Chọn nhận xét: đạt, quá sớm, quá muộn, không rõ biên hoặc không đánh giá được; ghi chú rồi lưu. '
            'Chuyển Previous/Next và quay lại vẫn giữ đánh dấu. Xuất CSV thường xuyên để sao lưu.\n\n'
            '**Chênh lệch = V1 time − Manual time.** Dương: V1 muộn hơn mốc bạn chọn; âm: V1 sớm hơn. '
            'Nếu thiếu một mốc thì không tính chênh lệch, không thay bằng0.\n\n'
            '**Blind/unblind:** ẩn prediction sau khi đã thấy không làm review trở lại độc lập. '
            'CSV này luôn là visual_review_only; muốn nghiên cứu accuracy, dùng quy trình annotation độc lập ở phụ lục.'))
        ''')
    section('4. Ví dụ trực tiếp: đủ biên, T rất ngắn, thiếu biên và thiếu median',
        'Các hình dưới minh họa chính signal và mốc V1 đã lưu. Không gán đúng/sai chỉ từ độ rộng, ST hoặc nhãn MI. '
        'Trong gallery, mọi recording đều có cùng cách xem.', '''
        for rid in ['patient104/s0306lre','patient051/s0179lre','patient034/s0109lre','patient033/s0157lre']:
            row=frame.set_index('record_id').loc[rid]
            display(Markdown(f"**{rid} — {row.label_name}, {row.split}, QC={row.qc_status}**"))
            display(row[['qrs_on_ms','qrs_off_ms','t_on_ms','t_peak_ms','t_off_ms']].to_frame('V1_ms'))
            image_row=panel_manifest.set_index('record_id').loc[rid]
            display(Image(filename=str(REVIEW_OUT/image_row.plot_path),width=1250))
        ''')
    section('14B. Tổng quan đủ448 median XYZ và các vùng V1 đã cắt',
        'Mỗi ca xuất hiện đúng một lần trên các trang hình. XYZ ngoài khoảng QRS/T vẫn hiện; '
        'ca không có median hiển thị nguyên nhân. Đây là hình tổng quan; mở gallery14A để zoom và đặt mốc.', '''
        contact_manifest = create_v1_contact_sheets(frame,ARTIFACTS,REVIEW_OUT,page_size=12)
        assert len(contact_manifest)==448 and contact_manifest.record_id.is_unique
        print('Toàn bộ',len(contact_manifest),'ca /',contact_manifest.page.nunique(),'trang XYZ')
        for page,rows in contact_manifest.groupby('page',sort=True):
            display(Markdown(f'**Trang {page}/{contact_manifest.page.nunique()}**'))
            display(Image(filename=str(REVIEW_OUT/rows.plot_path.iloc[0]),width=1400))
        ''')
    section('5. Tóm tắt kiểm tra trực quan và các ca thiếu/lỗi',
        'Báo số lượng khả dụng và khả năng mở hình; không báo tỷ lệ phân đoạn đúng khi chưa có tham chiếu. '
        'Mọi lỗi dữ liệu/render vẫn có dòng trong manifest và thông báo ở giao diện.', '''
        display(signal_manifest.groupby('segmentation_status').size().rename('recordings').to_frame())
        display(signal_manifest[signal_manifest.segmentation_status.ne('complete_bounds')].head(30))
        visual_checks = {'records':len(frame),'median_records':int(signal_manifest.has_median.sum()),
            'panel_records':len(panel_manifest),'contact_sheet_records':len(contact_manifest),
            'contact_sheet_pages':contact_manifest.page.nunique(),
            'manual_reviews_independent_reference':False,'test_used_for_tuning':False}
        write_json(REVIEW_OUT/'visual_review_execution_checks.json',visual_checks)
        print('Các đánh dấu được export riêng; không sửa reference_annotations.csv hoặc prediction V1.')
        ''')
    md('''
    # Phụ lục — Reference và sai số chỉ khi có annotation độc lập

    Phần dưới giữ quy trình đánh giá cũ để bổ sung bằng chứng khi có tham chiếu hợp lệ.
    Dữ liệu Manual Visual Review ở giao diện chính không tự được nhập làm reference.
    Hai candidate gần nhau vẫn có thể cùng sai, nên đồng thuận không được coi là accuracy.
    ''')
    titles={
        '3':('A1. Kiểm tra nguồn tham chiếu thực tế','Xác minh nguồn, record, signal/lead và hệ tọa độ thời gian.'),
        '4':('A2. Hệ thời gian của median beat','Giữ đúng fs/anchor và cửa sổ signal; không so raw với median khi chưa có ánh xạ hợp lệ.'),
        '5':('A3. Danh sách ưu tiên review trước đây','Chỉ là gợi ý; gallery chính vẫn hiển thị mọi recording.'),
        '6':('A4. Công cụ annotation độc lập riêng','Dành cho người đánh giá chưa xem prediction; tách khỏi Manual Visual Review ở gallery chính.'),
        '7':('A5. Đọc reference đã được cung cấp','Không ghi đè annotation hiện có và không tự nâng trạng thái chấp nhận.'),
        '8':('A6. Xác minh reference và tính bảng sai số','Chỉ chấp nhận reference độc lập đúng identity/time/hash/scope.'),
        '9':('A7. Sai số từng biên và coverage','Thiếu reference/prediction được giữ rõ; không thay sai số thiếu bằng0.'),
        '10':('A8. Theo bệnh nhân và giữa reviewer','Không coi recordings cùng bệnh nhân là độc lập.'),
        '11':('A9. Sai lệch khoảng cắt QRS/T','IoU và cắt thừa/thiếu chỉ tính khi có đủ cặp khoảng hợp lệ.'),
        '12':('A10. Sai lệch12 features theo reference','Giữ đúng công thức và vận tốc trên toàn median của V1.'),
        '13':('A11. Toàn bộ hình đối chiếu có reference','Nếu có reference, vẽ mọi ca hợp lệ; chưa có thì không tạo hình accuracy giả.'),
        '15':('A12. Báo cáo11 câu hỏi reference','Kết luận định lượng theo dữ liệu thật, không theo mức đồng thuận candidate.'),
    }
    for number,(title,purpose) in titles.items():
        section(title,purpose,old(number))
    section('B. Chẩn đoán tangent/threshold tùy chọn',
        'Đã chuyển khỏi nội dung xem chính. Mặc định không tạo thêm biểu đồ đồng thuận.', '''
        if SHOW_METHOD_DIAGNOSTICS:
            from vcg_all_case_visual_audit import build_disagreement_tables,plot_disagreement_summary
            method_tables=build_disagreement_tables(frame,OUT)
            for item in plot_disagreement_summary(method_tables,OUT).itertuples(index=False):
                display(Image(filename=str(OUT/item.plot_path),width=1100))
        else:
            display(Markdown('Phân tích candidate đang tắt. Bật SHOW_METHOD_DIAGNOSTICS nếu cần; '
                'hai phương pháp đồng thuận không chứng minh biên đúng.'))
        ''')
    section('6. Kiểm tra cuối và nguồn tái lập',
        'Đối chiếu hash dữ liệu/baseline; kiểm tra manifest448 ca, mọi hình và snapshot mã nguồn. '
        'Không sửa baseline hoặc tự tạo reference.', '''
        assert verify_protected(PROTECTED)
        assert set(signal_manifest.record_id)==set(frame.record_id)==set(panel_manifest.record_id)==set(contact_manifest.record_id)
        assert all((REVIEW_OUT/p).is_file() for p in panel_manifest.plot_path)
        assert all((REVIEW_OUT/p).is_file() for p in contact_manifest.plot_path.unique())
        assert (REVIEW_OUT/'all_cases_gallery.html').is_file()
        source_snapshot=snapshot_sources(OUT)
        write_json(OUT/'notebook_execution_checks.json',{
            **visual_checks,'mode':'VISUAL_V1_REVIEW_WITH_OPTIONAL_REFERENCE',
            'saved_predictions_used':True,'reference_mode':status['mode'],
            'accepted_boundary_references':status['n_valid_boundary_references'],
            'baseline_algorithm_modified':False,'protected_inputs_unchanged_during_final_execution':True,
            'source_snapshot_files':len(source_snapshot)})
        display(Markdown('**Hoàn tất:** [Gallery tương tác448 ca](outputs/vcg_boundary_reference_eval_v1/all_cases/all_cases_gallery.html) · '
            '[Báo cáo reference bổ sung](outputs/vcg_boundary_reference_eval_v1/reference_evaluation_report.md). '
            'Luồng chính dùng signal và mốc đã lưu; mốc thủ công luôn tách khỏi V1 và annotation độc lập.'))
        ''')
    from extend_notebook08_almeida import attach_wavelet_extension
    return attach_wavelet_extension(nb.v4.new_notebook(cells=cells,metadata=legacy.metadata))


def execute(notebook, path=TARGET):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(ROOT)
    shell = InteractiveShell.instance()
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    for i, cell in enumerate(code_cells, 1):
        with capture_output(display=True) as output:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = result.execution_count
        cell.outputs = []
        if output.stdout:
            cell.outputs.append(nb.v4.new_output('stream', name='stdout', text=output.stdout))
        if output.stderr:
            cell.outputs.append(nb.v4.new_output('stream', name='stderr', text=output.stderr))
        for item in output.outputs:
            cell.outputs.append(nb.v4.new_output('display_data', data=item.data, metadata=item.metadata))
        nb.write(notebook, path)
        if result.error_before_exec or result.error_in_exec:
            error = result.error_before_exec or result.error_in_exec
            raise RuntimeError(f'Notebook08 cell{i} failed: {error}') from error
        print(f'Notebook08 executed {i}/{len(code_cells)}', flush=True)
    nb.validate(notebook)
    print('Saved executed notebook:', path, flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    if args.rerun:
        notebook = nb.read(TARGET, as_version=4)
    else:
        if TARGET.exists():
            raise FileExistsError(TARGET)
        notebook = build()
        nb.write(notebook, TARGET)
    execute(notebook)


if __name__ == '__main__':
    main()

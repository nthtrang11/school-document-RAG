# Cổng tra cứu văn bản nhà trường — Hybrid RAG

Ứng dụng tra cứu văn bản tiếng Việt theo kiến trúc **BM25 + FAISS + Reciprocal Rank Fusion (RRF)**, hỗ trợ hai thí nghiệm chunking độc lập (Fixed-size và Recursive), trích dẫn theo tài liệu/trang/điều mục và đánh giá retrieval trên bộ câu hỏi đã được xác minh.

## Trạng thái dữ liệu ban đầu

Workspace ban đầu chứa bốn PDF trong `data/raw/` (12 trang tổng cộng); chưa có mã nguồn, tài liệu pipeline DOCX hoặc URL nguồn. Hai trang của một PDF có dấu hiệu lỗi mã hóa khi trích xuất. Các PDF gốc không bị sửa. Metadata hiện trạng hiệu lực, ngày ban hành, URL và số trang in được để chưa xác minh nếu không trích xuất chắc chắn. Các câu hỏi trong `data/evaluation/questions.csv` là **bản nháp cần người dùng kiểm tra**: ground truth và chunk liên quan để trống; không được dùng để tạo metrics thực nghiệm cho đến khi xác minh thủ công đối chiếu PDF.

## Kiến trúc

```text
PDF / HTML / TXT đã thu thập
  -> kiểm tra, SHA-256, page records, làm sạch bảo toàn nguồn
  -> Fixed-size hoặc Recursive chunks (index tách biệt)
  -> SentenceTransformers (CPU) -> FAISS inner product chuẩn hóa
  -> rank-bm25
  -> BM25 + FAISS -> RRF (k=60) -> top-5
  -> context có giới hạn -> LLM OpenAI-compatible tùy chọn
  -> citation đối chiếu evidence registry
```

`src/` tách riêng ingestion, chunking, embedding, indexing, retrieval, RAG, evaluation và UI. PDF loader dùng số trang 1-based trong file gốc; số trang in trên văn bản để null nếu không xác định an toàn. Web loader đọc file HTML/TXT đã thu thập trong máy, không crawl website.

## Công nghệ

Python 3.11+, Streamlit, PyMuPDF, pandas, SentenceTransformers, FAISS CPU, rank-bm25, Pydantic, python-dotenv, pytest và matplotlib. Mô hình embedding mặc định đa ngôn ngữ `paraphrase-multilingual-MiniLM-L12-v2`; tải model lần đầu cần mạng và dung lượng đĩa. Không cần GPU.

## Cấu trúc

- `data/raw/pdf/`: PDF đầu vào; hiện pipeline cũng quét PDF đặt trực tiếp trong `data/raw/`.
- `data/raw/web/`: HTML/TXT đã thu thập; có thể đặt URL gốc trong tệp `.url` cùng tên (ví dụ `notice.html.url`).
- `data/processed/`: page records, metadata registry và báo cáo ingestion (sinh tự động); page records sau xử lý đã lược bỏ dòng có thông tin liên hệ cá nhân phát hiện được.
- `data/evaluation/questions.csv`: câu hỏi nháp; chỉ đánh giá các hàng đã xác minh.
- `indexes/fixed_size/`, `indexes/recursive/`: hai corpus, BM25 và FAISS riêng biệt.
- `src/`: mã nguồn module.
- `tests/`: kiểm thử.
- `results/`: metrics, case records và biểu đồ/báo cáo (sinh tự động).

Các chỉ mục và dữ liệu processed bị loại khỏi Git theo mặc định vì có thể chứa nội dung nhạy cảm hoặc hạn chế chia sẻ.

## Cài đặt

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Có thể dùng Python 3.11 trở lên; phiên bản 3.13 phụ thuộc khả năng cài wheel của PyTorch/FAISS tại nền tảng.

## Cấu hình

Chỉnh `.env` nếu cần. `PDF_DIR` mặc định là `data/raw` để tương thích các PDF hiện có. Đặt tên file hoặc document ID cần loại trừ vào `EXCLUDED_DOCUMENTS`, phân cách bằng dấu phẩy. Cấu hình các tham số chunking, top-K, RRF và model qua `.env.example`.

LLM là tùy chọn và dùng endpoint OpenAI-compatible. Chỉ khi cần sinh câu trả lời, cấu hình `LLM_API_KEY`, `LLM_MODEL`, và endpoint trong `.env`; không thêm secret vào mã hoặc commit `.env`. Không có API key, ingestion, indexing, retrieval và evaluation retrieval vẫn hoạt động.

Nếu LLM báo timeout, ứng dụng chờ tối đa `LLM_TIMEOUT` giây (mặc định 180). Có thể tăng giá trị này trong `.env` nếu nhà cung cấp thường phản hồi lâu, sau đó khởi động lại Streamlit. Nếu vẫn timeout, kiểm tra endpoint, model được nhà cung cấp hỗ trợ, kết nối mạng hoặc trạng thái dịch vụ; không gửi API key trong báo lỗi.
Với endpoint Gemini OpenAI-compatible, ứng dụng tự đặt `reasoning_effort=low` để tránh mức suy luận mặc định kéo dài; có thể ghi đè bằng `LLM_REASONING_EFFORT` (`minimal`, `low`, `medium` hoặc `high`). Trường này không được gửi tới endpoint không phải Google nếu chưa cấu hình rõ.

## Citation tương tác

- Mã `[E1]`, `[E2]` trong câu trả lời được kiểm tra với evidence registry của lần truy xuất hiện tại; citation không có chunk, nguồn hoặc trang hợp lệ sẽ không được giữ.
- Citation nằm inline trong câu trả lời; rê chuột hoặc dùng phím Tab để xem thẻ preview tên tài liệu, trang và đoạn bằng chứng. Trên thiết bị cảm ứng, dùng nút thông tin cạnh citation.
- Nhấn citation mở trình xem PDF trong cùng phiên ứng dụng, tại đúng trang; mapping được đăng ký theo từng câu trả lời trong Session State trước khi hiển thị và callback component xử lý click mà không tạo URL/tab phiên mới.
- PDF được highlight khi PyMuPDF tìm thấy bằng chứng khớp chính xác hoặc có bbox nằm hợp lệ trong trang; nếu không có tọa độ đáng tin cậy, vẫn mở đúng trang nhưng không tạo vùng đánh dấu giả.
- PDF gốc không bị sửa. PDF scan, OCR lỗi hoặc nội dung đã biến đổi có thể không hỗ trợ tìm văn bản để highlight.
- `st.pdf` cần extra `streamlit[pdf]`; citation component dùng Custom Components v2 và `markdown-it-py`, đã được khai báo trong `requirements.txt`.

## Chạy hệ thống

1. Đặt tài liệu được phép xử lý vào `data/raw/pdf/` hoặc `data/raw/web/`.
2. Nạp, kiểm tra và làm sạch:

   ```powershell
   python -m src.ingestion.ingestion_pipeline
   ```

   Kết quả gồm `pages_raw.jsonl`, `pages_clean.jsonl`, `documents.jsonl`, danh sách hợp lệ/bị loại và `ingestion_summary.json` trong `data/processed/`. Tệp không có text được báo cần OCR và không tự OCR.

3. Tạo hai index độc lập:

   ```powershell
   python -m src.indexing.index_manager --strategy both
   ```

   Có thể chỉ tạo một nhánh (`fixed_size` hoặc `recursive`) hoặc buộc tạo lại với `--force`. Không thay đổi model/chunks thì fingerprint cho phép tái sử dụng index đã lưu.

4. Chạy web app:

   ```powershell
   streamlit run src/app.py
   ```

   Bộ lọc loại văn bản, đơn vị, trạng thái, ngày ban hành và chủ đề được áp dụng trực tiếp khi truy xuất; trường metadata chưa xác định hiển thị rõ thay vì suy đoán.

## Evaluation

Ground truth phải được xác minh thủ công từ PDF. Điền `ground_truth_answer`, `relevant_document_id`, `relevant_chunk_ids_fixed_size`, `relevant_chunk_ids_recursive`, `relevant_page`, `relevant_section` và đặt `verification_status=verified`. ID chunk phụ thuộc chiến lược; cần đánh dấu chunk liên quan ở cả hai index. Không đổi trạng thái nếu chưa kiểm tra nguồn gốc.

```powershell
python -m src.evaluation.report
```

Chương trình chỉ tính Precision@3, Recall@3, Hit@3, MRR và latency trên hàng đã xác minh; nếu không có hàng nào, summary báo bị chặn và không sinh số liệu. Kết quả thực nghiệm được ghi ở `results/metrics/`; sheet chấm thủ công correctness, completeness, groundedness và citation accuracy được tạo riêng. `results/cases/` luôn có ba tệp; case chưa thể phân tích được đánh dấu `not_run`, không chứa kết quả giả.

Citation được kiểm tra cấu trúc theo evidence registry (mã evidence có tồn tại); việc citation có hỗ trợ ngữ nghĩa cho từng mệnh đề cần được người đánh giá xác nhận thủ công.

## Kiểm thử

```powershell
pytest
```

## Giới hạn và giả định

- Chưa có DOCX tham chiếu hoặc URL chính thức, do đó chưa xác minh nguồn, hiệu lực/thay thế văn bản hay metadata thiếu.
- Pipeline không tự tạo OCR, không tự crawl web và không suy ra số trang in nếu thiếu bằng chứng.
- Nhận diện metadata dựa trên mẫu văn bản phổ biến; cần kiểm tra thủ công trước khi công bố.
- LLM được gọi qua chat-completions tương thích OpenAI; khi thiếu cấu hình, UI vẫn hiển thị retrieval và nguồn, không giả lập câu trả lời.
- Tính hợp lệ citation kiểm tra ID evidence/trang của nguồn; không khẳng định mức độ entailment ngữ nghĩa tự động.
- Chỉ số so sánh không có ý nghĩa khi evaluation chưa được xác minh hoặc hai index dùng corpus/model khác nhau. Mỗi index manifest ghi fingerprint, model và cấu hình chunk.

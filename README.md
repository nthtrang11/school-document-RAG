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

Python 3.11+, Streamlit, PyMuPDF, RapidOCR/ONNX Runtime, pandas, SentenceTransformers, FAISS CPU, rank-bm25, Pydantic, python-dotenv, pytest và matplotlib. Mô hình embedding mặc định đa ngôn ngữ `paraphrase-multilingual-MiniLM-L12-v2`; tải model lần đầu cần mạng và dung lượng đĩa. Không cần GPU.

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

LLM là tùy chọn và dùng endpoint OpenAI-compatible. Chỉ khi cần sinh câu trả lời, cấu hình `LLM_API_KEY`, `LLM_MODEL`, và endpoint trong `.env`; không thêm secret vào mã hoặc commit `.env`. Với Groq, đặt `LLM_PROVIDER=groq`, `LLM_API_KEY`, `LLM_MODEL` và `LLM_BASE_URL=https://api.groq.com/openai/v1`; ứng dụng tự nối `/chat/completions`. `LLM_ENDPOINT` nếu được cấu hình sẽ được ưu tiên hơn `LLM_BASE_URL`. Không có API key, ingestion, indexing, retrieval và evaluation retrieval vẫn hoạt động.

Nếu LLM báo timeout, ứng dụng chờ tối đa `LLM_TIMEOUT` giây (mặc định 180). Khi hết thời gian, ứng dụng vẫn hiển thị bằng chứng đã truy xuất và citation, kèm thông báo lỗi LLM. Có thể điều chỉnh giá trị này trong `.env` rồi khởi động lại Streamlit. Nếu vẫn timeout, kiểm tra endpoint, model được nhà cung cấp hỗ trợ, kết nối mạng hoặc trạng thái dịch vụ; không gửi API key trong báo lỗi.
Khi LLM không khả dụng, ứng dụng vẫn giữ các citation của bằng chứng truy xuất để xem preview và mở tài liệu nguồn; phần này không phải câu trả lời do LLM sinh.
Với lỗi HTTP 502, 503 hoặc 504, client tự thử lại một lần sau 1 giây. Nếu vẫn lỗi, nhà cung cấp có thể đang gián đoạn; kiểm tra trạng thái dịch vụ và thử lại sau. HTTP 401/403 thường cần kiểm tra API key/quyền model, còn HTTP 404 cần kiểm tra endpoint và model.
Lỗi HTTP từ provider hiển thị thêm provider, model, endpoint host, request ID và các trường `type`, `code`, `message` trong response body (API key được che) để hỗ trợ chẩn đoán.
Groq dùng OpenAI SDK với `User-Agent` nhận diện ứng dụng. Nếu phản hồi chứa Cloudflare error 1010, đó là chặn theo client signature, không phải bằng chứng API key/model sai; ứng dụng sẽ báo rõ trường hợp này.
Với endpoint Gemini OpenAI-compatible, ứng dụng tự đặt `reasoning_effort=low` để tránh mức suy luận mặc định kéo dài; có thể ghi đè bằng `LLM_REASONING_EFFORT` (`minimal`, `low`, `medium` hoặc `high`). Trường này không được gửi tới endpoint không phải Google nếu chưa cấu hình rõ.

## Citation tương tác

- Các dạng `[E1]`, `【E1】` và `【E1†L1-L3】` được đối chiếu với evidence registry; chỉ ID có mapping hợp lệ mới được chuẩn hóa thành citation tương tác `[E1]`. ID không hợp lệ được giữ nguyên dưới dạng văn bản, không được biến thành link.
- `L1-L3` là line reference do định dạng citation có thể kèm theo, không phải trang PDF hoặc bbox; trang và highlight chỉ lấy từ metadata mapping của evidence.
- Citation nằm inline trong câu trả lời; rê chuột hoặc dùng phím Tab để xem popover có tên file PDF, tên văn bản, trang, evidence nguyên văn và nút `Xem nguồn`. Có thể chuyển chuột vào popover; trên thiết bị không hỗ trợ hover, nút xem trước mở/đóng popover.
- Prompt yêu cầu citation đặt ngay sau từng ý nhỏ được hỗ trợ, không đặt ở tiêu đề nhóm. Trích đoạn văn bản được chọn theo nội dung câu hỏi; evidence không đủ liên quan sẽ giữ nguyên thay vì bị diễn giải hoặc viết lại. Với bảng, context chỉ lấy các cột phù hợp câu hỏi cùng cột nhận diện hàng; nội dung ô gốc giữ nguyên.
- Với câu hỏi tổng hợp, câu trả lời chỉ tạo nhóm có bằng chứng, dùng heading đánh số và bullet phân cấp theo chủ đề/đối tượng/mốc; câu hỏi đơn giản vẫn trả lời ngắn gọn. Citation được giữ inline ngay sau thông tin thực tế trong từng bullet.
- Nút `Xem nguồn` hoặc citation mở đúng PDF/trang trong cùng phiên ứng dụng; mapping riêng theo citation được đăng ký trong Session State trước khi hiển thị, không bị ghi đè khi mở nhiều nguồn.
- Nút quay lại nằm dưới thông báo trạng thái bằng chứng. Nút Back của trình duyệt hoặc chuột dùng cùng một mục history tạm; nút trong ứng dụng cũng tiêu thụ mục này để không làm Back tiếp theo bị dừng ở trang nguồn.
- PDF chỉ được highlight khi toàn bộ evidence hiển thị khớp trên đúng trang. Bảng dùng bbox gắn với từng ô liên quan của đúng hàng và chỉ đánh dấu nếu text trong mọi bbox khớp; đoạn văn dùng so khớp chính xác từng đoạn evidence, chuẩn hóa Unicode/khoảng trắng. Nếu chỉ khớp một phần hoặc không xác minh được vị trí, trang vẫn mở nhưng không đánh dấu.
- Ingestion trích bảng thành các hàng có header, nội dung từng ô, trạng thái ô, nguồn ô, span và bbox. Header gộp nhiều cột được giữ cùng phạm vi; nếu nhãn cột con không có trong PDF, context ghi rõ chưa xác định thay vì đặt tên suy đoán. Ô kế thừa từ rowspan được giữ nguồn gốc, còn ô bị colspan che phủ không được coi là giá trị riêng. Các bảng tiếp trang chỉ được liên kết khi header khớp và caption không mâu thuẫn. Citation vẫn dùng đúng hàng/cột gốc và tọa độ dùng để highlight.
- BM25, FAISS và RRF chia sẻ chunk ID; truy vấn tổng hợp giá trị ưu tiên bảng có header phù hợp rồi mở rộng toàn bộ hàng cùng nhóm trước khi dựng context, để Top-K không loại mất biến thể/ngoại lệ của bảng.
- Câu hỏi đã tra cứu được khôi phục khi quay lại từ tài liệu nguồn. Ô câu hỏi chỉ tự bỏ focus khi nhấn Enter; thao tác gửi bằng nút không bị đổi.
- Chunk được cắt ở ranh giới từ để citation không bắt đầu/kết thúc giữa một từ. Nội dung liên hệ được giữ nguyên trong evidence/index để citation khớp nguồn; chuỗi có thể là số định danh vẫn bị từ chối trong bước kiểm tra ingestion.
- PDF gốc không bị sửa. PDF scan, OCR lỗi hoặc nội dung đã biến đổi có thể không hỗ trợ tìm văn bản để highlight.
- `st.pdf` cần extra `streamlit[pdf]`; citation component dùng Custom Components v2 và `markdown-it-py`, đã được khai báo trong `requirements.txt`.

## Chạy hệ thống

1. Đặt tài liệu được phép xử lý vào `data/raw/pdf/` hoặc `data/raw/web/`.
2. Nạp, kiểm tra và làm sạch:

   ```powershell
   python -m src.ingestion.ingestion_pipeline
   ```

   Kết quả gồm `pages_raw.jsonl`, `pages_clean.jsonl`, `documents.jsonl`, danh sách hợp lệ/bị loại và `ingestion_summary.json` trong `data/processed/`. PDF có text-layer được đọc theo thứ tự tọa độ trang; bảng được lưu theo hàng/cột với header, trạng thái ô, span và bbox. Header hiện trích từ hàng header do PyMuPDF nhận diện; nhãn con không có trong nguồn được đánh dấu chưa xác định, không suy diễn. Nếu số hiệu hoặc ngày ở header vẫn thiếu/không đầy đủ, pipeline chỉ OCR vùng header trang đầu và lưu nguồn/trang bằng chứng. Tài liệu scan không có text-layer vẫn cần OCR toàn văn riêng trước khi index.

3. Tạo hai index độc lập:

   ```powershell
   python -m src.indexing.index_manager --strategy both
   ```

   Có thể chỉ tạo một nhánh (`fixed_size` hoặc `recursive`) hoặc buộc tạo lại với `--force`. Chunks bảng mới được nhận diện qua fingerprint; sau khi cập nhật ingestion, hãy chạy lại bước này để tái lập cả BM25 và FAISS.

4. Chạy web app:

   ```powershell
   streamlit run src/app.py
   ```

   Bộ lọc loại văn bản, đơn vị ban hành, ngày ban hành và chủ đề được áp dụng trực tiếp khi truy xuất; trường metadata chưa xác định hiển thị rõ thay vì suy đoán.

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
- Pipeline dùng RapidOCR cục bộ có điều kiện trên vùng header; model OCR có thể được tải khi khởi tạo lần đầu. OCR chỉ bổ sung trường còn thiếu, không thay thế bằng chứng text-layer, và kết quả OCR được đánh dấu `needs_manual_review`.
- Metadata số hiệu/ngày ban hành ưu tiên header có nhãn/ngữ cảnh, giữ nguồn (`text_layer`/`ocr`), trang và raw evidence; giá trị thiếu/không hợp lệ không được suy đoán. PDF scan không có text-layer vẫn cần OCR toàn văn riêng trước khi index.
- LLM được gọi qua chat-completions tương thích OpenAI; khi thiếu cấu hình, UI vẫn hiển thị retrieval và nguồn, không giả lập câu trả lời.
- Tính hợp lệ citation kiểm tra ID evidence/trang của nguồn; không khẳng định mức độ entailment ngữ nghĩa tự động.
- Chỉ số so sánh không có ý nghĩa khi evaluation chưa được xác minh hoặc hai index dùng corpus/model khác nhau. Mỗi index manifest ghi fingerprint, model và cấu hình chunk.

# 📰 Advanced Fake News Detection System Using NLP & Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)

Hệ thống phát hiện tin tức giả mạo (Fake News) ứng dụng các kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP) tiên tiến. Dự án kiểm thử và so sánh hiệu năng giữa hai phương pháp nhúng từ (**Word2Vec**, **BERT**) kết hợp với 4 thuật toán phân loại (**Naive Bayes**, **Support Vector Machine (SVM)**, **Recurrent Neural Network (RNN)**, và **Long Short-Term Memory (LSTM)**).

---

## 📌 1. Giới thiệu bài toán & Mục tiêu nghiên cứu
Trong kỷ nguyên bùng nổ thông tin, tin giả (Fake News) lan truyền trên các mạng xã hội gây ra những hệ lụy nghiêm trọng cho đời sống xã hội và an ninh thông tin. Dự án này được thực hiện nhằm:
* Xây dựng mô hình phân loại tin tức tự động (Nhãn: **REAL** hoặc **FAKE**).
* Đánh giá thực nghiệm sức mạnh của đặc trưng ngữ nghĩa tĩnh (**Word2Vec**) so với đặc trưng ngữ cảnh động (**BERT**).
* Ứng dụng cơ chế **Ensemble (Học tổ hợp)** theo phương pháp trung bình trọng số xác xuất để tối ưu hóa độ tin cậy của dự đoán cuối cùng, giảm thiểu sai số của các mô hình đơn lẻ.

---

## 📂 2. Cấu trúc thư mục dự án
Thư mục gốc được tổ chức đồng bộ và khoa học trên Google Drive / GitHub như sau:

```text
FakeNewsDetection_Project/
│
├── Demo_App.ipynb          # File Notebook chính cấu hình giao diện web tương tác Gradio
├── README.md               # File tài liệu hướng dẫn và giới thiệu tổng quan dự án
├── requirements.txt        # Danh sách các thư viện Python bắt buộc của hệ thống
│
├── Dataset/                # Thư mục lưu trữ tập dữ liệu huấn luyện và kiểm thử (.csv)
│   ├── true.csv
│   └── fake.csv
│
├── Models/                 # Nơi lưu trữ các file mô hình đã huấn luyện
│   ├── bert_lstm_model.pt
│   ├── bert_nb_classifier.pkl
│   ├── bert_rnn_model.pt
│   ├── bert_svm_classifier.pkl
│   ├── keras_tokenizer.pkl
│   ├── model_info.txt
│   ├── model_summary.csv
│   ├── nb_classifier.pkl
│   ├── svm_classifier.pkl
│   ├── tfidf_nb_classifier.pkl
│   ├── tfidf_svm_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── w2v_lstm_model.h5
│   ├── w2v_rnn_model.h5
│   └── w2v_word_embedding.model
│
├── Notebooks/              # Chuỗi file Notebook thực nghiệm huấn luyện riêng biệt
│   ├── BERT_LSTM.ipynb
│   ├── BERT_RNN.ipynb
│   ├── BERT_NB.ipynb
│   ├── BERT_SVM.ipynb
│   ├── Word2Vec_LSTM.ipynb
│   ├── Word2Vec_RNN.ipynb
│   ├── Word2Vec_NB.ipynb
│   └── Word2Vec_SVM.ipynb
│
└── Preprocessing/          # Các script tiền xử lý dữ liệu và làm sạch văn bản
---
🧠 3. Kiến trúc hệ thống & Phương pháp tiếp cận
Hệ thống xử lý thông tin đi qua đường ống (Pipeline) 3 giai đoạn liền mạch và khép kín:

Giai đoạn 1: Tiền xử lý dữ liệu (Preprocessing)
Chuẩn hóa văn bản bằng cách chuyển toàn bộ ký tự sang chữ thường (lowercase). Hệ thống tiến hành loại bỏ các ký tự đặc biệt, dấu câu, các đường dẫn liên kết URL thừa và bộ từ dừng (stopwords), sau đó thực hiện tách từ (Tokenization) để cấu trúc hóa câu văn bài báo tối ưu.

Giai đoạn 2: Trích xuất đặc trưng (Embedding)

Mô hình tĩnh (Word2Vec): Huấn luyện biểu diễn từ dựa trên ngữ cảnh cục bộ bằng Skip-gram/CBOW, tính trung bình vector để đại diện cho toàn bộ văn bản.

Mô hình động (BERT): Sử dụng mã nguồn BertModel và BertTokenizer trích xuất vector tầng ẩn cuối cùng (cls token) nhằm giữ trọn vẹn ngữ nghĩa ngữ cảnh hai chiều của câu văn bài báo.

Giai đoạn 3: Thuật toán phân loại & Tích hợp tổ hợp (Classification & Ensemble)

Cấu hình đơn lẻ: Chạy độc lập một trong bốn thuật toán phân loại cốt lõi bao gồm Naive Bayes, SVM, RNN hoặc LSTM.

Cấu hình tổ hợp (Ensemble Learning): Lấy giá trị xác suất (Probability) dự đoán từ hai thuật toán khác nhau, thực hiện tính toán hiệu chuẩn toán học theo công thức lấy trung bình cộng để tối ưu hóa độ chính xác và giảm thiểu sai số:

📊 4. Kết quả thực nghiệm (Tổng hợp từ Tiểu luận)
Qua quá trình thực nghiệm diện rộng trên tập dữ liệu kiểm thử, hiệu năng của hệ thống tuân thủ nghiêm ngặt các quy luật toán học và bản chất thuật toán:

Đánh giá về kỹ thuật Embedding
Nhánh cấu hình ứng dụng mô hình BERT mang lại kết quả vượt trội rõ rệt so với công cụ Word2Vec truyền thống nhờ khả năng nắm bắt ngữ cảnh động linh hoạt và mối quan hệ hai chiều của các từ đứng cạnh nhau trong văn bản văn cảnh.

Đánh giá về thuật toán phân loại

Mô hình LSTM: Đạt độ chính xác cao nhất (vùng tiệm cận 96% khi kết hợp với BERT) nhờ cơ chế các cổng nhớ dài hạn, giải quyết triệt để vấn đề mất mát đạo hàm trên văn bản dài.

Mô hình RNN: Xếp vị trí thứ hai với độ nhạy thông tin chuỗi thời gian ở mức khá tốt.

Mô hình SVM: Đạt mức độ phân tách biên ổn định, phù hợp với các đặc trưng tuyến tính phẳng.

Mô hình Naive Bayes: Hoạt động dựa trên giả định độc lập điều kiện, cho tốc độ tính toán cực nhanh nhưng độ tin cậy thấp hơn các mô hình học sâu (Deep Learning).

🚀 5. Hướng dẫn cài đặt & Khởi chạy ứng dụng
Hệ thống yêu cầu môi trường cài đặt cơ bản chạy trên nền tảng Python từ phiên bản 3.8 trở lên (Khuyên dùng môi trường Google Colab để tận dụng miễn phí phần cứng GPU khi khởi chạy nhánh BERT).

Bước 1: Cài đặt các gói thư viện phụ thuộc
Mở cửa sổ dòng lệnh (Terminal/Prompt) và chạy câu lệnh sau để tự động cài đặt tất cả thư viện có trong file requirements.txt:

Bash
pip install -r requirements.txt
Bước 2: Kích hoạt giao diện ứng dụng (Gradio UI)
Bạn có thể khởi chạy trực tiếp file mã nguồn hệ thống bằng lệnh:

Bash
python app.py
Hệ thống sẽ tự động thiết lập một máy chủ Web cục bộ và cung cấp một đường link công khai có đuôi dạng https://xxxx.gradio.live để bạn chia sẻ công khai.

Bước 3: Trải nghiệm thực nghiệm trên giao diện ứng dụng

Bước 1: Lựa chọn công cụ nhúng từ mong muốn tại ô 1. Chọn Embedding Tool (Word2Vec hoặc BERT).

Bước 2: Chọn thuật toán phân loại cốt lõi tại ô 2. Thuật toán phân loại 1.

Bước 3: Tùy chọn Ensemble: Nếu muốn kích hoạt cơ chế tích hợp học tổ hợp, bạn chọn tiếp thuật toán thứ hai tại ô 3. Thuật toán phân loại 2. Nếu chỉ muốn chạy mô hình đơn lẻ thông thường, hãy giữ ô này ở trạng thái None.

Bước 4: Dán nội dung bài báo cần kiểm tra vào ô văn bản và bấm nút 🚀 Dự đoán để nhận kết quả phân tích độ tin cậy kèm bảng xếp hạng trực quan.

👤 6. Thông tin tác giả & Bản quyền
Dự án này được nghiên cứu, phát triển và cấu hình hoàn thiện bởi tác giả:

Tác giả: Nguyễn Văn Minh

Mã số sinh viên: 4651050158

Học phần: Nhập môn Xử lý ngôn ngữ tự nhiên (NLP)

Mã nguồn dự án được chia sẻ công khai với mục đích học tập, tham khảo học thuật và nghiên cứu khoa học phi thương mại.

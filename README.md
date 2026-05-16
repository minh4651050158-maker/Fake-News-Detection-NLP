# 📰 Advanced Fake News Detection System Using NLP & Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)

Hệ thống phát hiện tin tức giả mạo (Fake News) ứng dụng các kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP) tiên tiến. Dự án kiểm thử và so sánh hiệu năng giữa hai phương pháp nhúng từ (**Word2Vec**, **BERT**) kết hợp với 4 thuật toán phân loại (**Naive Bayes**, **Support Vector Machine (SVM)**, **Recurrent Neural Network (RNN)**, và **Long Short-Term Memory (LSTM)**).

---

## 📌 1. Giới thiệu bài toán & Mục tiêu cứu
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
│   └──w2v_word_embedding.model
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
```text
---
---

## 📌 1. Giới thiệu bài toán & Mục tiêu cứu
Trong kỷ nguyên bùng nổ thông tin, tin giả (Fake News) lan truyền trên các mạng xã hội gây ra những hệ lụy nghiêm trọng cho đời sống xã hội và an ninh thông tin. Dự án này được thực hiện nhằm:
* Xây dựng mô hình phân loại tin tức tự động (Nhãn: **REAL** hoặc **FAKE**).
* Đánh giá thực nghiệm sức mạnh của đặc trưng ngữ nghĩa tĩnh (**Word2Vec**) so với đặc trưng ngữ cảnh động (**BERT**).
* Ứng dụng cơ chế **Ensemble (Học tổ hợp)** theo phương pháp trung bình trọng số xác xuất để tối ưu hóa độ tin cậy của dự đoán cuối cùng, giảm thiểu sai số của các mô hình đơn lẻ.

---
## 🧠 **3. Kiến trúc hệ thống & Phương pháp tiếp cậnHệ thống xử lý thông tin đi qua đường ống (Pipeline) 3 giai đoạn:**
- Giai đoạn 1: Tiền xử lý (Preprocessing)Chuẩn hóa văn bản (chuyển chữ thường, xóa ký tự đặc biệt, đường dẫn URL, xóa từ dừng - stopwords).Tách từ (Tokenization) cấu trúc câu văn bài báo.
- Giai đoạn 2: Trích xuất đặc trưng (Embedding)Mô hình tĩnh (Word2Vec): Huấn luyện biểu diễn từ dựa trên ngữ cảnh cục bộ bằng Skip-gram/CBOW, tính trung bình vector để đại diện cho toàn bộ văn bản.Mô hình động (BERT): Sử dụng BertModel và BertTokenizer trích xuất vector tầng ẩn cuối cùng (cls token) để giữ trọn vẹn ngữ nghĩa ngữ cảnh hai chiều.
- Giai đoạn 3: Phân loại & Tích hợp tổ hợp (Classification & Ensemble)Hệ thống cho phép người dùng tùy chọn linh hoạt cấu hình chạy trên giao diện:Mô hình đơn lẻ: Chạy độc lập một trong 4 thuật toán: Naive Bayes, SVM, RNN hoặc LSTM.Mô hình tổ hợp (Ensemble): Lấy giá trị xác suất (Probability) dự đoán từ hai thuật toán khác nhau, thực hiện tính toán hiệu chuẩn toán học:
<img width="390" height="69" alt="image" src="https://github.com/user-attachments/assets/d05a2764-ce77-42d2-9020-6a6f612b5b31" />
--
## 📊 4. Kết quả thực nghiệm (Tổng hợp từ Tiểu luận)Qua quá trình thực nghiệm diện rộng trên tập dữ liệu kiểm thử, hiệu năng của hệ thống tuân thủ nghiêm ngặt các quy luật toán học và bản chất thuật toán:
Về kỹ thuật Embedding: Nhánh cấu hình BERT cho kết quả vượt trội rõ rệt so với Word2Vec nhờ khả năng nắm bắt ngữ cảnh động của từ.
Về thuật toán phân loại:
- LSTM đạt độ chính xác cao nhất (vùng tiệm cận 96% với BERT) nhờ cơ chế cổng nhớ dài hạn, giải quyết triệt để vấn đề mất mát đạo hàm.
- RNN xếp thứ hai với độ nhạy thông tin chuỗi cao.
- SVM đạt mức độ phân tách biên phân loại ổn định cao ở các bài toán tuyến tính.
- Naive Bayes hoạt động dựa trên giả định độc lập lập điều kiện, cho tốc độ xử lý cực nhanh nhưng độ tin cậy thấp hơn các mô hình học sâu.
---
## 🚀 5. Hướng dẫn cài đặt & Khởi chạy ứng dụngYêu cầu môi trườngPython >= 3.8Google Colab hoặc Máy tính cá nhân có hỗ trợ GPU (khuyên dùng để chạy BERT).Cài đặt thư việnCài đặt toàn bộ các gói thư viện phụ thuộc bằng lệnh:Bashpip install -r requirements.txt
Khởi chạy Giao diện kiểm thử (Gradio UI)Bạn tiến hành chạy ô Cell cuối trong file Demo_App.ipynb hoặc chạy trực tiếp file script bằng lệnh Terminal:Bashpython app.py
Hệ thống sẽ khởi tạo một máy chủ Web cục bộ kèm theo một đường link công khai dạng https://xxxx.gradio.live.🔧 Hướng dẫn trải nghiệm giao diện:
- Bước 1: Lựa chọn công cụ nhúng từ mong muốn tại ô 1. Chọn Embedding Tool (Word2Vec hoặc BERT).
- Bước 2: Chọn thuật toán tại ô 2. Thuật toán phân loại 1.
- Bước 3: Chọn thuật toán thứ hai tại ô 3. Thuật toán phân loại 2 để kích hoạt cơ chế bầu chọn tích hợp. Nếu muốn chạy mô hình đơn lẻ, hãy để ô này ở trạng thái None.
- Bước 4: Dán nội dung bài báo cần phân tích vào ô văn bản và bấm 🚀 Dự đoán. Hệ thống sẽ xuất ra kết luận cuối cùng cùng bảng đối chiếu hiệu năng đơn lẻ trực quan.
---
## 👤6. Thông tin tác giả & Bản quyền
- Tác giả: Nguyễn Văn Minh
- Mã số sinh viên: 4651050158
- Học phần: Nhập môn Xử lý ngôn ngữ tự nhiên (NLP)
- Dự án được xây dựng và chia sẻ với mục đích học tập, nghiên cứu khoa học phi thương mại.


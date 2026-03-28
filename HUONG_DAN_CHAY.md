# 📚 HƯỚNG DẪN CHẠY DỰ ÁN DỰ ĐOÁN CHẤT LƯỢNG KHÔNG KHÍ (AQI)

## 🎯 Tổng quan dự án

Dự án này cung cấp một hệ thống hoàn chỉnh để:
- **Tiền xử lý dữ liệu** từ NASA Giovanni
- **Phân tích dữ liệu** AQI và các chất ô nhiễm
- **Trực quan hóa dữ liệu** bằng biểu đồ
- **Huấn luyện mô hình ML** để dự đoán AQI
- **Giao diện web** với Streamlit và Chatbot AI

---

## 📋 Yêu cầu hệ thống

- **Python**: >= 3.8
- **Hệ điều hành**: Windows, Linux, hoặc macOS
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Dung lượng ổ cứng**: Tối thiểu 500MB cho dữ liệu và mô hình

---

## 🔧 Cài đặt môi trường

### Bước 1: Clone/Tải dự án

```bash
# Nếu dùng Git
git clone <repository-url>
cd Final_project_DAP

# Hoặc giải nén file ZIP nếu tải về
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

**Lưu ý**: Nếu gặp lỗi khi cài đặt, thử:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

---

## 🔑 Cấu hình API Keys

### Tạo file `.env`

Tạo file `.env` trong thư mục gốc của dự án với nội dung:

```env
# Google Gemini API Key (cho Chatbot)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key (nếu có, không bắt buộc)
OPENAI_API_KEY=your_openai_api_key_here
```

### Lấy API Key:

1. **Google Gemini API**:
   - Truy cập: https://makersuite.google.com/app/apikey
   - Đăng nhập với tài khoản Google
   - Tạo API key mới
   - Copy và paste vào file `.env`

---

## 📁 Cấu trúc thư mục

```
Final_project_DAP/
├── tien_xu_ly_du_lieu.py         # Tiền xử lý dữ liệu
├── data_analysis.py              # Phân tích dữ liệu
├── data_visualization.py         # Trực quan hóa dữ liệu
├── Train_model.py                # Huấn luyện mô hình ML
├── streamlit.py                  # Ứng dụng web Streamlit
├── chatbot_gemini.py             # Chatbot AI
├── requirements.txt               # Danh sách thư viện
├── HUONG_DAN_CHAY.md            # File này
├── Data_download/                # Thư mục chứa dữ liệu CSV gốc
├── processed/                    # Thư mục chứa dữ liệu đã xử lý
├── visualizations/               # Thư mục chứa biểu đồ
├── models/                       # Thư mục chứa mô hình đã train
└── .env                          # File cấu hình API keys
```

---

## 🚀 Hướng dẫn chạy từng bước

### ⚠️ QUAN TRỌNG: Chạy theo thứ tự sau đây!

---

### **Bước 1: Tiền xử lý dữ liệu** 

```bash
python tien_xu_ly_du_lieu.py
```

**Chức năng:**
- Đọc tất cả file CSV từ thư mục `Data_download/`
- Gộp dữ liệu từ nhiều nguồn (PM2.5, PM10, NO2, SO2, CO, O3, weather)
- Xử lý missing values
- Tính toán các features phái sinh (AQI, wind speed, humidity, etc.)
- Tạo file `processed/processed_aqi_dataset.csv`

**Thời gian chạy:** 5-15 phút (tùy vào số lượng file CSV)

**Kết quả:** File `processed/processed_aqi_dataset.csv`

---

### **Bước 2: Phân tích dữ liệu**

```bash
python data_analysis.py
```

**Chức năng:**
- Phân tích dữ liệu AQI đã xử lý
- Thực hiện 5 truy vấn phân tích:
  1. Xu hướng AQI theo thời gian
  2. Chất ô nhiễm quan trọng nhất
  3. Tác động của thời tiết lên AQI
  4. Mẫu theo mùa
  5. Tương quan giữa các chất ô nhiễm
- Lưu kết quả vào database SQLite: `aqi_analysis.db`

**Thời gian chạy:** 1-3 phút

**Kết quả:** Database `aqi_analysis.db` và file báo cáo

---

### **Bước 3: Trực quan hóa dữ liệu**

```bash
python data_visualization.py
```

**Chức năng:**
- Tạo 13 biểu đồ khác nhau:
  - Xu hướng AQI theo thời gian
  - Tầm quan trọng của chất ô nhiễm
  - Mẫu theo mùa
  - Tác động thời tiết
  - Tương quan chất ô nhiễm
  - Và nhiều biểu đồ khác...
- Lưu biểu đồ vào thư mục `visualizations/`

**Thời gian chạy:** 2-5 phút

**Kết quả:** Các file PNG trong thư mục `visualizations/`

---

### **Bước 4: Huấn luyện mô hình Machine Learning**

```bash
python Train_model.py
```

**Chức năng:**
- Huấn luyện nhiều mô hình ML (Random Forest, Gradient Boosting, SVM, etc.)
- Dự đoán cho nhiều horizons: T+1, T+7, T+30, T+365 (ngày, tuần, tháng, năm)
- Dự đoán nhiều targets: AQI, temperature, và các chất ô nhiễm
- Tự động chọn top 2 mô hình tốt nhất cho mỗi horizon
- Lưu mô hình vào thư mục `models/`

**Thời gian chạy:** 30-60 phút (tùy vào cấu hình máy)

**Kết quả:** 
- Các file mô hình trong `models/horizon_1/`, `models/horizon_7/`, `models/horizon_30/`, `models/horizon_365/`

---

### **Bước 5: Chạy ứng dụng web Streamlit**

```bash
streamlit run streamlit.py
```

**Chức năng:**
- Giao diện web tương tác để:
  - Xem dữ liệu AQI
  - Phân tích xu hướng
  - Dự đoán AQI cho tương lai
  - Chat với AI về AQI và sức khỏe
- Tự động mở trình duyệt ở địa chỉ: `http://localhost:8501`

**Thời gian chạy:** Chạy liên tục (nhấn Ctrl+C để dừng)

**Kết quả:** Giao diện web trong trình duyệt

---

## 🔍 Kiểm tra kết quả

### Sau Bước 1:
- Kiểm tra file: `processed/processed_aqi_dataset.csv`
- File này phải có > 0 dòng dữ liệu

### Sau Bước 2:
- Kiểm tra file: `aqi_analysis.db`
- File này chứa kết quả phân tích

### Sau Bước 3:
- Kiểm tra thư mục: `visualizations/`
- Phải có 13 file PNG (hoặc nhiều hơn)

### Sau Bước 4:
- Kiểm tra thư mục: `models/horizon_1/`, `models/horizon_7/`, `models/horizon_30/`, `models/horizon_365/`
- Mỗi thư mục phải có ít nhất 2 file `.pkl` (mô hình) và file `metadata.json`

### Sau Bước 5:
- Mở trình duyệt và kiểm tra giao diện web hoạt động

---

## ❗ Xử lý lỗi thường gặp

### Lỗi 1: "Không tìm thấy file processed_aqi_dataset.csv"

**Nguyên nhân:** Chưa chạy Bước 1 hoặc chạy sai thứ tự

**Giải pháp:** Chạy lại `tien_xu_ly_du_lieu.py` trước

---

### Lỗi 2: "ModuleNotFoundError: No module named 'xxx'"

**Nguyên nhân:** Chưa cài đặt đầy đủ thư viện

**Giải pháp:**
```bash
pip install -r requirements.txt
```

---

### Lỗi 3: "GEMINI_API_KEY not found"

**Nguyên nhân:** Chưa tạo file `.env` hoặc chưa cấu hình API key

**Giải pháp:** 
1. Tạo file `.env` trong thư mục gốc
2. Thêm dòng: `GEMINI_API_KEY=your_api_key_here`

---

### Lỗi 4: "AttributeError: 'OutStream' object has no attribute 'buffer'"

**Nguyên nhân:** Chạy script trong Jupyter Notebook (không phải terminal)

**Giải pháp:** Script này được thiết kế để chạy trong terminal, không phải Jupyter. Nếu muốn chạy trong Jupyter, sử dụng các file `.ipynb` trong thư mục `File ipynb/`

---

### Lỗi 5: "Không tìm thấy model"

**Nguyên nhân:** Chưa chạy Bước 4 (train model)

**Giải pháp:** Chạy `Train_model.py` trước khi sử dụng tính năng dự đoán trong Streamlit

---

## 📊 Sử dụng dữ liệu mới

Nếu bạn có dữ liệu CSV mới từ NASA Giovanni:

1. Đặt các file CSV vào thư mục `Data_download/`
2. Chạy lại `tien_xu_ly_du_lieu.py` để xử lý dữ liệu mới
3. Chạy lại các bước tiếp theo nếu cần

**Lưu ý:** Format file CSV phải đúng chuẩn NASA Giovanni (có 8 dòng header metadata)

---

## 🎓 Tùy chỉnh

### Thay đổi đường dẫn dữ liệu:

Mở file và sửa biến `csv_path`:
- `tien_xu_ly_du_lieu.py`: Sửa đường dẫn `Data_download/`
- `data_analysis.py`: Sửa đường dẫn `processed/processed_aqi_dataset.csv`
- `data_visualization.py`: Tương tự
- `Train_model.py`: Tương tự

### Thay đổi cấu hình mô hình:

Mở file `Train_model.py` và sửa:
- `horizons`: Danh sách horizons muốn train
- `targets`: Danh sách targets muốn dự đoán
- Hyperparameters của các mô hình

---

## 📞 Hỗ trợ

Nếu gặp vấn đề, kiểm tra:
1. Đã cài đặt đầy đủ thư viện chưa?
2. Đã chạy các bước theo đúng thứ tự chưa?
3. File `.env` đã được cấu hình đúng chưa?
4. Dữ liệu CSV có đúng format không?

---

## 📝 Ghi chú

- **Python 3.8+**: Dự án yêu cầu Python 3.8 trở lên
- **Windows Encoding**: Scripts đã được cấu hình để xử lý encoding UTF-8 trên Windows
- **Memory**: Dữ liệu lớn có thể cần nhiều RAM, nếu gặp lỗi "Memory Error", thử giảm số lượng features hoặc sử dụng máy có RAM lớn hơn

---

## ✅ Checklist hoàn thành

- [ ] Đã cài đặt Python 3.8+
- [ ] Đã tạo virtual environment
- [ ] Đã cài đặt `requirements.txt`
- [ ] Đã tạo file `.env` với API keys
- [ ] Đã chạy `tien_xu_ly_du_lieu.py`
- [ ] Đã chạy `data_analysis.py`
- [ ] Đã chạy `data_visualization.py`
- [ ] Đã chạy `Train_model.py`
- [ ] Đã chạy `streamlit run streamlit.py`

---

**Chúc bạn thành công! 🎉**


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App - Dự đoán Chất lượng Không khí (AQI)
Giao diện web để phân tích và dự đoán AQI
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import json
from chatbot_gemini import ChatbotGemini
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Cấu hình trang
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .aqi-good { color: #00E400; font-weight: bold; }
    .aqi-moderate { color: #FFFF00; font-weight: bold; }
    .aqi-unhealthy-sens { color: #FF7E00; font-weight: bold; }
    .aqi-unhealthy { color: #FF0000; font-weight: bold; }
    .aqi-very-unhealthy { color: #8F3F97; font-weight: bold; }
    .aqi-hazardous { color: #7E0023; font-weight: bold; }
    .stChatbot {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 400px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'chatbot' not in st.session_state:
    try:
        st.session_state.chatbot = ChatbotGemini()
        st.session_state.chat_history = []
    except Exception as e:
        st.session_state.chatbot = None
        st.session_state.chat_error = str(e)

if 'data' not in st.session_state:
    st.session_state.data = None

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🌬️ Controls")
st.sidebar.markdown("---")

# Page selection
page = st.sidebar.selectbox(
    "Select page",
    ["📊 Overview", "📈 Phân Tích Dữ Liệu", "📉 Biểu Đồ", "🔮 Dự Đoán"],
    index=0
)

st.sidebar.markdown("---")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data_from_csv(csv_path=None):
    """Load dữ liệu từ CSV"""
    # Đặt đường dẫn mặc định từ thư mục processed
    if csv_path is None:
        csv_path = os.path.join("processed", "processed_aqi_dataset.csv")
    
    try:
        # Kiểm tra file có tồn tại không
        if not os.path.exists(csv_path):
            st.error(f"❌ Không tìm thấy file: {csv_path}")
            st.info("💡 Vui lòng chạy tien_xu_ly_du_lieu.py trước để tạo file dữ liệu đã xử lý")
            return None
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
        return None

def categorize_aqi(aqi):
    """Phân loại AQI và trả về màu nền/màu chữ phù hợp"""
    if pd.isna(aqi):
        return 'Unknown', '#808080', '#FFFFFF'  # Grey, White text
    elif aqi <= 50:
        return 'Good', '#00E400', '#000000'  # Green, Black text
    elif aqi <= 100:
        return 'Moderate', '#FFD700', '#000000'  # Gold (brighter yellow), Black text
    elif aqi <= 150:
        return 'Unhealthy for Sensitive', '#FF7E00', '#FFFFFF'  # Orange, White text
    elif aqi <= 200:
        return 'Unhealthy', '#FF0000', '#FFFFFF'  # Red, White text
    elif aqi <= 300:
        return 'Very Unhealthy', '#8F3F97', '#FFFFFF'  # Purple, White text
    else:
        return 'Hazardous', '#7E0023', '#FFFFFF'  # Dark Red, White text

def get_aqi_color(aqi):
    """Lấy màu nền cho AQI"""
    _, bg_color, _ = categorize_aqi(aqi)
    return bg_color

def get_aqi_text_color(aqi):
    """Lấy màu chữ cho AQI dựa trên category"""
    _, _, text_color = categorize_aqi(aqi)
    return text_color

def load_model(horizon=1, model_idx='model1'):
    """
    Load model đã train cho horizon cụ thể
    Args:
        horizon: Số ngày dự đoán (1, 7, 30, 365)
        model_idx: 'model1' hoặc 'model2' (top 2 models)
    """
    try:
        horizon_dir = f"models/horizon_{horizon}"
        
        # Đọc summary để biết model nào có sẵn
        summary_path = "models/models_summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # Tìm model cho horizon này
            model_key = f"T+{horizon}_{model_idx}"
            if model_key in summary['all_models']:
                model_info = summary['all_models'][model_key]
                model_name = model_info['model_name']
                
                model_path = os.path.join(horizon_dir, f"model_{model_idx}_{model_name}.pkl")
                metadata_path = os.path.join(horizon_dir, f"metadata_{model_idx}_{model_name}.json")
                
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Load scalers (X và Y)
                    scaler_X_path = "models/scaler_X.pkl"
                    scaler_y_path = "models/scaler_y.pkl"
                    
                    scaler_X = None
                    scaler_y = None
                    
                    if os.path.exists(scaler_X_path):
                        with open(scaler_X_path, 'rb') as f:
                            scaler_X = pickle.load(f)
                    
                    if os.path.exists(scaler_y_path):
                        with open(scaler_y_path, 'rb') as f:
                            scaler_y = pickle.load(f)
                    
                    # Fallback: thử load scaler cũ (backward compatibility)
                    if scaler_X is None:
                        scaler_path = "models/scaler.pkl"
                        if os.path.exists(scaler_path):
                            with open(scaler_path, 'rb') as f:
                                scaler_X = pickle.load(f)
                    
                    return model, scaler_X, metadata, scaler_y
        
        # Fallback: thử load model cũ (backward compatibility)
        model_path = "models/best_model_GradientBoosting.pkl"
        scaler_path = "models/scaler.pkl"
        metadata_path = "models/model_metadata.json"
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            scaler_X = None
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler_X = pickle.load(f)
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return model, scaler_X, metadata, None
        
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def predict_recursive(model, scaler_X, metadata, base_data, df, feature_cols, days_ahead, scaler_y=None):
    """
    Dự đoán recursive: dùng model T+1 để dự đoán tuần tự từng ngày
    Args:
        model: Model đã train (T+1)
        scaler: Scaler cho features
        metadata: Metadata của model
        base_data: Dữ liệu ngày bắt đầu
        df: DataFrame đầy đủ (để lấy thông tin về features)
        feature_cols: Danh sách features
        days_ahead: Số ngày cần dự đoán
    Returns:
        predictions: Dict chứa predictions cho tất cả targets
    """
    import pandas as pd
    import numpy as np
    
    # Lấy targets từ metadata
    targets = metadata.get('targets', ['aqi'])
    model_name = metadata.get('model_name', '')
    
    # Khởi tạo predictions
    predictions_sequence = []
    current_data = base_data.copy()
    
    # Đảm bảo season là số (0, 1, 2, 3) thay vì string
    if 'season' in current_data:
        season_val = current_data['season']
        if isinstance(season_val, str):
            # Chuyển string sang số
            season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3, 'Autumn': 3}
            current_data['season'] = season_map.get(season_val, 0)
        elif season_val not in [0, 1, 2, 3]:
            # Nếu không phải string và không phải 0-3, tính từ month
            month = current_data.get('month', pd.Timestamp(current_data['date']).month if 'date' in current_data else 1)
            if month in [12, 1, 2]:
                current_data['season'] = 0
            elif month in [3, 4, 5]:
                current_data['season'] = 1
            elif month in [6, 7, 8]:
                current_data['season'] = 2
            else:
                current_data['season'] = 3
    
    # Dự đoán từng ngày
    for day in range(days_ahead):
        # Chuẩn bị features từ current_data
        X_input = current_data[feature_cols].values.reshape(1, -1)
        
        # Scale features (tất cả models dùng scaled data)
        if scaler_X is not None:
            X_input_scaled = scaler_X.transform(X_input)
            pred_scaled = model.predict(X_input_scaled)[0]
            
            # Inverse transform nếu có scaler_y
            if scaler_y is not None:
                targets = metadata.get('targets', ['aqi'])
                pred = np.zeros(len(targets))
                for idx, target in enumerate(targets):
                    if 1 in scaler_y and target in scaler_y[1]:  # Dùng horizon 1 cho recursive
                        pred[idx] = scaler_y[1][target].inverse_transform(
                            pred_scaled[idx].reshape(-1, 1)
                        )[0, 0]
                    else:
                        pred[idx] = pred_scaled[idx]
            else:
                pred = pred_scaled
        else:
            pred = model.predict(X_input)[0]
        
        # Xử lý predictions (có thể là array hoặc scalar)
        if isinstance(pred, np.ndarray):
            pred_dict = {}
            for idx, target in enumerate(targets):
                if idx < len(pred):
                    pred_dict[target] = pred[idx]
        else:
            # Single target (backward compatibility)
            pred_dict = {'aqi': pred}
        
        # Giới hạn AQI trong range hợp lý
        if 'aqi' in pred_dict:
            pred_dict['aqi'] = max(0, min(500, pred_dict['aqi']))
        
        predictions_sequence.append(pred_dict)
        
        # Cập nhật current_data cho lần dự đoán tiếp theo
        # Tạo một row mới với predictions
        next_date = current_data['date'] + timedelta(days=1)
        
        # Cập nhật targets từ predictions
        for target in targets:
            if target in pred_dict:
                current_data[target] = pred_dict[target]
        
        # Cập nhật time features
        current_data['date'] = next_date
        current_data['year'] = next_date.year
        current_data['month'] = next_date.month
        current_data['day'] = next_date.day
        current_data['weekday'] = next_date.weekday()
        current_data['day_of_year'] = next_date.timetuple().tm_yday
        current_data['is_weekend'] = 1 if next_date.weekday() >= 5 else 0
        
        # Cập nhật season (phải là số: 0=Winter, 1=Spring, 2=Summer, 3=Fall)
        month = next_date.month
        if month in [12, 1, 2]:
            current_data['season'] = 0  # Winter
        elif month in [3, 4, 5]:
            current_data['season'] = 1  # Spring
        elif month in [6, 7, 8]:
            current_data['season'] = 2  # Summer
        else:
            current_data['season'] = 3  # Fall
        
        # Cập nhật lag features (shift về 1 ngày)
        # Ví dụ: PM2_5_lag1 = PM2_5 hiện tại, PM2_5_lag7 = PM2_5 của 6 ngày trước
        for target in targets:
            if f'{target}_lag1' in feature_cols:
                current_data[f'{target}_lag1'] = pred_dict.get(target, current_data.get(target, np.nan))
            if f'{target}_lag7' in feature_cols:
                # Lấy từ predictions_sequence (nếu có)
                if len(predictions_sequence) >= 7:
                    current_data[f'{target}_lag7'] = predictions_sequence[-7][target]
                elif len(predictions_sequence) >= 6:
                    # Nếu chưa đủ 7 ngày, dùng giá trị từ base_data
                    lag7_date = next_date - timedelta(days=7)
                    if lag7_date in df['date'].values:
                        lag7_data = df[df['date'] == lag7_date].iloc[0]
                        current_data[f'{target}_lag7'] = lag7_data.get(target, np.nan)
        
        # Cập nhật rolling features (tính từ predictions_sequence)
        for target in targets:
            # Rolling 7 mean
            if f'{target}_rolling7_mean' in feature_cols:
                if len(predictions_sequence) >= 7:
                    recent_vals = [p[target] for p in predictions_sequence[-7:]]
                    current_data[f'{target}_rolling7_mean'] = np.mean(recent_vals)
                elif len(predictions_sequence) > 0:
                    recent_vals = [p[target] for p in predictions_sequence]
                    current_data[f'{target}_rolling7_mean'] = np.mean(recent_vals)
            
            # Rolling 7 std
            if f'{target}_rolling7_std' in feature_cols:
                if len(predictions_sequence) >= 7:
                    recent_vals = [p[target] for p in predictions_sequence[-7:]]
                    current_data[f'{target}_rolling7_std'] = np.std(recent_vals) if len(recent_vals) > 1 else 0
                elif len(predictions_sequence) > 1:
                    recent_vals = [p[target] for p in predictions_sequence]
                    current_data[f'{target}_rolling7_std'] = np.std(recent_vals)
                else:
                    current_data[f'{target}_rolling7_std'] = 0
            
            # Rolling 14 mean
            if f'{target}_rolling14_mean' in feature_cols:
                if len(predictions_sequence) >= 14:
                    recent_vals = [p[target] for p in predictions_sequence[-14:]]
                    current_data[f'{target}_rolling14_mean'] = np.mean(recent_vals)
                elif len(predictions_sequence) > 0:
                    recent_vals = [p[target] for p in predictions_sequence]
                    current_data[f'{target}_rolling14_mean'] = np.mean(recent_vals)
            
            # Rolling 14 std
            if f'{target}_rolling14_std' in feature_cols:
                if len(predictions_sequence) >= 14:
                    recent_vals = [p[target] for p in predictions_sequence[-14:]]
                    current_data[f'{target}_rolling14_std'] = np.std(recent_vals) if len(recent_vals) > 1 else 0
                elif len(predictions_sequence) > 1:
                    recent_vals = [p[target] for p in predictions_sequence]
                    current_data[f'{target}_rolling14_std'] = np.std(recent_vals)
                else:
                    current_data[f'{target}_rolling14_std'] = 0
        
        # Fill missing features với giá trị từ base_data hoặc mean
        for col in feature_cols:
            if col not in current_data or pd.isna(current_data[col]):
                if col in base_data:
                    current_data[col] = base_data[col]
                else:
                    # Lấy từ df nếu có
                    if col in df.columns:
                        current_data[col] = df[col].mean()
                    else:
                        current_data[col] = 0
    
    # Trả về prediction cuối cùng (cho ngày được yêu cầu)
    return predictions_sequence[-1] if predictions_sequence else {}

def get_ai_recommendations(aqi, base_data, forecast_date, predictions_dict=None):
    """
    Tạo khuyến nghị bằng AI (Gemini) dựa trên AQI và các yếu tố khác
    Nếu không có API key hoặc lỗi, sẽ fallback về logic cũ
    """
    try:
        # Kiểm tra xem có chatbot trong session state không
        if 'chatbot' not in st.session_state:
            return None
        
        chatbot = st.session_state.chatbot
        
        # Chuẩn bị dữ liệu đầu vào
        category, _, _ = categorize_aqi(aqi)
        
        # Lấy thông tin mùa và nhiệt độ
        month = forecast_date.month
        temp_max_raw = base_data.get('temperature_max', np.nan)
        temp_min_raw = base_data.get('temperature_min', np.nan)
        temp_mean_raw = base_data.get('temperature_mean', np.nan)
        
        # Convert nhiệt độ từ Kelvin sang Celsius nếu cần
        if not pd.isna(temp_max_raw) and temp_max_raw > 50:
            temp_max = temp_max_raw - 273.15
        else:
            temp_max = temp_max_raw
        
        if not pd.isna(temp_min_raw) and temp_min_raw > 50:
            temp_min = temp_min_raw - 273.15
        else:
            temp_min = temp_min_raw
            
        if not pd.isna(temp_mean_raw) and temp_mean_raw > 50:
            temp_mean = temp_mean_raw - 273.15
        else:
            temp_mean = temp_mean_raw
        
        # Xác định mùa
        if month in [12, 1, 2]:
            season_name = "Đông"
        elif month in [3, 4, 5]:
            season_name = "Xuân"
        elif month in [6, 7, 8]:
            season_name = "Hè"
        else:
            season_name = "Thu"
        
        # Lấy thông tin các chất ô nhiễm từ predictions_dict hoặc base_data
        pm25 = predictions_dict.get('PM2_5', base_data.get('PM2_5', np.nan)) if predictions_dict else base_data.get('PM2_5', np.nan)
        pm10 = predictions_dict.get('PM10', base_data.get('PM10', np.nan)) if predictions_dict else base_data.get('PM10', np.nan)
        humidity = predictions_dict.get('humidity', base_data.get('humidity', np.nan)) if predictions_dict else base_data.get('humidity', np.nan)
        wind_speed = predictions_dict.get('wind_speed', base_data.get('wind_speed', np.nan)) if predictions_dict else base_data.get('wind_speed', np.nan)
        
        # Xác định mùa hoa sữa (1/10 - 15/11)
        day = forecast_date.day
        is_flower_season = (month == 10) or (month == 11 and day <= 15)
        
        # Tạo prompt cho Gemini
        prompt = f"""BẠN LÀ MỘT CHUYÊN GIA VỀ CHẤT LƯỢNG KHÔNG KHÍ VÀ SỨC KHỎE CỘNG ĐỒNG TẠI VIỆT NAM.

DỰA VÀO CÁC THÔNG TIN SAU, HÃY ĐƯA RA KHUYẾN NGHỊ CHI TIẾT VÀ CỤ THỂ VỀ SỨC KHỎE VÀ HOẠT ĐỘNG:

📊 THÔNG TIN DỰ ĐOÁN:
- Ngày dự đoán: {forecast_date.strftime('%d/%m/%Y')}
- Chỉ số AQI: {aqi:.1f}
- Phân loại: {category}
- Mùa: {season_name}
- Nhiệt độ: {temp_mean:.1f}°C (min: {temp_min:.1f}°C, max: {temp_max:.1f}°C)
- Độ ẩm: {humidity:.1f}% (nếu có)
- Tốc độ gió: {wind_speed:.1f} m/s (nếu có)
- PM2.5: {pm25:.2f} μg/m³ (nếu có)
- PM10: {pm10:.2f} μg/m³ (nếu có)
"""
        
        if is_flower_season:
            prompt += f"- ⚠️ Mùa hoa sữa (từ 1/10 đến 15/11) - Cảnh báo cho người dị ứng phấn hoa\n"
        
        prompt += f"""
YÊU CẦU:
1. Đưa ra tiêu đề ngắn gọn (1 dòng) về tình trạng không khí
2. Liệt kê 3-5 hành động cụ thể người dùng NÊN LÀM (mỗi hành động 1 dòng, ngắn gọn)
3. Đánh giá rủi ro sức khỏe (1-2 dòng)
4. Trả lời bằng tiếng Việt, thân thiện, dễ hiểu
5. Có thể đề cập đến:
   - Khuyến nghị đeo khẩu trang (loại nào, khi nào)
   - Hoạt động ngoài trời (nên/không nên làm gì)
   - Bảo vệ sức khỏe (đặc biệt cho người nhạy cảm, trẻ em, người già)
   - Mùa {season_name} (nếu có ảnh hưởng)
   - Nhiệt độ (nếu quá lạnh hoặc quá nóng)
   - Mùa hoa sữa (nếu từ 1/10 đến 15/11, cảnh báo người dị ứng)

ĐỊNH DẠNG TRẢ LỜI (JSON):
{{
    "title": "Tiêu đề ngắn gọn",
    "actions": [
        "Hành động 1",
        "Hành động 2",
        "Hành động 3"
    ],
    "health": "Đánh giá rủi ro sức khỏe"
}}

CHỈ TRẢ LỜI JSON, KHÔNG THÊM BẤT KỲ VĂN BẢN NÀO KHÁC.
"""
        
        # Gọi API Gemini
        response = chatbot.model.generate_content(prompt)
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Xóa markdown code blocks nếu có
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        response_text = response_text.strip()
        
        # Parse JSON
        import json
        rec = json.loads(response_text)
        
        # Đảm bảo format đúng
        if 'title' not in rec:
            rec['title'] = f"⚠️ {category}"
        if 'actions' not in rec:
            rec['actions'] = []
        if 'health' not in rec:
            rec['health'] = 'Cần theo dõi sức khỏe'
        
        return rec
        
    except Exception as e:
        # Nếu có lỗi, fallback về logic cũ
        return None

def get_recommendations(aqi, base_data, forecast_date, predictions_dict=None):
    """
    Lấy khuyến nghị dựa trên AQI, mùa, nhiệt độ và thời tiết
    Ưu tiên dùng AI, nếu không có thì dùng logic cũ
    """
    # Thử dùng AI trước
    ai_rec = get_ai_recommendations(aqi, base_data, forecast_date, predictions_dict)
    if ai_rec is not None:
        return ai_rec
    
    # Fallback về logic cũ
    category, _, _ = categorize_aqi(aqi)
    
    # Lấy thông tin mùa và nhiệt độ
    month = forecast_date.month
    season = base_data.get('season', 'Unknown')
    temp_max_raw = base_data.get('temperature_max', np.nan)
    temp_min_raw = base_data.get('temperature_min', np.nan)
    temp_mean_raw = base_data.get('temperature_mean', np.nan)
    
    # Convert nhiệt độ từ Kelvin sang Celsius nếu cần
    if not pd.isna(temp_max_raw) and temp_max_raw > 50:
        temp_max = temp_max_raw - 273.15
    else:
        temp_max = temp_max_raw
    
    if not pd.isna(temp_min_raw) and temp_min_raw > 50:
        temp_min = temp_min_raw - 273.15
    else:
        temp_min = temp_min_raw
    
    if not pd.isna(temp_mean_raw) and temp_mean_raw > 50:
        temp_mean = temp_mean_raw - 273.15
    else:
        temp_mean = temp_mean_raw
    
    # Xác định mùa
    day = forecast_date.day
    
    if month in [12, 1, 2]:
        season_name = "Đông"
        is_winter = True
        is_flower_season = False
    elif month in [3, 4, 5]:
        season_name = "Xuân"
        is_winter = False
        is_flower_season = False
    elif month in [6, 7, 8]:
        season_name = "Hè"
        is_winter = False
        is_flower_season = False
    else:  # 9, 10, 11
        season_name = "Thu"
        is_winter = False
        # Mùa hoa sữa: từ 1/10 đến 15/11 (khoảng 45 ngày)
        is_flower_season = (month == 10) or (month == 11 and day <= 15)
    
    # Xác định nhiệt độ
    is_cold = not pd.isna(temp_mean) and temp_mean < 20
    is_hot = not pd.isna(temp_mean) and temp_mean > 30
    
    # Khuyến nghị cơ bản
    base_recommendations = {
        'Good': {
            'title': '✅ Không khí trong lành, an toàn',
            'actions': [],
            'health': 'Không có rủi ro sức khỏe'
        },
        'Moderate': {
            'title': '⚠️ Không khí bình thường',
            'actions': [],
            'health': 'Rủi ro thấp cho người nhạy cảm'
        },
        'Unhealthy for Sensitive': {
            'title': '⚠️ Không tốt cho người nhạy cảm',
            'actions': [],
            'health': 'Rủi ro trung bình cho người nhạy cảm'
        },
        'Unhealthy': {
            'title': '🔴 Không tốt cho mọi người',
            'actions': [],
            'health': 'Rủi ro cao cho sức khỏe'
        },
        'Very Unhealthy': {
            'title': '🔴 Rất không tốt',
            'actions': [],
            'health': 'Rủi ro rất cao cho sức khỏe'
        },
        'Hazardous': {
            'title': '🔴 Nguy hiểm',
            'actions': [],
            'health': 'Rủi ro cực kỳ cao - Khẩn cấp'
        }
    }
    
    rec = base_recommendations.get(category, base_recommendations['Moderate'])
    
    # Thêm khuyến nghị theo AQI
    if category == 'Good':
        rec['actions'].extend([
            'Có thể hoạt động ngoài trời bình thường',
            'Tập thể dục ngoài trời an toàn',
            'Mở cửa sổ để thông gió'
        ])
    elif category == 'Moderate':
        rec['actions'].extend([
            'Người nhạy cảm nên hạn chế hoạt động ngoài trời',
            'Có thể đeo khẩu trang khi ra ngoài',
            'Tránh tập thể dục cường độ cao ngoài trời'
        ])
    elif category in ['Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy', 'Hazardous']:
        rec['actions'].extend([
            'Nên đeo khẩu trang khi ra ngoài',
            'Hạn chế hoạt động ngoài trời',
            'Đóng cửa sổ khi ở nhà',
            'Sử dụng máy lọc không khí'
        ])
        if category in ['Unhealthy', 'Very Unhealthy', 'Hazardous']:
            rec['actions'].insert(0, 'Bắt buộc đeo khẩu trang N95 khi ra ngoài')
    
    # Thêm khuyến nghị theo mùa và nhiệt độ
    if is_winter or is_cold:
        rec['actions'].append('🌨️ Mùa đông lạnh - Nên mặc áo khoác, giữ ấm cơ thể')
    
    if is_hot:
        rec['actions'].append('☀️ Thời tiết nắng nóng - Nên uống đủ nước, tránh ánh nắng trực tiếp')
    
    if is_flower_season:
        rec['actions'].append(f'🌸 Mùa hoa sữa ({forecast_date.strftime("%d/%m")} - 15/11) - Người bị dị ứng phấn hoa nên hạn chế ra ngoài, đeo khẩu trang')
    
    return rec

# ============================================================================
# PAGE 1: OVERVIEW - Upload Data & Analyze
# ============================================================================
def page_overview():
    st.markdown('<h1 class="main-header">📊 Overview - Phân Tích Dữ Liệu</h1>', unsafe_allow_html=True)
    
    # Mục tiêu dự án - Đẩy lên đầu
    st.markdown("### 🎯 Mục Đích Dự Án")
    st.info("""
    **Dự án này nhằm dự đoán chất lượng không khí (AQI) vào ngày mai dựa trên dữ liệu hôm nay.**
    
    - 📊 **Phân tích**: Hiểu rõ các yếu tố ảnh hưởng đến AQI
    - 🔮 **Dự đoán**: Sử dụng Machine Learning để dự đoán AQI ngày mai
    - 💡 **Khuyến nghị**: Đưa ra lời khuyên về sức khỏe dựa trên AQI dự đoán
    - 🛡️ **Bảo vệ**: Giúp người dùng bảo vệ sức khỏe trước ô nhiễm không khí
    """)
    
    # Load data mặc định
    if st.session_state.data is None:
        with st.spinner("Đang tải dữ liệu..."):
            st.session_state.data = load_data_from_csv()
    
    if st.session_state.data is None:
        csv_path = os.path.join("processed", "processed_aqi_dataset.csv")
        st.error(f"❌ Không thể tải dữ liệu từ file {csv_path}")
        st.info("💡 Vui lòng chạy tien_xu_ly_du_lieu.py trước để tạo file dữ liệu đã xử lý")
        return
    
    df = st.session_state.data
    
    if df is None or len(df) == 0:
        st.error("❌ Không có dữ liệu")
        return
    
    st.success(f"✅ Đã tải {len(df):,} records từ file mặc định")
    
    # Hiển thị thông tin cơ bản
    st.markdown("---")
    st.markdown("### 📋 Thông Tin Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng số records", f"{len(df):,}")
    with col2:
        st.metric("Số features", len(df.columns))
    with col3:
        if 'aqi' in df.columns:
            st.metric("AQI trung bình", f"{df['aqi'].mean():.1f}")
    
    # Demo 5 dòng đầu
    st.markdown("---")
    st.markdown("### 📊 Demo 5 Dòng Đầu")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Giải thích features
    st.markdown("---")
    st.markdown("### 📖 Giải Thích Features")
    
    feature_groups = {
        "📅 Time Features": {
            "year": "Năm",
            "month": "Tháng (1-12)",
            "day": "Ngày trong tháng",
            "weekday": "Thứ trong tuần (0=Monday, 6=Sunday)",
            "day_of_year": "Ngày thứ bao nhiêu trong năm (1-365/366)",
            "is_weekend": "Có phải cuối tuần không (0/1)",
            "season": "Mùa (Spring/Summer/Fall/Winter)"
        },
        "🌫️ Pollutants (Chất ô nhiễm)": {
            "PM2_5": "Bụi mịn PM2.5 (μg/m³) - Bụi có đường kính ≤ 2.5μm, rất nguy hiểm",
            "PM10": "Bụi mịn PM10 (μg/m³) - Bụi có đường kính ≤ 10μm",
            "NO2_trop": "Nitrogen Dioxide tropospheric (molecules/cm²) - Khí độc từ xe cộ",
            "SO2_column": "Sulfur Dioxide column (DU) - Khí độc từ đốt than, dầu",
            "CO": "Carbon Monoxide (ppbv) - Khí độc không màu, không mùi",
            "O3": "Ozone column (DU) - Ozone tầng cao, có thể gây hại ở mặt đất"
        },
        "🌤️ Weather Features (Thời tiết)": {
            "temperature_max": "Nhiệt độ tối đa (°C)",
            "temperature_min": "Nhiệt độ tối thiểu (°C)",
            "humidity": "Độ ẩm (%)",
            "dew_point": "Điểm sương (°C)",
            "wind_v10m": "Tốc độ gió (m/s)",
            "wind_direction": "Hướng gió (degrees)",
            "precipitation": "Lượng mưa (mm)"
        },
        "🔍 Derived Features": {
            "AOD_550": "Aerosol Optical Depth - Độ mờ của khí quyển",
            "SO2_surface": "SO2 ở mặt đất",
            "NO2_total": "NO2 tổng"
        },
        "⏱️ Lag Features (Giá trị quá khứ)": {
            "PM2_5_lag1": "PM2.5 của 1 ngày trước",
            "PM2_5_lag7": "PM2.5 của 7 ngày trước",
            "PM10_lag1": "PM10 của 1 ngày trước",
            "PM10_lag7": "PM10 của 7 ngày trước",
            "...": "... (tương tự cho các pollutants khác)"
        },
        "📊 Rolling Statistics (Thống kê trượt)": {
            "PM2_5_rolling7_mean": "Trung bình PM2.5 trong 7 ngày",
            "PM2_5_rolling14_mean": "Trung bình PM2.5 trong 14 ngày",
            "PM2_5_rolling7_std": "Độ lệch chuẩn PM2.5 trong 7 ngày",
            "...": "... (tương tự cho các pollutants khác)"
        },
        "🎯 Target Variable": {
            "aqi": "Air Quality Index - Chỉ số chất lượng không khí (0-500)",
            "aqi_category": "Phân loại AQI (Good/Moderate/Unhealthy/...)",
            "pollution_level": "Mức độ ô nhiễm"
        }
    }
    
    for group_name, features in feature_groups.items():
        with st.expander(group_name):
            for feature, description in features.items():
                st.markdown(f"**{feature}**: {description}")
    

# ============================================================================
# PAGE 2: DATA ANALYSIS
# ============================================================================
def page_analysis():
    st.markdown('<h1 class="main-header">📈 Phân Tích Dữ Liệu</h1>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải dữ liệu ở trang Overview trước")
        return
    
    df = st.session_state.data
    
    # Import và chạy analyzer
    if st.session_state.analyzer is None:
        try:
            from data_analysis import AQIDataAnalyzer
            # Tạo analyzer với data trực tiếp (sẽ dùng đường dẫn mặc định từ processed/)
            analyzer = AQIDataAnalyzer()
            analyzer.data = df.copy()
            # Kiểm tra AQI đã có chưa
            if 'aqi' not in analyzer.data.columns:
                analyzer.calculate_aqi()
            else:
                # Đảm bảo có aqi_category
                if 'aqi_category' not in analyzer.data.columns:
                    analyzer.calculate_aqi()
            st.session_state.analyzer = analyzer
        except Exception as e:
            st.error(f"❌ Lỗi khi khởi tạo analyzer: {e}")
            import traceback
            st.error(traceback.format_exc())
            return
    
    analyzer = st.session_state.analyzer
    
    # Mô tả dataset
    st.markdown("### 📚 Mô Tả Dataset")
    st.info("""
    **Dataset này được tạo để XÂY DỰNG MÔ HÌNH DỰ ĐOÁN AQI (Air Quality Index)**
    
    - **Mục tiêu**: Dự đoán chất lượng không khí NGÀY MAI dựa trên dữ liệu hôm nay
    - **Giúp**: Người dân biết trước khi nào cần đeo khẩu trang, hạn chế ra ngoài
    - **Dữ liệu**: {:,} ngày từ {} đến {}
    - **Features**: {} features bao gồm Pollutants, Weather, Time, Lag, Rolling Statistics
    """.format(
        len(df),
        df['date'].min().date(),
        df['date'].max().date(),
        len(df.columns)
    ))
    
    # Hiển thị 5 câu truy vấn
    st.markdown("---")
    st.markdown("### 🔍 5 Câu Truy Vấn Quan Trọng")
    
    # Query 1: AQI Trend
    with st.expander("📈 Query 1: Xu hướng AQI theo thời gian", expanded=True):
        try:
            results = analyzer.query_1_aqi_trend_over_time()
            if results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Năm đầu", results['first_year'], f"AQI: {results['first_aqi']:.1f}")
                with col2:
                    st.metric("Năm cuối", results['last_year'], f"AQI: {results['last_aqi']:.1f}")
                with col3:
                    change = results['change']
                    change_pct = results['change_pct']
                    st.metric("Thay đổi", f"{change:+.1f} điểm", f"{change_pct:+.1f}%")
                
                # Vẽ biểu đồ xu hướng
                yearly_trend = results['yearly_trend']
                
                # Xử lý yearly_trend (dict với structure: {year: {'mean': ..., 'min': ..., 'max': ...}})
                years = []
                aqis = []
                
                for year_key, year_data in yearly_trend.items():
                    # year_key là số năm (int)
                    year = int(year_key)
                    
                    # year_data là dict với keys: 'mean', 'min', 'max', 'std'
                    if isinstance(year_data, dict):
                        # Lấy 'mean' từ dict
                        aqi_val = year_data.get('mean', np.nan)
                    else:
                        # Nếu không phải dict, lấy trực tiếp
                        aqi_val = year_data
                    
                    if not pd.isna(aqi_val) and aqi_val is not None:
                        years.append(year)
                        aqis.append(float(aqi_val))
                
                # Sắp xếp theo năm
                if years and aqis:
                    sorted_data = sorted(zip(years, aqis))
                    years, aqis = zip(*sorted_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(years),
                        y=list(aqis),
                        mode='lines+markers',
                        name='AQI trung bình',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8, color='#1f77b4')
                    ))
                    fig.update_layout(
                        title="Xu hướng AQI theo năm",
                        xaxis_title="Năm",
                        yaxis_title="AQI",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Không có dữ liệu để vẽ biểu đồ")
                
                st.info("💡 **Ý nghĩa**: Hiểu xu hướng giúp đánh giá hiệu quả các chính sách bảo vệ môi trường và dự đoán tương lai")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
    
    # Query 2: Most Important Pollutant
    with st.expander("🌫️ Query 2: Pollutant nào quan trọng nhất?", expanded=True):
        try:
            results = analyzer.query_2_most_important_pollutant()
            if results:
                pollutant_aqi_avg = results.get('pollutant_aqi_avg', {})
                sorted_pollutants = sorted(pollutant_aqi_avg.items(), key=lambda x: x[1], reverse=True)
                
                # Hiển thị bảng
                df_poll = pd.DataFrame(sorted_pollutants, columns=['Pollutant', 'AQI Trung Bình'])
                st.dataframe(df_poll, use_container_width=True)
                
                # Vẽ bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[p[0] for p in sorted_pollutants],
                    y=[p[1] for p in sorted_pollutants],
                    marker_color='#FF6B6B'
                ))
                fig.update_layout(
                    title="AQI trung bình từ từng Pollutant",
                    xaxis_title="Pollutant",
                    yaxis_title="AQI",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                most_important = results.get('most_important', 'N/A')
                st.info(f"💡 **Ý nghĩa**: {most_important} ảnh hưởng nhiều nhất đến AQI. Đây là chất ô nhiễm cần ưu tiên kiểm soát.")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
    
    # Query 3: Weather Impact
    with st.expander("🌤️ Query 3: Thời tiết ảnh hưởng đến AQI như thế nào?", expanded=True):
        try:
            results = analyzer.query_3_weather_impact_on_aqi()
            if results:
                correlations = results.get('correlations', {})
                if correlations:
                    weather_factors = [(k, v) for k, v in correlations.items()]
                    weather_factors.sort(key=lambda x: abs(x[1]), reverse=True)
                    df_weather = pd.DataFrame(weather_factors, columns=['Weather Factor', 'Correlation'])
                    st.dataframe(df_weather, use_container_width=True)
                    
                    # Vẽ bar chart
                    fig = go.Figure()
                    corr_values = [abs(w[1]) for w in weather_factors]
                    colors = ['#FF6B6B' if c > 0.3 else '#4ECDC4' for c in corr_values]
                    fig.add_trace(go.Bar(
                        x=[w[0] for w in weather_factors],
                        y=[abs(w[1]) for w in weather_factors],
                        marker_color=colors
                    ))
                    fig.update_layout(
                        title="Tương quan giữa Thời tiết và AQI",
                        xaxis_title="Weather Factor",
                        yaxis_title="|Correlation|",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
    
    # Query 4: Seasonal Patterns
    with st.expander("🍂 Query 4: Mùa nào ô nhiễm nhất?", expanded=True):
        try:
            results = analyzer.query_4_seasonal_patterns()
            if results:
                seasonal_stats = results.get('seasonal_stats', {})
                if seasonal_stats:
                    # seasonal_stats có thể là dict với MultiIndex keys
                    seasons = []
                    aqis = []
                    for key, value in seasonal_stats.items():
                        if isinstance(key, tuple):
                            season_name = key[0] if len(key) > 0 else str(key)
                        else:
                            season_name = str(key)
                        if isinstance(value, dict):
                            aqi_val = value.get('aqi_mean', value.get('avg_aqi', 0))
                        else:
                            aqi_val = value
                        seasons.append(season_name)
                        aqis.append(aqi_val)
                    
                    if seasons and aqis:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=seasons,
                            y=aqis,
                            marker_color='#95E1D3'
                        ))
                        fig.update_layout(
                            title="AQI trung bình theo mùa",
                            xaxis_title="Mùa",
                            yaxis_title="AQI",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Hiển thị worst và best season
                        worst_idx = np.argmax(aqis)
                        best_idx = np.argmin(aqis)
                        st.info(f"💡 **Mùa ô nhiễm nhất**: {seasons[worst_idx]} với AQI trung bình {aqis[worst_idx]:.1f}")
                        st.success(f"✅ **Mùa sạch nhất**: {seasons[best_idx]} với AQI trung bình {aqis[best_idx]:.1f}")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            import traceback
            st.error(traceback.format_exc())
    
    # Query 5: Pollutant Correlations
    with st.expander("🔗 Query 5: Tương quan giữa các Pollutants", expanded=True):
        try:
            results = analyzer.query_5_pollutant_correlations()
            if results:
                corr_dict = results.get('correlation_matrix', None)
                if corr_dict is not None:
                    # Convert dict to DataFrame
                    corr_matrix = pd.DataFrame(corr_dict)
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size":10}
                    ))
                    fig.update_layout(
                        title="Ma trận tương quan giữa các Pollutants",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Hiển thị các cặp tương quan mạnh
                high_corr_pairs = results.get('high_correlation_pairs', [])
                if high_corr_pairs:
                    st.markdown("**Các cặp tương quan mạnh (|r| > 0.5):**")
                    for p1, p2, corr in high_corr_pairs[:10]:
                        direction = "Cùng tăng" if corr > 0 else "Ngược chiều"
                        st.markdown(f"- **{p1}** ↔ **{p2}**: {corr:+.3f} ({direction})")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            import traceback
            st.error(traceback.format_exc())

# ============================================================================
# PAGE 3: VISUALIZATION
# ============================================================================
def page_visualization():
    st.markdown('<h1 class="main-header">📉 Biểu Đồ Trực Quan</h1>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải dữ liệu ở trang Overview trước")
        return
    
    df = st.session_state.data
    
    # Load các biểu đồ từ thư mục visualizations
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        st.warning("⚠️ Không tìm thấy thư mục visualizations")
        return
    
    # Danh sách biểu đồ
    charts = [
        ("1_aqi_trend_over_time.png", "📈 Biểu đồ 1: Xu hướng AQI theo thời gian", "Hiển thị xu hướng AQI từ 2004-2025"),
        ("2_pollutant_importance.png", "🌫️ Biểu đồ 2: Tầm quan trọng của Pollutants", "So sánh đóng góp của từng pollutant vào AQI"),
        ("3_seasonal_patterns.png", "🍂 Biểu đồ 3: Mẫu theo mùa", "AQI trung bình theo từng mùa"),
        ("4_weather_impact.png", "🌤️ Biểu đồ 4: Ảnh hưởng của thời tiết", "Mối quan hệ giữa thời tiết và AQI"),
        ("5_pollutant_correlations.png", "🔗 Biểu đồ 5: Tương quan Pollutants", "Ma trận tương quan giữa các pollutants"),
        ("6_pollutants_timeseries.png", "📊 Biểu đồ 6: Time Series Pollutants", "Diễn biến các pollutants theo thời gian"),
        ("7_aqi_distribution.png", "📊 Biểu đồ 7: Phân phối AQI", "Histogram phân phối giá trị AQI"),
        ("8_calendar_heatmap.png", "📅 Biểu đồ 8: Calendar Heatmap", "AQI theo tháng và năm"),
        ("9_pm25_vs_pm10.png", "🌫️ Biểu đồ 9: PM2.5 vs PM10", "So sánh PM2.5 và PM10"),
        ("10_lag_features.png", "⏱️ Biểu đồ 10: Lag Features", "Ảnh hưởng của giá trị quá khứ"),
        ("11_rolling_statistics.png", "📊 Biểu đồ 11: Rolling Statistics", "Thống kê trượt của pollutants"),
        ("12_monthly_pollutants.png", "📅 Biểu đồ 12: Pollutants theo tháng", "Giá trị trung bình pollutants theo tháng"),
        ("13_pollution_vs_normal_pie.png", "🥧 Biểu đồ 13: Ngày ô nhiễm vs Bình thường", "Tỷ lệ ngày ô nhiễm và bình thường")
    ]
    
    # Hiển thị từng biểu đồ
    for chart_file, chart_title, chart_desc in charts:
        chart_path = os.path.join(viz_dir, chart_file)
        if os.path.exists(chart_path):
            st.markdown(f"### {chart_title}")
            st.markdown(f"**Mô tả**: {chart_desc}")
            st.image(chart_path, use_container_width=True)
            st.markdown("---")
    
    # Thêm biểu đồ 3D nếu có thể
    if 'aqi' in df.columns and 'PM2_5' in df.columns and 'PM10' in df.columns:
        st.markdown("### 🌐 Biểu Đồ 3D: AQI vs PM2.5 vs PM10")
        st.markdown("**Mô tả**: Hiển thị mối quan hệ 3D giữa AQI, PM2.5 và PM10")
        
        # Sample data để vẽ nhanh hơn
        sample_df = df.sample(min(1000, len(df)))
        
        fig = go.Figure(data=go.Scatter3d(
            x=sample_df['PM2_5'],
            y=sample_df['PM10'],
            z=sample_df['aqi'],
            mode='markers',
            marker=dict(
                size=5,
                color=sample_df['aqi'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="AQI")
            ),
            text=sample_df['date'].dt.strftime('%Y-%m-%d'),
            hovertemplate='<b>Date:</b> %{text}<br>' +
                          'PM2.5: %{x:.1f}<br>' +
                          'PM10: %{y:.1f}<br>' +
                          'AQI: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='PM2.5 (μg/m³)',
                yaxis_title='PM10 (μg/m³)',
                zaxis_title='AQI'
            ),
            title="3D Scatter: AQI vs PM2.5 vs PM10",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Ý nghĩa**: Biểu đồ này giúp hiểu mối quan hệ phức tạp giữa các pollutants và AQI trong không gian 3 chiều")

# ============================================================================
# PAGE 4: PREDICTION
# ============================================================================
def page_prediction():
    st.markdown('<h1 class="main-header">🔮 Dự Đoán Chất Lượng Không Khí</h1>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải dữ liệu ở trang Overview trước")
        return
    
    df = st.session_state.data
    
    # Chọn horizon và model
    st.markdown("### ⚙️ Cấu Hình Dự Đoán")
    col1, col2 = st.columns(2)
    
    with col1:
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        max_predict_date = max_date + timedelta(days=365)
        
        selected_date = st.date_input(
            "Ngày dự đoán",
            value=max_date + timedelta(days=1),
            min_value=min_date,
            max_value=max_predict_date
        )
        
        # Tính số ngày cần dự đoán
        days_ahead = (selected_date - max_date).days if selected_date > max_date else 1
        
        # Chọn horizon gần nhất
        horizons = [1, 7, 30, 365]
        selected_horizon = min(horizons, key=lambda x: abs(x - days_ahead))
        if days_ahead > 365:
            selected_horizon = 365
        elif days_ahead <= 1:
            selected_horizon = 1
        elif days_ahead <= 7:
            selected_horizon = 7
        elif days_ahead <= 30:
            selected_horizon = 30
    
    with col2:
        model_idx = st.selectbox(
            "Chọn Model",
            ['model1', 'model2'],
            index=0,
            help="Model1: Model tốt nhất, Model2: Model tốt thứ 2"
        )
        
        # Load model
        model, scaler_X, metadata, scaler_y = load_model(horizon=selected_horizon, model_idx=model_idx)
        
        if model is None:
            st.error("❌ Không tìm thấy model. Vui lòng chạy Train_model.py trước")
            return
        
        # Hiển thị thông tin model (model ban đầu được chọn, có thể sẽ được thay đổi bởi logic tối ưu)
        if 'model_name' in metadata:
            model_name = metadata['model_name']
            if 'metrics' in metadata:
                if 'overall_r2_test' in metadata['metrics']:
                    r2_score = metadata['metrics']['overall_r2_test']
                elif 'aqi_r2_test' in metadata['metrics']:
                    r2_score = metadata['metrics']['aqi_r2_test']
                else:
                    r2_score = 0.0
            else:
                r2_score = 0.0
            st.info(f"📊 Model đã chọn: {model_name}\n📈 Weighted R²: {r2_score:.3f}\n🎯 Horizon: T+{selected_horizon}\n💡 (Có thể tự động chuyển sang model tối ưu hơn)")
        else:
            st.info(f"📊 Model đã chọn: {metadata.get('model_name', 'Unknown')}\n🎯 Horizon: T+{selected_horizon}\n💡 (Có thể tự động chuyển sang model tối ưu hơn)")
    
    # Xác định base_date và forecast_date
    # Người dùng muốn dự đoán từ 5/11/2025 đến 31/12/2025
    target_base_date = pd.Timestamp('2025-11-05').date()
    target_end_date = pd.Timestamp('2025-12-31').date()
    
    if selected_date <= max_date:
        # Nếu chọn ngày trong quá khứ
        base_date = selected_date - timedelta(days=1)
        forecast_date = selected_date
        days_ahead = 1
        use_base_date = base_date
    else:
        # Nếu chọn ngày tương lai
        forecast_date = selected_date
        
        # Kiểm tra xem có phải trong khoảng 5/11 - 31/12/2025 không
        if selected_date >= target_base_date and selected_date <= target_end_date:
            # Dùng ngày 5/11/2025 làm base (hoặc ngày cuối cùng có dữ liệu nếu 5/11 chưa đến)
            if target_base_date <= max_date:
                # Nếu 5/11/2025 đã có trong dữ liệu
                use_base_date = target_base_date
            else:
                # Nếu 5/11/2025 chưa đến, dùng ngày cuối cùng có dữ liệu
                use_base_date = max_date
                st.info(f"ℹ️ Dùng ngày {use_base_date.strftime('%d/%m/%Y')} (ngày cuối cùng có dữ liệu) làm base")
            
            days_ahead = (forecast_date - use_base_date).days
            if days_ahead > 0:
                st.success(f"✅ Có thể dự đoán: Từ {use_base_date.strftime('%d/%m/%Y')} đến {forecast_date.strftime('%d/%m/%Y')} ({days_ahead} ngày)")
        else:
            # Dùng ngày cuối cùng có dữ liệu
            use_base_date = max_date
            days_ahead = (forecast_date - use_base_date).days
    
    # Kiểm tra dữ liệu base
    base_date_str = pd.Timestamp(use_base_date)
    
    if base_date_str not in df['date'].values:
        st.error(f"❌ Không có dữ liệu cho ngày {use_base_date}. Vui lòng chọn ngày khác.")
        return
    
    # Lấy dữ liệu ngày base
    base_data = df[df['date'] == base_date_str].iloc[0]
    
    # Đảm bảo season là số (0, 1, 2, 3) thay vì string (nếu có)
    if 'season' in base_data:
        season_val = base_data['season']
        if isinstance(season_val, str):
            # Chuyển string sang số
            season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3, 'Autumn': 3}
            base_data['season'] = season_map.get(season_val, 0)
        elif season_val not in [0, 1, 2, 3]:
            # Nếu không phải string và không phải 0-3, tính từ month
            month = base_data.get('month', pd.Timestamp(base_data['date']).month if 'date' in base_data else 1)
            if month in [12, 1, 2]:
                base_data['season'] = 0
            elif month in [3, 4, 5]:
                base_data['season'] = 1
            elif month in [6, 7, 8]:
                base_data['season'] = 2
            else:
                base_data['season'] = 3
    
    # Chuẩn bị features
    if 'feature_names' in metadata:
        feature_cols = metadata['feature_names']
    else:
        exclude_features = ['date', 'aqi', 'aqi_category', 'pollution_level']
        feature_cols = [col for col in df.columns if col not in exclude_features]
    
    # Quyết định dùng recursive prediction hay direct prediction
    # Logic tối ưu: Ưu tiên dùng model có horizon phù hợp nhất
    use_recursive = False
    optimal_horizon = None
    
    if days_ahead == 1:
        # Dự đoán 1 ngày: dùng T+1 (direct)
        optimal_horizon = 1
        use_recursive = False
        st.info(f"✅ Dùng Direct Prediction (T+1 model) cho 1 ngày")
    elif days_ahead in [7, 30, 365]:
        # Dự đoán chính xác 7, 30, hoặc 365 ngày: dùng model tương ứng (direct)
        optimal_horizon = days_ahead
        use_recursive = False
        st.info(f"✅ Dùng Direct Prediction (T+{days_ahead} model) cho {days_ahead} ngày - Độ chính xác cao hơn, không tích lũy sai số")
    elif days_ahead > 365:
        # Nếu > 365 ngày: dùng T+365 (best long-term model)
        optimal_horizon = 365
        use_recursive = False
        st.warning(f"⚠️ Dự đoán {days_ahead} ngày > 365. Dùng model T+365 (best available) thay vì recursive")
    elif days_ahead > 30:
        # Nếu 30 < days_ahead <= 365: dùng T+30 hoặc T+365 tùy gần hơn
        if days_ahead <= 60:
            optimal_horizon = 30
            st.info(f"✅ Dự đoán {days_ahead} ngày gần với T+30. Dùng Direct Prediction (T+30 model)")
        else:
            optimal_horizon = 365
            st.info(f"✅ Dự đoán {days_ahead} ngày gần với T+365. Dùng Direct Prediction (T+365 model)")
        use_recursive = False
    elif days_ahead > 7:
        # Nếu 7 < days_ahead <= 30: dùng T+7 hoặc T+30 tùy gần hơn
        if days_ahead <= 14:
            optimal_horizon = 7
            st.info(f"✅ Dự đoán {days_ahead} ngày gần với T+7. Dùng Direct Prediction (T+7 model)")
        else:
            optimal_horizon = 30
            st.info(f"✅ Dự đoán {days_ahead} ngày gần với T+30. Dùng Direct Prediction (T+30 model)")
        use_recursive = False
    else:
        # Nếu 1 < days_ahead <= 7: dùng T+7 hoặc recursive tùy gần hơn
        if days_ahead <= 3:
            # Dùng recursive T+1 cho 2-3 ngày (chính xác hơn)
            optimal_horizon = 1
            use_recursive = True
            st.info(f"🔄 Dự đoán {days_ahead} ngày: Dùng Recursive Prediction (T+1 model) - Chính xác hơn cho ngắn hạn")
        else:
            # Dùng T+7 cho 4-7 ngày
            optimal_horizon = 7
            use_recursive = False
            st.info(f"✅ Dự đoán {days_ahead} ngày gần với T+7. Dùng Direct Prediction (T+7 model)")
    
    # Load model tối ưu nếu khác với model đã chọn
    if optimal_horizon is not None and optimal_horizon != selected_horizon:
        optimal_model, optimal_scaler_X, optimal_metadata, optimal_scaler_y = load_model(
            horizon=optimal_horizon, model_idx='model1'
        )
        if optimal_model is not None:
            model = optimal_model
            scaler_X = optimal_scaler_X
            scaler_y = optimal_scaler_y
            metadata = optimal_metadata
            selected_horizon = optimal_horizon  # Cập nhật selected_horizon để dùng cho phần sau
            st.success(f"🔄 Đã tự động chuyển sang model T+{optimal_horizon} (tối ưu cho {days_ahead} ngày)")
        else:
            st.warning(f"⚠️ Không tìm thấy model T+{optimal_horizon}, dùng model đã chọn (T+{selected_horizon})")
    elif optimal_horizon is None:
        # Nếu không có optimal_horizon (days_ahead = 1 và không cần recursive)
        optimal_horizon = selected_horizon
    
    # Nếu dùng recursive, đảm bảo có model T+1
    if use_recursive:
        model_t1, scaler_X_t1, metadata_t1, scaler_y_t1 = load_model(horizon=1, model_idx='model1')
        if model_t1 is not None:
            model = model_t1
            scaler_X = scaler_X_t1
            scaler_y = scaler_y_t1
            metadata = metadata_t1
        else:
            st.warning("⚠️ Không tìm thấy model T+1 cho recursive, dùng model hiện tại")
            use_recursive = False
    
    # Dự đoán
    if use_recursive and days_ahead > 1:
        # Recursive prediction
        predictions_dict = predict_recursive(
            model, scaler_X, metadata, base_data, df, feature_cols, days_ahead, scaler_y
        )
    else:
        # Direct prediction
        X_input = base_data[feature_cols].values.reshape(1, -1)
        
        # Scale features (tất cả models dùng scaled data trong version mới)
        model_name = metadata.get('model_name', '')
        if scaler_X is not None:
            X_input_scaled = scaler_X.transform(X_input)
            predictions_scaled = model.predict(X_input_scaled)[0]
            
            # Inverse transform predictions nếu có scaler_y
            if scaler_y is not None and selected_horizon in scaler_y:
                targets = metadata.get('targets', ['aqi'])
                predictions = np.zeros(len(targets))
                for idx, target in enumerate(targets):
                    if target in scaler_y[selected_horizon]:
                        predictions[idx] = scaler_y[selected_horizon][target].inverse_transform(
                            predictions_scaled[idx].reshape(-1, 1)
                        )[0, 0]
                    else:
                        predictions[idx] = predictions_scaled[idx]
            else:
                predictions = predictions_scaled
        else:
            predictions = model.predict(X_input)[0]
        
        # Xử lý predictions (có thể là array hoặc scalar)
        if isinstance(predictions, np.ndarray):
            # Multi-target prediction
            targets = metadata.get('targets', ['aqi'])
            predictions_dict = {}
            for idx, target in enumerate(targets):
                if idx < len(predictions):
                    predictions_dict[target] = predictions[idx]
            
            # Nếu không có 'aqi' trong dict, lấy giá trị đầu tiên
            if 'aqi' not in predictions_dict and len(predictions) > 0:
                predictions_dict['aqi'] = predictions[0]
        else:
            # Single target (backward compatibility)
            predictions_dict = {'aqi': predictions}
        
        # Giới hạn AQI trong range hợp lý
        if 'aqi' in predictions_dict:
            predictions_dict['aqi'] = max(0, min(500, predictions_dict['aqi']))
    
    # Lấy predictions từ model (hoặc fallback từ base_data)
    aqi_pred = predictions_dict.get('aqi', base_data.get('aqi', np.nan))
    
    # Convert nhiệt độ từ predictions hoặc base_data
    def convert_temp(temp_raw):
        if pd.isna(temp_raw):
            return np.nan
        if temp_raw > 50:  # Kelvin
            return temp_raw - 273.15
        return temp_raw
    
    temp_pred_max = predictions_dict.get('temperature_max', base_data.get('temperature_max', np.nan))
    temp_pred_min = predictions_dict.get('temperature_min', base_data.get('temperature_min', np.nan))
    temp_pred_mean = predictions_dict.get('temperature_mean', base_data.get('temperature_mean', np.nan))
    
    temp_pred_max = convert_temp(temp_pred_max)
    temp_pred_min = convert_temp(temp_pred_min)
    temp_pred_mean = convert_temp(temp_pred_mean)
    
    # Lấy pollutants từ predictions hoặc base_data
    pm25_pred = predictions_dict.get('PM2_5', base_data.get('PM2_5', np.nan))
    pm10_pred = predictions_dict.get('PM10', base_data.get('PM10', np.nan))
    no2_pred = predictions_dict.get('NO2_trop', base_data.get('NO2_trop', np.nan))
    co_pred = predictions_dict.get('CO', base_data.get('CO', np.nan))
    o3_pred = predictions_dict.get('O3', base_data.get('O3', np.nan))
    so2_pred = predictions_dict.get('SO2_column', base_data.get('SO2_column', np.nan))
    humidity_pred = predictions_dict.get('humidity', base_data.get('humidity', np.nan))
    wind_pred = predictions_dict.get('wind_speed', base_data.get('wind_speed', np.nan))
    
    # Hiển thị kết quả
    st.markdown("---")
    st.markdown("### 📊 Kết Quả Dự Đoán")
    
    category, bg_color, text_color = categorize_aqi(aqi_pred)
    
    # Hiển thị AQI
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ngày dự đoán", forecast_date.strftime("%d/%m/%Y"))
        st.caption(f"Dự đoán cho {days_ahead} ngày tương lai từ ngày {use_base_date.strftime('%d/%m/%Y')}")
        if use_recursive and days_ahead > 1:
            st.caption("🔄 Sử dụng Recursive Prediction (T+1 model) - Dự đoán từng ngày tuần tự")
        else:
            st.caption(f"🎯 Direct Prediction - Model T+{selected_horizon} (dự đoán trực tiếp {selected_horizon} ngày)")
    with col2:
        # Hiển thị AQI với màu nền theo AQI (không dùng màu trắng)
        st.markdown(f'<div style="background-color: {bg_color}; padding: 1.5rem; border-radius: 0.5rem; text-align: center;"><h1 style="color: {text_color}; margin: 0; font-size: 3rem;">AQI: {aqi_pred:.1f}</h1><p style="color: {text_color}; margin: 0.5rem 0 0 0; font-size: 1.2rem;">{category}</p></div>', unsafe_allow_html=True)
    with col3:
        st.metric("Phân loại", category)
    
    # Hiển thị dự đoán nhiệt độ và chất lượng không khí
    st.markdown("---")
    st.markdown("### 🌡️ Dự Đoán Nhiệt Độ & Chất Lượng Không Khí")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if not pd.isna(temp_pred_max):
            st.metric("🌡️ Nhiệt độ cao nhất", f"{temp_pred_max:.1f}°C")
        else:
            st.metric("🌡️ Nhiệt độ cao nhất", "N/A")
    with col2:
        if not pd.isna(temp_pred_min):
            st.metric("🌡️ Nhiệt độ thấp nhất", f"{temp_pred_min:.1f}°C")
        else:
            st.metric("🌡️ Nhiệt độ thấp nhất", "N/A")
    with col3:
        if not pd.isna(humidity_pred):
            st.metric("💧 Độ ẩm", f"{humidity_pred:.1f}%")
        else:
            st.metric("💧 Độ ẩm", "N/A")
    with col4:
        if not pd.isna(wind_pred):
            st.metric("💨 Tốc độ gió", f"{wind_pred:.1f} m/s")
        else:
            st.metric("💨 Tốc độ gió", "N/A")
    
    # Hiển thị các pollutants
    st.markdown("#### 🌫️ Dự Đoán Các Chất Ô Nhiễm")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**PM2.5 (Bụi mịn)**: " + (f"{pm25_pred:.2f} μg/m³" if not pd.isna(pm25_pred) else "N/A"))
        st.markdown("**PM10 (Bụi)**: " + (f"{pm10_pred:.2f} μg/m³" if not pd.isna(pm10_pred) else "N/A"))
    with col2:
        st.markdown("**NO₂ (Nitrogen Dioxide)**: " + (f"{no2_pred:.2e} molecules/cm²" if not pd.isna(no2_pred) else "N/A"))
        st.markdown("**CO (Carbon Monoxide)**: " + (f"{co_pred:.2f} ppbv" if not pd.isna(co_pred) else "N/A"))
    with col3:
        st.markdown("**O₃ (Ozone)**: " + (f"{o3_pred:.2f} DU" if not pd.isna(o3_pred) else "N/A"))
        st.markdown("**SO₂ (Sulfur Dioxide)**: " + (f"{so2_pred:.2f} DU" if not pd.isna(so2_pred) else "N/A"))
    
    # Khuyến nghị
    st.markdown("---")
    st.markdown("### 💡 Khuyến Nghị")
    
    # Thử dùng AI recommendations
    ai_rec = get_ai_recommendations(aqi_pred, base_data, forecast_date, predictions_dict)
    if ai_rec is not None:
        st.info("🤖 Khuyến nghị được tạo bởi AI (Gemini)")
        recommendations = ai_rec
    else:
        recommendations = get_recommendations(aqi_pred, base_data, forecast_date, predictions_dict)
    
    st.markdown(f"#### {recommendations['title']}")
    
    st.markdown("**Hành động nên làm:**")
    for action in recommendations['actions']:
        st.markdown(f"- {action}")
    
    st.markdown(f"**Sức khỏe**: {recommendations['health']}")
    
    # Hiển thị môi trường tương ứng theo chất lượng không khí
    st.markdown("---")
    st.markdown("### 🌳 Môi Trường Tương Ứng")
    
    if aqi_pred <= 50:  # Tốt - Không khí trong lành
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #00E400 0%, #00FF7F 100%); border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color: white; font-size: 5rem; margin: 0;">🌳🌬️🌈</h1>
            <h2 style="color: white; margin: 1rem 0; font-size: 2rem;">Không khí trong lành</h2>
            <p style="color: white; font-size: 1.3rem; margin: 0.5rem 0;">Bầu trời trong xanh, không khí sạch sẽ</p>
            <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">An toàn tuyệt đối cho mọi hoạt động ngoài trời</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif aqi_pred <= 100:  # Bình thường
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color: white; font-size: 5rem; margin: 0;">☀️🌧️</h1>
            <h2 style="color: white; margin: 1rem 0; font-size: 2rem;">Không khí bình thường</h2>
            <p style="color: white; font-size: 1.3rem; margin: 0.5rem 0;">Thời tiết ổn định, nắng mưa nhẹ</p>
            <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">Phù hợp cho hầu hết các hoạt động ngoài trời</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:  # Ô nhiễm - Khói bụi, mù trời
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #8B0000 0%, #FF0000 100%); border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color: white; font-size: 5rem; margin: 0;">🌫️💨</h1>
            <h2 style="color: white; margin: 1rem 0; font-size: 2rem;">Không khí ô nhiễm</h2>
            <p style="color: white; font-size: 1.3rem; margin: 0.5rem 0;">Bụi mịn rất nhiều, không khí ô nhiễm nặng</p>
            <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">Cần hạn chế ra ngoài</p>
        </div>
        """, unsafe_allow_html=True)
    
    # So sánh với AQI thực tế (nếu có)
    if forecast_date in df['date'].dt.date.values:
        actual_date_str = pd.Timestamp(forecast_date)
        actual_aqi = df[df['date'] == actual_date_str]['aqi'].iloc[0]
        
        st.markdown("---")
        st.markdown("### 📊 So Sánh với Giá Trị Thực Tế")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dự đoán", f"{aqi_pred:.1f}", f"{aqi_pred - actual_aqi:+.1f}")
        with col2:
            st.metric("Thực tế", f"{actual_aqi:.1f}")
        
        # Vẽ biểu đồ so sánh
        fig = go.Figure()
        _, actual_bg_color, _ = categorize_aqi(actual_aqi)
        fig.add_trace(go.Bar(
            x=['Dự đoán', 'Thực tế'],
            y=[aqi_pred, actual_aqi],
            marker_color=[bg_color, actual_bg_color],
            text=[f"{aqi_pred:.1f}", f"{actual_aqi:.1f}"],
            textposition='auto'
        ))
        fig.update_layout(
            title="So sánh Dự đoán vs Thực tế",
            yaxis_title="AQI",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tính sai số
        error = abs(aqi_pred - actual_aqi)
        error_pct = (error / actual_aqi) * 100
        st.info(f"📊 Sai số: {error:.1f} điểm AQI ({error_pct:.1f}%)")
    else:
        st.info(f"💡 Dự đoán cho ngày {forecast_date.strftime('%d/%m/%Y')} - Không có dữ liệu thực tế để so sánh")

# ============================================================================
# CHATBOT COMPONENT
# ============================================================================
def chatbot_component():
    """Component chatbot ở góc phải"""
    if st.session_state.chatbot is None:
        return
    
    # Sidebar cho chatbot
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 AI Chatbot")
        
        # Hiển thị lịch sử chat
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history[-5:]:  # Chỉ hiển thị 5 tin nhắn gần nhất
                with st.chat_message("user"):
                    st.write(chat['user'])
                with st.chat_message("assistant"):
                    st.write(chat['assistant'])
        
        # Input cho câu hỏi
        user_question = st.text_input("Hỏi về AQI và sức khỏe...")
        
        if st.button("Gửi") and user_question:
            try:
                with st.spinner("Đang suy nghĩ..."):
                    response = st.session_state.chatbot.generate_response(user_question)
                    st.session_state.chat_history.append({
                        'user': user_question,
                        'assistant': response
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
        
        # Câu hỏi gợi ý
        if st.session_state.chatbot:
            suggested = st.session_state.chatbot.get_suggested_questions()
            st.markdown("**Câu hỏi gợi ý:**")
            for q in suggested[:5]:
                if st.button(q, key=f"suggest_{q}"):
                    try:
                        with st.spinner("Đang suy nghĩ..."):
                            response = st.session_state.chatbot.generate_response(q)
                            st.session_state.chat_history.append({
                                'user': q,
                                'assistant': response
                            })
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Lỗi: {e}")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Determine current page
    if page == "📊 Overview":
        page_overview()
    elif page == "📈 Phân Tích Dữ Liệu":
        page_analysis()
    elif page == "📉 Biểu Đồ":
        page_visualization()
    elif page == "🔮 Dự Đoán":
        page_prediction()
    
    # Hiển thị chatbot ở sidebar
    chatbot_component()

if __name__ == "__main__":
    main()


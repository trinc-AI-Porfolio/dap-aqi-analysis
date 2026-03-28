#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script phân tích dữ liệu AQI đã được tiền xử lý (2004-2025)
Phân tích dữ liệu thực từ NASA Giovanni
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import json
import warnings
import sys
import codecs
import os

# Fix encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

warnings.filterwarnings('ignore')

class AQIDataAnalyzer:
    def __init__(self, csv_path=None, db_path="aqi_analysis.db"):
        # Đặt đường dẫn mặc định từ thư mục processed
        if csv_path is None:
            csv_path = os.path.join("processed", "processed_aqi_dataset.csv")
        
        self.csv_path = csv_path
        self.db_path = db_path
        self.data = None
        self.init_database()
        
    def load_data(self):
        """Tải dữ liệu từ CSV đã xử lý"""
        print("📊 ĐANG TẢI DỮ LIỆU...")
        print(f"   📂 Đường dẫn: {self.csv_path}")
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(self.csv_path):
            print(f"❌ Không tìm thấy file: {self.csv_path}")
            print(f"💡 Vui lòng chạy tien_xu_ly_du_lieu.py trước để tạo file dữ liệu đã xử lý")
            return False
        
        try:
            self.data = pd.read_csv(self.csv_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            # Tính AQI từ các features có sẵn
            self.calculate_aqi()
            
            print(f"✅ Đã tải {len(self.data):,} records")
            print(f"   Date range: {self.data['date'].min().date()} đến {self.data['date'].max().date()}")
            print(f"   Features: {len(self.data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_aqi(self):
        """Kiểm tra AQI - AQI đã được tính ở bước tiền xử lý"""
        # Kiểm tra xem đã có cột AQI chưa
        if 'aqi' not in self.data.columns:
            raise ValueError("❌ LỖI: Không tìm thấy cột 'aqi' trong dataset. "
                           "Vui lòng chạy tien_xu_ly_du_lieu.py trước để tính AQI.")
        
        print("✅ AQI đã có sẵn trong dataset (từ bước tiền xử lý)")
        
        # Kiểm tra xem có aqi_category chưa
        if 'aqi_category' not in self.data.columns:
            # Tính lại category nếu chưa có
            def categorize_aqi(aqi):
                if pd.isna(aqi):
                    return 'Unknown'
                elif aqi <= 50:
                    return 'Good'
                elif aqi <= 100:
                    return 'Moderate'
                elif aqi <= 150:
                    return 'Unhealthy for Sensitive'
                elif aqi <= 200:
                    return 'Unhealthy'
                elif aqi <= 300:
                    return 'Very Unhealthy'
                else:
                    return 'Hazardous'
            self.data['aqi_category'] = self.data['aqi'].apply(categorize_aqi)
        
        print(f"   AQI: Min={self.data['aqi'].min():.1f}, Max={self.data['aqi'].max():.1f}, Mean={self.data['aqi'].mean():.1f}")
    
    def init_database(self):
        """Khởi tạo database SQLite"""
        print("🗄️ KHỞI TẠO DATABASE...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tạo bảng AQI data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aqi_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    year INTEGER,
                    month INTEGER,
                    day INTEGER,
                    weekday INTEGER,
                    day_of_year INTEGER,
                    is_weekend INTEGER,
                    season INTEGER,
                    aqi REAL,
                    aod_550 REAL,
                    pm2_5 REAL,
                    pm10 REAL,
                    no2_trop REAL,
                    no2_total REAL,
                    so2_column REAL,
                    so2_surface REAL,
                    o3 REAL,
                    co REAL,
                    temperature_max REAL,
                    temperature_mean REAL,
                    temperature_min REAL,
                    skin_temperature REAL,
                    dew_point REAL,
                    humidity REAL,
                    wind_u10m REAL,
                    wind_v10m REAL,
                    wind_speed REAL,
                    wind_direction REAL,
                    aqi_category TEXT,
                    pollution_level TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tạo bảng Analysis results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    results_json TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tạo bảng Trends
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER,
                    aqi_avg REAL,
                    aqi_max REAL,
                    aqi_min REAL,
                    pm2_5_avg REAL,
                    pm10_avg REAL,
                    temperature_avg REAL,
                    humidity_avg REAL,
                    wind_speed_avg REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database đã được khởi tạo thành công")
            
        except Exception as e:
            print(f"❌ Lỗi khi khởi tạo database: {e}")
    
    def save_to_database(self):
        """Lưu dữ liệu vào database"""
        print("💾 LƯU DỮ LIỆU VÀO DATABASE...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Chuẩn bị dữ liệu
            db_data = self.data.copy()
            
            # Lưu vào database
            db_data.to_sql('aqi_data', conn, if_exists='replace', index=False)
            print(f"✅ Đã lưu {len(db_data):,} records vào database")
            
            conn.close()
            print("✅ Hoàn thành lưu dữ liệu vào database")
            
        except Exception as e:
            print(f"❌ Lỗi khi lưu vào database: {e}")
            import traceback
            traceback.print_exc()
    
    def _flatten_multiindex_dict(self, df_dict):
        """Chuyển DataFrame MultiIndex columns thành flat dict"""
        if isinstance(df_dict, dict):
            result = {}
            for key, value in df_dict.items():
                if isinstance(key, tuple):
                    # MultiIndex key: chuyển thành string
                    flat_key = '_'.join(str(k) for k in key)
                    result[flat_key] = self._flatten_multiindex_dict(value)
                elif isinstance(value, dict):
                    result[key] = self._flatten_multiindex_dict(value)
                elif isinstance(value, (pd.Series, np.ndarray, list)):
                    # Nếu là Series/array/list, convert thành dict hoặc list
                    if isinstance(value, pd.Series):
                        result[key] = {str(k): float(v) if not pd.isna(v) else None 
                                     for k, v in value.items()}
                    else:
                        result[key] = [float(v) if isinstance(v, (int, float)) and not pd.isna(v) else None 
                                     for v in value]
                else:
                    # Chuyển numpy types sang Python types
                    try:
                        if pd.isna(value):
                            result[key] = None
                        elif isinstance(value, (np.integer, np.int64, np.int32)):
                            result[key] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            result[key] = float(value)
                        else:
                            result[key] = value
                    except (ValueError, TypeError):
                        # Nếu không thể xử lý, convert thành string
                        result[key] = str(value)
            return result
        return df_dict
    
    def save_analysis_results(self, analysis_type, results, summary=""):
        """Lưu kết quả phân tích vào database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Flatten MultiIndex dicts trước khi JSON serialize
            flattened_results = self._flatten_multiindex_dict(results)
            results_json = json.dumps(flattened_results, ensure_ascii=False, default=str)
            
            cursor.execute('''
                INSERT INTO analysis_results (analysis_type, analysis_date, results_json, summary)
                VALUES (?, ?, ?, ?)
            ''', (analysis_type, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), results_json, summary))
            
            conn.commit()
            conn.close()
            print(f"✅ Đã lưu kết quả phân tích: {analysis_type}")
            
        except Exception as e:
            print(f"❌ Lỗi khi lưu kết quả phân tích: {e}")
            import traceback
            traceback.print_exc()
    
    def save_trends_to_database(self):
        """Lưu xu hướng theo năm vào database"""
        print("📈 LƯU XU HƯỚNG VÀO DATABASE...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM trends')
            
            # Tính toán xu hướng theo năm
            yearly_stats = self.data.groupby('year').agg({
                'aqi': ['mean', 'max', 'min'],
                'PM2_5': 'mean',
                'PM10': 'mean',
                'temperature_mean': 'mean',
                'humidity': 'mean',
                'wind_speed': 'mean'
            }).round(2)
            
            yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns.values]
            
            for year, row in yearly_stats.iterrows():
                cursor.execute('''
                    INSERT INTO trends (year, aqi_avg, aqi_max, aqi_min, pm2_5_avg, pm10_avg, 
                                     temperature_avg, humidity_avg, wind_speed_avg)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(year),
                    float(row['aqi_mean']),
                    float(row['aqi_max']),
                    float(row['aqi_min']),
                    float(row['PM2_5_mean']),
                    float(row['PM10_mean']),
                    float(row['temperature_mean_mean']),
                    float(row['humidity_mean']),
                    float(row['wind_speed_mean'])
                ))
            
            conn.commit()
            conn.close()
            print(f"✅ Đã lưu xu hướng {len(yearly_stats)} năm vào database")
            
        except Exception as e:
            print(f"❌ Lỗi khi lưu xu hướng: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_summary_report(self):
        """Tạo báo cáo tổng hợp ngắn gọn"""
        print("\n" + "="*80)
        print("📋 BÁO CÁO TỔNG HỢP")
        print("="*80)
        
        if self.data is None:
            print("❌ Chưa có dữ liệu")
            return
        
        print(f"\n📊 THỐNG KÊ CƠ BẢN:")
        print(f"   - Tổng số records: {len(self.data):,} ngày")
        print(f"   - Thời gian: {self.data['date'].min().date()} đến {self.data['date'].max().date()}")
        print(f"   - AQI trung bình: {self.data['aqi'].mean():.1f} (Min: {self.data['aqi'].min():.1f}, Max: {self.data['aqi'].max():.1f})")
        
        # Phân loại AQI
        aqi_counts = self.data['aqi_category'].value_counts()
        print(f"\n   Phân loại AQI:")
        for category, count in aqi_counts.head(3).items():
            percentage = (count / len(self.data)) * 100
            print(f"     - {category}: {count:,} ngày ({percentage:.1f}%)")
        
        print(f"\n✅ KẾT LUẬN:")
        print(f"   ✅ Dữ liệu đã được tiền xử lý hoàn chỉnh")
        print(f"   ✅ Sẵn sàng cho Data Visualization và ML Model")

    def describe_dataset(self):
        """Mô tả chi tiết về dataset để người dùng hiểu rõ dữ liệu có gì"""
        print("\n" + "="*80)
        print("📚 MÔ TẢ DATASET - BẠN CÓ GÌ TRONG DỮ LIỆU?")
        print("="*80)
        
        if self.data is None:
            print("❌ Chưa có dữ liệu")
            return
        
        print("\n🎯 MỤC ĐÍCH CỦA DATASET:")
        print("   Dataset này được tạo để XÂY DỰNG MÔ HÌNH DỰ ĐOÁN AQI (Air Quality Index)")
        print("   → Dự đoán chất lượng không khí NGÀY MAI dựa trên dữ liệu hôm nay")
        print("   → Giúp người dân biết trước khi nào cần đeo khẩu trang, hạn chế ra ngoài")
        
        print("\n📊 CẤU TRÚC DỮ LIỆU:")
        print(f"   - Tổng số ngày: {len(self.data):,} ngày")
        print(f"   - Thời gian: {self.data['date'].min().date()} đến {self.data['date'].max().date()}")
        print(f"   - Khoảng: {self.data['year'].max() - self.data['year'].min() + 1} năm")
        print(f"   - Tổng số features: {len(self.data.columns)} features")
        
        print("\n🔬 CÁC NHÓM DỮ LIỆU CHÍNH:")
        
        # 1. Pollutants (chất ô nhiễm)
        pollutants = ['PM2_5', 'PM10', 'NO2_trop', 'NO2_total', 'SO2_column', 'SO2_surface', 'O3', 'CO', 'AOD_550']
        available_pollutants = [p for p in pollutants if p in self.data.columns]
        print(f"\n   1️⃣ POLLUTANTS (Chất ô nhiễm) - {len(available_pollutants)} features:")
        print("      → Đây là các chất gây ô nhiễm không khí")
        for pol in available_pollutants:
            missing = self.data[pol].isnull().sum()
            missing_pct = (missing / len(self.data)) * 100
            if missing == 0:
                print(f"        ✅ {pol}: Đầy đủ ({self.data[pol].mean():.2f} trung bình)")
            else:
                print(f"        ⚠️  {pol}: {missing_pct:.1f}% missing")
        print("      💡 Ý nghĩa: Dùng để TÍNH AQI và DỰ ĐOÁN AQI tương lai")
        
        # 2. Weather (thời tiết)
        weather = ['temperature_max', 'temperature_mean', 'temperature_min', 'skin_temperature', 
                   'dew_point', 'humidity', 'wind_u10m', 'wind_v10m', 'wind_speed', 
                   'wind_direction', 'precipitation']
        available_weather = [w for w in weather if w in self.data.columns]
        print(f"\n   2️⃣ WEATHER (Thời tiết) - {len(available_weather)} features:")
        print("      → Nhiệt độ, độ ẩm, gió, mưa")
        for w in available_weather[:5]:  # Show first 5
            print(f"        ✅ {w}")
        if len(available_weather) > 5:
            print(f"        ... và {len(available_weather) - 5} features khác")
        print("      💡 Ý nghĩa: Thời tiết ẢNH HƯỞNG đến AQI (gió làm phân tán, mưa làm sạch)")
        
        # 3. Time features
        time_features = ['year', 'month', 'day', 'weekday', 'day_of_year', 'is_weekend', 'season']
        print(f"\n   3️⃣ TIME FEATURES (Thời gian) - {len(time_features)} features:")
        print("      → Năm, tháng, ngày, thứ trong tuần, mùa")
        print("      💡 Ý nghĩa: AQI có PATTERN theo thời gian (mùa đông ô nhiễm hơn, cuối tuần khác ngày thường)")
        
        # 4. Lag features
        lag_features = [col for col in self.data.columns if '_lag' in col]
        print(f"\n   4️⃣ LAG FEATURES (Giá trị quá khứ) - {len(lag_features)} features:")
        print("      → Giá trị của pollutants/weather 1-7 ngày trước")
        if lag_features:
            print(f"        Ví dụ: {lag_features[0]}, {lag_features[1]}")
        print("      💡 Ý nghĩa: AQI có QUÁN TÍNH (hôm nay giống hôm qua), dùng để dự đoán")
        
        # 5. Rolling features
        rolling_features = [col for col in self.data.columns if '_rolling' in col]
        print(f"\n   5️⃣ ROLLING FEATURES (Thống kê trượt) - {len(rolling_features)} features:")
        print("      → Trung bình/độ lệch chuẩn của pollutants/weather trong 7-14 ngày qua")
        if rolling_features:
            print(f"        Ví dụ: {rolling_features[0]}, {rolling_features[1]}")
        print("      💡 Ý nghĩa: Nắm bắt XU HƯỚNG và BIẾN ĐỘNG dài hạn")
        
        # 6. Target variable
        print(f"\n   6️⃣ TARGET VARIABLE (Biến mục tiêu):")
        print("      → AQI (Air Quality Index) - Chỉ số chất lượng không khí")
        print(f"        - Giá trị: {self.data['aqi'].min():.1f} - {self.data['aqi'].max():.1f}")
        print(f"        - Trung bình: {self.data['aqi'].mean():.1f}")
        print(f"        - AQI Category: {self.data['aqi_category'].value_counts().to_dict()}")
        print("      💡 Ý nghĩa: Đây là cái chúng ta muốn DỰ ĐOÁN (ngày mai AQI là bao nhiêu?)")
        
        print("\n🎯 TỔNG KẾT:")
        print("   ✅ Dataset này có đầy đủ dữ liệu để:")
        print("      1. Hiểu xu hướng ô nhiễm theo thời gian")
        print("      2. Phân tích mối quan hệ giữa pollutants, thời tiết và AQI")
        print("      3. Xây dựng mô hình ML dự đoán AQI ngày mai")
        print("      4. Đưa ra cảnh báo sức khỏe cho người dân")
        
        print("\n" + "="*80)
    
    def query_1_aqi_trend_over_time(self):
        """Câu truy vấn 1: Xu hướng AQI theo thời gian - Có cải thiện không?"""
        print("\n" + "="*80)
        print("🔍 CÂU TRUY VẤN 1: XU HƯỚNG AQI THEO THỜI GIAN")
        print("   Câu hỏi: 'Chất lượng không khí có cải thiện qua các năm không?'")
        print("="*80)
        
        if self.data is None:
            print("❌ Chưa có dữ liệu")
            return
        
        # Tính AQI trung bình theo năm
        yearly_aqi = self.data.groupby('year')['aqi'].agg(['mean', 'min', 'max', 'std']).round(2)
        
        print("\n📊 KẾT QUẢ:")
        print(yearly_aqi)
        
        first_year = yearly_aqi.index[0]
        last_year = yearly_aqi.index[-1]
        first_aqi = yearly_aqi.loc[first_year, 'mean']
        last_aqi = yearly_aqi.loc[last_year, 'mean']
        change = last_aqi - first_aqi
        change_pct = (change / first_aqi) * 100
        
        print(f"\n💡 PHÂN TÍCH:")
        print(f"   - Năm {first_year}: AQI trung bình = {first_aqi:.1f}")
        print(f"   - Năm {last_year}: AQI trung bình = {last_aqi:.1f}")
        
        if change < -5:
            print(f"   ✅ CẢI THIỆN: Giảm {abs(change):.1f} điểm AQI ({abs(change_pct):.1f}%)")
            print("      → Chất lượng không khí đã tốt hơn!")
        elif change > 5:
            print(f"   ⚠️  XẤU ĐI: Tăng {change:.1f} điểm AQI ({change_pct:.1f}%)")
            print("      → Chất lượng không khí đã xấu hơn!")
        else:
            print(f"   ➡️  ỔN ĐỊNH: Thay đổi {abs(change):.1f} điểm AQI ({abs(change_pct):.1f}%)")
            print("      → Chất lượng không khí tương đối ổn định")
        
        # Phân tích theo thập kỷ
        self.data['decade'] = (self.data['year'] // 10) * 10
        decade_aqi = self.data.groupby('decade')['aqi'].mean().round(2)
        
        print(f"\n📈 XU HƯỚNG THEO THẬP KỶ:")
        for decade, aqi in decade_aqi.items():
            print(f"   - {int(decade)}s: AQI = {aqi:.1f}")
        
        print("\n🎯 Ý NGHĨA:")
        print("   → Hiểu xu hướng giúp đánh giá hiệu quả các chính sách bảo vệ môi trường")
        print("   → Dự đoán được xu hướng tương lai để lập kế hoạch")
        
        # Lưu kết quả
        results = {
            'first_year': int(first_year),
            'last_year': int(last_year),
            'first_aqi': float(first_aqi),
            'last_aqi': float(last_aqi),
            'change': float(change),
            'change_pct': float(change_pct),
            'yearly_trend': yearly_aqi.to_dict('index'),
            'decade_trend': {int(k): float(v) for k, v in decade_aqi.items()}
        }
        self.save_analysis_results("Query_1_AQI_Trend", results, 
                                   f"Xu hướng AQI: {first_year}-{last_year}, thay đổi {change:+.1f} điểm")
        
        return results
    
    def query_2_most_important_pollutant(self):
        """Câu truy vấn 2: Pollutant nào quan trọng nhất đối với AQI?"""
        print("\n" + "="*80)
        print("🔍 CÂU TRUY VẤN 2: POLLUTANT NÀO QUAN TRỌNG NHẤT?")
        print("   Câu hỏi: 'Chất ô nhiễm nào ảnh hưởng nhiều nhất đến AQI?'")
        print("="*80)
        
        if self.data is None:
            print("❌ Chưa có dữ liệu")
            return
        
        # Tính AQI cho từng pollutant riêng lẻ (6 pollutants theo chuẩn EPA)
        pollutants = ['PM2_5', 'PM10', 'NO2_trop', 'SO2_column', 'CO', 'O3']
        available_pollutants = [p for p in pollutants if p in self.data.columns]
        
        pollutant_aqi = {}
        for pol in available_pollutants:
            # Tính AQI từ pollutant này (giống như trong calculate_aqi)
            if pol == 'PM2_5':
                pm25_breaks = [(0, 0), (12.0, 50), (35.4, 100), (55.4, 150), (150.4, 200), (250.4, 300), (350.4, 400), (500.4, 500)]
                aqi_values = self.data[pol].apply(lambda x: self._calc_aqi_single(x, pm25_breaks))
            elif pol == 'PM10':
                pm10_breaks = [(0, 0), (54, 50), (154, 100), (254, 150), (354, 200), (424, 300), (504, 400), (604, 500)]
                aqi_values = self.data[pol].apply(lambda x: self._calc_aqi_single(x, pm10_breaks))
            elif pol == 'NO2_trop':
                no2_breaks = [(0, 0), (53, 50), (100, 100), (360, 150), (649, 200), (1249, 300), (1649, 400), (2049, 500)]
                no2_scaling = 2.67e-16
                no2_ppb = self.data[pol] * no2_scaling
                no2_ppb = no2_ppb.clip(0, 5000)
                aqi_values = no2_ppb.apply(lambda x: self._calc_aqi_single(x, no2_breaks))
            elif pol == 'SO2_column':
                so2_breaks = [(0, 0), (35, 50), (75, 100), (185, 150), (304, 200), (604, 300), (804, 400), (1004, 500)]
                so2_scaling = 0.1 * 1000
                so2_ppb = self.data[pol] * so2_scaling
                so2_ppb = so2_ppb.clip(0, 5000)
                aqi_values = so2_ppb.apply(lambda x: self._calc_aqi_single(x, so2_breaks))
            elif pol == 'CO':
                co_breaks = [(0, 0), (4.4, 50), (9.4, 100), (12.4, 150), (15.4, 200), (30.4, 300), (40.4, 400), (50.4, 500)]
                co_ppm = self.data[pol] / 1000
                aqi_values = co_ppm.apply(lambda x: self._calc_aqi_single(x, co_breaks))
            elif pol == 'O3':
                o3_breaks = [(0, 0), (54, 50), (70, 100), (85, 150), (105, 200), (200, 300), (300, 400), (400, 500)]
                o3_scaling = 0.2
                o3_ppb = self.data[pol] * o3_scaling
                o3_ppb = o3_ppb.clip(0, 1000)
                aqi_values = o3_ppb.apply(lambda x: self._calc_aqi_single(x, o3_breaks))
            else:
                continue
            
            pollutant_aqi[pol] = aqi_values.mean()
        
        # Sắp xếp theo AQI trung bình
        sorted_pollutants = sorted(pollutant_aqi.items(), key=lambda x: x[1], reverse=True)
        
        print("\n📊 KẾT QUẢ - AQI TRUNG BÌNH TỪ TỪNG POLLUTANT:")
        for i, (pol, aqi) in enumerate(sorted_pollutants, 1):
            print(f"   {i}. {pol:15s}: AQI = {aqi:.1f}")
        
        # So sánh với AQI thực tế
        actual_aqi = self.data['aqi'].mean()
        print(f"\n   🌍 AQI THỰC TẾ (tổng hợp): {actual_aqi:.1f}")
        
        # Pollutant nào thường "quyết định" AQI (tức là AQI = max của các AQI thành phần)
        aqi_components = pd.DataFrame()
        for pol in available_pollutants:
            col_name = f'aqi_{pol}'
            # Tính lại AQI cho pollutant này
            if pol == 'PM2_5':
                pm25_breaks = [(0, 0), (12.0, 50), (35.4, 100), (55.4, 150), (150.4, 200), (250.4, 300), (350.4, 400), (500.4, 500)]
                aqi_components[col_name] = self.data[pol].apply(lambda x: self._calc_aqi_single(x, pm25_breaks))
            elif pol == 'PM10':
                pm10_breaks = [(0, 0), (54, 50), (154, 100), (254, 150), (354, 200), (424, 300), (504, 400), (604, 500)]
                aqi_components[col_name] = self.data[pol].apply(lambda x: self._calc_aqi_single(x, pm10_breaks))
            elif pol == 'NO2_trop':
                no2_breaks = [(0, 0), (53, 50), (100, 100), (360, 150), (649, 200), (1249, 300), (1649, 400), (2049, 500)]
                no2_scaling = 2.67e-16
                no2_ppb = self.data[pol] * no2_scaling
                no2_ppb = no2_ppb.clip(0, 5000)
                aqi_components[col_name] = no2_ppb.apply(lambda x: self._calc_aqi_single(x, no2_breaks))
            elif pol == 'SO2_column':
                so2_breaks = [(0, 0), (35, 50), (75, 100), (185, 150), (304, 200), (604, 300), (804, 400), (1004, 500)]
                so2_scaling = 0.1 * 1000
                so2_ppb = self.data[pol] * so2_scaling
                so2_ppb = so2_ppb.clip(0, 5000)
                aqi_components[col_name] = so2_ppb.apply(lambda x: self._calc_aqi_single(x, so2_breaks))
            elif pol == 'CO':
                co_breaks = [(0, 0), (4.4, 50), (9.4, 100), (12.4, 150), (15.4, 200), (30.4, 300), (40.4, 400), (50.4, 500)]
                co_ppm = self.data[pol] / 1000
                aqi_components[col_name] = co_ppm.apply(lambda x: self._calc_aqi_single(x, co_breaks))
            elif pol == 'O3':
                o3_breaks = [(0, 0), (54, 50), (70, 100), (85, 150), (105, 200), (200, 300), (300, 400), (400, 500)]
                o3_scaling = 0.2
                o3_ppb = self.data[pol] * o3_scaling
                o3_ppb = o3_ppb.clip(0, 1000)
                aqi_components[col_name] = o3_ppb.apply(lambda x: self._calc_aqi_single(x, o3_breaks))
        
        # Đếm số lần mỗi pollutant là "quyết định" (AQI = max)
        if len(aqi_components.columns) > 0:
            aqi_max = aqi_components.max(axis=1)
            dominant_pollutant = aqi_components.idxmax(axis=1)
            dominant_count = dominant_pollutant.value_counts()
            
            print(f"\n💡 PHÂN TÍCH:")
            print(f"   Pollutant 'quyết định' AQI (số lần cao nhất):")
            for pol_col, count in dominant_count.head(5).items():
                pol_name = pol_col.replace('aqi_', '')
                pct = (count / len(dominant_pollutant)) * 100
                print(f"   - {pol_name:15s}: {count:,} lần ({pct:.1f}%)")
        
        # Tương quan giữa pollutants và AQI
        print(f"\n🔗 TƯƠNG QUAN VỚI AQI THỰC TẾ:")
        correlations = {}
        for pol in available_pollutants:
            if pol in self.data.columns:
                corr = self.data[[pol, 'aqi']].corr().iloc[0, 1]
                correlations[pol] = corr
        
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        for pol, corr in sorted_corr[:5]:
            print(f"   - {pol:15s}: {corr:+.3f}")
        
        most_important = sorted_corr[0][0] if sorted_corr else "N/A"
        print(f"\n🎯 KẾT LUẬN:")
        print(f"   → Pollutant quan trọng nhất: {most_important}")
        print(f"   → Đây là chất ô nhiễm cần ưu tiên kiểm soát")
        
        results = {
            'pollutant_aqi_avg': {k: float(v) for k, v in pollutant_aqi.items()},
            'actual_aqi': float(actual_aqi),
            'correlations': {k: float(v) for k, v in correlations.items()},
            'most_important': most_important
        }
        self.save_analysis_results("Query_2_Most_Important_Pollutant", results, 
                                   f"Pollutant quan trọng nhất: {most_important}")
        
        return results
    
    def _calc_aqi_single(self, conc, breakpoints):
        """Helper function để tính AQI cho một pollutant"""
        if pd.isna(conc) or conc < 0:
            return np.nan
        
        for i in range(len(breakpoints) - 1):
            if breakpoints[i][0] <= conc <= breakpoints[i+1][0]:
                aqi_low = breakpoints[i][1]
                aqi_high = breakpoints[i+1][1]
                conc_low = breakpoints[i][0]
                conc_high = breakpoints[i+1][0]
                
                aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (conc - conc_low) + aqi_low
                return round(aqi)
        
        return 500
    
    def query_3_weather_impact_on_aqi(self):
        """Câu truy vấn 3: Thời tiết ảnh hưởng như thế nào đến AQI?"""
        print("\n" + "="*80)
        print("🔍 CÂU TRUY VẤN 3: ẢNH HƯỞNG CỦA THỜI TIẾT LÊN AQI")
        print("   Câu hỏi: 'Nhiệt độ, độ ẩm, gió ảnh hưởng đến AQI như thế nào?'")
        print("="*80)
        
        if self.data is None:
            print("❌ Chưa có dữ liệu")
            return
        
        weather_vars = ['temperature_mean', 'humidity', 'wind_speed', 'precipitation']
        available_weather = [w for w in weather_vars if w in self.data.columns]
        
        print("\n📊 KẾT QUẢ - TƯƠNG QUAN GIỮA THỜI TIẾT VÀ AQI:")
        
        correlations = {}
        for w in available_weather:
            corr = self.data[[w, 'aqi']].corr().iloc[0, 1]
            correlations[w] = corr
            
            # Giải thích ý nghĩa
            if abs(corr) < 0.1:
                impact = "Không đáng kể"
            elif abs(corr) < 0.3:
                impact = "Yếu"
            elif abs(corr) < 0.5:
                impact = "Trung bình"
            elif abs(corr) < 0.7:
                impact = "Mạnh"
            else:
                impact = "Rất mạnh"
            
            direction = "Tăng" if corr > 0 else "Giảm"
            print(f"   - {w:20s}: {corr:+.3f} ({impact})")
            print(f"     → {direction} {w} → {direction} AQI")
        
        # Phân tích chi tiết theo từng biến
        print(f"\n🌡️ PHÂN TÍCH THEO NHIỆT ĐỘ:")
        temp_bins = pd.qcut(self.data['temperature_mean'], q=5, labels=['Rất lạnh', 'Lạnh', 'Trung bình', 'Ấm', 'Nóng'], duplicates='drop')
        temp_analysis = self.data.groupby(temp_bins).agg({
            'aqi': ['mean', 'count'],
            'PM2_5': 'mean'
        }).round(2)
        print(temp_analysis)
        
        print(f"\n💧 PHÂN TÍCH THEO ĐỘ ẨM:")
        humidity_bins = pd.qcut(self.data['humidity'], q=5, labels=['Rất khô', 'Khô', 'Trung bình', 'Ẩm', 'Rất ẩm'], duplicates='drop')
        humidity_analysis = self.data.groupby(humidity_bins).agg({
            'aqi': ['mean', 'count']
        }).round(2)
        print(humidity_analysis)
        
        print(f"\n💨 PHÂN TÍCH THEO TỐC ĐỘ GIÓ:")
        wind_bins = pd.qcut(self.data['wind_speed'], q=5, labels=['Rất yếu', 'Yếu', 'Trung bình', 'Mạnh', 'Rất mạnh'], duplicates='drop')
        wind_analysis = self.data.groupby(wind_bins).agg({
            'aqi': ['mean', 'count']
        }).round(2)
        print(wind_analysis)
        
        print(f"\n🎯 Ý NGHĨA:")
        print("   → Hiểu ảnh hưởng thời tiết giúp dự đoán AQI chính xác hơn")
        print("   → Gió mạnh → phân tán ô nhiễm → AQI thấp")
        print("   → Mưa → rửa sạch không khí → AQI thấp")
        print("   → Nhiệt độ cao + không có gió → tích tụ ô nhiễm → AQI cao")
        
        # Flatten MultiIndex columns trước khi convert sang dict
        temp_analysis_flat = temp_analysis.copy()
        temp_analysis_flat.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                      for col in temp_analysis_flat.columns]
        humidity_analysis_flat = humidity_analysis.copy()
        humidity_analysis_flat.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                         for col in humidity_analysis_flat.columns]
        wind_analysis_flat = wind_analysis.copy()
        wind_analysis_flat.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                     for col in wind_analysis_flat.columns]
        
        results = {
            'correlations': {k: float(v) if not pd.isna(v) else None for k, v in correlations.items()},
            'temperature_analysis': temp_analysis_flat.to_dict('index'),
            'humidity_analysis': humidity_analysis_flat.to_dict('index'),
            'wind_analysis': wind_analysis_flat.to_dict('index')
        }
        self.save_analysis_results("Query_3_Weather_Impact", results, 
                                   "Phân tích ảnh hưởng thời tiết lên AQI")
        
        return results
    
    def query_4_seasonal_patterns(self):
        """Câu truy vấn 4: Mùa nào ô nhiễm nhất?"""
        print("\n" + "="*80)
        print("🔍 CÂU TRUY VẤN 4: PHÂN BỐ AQI THEO MÙA/THÁNG")
        print("   Câu hỏi: 'Mùa nào/tháng nào ô nhiễm nhất? Có pattern theo thời gian không?'")
        print("="*80)
        
        if self.data is None:
            print("❌ Chưa có dữ liệu")
            return
        
        # Phân tích theo mùa
        season_names = {0: 'Đông (Dec-Feb)', 1: 'Xuân (Mar-May)', 2: 'Hè (Jun-Aug)', 3: 'Thu (Sep-Nov)'}
        seasonal_stats = self.data.groupby('season').agg({
            'aqi': ['mean', 'std', 'min', 'max'],
            'PM2_5': 'mean',
            'temperature_mean': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }).round(2)
        
        print("\n📊 KẾT QUẢ THEO MÙA:")
        for season in range(4):
            if season in seasonal_stats.index:
                season_name = season_names[season]
                aqi_mean = seasonal_stats.loc[season, ('aqi', 'mean')]
                print(f"   {season_name}:")
                print(f"      - AQI trung bình: {aqi_mean:.1f}")
                print(f"      - Nhiệt độ: {seasonal_stats.loc[season, ('temperature_mean', 'mean')]:.1f}°C")
                print(f"      - Độ ẩm: {seasonal_stats.loc[season, ('humidity', 'mean')]:.1f}%")
                print(f"      - Gió: {seasonal_stats.loc[season, ('wind_speed', 'mean')]:.2f} m/s")
        
        worst_season = seasonal_stats[('aqi', 'mean')].idxmax()
        best_season = seasonal_stats[('aqi', 'mean')].idxmin()
        print(f"\n   ⚠️  MÙA Ô NHIỄM NHẤT: {season_names[worst_season]} (AQI = {seasonal_stats.loc[worst_season, ('aqi', 'mean')]:.1f})")
        print(f"   ✅ MÙA SẠCH NHẤT: {season_names[best_season]} (AQI = {seasonal_stats.loc[best_season, ('aqi', 'mean')]:.1f})")
        
        # Phân tích theo tháng
        monthly_stats = self.data.groupby('month').agg({
            'aqi': 'mean',
            'PM2_5': 'mean'
        }).round(2)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"\n📅 KẾT QUẢ THEO THÁNG:")
        print("   Tháng | AQI trung bình | PM2.5")
        print("   " + "-" * 40)
        for month in range(1, 13):
            if month in monthly_stats.index:
                aqi = monthly_stats.loc[month, 'aqi']
                pm25 = monthly_stats.loc[month, 'PM2_5']
                print(f"   {month:2d} ({month_names[month-1]:3s}) | {aqi:13.1f} | {pm25:.2f}")
        
        worst_month = monthly_stats['aqi'].idxmax()
        best_month = monthly_stats['aqi'].idxmin()
        print(f"\n   ⚠️  THÁNG Ô NHIỄM NHẤT: Tháng {int(worst_month)} ({month_names[int(worst_month)-1]}) - AQI = {monthly_stats.loc[worst_month, 'aqi']:.1f}")
        print(f"   ✅ THÁNG SẠCH NHẤT: Tháng {int(best_month)} ({month_names[int(best_month)-1]}) - AQI = {monthly_stats.loc[best_month, 'aqi']:.1f}")
        
        # Phân tích theo ngày trong tuần
        weekday_stats = self.data.groupby('weekday').agg({
            'aqi': 'mean'
        }).round(2)
        
        weekday_names = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
        print(f"\n📆 KẾT QUẢ THEO NGÀY TRONG TUẦN:")
        for day, aqi in weekday_stats.iterrows():
            print(f"   {weekday_names[int(day)]}: AQI = {aqi['aqi']:.1f}")
        
        print(f"\n🎯 Ý NGHĨA:")
        print("   → Hiểu pattern theo mùa/tháng giúp dự đoán chính xác hơn")
        print("   → Mùa đông thường ô nhiễm hơn do: đốt nhiên liệu, nghịch nhiệt, ít gió")
        print("   → Mùa hè có thể ô nhiễm do: nắng nóng, O3 tăng cao")
        print("   → Cuối tuần có thể khác ngày thường do: giao thông giảm")
        
        # Flatten MultiIndex columns trước khi convert sang dict
        seasonal_stats_flat = seasonal_stats.copy()
        seasonal_stats_flat.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                      for col in seasonal_stats_flat.columns]
        
        results = {
            'seasonal_stats': seasonal_stats_flat.to_dict('index'),
            'monthly_stats': monthly_stats.to_dict('index'),
            'weekday_stats': weekday_stats.to_dict('index'),
            'worst_season': int(worst_season),
            'best_season': int(best_season),
            'worst_month': int(worst_month),
            'best_month': int(best_month)
        }
        self.save_analysis_results("Query_4_Seasonal_Patterns", results, 
                                   f"Mùa ô nhiễm nhất: {season_names[worst_season]}")
        
        return results
    
    def query_5_pollutant_correlations(self):
        """Câu truy vấn 5: Tương quan giữa các pollutants"""
        print("\n" + "="*80)
        print("🔍 CÂU TRUY VẤN 5: TƯƠNG QUAN GIỮA CÁC POLLUTANTS")
        print("   Câu hỏi: 'Các chất ô nhiễm có xuất hiện cùng nhau không? Nhóm nào tương quan cao?'")
        print("="*80)
        
        if self.data is None:
            print("❌ Chưa có dữ liệu")
            return
        
        pollutants = ['PM2_5', 'PM10', 'NO2_trop', 'NO2_total', 'SO2_column', 'SO2_surface', 'O3', 'CO', 'AOD_550']
        available_pollutants = [p for p in pollutants if p in self.data.columns]
        
        # Tính correlation matrix
        corr_matrix = self.data[available_pollutants].corr()
        
        print("\n📊 MA TRẬN TƯƠNG QUAN (Top correlations):")
        
        # Tìm các cặp tương quan cao nhất
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                pol1 = corr_matrix.columns[i]
                pol2 = corr_matrix.columns[j]
                corr_value = corr_matrix.loc[pol1, pol2]
                
                if abs(corr_value) > 0.5:  # Tương quan mạnh
                    high_corr_pairs.append((pol1, pol2, corr_value))
        
        # Sắp xếp theo độ lớn
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("\n   🔗 CÁC CẶP TƯƠNG QUAN MẠNH (|r| > 0.5):")
        for pol1, pol2, corr in high_corr_pairs[:10]:
            direction = "Cùng tăng" if corr > 0 else "Ngược chiều"
            print(f"   - {pol1:15s} ↔ {pol2:15s}: {corr:+.3f} ({direction})")
        
        # Phân tích nhóm
        print(f"\n💡 PHÂN TÍCH NHÓM:")
        
        # Nhóm 1: PM (Particulate Matter)
        pm_group = ['PM2_5', 'PM10', 'AOD_550']
        available_pm = [p for p in pm_group if p in available_pollutants]
        if len(available_pm) >= 2:
            pm_corr = corr_matrix.loc[available_pm, available_pm]
            avg_pm_corr = pm_corr.values[np.triu_indices_from(pm_corr.values, k=1)].mean()
            print(f"   📍 NHÓM PM (Bụi mịn): {', '.join(available_pm)}")
            print(f"      → Tương quan trung bình: {avg_pm_corr:+.3f}")
            print(f"      → Giải thích: PM2.5 và PM10 cùng nguồn (bụi), AOD đo bụi từ vệ tinh")
        
        # Nhóm 2: NO2, SO2 (Khí từ đốt nhiên liệu)
        gas_group = ['NO2_trop', 'NO2_total', 'SO2_column', 'SO2_surface', 'CO']
        available_gas = [p for p in gas_group if p in available_pollutants]
        if len(available_gas) >= 2:
            gas_corr = corr_matrix.loc[available_gas, available_gas]
            avg_gas_corr = gas_corr.values[np.triu_indices_from(gas_corr.values, k=1)].mean()
            print(f"   📍 NHÓM KHÍ (NO2, SO2, CO): {', '.join(available_gas)}")
            print(f"      → Tương quan trung bình: {avg_gas_corr:+.3f}")
            print(f"      → Giải thích: Cùng từ đốt nhiên liệu (xe, nhà máy)")
        
        # O3 (đặc biệt)
        if 'O3' in available_pollutants:
            o3_corr = corr_matrix.loc['O3', available_pollutants].drop('O3')
            print(f"   📍 O3 (Ozone):")
            print(f"      → Tương quan với các pollutants khác: Yếu (O3 hình thành từ phản ứng hóa học)")
            print(f"      → O3 thường ngược chiều với NO2 (O3 + NO → NO2 + O2)")
        
        print(f"\n🎯 Ý NGHĨA:")
        print("   → Hiểu tương quan giúp:")
        print("      • Giảm số features (dùng 1 pollutant đại diện cho nhóm)")
        print("      • Phát hiện nguồn ô nhiễm (cùng nguồn → tương quan cao)")
        print("      • Dự đoán tốt hơn (nếu biết PM2.5, có thể ước lượng PM10)")
        
        # Convert correlation matrix thành dict đơn giản
        corr_dict = {}
        for col in corr_matrix.columns:
            corr_dict[col] = {}
            for idx in corr_matrix.index:
                val = corr_matrix.loc[idx, col]
                corr_dict[col][idx] = float(val) if not pd.isna(val) else None
        
        results = {
            'correlation_matrix': corr_dict,
            'high_correlation_pairs': [(p1, p2, float(c)) for p1, p2, c in high_corr_pairs],
            'pm_group_correlation': float(avg_pm_corr) if len(available_pm) >= 2 else None,
            'gas_group_correlation': float(avg_gas_corr) if len(available_gas) >= 2 else None
        }
        self.save_analysis_results("Query_5_Pollutant_Correlations", results, 
                                   f"Tìm thấy {len(high_corr_pairs)} cặp tương quan mạnh")
        
        return results

def main():
    """Hàm chính để phân tích dữ liệu"""
    print("🔍 BẮT ĐẦU PHÂN TÍCH DỮ LIỆU AQI (2004-2025)")
    print("=" * 80)
    
    # Khởi tạo analyzer
    analyzer = AQIDataAnalyzer()
    
    # Tải dữ liệu
    if not analyzer.load_data():
        print("❌ Không thể tải dữ liệu")
        return
    
    try:
        # Mô tả dataset
        analyzer.describe_dataset()
        
        # 5 CÂU TRUY VẤN QUAN TRỌNG NHẤT
        print("\n" + "="*80)
        print("🔍 THỰC HIỆN 5 CÂU TRUY VẤN QUAN TRỌNG NHẤT CHO DỰ ÁN")
        print("="*80)
        
        query_1 = analyzer.query_1_aqi_trend_over_time()
        query_2 = analyzer.query_2_most_important_pollutant()
        query_3 = analyzer.query_3_weather_impact_on_aqi()
        query_4 = analyzer.query_4_seasonal_patterns()
        query_5 = analyzer.query_5_pollutant_correlations()
        
        # Lưu dữ liệu vào database
        analyzer.save_to_database()
        analyzer.save_trends_to_database()
        
        # Báo cáo tổng hợp ngắn gọn
        analyzer.generate_summary_report()
        
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH PHÂN TÍCH DỮ LIỆU!")
        print("="*80)
        print("📊 Đã thực hiện 5 câu truy vấn quan trọng:")
        print("   1. ✅ Xu hướng AQI theo thời gian")
        print("   2. ✅ Pollutant quan trọng nhất")
        print("   3. ✅ Ảnh hưởng của thời tiết")
        print("   4. ✅ Phân bố AQI theo mùa/tháng")
        print("   5. ✅ Tương quan giữa các pollutants")
        print("\n📊 Dữ liệu đã sẵn sàng cho bước tiếp theo: Data Visualization")
        print("🗄️ Database đã được cập nhật với tất cả dữ liệu và kết quả phân tích")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình phân tích: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



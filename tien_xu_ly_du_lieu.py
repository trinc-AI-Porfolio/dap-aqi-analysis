import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import warnings
import sys
import codecs

# Fix encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

warnings.filterwarnings('ignore')

# Set random seed để đảm bảo reproducibility
np.random.seed(42)

print("="*80)
print("🔄 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
print("="*80)

# ============================================================================
# 1. ĐỊNH NGHĨA HÀM ĐỌC FILE CSV
# ============================================================================

def read_giovanni_csv(filepath, feature_name=None):
    """
    Đọc file CSV từ Giovanni, xử lý fill values CHÍNH XÁC và trả về DataFrame
    CHỈ thay thế fill values được khai báo trong header, KHÔNG tự động loại bỏ dữ liệu thực
    """
    try:
        # Đọc file, skip 8 dòng đầu (header metadata)
        df = pd.read_csv(filepath, skiprows=8)
        
        # Lấy tên file để xác định feature
        filename = os.path.basename(filepath)
        
        # Tìm column chứa data (không phải 'time')
        data_col = [col for col in df.columns if col != 'time'][0]
        
        # Đọc fill value từ header (CHÍNH XÁC)
        fill_value = None
        fill_value_str = None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Fill Value' in line:
                        try:
                            # Lấy fill value từ dòng "Fill Value (column_name):, value"
                            fill_val_str = line.split(',')[-1].strip()
                            fill_value = float(fill_val_str)
                            fill_value_str = fill_val_str
                            break
                        except:
                            pass
        except:
            pass
        
        # Parse time
        df['time'] = pd.to_datetime(df['time'])
        
        # Đếm số records ban đầu
        total_records = len(df)
        
        # XỬ LÝ FILL VALUES - CHỈ THAY THẾ CHÍNH XÁC GIÁ TRỊ ĐƯỢC KHAI BÁO
        fill_count = 0
        if fill_value is not None:
            # Kiểm tra xem có bao nhiêu giá trị khớp với fill_value
            # Sử dụng np.isclose để xử lý floating point precision
            if abs(fill_value) > 1e6:  # Fill value rất lớn (như 1e15, -1e30)
                # So sánh chính xác cho giá trị lớn
                fill_mask = df[data_col] == fill_value
            else:
                # So sánh với tolerance cho giá trị nhỏ
                fill_mask = np.isclose(df[data_col], fill_value, rtol=1e-10, atol=1e-10)
            
            fill_count = fill_mask.sum()
            
            # CHỈ thay thế các giá trị khớp chính xác với fill_value
            if fill_count > 0:
                df.loc[fill_mask, data_col] = np.nan
                print(f"      ⚠️  Thay thế {fill_count:,}/{total_records:,} fill values ({fill_value_str})")
        
        # KIỂM TRA GIÁ TRỊ BẤT THƯỜNG
        # Chỉ log để user biết, không xóa dữ liệu thực
        if df[data_col].dtype in [np.float64, np.float32]:
            valid_data = df[data_col].dropna()
            
            if len(valid_data) > 0:
                # Tính statistical outliers (IQR method) - CHỈ ĐỂ THÔNG BÁO
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # 3-sigma để tránh loại bỏ nhầm
                upper_bound = Q3 + 3 * IQR
                
                outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
                
                if len(outliers) > 0:
                    outlier_pct = (len(outliers) / len(valid_data)) * 100
                    # Chỉ cảnh báo nếu outlier > 5% (có thể là vấn đề)
                    if outlier_pct > 5:
                        print(f"      ⚠️  Phát hiện {len(outliers):,} outliers ({outlier_pct:.1f}%) - GIỮ LẠI để giữ độ chính xác")
                
                # Kiểm tra giá trị cực đoan (có thể là fill value nhưng không khớp chính xác)
                # MERRA-2 thường dùng 1e15, OMI dùng -1e30
                if fill_value is not None and abs(fill_value) > 1e10:
                    # Chỉ kiểm tra nếu fill_value là giá trị cực lớn
                    extreme_mask = (abs(df[data_col]) > 1e12) | (df[data_col] < -1e15)
                    extreme_count = extreme_mask.sum()
                    
                    if extreme_count > 0:
                        extreme_pct = (extreme_count / len(valid_data)) * 100
                        if extreme_pct > 0.1:  # Nếu > 0.1% có thể là fill values
                            # Kiểm tra xem các giá trị cực đoan có gần với fill_value không
                            extreme_values = df.loc[extreme_mask, data_col]
                            # Thay thế các giá trị rất gần với fill_value (trong 1% tolerance)
                            close_to_fill = np.isclose(extreme_values, fill_value, rtol=0.01, atol=0)
                            close_count = close_to_fill.sum()
                            
                            if close_count > 0:
                                # Thay thế các giá trị gần fill_value
                                close_indices = extreme_values[close_to_fill].index
                                df.loc[close_indices, data_col] = np.nan
                                print(f"      ⚠️  Phát hiện {extreme_count:,} giá trị cực đoan, đã thay {close_count:,} giá trị gần fill_value")
        
        # Đếm số records hợp lệ sau xử lý
        valid_records = df[data_col].notna().sum()
        valid_pct = (valid_records / total_records) * 100 if total_records > 0 else 0
        
        # Đổi tên column thành tên feature dễ hiểu
        if feature_name:
            df = df.rename(columns={data_col: feature_name})
        
        result_df = df[['time', feature_name if feature_name else data_col]].copy()
        
        # Log summary
        if fill_count > 0:
            print(f"      ✅ Còn lại {valid_records:,}/{total_records:,} records hợp lệ ({valid_pct:.1f}%)")
        
        return result_df
    
    except Exception as e:
        print(f"   ❌ Lỗi đọc file {os.path.basename(filepath)}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 2. ĐỌC TẤT CẢ FILE CSV
# ============================================================================

print("\n📂 Bước 1: Đọc tất cả file CSV...")

# Định nghĩa đường dẫn thư mục input và output
INPUT_DIR = "Data_download"  # Thư mục chứa file CSV gốc
OUTPUT_DIR = "processed"     # Thư mục lưu file đã xử lý

# Tạo thư mục output nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"   📁 Đã tạo thư mục: {OUTPUT_DIR}")

# Kiểm tra thư mục input
if not os.path.exists(INPUT_DIR):
    print(f"   ❌ Không tìm thấy thư mục {INPUT_DIR}")
    print(f"   💡 Vui lòng tạo thư mục {INPUT_DIR} và đặt các file CSV vào đó")
    sys.exit(1)

print(f"   📂 Đọc file từ: {INPUT_DIR}/")
print(f"   💾 Lưu file vào: {OUTPUT_DIR}/")

# Mapping file patterns → feature names
file_patterns = {
    # Pollution Features
    'MOD08_D3_6_1_AOD_550': ('AOD_550', 'daily'),
    'M2T1NXAER_5_12_4_TOTSMASS25': ('PM2_5', 'hourly'),
    'OMNO2d_003_ColumnAmountNO2TropCloudScreened': ('NO2_trop', 'daily'),
    'OMNO2d_003_ColumnAmountNO2CloudScreened': ('NO2_total', 'daily'),
    'OMSO2e_003_ColumnAmountSO2': ('SO2_column', 'daily'),
    'M2T1NXAER_5_12_4_SO2SMASS': ('SO2_surface', 'hourly'),
    'OMTO3e_003_ColumnAmountO3': ('O3', 'daily'),
    'AIRS3STD_7_0_CO_VMR_A': ('CO', 'daily'),
    
    # Weather Features
    'M2SDNXSLV_5_12_4_T2MMAX': ('temperature_max', 'daily'),
    'M2SDNXSLV_5_12_4_T2MMEAN': ('temperature_mean', 'daily'),
    'M2SDNXSLV_5_12_4_T2MMIN': ('temperature_min', 'daily'),
    'M2T1NXSLV_5_12_4_TS': ('skin_temperature', 'hourly'),
    'M2TMNXSLV_5_12_4_T2MDEW': ('dew_point', 'monthly'),
    'M2T1NXSLV_5_12_4_U10M': ('wind_u10m', 'hourly'),
    'M2T1NXSLV_5_12_4_V10M': ('wind_v10m', 'hourly'),
}

# Tìm và đọc các file
dataframes = {}
for pattern, (feature_name, resolution) in file_patterns.items():
    # Tìm file trong thư mục INPUT_DIR
    files = glob.glob(os.path.join(INPUT_DIR, f"g4.areaAvgTimeSeries.{pattern}*.csv"))
    if files:
        filepath = files[0]  # Lấy file đầu tiên nếu có nhiều
        print(f"   📄 Đọc {feature_name} ({resolution})...")
        df = read_giovanni_csv(filepath, feature_name)
        if df is not None:
            dataframes[feature_name] = {'df': df, 'resolution': resolution}
            print(f"      ✅ {len(df):,} records")
        else:
            print(f"      ❌ Lỗi đọc file")
    else:
        print(f"   ⚠️  Không tìm thấy file cho {feature_name} trong {INPUT_DIR}/")

# Kiểm tra Precipitation (nếu đã download)
precip_files = glob.glob(os.path.join(INPUT_DIR, "*precipitation*.csv")) + \
               glob.glob(os.path.join(INPUT_DIR, "*PRECIP*.csv")) + \
               glob.glob(os.path.join(INPUT_DIR, "*PRCP*.csv"))
if precip_files:
    print(f"   📄 Đọc precipitation...")
    df_precip = read_giovanni_csv(precip_files[0], 'precipitation')
    if df_precip is not None:
        # Kiểm tra resolution
        if len(df_precip) > 1000:
            resolution = 'hourly'
        else:
            resolution = 'daily'
        dataframes['precipitation'] = {'df': df_precip, 'resolution': resolution}
        print(f"      ✅ {len(df_precip):,} records ({resolution})")

# ============================================================================
# 3. XỬ LÝ RESAMPLE/AGGREGATE
# ============================================================================

print("\n🔄 Bước 2: Resample/Aggregate dữ liệu về Daily...")

# ============================================================================
# CẤU HÌNH: GIỚI HẠN THỜI GIAN VÀ XỬ LÝ MISSING VALUES
# ============================================================================

# Ngày bắt đầu (đồng bộ thời gian)
START_DATE = '2004-10-01'
END_DATE = '2025-09-30'

# Tạo date range chung (từ ngày bắt đầu đến ngày kết thúc)
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
master_df = pd.DataFrame({'date': date_range})

print(f"\n📅 Giới hạn thời gian: {START_DATE} đến {END_DATE}")
print(f"   Tổng số ngày: {len(date_range):,}")

# Mapping lý do missing values cho từng feature
missing_reasons = {
    'NO2_trop': 'Vệ tinh OMI bắt đầu hoạt động từ 2004-10-01',
    'NO2_total': 'Vệ tinh OMI bắt đầu hoạt động từ 2004-10-01',
    'SO2_column': 'Vệ tinh OMI bắt đầu hoạt động từ 2004-10-01',
    'O3': 'Vệ tinh OMI bắt đầu hoạt động từ 2004-10-01',
    'CO': 'Vệ tinh AIRS bắt đầu hoạt động từ 2002-08-31',
    'AOD_550': 'MODIS có thể có missing do mây che phủ',
    'PM2_5': 'MERRA-2 reanalysis, có thể có missing do điều kiện thời tiết',
    'SO2_surface': 'MERRA-2 reanalysis, có thể có missing',
    'precipitation': 'Có thể có missing nếu chưa download'
}

# Xử lý từng feature
for feature_name, info in dataframes.items():
    df = info['df'].copy()
    resolution = info['resolution']
    
    # Đổi tên column time thành date
    df = df.rename(columns={'time': 'date'})
    
    # LỌC THEO NGÀY BẮT ĐẦU (2024-10-01)
    df = df[df['date'] >= START_DATE].copy()
    
    if len(df) == 0:
        print(f"   ⚠️  {feature_name}: Không có data từ {START_DATE} trở đi")
        continue
    
    # Log ngày bắt đầu thực tế của data
    actual_start = df['date'].min()
    if actual_start > pd.to_datetime(START_DATE):
        print(f"   ⚠️  {feature_name}: Data bắt đầu từ {actual_start.date()} (sau {START_DATE})")
        if feature_name in missing_reasons:
            print(f"      → Lý do: {missing_reasons[feature_name]}")
    
    # Xử lý theo resolution
    if resolution == 'hourly':
        # Resample hourly → daily
        if feature_name == 'precipitation':
            # Precipitation: tổng daily (sum)
            df_daily = df.set_index('date').resample('D')[feature_name].sum().reset_index()
        else:
            # Các features khác: trung bình daily (mean)
            df_daily = df.set_index('date').resample('D')[feature_name].mean().reset_index()
        
        print(f"   ✅ {feature_name}: {len(df):,} hourly → {len(df_daily):,} daily")
        
    elif resolution == 'monthly':
        # Interpolate monthly → daily
        df = df.set_index('date')
        # Tạo index daily (chỉ trong date_range)
        df_daily = df.reindex(date_range)
        # Interpolate (chỉ nội suy giữa các giá trị có sẵn)
        df_daily[feature_name] = df_daily[feature_name].interpolate(method='linear', limit_direction='both')
        df_daily = df_daily.reset_index().rename(columns={'index': 'date'})
        
        print(f"   ✅ {feature_name}: {len(df):,} monthly → {len(df_daily):,} daily (interpolated)")
        
    else:  # daily
        # Đã là daily, chỉ cần đảm bảo format date
        df_daily = df.copy()
        print(f"   ✅ {feature_name}: {len(df_daily):,} daily (giữ nguyên)")
    
    # Merge vào master_df (left join để giữ tất cả dates trong date_range)
    master_df = master_df.merge(df_daily, on='date', how='left')

# ============================================================================
# 4. TÍNH DERIVED FEATURES
# ============================================================================

print("\n🧮 Bước 3: Tính Derived Features...")

# 4.1. PM10 từ PM2.5
if 'PM2_5' in master_df.columns:
    # Convert từ kg/m³ sang μg/m³ (nhân 1e9)
    # PM2.5 trong file là kg/m³, cần convert
    # CHỈ tính PM10 nếu PM2.5 có giá trị (giữ missing nếu PM2.5 missing)
    valid_mask = master_df['PM2_5'].notna()
    pm2_5_ug_m3 = master_df['PM2_5'].copy()
    pm2_5_ug_m3[valid_mask] = pm2_5_ug_m3[valid_mask] * 1e9  # kg/m³ → μg/m³
    
    PM10 = pd.Series(index=master_df.index, dtype=float)
    PM10[valid_mask] = pm2_5_ug_m3[valid_mask] * 1.6  # PM10 ≈ PM2.5 × 1.6
    
    master_df['PM10'] = PM10
    master_df['PM2_5'] = pm2_5_ug_m3  # Update PM2.5 về μg/m³
    print("   ✅ PM10 = PM2.5 × 1.6 (từ μg/m³)")
    print(f"      → Giữ nguyên missing nếu PM2.5 missing")

# 4.2. Relative Humidity từ Temperature + Dew Point
if 'temperature_mean' in master_df.columns and 'dew_point' in master_df.columns:
    # Magnus equation: RH = 100 * exp((17.27 * Td)/(Td + 237.3)) / exp((17.27 * T)/(T + 237.3))
    # T: temperature (K) → C
    # Td: dew point (K) → C
    
    # CHỈ tính RH nếu cả T và Td đều có giá trị (không missing)
    valid_mask = master_df['temperature_mean'].notna() & master_df['dew_point'].notna()
    
    T = master_df['temperature_mean'] - 273.15  # K → C
    Td = master_df['dew_point'] - 273.15  # K → C
    
    # Tính RH (chỉ tính cho các giá trị hợp lệ)
    RH = pd.Series(index=master_df.index, dtype=float)
    RH[valid_mask] = (
        100 * (
            6.112 * np.exp((17.27 * Td[valid_mask]) / (Td[valid_mask] + 237.3)) /
            (6.112 * np.exp((17.27 * T[valid_mask]) / (T[valid_mask] + 237.3)))
        )
    )
    
    # Clip về [0, 100]
    RH = np.clip(RH, 0, 100)
    master_df['humidity'] = RH
    print("   ✅ Relative Humidity (từ Temperature + Dew Point)")
    print(f"      → Giữ nguyên missing nếu T hoặc Td missing")

# 4.3. Wind Speed từ U10M + V10M
if 'wind_u10m' in master_df.columns and 'wind_v10m' in master_df.columns:
    # CHỈ tính nếu cả U và V đều có giá trị
    valid_mask = master_df['wind_u10m'].notna() & master_df['wind_v10m'].notna()
    wind_speed = pd.Series(index=master_df.index, dtype=float)
    wind_speed[valid_mask] = np.sqrt(
        master_df.loc[valid_mask, 'wind_u10m']**2 + 
        master_df.loc[valid_mask, 'wind_v10m']**2
    )
    master_df['wind_speed'] = wind_speed
    print("   ✅ Wind Speed (từ U10M + V10M)")
    print(f"      → Giữ nguyên missing nếu U10M hoặc V10M missing")

# 4.4. Wind Direction từ U10M + V10M
if 'wind_u10m' in master_df.columns and 'wind_v10m' in master_df.columns:
    # CHỈ tính nếu cả U và V đều có giá trị
    valid_mask = master_df['wind_u10m'].notna() & master_df['wind_v10m'].notna()
    wind_direction = pd.Series(index=master_df.index, dtype=float)
    # atan2(U, V) cho wind direction (0° = North, 90° = East)
    wind_direction[valid_mask] = np.degrees(
        np.arctan2(
            master_df.loc[valid_mask, 'wind_u10m'],
            master_df.loc[valid_mask, 'wind_v10m']
        )
    )
    # Chuyển về 0-360
    wind_direction[valid_mask] = (wind_direction[valid_mask] + 360) % 360
    master_df['wind_direction'] = wind_direction
    print("   ✅ Wind Direction (từ U10M + V10M)")
    print(f"      → Giữ nguyên missing nếu U10M hoặc V10M missing")

# ============================================================================
# 5. TÍNH TIME-BASED FEATURES
# ============================================================================

print("\n📅 Bước 4: Tính Time-based Features...")

master_df['year'] = master_df['date'].dt.year
master_df['month'] = master_df['date'].dt.month
master_df['day'] = master_df['date'].dt.day
master_df['weekday'] = master_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
master_df['day_of_year'] = master_df['date'].dt.dayofyear
master_df['is_weekend'] = (master_df['weekday'] >= 5).astype(int)

# Season (Northern Hemisphere)
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

master_df['season'] = master_df['month'].apply(get_season)
print("   ✅ Year, Month, Day, Weekday, Day_of_year, Is_weekend, Season")

# ============================================================================
# 6. XỬ LÝ MISSING VALUES & CLEANUP
# ============================================================================

print("\n🧹 Bước 5: Xử lý Missing Values...")

# Đếm missing values trước
missing_before = master_df.isnull().sum().sum()
print(f"   Missing values trước: {missing_before:,}")

# Đảm bảo data được sắp xếp theo date (QUAN TRỌNG cho time series!)
master_df = master_df.sort_values('date').reset_index(drop=True)

# LƯU FILE TRƯỚC KHI XỬ LÝ MISSING VALUES
print("\n💾 Lưu file trước khi xử lý missing values...")
file_before_missing = os.path.join(OUTPUT_DIR, "processed_aqi_dataset_before_missing_fill.csv")
master_df.to_csv(file_before_missing, index=False)
print(f"   ✅ Đã lưu: {file_before_missing}")

# ============================================================================
# THỐNG KÊ MISSING VALUES TRƯỚC KHI XỬ LÝ
# ============================================================================

print("\n📋 DANH SÁCH FEATURES:")
missing_stats = master_df.isnull().sum()
total_rows = len(master_df)

for idx, (feature, missing_count) in enumerate(missing_stats.items(), 1):
    missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
    print(f"   {idx:3d}. {feature:25s} - Missing: {missing_count:6,} ({missing_pct:5.1f}%)")

print(f"\n   📊 Tổng số features: {len(master_df.columns)}")
print(f"   📊 Tổng số missing values: {missing_before:,}")
print(f"   📊 Tỷ lệ missing: {(missing_before / (total_rows * len(master_df.columns)) * 100):.2f}% tổng số cells")

# ============================================================================
# CHIẾN LƯỢC XỬ LÝ MISSING VALUES THEO TỪNG LOẠI FEATURE
# ============================================================================

def fill_missing_values(df):
    """
    Xử lý missing values với các phương pháp phù hợp cho từng loại feature
    """
    df = df.copy()
    original_missing = df.isnull().sum().sum()
    
    # 1. POLLUTANTS: Time series interpolation + forward/backward fill
    pollutant_features = ['PM2_5', 'PM10', 'NO2_trop', 'NO2_total', 'SO2_column', 
                         'SO2_surface', 'O3', 'CO', 'AOD_550']
    available_pollutants = [f for f in pollutant_features if f in df.columns]
    
    print("\n   📊 XỬ LÝ POLLUTANTS:")
    
    # Định nghĩa phạm vi hợp lý cho từng pollutant (để clip sau khi fill)
    pollutant_ranges = {
        'PM2_5': (0, 1000),  # μg/m³
        'PM10': (0, 2000),   # μg/m³
        'NO2_trop': (0, 1e16),  # molecules/cm²
        'NO2_total': (0, 1e16),
        'SO2_column': (0, 1),  # DU
        'SO2_surface': (0, 1e-3),  # kg/m³
        'O3': (0, 500),  # DU
        'CO': (0, 500),  # ppbv
        'AOD_550': (0, 5)  # unitless
    }
    
    for feature in available_pollutants:
        missing_count = df[feature].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            print(f"      {feature}: {missing_count:,} missing ({missing_pct:.1f}%)")
            
            # Phương pháp: Interpolation (linear) cho time series - ƯU TIÊN HÀNG ĐẦU
            # Interpolation tạo giá trị thay đổi dần, không fill bằng cùng một giá trị
            df[feature] = df[feature].interpolate(method='linear', limit_direction='both', limit=30)
            
            # Nếu vẫn còn missing ở đầu/cuối, dùng forward/backward fill với limit nhỏ
            # Chỉ fill 1-2 giá trị ở biên để tránh fill nhiều giá trị bằng nhau
            df[feature] = df[feature].ffill(limit=2).bfill(limit=2)
            
            # QUAN TRỌNG: Clip về phạm vi hợp lý sau khi fill
            if feature in pollutant_ranges:
                min_val, max_val = pollutant_ranges[feature]
                # Chỉ clip các giá trị ngoài phạm vi hợp lý
                invalid_mask = (df[feature] < min_val) | (df[feature] > max_val)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    print(f"         ⚠️  Phát hiện {invalid_count:,} giá trị ngoài phạm vi hợp lý [{min_val:.2e}, {max_val:.2e}]")
                    # Thay thế giá trị không hợp lý bằng NaN, sau đó fill lại
                    df.loc[invalid_mask, feature] = np.nan
                    # Fill lại bằng interpolation (không dùng ffill/bfill nhiều)
                    df[feature] = df[feature].interpolate(method='linear', limit_direction='both', limit=30)
                    df[feature] = df[feature].ffill(limit=2).bfill(limit=2)
                    # Clip về phạm vi hợp lý
                    df[feature] = df[feature].clip(lower=min_val, upper=max_val)
                    print(f"         ✅ Đã clip về phạm vi hợp lý")
            
            filled = original_missing - df.isnull().sum().sum()
            if filled > 0:
                print(f"         → Đã fill: {missing_count:,} missing values")
    
    # 2. WEATHER FEATURES: Interpolation + seasonal average
    weather_features = ['temperature_max', 'temperature_mean', 'temperature_min', 
                       'skin_temperature', 'dew_point', 'wind_u10m', 'wind_v10m']
    available_weather = [f for f in weather_features if f in df.columns]
    
    print("\n   🌡️ XỬ LÝ WEATHER FEATURES:")
    for feature in available_weather:
        missing_count = df[feature].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            # ƯU TIÊN: Interpolation cho weather (tạo giá trị thay đổi dần)
            if missing_pct < 10:  # Nếu missing < 10%
                df[feature] = df[feature].interpolate(method='linear', limit_direction='both')
                df[feature] = df[feature].ffill(limit=2).bfill(limit=2)
                print(f"      ✅ {feature}: Đã fill bằng interpolation ({missing_pct:.1f}% missing)")
            else:
                # Nếu missing nhiều, dùng interpolation với seasonal trend
                # Thay vì fill bằng cùng một giá trị, dùng interpolation với seasonal adjustment
                df['month'] = df['date'].dt.month
                # Tính seasonal average để làm baseline
                seasonal_avg = df.groupby('month')[feature].mean()
                # Thay vì fill bằng cùng một giá trị, dùng interpolation với seasonal baseline
                # Tạo một series với seasonal baseline
                seasonal_baseline = df['month'].map(seasonal_avg)
                # Interpolate phần missing
                df[feature] = df[feature].interpolate(method='linear', limit_direction='both', limit=30)
                # Chỉ fill các missing còn lại bằng seasonal baseline (không phải tất cả)
                # Nhưng thêm noise nhỏ để tránh các giá trị bằng nhau hoàn toàn
                missing_mask = df[feature].isna()
                if missing_mask.sum() > 0:
                    # Tính std của feature để thêm noise hợp lý
                    feature_std = df[feature].std()
                    if pd.notna(feature_std) and feature_std > 0:
                        noise_scale = feature_std * 0.05  # 5% noise
                        noise = np.random.normal(0, noise_scale, size=missing_mask.sum())
                        df.loc[missing_mask, feature] = seasonal_baseline[missing_mask] + noise
                    else:
                        df.loc[missing_mask, feature] = seasonal_baseline[missing_mask]
                print(f"      ✅ {feature}: Đã fill bằng interpolation + seasonal baseline ({missing_pct:.1f}% missing)")
    
    # 3. DERIVED FEATURES: Tính lại nếu có thể, hoặc fill
    derived_features = ['humidity', 'wind_speed', 'wind_direction', 'PM10']
    print("\n   🔄 XỬ LÝ DERIVED FEATURES:")
    
    # PM10: Tính lại từ PM2.5 nếu có thể
    if 'PM10' in df.columns and 'PM2_5' in df.columns:
        pm10_missing = df['PM10'].isnull().sum()
        if pm10_missing > 0:
            # Tính lại PM10 từ PM2.5 nếu PM2.5 có giá trị
            valid_pm25 = df['PM2_5'].notna()
            df.loc[valid_pm25 & df['PM10'].isna(), 'PM10'] = df.loc[valid_pm25 & df['PM10'].isna(), 'PM2_5'] * 1.6
            print(f"      ✅ PM10: Tính lại từ PM2.5")
    
    # Humidity: Tính lại nếu có temperature và dew_point
    if 'humidity' in df.columns and 'temperature_mean' in df.columns and 'dew_point' in df.columns:
        humidity_missing = df['humidity'].isnull().sum()
        if humidity_missing > 0:
            # Tính lại RH từ T và Td
            valid_mask = df['temperature_mean'].notna() & df['dew_point'].notna()
            T = df['temperature_mean'] - 273.15
            Td = df['dew_point'] - 273.15
            RH_new = 100 * (
                6.112 * np.exp((17.27 * Td) / (Td + 237.3)) /
                (6.112 * np.exp((17.27 * T) / (T + 237.3)))
            )
            df.loc[valid_mask & df['humidity'].isna(), 'humidity'] = RH_new[valid_mask & df['humidity'].isna()]
            df['humidity'] = np.clip(df['humidity'], 0, 100)
            print(f"      ✅ humidity: Tính lại từ T và Td")
    
    # Wind speed/direction: Tính lại từ U10M và V10M
    if 'wind_speed' in df.columns and 'wind_u10m' in df.columns and 'wind_v10m' in df.columns:
        wind_speed_missing = df['wind_speed'].isnull().sum()
        if wind_speed_missing > 0:
            valid_mask = df['wind_u10m'].notna() & df['wind_v10m'].notna()
            wind_speed_new = np.sqrt(df['wind_u10m']**2 + df['wind_v10m']**2)
            df.loc[valid_mask & df['wind_speed'].isna(), 'wind_speed'] = wind_speed_new[valid_mask & df['wind_speed'].isna()]
            print(f"      ✅ wind_speed: Tính lại từ U10M và V10M")
    
    if 'wind_direction' in df.columns and 'wind_u10m' in df.columns and 'wind_v10m' in df.columns:
        wind_dir_missing = df['wind_direction'].isnull().sum()
        if wind_dir_missing > 0:
            valid_mask = df['wind_u10m'].notna() & df['wind_v10m'].notna()
            wind_dir_new = np.degrees(np.arctan2(df['wind_u10m'], df['wind_v10m']))
            wind_dir_new = (wind_dir_new + 360) % 360
            df.loc[valid_mask & df['wind_direction'].isna(), 'wind_direction'] = wind_dir_new[valid_mask & df['wind_direction'].isna()]
            print(f"      ✅ wind_direction: Tính lại từ U10M và V10M")
    
    # Sau khi tính lại, fill các missing còn lại bằng interpolation
    for feature in derived_features:
        if feature in df.columns:
            missing_count = df[feature].isnull().sum()
            if missing_count > 0 and missing_count < len(df) * 0.1:  # Nếu missing < 10%
                df[feature] = df[feature].interpolate(method='linear', limit_direction='both')
                df[feature] = df[feature].ffill(limit=2).bfill(limit=2)
    
    # Xóa cột month tạm thời nếu đã tạo
    if 'month' in df.columns and 'month' not in [col for col in df.columns if col == 'month']:
        # Giữ lại month vì nó là feature chính
        pass
    
    final_missing = df.isnull().sum().sum()
    filled_count = original_missing - final_missing
    
    print(f"\n   📊 TỔNG KẾT:")
    print(f"      - Missing trước: {original_missing:,}")
    print(f"      - Missing sau: {final_missing:,}")
    print(f"      - Đã fill: {filled_count:,} values ({filled_count/original_missing*100:.1f}%)")
    
    return df

# Áp dụng xử lý missing values
master_df = fill_missing_values(master_df)

missing_after = master_df.isnull().sum().sum()
print(f"\n   ✅ Missing values sau xử lý: {missing_after:,}")

# ============================================================================
# 7. SẮP XẾP VÀ CHUẨN HÓA TÊN COLUMNS
# ============================================================================

print("\n📋 Bước 6: Sắp xếp và chuẩn hóa columns...")

# Sắp xếp columns: date, time features, pollution, weather, derived
column_order = [
    'date', 'year', 'month', 'day', 'weekday', 'day_of_year', 'is_weekend', 'season',
    'AOD_550',
    'PM2_5', 'PM10',
    'NO2_trop', 'NO2_total',
    'SO2_column', 'SO2_surface',
    'O3', 'CO',
    'temperature_max', 'temperature_mean', 'temperature_min', 'skin_temperature',
    'dew_point', 'humidity',
    'wind_u10m', 'wind_v10m', 'wind_speed', 'wind_direction',
    'precipitation'
]

# Chỉ lấy columns có trong master_df
column_order = [col for col in column_order if col in master_df.columns]
# Thêm các columns còn lại
remaining_cols = [col for col in master_df.columns if col not in column_order]
final_columns = column_order + remaining_cols

master_df = master_df[final_columns]

# ============================================================================
# 7. TẠO LAG FEATURES VÀ ROLLING FEATURES
# ============================================================================

print("\n🔄 Bước 7: Tạo Lag Features và Rolling Features...")

# Đảm bảo data được sắp xếp theo date (QUAN TRỌNG!)
master_df = master_df.sort_values('date').reset_index(drop=True)

# Danh sách features cần tạo lag và rolling
pollutant_features = ['PM2_5', 'PM10', 'NO2_trop', 'NO2_total', 'SO2_column', 'O3', 'CO', 'AOD_550']
weather_features = ['temperature_mean', 'humidity', 'wind_speed', 'precipitation']

# Chỉ lấy features có trong dataset
pollutant_features = [f for f in pollutant_features if f in master_df.columns]
weather_features = [f for f in weather_features if f in master_df.columns]

all_lag_features = pollutant_features  # Chỉ tạo lag/rolling cho pollutants (bao gồm AOD)

print(f"   📊 Tạo lag và rolling cho {len(all_lag_features)} features:")
print(f"      - Pollutants: {len(pollutant_features)} features")
print(f"      - Weather: {len(weather_features)} features")

# ============================================================================
# 7.1. TẠO LAG FEATURES (1, 7 ngày trước)
# ============================================================================

lag_periods = [1, 7]
lag_count = 0

for feature in all_lag_features:
    for lag in lag_periods:
        lag_col_name = f"{feature}_lag{lag}"
        master_df[lag_col_name] = master_df[feature].shift(lag)
        lag_count += 1

print(f"   ✅ Đã tạo {lag_count} lag features (lag 1, 2, 3, 7 ngày)")

# ============================================================================
# 7.2. TẠO ROLLING STATISTICS (7, 14 ngày)
# ============================================================================

rolling_windows = [7, 14]
rolling_stats = ['mean', 'std']
rolling_count = 0

for feature in all_lag_features:
    for window in rolling_windows:
        for stat in rolling_stats:
            rolling_col_name = f"{feature}_rolling{window}_{stat}"
            
            if stat == 'mean':
                master_df[rolling_col_name] = master_df[feature].rolling(window=window, min_periods=1).mean()
            elif stat == 'std':
                master_df[rolling_col_name] = master_df[feature].rolling(window=window, min_periods=1).std()
            elif stat == 'min':
                master_df[rolling_col_name] = master_df[feature].rolling(window=window, min_periods=1).min()
            elif stat == 'max':
                master_df[rolling_col_name] = master_df[feature].rolling(window=window, min_periods=1).max()
            
            rolling_count += 1

print(f"   ✅ Đã tạo {rolling_count} rolling features (window 3, 7, 14 ngày)")

# ============================================================================
# 7.3. XỬ LÝ MISSING VALUES CHO LAG/ROLLING FEATURES
# ============================================================================

print("\n   🔄 XỬ LÝ MISSING VALUES CHO LAG/ROLLING FEATURES:")

# Lag features: Các giá trị đầu tiên sẽ là NaN (do không có data quá khứ)
# → Fill bằng forward fill từ feature gốc (nếu feature gốc đã được fill)
lag_features_missing = 0
for feature in all_lag_features:
    for lag in lag_periods:
        lag_col = f"{feature}_lag{lag}"
        if lag_col in master_df.columns:
            missing_count = master_df[lag_col].isnull().sum()
            if missing_count > 0:
                lag_features_missing += missing_count
                # Fill bằng forward fill từ feature gốc (nếu feature gốc có giá trị)
                master_df[lag_col] = master_df[lag_col].fillna(master_df[feature].shift(-lag))
                # Nếu vẫn còn missing ở đầu, dùng forward fill với limit nhỏ
                master_df[lag_col] = master_df[lag_col].ffill(limit=min(lag, 2))

if lag_features_missing > 0:
    print(f"      ✅ Đã fill {lag_features_missing:,} missing values trong lag features")

# Rolling features: Đã dùng min_periods=1 nên sẽ có giá trị từ đầu
# Tuy nhiên, nếu feature gốc có missing, rolling cũng có thể có missing
rolling_features_missing = 0
for feature in all_lag_features:
    for window in rolling_windows:
        for stat in rolling_stats:
            rolling_col = f"{feature}_rolling{window}_{stat}"
            if rolling_col in master_df.columns:
                missing_count = master_df[rolling_col].isnull().sum()
                if missing_count > 0:
                    rolling_features_missing += missing_count
                    # Fill bằng interpolation thay vì ffill để tránh fill bằng cùng một giá trị
                    master_df[rolling_col] = master_df[rolling_col].interpolate(method='linear', limit_direction='both', limit=window)
                    # Chỉ ffill/bfill ở biên với limit nhỏ
                    master_df[rolling_col] = master_df[rolling_col].ffill(limit=min(window, 2)).bfill(limit=min(window, 2))

if rolling_features_missing > 0:
    print(f"      ✅ Đã fill {rolling_features_missing:,} missing values trong rolling features")

# Final check: Fill các missing còn lại bằng interpolation cho tất cả features
print("\n   🔄 XỬ LÝ MISSING VALUES CUỐI CÙNG (interpolation cho tất cả):")
final_missing_before = master_df.isnull().sum().sum()

# Chỉ fill các features số (không fill date, year, month, etc.)
numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col != 'date':  # Không fill date
        missing_count = master_df[col].isnull().sum()
        if missing_count > 0 and missing_count < len(master_df) * 0.5:  # Nếu missing < 50%
            # Interpolation cho time series (ưu tiên để tránh fill bằng cùng một giá trị)
            master_df[col] = master_df[col].interpolate(method='linear', limit_direction='both', limit=30)
            # Forward/backward fill với limit nhỏ (chỉ fill biên)
            master_df[col] = master_df[col].ffill(limit=2).bfill(limit=2)

final_missing_after = master_df.isnull().sum().sum()
filled_final = final_missing_before - final_missing_after

print(f"      ✅ Đã fill thêm {filled_final:,} missing values")
total_cells = len(master_df) * len(numeric_cols) if len(numeric_cols) > 0 else len(master_df) * len(master_df.columns)
missing_pct = (final_missing_after / total_cells * 100) if total_cells > 0 else 0
print(f"      📊 Missing cuối cùng: {final_missing_after:,} ({missing_pct:.2f}% tổng số cells)")

# ============================================================================
# 8. TÍNH AQI (THEO CHUẨN EPA - 6 POLLUTANTS)
# ============================================================================

print("\n🧮 Bước 8: Tính AQI từ các pollutants...")

def calc_aqi_component(conc, breakpoints):
    """Tính AQI cho một pollutant"""
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
    
    # Nếu vượt quá ngưỡng cao nhất
    return 500

# Breakpoints cho các pollutants (US EPA)
pm25_breaks = [(0, 0), (12.0, 50), (35.4, 100), (55.4, 150), (150.4, 200), (250.4, 300), (350.4, 400), (500.4, 500)]
pm10_breaks = [(0, 0), (54, 50), (154, 100), (254, 150), (354, 200), (424, 300), (504, 400), (604, 500)]
no2_breaks = [(0, 0), (53, 50), (100, 100), (360, 150), (649, 200), (1249, 300), (1649, 400), (2049, 500)]
so2_breaks = [(0, 0), (35, 50), (75, 100), (185, 150), (304, 200), (604, 300), (804, 400), (1004, 500)]
co_breaks = [(0, 0), (4.4, 50), (9.4, 100), (12.4, 150), (15.4, 200), (30.4, 300), (40.4, 400), (50.4, 500)]
o3_breaks = [(0, 0), (54, 50), (70, 100), (85, 150), (105, 200), (200, 300), (300, 400), (400, 500)]

# Tính AQI cho từng pollutant (THEO CHUẨN EPA - 6 pollutants)
# PM2.5 và PM10: đã là μg/m³ (surface concentration)
aqi_pm25 = master_df['PM2_5'].apply(lambda x: calc_aqi_component(x, pm25_breaks))
aqi_pm10 = master_df['PM10'].apply(lambda x: calc_aqi_component(x, pm10_breaks))

# CO: đã là ppbv (surface concentration), convert sang ppm
co_ppm = master_df['CO'] / 1000  # ppbv to ppm
aqi_co = co_ppm.apply(lambda x: calc_aqi_component(x, co_breaks))

# NO2: Convert từ column amount (molecules/cm²) sang surface ppb
# Công thức: NO2 surface ppb ≈ NO2_column * 2.67e-16
no2_scaling = 2.67e-16
no2_ppb = master_df['NO2_trop'] * no2_scaling
no2_ppb = no2_ppb.clip(0, 5000)  # Clip về giá trị hợp lý
aqi_no2 = no2_ppb.apply(lambda x: calc_aqi_component(x, no2_breaks))

# SO2: Convert từ column amount (DU) sang surface ppb
# Công thức: SO2 surface ppb ≈ SO2_column * 0.1 * 1000
so2_scaling = 0.1 * 1000  # Convert DU sang ppb
so2_ppb = master_df['SO2_column'] * so2_scaling
so2_ppb = so2_ppb.clip(0, 5000)  # Clip về giá trị hợp lý
aqi_so2 = so2_ppb.apply(lambda x: calc_aqi_component(x, so2_breaks))

# O3: Convert từ column amount (DU) sang surface ppb
# Công thức: O3 surface ppb ≈ O3_DU * 0.2
o3_scaling = 0.2
o3_ppb = master_df['O3'] * o3_scaling
o3_ppb = o3_ppb.clip(0, 1000)  # Clip về giá trị hợp lý
aqi_o3 = o3_ppb.apply(lambda x: calc_aqi_component(x, o3_breaks))

# AQI tổng = max của tất cả các AQI thành phần (6 pollutants theo chuẩn EPA)
aqi_components = pd.DataFrame({
    'PM2.5': aqi_pm25,
    'PM10': aqi_pm10,
    'NO2': aqi_no2,
    'SO2': aqi_so2,
    'CO': aqi_co,
    'O3': aqi_o3
})

master_df['aqi'] = aqi_components.max(axis=1)

# Phân loại AQI
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

master_df['aqi_category'] = master_df['aqi'].apply(categorize_aqi)

print(f"   ✅ Đã tính AQI: Min={master_df['aqi'].min():.1f}, Max={master_df['aqi'].max():.1f}, Mean={master_df['aqi'].mean():.1f}")
print(f"   📊 AQI categories: {master_df['aqi_category'].value_counts().to_dict()}")

# ============================================================================
# 9. LƯU FILE CUỐI CÙNG
# ============================================================================

print("\n💾 Bước 9: Lưu file cuối cùng...")

# Lưu file sau khi xử lý missing values
output_file = os.path.join(OUTPUT_DIR, "processed_aqi_dataset.csv")
master_df.to_csv(output_file, index=False)
print(f"   ✅ Đã lưu file sau khi xử lý missing values: {output_file}")

print(f"\n✅ HOÀN THÀNH!")
print(f"\n📁 CÁC FILE ĐÃ TẠO:")
print(f"   📂 Thư mục output: {OUTPUT_DIR}/")
print(f"   1. {os.path.basename(file_before_missing)} - Dữ liệu TRƯỚC khi xử lý missing values")
print(f"   2. {os.path.basename(output_file)} - Dữ liệu SAU khi xử lý missing values")
print(f"\n📊 Tổng số records: {len(master_df):,}")
print(f"📊 Tổng số features: {len(master_df.columns)}")
print(f"   - Features gốc: {len(final_columns)}")
print(f"   - Lag features: {lag_count}")
print(f"   - Rolling features: {rolling_count}")
print(f"   - AQI: 1 feature (đã tính từ 6 pollutants)")
print(f"📅 Date range: {master_df['date'].min()} đến {master_df['date'].max()}")

# In thống kê
print("\n📊 THỐNG KÊ:")
print(f"   - Records có đủ data: {master_df.notna().all(axis=1).sum():,}")
print(f"   - Records có missing: {master_df.isnull().any(axis=1).sum():,}")

# In danh sách features với lý do missing
print("\n📋 DANH SÁCH FEATURES:")
for i, col in enumerate(master_df.columns, 1):
    missing_count = master_df[col].isnull().sum()
    missing_pct = (missing_count / len(master_df)) * 100
    
    # Lấy lý do missing nếu có
    reason = missing_reasons.get(col, '')
    if missing_count > 0 and reason:
        print(f"   {i:2d}. {col:25s} - Missing: {missing_count:5,} ({missing_pct:5.1f}%)")
        print(f"       → {reason}")
    else:
        print(f"   {i:2d}. {col:25s} - Missing: {missing_count:5,} ({missing_pct:5.1f}%)")

print("\n" + "="*80)
print("🎉 TIỀN XỬ LÝ DỮ LIỆU HOÀN TẤT!")
print("="*80)


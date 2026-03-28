#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Visualization cho dữ liệu AQI thực tế (2004-2025)
Đầu vào: processed_aqi_dataset.csv (đã chuẩn hóa daily, giữ dữ liệu thật)
Đầu ra: Thư mục visualizations/ với các biểu đồ PNG/HTML
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Fix encoding cho Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình hiển thị tiếng Việt và style
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AQIDataVisualization:
    def __init__(self, csv_path: str = None, output_dir: str = "visualizations"):
        # Đặt đường dẫn mặc định từ thư mục processed
        if csv_path is None:
            csv_path = os.path.join("processed", "processed_aqi_dataset.csv")
        
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.data: pd.DataFrame | None = None
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✅ Đã tạo thư mục: {self.output_dir}")

    def load_data(self) -> bool:
        print("📊 ĐANG TẢI DỮ LIỆU CHO VISUALIZATION...")
        print(f"   📂 Đường dẫn: {self.csv_path}")
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(self.csv_path):
            print(f"❌ Không tìm thấy file: {self.csv_path}")
            print(f"💡 Vui lòng chạy tien_xu_ly_du_lieu.py trước để tạo file dữ liệu đã xử lý")
            return False
        
        try:
            self.data = pd.read_csv(self.csv_path)
            self.data['date'] = pd.to_datetime(self.data['date'])

            # Bảo toàn dữ liệu thật: không fill các cột gốc; chỉ chuẩn hóa cột trợ giúp
            # Tạo cột season_name (nếu season là số 0..3)
            if 'season' in self.data.columns and self.data['season'].dtype != 'O':
                season_map = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Fall'}
                self.data['season_name'] = self.data['season'].map(season_map).fillna(self.data['season'].astype(str))
            else:
                self.data['season_name'] = self.data.get('season', pd.Series(index=self.data.index, dtype='object'))

            # Nếu chưa có AQI, tính AQI giống data_analysis.py
            if 'aqi' not in self.data.columns:
                self._calculate_aqi_inplace()

            # Nếu chưa có aqi_category/pollution_level thì thêm từ aqi
            if 'aqi_category' not in self.data.columns:
                self.data['aqi_category'] = self.data['aqi'].apply(self._categorize_aqi)
            if 'pollution_level' not in self.data.columns:
                self.data['pollution_level'] = self.data['aqi'].apply(self._pollution_level)

            print(f"✅ Đã tải {len(self.data):,} bản ghi | {self.data['date'].min().date()} → {self.data['date'].max().date()}")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu: {e}")
            return False

    # ==== AQI helpers (đồng bộ với data_analysis.py) ====
    @staticmethod
    def _categorize_aqi(aqi: float) -> str:
        if pd.isna(aqi):
            return 'Unknown'
        if aqi <= 50:
            return 'Good'
        if aqi <= 100:
            return 'Moderate'
        if aqi <= 150:
            return 'Unhealthy for Sensitive'
        if aqi <= 200:
            return 'Unhealthy'
        if aqi <= 300:
            return 'Very Unhealthy'
        return 'Hazardous'

    @staticmethod
    def _pollution_level(aqi: float) -> str:
        if pd.isna(aqi):
            return 'Unknown'
        if aqi <= 50:
            return 'Low'
        if aqi <= 100:
            return 'Moderate'
        if aqi <= 150:
            return 'High'
        return 'Very High'

    def _calculate_aqi_inplace(self) -> None:
        """Kiểm tra AQI - AQI đã được tính ở bước tiền xử lý"""
        df = self.data
        
        # Kiểm tra xem đã có cột AQI chưa
        if 'aqi' not in df.columns:
            raise ValueError("❌ LỖI: Không tìm thấy cột 'aqi' trong dataset. "
                           "Vui lòng chạy tien_xu_ly_du_lieu.py trước để tính AQI.")
        
        print("✅ AQI đã có sẵn trong dataset (từ bước tiền xử lý)")
        print(f"✅ AQI: Min={df['aqi'].min():.1f}, Max={df['aqi'].max():.1f}, Mean={df['aqi'].mean():.1f}")

    # ==== Visualization functions - CHỌN LỌC BIỂU ĐỒ Ý NGHĨA NHẤT ====
    
    def plot_1_aqi_trend_over_time(self) -> None:
        """BIỂU ĐỒ 1: XU HƯỚNG AQI THEO THỜI GIAN (2004-2025)
        Phục vụ Query 1: Chất lượng không khí có cải thiện qua các năm không?"""
        print("📈 BIỂU ĐỒ 1: Xu hướng AQI theo thời gian...")
        df = self.data.copy()
        
        # Tính AQI trung bình theo năm
        yearly = df.groupby('year')['aqi'].agg(['mean', 'max', 'min']).reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(yearly['year'], yearly['mean'], label='AQI trung bình', 
                color='#2E86C1', marker='o', linewidth=2, markersize=6)
        ax.fill_between(yearly['year'], yearly['min'], yearly['max'], 
                       color='#AED6F1', alpha=0.3, label='Khoảng (min-max)')
        
        # Highlight xu hướng
        first_year = yearly['year'].iloc[0]
        last_year = yearly['year'].iloc[-1]
        first_aqi = yearly['mean'].iloc[0]
        last_aqi = yearly['mean'].iloc[-1]
        
        ax.scatter([first_year, last_year], [first_aqi, last_aqi], 
                  color='red', s=100, zorder=5)
        ax.plot([first_year, last_year], [first_aqi, last_aqi], 
               'r--', linewidth=2, alpha=0.7, label='Xu hướng tổng thể')
        
        ax.set_title('BIỂU ĐỒ 1: XU HƯỚNG AQI THEO THỜI GIAN (2004-2025)', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Năm', fontsize=12)
        ax.set_ylabel('AQI', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/1_aqi_trend_over_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # In giải thích ra console
        change = last_aqi - first_aqi
        change_pct = (change / first_aqi) * 100
        print(f"   💡 Ý NGHĨA: Năm {first_year}: AQI = {first_aqi:.1f} → Năm {last_year}: AQI = {last_aqi:.1f}")
        print(f"      Thay đổi: {change:+.1f} điểm ({change_pct:+.1f}%) - {'XẤU ĐI' if change > 0 else 'CẢI THIỆN'}")
        print("   ✅ Đã lưu: 1_aqi_trend_over_time.png")

    def plot_2_pollutant_importance(self) -> None:
        """BIỂU ĐỒ 2: POLLUTANT QUAN TRỌNG NHẤT
        Phục vụ Query 2: Chất ô nhiễm nào ảnh hưởng nhiều nhất đến AQI?"""
        print("📊 BIỂU ĐỒ 2: Pollutant quan trọng nhất...")
        df = self.data.copy()
        
        # Tính tương quan giữa pollutants và AQI
        pollutants = ['PM2_5', 'PM10', 'NO2_trop', 'SO2_column', 'CO', 'O3']
        available_pollutants = [p for p in pollutants if p in df.columns]
        
        correlations = {}
        for pol in available_pollutants:
            corr = df[[pol, 'aqi']].corr().iloc[0, 1]
            if not pd.isna(corr):
                correlations[pol] = abs(corr)
        
        if not correlations:
            print("   ⚠️ Không có dữ liệu để vẽ")
            return
        
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#E74C3C' if v > 0.5 else '#F39C12' if v > 0.3 else '#3498DB' 
                 for _, v in sorted_corr]
        bars = ax.barh([p[0] for p in sorted_corr], [p[1] for p in sorted_corr], 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Thêm giá trị trên thanh
        for i, (pol, corr) in enumerate(sorted_corr):
            ax.text(corr + 0.01, i, f'{corr:.3f}', 
                   va='center', fontweight='bold', fontsize=10)
        
        ax.set_title('BIỂU ĐỒ 2: POLLUTANT QUAN TRỌNG NHẤT (Tương quan với AQI)', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('|Tương quan| với AQI', fontsize=12)
        ax.set_ylabel('Pollutant', fontsize=12)
        ax.set_xlim(0, max([v for _, v in sorted_corr]) * 1.2)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/2_pollutant_importance.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # In giải thích ra console
        most_important = sorted_corr[0][0]
        print(f"   💡 Ý NGHĨA: Pollutant quan trọng nhất = {most_important} (tương quan: {sorted_corr[0][1]:.3f})")
        print("   ✅ Đã lưu: 2_pollutant_importance.png")

    def plot_3_seasonal_patterns(self) -> None:
        """BIỂU ĐỒ 3: PHÂN BỐ AQI THEO MÙA/THÁNG
        Phục vụ Query 4: Mùa nào/tháng nào ô nhiễm nhất?"""
        print("🍂 BIỂU ĐỒ 3: Phân bố AQI theo mùa và tháng...")
        df = self.data.copy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Biểu đồ 3a: Theo mùa
        season_names = {0: 'Đông\n(Dec-Feb)', 1: 'Xuân\n(Mar-May)', 
                       2: 'Hè\n(Jun-Aug)', 3: 'Thu\n(Sep-Nov)'}
        seasonal_aqi = df.groupby('season')['aqi'].mean().sort_index()
        
        colors_season = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
        bars1 = ax1.bar(range(len(seasonal_aqi)), seasonal_aqi.values, 
                       color=colors_season, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(seasonal_aqi)))
        ax1.set_xticklabels([season_names.get(i, f'Mùa {i}') for i in seasonal_aqi.index])
        ax1.set_title('AQI trung bình theo MÙA', fontweight='bold', fontsize=12)
        ax1.set_ylabel('AQI', fontsize=11)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Thêm giá trị trên thanh
        for i, (bar, val) in enumerate(zip(bars1, seasonal_aqi.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', fontweight='bold', fontsize=10)
        
        worst_season_idx = seasonal_aqi.idxmax()
        best_season_idx = seasonal_aqi.idxmin()
        
        # Biểu đồ 3b: Theo tháng
        monthly_aqi = df.groupby('month')['aqi'].mean().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        colors_month = ['#E74C3C' if m == monthly_aqi.idxmax() else 
                       '#2ECC71' if m == monthly_aqi.idxmin() else '#3498DB'
                       for m in monthly_aqi.index]
        bars2 = ax2.bar(range(len(monthly_aqi)), monthly_aqi.values,
                       color=colors_month, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(monthly_aqi)))
        ax2.set_xticklabels(month_names, rotation=45, ha='right')
        ax2.set_title('AQI trung bình theo THÁNG', fontweight='bold', fontsize=12)
        ax2.set_ylabel('AQI', fontsize=11)
        ax2.grid(True, axis='y', alpha=0.3)
        
        worst_month = monthly_aqi.idxmax()
        best_month = monthly_aqi.idxmin()
        
        fig.suptitle('BIỂU ĐỒ 3: PHÂN BỐ AQI THEO MÙA VÀ THÁNG', 
                    fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/3_seasonal_patterns.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # In giải thích ra console
        print(f"   💡 Ý NGHĨA: Mùa ô nhiễm nhất = {season_names[worst_season_idx]} (AQI = {seasonal_aqi[worst_season_idx]:.1f})")
        print(f"      Mùa sạch nhất = {season_names[best_season_idx]} (AQI = {seasonal_aqi[best_season_idx]:.1f})")
        print(f"      Tháng ô nhiễm nhất = {month_names[int(worst_month)-1]} (AQI = {monthly_aqi[worst_month]:.1f})")
        print(f"      Tháng sạch nhất = {month_names[int(best_month)-1]} (AQI = {monthly_aqi[best_month]:.1f})")
        print("   ✅ Đã lưu: 3_seasonal_patterns.png")

    def plot_4_weather_impact(self) -> None:
        """BIỂU ĐỒ 4: ẢNH HƯỞNG CỦA THỜI TIẾT LÊN AQI
        Phục vụ Query 3: Nhiệt độ, độ ẩm, gió ảnh hưởng đến AQI như thế nào?"""
        print("🌡️ BIỂU ĐỒ 4: Ảnh hưởng của thời tiết lên AQI...")
        df = self.data.copy()
        
        weather_vars = ['temperature_mean', 'humidity', 'wind_speed']
        available_weather = [w for w in weather_vars if w in df.columns]
        
        if len(available_weather) < 2:
            print("   ⚠️ Không đủ dữ liệu thời tiết")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        correlations = {}
        for i, (ax, weather) in enumerate(zip(axes, available_weather)):
            # Tính tương quan
            corr = df[[weather, 'aqi']].corr().iloc[0, 1]
            correlations[weather] = corr
            
            # Vẽ scatter plot
            ax.scatter(df[weather], df['aqi'], alpha=0.3, s=10, color='#3498DB')
            
            # Vẽ trend line
            z = np.polyfit(df[weather].dropna(), df.loc[df[weather].dropna().index, 'aqi'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[weather].min(), df[weather].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label=f'Trend (r={corr:.3f})')
            
            # Labels
            weather_labels = {
                'temperature_mean': 'Nhiệt độ trung bình (K)',
                'humidity': 'Độ ẩm (%)',
                'wind_speed': 'Tốc độ gió (m/s)'
            }
            ax.set_xlabel(weather_labels.get(weather, weather), fontsize=11)
            ax.set_ylabel('AQI', fontsize=11)
            ax.set_title(f'{weather_labels.get(weather, weather)}\nTương quan: {corr:+.3f}', 
                        fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        fig.suptitle('BIỂU ĐỒ 4: ẢNH HƯỞNG CỦA THỜI TIẾT LÊN AQI', 
                    fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/4_weather_impact.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # In giải thích ra console
        corr_str = ', '.join([f'{k}: {v:+.3f}' for k, v in correlations.items()])
        print(f"   💡 Ý NGHĨA: Tương quan với AQI - {corr_str}")
        print("   ✅ Đã lưu: 4_weather_impact.png")

    def plot_5_pollutant_correlations(self) -> None:
        """BIỂU ĐỒ 5: TƯƠNG QUAN GIỮA CÁC POLLUTANTS
        Phục vụ Query 5: Các chất ô nhiễm có xuất hiện cùng nhau không?"""
        print("🔗 BIỂU ĐỒ 5: Tương quan giữa các pollutants...")
        df = self.data.copy()
        
        pollutants = ['PM2_5', 'PM10', 'NO2_trop', 'NO2_total', 'SO2_column', 
                     'SO2_surface', 'O3', 'CO', 'AOD_550']
        available_pollutants = [p for p in pollutants if p in df.columns]
        
        if len(available_pollutants) < 3:
            print("   ⚠️ Không đủ dữ liệu pollutants")
            return
        
        corr = df[available_pollutants].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Chỉ hiển thị nửa dưới
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax, annot_kws={'fontsize': 9})
        
        ax.set_title('BIỂU ĐỒ 5: MA TRẬN TƯƠNG QUAN GIỮA CÁC POLLUTANTS', 
                    fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/5_pollutant_correlations.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Tìm các cặp tương quan mạnh và in ra console
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], val))
        
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if high_corr_pairs:
            top_pairs = high_corr_pairs[:5]
            print(f"   💡 Ý NGHĨA: Tìm thấy {len(high_corr_pairs)} cặp tương quan mạnh (|r| > 0.5)")
            for p1, p2, c in top_pairs[:3]:
                print(f"      - {p1} ↔ {p2}: {c:+.3f}")
        else:
            print("   💡 Ý NGHĨA: Không có cặp tương quan mạnh (|r| > 0.5)")
        print("   ✅ Đã lưu: 5_pollutant_correlations.png")

    def plot_6_pollutants_timeseries(self) -> None:
        """BIỂU ĐỒ 6: XU HƯỚNG CÁC POLLUTANTS THEO THỜI GIAN (6 pollutants EPA)"""
        print("📊 BIỂU ĐỒ 6: Xu hướng các pollutants theo thời gian...")
        df = self.data.copy()
        
        # 6 pollutants theo chuẩn EPA với thông tin đơn vị
        pollutants = {
            'PM2_5': {'unit': 'μg/m³', 'label': 'PM2.5', 'color': '#E74C3C'},
            'PM10': {'unit': 'μg/m³', 'label': 'PM10', 'color': '#3498DB'},
            'NO2_trop': {'unit': 'molecules/cm²', 'label': 'NO2', 'color': '#2ECC71', 'scale': 1e-15, 'scale_label': '(×10^15)'},
            'CO': {'unit': 'ppbv', 'label': 'CO', 'color': '#F39C12'},
            'O3': {'unit': 'DU', 'label': 'O3', 'color': '#9B59B6'},
            'SO2_column': {'unit': 'DU', 'label': 'SO2', 'color': '#E67E22'}
        }
        
        available_pollutants = {k: v for k, v in pollutants.items() if k in df.columns}
        
        if len(available_pollutants) < 2:
            print("   ⚠️ Không đủ dữ liệu pollutants")
            return
        
        # Tính trung bình theo năm
        yearly_pollutants = df.groupby('year')[list(available_pollutants.keys())].mean()
        
        # Tạo subplot: 2 hàng, 3 cột
        n_pols = len(available_pollutants)
        n_cols = 3
        n_rows = (n_pols + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (pol, info) in enumerate(available_pollutants.items()):
            ax = axes[idx]
            
            # Lấy dữ liệu và scale nếu cần
            data = yearly_pollutants[pol].values
            years = yearly_pollutants.index.values
            
            if 'scale' in info:
                data = data * info['scale']
                ylabel = f"{info['label']} ({info['unit']} {info.get('scale_label', '')})"
            else:
                ylabel = f"{info['label']} ({info['unit']})"
            
            # Vẽ line chart
            ax.plot(years, data, marker='o', linewidth=2, markersize=5, 
                   color=info['color'], label=info['label'])
            
            ax.set_title(f"{info['label']} ({info['unit']})", fontweight='bold', fontsize=11)
            ax.set_xlabel('Năm', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            
            # Format x-axis years
            ax.set_xticks(years[::2] if len(years) > 10 else years)
            ax.tick_params(axis='x', rotation=45)
        
        fig.suptitle('BIỂU ĐỒ 6: XU HƯỚNG CÁC POLLUTANTS THEO THỜI GIAN (6 POLLUTANTS EPA)', 
                    fontweight='bold', fontsize=14)
        
        # Ẩn các subplot không dùng
        for idx in range(len(available_pollutants), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/6_pollutants_timeseries.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # In thống kê xu hướng
        print(f"   💡 Ý NGHĨA: Xu hướng của {len(available_pollutants)} pollutants (EPA) theo năm:")
        for pol, info in available_pollutants.items():
            first_val = yearly_pollutants[pol].iloc[0]
            last_val = yearly_pollutants[pol].iloc[-1]
            change = last_val - first_val
            change_pct = (change / first_val * 100) if first_val > 0 else 0
            
            if 'scale' in info:
                first_display = first_val * info['scale']
                last_display = last_val * info['scale']
                change_display = change * info['scale']
                print(f"      - {info['label']}: {yearly_pollutants.index[0]} = {first_display:.2f} → "
                      f"{yearly_pollutants.index[-1]} = {last_display:.2f} ({change_pct:+.1f}%)")
            elif pol == 'SO2_column':
                print(f"      - {info['label']}: {yearly_pollutants.index[0]} = {first_val:.3f} → "
                      f"{yearly_pollutants.index[-1]} = {last_val:.3f} ({change_pct:+.1f}%)")
            else:
                print(f"      - {info['label']}: {yearly_pollutants.index[0]} = {first_val:.1f} → "
                      f"{yearly_pollutants.index[-1]} = {last_val:.1f} ({change_pct:+.1f}%)")
        print("   ✅ Đã lưu: 6_pollutants_timeseries.png")

    def plot_7_aqi_distribution(self) -> None:
        """BIỂU ĐỒ 7: PHÂN BỐ AQI THEO CATEGORY"""
        print("📊 BIỂU ĐỒ 7: Phân bố AQI theo category...")
        df = self.data.copy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram với categories
        categories = df['aqi_category'].value_counts().sort_index()
        colors_map = {
            'Good': '#2ECC71',
            'Moderate': '#F39C12',
            'Unhealthy for Sensitive': '#E67E22',
            'Unhealthy': '#E74C3C',
            'Very Unhealthy': '#8E44AD',
            'Hazardous': '#7F8C8D'
        }
        bar_colors = [colors_map.get(cat, '#3498DB') for cat in categories.index]
        
        ax1.bar(categories.index, categories.values, color=bar_colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('Số ngày theo AQI Category', fontweight='bold', fontsize=12)
        ax1.set_xlabel('AQI Category', fontsize=11)
        ax1.set_ylabel('Số ngày', fontsize=11)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Thêm giá trị trên thanh
        for i, (cat, count) in enumerate(categories.items()):
            ax1.text(i, count + 50, f'{count:,}', ha='center', fontweight='bold', fontsize=9)
        
        # Boxplot theo category
        category_order = ['Good', 'Moderate', 'Unhealthy for Sensitive', 
                         'Unhealthy', 'Very Unhealthy', 'Hazardous']
        existing_categories = [c for c in category_order if c in df['aqi_category'].values]
        
        if existing_categories:
            data_for_box = [df[df['aqi_category'] == cat]['aqi'].values for cat in existing_categories]
            bp = ax2.boxplot(data_for_box, labels=existing_categories, patch_artist=True)
            
            for patch, cat in zip(bp['boxes'], existing_categories):
                patch.set_facecolor(colors_map.get(cat, '#3498DB'))
                patch.set_alpha(0.7)
            
            ax2.set_title('Phân bố AQI theo Category (Boxplot)', fontweight='bold', fontsize=12)
            ax2.set_xlabel('AQI Category', fontsize=11)
            ax2.set_ylabel('AQI', fontsize=11)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, axis='y', alpha=0.3)
        
        fig.suptitle('BIỂU ĐỒ 7: PHÂN BỐ AQI THEO CATEGORY', 
                    fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/7_aqi_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # In thông tin về các categories
        print(f"   💡 Ý NGHĨA: Phân bố AQI:")
        for cat, count in categories.items():
            print(f"      - {cat}: {count:,} ngày ({count/len(df)*100:.1f}%)")
        print("   ✅ Đã lưu: 7_aqi_distribution.png")

    def plot_8_calendar_heatmap(self) -> None:
        """BIỂU ĐỒ 8: CALENDAR HEATMAP AQI THEO THÁNG/NĂM"""
        print("📅 BIỂU ĐỒ 8: Calendar heatmap AQI theo tháng/năm...")
        df = self.data.copy()
        
        # Tính AQI trung bình theo tháng/năm
        monthly_aqi = df.groupby(['year', 'month'])['aqi'].mean().reset_index()
        pivot_table = monthly_aqi.pivot(index='year', columns='month', values='aqi')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        # Điều chỉnh color scale để phản ánh đúng mức độ ô nhiễm
        # Với dải giá trị thực tế: Min=43, Max=160, Mean=67
        # Sử dụng color scale phù hợp với breakpoints AQI: 0-50 (Good), 51-100 (Moderate), 101-150 (Unhealthy for Sensitive), 151-200 (Unhealthy)
        aqi_min = df['aqi'].min()
        aqi_max = df['aqi'].max()
        aqi_mean = df['aqi'].mean()
        
        # Sử dụng color scale từ 0 đến 200 (phù hợp với giá trị max thực tế)
        # Center ở 75 (giữa Moderate range) để phân biệt rõ Good và Moderate
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                   center=75, vmin=0, vmax=200, cbar_kws={'label': 'AQI'},
                   linewidths=0.5, ax=ax, annot_kws={'fontsize': 8})
        
        ax.set_title('BIỂU ĐỒ 8: CALENDAR HEATMAP - AQI TRUNG BÌNH THEO THÁNG/NĂM', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Tháng', fontsize=12)
        ax.set_ylabel('Năm', fontsize=12)
        
        # Đổi nhãn tháng
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/8_calendar_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        worst_year_month = monthly_aqi.loc[monthly_aqi['aqi'].idxmax()]
        best_year_month = monthly_aqi.loc[monthly_aqi['aqi'].idxmin()]
        print(f"   💡 Ý NGHĨA: Tháng ô nhiễm nhất = {int(worst_year_month['year'])}/{int(worst_year_month['month'])} (AQI = {worst_year_month['aqi']:.1f})")
        print(f"      Tháng sạch nhất = {int(best_year_month['year'])}/{int(best_year_month['month'])} (AQI = {best_year_month['aqi']:.1f})")
        print("   ✅ Đã lưu: 8_calendar_heatmap.png")

    def plot_9_pm25_vs_pm10(self) -> None:
        """BIỂU ĐỒ 9: PM2.5 vs PM10 VỚI AQI COLORING"""
        print("🔍 BIỂU ĐỒ 9: PM2.5 vs PM10 với AQI coloring...")
        df = self.data.copy()
        
        if 'PM2_5' not in df.columns or 'PM10' not in df.columns:
            print("   ⚠️ Không có dữ liệu PM2.5 hoặc PM10")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color theo AQI category
        scatter = ax.scatter(df['PM2_5'], df['PM10'], c=df['aqi'], 
                           cmap='RdYlGn_r', s=20, alpha=0.6, vmin=0, vmax=500)
        
        # Thêm đường tỷ lệ 1:1.6 (PM10 = PM2.5 * 1.6)
        max_pm25 = df['PM2_5'].max()
        x_line = np.linspace(0, max_pm25, 100)
        y_line = x_line * 1.6
        ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7, label='PM10 = PM2.5 × 1.6')
        
        ax.set_title('BIỂU ĐỒ 9: PM2.5 vs PM10 (Màu theo AQI)', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('PM2.5 (μg/m³)', fontsize=12)
        ax.set_ylabel('PM10 (μg/m³)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('AQI', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/9_pm25_vs_pm10.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        corr = df[['PM2_5', 'PM10']].corr().iloc[0, 1]
        print(f"   💡 Ý NGHĨA: Tương quan PM2.5 vs PM10 = {corr:.3f} (rất cao)")
        print("   ✅ Đã lưu: 9_pm25_vs_pm10.png")

    def plot_10_lag_features(self) -> None:
        """BIỂU ĐỒ 10: AQI HÔM NAY vs AQI HÔM QUA (LAG FEATURES)"""
        print("🔄 BIỂU ĐỒ 10: AQI hôm nay vs AQI hôm qua...")
        df = self.data.copy()
        
        # Tạo lag feature nếu chưa có
        if 'aqi_lag1' not in df.columns:
            df['aqi_lag1'] = df['aqi'].shift(1)
        
        # Xóa NaN
        df_lag = df[['aqi', 'aqi_lag1']].dropna()
        
        if len(df_lag) < 100:
            print("   ⚠️ Không đủ dữ liệu lag")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(df_lag['aqi_lag1'], df_lag['aqi'], alpha=0.3, s=10, color='#3498DB')
        
        # Vẽ đường y=x
        min_val = min(df_lag['aqi'].min(), df_lag['aqi_lag1'].min())
        max_val = max(df_lag['aqi'].max(), df_lag['aqi_lag1'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='y = x')
        
        # Vẽ trend line
        z = np.polyfit(df_lag['aqi_lag1'], df_lag['aqi'], 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], p([min_val, max_val]), 'g--', linewidth=2, alpha=0.7, 
               label=f'Trend (y = {z[0]:.2f}x + {z[1]:.2f})')
        
        corr = df_lag['aqi'].corr(df_lag['aqi_lag1'])
        
        ax.set_title('BIỂU ĐỒ 10: AQI HÔM NAY vs AQI HÔM QUA (LAG FEATURES)', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('AQI hôm qua (t-1)', fontsize=12)
        ax.set_ylabel('AQI hôm nay (t)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/10_lag_features.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   💡 Ý NGHĨA: Tương quan AQI(t) vs AQI(t-1) = {corr:.3f} (quán tính cao)")
        print("   ✅ Đã lưu: 10_lag_features.png")

    def plot_11_rolling_statistics(self) -> None:
        """BIỂU ĐỒ 11: ROLLING STATISTICS CỦA AQI"""
        print("📈 BIỂU ĐỒ 11: Rolling statistics của AQI...")
        df = self.data.copy()
        df = df.sort_values('date')
        
        # Tính rolling mean và std
        df['aqi_rolling7_mean'] = df['aqi'].rolling(window=7, min_periods=1).mean()
        df['aqi_rolling30_mean'] = df['aqi'].rolling(window=30, min_periods=1).mean()
        df['aqi_rolling7_std'] = df['aqi'].rolling(window=7, min_periods=1).std()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Time series với rolling mean
        ax1.plot(df['date'], df['aqi'], alpha=0.3, color='lightblue', linewidth=0.5, label='AQI daily')
        ax1.plot(df['date'], df['aqi_rolling7_mean'], color='#2ECC71', linewidth=2, label='Rolling 7 ngày')
        ax1.plot(df['date'], df['aqi_rolling30_mean'], color='#E74C3C', linewidth=2, label='Rolling 30 ngày')
        ax1.set_title('AQI Time Series với Rolling Mean', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Ngày', fontsize=11)
        ax1.set_ylabel('AQI', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Rolling std
        ax2.fill_between(df['date'], df['aqi_rolling7_mean'] - df['aqi_rolling7_std'],
                        df['aqi_rolling7_mean'] + df['aqi_rolling7_std'],
                        alpha=0.3, color='#3498DB', label='±1 STD (7 ngày)')
        ax2.plot(df['date'], df['aqi_rolling7_mean'], color='#2E86C1', linewidth=2, label='Rolling 7 ngày')
        ax2.set_title('Rolling Mean với Standard Deviation (7 ngày)', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Ngày', fontsize=11)
        ax2.set_ylabel('AQI', fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle('BIỂU ĐỒ 11: ROLLING STATISTICS CỦA AQI', 
                    fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/11_rolling_statistics.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   💡 Ý NGHĨA: Rolling mean giúp nhận diện xu hướng dài hạn, giảm nhiễu")
        print("   ✅ Đã lưu: 11_rolling_statistics.png")

    def plot_12_monthly_pollutants(self) -> None:
        """BIỂU ĐỒ 12: TRUNG BÌNH POLLUTANTS THEO THÁNG (6 pollutants EPA)"""
        print("📊 BIỂU ĐỒ 12: Trung bình pollutants theo tháng...")
        df = self.data.copy()
        
        # 6 pollutants theo chuẩn EPA
        pollutants = {
            'PM2_5': {'unit': 'μg/m³', 'label': 'PM2.5', 'color': '#E74C3C'},
            'PM10': {'unit': 'μg/m³', 'label': 'PM10', 'color': '#3498DB'},
            'NO2_trop': {'unit': 'molecules/cm²', 'label': 'NO2', 'color': '#2ECC71', 'scale': 1e-15, 'scale_label': '(×10^15)'},
            'CO': {'unit': 'ppbv', 'label': 'CO', 'color': '#F39C12'},
            'O3': {'unit': 'DU', 'label': 'O3', 'color': '#9B59B6'},
            'SO2_column': {'unit': 'DU', 'label': 'SO2', 'color': '#E67E22'}
        }
        
        available_pollutants = {k: v for k, v in pollutants.items() if k in df.columns}
        
        if len(available_pollutants) < 2:
            print("   ⚠️ Không đủ dữ liệu pollutants")
            return
        
        monthly_pollutants = df.groupby('month')[list(available_pollutants.keys())].mean()
        
        # Tạo subplot: 2 hàng, 3 cột
        n_pols = len(available_pollutants)
        n_cols = 3
        n_rows = (n_pols + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for idx, (pol, info) in enumerate(available_pollutants.items()):
            ax = axes[idx]
            
            # Lấy dữ liệu và scale nếu cần
            data = monthly_pollutants[pol].values
            if 'scale' in info:
                data = data * info['scale']
                ylabel = f"{info['label']} ({info['unit']} {info.get('scale_label', '')})"
            else:
                ylabel = f"{info['label']} ({info['unit']})"
            
            # Vẽ bar chart
            x = np.arange(len(month_names))
            bars = ax.bar(x, data, color=info['color'], edgecolor='black', linewidth=1, alpha=0.8)
            
            # Thêm giá trị trên mỗi bar
            for i, (bar, val) in enumerate(zip(bars, data)):
                height = bar.get_height()
                if 'scale' in info:
                    # Format cho NO2_trop (đã scale về 1e15)
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2f}',
                           ha='center', va='bottom', fontsize=8)
                else:
                    if val < 1:
                        # Format cho SO2 (giá trị < 1)
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.3f}',
                               ha='center', va='bottom', fontsize=8)
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.1f}',
                               ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f"{info['label']} ({info['unit']})", fontweight='bold', fontsize=11)
            ax.set_xlabel('Tháng', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(month_names, rotation=0, fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Tìm tháng cao nhất và thấp nhất
            max_idx = data.argmax()
            min_idx = data.argmin()
            bars[max_idx].set_alpha(1.0)
            bars[max_idx].set_edgecolor('red')
            bars[max_idx].set_linewidth(2)
        
        fig.suptitle('BIỂU ĐỒ 12: TRUNG BÌNH POLLUTANTS THEO THÁNG (6 POLLUTANTS EPA)', 
                    fontweight='bold', fontsize=14)
        
        # Ẩn các subplot không dùng
        for idx in range(len(available_pollutants), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/12_monthly_pollutants.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # In thống kê
        print(f"   💡 Ý NGHĨA: Pattern theo tháng của {len(available_pollutants)} pollutants (EPA)")
        for pol, info in available_pollutants.items():
            max_month = monthly_pollutants[pol].idxmax()
            min_month = monthly_pollutants[pol].idxmin()
            max_val = monthly_pollutants[pol].loc[max_month]
            min_val = monthly_pollutants[pol].loc[min_month]
            
            # Format số tùy theo loại pollutant
            if 'scale' in info:
                # NO2_trop: hiển thị với scale
                max_val_scaled = max_val * info['scale']
                min_val_scaled = min_val * info['scale']
                print(f"      - {info['label']}: Cao nhất = {month_names[max_month-1]} ({max_val_scaled:.2f}×10¹⁵), "
                      f"Thấp nhất = {month_names[min_month-1]} ({min_val_scaled:.2f}×10¹⁵)")
            elif pol == 'SO2_column':
                print(f"      - {info['label']}: Cao nhất = {month_names[max_month-1]} ({max_val:.3f}), "
                      f"Thấp nhất = {month_names[min_month-1]} ({min_val:.3f})")
            else:
                print(f"      - {info['label']}: Cao nhất = {month_names[max_month-1]} ({max_val:.1f}), "
                      f"Thấp nhất = {month_names[min_month-1]} ({min_val:.1f})")
        print("   ✅ Đã lưu: 12_monthly_pollutants.png")

    def plot_13_pollution_vs_normal_pie(self) -> None:
        """BIỂU ĐỒ 13: TỔNG SỐ NGÀY Ô NHIỄM vs NGÀY BÌNH THƯỜNG (PIE CHART)"""
        print("🥧 BIỂU ĐỒ 13: Tổng số ngày ô nhiễm vs ngày bình thường (Pie chart)...")
        df = self.data.copy()
        
        # Phân loại: Good/Moderate = Bình thường, còn lại = Ô nhiễm
        df['pollution_status'] = df['aqi_category'].apply(
            lambda x: 'Bình thường' if x in ['Good', 'Moderate'] else 'Ô nhiễm'
        )
        
        status_counts = df['pollution_status'].value_counts()
        
        colors = ['#2ECC71', '#E74C3C']  # Xanh = Bình thường, Đỏ = Ô nhiễm
        labels = ['Bình thường (Good/Moderate)', 'Ô nhiễm (Unhealthy trở lên)']
        
        total = status_counts.sum()
        percentages = status_counts / total * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tạo labels với số lượng và phần trăm
        pie_labels = []
        for i, (status, count) in enumerate(status_counts.items()):
            label = labels[i] if i < len(labels) else status
            pct = percentages[status]
            pie_labels.append(f'{label}\n({count:,} ngày - {pct:.1f}%)')
        
        wedges, texts, autotexts = ax.pie(status_counts.values, labels=pie_labels, 
                                         colors=colors, autopct='', 
                                         startangle=90, textprops={'fontsize': 11})
        
        ax.set_title('BIỂU ĐỒ 13: TỔNG SỐ NGÀY Ô NHIỄM vs NGÀY BÌNH THƯỜNG (PIE CHART)', 
                    fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/13_pollution_vs_normal_pie.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        normal_days = status_counts.get('Bình thường', 0)
        pollution_days = status_counts.get('Ô nhiễm', 0)
        normal_pct = percentages.get('Bình thường', 0)
        pollution_pct = percentages.get('Ô nhiễm', 0)
        
        print(f"   💡 Ý NGHĨA: {normal_days:,} ngày bình thường ({normal_pct:.1f}%) vs {pollution_days:,} ngày ô nhiễm ({pollution_pct:.1f}%)")
        print("   ✅ Đã lưu: 13_pollution_vs_normal_pie.png")

    def run_all(self) -> None:
        """Chạy tất cả các biểu đồ quan trọng nhất"""
        print("\n" + "="*80)
        print("🎨 BẮT ĐẦU TẠO CÁC BIỂU ĐỒ Ý NGHĨA NHẤT")
        print("="*80)
        
        self.plot_1_aqi_trend_over_time()
        self.plot_2_pollutant_importance()
        self.plot_3_seasonal_patterns()
        self.plot_4_weather_impact()
        self.plot_5_pollutant_correlations()
        self.plot_6_pollutants_timeseries()
        self.plot_7_aqi_distribution()
        self.plot_8_calendar_heatmap()
        self.plot_9_pm25_vs_pm10()
        self.plot_10_lag_features()
        self.plot_11_rolling_statistics()
        self.plot_12_monthly_pollutants()
        self.plot_13_pollution_vs_normal_pie()
        
        print("\n" + "="*80)
        print("✅ ĐÃ HOÀN THÀNH TẤT CẢ CÁC BIỂU ĐỒ QUAN TRỌNG")
        print("="*80)


def main() -> None:
    print("🎨 BẮT ĐẦU TẠO DATA VISUALIZATION (AQI thực)")
    print("=" * 80)

    viz = AQIDataVisualization()
    if not viz.load_data():
        print("❌ Không thể tải dữ liệu, dừng.")
        return

    try:
        viz.run_all()

        # Báo cáo danh sách file xuất
        outputs = [
            '1_aqi_trend_over_time.png',
            '2_pollutant_importance.png',
            '3_seasonal_patterns.png',
            '4_weather_impact.png',
            '5_pollutant_correlations.png',
            '6_pollutants_timeseries.png',
            '7_aqi_distribution.png',
            '8_calendar_heatmap.png',
            '9_pm25_vs_pm10.png',
            '10_lag_features.png',
            '11_rolling_statistics.png',
            '12_monthly_pollutants.png',
            '13_pollution_vs_normal_pie.png'
        ]
        outputs = [os.path.join(viz.output_dir, f) for f in outputs]
        
        print("\n📁 CÁC BIỂU ĐỒ ĐÃ TẠO:")
        print("="*80)
        for i, f in enumerate(outputs, 1):
            if os.path.exists(f):
                size_kb = os.path.getsize(f) / 1024
                print(f"   {i}. {os.path.basename(f)} ({size_kb:.1f} KB)")
            else:
                print(f"   {i}. {os.path.basename(f)} (⚠️ Chưa tạo)")
        
        print("\n💡 LƯU Ý:")
        print("   - Mỗi biểu đồ đều có giải thích ý nghĩa in ra console")
        print("   - Các biểu đồ phục vụ phân tích và xây dựng mô hình ML")
        print("   - Tổng cộng 13 biểu đồ (12 biểu đồ thường + 1 biểu đồ tròn)")
        
        # Lưu metadata
        meta = {
            'records': len(viz.data),
            'date_min': str(viz.data['date'].min().date()),
            'date_max': str(viz.data['date'].max().date()),
            'total_plots': 13,
            'plots_description': {
                '1_aqi_trend_over_time.png': 'Xu hướng AQI theo thời gian (2004-2025)',
                '2_pollutant_importance.png': 'Pollutant quan trọng nhất (tương quan với AQI)',
                '3_seasonal_patterns.png': 'Phân bố AQI theo mùa và tháng',
                '4_weather_impact.png': 'Ảnh hưởng của thời tiết lên AQI',
                '5_pollutant_correlations.png': 'Ma trận tương quan giữa các pollutants',
                '6_pollutants_timeseries.png': 'Xu hướng các pollutants theo thời gian',
                '7_aqi_distribution.png': 'Phân bố AQI theo category',
                '8_calendar_heatmap.png': 'Calendar heatmap AQI theo tháng/năm',
                '9_pm25_vs_pm10.png': 'PM2.5 vs PM10 với AQI coloring',
                '10_lag_features.png': 'AQI hôm nay vs AQI hôm qua (lag features)',
                '11_rolling_statistics.png': 'Rolling statistics của AQI',
                '12_monthly_pollutants.png': 'Trung bình pollutants theo tháng',
                '13_pollution_vs_normal_pie.png': 'Tổng số ngày ô nhiễm vs ngày bình thường (Pie chart)'
            }
        }
        with open(os.path.join(viz.output_dir, 'viz_metadata.json'), 'w', encoding='utf-8') as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        print("\n✅ Đã lưu visualizations và metadata")

    except Exception as e:
        print(f"❌ Lỗi khi tạo biểu đồ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()



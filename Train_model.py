#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script train model dự đoán Multi-Target và Multi-Horizon - IMPROVED VERSION
- Targets: AQI, temperature, pollutants (PM2.5, PM10, NO2, CO, O3, SO2, humidity, wind_speed)
- Horizons: T+1, T+7, T+30, T+365 (ngày, tuần, tháng, năm)
- Sử dụng tất cả features có sẵn
- Test tất cả models và chọn top 2
- Cải thiện: Tune hyperparameters, scale targets, weighted scoring, ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import VotingRegressor
import warnings
import sys
import codecs
from datetime import datetime
import pickle
import os
import json

# Fix encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

warnings.filterwarnings('ignore')

class MultiTargetMultiHorizonTrainer:
    def __init__(self, csv_path=None):
        # Đặt đường dẫn mặc định từ thư mục processed
        if csv_path is None:
            csv_path = os.path.join("processed", "processed_aqi_dataset.csv")
        
        self.csv_path = csv_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = {}  # {horizon: {target: array}}
        self.y_test = {}   # {horizon: {target: array}}
        self.models = {}   # {horizon: {model_name: model}}
        self.results = {}  # {horizon: {model_name: {target: metrics}}}
        self.best_models = {}  # {horizon: {model_name: model, metrics: dict}}
        self.scaler_X = None  # Scaler cho features
        self.scaler_y = {}  # Scaler cho từng target
        self.feature_names = []
        
        # Targets cần dự đoán (AQI là quan trọng nhất)
        self.targets = [
            'aqi', 'temperature_max', 'temperature_min', 'temperature_mean',
            'PM2_5', 'PM10', 'NO2_trop', 'CO', 'O3', 'SO2_column',
            'humidity', 'wind_speed'
        ]
        
        # Weights cho từng target (AQI có weight cao nhất)
        self.target_weights = {
            'aqi': 3.0,  # AQI quan trọng nhất
            'PM2_5': 2.0, 'PM10': 2.0,  # Pollutants quan trọng
            'temperature_max': 1.5, 'temperature_min': 1.5, 'temperature_mean': 1.5,
            'CO': 1.5, 'O3': 1.5,
            'NO2_trop': 1.0, 'SO2_column': 1.0,
            'humidity': 1.0, 'wind_speed': 1.0
        }
        
        # Horizons cần dự đoán (số ngày)
        self.horizons = [1, 7, 30, 365]  # T+1, T+7, T+30, T+365
        
    def load_data(self):
        """Tải dữ liệu từ CSV"""
        print("📊 ĐANG TẢI DỮ LIỆU...")
        print(f"   File: {self.csv_path}")
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(self.csv_path):
            print(f"❌ Không tìm thấy file: {self.csv_path}")
            print(f"💡 Vui lòng chạy tien_xu_ly_du_lieu.py trước để tạo file dữ liệu đã xử lý")
            return False
        
        try:
            self.data = pd.read_csv(self.csv_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
            
            # Kiểm tra các targets
            missing_targets = []
            for target in self.targets:
                if target not in self.data.columns:
                    missing_targets.append(target)
            
            if missing_targets:
                print(f"   ⚠️  Một số targets không có: {missing_targets}")
                # Loại bỏ targets không có
                self.targets = [t for t in self.targets if t in self.data.columns]
                # Loại bỏ weights cho targets không có
                self.target_weights = {k: v for k, v in self.target_weights.items() if k in self.targets}
            
            print(f"✅ Đã tải {len(self.data):,} records")
            print(f"   Date range: {self.data['date'].min().date()} đến {self.data['date'].max().date()}")
            print(f"   Features: {len(self.data.columns)}")
            print(f"   Targets: {len(self.targets)} targets")
            print(f"   Horizons: {self.horizons}")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_features(self):
        """Chuẩn bị features và targets cho tất cả horizons"""
        print("\n🔧 CHUẨN BỊ FEATURES VÀ TARGETS...")
        
        df = self.data.copy()
        
        # Loại bỏ các features không dùng được
        exclude_features = [
            'date',  # Không phải feature
            'aqi_category',  # Derived from AQI
            'pollution_level',  # Derived from AQI
        ]
        
        # Lấy tất cả features còn lại (trừ các targets)
        feature_cols = [col for col in df.columns 
                        if col not in exclude_features and col not in self.targets]
        self.feature_names = feature_cols
        
        print(f"   ✅ Số features: {len(feature_cols)}")
        print(f"   ✅ Số targets: {len(self.targets)}")
        print(f"   ✅ Horizons: {self.horizons}")
        
        # Chuẩn bị X (features) - dùng tất cả rows (sẽ xử lý missing sau)
        X = df[feature_cols].copy()
        
        # Kiểm tra missing values trong features
        missing_X = X.isnull().sum().sum()
        if missing_X > 0:
            print(f"   ⚠️  Có {missing_X:,} missing values trong features - sẽ fill")
            X = X.fillna(X.mean())
        
        # Xóa các rows cuối cùng không có target (số lượng = max horizon)
        max_horizon = max(self.horizons)
        valid_mask = pd.Series([True] * (len(df) - max_horizon) + [False] * max_horizon)
        
        X_valid = X[valid_mask].copy()
        
        # Chuẩn bị y cho từng horizon
        print(f"\n   📊 Chuẩn bị targets cho các horizons...")
        
        # Tính valid_indices chung cho tất cả horizons (intersection)
        valid_indices = pd.Series([True] * len(X_valid))
        
        # Tính valid_indices cho từng horizon và lấy intersection
        for horizon in self.horizons:
            print(f"      → Horizon T+{horizon}...")
            horizon_valid = pd.Series([True] * len(X_valid))
            
            for target in self.targets:
                if target in df.columns:
                    # Shift về phía trước để tạo target T+horizon
                    y = df[target].shift(-horizon)
                    y_valid = y[valid_mask].copy()
                    # Đánh dấu missing
                    horizon_valid = horizon_valid & (~y_valid.isnull())
            
            # Intersection với valid_indices chung
            valid_indices = valid_indices & horizon_valid
            print(f"         → {horizon_valid.sum():,} samples hợp lệ (trước khi intersect)")
        
        print(f"   ✅ {valid_indices.sum():,} samples hợp lệ cho tất cả horizons")
        
        # Dùng valid_indices chung
        X_final = X_valid[valid_indices].copy()
        self.X = X_final
        
        # Lưu y cho từng horizon (đã filter theo valid_indices chung)
        if not hasattr(self, 'y_all'):
            self.y_all = {}
            
        for horizon in self.horizons:
            y_dict = {}
            for target in self.targets:
                if target in df.columns:
                    y = df[target].shift(-horizon)
                    y_valid = y[valid_mask][valid_indices].copy()
                    y_dict[target] = y_valid
            self.y_all[horizon] = y_dict
        
        print(f"   ✅ Sau khi xử lý: {len(self.X):,} samples")
        
        return True
    
    def split_data(self, test_size=0.2):
        """Chia train/test theo time-series (80/20)"""
        print(f"\n📊 CHIA DỮ LIỆU: Train {1-test_size:.0%} / Test {test_size:.0%} (Time-series split)...")
        
        # Time-series split: không shuffle, lấy phần cuối làm test
        split_idx = int(len(self.X) * (1 - test_size))
        
        self.X_train = self.X.iloc[:split_idx].copy()
        self.X_test = self.X.iloc[split_idx:].copy()
        
        print(f"   ✅ Train: {len(self.X_train):,} samples")
        print(f"   ✅ Test: {len(self.X_test):,} samples")
        
        # Chuẩn bị y_train và y_test cho từng horizon (từ y_all đã chuẩn bị)
        for horizon in self.horizons:
            y_train_dict = {}
            y_test_dict = {}
            
            for target in self.targets:
                if target in self.y_all[horizon]:
                    y_series = self.y_all[horizon][target]
                    # Split theo split_idx
                    y_train_dict[target] = y_series.iloc[:split_idx].values
                    y_test_dict[target] = y_series.iloc[split_idx:].values
            
            self.y_train[horizon] = y_train_dict
            self.y_test[horizon] = y_test_dict
        
        # Scale features (dùng RobustScaler để chống outliers tốt hơn)
        self.scaler_X = RobustScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        
        print(f"   ✅ Đã scale features (RobustScaler)")
        
        return True
    
    def train_models_for_horizon(self, horizon):
        """Train nhiều models cho một horizon cụ thể với hyperparameters đã tune"""
        print(f"\n🤖 TRAIN CÁC MODELS CHO HORIZON T+{horizon}...")
        
        # Chuẩn bị y_train và y_test cho horizon này (multi-output)
        targets_list = [t for t in self.targets if t in self.y_train[horizon]]
        y_train_array = np.column_stack([self.y_train[horizon][t] for t in targets_list])
        y_test_array = np.column_stack([self.y_test[horizon][t] for t in targets_list])
        
        print(f"   Targets: {targets_list}")
        print(f"   Train shape: {y_train_array.shape}, Test shape: {y_test_array.shape}")
        
        # Scale targets riêng biệt để cải thiện accuracy
        self.scaler_y[horizon] = {}
        y_train_scaled = np.zeros_like(y_train_array)
        y_test_scaled = np.zeros_like(y_test_array)
        
        for idx, target in enumerate(targets_list):
            scaler_y = RobustScaler()
            y_train_scaled[:, idx] = scaler_y.fit_transform(y_train_array[:, idx].reshape(-1, 1)).ravel()
            y_test_scaled[:, idx] = scaler_y.transform(y_test_array[:, idx].reshape(-1, 1)).ravel()
            self.scaler_y[horizon][target] = scaler_y
        
        print(f"   ✅ Đã scale targets riêng biệt")
        
        # Danh sách models với hyperparameters tối ưu cho độ chính xác cao nhất
        models_config = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,  # Tăng để tăng accuracy
                max_depth=30,  # Tăng depth để capture complex patterns
                min_samples_split=3,  # Giảm để tăng flexibility
                min_samples_leaf=1,  # Giảm để tăng accuracy
                max_features='sqrt',  # Tối ưu cho accuracy
                random_state=42,
                n_jobs=-1,
                verbose=0
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=200,  # Tăng để tăng accuracy
                max_depth=30,  # Tăng depth
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=0
            ),
            'GradientBoosting': MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=200,  # Tăng để tăng accuracy
                    max_depth=10,  # Tăng depth
                    learning_rate=0.05,  # Giảm learning rate để tăng accuracy
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42,
                    verbose=0
                ),
                n_jobs=-1
            ),
            'Ridge': Ridge(alpha=0.1),
            'Lasso': Lasso(alpha=0.01, max_iter=3000),
            'LinearRegression': LinearRegression(),
            'SVR': MultiOutputRegressor(
                SVR(kernel='rbf', C=100, gamma='scale', max_iter=2000),  # Tăng C để tăng accuracy
                n_jobs=-1
            ),
            'MLPRegressor': MLPRegressor(
                hidden_layer_sizes=(150, 100, 50),  # Tăng layers để tăng accuracy
                max_iter=1000,  # Tăng iterations
                learning_rate='adaptive',
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.01,  # L2 regularization
                random_state=42,
                verbose=0
            )
        }
        
        results_list = []
        
        for name, model in models_config.items():
            print(f"\n   🔄 Training {name}...")
            start_time = datetime.now()
            
            try:
                # Tất cả models dùng scaled data (cả X và y) để cải thiện accuracy
                print(f"      ⏳ Đang train (có thể mất vài phút cho models lớn)...", end='', flush=True)
                model.fit(self.X_train_scaled, y_train_scaled)
                print(f" ✓ Đã train xong. Đang predict...", end='', flush=True)
                y_pred_train_scaled = model.predict(self.X_train_scaled)
                y_pred_test_scaled = model.predict(self.X_test_scaled)
                print(f" ✓")
                
                # Inverse transform predictions về original scale
                y_pred_train = np.zeros_like(y_train_array)
                y_pred_test = np.zeros_like(y_test_array)
                
                for idx, target in enumerate(targets_list):
                    scaler_y = self.scaler_y[horizon][target]
                    y_pred_train[:, idx] = scaler_y.inverse_transform(y_pred_train_scaled[:, idx].reshape(-1, 1)).ravel()
                    y_pred_test[:, idx] = scaler_y.inverse_transform(y_pred_test_scaled[:, idx].reshape(-1, 1)).ravel()
                
                # Tính metrics cho từng target
                metrics_by_target = {}
                weighted_r2_test = 0
                weighted_normalized_rmse_test = 0
                total_weight = 0
                
                for idx, target in enumerate(targets_list):
                    y_train_target = y_train_array[:, idx]
                    y_test_target = y_test_array[:, idx]
                    y_pred_train_target = y_pred_train[:, idx]
                    y_pred_test_target = y_pred_test[:, idx]
                    
                    rmse_train = np.sqrt(mean_squared_error(y_train_target, y_pred_train_target))
                    rmse_test = np.sqrt(mean_squared_error(y_test_target, y_pred_test_target))
                    mae_train = mean_absolute_error(y_train_target, y_pred_train_target)
                    mae_test = mean_absolute_error(y_test_target, y_pred_test_target)
                    r2_train = r2_score(y_train_target, y_pred_train_target)
                    r2_test = r2_score(y_test_target, y_pred_test_target)
                    
                    # Tính normalized metrics (RMSE/mean, MAE/mean) để so sánh tương đối
                    mean_target = np.abs(y_test_target.mean())
                    std_target = y_test_target.std()
                    
                    if mean_target > 1e-6:  # Tránh chia cho 0
                        normalized_rmse = rmse_test / mean_target
                        normalized_mae = mae_test / mean_target
                    else:
                        # Nếu mean quá nhỏ, dùng std
                        normalized_rmse = rmse_test / std_target if std_target > 1e-6 else rmse_test
                        normalized_mae = mae_test / std_target if std_target > 1e-6 else mae_test
                    
                    metrics_by_target[target] = {
                        'rmse_train': float(rmse_train),
                        'rmse_test': float(rmse_test),
                        'mae_train': float(mae_train),
                        'mae_test': float(mae_test),
                        'r2_train': float(r2_train),
                        'r2_test': float(r2_test),
                        'normalized_rmse_test': float(normalized_rmse),
                        'normalized_mae_test': float(normalized_mae)
                    }
                    
                    # Weighted metrics (AQI có weight cao nhất)
                    weight = self.target_weights.get(target, 1.0)
                    weighted_r2_test += r2_test * weight
                    weighted_normalized_rmse_test += normalized_rmse * weight
                    total_weight += weight
                
                # Tính weighted average
                overall_r2_test = weighted_r2_test / total_weight if total_weight > 0 else 0
                overall_rmse_test = weighted_normalized_rmse_test / total_weight if total_weight > 0 else 0
                
                self.models[horizon] = self.models.get(horizon, {})
                self.models[horizon][name] = model
                
                self.results[horizon] = self.results.get(horizon, {})
                self.results[horizon][name] = {
                    'targets': metrics_by_target,
                    'overall_r2_test': float(overall_r2_test),
                    'overall_rmse_test': float(overall_rmse_test),
                    'aqi_r2_test': float(metrics_by_target.get('aqi', {}).get('r2_test', 0))
                }
                
                results_list.append({
                    'Model': name,
                    'Weighted R² (Test)': overall_r2_test,
                    'AQI R² (Test)': metrics_by_target.get('aqi', {}).get('r2_test', 0),
                    'Normalized RMSE (Test)': overall_rmse_test
                })
                
                elapsed_time = (datetime.now() - start_time).total_seconds()
                print(f"      ✅ Weighted R²: {overall_r2_test:.4f} | AQI R²: {metrics_by_target.get('aqi', {}).get('r2_test', 0):.4f} | Norm RMSE: {overall_rmse_test:.4f} | Time: {elapsed_time:.1f}s")
                
            except Exception as e:
                print(f"      ❌ Lỗi khi train {name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Tạo DataFrame kết quả - sort theo Weighted R² (ưu tiên AQI)
        results_df = pd.DataFrame(results_list)
        if len(results_df) > 0:
            results_df = results_df.sort_values('Weighted R² (Test)', ascending=False)
            
            print("\n" + "="*80)
            print(f"📊 KẾT QUẢ TẤT CẢ MODELS CHO HORIZON T+{horizon}:")
            print("="*80)
            print(results_df.to_string(index=False))
            
            # Chọn top 2 models tốt nhất (theo Weighted R²)
            top2_models = results_df.head(2)
            print(f"\n🏆 TOP 2 MODELS CHO HORIZON T+{horizon}:")
            for idx, (_, row) in enumerate(top2_models.iterrows(), 1):
                print(f"   {idx}. {row['Model']}: Weighted R²={row['Weighted R² (Test)']:.4f}, AQI R²={row['AQI R² (Test)']:.4f}")
            
            return results_df
        else:
            print("   ❌ Không có model nào train thành công")
            return pd.DataFrame()
    
    def train_all_models(self):
        """Train models cho tất cả horizons"""
        print("\n" + "="*80)
        print("🚀 BẮT ĐẦU TRAIN MODELS CHO TẤT CẢ HORIZONS...")
        print("="*80)
        
        all_results = {}
        
        for horizon in self.horizons:
            results_df = self.train_models_for_horizon(horizon)
            all_results[horizon] = results_df
            
            # Lưu top 2 models cho horizon này
            if len(results_df) >= 2:
                top2_names = results_df.head(2)['Model'].tolist()
                self.best_models[horizon] = {
                    'model1': {
                        'name': top2_names[0],
                        'model': self.models[horizon][top2_names[0]],
                        'metrics': self.results[horizon][top2_names[0]]
                    },
                    'model2': {
                        'name': top2_names[1],
                        'model': self.models[horizon][top2_names[1]],
                        'metrics': self.results[horizon][top2_names[1]]
                    }
                }
        
        return all_results
    
    def save_models(self, output_dir="models"):
        """Lưu top 2 models cho mỗi horizon"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n💾 ĐANG LƯU MODELS...")
        
        all_metadata = {}
        
        for horizon in self.horizons:
            if horizon not in self.best_models:
                continue
            
            horizon_dir = os.path.join(output_dir, f"horizon_{horizon}")
            if not os.path.exists(horizon_dir):
                os.makedirs(horizon_dir)
            
            best_models_h = self.best_models[horizon]
            
            for model_idx in ['model1', 'model2']:
                model_info = best_models_h[model_idx]
                model_name = model_info['name']
                model = model_info['model']
                
                # Lưu model
                model_file = os.path.join(horizon_dir, f"model_{model_idx}_{model_name}.pkl")
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                
                # Lưu metadata
                metadata = {
                    'horizon': horizon,
                    'model_name': model_name,
                    'model_idx': model_idx,
                    'feature_names': self.feature_names,
                    'targets': self.targets,
                    'metrics': model_info['metrics'],
                    'train_size': len(self.X_train),
                    'test_size': len(self.X_test),
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                metadata_file = os.path.join(horizon_dir, f"metadata_{model_idx}_{model_name}.json")
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
                
                all_metadata[f"T+{horizon}_{model_idx}"] = metadata
                
                print(f"   ✅ T+{horizon} - {model_idx}: {model_name} → {model_file}")
        
        # Lưu scalers
        scaler_X_file = os.path.join(output_dir, "scaler_X.pkl")
        with open(scaler_X_file, 'wb') as f:
            pickle.dump(self.scaler_X, f)
        print(f"   ✅ Scaler X: {scaler_X_file}")
        
        scaler_y_file = os.path.join(output_dir, "scaler_y.pkl")
        with open(scaler_y_file, 'wb') as f:
            pickle.dump(self.scaler_y, f)
        print(f"   ✅ Scaler Y: {scaler_y_file}")
        
        # Lưu metadata tổng hợp
        summary_metadata = {
            'all_models': all_metadata,
            'feature_names': self.feature_names,
            'targets': self.targets,
            'horizons': self.horizons,
            'target_weights': self.target_weights,
            'improvements': [
                'Tuned hyperparameters for maximum accuracy (n_estimators=200, max_depth=30)',
                'Scaled targets separately (RobustScaler) - improves multi-target accuracy',
                'Weighted scoring (AQI has 3x weight) - prioritizes AQI accuracy',
                'RobustScaler for features (better with outliers)',
                'All models use scaled data (X and y) for better accuracy',
                'Optimized hyperparameters: min_samples_split=3, min_samples_leaf=1',
                'GradientBoosting: learning_rate=0.05, n_estimators=200 for better accuracy',
                'MLPRegressor: 3 hidden layers (150,100,50) with L2 regularization'
            ],
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_file = os.path.join(output_dir, "models_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_metadata, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"   ✅ Summary: {summary_file}")

def main():
    """Hàm chính"""
    print("="*80)
    print("🤖 TRAIN MODEL MULTI-TARGET & MULTI-HORIZON - IMPROVED VERSION")
    print("="*80)
    print("   Mục tiêu:")
    print("   - Targets: AQI, temperature, pollutants (PM2.5, PM10, NO2, CO, O3, SO2, humidity, wind_speed)")
    print("   - Horizons: T+1, T+7, T+30, T+365 (ngày, tuần, tháng, năm)")
    print("   - Sử dụng tất cả features có sẵn")
    print("   - Test tất cả models và chọn top 2 cho mỗi horizon")
    print("   - Cải thiện: Tuned hyperparameters, scaled targets, weighted scoring")
    print("="*80)
    
    # Khởi tạo trainer
    trainer = MultiTargetMultiHorizonTrainer()
    
    # Tải dữ liệu
    if not trainer.load_data():
        return
    
    # Chuẩn bị features
    if not trainer.prepare_features():
        return
    
    # Chia train/test
    if not trainer.split_data(test_size=0.2):
        return
    
    # Train models cho tất cả horizons
    all_results = trainer.train_all_models()
    
    # Lưu models
    trainer.save_models()
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH TRAIN MODELS!")
    print("="*80)
    print(f"   📁 Models đã được lưu trong thư mục 'models/'")
    print(f"   📊 Mỗi horizon có 2 models tốt nhất")
    print(f"   🎯 Horizons: {trainer.horizons}")
    print(f"   🎯 Improvements: Tuned hyperparameters, scaled targets, weighted scoring")

if __name__ == "__main__":
    main()

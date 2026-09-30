import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    """
    資料型態轉換與特徵工程準備。
    """
    data = df.copy()
    
    # 1. 確保欄位名稱沒有前後空白
    data.columns = data.columns.str.strip()
    
    # 2. 自動相容原生 plate_x 或轉換後的 plate_x_2026
    if 'plate_x_2026' not in data.columns and 'plate_x' in data.columns:
        data['plate_x_2026'] = data['plate_x']
    if 'plate_z_2026' not in data.columns and 'plate_z' in data.columns:
        data['plate_z_2026'] = data['plate_z']
    
    # 3. 類別特徵轉換
    categorical_cols = ['umpire', 'stand', 'p_throws', 'fielder_2']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')
            
    # 4. 定義 Ground Truth (ABS 基準：本壘板半寬 8.5 吋 ≈ 0.708 ft)
    data['is_true_strike'] = (
        (data['plate_x_2026'].abs() <= 0.708) & 
        (data['plate_z_2026'] >= data['sz_bot']) & 
        (data['plate_z_2026'] <= data['sz_top'])
    )

    # 5. 定義裁判判決與誤判標籤
    data['is_called_strike'] = data['description'] == 'called_strike'
    data['is_miscall'] = (data['is_true_strike'] != data['is_called_strike']).astype(int)
    
    return data

def main():
    print("啟動方案 B：雙層堆疊 LightGBM 訓練管線 (Stacked Architecture)...")
    
    # 1. 載入資料並清除欄位多餘空白
    df = pd.read_csv('statcast_2025.csv')
    df.columns = df.columns.str.strip()
    
    # 確保日期格式解析正確
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # 2. 特徵前處理
    df = prepare_features(df)
    
    # 3. 時間軸切分
    df_2025 = df[df['game_date'].dt.year == 2025].copy()
    df_2026_H1 = df[(df['game_date'] >= '2026-03-26') & (df['game_date'] <= '2026-06-30')].copy()
    df_test = df[(df['game_date'] >= '2026-07-01') & (df['game_date'] <= '2026-07-31')].copy()
    
    print(f"資料切分完成 -> 2025 筆數: {len(df_2025)}, 2026 H1 筆數: {len(df_2026_H1)}, 2026 Jul 筆數: {len(df_test)}")

    if len(df_2025) == 0:
        raise ValueError("錯誤：df_2025 筆數為 0，請檢查 CSV 檔案內的 game_date 是否包含 2025 年資料。")

    # ==========================================
    # 4. Layer 1: 物理基底模型 (Model 1) 
    # ==========================================
    phys_features = [
        'plate_x_2026', 'plate_z_2026', 'sz_top', 'sz_bot', 
        'release_speed', 'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_z'
    ]
    
    print("\n[Layer 1] 訓練純物理模型 (Training Physics Model)...")
    model_1 = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.09,
        max_depth=5,
        num_leaves=1,
        random_state=42,
        n_jobs=-1
    )
    
    model_1.fit(df_2025[phys_features], df_2025['is_miscall'])
    joblib.dump(model_1, 'model_1_physics.pkl')
    
    # ==========================================
    # 5. 建立 Meta-Feature (堆疊特徵轉移)
    # ==========================================
    print("生成 Meta-Feature (P_physics)...")
    df_2026_H1['P_physics'] = model_1.predict_proba(df_2026_H1[phys_features])[:, 1]
    df_test['P_physics'] = model_1.predict_proba(df_test[phys_features])[:, 1]

    # ==========================================
    # 6. Layer 2: 賽局裁判模型 (Model 2) + 機率校準
    # ==========================================
    layer2_features = phys_features + ['P_physics', 'umpire', 'stand', 'p_throws', 'fielder_2']
    
    print("\n[Layer 2] 訓練裁判情境模型並進行機率校準 (Training & Calibrating Context Model)...")
    base_model_2 = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.09,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    calibrated_model_2 = CalibratedClassifierCV(
        estimator=base_model_2, 
        method='isotonic', 
        cv=5
    )
    
    calibrated_model_2.fit(df_2026_H1[layer2_features], df_2026_H1['is_miscall'])
    joblib.dump(calibrated_model_2, 'model_2_umpire_calibrated.pkl')
    
    # ==========================================
    # 7. Out-of-Time (OOT) 最終驗證 (2026 July)
    # ==========================================
    print("\n[Evaluation] 在 2026 年 7 月份資料上進行 OOT 驗證...")
    
    y_pred_prob = calibrated_model_2.predict_proba(df_test[layer2_features])[:, 1]
    y_true = df_test['is_miscall']
    
    brier = brier_score_loss(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    
    print(" [OOT 驗證結果]")
    print(f"AUC Score    : {auc:.4f} (模型區分誤判的能力)")
    print(f"Brier Score  : {brier:.4f} (機率絕對精準度，越接近 0 越好)")
    print(" 雙層模型訓練完成！")

if __name__ == "__main__":
    main()
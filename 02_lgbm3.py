import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pltfrom sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 準備階段：定義目標變數與特徵
# ==========================================
def prepare_data(df):
    data = df.copy()
    # 確保類別特徵被正確設定
    cat_cols = ['umpire', 'stand', 'p_throws', 'fielder_2']
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')
            
    # 定義絕對物理好壞球 (以 2026 年 2D 中板 8.5 吋為準)
    data['is_true_strike'] = (
        (data['plate_x_2026'].abs() <= 0.708) & 
        (data['plate_z_2026'] >= data['sz_bot']) & 
        (data['plate_z_2026'] <= data['sz_top'])
    )
    data['is_called_strike'] = data['description'] == 'called_strike'
    # Target: 挑戰會成功的誤判球
    data['is_miscall'] = (data['is_true_strike'] != data['is_called_strike']).astype(int)
    return data

# ==========================================
# 階段一：Optuna 自動調優 Layer 1 (物理模型)
# ==========================================
def objective_layer1(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**param, n_estimators=150, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        loss = log_loss(y_va, preds)
        cv_scores.append(loss)
        
    return np.mean(cv_scores)

# ==========================================
# 階段一之二：Optuna 自動調優 Layer 2 (裁判情境模型)
# ==========================================
def objective_layer2(trial, X, y):
    """
    這層的特徵包含了 P_physics 以及類別特徵 (umpire 等)
    """
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        # 為了實驗，我們把深度範圍拉寬，看 Optuna 會不會選擇較深的樹
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # 由於後續會做 Isotonic Calibration，這裡的 n_estimators 不用設太高
        model = lgb.LGBMClassifier(**param, n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        loss = log_loss(y_va, preds)
        cv_scores.append(loss)
        
    return np.mean(cv_scores)

# ==========================================
# 階段二：主程式 (雙模型堆疊與雙 Optuna)
# ==========================================
def main():
    print("啟動雙層全自動調優訓練管線 (Layer 1 Optuna + Layer 2 Optuna)...")
    
    # 讀取資料 (實戰請換成你的 pd.read_csv)
    df_2025 = prepare_data(pd.read_csv("tra_25.csv"))
    df_2026_H1 = prepare_data(pd.read_csv("statcast_2026H1_sha.csv"))
    df_2026_test = prepare_data(pd.read_csv("statcast_2026H2_sha.csv"))
    
    #df_2025 = pd.DataFrame() 
    #df_2026_H1 = pd.DataFrame()
    #df_2026_test = pd.DataFrame()
    
    phys_features = ['plate_x_2026', 'plate_z_2026', 'sz_top', 'sz_bot', 'release_speed', 'pfx_x', 'pfx_z']
    context_features = phys_features + ['P_physics', 'umpire', 'stand', 'p_throws', 'fielder_2']
    
    # ------------------------------------------
    # 1. 訓練 Layer 1 (物理基礎)
    # ------------------------------------------
    print("\n[Layer 1] 開始 Optuna 調參 (尋找最佳物理邊界)...")
    X_phys = df_2025[phys_features]
    y_phys = df_2025['is_miscall']
    
    study_1 = optuna.create_study(direction='minimize')
    study_1.optimize(lambda trial: objective_layer1(trial, X_phys, y_phys), n_trials=15)
    print(f"Layer 1 最佳參數: {study_1.best_params}")
    
    best_model_1 = lgb.LGBMClassifier(**study_1.best_params, n_estimators=150, random_state=42)
    best_model_1.fit(X_phys, y_phys)
    
    # ------------------------------------------
    # 2. 建立 Meta-Feature
    # ------------------------------------------
    print("\n[Meta-Feature] 讓 2026 年資料向 Model 1 取得純物理機率...")
    df_2026_H1['P_physics'] = best_model_1.predict_proba(df_2026_H1[phys_features])[:, 1]
    df_2026_test['P_physics'] = best_model_1.predict_proba(df_2026_test[phys_features])[:, 1]
    
    # ------------------------------------------
    # 3. 訓練 Layer 2 (裁判情境模型)
    # ------------------------------------------
    print("\n[Layer 2] 開始 Optuna 調參 (尋找裁判尺度微調最佳參數)...")
    X_ctx = df_2026_H1[context_features]
    y_ctx = df_2026_H1['is_miscall']
    
    # 建立第二個 Study 給 Layer 2 專用
    study_2 = optuna.create_study(direction='minimize')
    study_2.optimize(lambda trial: objective_layer2(trial, X_ctx, y_ctx), n_trials=15)
    print(f"Layer 2 最佳參數: {study_2.best_params}")
    
    # 用 Layer 2 的最佳參數建立 Base Model
    base_model_2 = lgb.LGBMClassifier(**study_2.best_params, n_estimators=100, random_state=42)
    
    print("\n[Calibration] 進行 Isotonic 機率校準...")
    calibrated_model_2 = CalibratedClassifierCV(base_model_2, method='isotonic', cv=5)
    calibrated_model_2.fit(X_ctx, y_ctx)
    
    # ------------------------------------------
    # 4. Out-of-Time 驗證 (2026 July)
    # ------------------------------------------
    print("\n[Evaluation] OOT 驗證 (2026 July)...")
    X_test = df_2026_test[context_features]
    y_test = df_2026_test['is_miscall']
    
    y_pred_prob = calibrated_model_2.predict_proba(X_test)[:, 1]
    
    print(f"Log Loss   : {log_loss(y_test, y_pred_prob):.4f}")
    print(f"Brier Score: {brier_score_loss(y_test, y_pred_prob):.4f}")
    print(f"AUC        : {roc_auc_score(y_test, y_pred_prob):.4f}")

if __name__ == "__main__":
    main()
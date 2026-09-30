import os
import math
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# 準備階段：定義目標變數與特徵
# ==========================================
def prepare_data(df):
    data = df.copy()
    cat_cols = ['umpire', 'stand', 'p_throws', 'fielder_2']
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')
            
    # 定義絕對物理好壞球 (以 2D 中板 8.5 吋為準)
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
# 輔助函式：球種家族映射 (Pitch Grouping)
# ==========================================
def map_pitch_group(pitch_type):
    """
    將 Statcast 球種代碼歸納為三大球種家族：
    - Fastballs: FF, SI, FC, FA
    - Breaking Balls: SL, ST, CU, KC, SV, CS
    - Offspeed: CH, FS, FO
    其他稀有球/非主流球種映射為 None
    """
    group_mapping = {
        # Fastballs
        'FF': 'Fastballs',
        'SI': 'Fastballs',
        'FC': 'Fastballs',
        'FA': 'Fastballs',
        
        # Breaking Balls
        'SL': 'Breaking Balls',
        'ST': 'Breaking Balls',
        'CU': 'Breaking Balls',
        'KC': 'Breaking Balls',
        'SV': 'Breaking Balls',
        'CS': 'Breaking Balls',
        
        # Offspeed
        'CH': 'Offspeed',
        'FS': 'Offspeed',
        'FO': 'Offspeed'
    }
    return group_mapping.get(pitch_type, None)

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
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
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
        
        model = lgb.LGBMClassifier(**param, n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        loss = log_loss(y_va, preds)
        cv_scores.append(loss)
        
    return np.mean(cv_scores)

# ==========================================
# 輔助函式：繪製多類別球種 Confusion Matrix
# ==========================================
def plot_multiclass_pitch_confusion_matrix(
    y_true_pitch, 
    y_pred_pitch, 
    output_image='pitch_confusion_matrix.png',
    output_excel='pitch_confusion_matrix.xlsx'
):
    y_true = pd.Series(y_true_pitch).dropna().astype(str).reset_index(drop=True)
    y_pred = pd.Series(y_pred_pitch).dropna().astype(str).reset_index(drop=True)

    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred.iloc[:min_len]

    labels = y_true.value_counts().index.tolist()
    for label in y_pred.unique():
        if label not in labels:
            labels.append(label)

    cm_df = pd.crosstab(
        y_true, 
        y_pred, 
        rownames=['Actual'], 
        colnames=['Predicted'], 
        dropna=False
    ).reindex(index=labels, columns=labels, fill_value=0)

    cm_df.to_excel(output_excel)
    print(f"球種混淆矩陣數據已輸出至: {output_excel}")

    fig_size = max(8, len(labels) * 0.85)
    plt.figure(figsize=(fig_size, fig_size * 0.85))

    ax = sns.heatmap(
        cm_df, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        linewidths=0.5,
        linecolor='lightgrey',
        square=True
    )

    ax.set_title('Pitch Type Confusion Matrix (Actual vs Predicted)', fontsize=14, pad=15)
    ax.set_xlabel('Predicted Pitch', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual Pitch', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"修正後的球種混淆矩陣圖表已輸出至: {output_image}")

# ==========================================
# 階段二：主程式
# ==========================================
def main():
    print("啟動雙層全自動調優訓練管線 (Layer 1 Optuna + Layer 2 Optuna)...")
    
    # 建立模型儲存資料夾
    output_model_dir = "saved_models"
    os.makedirs(output_model_dir, exist_ok=True)
    
    df_2025 = prepare_data(pd.read_csv("tra_25.csv"))
    df_2026_H1 = prepare_data(pd.read_csv("statcast_2026H1_sha.csv"))
    df_2026_test = prepare_data(pd.read_csv("statcast_2026H2_sha.csv"))
    
    phys_features = ['plate_x_2026', 'plate_z_2026', 'sz_top', 'sz_bot', 'release_speed', 'pfx_x', 'pfx_z']
    context_features = phys_features + ['P_physics', 'umpire', 'stand', 'p_throws', 'fielder_2']
    
    # 1. 訓練 Layer 1
    print("\n[Layer 1] 開始 Optuna 調參...")
    X_phys = df_2025[phys_features]
    y_phys = df_2025['is_miscall']
    
    study_1 = optuna.create_study(direction='minimize')
    study_1.optimize(lambda trial: objective_layer1(trial, X_phys, y_phys), n_trials=15)
    
    best_model_1 = lgb.LGBMClassifier(**study_1.best_params, n_estimators=150, random_state=42)
    best_model_1.fit(X_phys, y_phys)
    
    # 儲存 Layer 1 模型
    joblib.dump(best_model_1, os.path.join(output_model_dir, "layer1_physics_model.joblib"))
    print(f"Layer 1 物理模型已儲存至: {output_model_dir}/layer1_physics_model.joblib")
    
    # 2. 建立 Meta-Feature
    print("\n[Meta-Feature] 計算 P_physics...")
    df_2026_H1['P_physics'] = best_model_1.predict_proba(df_2026_H1[phys_features])[:, 1]
    df_2026_test['P_physics'] = best_model_1.predict_proba(df_2026_test[phys_features])[:, 1]
    
    # 3. 訓練 Layer 2
    print("\n[Layer 2] 開始 Optuna 調參...")
    X_ctx = df_2026_H1[context_features]
    y_ctx = df_2026_H1['is_miscall']
    
    study_2 = optuna.create_study(direction='minimize')
    study_2.optimize(lambda trial: objective_layer2(trial, X_ctx, y_ctx), n_trials=15)
    
    base_model_2 = lgb.LGBMClassifier(**study_2.best_params, n_estimators=100, random_state=42)
    
    print("\n[Calibration] 進行 Isotonic 機率校準...")
    calibrated_model_2 = CalibratedClassifierCV(base_model_2, method='isotonic', cv=5)
    calibrated_model_2.fit(X_ctx, y_ctx)
    
    # 儲存 Layer 2 校準模型
    joblib.dump(calibrated_model_2, os.path.join(output_model_dir, "layer2_calibrated_context_model.joblib"))
    print(f"Layer 2 情境校準模型已儲存至: {output_model_dir}/layer2_calibrated_context_model.joblib")
    
    # 4. Out-of-Time 驗證
    print("\n[Evaluation] OOT 驗證...")
    X_test = df_2026_test[context_features]
    y_test = df_2026_test['is_miscall']
    
    y_pred_prob = calibrated_model_2.predict_proba(X_test)[:, 1]
    
    print(f"Log Loss   : {log_loss(y_test, y_pred_prob):.4f}")
    print(f"Brier Score: {brier_score_loss(y_test, y_pred_prob):.4f}")
    print(f"AUC        : {roc_auc_score(y_test, y_pred_prob):.4f}")

    # ==========================================
    # 5. 球種分類模型 (Fastballs / Breaking Balls / Offspeed)
    # ==========================================
    print("\n[Pitch Classifier] 訓練三大球種家族辨識模型並生成混淆矩陣...")
    
    pitch_features = ['release_speed', 'pfx_x', 'pfx_z', 'plate_x_2026', 'plate_z_2026']
    
    df_2025['pitch_group'] = df_2025['pitch_type'].apply(map_pitch_group)
    df_2026_test['pitch_group'] = df_2026_test['pitch_type'].apply(map_pitch_group)
    
    train_pitch_df = df_2025.dropna(subset=['pitch_group'] + pitch_features).copy()
    test_pitch_df = df_2026_test.dropna(subset=['pitch_group'] + pitch_features).copy()
    
    print(f"訓練集有效樣本數: {len(train_pitch_df)} (類別分布:\n{train_pitch_df['pitch_group'].value_counts()})")
    print(f"測試集有效樣本數: {len(test_pitch_df)} (類別分布:\n{test_pitch_df['pitch_group'].value_counts()})")

    le = LabelEncoder()
    y_pitch_train = le.fit_transform(train_pitch_df['pitch_group'])

    pitch_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=120,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    pitch_model.fit(train_pitch_df[pitch_features], y_pitch_train)

    # 儲存球種分類模型及對應的 LabelEncoder
    joblib.dump(pitch_model, os.path.join(output_model_dir, "pitch_group_classifier.joblib"))
    joblib.dump(le, os.path.join(output_model_dir, "pitch_label_encoder.joblib"))
    print(f"三大球種分類模型與 Encoder 已儲存至: {output_model_dir}/")

    y_pred_pitch_encoded = pitch_model.predict(test_pitch_df[pitch_features])
    y_pred_pitch = le.inverse_transform(y_pred_pitch_encoded)

    plot_multiclass_pitch_confusion_matrix(
        y_true_pitch=test_pitch_df['pitch_group'],
        y_pred_pitch=y_pred_pitch,
        output_image='pitch_group_confusion_matrix.png',
        output_excel='pitch_group_confusion_matrix.xlsx'
    )

    # ==========================================
    # 6. 整合測試集預測明細並輸出 CSV
    # ==========================================
    print("\n[Export Predictions] 輸出測試集完整預測明細...")
    df_2026_test['pred_miscall_prob'] = y_pred_prob
    df_2026_test['pred_miscall_label'] = (y_pred_prob >= 0.5).astype(int)
    
    # 合併球種預測標籤
    test_pitch_df['pred_pitch_group'] = y_pred_pitch
    df_2026_test['pred_pitch_group'] = test_pitch_df['pred_pitch_group']

    output_csv = "test_predictions_2026.csv"
    df_2026_test.to_csv(output_csv, index=False)
    print(f"測試集預測結果已成功匯出至: {output_csv}")

if __name__ == "__main__":
    main()
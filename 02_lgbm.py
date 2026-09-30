import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
import joblib


# 1. 讀取模組 A 處理好的乾淨資料
df = pd.read_csv("tra2526.csv")

# ==========================================
# 假定 df 已經讀入，以下為資料前處理
# ==========================================

# 2. 定義 Ground Truth (絕對好球帶) 與 Target (是不是誤判)
# 2026 ABS 基準：本壘板寬 17 吋，所以左右好球帶邊界為 8.5 吋 (約 0.708 ft)
df['is_true_strike'] = (
    (df['plate_x_2026'].abs() <= 0.708) & 
    (df['plate_z_2026'] >= df['sz_bot']) & 
    (df['plate_z_2026'] <= df['sz_top'])
)

df['is_called_strike'] = df['description'] == 'called_strike'

# 定義目標變數 y (1: 主審誤判，挑戰會成功 / 0: 主審判對，挑戰會失敗)
df['is_miscall'] = (df['is_true_strike'] != df['is_called_strike']).astype(int)

# 3. Hard Thresholding: 只過濾出 Shadow Zone (邊緣 ±2 吋 = 約 0.166 ft)
# 這是為了避免模型學到一堆明顯好壞球的廢話
shadow_zone_mask = (
    ((df['plate_x_2026'].abs() > 0.542) & (df['plate_x_2026'].abs() < 0.874)) | 
    ((df['plate_z_2026'] > df['sz_bot'] - 0.166) & (df['plate_z_2026'] < df['sz_bot'] + 0.166)) |
    ((df['plate_z_2026'] > df['sz_top'] - 0.166) & (df['plate_z_2026'] < df['sz_top'] + 0.166))
)
df_shadow = df[shadow_zone_mask].copy()

# 4. 準備特徵 (X) 與標籤 (y)
features = ['plate_x_2026', 'plate_z_2026', 'sz_top', 'sz_bot', 'release_speed', 'pfx_x', 'pfx_z']
X = df_shadow[features]
y = df_shadow['is_miscall']

# 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 模型訓練與校準
# ==========================================

# 5. 建立基礎 LightGBM 模型 (參數先用預設，不花時間 Tuning)
base_lgb = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# 6. 【專家捷徑】套用 Isotonic Regression 進行機率校準
# 這會利用交叉驗證，確保模型吐出來的 80% 機率，在現實中真的等於 80% 翻盤率
calibrated_lgb = CalibratedClassifierCV(estimator=base_lgb, method='isotonic', cv=5)
calibrated_lgb.fit(X_train, y_train)

# ==========================================
# 模型評估與儲存
# ==========================================

# 預測機率
y_pred_prob = calibrated_lgb.predict_proba(X_test)[:, 1]

# 評估指標 (不要看 Accuracy，看 Brier Score 與 AUC)
brier = brier_score_loss(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Shadow Zone AUC: {auc:.4f} (區分誤判的能力)")
print(f"Brier Score: {brier:.4f} (越接近 0 機率越精準)")

# 將訓練好的模型存檔，給模組 D 使用
joblib.dump(calibrated_lgb, "module_B_calibrated_lgb.pkl")
print("✅ 模組 B 模型已儲存為 module_B_calibrated_lgb.pkl")
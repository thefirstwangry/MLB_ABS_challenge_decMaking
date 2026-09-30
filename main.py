from pybaseball import statcast
import pandas as pd

# 1. 發射追蹤子彈：先抓 2025 年的 1 週資料來測試
df_raw = statcast(start_dt="2025-05-01", end_dt="2025-05-10")

# 2. 定義你需要保留的核心欄位
core_columns = [
    'game_pk', 'game_date', 'pitcher', 'batter', 'fielder_2', 
    'stand', 'p_throws', 'inning', 'inning_topbot', 
    'outs_when_up', 'balls', 'strikes', 'on_1b', 'on_2b', 'on_3b', 
    'bat_score', 'fld_score', 'description', 
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 
    'plate_x', 'plate_z', 'sz_top', 'sz_bot', 
    'release_speed', 'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_z'
]

# 3. 過濾欄位並剔除不要的揮棒資料
df_clean = df_raw[core_columns]
df_clean = df_clean[df_clean['description'].isin(['called_strike', 'ball'])]

# 4. 存成 csv 或 parquet，以後測試就直接讀這個小檔案，不要一直 call API
df_clean.to_csv("tracer_bullet_data.csv", index=False)
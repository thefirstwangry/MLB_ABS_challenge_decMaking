import pandas as pd
import numpy as np

def reconstruct_2026_midpoint_coordinates(df):
    """
    將 Statcast 歷史軌跡推平至 2026 ABS 本壘板中線 (y = 8.5/12 = 0.70833 ft)
    解決 pybaseball 缺乏 x0, y0, z0 的問題，透過 plate_x, plate_z 逆推初始座標。
    """
    # 目標 Y 座標 (2026 ABS 中線，距離尖端 8.5 吋)
    Y_TARGET = 8.5 / 12.0 
    
    # 本壘板前緣 Y 座標 (距離尖端 17 吋)
    Y_PLATE_FRONT = 17.0 / 12.0
    
    # 避免 SettingWithCopyWarning
    data = df.copy()
    
    # Statcast 9-param 的基準點 y0 統一為 50 ft
    y0 = 50.0

    # 一元二次方程式係數 (A, B) - 整個推算過程不變
    A = 0.5 * data['ay']
    B = data['vy0']
    
    # ==========================================
    # 步驟 1：算出球抵達「本壘板前緣 (y = 17/12)」的時間 t_plate
    # ==========================================
    C_plate = y0 - Y_PLATE_FRONT
    discriminant_plate = B**2 - 4 * A * C_plate
    
    valid_mask_plate = discriminant_plate >= 0
    t_plate = pd.Series(np.nan, index=data.index)
    t_plate[valid_mask_plate] = (-B[valid_mask_plate] - np.sqrt(discriminant_plate[valid_mask_plate])) / (2 * A[valid_mask_plate])
    
    # ==========================================
    # 步驟 2：利用 t_plate, plate_x, plate_z 逆推 y=50 呎時的 x0, z0
    # ==========================================
    x0 = data['plate_x'] - (data['vx0'] * t_plate + 0.5 * data['ax'] * (t_plate**2))
    z0 = data['plate_z'] - (data['vz0'] * t_plate + 0.5 * data['az'] * (t_plate**2))

    # ==========================================
    # 步驟 3：計算球抵達「2026 ABS 中線 (y = 8.5/12)」的時間 t_mid，並代入 x0, z0
    # ==========================================
    C_mid = y0 - Y_TARGET
    discriminant_mid = B**2 - 4 * A * C_mid
    
    valid_mask_mid = discriminant_mid >= 0
    data['t_mid'] = np.nan
    data.loc[valid_mask_mid, 't_mid'] = (-B[valid_mask_mid] - np.sqrt(discriminant_mid[valid_mask_mid])) / (2 * A[valid_mask_mid])
    
    t = data['t_mid']
    
    # 產出最終 2026 年的進壘點座標
    data['plate_x_2026'] = x0 + data['vx0'] * t + 0.5 * data['ax'] * (t**2)
    data['plate_z_2026'] = z0 + data['vz0'] * t + 0.5 * data['az'] * (t**2)
    
    return data

def main():
    df = pd.read_csv('statcast_2025_sha.csv')
    df_transfer = reconstruct_2026_midpoint_coordinates(df)
    df_transfer.to_csv("tra_25.csv", index=False)


if __name__ == '__main__':
    main()
import pandas as pd

# ==========================================
# 1. 讀取你的 RE288 矩陣與機會成本矩陣
# ==========================================
def load_matrices():
    # 讀取你的真實資料表
    df_re288 = pd.read_csv('re288_baseline_wpa_matrix.csv')

    
    # 將 DataFrame 轉換為 Dictionary 以達到 O(1) 極速查表
    # Key: (Inning, Half_Inning, Outs, Base_State, Balls, Strikes, Home_Score_Diff)
    re_dict = df_re288.set_index(['Inning', 'Half_Inning', 'Outs', 'Base_State', 'Balls', 'Strikes', 'Home_Score_Diff'])['RE_Value'].to_dict()
    
    return re_dict

# ==========================================
# 2. 定義查表與狀態轉移邏輯
# ==========================================
def get_re_value(re_dict, state):
    """從字典中快速查出得分期望值"""
    # 如果 3 出局，該半局結束，期望值直接歸零
    if state['Outs'] >= 3:
        return 0.0
    
    # 組合查詢的 Key
    key = (state['Inning'], state['Half_Inning'], state['Outs'], 
            state['Base_State'], state['Balls'], state['Strikes'], state['Home_Score_Diff'])
    
    # 查表，若遇到查不到的極端狀態(如四壞保送後的壘包狀態尚未寫入)，先回傳一個預設值
    return re_dict.get(key, 0.0)

def calculate_delta_re(re_dict, current_state):
    """
    計算挑戰改判所帶來的收益 (Delta RE)
    假設這顆球原本被主審判為好球 (Called Strike)，但球員想挑戰它是壞球 (Ball)
    """
    # 平行時空 1：挑戰失敗，維持原判 (Strike + 1)
    state_called = current_state.copy()
    state_called['Strikes'] += 1
    # TODO: 未來要加入「如果 Strikes == 3，Outs + 1, 壘包與球數歸零」的邏輯
    
    # 平行時空 2：挑戰成功，改判壞球 (Ball + 1)
    state_overturned = current_state.copy()
    state_overturned['Balls'] += 1
    # TODO: 未來要加入「如果 Balls == 4，打者上壘，Base_State 改變」的邏輯
    
    # 查表取得兩個時空的期望值
    re_called = get_re_value(re_dict, state_called)
    re_overturned = get_re_value(re_dict, state_overturned)
    
    # 計算淨收益
    delta_re = re_overturned - re_called
    return delta_re, re_called, re_overturned

# ==========================================
# 3. 決策引擎主程式 (整合所有資料)
# ==========================================
def decision_engine_pipeline():
    
    # 1. 載入查表矩陣
    re_dict = load_matrices()
    
    # 2. 模擬當下一顆球的實戰狀態 (例如：1局上半，無人出局，0好0壞)
    current_state = {
        'Inning': 8, 'Half_Inning': 'Top', 'Outs': 0, 
        'Base_State': 0, 'Balls': 0, 'Strikes': 0, 'Home_Score_Diff': 0
    }
    
    # 3. 模擬從 LightGBM (模組 B) 吐出的翻盤機率
    # 假設我們預測這顆邊角球有 65% 機率能改判成功
    p_success = 1
    
    # 4. 模擬從機會成本矩陣 (模組 C) 查到的定價
    # 假設這張挑戰權當下市價為 0.05 分
    opp_cost_re = 0.0000000000011
    
    # ---------------- 執行計算 ----------------
    
    # 計算 Delta RE
    delta_re, re_called, re_overturned = calculate_delta_re(re_dict, current_state)
    
    # 計算最終 EV
    # 公式：EV = (成功率 * 賺到的分數) - (失敗率 * 失去挑戰權的代價)
    expected_gain = p_success * delta_re
    expected_loss = (1 - p_success) * opp_cost_re
    final_ev = expected_gain - expected_loss
    
    # ---------------- 輸出結果 ----------------
    print("\n--- 賽局分析 ---")
    print(f"原判好球期望值: {re_called:.3f}")
    print(f"改判壞球期望值: {re_overturned:.3f}")
    print(f"改判成功淨收益 (Delta RE): +{delta_re:.3f} 分")
    
    print("\n--- 決策輸出 ---")
    print(f"預測翻盤率 (P_success): {p_success*100:.1f}%")
    print(f"挑戰權機會成本: {opp_cost_re:.3f} 分")
    print(f"最終挑戰 EV 值: {final_ev:.4f}")
    
    if final_ev > 0:
        print("🟢 系統建議：EV 為正，立刻拍頭盔挑戰！")
    else:
        print("🔴 系統建議：EV 為負，保留挑戰權！")

if __name__ == "__main__":
    decision_engine_pipeline()
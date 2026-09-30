"""
MLB ABS Challenge Decision Engine & Statistical Audit Pipeline
Re-implemented for 2026 Production Standards.

Features:
1. O(1) RE288 Matrix Lookup with Score Clamping.
2. Rigorous Markov State Transitions (Force-advance logic).
3. Spiegelhalter's Z-Test & Subgroup Calibration.
4. Bootstrap Confidence Intervals for Expected Value (EV).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List, Union


class GameStateEngine:
    """賽局查表引擎與狀態轉移處理器"""

    def __init__(self, re_matrix_path: str = 're288_baseline_wpa_matrix.csv'):
        """
        初始化引擎並載入 RE 矩陣至本機記憶體雜湊表，確保 O(1) 查詢時間。
        """
        self.re_dict: Dict[Tuple[int, str, int, int, int, int, int], float] = {}
        self._load_or_mock_re_matrix(re_matrix_path)

    def _load_or_mock_re_matrix(self, path: str):
        """讀取 RE288 csv。若不存在，則自動生成測試用 Dummy 資料。"""
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            print(f"⚠️ {path} 不存在，自動生成 Dummy RE 資料供測試使用。")
            df = self._generate_dummy_re_data()

        # 將 DataFrame 轉為 Hash Map 達成 O(1) 查詢
        for _, row in df.iterrows():
            # 強制分差截斷 (Clamping) 於載入時或查詢時確保邏輯一致
            clamped_diff = max(-5, min(5, int(row['Home_Score_Diff'])))
            key = (
                int(row['Inning']),
                str(row['Half_Inning']),
                int(row['Outs']),
                int(row['Base_State']),
                int(row['Balls']),
                int(row['Strikes']),
                clamped_diff
            )
            self.re_dict[key] = float(row['RE_Value'])

    def _generate_dummy_re_data(self) -> pd.DataFrame:
        """生成基礎測試資料，涵蓋滿壘與三振等關鍵邊界狀態"""
        records = []
        for outs in [0, 1, 2]:
            for bases in range(8):
                for b in range(4):
                    for s in range(3):
                        for diff in range(-6, 7):
                            # 隨機設計基礎 RE，滿壘 (7) 期望值較高，出局數越多期望值越低
                            base_re = (bases * 0.25) * ((3 - outs) / 3.0)
                            records.append({
                                'Inning': 9,
                                'Half_Inning': 'Bot',
                                'Outs': outs,
                                'Base_State': bases,
                                'Balls': b,
                                'Strikes': s,
                                'Home_Score_Diff': diff,
                                'RE_Value': round(base_re, 4)
                            })
        return pd.DataFrame(records)

    def _get_re(self, state: Tuple[int, str, int, int, int, int, int]) -> float:
        """執行 O(1) 查表，並動態處理分差截斷與 3 出局吸收態"""
        inning, half, outs, bases, balls, strikes, diff = state
        
        # 吸收態防禦：3 出局期望值得分直接歸零
        if outs >= 3:
            return 0.0
            
        # 邊界截斷規則：大於 5 分視為 5，小於 -5 視為 -5
        clamped_diff = max(-5, min(5, diff))
        
        key = (inning, half, outs, bases, balls, strikes, clamped_diff)
        return self.re_dict.get(key, 0.0)

    @staticmethod
    def resolve_force_advance(base_state: int) -> Tuple[int, int]:
        """
        處理保送上壘時的「擠壘/強制推進」演算法。
        Base_State 為 0~7 的二進位整數 (1B=1, 2B=2, 3B=4)。
        
        :return: (new_base_state, runs_scored)
        """
        runs_scored = 0
        if base_state in [0, 2, 4, 6]:
            # 一壘空著，打者直接上一壘
            new_state = base_state + 1
        elif base_state in [1, 5]:
            # 一壘有人，二壘空著，推進至一二壘
            new_state = base_state + 2
        elif base_state == 3:
            # 一二壘有人，推進至滿壘
            new_state = 7
        elif base_state == 7:
            # 滿壘，擠回一分，狀態維持滿壘
            new_state = 7
            runs_scored = 1
        else:
            new_state = base_state
            
        return new_state, runs_scored

    def calculate_delta_re(self, current_state: Tuple[int, str, int, int, int, int, int]) -> float:
        """
        精準計算挑戰前後來回的期望得分差 (Delta RE)。
        假設情境：主審原判好球 (Called Strike)，挑戰改判為壞球 (Overturned Ball)
        """
        inning, half, outs, bases, balls, strikes, diff = current_state

        # ==========================================
        # 平行時空 A：維持原判 (Called Strike)
        # ==========================================
        if strikes < 2:
            state_a = (inning, half, outs, bases, balls, strikes + 1, diff)
            re_a = self._get_re(state_a)
        else:
            # 三振出局，球數重置，壘包維持
            state_a = (inning, half, outs + 1, bases, 0, 0, diff)
            re_a = self._get_re(state_a)

        # ==========================================
        # 平行時空 B：挑戰成功 (Overturned Ball)
        # ==========================================
        runs_scored_b = 0
        if balls < 3:
            state_b = (inning, half, outs, bases, balls + 1, strikes, diff)
            re_b = self._get_re(state_b)
        else:
            # 四壞保送，觸發強制推進演算法
            new_bases, runs_scored_b = self.resolve_force_advance(bases)
            state_b = (inning, half, outs, new_bases, 0, 0, diff)
            re_b = self._get_re(state_b)

        # 淨收益 = (新狀態期望值 + 當下得分) - 原判狀態期望值
        return (re_b + runs_scored_b) - re_a


class StatisticalOutputAuditor:
    """輸出統計檢定套件 (Rigorous Statistical Validation)"""

    @staticmethod
    def spiegelhalter_z_test(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        """
        實作 Spiegelhalter's Z-Test 檢驗機率校準。
        公式: Z = sum((Y - P)*(1 - 2P)) / sqrt(sum((1 - 2P)^2 * P * (1 - P)))
        """
        n = len(y_true)
        if n == 0:
            return 0.0, 1.0

        numerator = np.sum((y_true - y_prob) * (1.0 - 2.0 * y_prob))
        variance_terms = ((1.0 - 2.0 * y_prob) ** 2) * y_prob * (1.0 - y_prob)
        denominator = np.sqrt(np.sum(variance_terms))

        if denominator == 0:
            return 0.0, 1.0

        z_stat = numerator / denominator
        # 計算雙尾 p-value
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
        
        return z_stat, p_value

    @staticmethod
    def stratified_brier_score(y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray) -> Dict[str, dict]:
        """
        條件式局部校準檢驗：依據給定組別計算 Brier Score。
        """
        results = {}
        unique_groups = np.unique(groups)
        
        for g in unique_groups:
            mask = (groups == g)
            y_t = y_true[mask]
            y_p = y_prob[mask]
            
            brier = np.mean((y_t - y_p) ** 2) if len(y_t) > 0 else 0.0
            results[str(g)] = {
                'Brier_Score': round(brier, 4),
                'Sample_Size': len(y_t)
            }
            
        return results

    @staticmethod
    def bootstrap_ev_ci(p_samples: np.ndarray, delta_v: float, oc: float, n_bootstraps: int = 1000) -> Tuple[float, float, str]:
        """
        利用 Bootstrap 重抽樣預測機率，估算 EV 的 95% 置信區間。
        EV = E[P] * Delta_V - (1 - E[P]) * OC
        """
        ev_estimates = np.zeros(n_bootstraps)
        n = len(p_samples)
        
        for i in range(n_bootstraps):
            # 抽樣並計算平均機率
            sampled_p = np.random.choice(p_samples, size=n, replace=True)
            mean_p = np.mean(sampled_p)
            ev_estimates[i] = mean_p * delta_v - (1.0 - mean_p) * oc
            
        ev_lower = np.percentile(ev_estimates, 2.5)
        ev_upper = np.percentile(ev_estimates, 97.5)
        
        # 門檻判定
        if ev_lower > 0:
            flag = "🟢 高信心挑戰 (High Confidence Challenge)"
        elif ev_upper < 0:
            flag = "🔴 拒絕挑戰 (Hold)"
        else:
            flag = "🟧 邊緣決策 (Marginal Decision)"
            
        return round(ev_lower, 4), round(ev_upper, 4), flag


# ==========================================
# 執行示範區塊
# ==========================================
if __name__ == '__main__':
    print("=== 1. 賽局查表引擎與邊界條件測試 ===")
    engine = GameStateEngine(re_matrix_path='non_existent_re288.csv')
    
    # 測試 1: 滿球數 (3-2) 滿壘 (Base_State=7) 2 出局，分差落後 7 分 (應自動截斷至 -5)
    test_state = (9, 'Bot', 2, 7, 3, 2, -7)
    delta_re = engine.calculate_delta_re(test_state)
    print(f"測試狀態: {test_state}")
    print(f"-> 預期: 保送擠回 1 分且局數繼續，三振則 3 出局無得分。")
    print(f"-> 計算所得 Delta RE: {delta_re:.4f}\n")
    
    
    print("=== 2. 輸出統計檢定套件測試 ===")
    auditor = StatisticalOutputAuditor()
    
    # 生成 1,000 筆模擬的預測資料
    np.random.seed(42)
    N = 1000
    sim_prob = np.random.beta(2, 5, N)  # 模擬右偏機率預測
    # 注入些微未校準偏差：實際成功率略低於預測率
    sim_true = np.random.binomial(1, sim_prob * 0.85)
    sim_groups = np.random.choice(["Fastball_Corner", "Breaking_Horizontal"], N)
    
    # A. Spiegelhalter's Z-Test
    z_stat, p_val = auditor.spiegelhalter_z_test(sim_true, sim_prob)
    print(f"[全域校準檢定] Z-Statistic: {z_stat:.4f}, p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("-> ⚠️ 拒絕原假設：預測機率存在顯著過度自信或偏誤\n")
    else:
        print("-> ✅ 接受原假設：模型具備良好校準度\n")
        
    # B. 分組 Brier Score
    stratified_res = auditor.stratified_brier_score(sim_true, sim_prob, sim_groups)
    print("[條件式局部校準檢驗]")
    for grp, metrics in stratified_res.items():
        print(f"-> {grp}: Brier={metrics['Brier_Score']}, N={metrics['Sample_Size']}")
    print()
    
    # C. Bootstrap EV 區間估算
    # 假設我們抽取特定情境下的 50 筆相似球種機率預測，代入剛才的 delta_re，以及 0.25 的機會成本
    context_probs = np.random.normal(0.6, 0.1, 50)
    context_probs = np.clip(context_probs, 0.01, 0.99)
    
    opportunity_cost = 0.25
    ev_low, ev_high, decision_flag = auditor.bootstrap_ev_ci(
        p_samples=context_probs, 
        delta_v=delta_re, 
        oc=opportunity_cost
    )
    print("[單一決策 EV 置信區間 (Bootstrap 1000次)]")
    print(f"-> Delta RE: {delta_re:.4f}, 機會成本: {opportunity_cost}")
    print(f"-> 95% 置信區間: [{ev_low}, {ev_high}]")
    print(f"-> 系統決策指示: {decision_flag}")
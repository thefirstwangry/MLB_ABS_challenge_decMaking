import time
import pandas as pd
from pybaseball import statcast
from tqdm import tqdm

# 啟用 pybaseball 的本地快取以避免重複查詢
from pybaseball import cache
cache.enable()

# 設定 2025 球季區間（涵蓋海外開幕戰至世界大賽結束）
START_DATE = '2026-07-01'
END_DATE = '2026-08-20'
CHUNK_DAYS = 5  # 每次抓取 5 天的資料，降低伺服器負擔

def download_season_statcast(start_date: str, end_date: str, chunk_days: int = 5, output_file: str = 'statcast_2025.parquet'):
    date_ranges = pd.date_range(start=start_date, end=end_date, freq=f'{chunk_days}D')
    all_chunks = []
    
    print(f"開始下載 2025 球季逐球資料（區間: {start_date} ~ {end_date}）...")

    for i in tqdm(range(len(date_ranges)), desc="下載進度"):
        chunk_start = date_ranges[i].strftime('%Y-%m-%d')
        
        # 計算每批次的結束日期
        if i + 1 < len(date_ranges):
            chunk_end = (date_ranges[i+1] - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            chunk_end = end_date
            
        if chunk_start > end_date:
            break

        # 下載與重試機制
        success = False
        attempts = 0
        while not success and attempts < 3:
            try:
                # pybaseball statcast 查詢
                df_chunk = statcast(start_dt=chunk_start, end_dt=chunk_end, verbose=False)
                
                if df_chunk is not None and not df_chunk.empty:
                    all_chunks.append(df_chunk)
                
                success = True
            except Exception as e:
                attempts += 1
                time.sleep(3)  # 失敗時稍作等待再重試
                if attempts == 3:
                    print(f"\n[警告] 下載區間 {chunk_start} ~ {chunk_end} 失敗: {e}")
        
        # 避免發送過於頻繁的 Request
        time.sleep(0.5)

    if not all_chunks:
        print("未抓取到任何資料。")
        return None

    # 合併所有區間資料
    print("\n正在合併資料...")
    full_df = pd.concat(all_chunks, ignore_index=True)
    
    # 儲存檔案（推薦 Parquet 格式，比 CSV 更小且讀寫更快）
    if output_file.endswith('.parquet'):
        full_df.to_parquet(output_file, index=False)
    else:
        full_df.to_csv(output_file, index=False)

    print(f"下載完成！共 {len(full_df):,} 筆逐球資料，已儲存至 {output_file}")
    return full_df

if __name__ == '__main__':
    # 執行下載（若需 CSV 可將副檔名改為 .csv）

    df_2025 = download_season_statcast(
        start_date=START_DATE,
        end_date=END_DATE,
        chunk_days=CHUNK_DAYS,
        output_file='statcast_2026H2_sha.csv'
    )
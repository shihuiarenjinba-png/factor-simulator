import pandas as pd
import numpy as np
import datetime
import streamlit as st
from quant_engine import QuantEngine

# --- フォールバック用リスト (スクレイピング失敗時・サイト仕様変更時の保険) ---
FALLBACK_NIKKEI_225 = [
    "7203.T", "6758.T", "8035.T", "9984.T", "9983.T", "6098.T", "4063.T", "6367.T", "9432.T", "4502.T",
    "4503.T", "6501.T", "7267.T", "8058.T", "8001.T", "6954.T", "6981.T", "9020.T", "9022.T", "7741.T",
    "5108.T", "4452.T", "6902.T", "7974.T", "8031.T", "4519.T", "4568.T", "6273.T", "4543.T", "6702.T",
    "6503.T", "4901.T", "4911.T", "2502.T", "2802.T", "3382.T", "8306.T", "8316.T", "8411.T", "8766.T",
    "8591.T", "8801.T", "8802.T", "9021.T", "9101.T", "9433.T", "9434.T", "9501.T", "9502.T"
]

FALLBACK_TOPIX_CORE30 = [
    "7203.T", "6758.T", "8306.T", "9984.T", "9432.T", "6861.T", "8035.T", "6098.T", "8316.T", "4063.T",
    "9983.T", "6367.T", "4502.T", "7974.T", "8058.T", "8001.T", "2914.T", "6501.T", "7267.T", "8411.T",
    "6954.T", "6902.T", "7741.T", "9020.T", "9022.T", "4452.T", "5108.T", "8801.T", "6752.T", "6273.T"
]

class MarketMonitor:
    """
    【新規追加 Module】市場監視・自動オーケストレーションモジュール
    市場の「今」を監視し、構成銘柄の最新リスト自動取得と、
    データの一括（バルク）取得・キャッシュ管理を一手に行う心臓部。
    """
    
    @staticmethod
    @st.cache_data(ttl=86400) # 銘柄の入れ替えは頻繁ではないため、リスト取得は1日1回更新
    def get_latest_tickers(bench_mode):
        if bench_mode == "Nikkei 225":
            try:
                # 日経平均プロファイル公式サイトから構成銘柄を直接抽出
                url = "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225"
                dfs = pd.read_html(url)
                for df in dfs:
                    if 'コード' in df.columns:
                        tickers = []
                        for code in df['コード'].dropna():
                            try:
                                tickers.append(f"{int(code)}.T")
                            except ValueError:
                                pass
                        if len(tickers) >= 200:
                            return tickers
            except Exception:
                pass
            return FALLBACK_NIKKEI_225
            
        else: # TOPIX Core 30
            try:
                # TOPIX Core 30はWikipediaのリストがパースしやすく安定しているため利用
                url = "https://ja.wikipedia.org/wiki/TOPIX_Core30"
                dfs = pd.read_html(url)
                for df in dfs:
                    if '証券コード' in df.columns:
                        tickers = []
                        for code in df['証券コード'].dropna():
                            try:
                                tickers.append(f"{int(code)}.T")
                            except ValueError:
                                pass
                        if len(tickers) >= 25:
                            return tickers
            except Exception:
                pass
            return FALLBACK_TOPIX_CORE30

    @staticmethod
    @st.cache_data(ttl=1800) # 市場データは30分間隔でリフレッシュし、時差を最小化
    def get_market_intelligence(bench_mode, benchmark_etf):
        """
        最新の銘柄リストでバルク取得を行い、市場の「ものさし」を計算してメモリに保持する
        """
        # 循環参照を防ぐためにメソッド内でインポート
        from data_provider import DataProvider  
        
        # 1. 公式サイトから最新銘柄リストを取得
        tickers = MarketMonitor.get_latest_tickers(bench_mode)
        
        # 2. バルクデータ取得 (DataProviderがyf.Tickersで一括処理するため速い)
        df_fund = DataProvider.fetch_fundamentals(tickers)
        df_hist = DataProvider.fetch_historical_prices(tickers + [benchmark_etf])
        
        # 3. エンジンでBetaなどを計算
        df_fund = QuantEngine.calculate_beta(df_fund, df_hist, benchmark_etf)
        
        # 4. UniverseManagerで市場全体の平均・標準偏差（ものさし）を算出
        stats, processed_data = UniverseManager.generate_market_stats(df_fund)
        
        return {
            'stats': stats,
            'processed_data': processed_data,
            'last_updated': datetime.datetime.now().strftime("%H:%M:%S")
        }


class UniverseManager:
    """
    【Module 3】 市場統計管理 (Pro Version)
    【完了版 Step 5】直交化パラメータの永続化と、市場の「ものさし」の完全固定
    """

    @staticmethod
    def generate_market_stats(df_universe_raw):
        """
        市場全体の生データを受け取り、統計情報(Stats)と処理済みデータを返す
        """
        # 1. 生データを計算可能な指標に変換 
        # (ここで Investment_Raw 等が生成される)
        df_proc = QuantEngine.process_raw_factors(df_universe_raw)

        # 2. 統計作成用の外れ値処理 (Winsorization)
        # Zスコア計算の基となるカラムを指定 (モメンタムを除外した5ファクター)
        numeric_cols = [
            'Size_Log', 
            'Value_Raw',
            'Quality_Raw',
            'Investment_Raw',
            'Beta_Raw'
        ]
        
        df_for_stats = df_proc.copy()
        
        for col in numeric_cols:
            if col in df_for_stats.columns:
                # データ型を確実に数値にする
                df_for_stats[col] = pd.to_numeric(df_for_stats[col], errors='coerce')
                
                # 上下1%をクリップして異常値（極端な資産変動など）を除外
                lower = df_for_stats[col].quantile(0.01)
                upper = df_for_stats[col].quantile(0.99)
                df_for_stats[col] = df_for_stats[col].clip(lower, upper)

        # 3. 直交化パラメータ & R² の算出 (Investmentに対してQualityを直交化)
        # Investment(資産拡大)の影響をQuality(ROE)から取り除く
        df_ortho, ortho_params = QuantEngine.calculate_orthogonalization(
            df_for_stats, 
            x_col='Investment_Raw', 
            y_col='Quality_Raw'
        )

        # 4. 各ファクターの統計量(Median, MAD)を算出
        # 【修正】市場の「ものさし（傾き・切片）」を固定。KeyErrorを防ぐため get() で安全に取得
        stats = {
            'ortho_slope': ortho_params.get('slope', 0.0),
            'ortho_intercept': ortho_params.get('intercept', 0.0),
            'ortho_r_squared': ortho_params.get('r_squared', 0.0)
        }

        # 統計を抽出する対象と、参照するカラム名のマッピング (5ファクターに完全同期)
        target_factors = {
            'Beta': 'Beta_Raw',          
            'Size': 'Size_Log',
            'Value': 'Value_Raw',
            'Quality': 'Quality_Raw_Orthogonal', # 直交化後のクオリティ
            'Investment': 'Investment_Raw'       # 総資産増加率
        }

        for factor, col in target_factors.items():
            # カラムが存在しない場合の救済措置
            if col not in df_ortho.columns:
                if factor == 'Quality' and 'Quality_Raw' in df_ortho.columns:
                    col = 'Quality_Raw'
                else:
                    stats[factor] = {'median': 0.0, 'mad': 1.0, 'col': col}
                    continue

            series = df_ortho[col].dropna()

            if series.empty:
                stats[factor] = {'median': 0.0, 'mad': 1.0, 'col': col}
            else:
                # 中央値 (Median)
                median_val = series.median()
                
                # MAD (Median Absolute Deviation)
                abs_deviation = np.abs(series - median_val)
                mad_val = abs_deviation.median()
                
                # 安全策: MADが0の場合は標準偏差で代用
                if mad_val == 0:
                    mad_val = series.std() if series.std() > 0 else 1.0

                stats[factor] = {
                    'median': median_val,
                    'mad': mad_val,
                    'col': col
                }
                
                if factor == 'Quality':
                    stats[factor]['r_squared'] = ortho_params.get('r_squared', 0.0)

        return stats, df_ortho

import pandas as pd
import numpy as np
import datetime
import streamlit as st
import re  # 4桁の証券コード抽出用に追加
from scipy.stats import linregress

# 循環参照を避けるための遅延インポート
# from quant_engine import QuantEngine

# --- フォールバック用リスト (スクレイピング失敗時・サイト仕様変更時の保険・確定版) ---
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
    【Module】市場監視・自動オーケストレーションモジュール
    市場の「今」を監視し、構成銘柄の最新リスト自動取得とキャッシュ管理を行う。
    Wikipediaなどの不確実なソースを排除し、公式ファイル(CSV/Excel)の読み込みに対応。
    """
    
    @staticmethod
    def load_tickers_from_file(uploaded_file):
        """
        ユーザーがアップロードした構成銘柄リスト(CSV/Excel)を読み込む。
        特定の列名に依存せず、全データから「4桁の数字」を自動スキャンして抽出する。
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                return None
            
            best_tickers = []
            max_matches = 0
            
            # 全ての列をスキャンし、4桁の数字が最も多く含まれる列を特定する
            for col in df.columns:
                col_strs = df[col].astype(str).fillna('')
                tickers_in_col = []
                
                for val in col_strs:
                    # 単独の4桁の数字を抽出（例: "7203", " 7203 ", "7203.0" の一部など）
                    matches = re.findall(r'\b\d{4}\b', val)
                    for match in matches:
                        tickers_in_col.append(f"{match}.T")
                
                # 年号(2023など)の列を誤検知しないよう、ユニーク数も加味して最も多くマッチした列を採用
                unique_tickers = list(set(tickers_in_col))
                if len(unique_tickers) > max_matches:
                    max_matches = len(unique_tickers)
                    best_tickers = unique_tickers
                    
            if best_tickers:
                return best_tickers
                
        except Exception as e:
            st.warning(f"ファイルのパース中にエラーが発生しました: {e}")
            
        return None

    @staticmethod
    @st.cache_data(ttl=86400)
    def get_latest_tickers(bench_mode, custom_list=None):
        """
        対象ベンチマークのティッカーリストを取得する。
        カスタムリストが渡された場合はそれを最優先する。
        """
        if custom_list is not None and len(custom_list) > 0:
            return custom_list

        if bench_mode == "Nikkei 225":
            try:
                # 日経平均プロファイル公式サイトから構成銘柄を直接抽出（信頼性高）
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
            # Wikipediaへの依存を排除。
            # 外部ファイル読み込みがない場合は、安全なマネージド・リストを返す。
            return FALLBACK_TOPIX_CORE30

    @staticmethod
    @st.cache_data(ttl=1800) # 市場データは30分間隔でリフレッシュし、時差を最小化
    def get_market_intelligence(bench_mode, benchmark_etf):
        """
        最新の銘柄リストでバルク取得を行い、市場の「ものさし」を計算してメモリに保持する
        """
        from data_provider import DataProvider  
        from quant_engine import QuantEngine
        
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
    【修正版】Winsorization (外れ値処理) の強化と統計量の強力なキャッシュ化。
    Zスコアの分母・分子が極端な一部の銘柄に引っ張られないよう保護する。
    """

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate_market_stats(df_universe_raw):
        """
        市場全体の生データを受け取り、統計情報(Stats)と処理済みデータを返す
        st.cache_data により、同じユニバースでの再計算をスキップし高速化。
        """
        from quant_engine import QuantEngine
        
        # --- 0.00病の防止: 入力データの空チェックを追加 ---
        if df_universe_raw is None or df_universe_raw.empty:
            raise ValueError("ユニバース（市場基準）データが空です。データの取得に失敗した可能性があります。")

        # 1. 生データを計算可能な指標に変換 
        # (ここで Investment_Raw 等が生成される)
        df_proc = QuantEngine.process_raw_factors(df_universe_raw)
        
        if df_proc is None or df_proc.empty:
            raise ValueError("ファクター算出後のデータが空になりました。入力データを確認してください。")

        # 2. 統計作成用の外れ値処理 (Winsorization) の強化
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
                df_for_stats[col] = pd.to_numeric(df_for_stats[col], errors='coerce')
                
                # 【Winsorization】 上下2.5%をクリップし、異常値が中央値・MADを歪めるのを防ぐ
                valid_data = df_for_stats[col].dropna()
                if len(valid_data) > 20: # サンプル数が十分な場合のみ実行
                    lower = valid_data.quantile(0.025)
                    upper = valid_data.quantile(0.975)
                    df_for_stats[col] = df_for_stats[col].clip(lower, upper)

        # 3. 直交化パラメータ & R² の算出 (Investmentに対してQualityを直交化)
        # Investment(資産拡大)の影響をQuality(ROE)から取り除く
        df_ortho, ortho_params = QuantEngine.calculate_orthogonalization(
            df_for_stats, 
            x_col='Investment_Raw', 
            y_col='Quality_Raw'
        )

        # 4. 各ファクターの統計量(Median, MAD)を算出
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
                median_val = float(series.median())
                
                # MAD (Median Absolute Deviation) を堅牢に算出
                abs_deviation = np.abs(series - median_val)
                mad_val = float(abs_deviation.median())
                
                # 安全策: MADが極端に小さい(0に近い)場合は標準偏差で代用
                if mad_val < 1e-6:
                    mad_val = float(series.std())
                    if pd.isna(mad_val) or mad_val < 1e-6:
                        mad_val = 1.0

                stats[factor] = {
                    'median': median_val,
                    'mad': mad_val,
                    'col': col
                }
                
                if factor == 'Quality':
                    stats[factor]['r_squared'] = ortho_params.get('r_squared', 0.0)

        return stats, df_ortho

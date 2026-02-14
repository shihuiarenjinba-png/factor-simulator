import pandas as pd
import numpy as np
from quant_engine import QuantEngine  # 先ほど作成した計算エンジンを利用

class UniverseManager:
    """
    【Module 3】 比較対象（ベンチマーク）の統計管理
    
    市場全体（日経225やTOPIX）のデータを読み込み、
    「平均(μ)」と「標準偏差(σ)」という『モノサシ』を作成する責任を持つ。
    """

    @staticmethod
    def generate_market_stats(df_universe_raw):
        """
        市場全体の生データを受け取り、統計情報(Stats)と処理済みデータを返す
        
        Args:
            df_universe_raw (pd.DataFrame): DataProviderから取得した生の市場データ
            
        Returns:
            stats (dict): 各ファクターの平均・標準偏差・直交化パラメータ
            df_processed (pd.DataFrame): 統計計算に使用した処理済みデータ
        """
        # 1. 生データを計算可能な指標に変換 (Log化, 逆数化など)
        # Module 2 (QuantEngine) のロジックを借用して統一性を保つ
        df_proc = QuantEngine.process_raw_factors(df_universe_raw)

        # 2. 【重要】統計作成用の外れ値処理 (Winsorization)
        # 平均値を計算する前に、極端な異常値（例: ROE 500% や エラー値）を丸める。
        # これを行わないと、たった1つの異常値のせいで標準偏差が巨大になり、
        # 他の全銘柄のスコアが 0.0 に潰れてしまうのを防ぐ。
        
        numeric_cols = [
            'Size_Log', 
            'Value_Metric', 
            'Momentum_Metric', 
            'Quality_Metric', 
            'Investment_Metric'
        ]
        
        # 統計計算用の一時データフレーム（元のデータは破壊しない）
        df_for_stats = df_proc.copy()
        
        for col in numeric_cols:
            if col in df_for_stats.columns:
                # 上下1%をカットして、分布を安定させる
                df_for_stats[col] = QuantEngine.winsorize_series(df_for_stats[col], 0.01, 0.99)

        # 3. 直交化パラメータの算出 (Quality vs Investment)
        # 「市場全体」の分布において、成長株要素をどれくらい差し引くべきか(傾き)を決定する
        df_ortho, ortho_params = QuantEngine.calculate_orthogonalization(df_for_stats)

        # 4. 各ファクターの平均(mu)と標準偏差(sigma)を算出
        # ここで作られる数値が、全銘柄を評価する「基準点」となる
        stats = {
            'ortho_slope': ortho_params['slope'],
            'ortho_intercept': ortho_params['intercept']
        }

        # 統計をとる対象カラムの定義
        target_factors = {
            'Beta': 'Beta_Raw',          # Betaは外部(DataProvider)ですでに計算済み前提
            'Size': 'Size_Log',
            'Value': 'Value_Metric',
            'Momentum': 'Momentum_Metric',
            'Quality': 'Quality_Orthogonal', # 直交化後の値（純粋Quality）を使用
            'Investment': 'Investment_Metric'
        }

        for factor, col in target_factors.items():
            if col not in df_ortho.columns:
                # データが欠落している場合の安全策 (Z=0になる設定)
                stats[factor] = {'mean': 0.0, 'std': 1.0, 'col': col}
                continue

            series = df_ortho[col].dropna()

            if series.empty:
                stats[factor] = {'mean': 0.0, 'std': 1.0, 'col': col}
            else:
                # ここが「モノサシ」の本体
                stats[factor] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'col': col
                }

        return stats, df_ortho

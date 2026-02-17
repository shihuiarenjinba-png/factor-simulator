import pandas as pd
import numpy as np
from quant_engine import QuantEngine

class UniverseManager:
    """
    【Module 3】 市場統計管理 (Pro Version)
    【修正版 Step 3】QuantEngineとの整合性を確保し、エラーを解消
    """

    @staticmethod
    def generate_market_stats(df_universe_raw):
        """
        市場全体の生データを受け取り、統計情報(Stats)と処理済みデータを返す
        """
        # 1. 生データを計算可能な指標に変換
        df_proc = QuantEngine.process_raw_factors(df_universe_raw)

        # 2. 統計作成用の外れ値処理 (Winsorization)
        # エンジンとの名称統一 (Metric -> Raw)
        numeric_cols = [
            'Size_Log', 
            'Value_Raw',      # Value_Metric -> Value_Raw
            'Momentum_Raw',   # Momentum_Metric -> Momentum_Raw
            'Quality_Raw',    # Quality_Metric -> Quality_Raw
            'Investment_Raw'  # Investment_Metric -> Investment_Raw
        ]
        
        # 統計計算用の一時データフレーム
        df_for_stats = df_proc.copy()
        
        for col in numeric_cols:
            if col in df_for_stats.columns:
                # 【修正】QuantEngineにwinsorize_seriesがないため、pandas標準機能で代用
                # 上下1%をクリップする処理
                lower = df_for_stats[col].quantile(0.01)
                upper = df_for_stats[col].quantile(0.99)
                df_for_stats[col] = df_for_stats[col].clip(lower, upper)

        # 3. 直交化パラメータ & R² の算出 (Quality vs Investment)
        # 【修正】引数を新しいQuantEngineの仕様 (df, x_col, y_col) に合わせる
        # Investment(成長)に対してQuality(質)を直交化する
        df_ortho, ortho_params = QuantEngine.calculate_orthogonalization(
            df_for_stats, 
            x_col='Investment_Raw', 
            y_col='Quality_Raw'
        )

        # 4. 各ファクターの統計量(Median, MAD)を算出
        stats = {
            'ortho_slope': ortho_params['slope'],
            'ortho_intercept': ortho_params['intercept'],
            'ortho_r_squared': ortho_params.get('r_squared', 0.0)
        }

        # 統計をとる対象カラム
        target_factors = {
            'Beta': 'Beta_Raw',          
            'Size': 'Size_Log',
            'Value': 'Value_Raw',       # Value_Metric -> Value_Raw
            'Momentum': 'Momentum_Raw', # Momentum_Metric -> Momentum_Raw
            'Quality': 'Quality_Orthogonal', # 直交化後の値
            'Investment': 'Investment_Raw'   # Investment_Metric -> Investment_Raw
        }

        for factor, col in target_factors.items():
            if col not in df_ortho.columns:
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

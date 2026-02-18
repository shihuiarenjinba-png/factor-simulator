import pandas as pd
import numpy as np
from quant_engine import QuantEngine

class UniverseManager:
    """
    【Module 3】 市場統計管理 (Pro Version)
    【修正版 Step 2】指標名称の完全統一と直交化フローの整合性確保
    """

    @staticmethod
    def generate_market_stats(df_universe_raw):
        """
        市場全体の生データを受け取り、統計情報(Stats)と処理済みデータを返す
        """
        # 1. 生データを計算可能な指標に変換 (_Raw, _Log カラムが生成される)
        df_proc = QuantEngine.process_raw_factors(df_universe_raw)

        # 2. 統計作成用の外れ値処理 (Winsorization)
        # エンジン側で使用するすべての「Raw指標」を対象にする
        numeric_cols = [
            'Size_Log', 
            'Value_Raw',
            'Momentum_Raw',
            'Quality_Raw',
            'Investment_Raw',
            'Beta_Raw'
        ]
        
        df_for_stats = df_proc.copy()
        
        for col in numeric_cols:
            if col in df_for_stats.columns:
                # 上下1%をクリップして異常値による中央値・MADの歪みを防ぐ
                lower = df_for_stats[col].quantile(0.01)
                upper = df_for_stats[col].quantile(0.99)
                df_for_stats[col] = df_for_stats[col].clip(lower, upper)

        # 3. 直交化パラメータ & R² の算出 (Investmentに対してQualityを直交化)
        # 【重要】Step 1 で修正したエンジンを呼び出し、計算済みの df_ortho を受け取る
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

        # 統計を抽出する対象と、参照するカラム名のマッピング
        # ここで名称がズレると、Zスコア計算時に「市場平均=0」となり、解説がバグる原因になる
        target_factors = {
            'Beta': 'Beta_Raw',          
            'Size': 'Size_Log',
            'Value': 'Value_Raw',
            'Momentum': 'Momentum_Raw',
            'Quality': 'Quality_Raw_Orthogonal', # エンジン側で生成される名称に合わせる
            'Investment': 'Investment_Raw'
        }

        for factor, col in target_factors.items():
            # カラムが存在しない場合の救済措置
            if col not in df_ortho.columns:
                # Quality_Raw_Orthogonal が見つからない場合は Quality_Raw で代用（フォールバック）
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
                
                # 安全策: 分散が0の場合は標準偏差、それも0なら1.0
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

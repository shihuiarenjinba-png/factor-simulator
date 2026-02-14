import pandas as pd
import numpy as np
from quant_engine import QuantEngine

class UniverseManager:
    """
    【Module 3】 市場統計管理 (Pro Version)
    
    市場全体（ベンチマーク）のデータを分析し、
    「ロバストな基準値（中央値・MAD）」と「直交化パラメータ」を生成する。
    これが全てのスコア計算の「原点（ゼロ地点）」となる。
    """

    @staticmethod
    def generate_market_stats(df_universe_raw):
        """
        市場全体の生データを受け取り、統計情報(Stats)と処理済みデータを返す
        
        Args:
            df_universe_raw (pd.DataFrame): DataProviderから取得した生の市場データ
            
        Returns:
            stats (dict): 各ファクターの中央値(median)、MAD、直交化パラメータ
            df_processed (pd.DataFrame): 統計計算に使用した処理済みデータ
        """
        # 1. 生データを計算可能な指標に変換
        # (Module 2のロジックを流用して定義を統一)
        df_proc = QuantEngine.process_raw_factors(df_universe_raw)

        # 2. 統計作成用の外れ値処理 (Winsorization)
        # 基準（モノサシ）を作るために、極端な異常値をカットしたデータを用意する
        numeric_cols = [
            'Size_Log', 
            'Value_Metric', 
            'Momentum_Metric', 
            'Quality_Metric', 
            'Investment_Metric'
        ]
        
        # 統計計算用の一時データフレーム
        df_for_stats = df_proc.copy()
        
        for col in numeric_cols:
            if col in df_for_stats.columns:
                # 上下1%を丸める (TrimmingではなくWinsorizing)
                df_for_stats[col] = QuantEngine.winsorize_series(df_for_stats[col], 0.01, 0.99)

        # 3. 直交化パラメータ & R² の算出 (Quality vs Investment)
        # 市場全体で「成長率に対してROEがどれくらいあるのが普通か」を回帰分析する
        # ここで返ってくる r_squared が「市場の規律正しさ」を示す指標になる
        df_ortho, ortho_params = QuantEngine.calculate_orthogonalization(df_for_stats)

        # 4. 各ファクターの統計量(Median, MAD)を算出
        stats = {
            # 直交化パラメータを保存（ユーザー銘柄の補正に使う）
            'ortho_slope': ortho_params['slope'],
            'ortho_intercept': ortho_params['intercept'],
            'ortho_r_squared': ortho_params.get('r_squared', 0.0) # R²も保存
        }

        # 統計をとる対象カラム
        target_factors = {
            'Beta': 'Beta_Raw',          # Betaは外部計算済み前提
            'Size': 'Size_Log',
            'Value': 'Value_Metric',
            'Momentum': 'Momentum_Metric',
            'Quality': 'Quality_Orthogonal', # 直交化後の値を使用
            'Investment': 'Investment_Metric'
        }

        for factor, col in target_factors.items():
            if col not in df_ortho.columns:
                # データ欠落時の安全策
                stats[factor] = {'median': 0.0, 'mad': 1.0, 'col': col}
                continue

            series = df_ortho[col].dropna()

            if series.empty:
                stats[factor] = {'median': 0.0, 'mad': 1.0, 'col': col}
            else:
                # ---------------------------------------------------------
                # 【ロバスト統計の核心】 平均・標準偏差ではなく、中央値・MADを使う
                # ---------------------------------------------------------
                median_val = series.median()
                
                # MAD (Median Absolute Deviation) の計算
                # データと中央値の差の絶対値をとり、そのさらに中央値をとる
                abs_deviation = np.abs(series - median_val)
                mad_val = abs_deviation.median()
                
                # MADが0になってしまう場合（データの過半数が同じ値など）の安全策
                if mad_val == 0:
                    # 代わりに標準偏差を使うか、最小値を設定してゼロ除算を防ぐ
                    mad_val = series.std() if series.std() > 0 else 1.0

                stats[factor] = {
                    'median': median_val,
                    'mad': mad_val,
                    'col': col
                }
                
                # 直交化R²がある場合は、Qualityファクター情報に追加しておく
                if factor == 'Quality':
                    stats[factor]['r_squared'] = ortho_params.get('r_squared', 0.0)

        return stats, df_ortho

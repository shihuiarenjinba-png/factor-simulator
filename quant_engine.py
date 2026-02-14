import pandas as pd
import numpy as np
from scipy.stats import linregress

class QuantEngine:
    """
    【Module 2】 計算ロジック担当
    生データを統計的に処理し、投資判断可能なZスコアに変換するクラス
    """

    @staticmethod
    def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
        """
        【外れ値処理】
        極端な異常値（上下1%など）を閾値で丸める処理。
        これにより、1つの異常値で全体の偏差が壊れるのを防ぐ。
        """
        if series.empty:
            return series
        
        lower = series.quantile(lower_percentile)
        upper = series.quantile(upper_percentile)
        return series.clip(lower=lower, upper=upper)

    @staticmethod
    def process_raw_factors(df):
        """
        生データ(Raw)から、分析用の指標(Metrics)を計算する
        - Size: 対数変換
        - Value: PBRの逆数
        """
        df = df.copy()
        
        # 1. Size: 対数変換 (対数正規化)
        # 時価総額の巨大な格差を圧縮する
        # 0以下の値は計算できないのでNaNにする
        df['Size_Log'] = np.log(pd.to_numeric(df['Size_Raw'], errors='coerce').replace(0, np.nan))
        
        # 2. Value: PBRの逆数 (1/PBR)
        # PBRが低いほどValueスコアが高くなるように逆数をとる
        df['Value_Raw'] = pd.to_numeric(df['PBR'], errors='coerce')
        df['Value_Metric'] = df['Value_Raw'].apply(lambda x: 1/x if (pd.notnull(x) and x > 0) else np.nan)
        
        # 他の指標はそのまま使用（名称統一のためコピー）
        df['Momentum_Metric'] = pd.to_numeric(df['Momentum_Raw'], errors='coerce')
        df['Quality_Metric'] = pd.to_numeric(df['ROE'], errors='coerce')      # ROE
        df['Investment_Metric'] = pd.to_numeric(df['Growth'], errors='coerce') # 売上成長率
        
        return df

    @staticmethod
    def calculate_orthogonalization(df):
        """
        【直交化】
        Quality (ROE) と Investment (Growth) の重複を取り除く。
        成長要因を差し引いた「純粋な収益力」を算出する。
        """
        df = df.copy()
        
        # 両方のデータがある銘柄だけを対象に回帰分析
        mask = df['Quality_Metric'].notna() & df['Investment_Metric'].notna()
        
        # データが少なすぎる場合は直交化しない
        if mask.sum() < 10:
            df['Quality_Orthogonal'] = df['Quality_Metric']
            return df, {'slope': 0, 'intercept': 0}
            
        # 回帰分析: Quality = a * Investment + b + Error
        # この Error (残差) こそが、成長に依存しない純粋なQuality
        x = df.loc[mask, 'Investment_Metric']
        y = df.loc[mask, 'Quality_Metric']
        res = linregress(x, y)
        
        slope = res.slope
        intercept = res.intercept
        
        # 全銘柄に対して残差を計算
        # 残差 = 実測ROE - (予測ROE)
        def get_residual(row):
            q = row['Quality_Metric']
            i = row['Investment_Metric']
            if pd.isna(q): return np.nan
            if pd.isna(i): return q # 相手がない場合はそのまま
            return q - (slope * i + intercept)
            
        df['Quality_Orthogonal'] = df.apply(get_residual, axis=1)
        
        return df, {'slope': slope, 'intercept': intercept}

    @staticmethod
    def compute_z_scores(target_df, stats_dict):
        """
        Zスコア計算 & SMB反転
        事前に計算された市場統計(stats_dict)を用いてスコアリングする
        """
        df = target_df.copy()
        
        factors_map = {
            'Beta': {'col': 'Beta_Raw', 'invert': False},
            'Size': {'col': 'Size_Log', 'invert': True},     # ★重要: SMB反転 (大型=-, 小型=+)
            'Value': {'col': 'Value_Metric', 'invert': False},
            'Momentum': {'col': 'Momentum_Metric', 'invert': False},
            'Quality': {'col': 'Quality_Orthogonal', 'invert': False},
            'Investment': {'col': 'Investment_Metric', 'invert': False}
        }
        
        for factor, config in factors_map.items():
            if factor not in stats_dict:
                continue
                
            col_name = config['col']
            mu = stats_dict[factor]['mean']
            sigma = stats_dict[factor]['std']
            invert = config['invert']
            
            z_col = f"{factor}_Z"
            
            # Zスコア計算関数
            def calc_z(val):
                if pd.isna(val) or sigma == 0:
                    return 0.0
                
                # 外れ値処理（計算時のみ適用、表示用データは元のまま）
                # ここでは簡易的に±3σを超えるものは丸めるロジックを入れることも可能
                
                z = (val - mu) / sigma
                
                # 符号反転 (SMBなど)
                if invert:
                    z = -z
                return z
            
            df[z_col] = df[col_name].apply(calc_z)
            
            # 表示用データの整形（UI側で使いやすくするため）
            # ここでは計算ロジックのみを提供し、フォーマットはView側(App)に任せるのが設計の基本だが
            # 便宜上、生データも分かりやすい列名で残しておく
            df[f"{factor}_Display_Raw"] = df[col_name]
            
        return df

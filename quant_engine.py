import pandas as pd
import numpy as np
from scipy.stats import linregress

class QuantEngine:
    """
    【Module 2】 高精度計算エンジン (Pro Version)
    
    特徴:
    1. ロバスト統計 (Robust Statistics): 平均・標準偏差ではなく、中央値・MADを用いて外れ値の影響を排除。
    2. 因子直交化 (Orthogonalization): 成長要因を排除した純粋なQualityを算出。
    3. 統計的適合度 (R-squared): モデルの信頼性を数値化して返す。
    """

    @staticmethod
    def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
        """
        【外れ値処理】
        極端な異常値（上下1%など）を閾値で丸める処理。
        計算が破綻するのを防ぐための第一防衛ライン。
        """
        if series.empty:
            return series
        
        # 数値データのみを対象にする
        series = pd.to_numeric(series, errors='coerce')
        
        lower = series.quantile(lower_percentile)
        upper = series.quantile(upper_percentile)
        return series.clip(lower=lower, upper=upper)

    @staticmethod
    def calculate_robust_z_score(val, median, mad):
        """
        【ロバストZスコア計算】
        Modified Z-Score = 0.6745 * (x - median) / MAD
        
        通常のZスコアは異常値に弱いが、これは中央値ベースなので
        「トヨタ」や「ユニクロ」のような巨大企業が混ざっても基準がズレない。
        """
        if pd.isna(val) or mad == 0:
            return 0.0
        
        # 正規分布相当に補正するための係数 0.6745 (approx 1/1.4826)
        z = 0.6745 * (val - median) / mad
        
        # 実用的な範囲（±5.0）にクリップしてグラフを見やすくする
        return max(min(z, 5.0), -5.0)

    @staticmethod
    def process_raw_factors(df):
        """
        生データ(Raw)から、分析用の指標(Metrics)を厳密に計算する
        """
        df = df.copy()
        
        # 1. Size: 対数変換 (Log Normalization)
        # 時価総額は桁が違いすぎるため、対数をとって正規分布に近づける
        df['Size_Log'] = np.log(pd.to_numeric(df['Size_Raw'], errors='coerce').replace(0, np.nan))
        
        # 2. Value: PBRの逆数 (Book-to-Market)
        # 学術的には B/M (1/PBR) が正統なバリュー指標
        df['Value_Raw'] = pd.to_numeric(df['PBR'], errors='coerce')
        df['Value_Metric'] = df['Value_Raw'].apply(lambda x: 1/x if (pd.notnull(x) and x > 0) else np.nan)
        
        # 他の指標の数値化（エラーハンドリング付き）
        df['Momentum_Metric'] = pd.to_numeric(df['Momentum_Raw'], errors='coerce')
        df['Quality_Metric'] = pd.to_numeric(df['ROE'], errors='coerce')      # ROE
        df['Investment_Metric'] = pd.to_numeric(df['Growth'], errors='coerce') # 売上成長率
        
        return df

    @staticmethod
    def calculate_orthogonalization(df):
        """
        【直交化 & R²算出】
        Quality (ROE) から Investment (Growth) の影響を回帰分析で取り除く。
        同時に、その回帰の「決定係数(R²)」を算出し、分析の信頼度として返す。
        """
        df = df.copy()
        mask = df['Quality_Metric'].notna() & df['Investment_Metric'].notna()
        
        # データ不足時の安全策
        if mask.sum() < 10:
            df['Quality_Orthogonal'] = df['Quality_Metric']
            return df, {'slope': 0, 'intercept': 0, 'r_squared': 0.0}
            
        x = df.loc[mask, 'Investment_Metric']
        y = df.loc[mask, 'Quality_Metric']
        
        # 線形回帰 (SciPy使用)
        res = linregress(x, y)
        
        slope = res.slope
        intercept = res.intercept
        r_squared = res.rvalue ** 2  # 決定係数
        
        # 残差（Residuals）= 純粋なQuality
        def get_residual(row):
            q = row['Quality_Metric']
            i = row['Investment_Metric']
            if pd.isna(q): return np.nan
            if pd.isna(i): return q
            return q - (slope * i + intercept)
            
        df['Quality_Orthogonal'] = df.apply(get_residual, axis=1)
        
        return df, {'slope': slope, 'intercept': intercept, 'r_squared': r_squared}

    @staticmethod
    def compute_z_scores(target_df, stats_dict):
        """
        Zスコア計算 & SMB反転 & R²付与
        Module 3で作られた「市場のモノサシ(stats_dict)」を使ってスコアリングする。
        """
        df = target_df.copy()
        
        # ファクター設定
        # invert: Trueなら「値が大きいほどスコアをマイナスにする」（例：サイズ因子）
        factors_map = {
            'Beta':      {'col': 'Beta_Raw',           'invert': False},
            'Size':      {'col': 'Size_Log',           'invert': True},  # 小型株効果（Small is Plus）
            'Value':     {'col': 'Value_Metric',       'invert': False},
            'Momentum':  {'col': 'Momentum_Metric',    'invert': False},
            'Quality':   {'col': 'Quality_Orthogonal', 'invert': False},
            'Investment':{'col': 'Investment_Metric',  'invert': False}
        }
        
        r_squared_values = {} # 各ファクターの信頼度（あれば）
        
        for factor, config in factors_map.items():
            if factor not in stats_dict:
                continue
            
            # 統計データの取り出し
            stat = stats_dict[factor]
            median = stat.get('median', 0)
            mad = stat.get('mad', 1) # MADがない場合は1（標準偏差代用）として扱う
            
            col_name = config['col']
            invert = config['invert']
            z_col = f"{factor}_Z"
            
            # ロバストZスコアの適用
            df[z_col] = df[col_name].apply(lambda x: QuantEngine.calculate_robust_z_score(x, median, mad))
            
            # 符号反転 (SMB: 大型株をマイナス評価にする等)
            if invert:
                df[z_col] = -df[z_col]
            
            # UI表示用に生データも残す
            df[f"{factor}_Display_Raw"] = df[col_name]

            # R²情報の取得（もし統計データに含まれていれば）
            if 'r_squared' in stat:
                r_squared_values[factor] = stat['r_squared']
            else:
                r_squared_values[factor] = None
                
        return df, r_squared_values

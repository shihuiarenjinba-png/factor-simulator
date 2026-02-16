import pandas as pd
import numpy as np
from scipy.stats import linregress

class QuantEngine:
    """
    ポートフォリオの数値計算、スコアリング、インサイト生成を担当するエンジン
    【修正版 Step 1】入力データの型変換とエラーハンドリングを強化
    """
    
    @staticmethod
    def calculate_beta_momentum(df_fund, df_hist, benchmark_ticker="1321.T"):
        """
        時系列データからBetaとMomentumを計算し、Fundamental DataFrameに結合して返す
        """
        # --- [Step 1 修正] 入力データの安全化 ---
        
        # 1. df_fund が DataFrame でない場合（リスト等が来た場合）の救済
        if not isinstance(df_fund, pd.DataFrame):
            # リストなら DataFrame に変換を試みる
            try:
                df = pd.DataFrame(df_fund)
                if 'Ticker' not in df.columns and 0 in df.columns:
                    df.rename(columns={0: 'Ticker'}, inplace=True)
            except:
                # 変換不可なら空のDataFrameにして返す（エラー回避）
                return pd.DataFrame()
        else:
            df = df_fund.copy()

        # 2. df_hist が DataFrame でない、または空の場合のデフォルト値設定
        #    文字列などが渡された場合の AttributeError を防ぐ
        if not isinstance(df_hist, pd.DataFrame) or df_hist.empty:
            # カラムが存在しない場合は作成
            if 'Beta_Raw' not in df.columns:
                df['Beta_Raw'] = 1.0
            if 'Momentum_Raw' not in df.columns:
                df['Momentum_Raw'] = 0.0
            return df

        # --- 以下、計算ロジック ---

        # リターン計算
        try:
            rets = df_hist.pct_change().dropna()
        except Exception:
            df['Beta_Raw'] = 1.0
            df['Momentum_Raw'] = 0.0
            return df
        
        # ベンチマークが存在しない場合のフォールバック
        if benchmark_ticker not in rets.columns:
            df['Beta_Raw'] = 1.0
            df['Momentum_Raw'] = 0.0
            return df

        bench_ret = rets[benchmark_ticker]
        bench_var = bench_ret.var()

        # 結果を格納する辞書
        betas = {}
        momenta = {}

        for t in df['Ticker']:
            # Beta
            if t in rets.columns:
                try:
                    cov = rets[t].cov(bench_ret)
                    betas[t] = cov / bench_var if bench_var > 1e-8 else 1.0
                except:
                    betas[t] = 1.0
                
                # Momentum
                try:
                    if t in df_hist.columns:
                        series = df_hist[t].dropna()
                        if not series.empty:
                            p_start = series.iloc[0]
                            p_end = series.iloc[-1]
                            momenta[t] = (p_end / p_start) - 1 if p_start > 0 else 0.0
                        else:
                            momenta[t] = 0.0
                    else:
                        momenta[t] = 0.0
                except:
                    momenta[t] = 0.0
            else:
                betas[t] = 1.0
                momenta[t] = 0.0
        
        df['Beta_Raw'] = df['Ticker'].map(betas)
        df['Momentum_Raw'] = df['Ticker'].map(momenta)
        return df

    @staticmethod
    def process_raw_factors(df):
        """生データをファクター分析用の形式に加工"""
        # Value (PBRの逆数)
        if 'PBR' in df.columns:
            df['Value_Raw'] = df['PBR'].apply(lambda x: 1/x if (pd.notnull(x) and x > 0) else np.nan)
        
        # Size (時価総額の対数)
        if 'Size_Raw' in df.columns:
            df['Size_Log'] = np.log(pd.to_numeric(df['Size_Raw'], errors='coerce').replace(0, np.nan))
        
        # カラム名統一
        if 'ROE' in df.columns:
            df['Quality_Raw'] = df['ROE']
        if 'Growth' in df.columns:
            df['Investment_Raw'] = df['Growth']
            
        return df

    @staticmethod
    def compute_z_scores(df_target, stats):
        """市場統計(stats)を用いてZスコアを計算する"""
        df = df_target.copy()
        
        # 1. 直交化
        slope = stats.get('ortho_slope', 0)
        intercept = stats.get('ortho_intercept', 0)
        
        def apply_ortho(row):
            q = row.get('Quality_Raw', np.nan)
            i = row.get('Investment_Raw', np.nan)
            if pd.isna(q): return np.nan
            if pd.isna(i): return q 
            return q - (slope * i + intercept)
            
        df['Quality_Orthogonal'] = df.apply(apply_ortho, axis=1)

        # 2. Zスコア計算
        factors = ['Beta', 'Value', 'Size', 'Momentum', 'Quality', 'Investment']
        r_squared_map = {} 

        for f in factors:
            if f not in stats: continue
            
            if f == 'Quality': col_name = 'Quality_Orthogonal'
            else: col_name = stats[f]['col']
            
            if col_name not in df.columns: continue

            mu = stats[f]['mean']
            sigma = stats[f]['std']
            z_col = f"{f}_Z"
            
            def calc_z(val):
                if pd.isna(val) or sigma == 0: return 0.0
                z = (val - mu) / sigma
                if f == 'Size': z = -z 
                return z
            
            df[z_col] = df[col_name].apply(calc_z)
            
        return df, r_squared_map

    @staticmethod
    def generate_insights(z_scores):
        """Zスコア辞書からインサイト文章を生成"""
        insights = []
        
        # Size
        if z_scores.get('Size', 0) < -1.0:
            insights.append("✅ **大型株中心**: 財務基盤が安定した大型株への配分が高く、市場変動に対する耐久性が期待できます。")
        elif z_scores.get('Size', 0) > 1.0:
            insights.append("🚀 **小型株効果**: 時価総額の小さい銘柄が多く、市場平均を上回る成長ポテンシャルを秘めています。")
            
        # Value
        if z_scores.get('Value', 0) > 1.0:
            insights.append("💰 **バリュー投資**: 純資産に対して割安な銘柄が多く、下値リスクが限定的である可能性があります。")
            
        # Quality
        if z_scores.get('Quality', 0) > 1.0:
            insights.append("💎 **高クオリティ**: ROE等の収益性が市場平均より高く、経営効率の良い企業群です。")
            
        # Momentum
        mom_z = z_scores.get('Momentum', 0)
        if mom_z < -1.0:
            insights.append("🔄 **リバーサル狙い**: 直近で株価が出遅れている銘柄が多く、反発（見直し買い）を狙う構成です。")
        elif mom_z > 1.0:
            insights.append("📈 **モメンタム重視**: 直近の株価パフォーマンスが良い銘柄に乗る「順張り」の傾向があります。")

        if not insights:
            insights.append("⚖️ **市場中立**: 特定のファクターへの極端な偏りがなく、市場全体（インデックス）に近いバランスです。")
            
        return insights

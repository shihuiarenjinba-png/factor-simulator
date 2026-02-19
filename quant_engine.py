import pandas as pd
import numpy as np
from scipy.stats import linregress

class QuantEngine:
    """
    ポートフォリオの数値計算、スコアリング、インサイト生成を担当するエンジン
    【修正版 Step 2】内部名称の統一と欠損フォールバックの強化
    """
    
    @staticmethod
    def calculate_beta_momentum(df_fund, df_hist, benchmark_ticker="1321.T"):
        """時系列データからBetaとMomentumを計算"""
        # 1. df_fund救済
        if not isinstance(df_fund, pd.DataFrame):
            try:
                df = pd.DataFrame(df_fund)
                if 'Ticker' not in df.columns and 0 in df.columns:
                    df.rename(columns={0: 'Ticker'}, inplace=True)
            except:
                return pd.DataFrame()
        else:
            df = df_fund.copy()

        # 2. df_hist救済
        if not isinstance(df_hist, pd.DataFrame) or df_hist.empty:
            if 'Beta_Raw' not in df.columns: df['Beta_Raw'] = 1.0
            if 'Momentum_Raw' not in df.columns: df['Momentum_Raw'] = 0.0
            return df

        # 計算ロジック
        try:
            rets = df_hist.pct_change(fill_method=None).dropna()
        except Exception:
            df['Beta_Raw'] = 1.0
            df['Momentum_Raw'] = 0.0
            return df
        
        if benchmark_ticker not in rets.columns:
            df['Beta_Raw'] = 1.0
            df['Momentum_Raw'] = 0.0
            return df

        bench_ret = rets[benchmark_ticker]
        bench_var = bench_ret.var()

        betas = {}
        momenta = {}

        for t in df['Ticker']:
            if t in rets.columns:
                try:
                    cov = rets[t].cov(bench_ret)
                    betas[t] = cov / bench_var if bench_var > 1e-8 else 1.0
                except:
                    betas[t] = 1.0
                
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
        """
        生データをファクター分析用の形式に加工
        【修正】Sizeの名称同期 (MarketCap) と、Investmentのフォールバック強化
        """
        # Value (PBR逆数)
        if 'PBR' in df.columns:
            df['Value_Raw'] = df['PBR'].apply(lambda x: 1/x if (pd.notnull(x) and x > 0) else np.nan)
        
        # Size (時価総額対数)
        if 'Size_Raw' in df.columns:
            df['Size_Log'] = np.log(pd.to_numeric(df['Size_Raw'], errors='coerce').replace(0, np.nan))
            # 【追加】app.pyの表示ロジックに合わせて 'MarketCap' カラムを明示的に作成
            df['MarketCap'] = pd.to_numeric(df['Size_Raw'], errors='coerce')
        
        # Quality (ROE)
        if 'ROE' in df.columns:
            df['Quality_Raw'] = df['ROE']
        
        # Investment (総資産増加率)
        # Formula: (当期総資産 / 前期総資産) - 1
        if 'Total_Assets' in df.columns and 'Total_Assets_Prev' in df.columns:
            prev = pd.to_numeric(df['Total_Assets_Prev'], errors='coerce')
            curr = pd.to_numeric(df['Total_Assets'], errors='coerce')
            
            # 0除算回避のため、prevが0の場合はNaNにする
            ratio = curr / prev.replace(0, np.nan)
            df['Investment_Raw'] = ratio - 1.0
        else:
            df['Investment_Raw'] = np.nan
            
        # 【追加】総資産が取得できず Investment_Raw が NaN の場合、Growth (売上成長) で穴埋めする
        if 'Growth' in df.columns:
            df['Investment_Raw'] = df['Investment_Raw'].fillna(pd.to_numeric(df['Growth'], errors='coerce'))
            
        return df

    @staticmethod
    def calculate_orthogonalization(df, x_col, y_col):
        """直交化メソッド"""
        df_out = df.copy()
        params = {'slope': 0, 'intercept': 0, 'r_squared': 0}
        col_name = f"{y_col}_Orthogonal"

        try:
            valid_data = df[[x_col, y_col]].dropna()
            if len(valid_data) < 5:
                df_out[col_name] = df_out[y_col]
                return df_out, params

            slope, intercept, r_value, p_value, std_err = linregress(valid_data[x_col], valid_data[y_col])
            
            params = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2
            }
            
            def apply_resid(row):
                y = row.get(y_col, np.nan)
                x = row.get(x_col, np.nan)
                if pd.isna(y) or pd.isna(x):
                    return y 
                return y - (slope * x + intercept)

            df_out[col_name] = df_out.apply(apply_resid, axis=1)
            return df_out, params

        except Exception as e:
            if col_name not in df_out.columns:
                df_out[col_name] = df_out[y_col]
            return df_out, params

    @staticmethod
    def compute_z_scores(df_target, stats):
        """
        Zスコア計算
        """
        df = df_target.copy()
        
        slope = stats.get('ortho_slope', 0)
        intercept = stats.get('ortho_intercept', 0)
        
        def apply_ortho(row):
            q = row.get('Quality_Raw', np.nan)
            i = row.get('Investment_Raw', np.nan)
            if pd.isna(q): return np.nan
            if pd.isna(i): return q
            return q - (slope * i + intercept)
            
        df['Quality_Raw_Orthogonal'] = df.apply(apply_ortho, axis=1)
        df['Quality_Orthogonal'] = df['Quality_Raw_Orthogonal']

        factors = ['Beta', 'Value', 'Size', 'Momentum', 'Quality', 'Investment']
        r_squared_map = {} 

        for f in factors:
            if f not in stats: continue
            
            target_col = stats[f]['col']
            
            if target_col not in df.columns:
                if f == 'Quality':
                    if 'Quality_Raw_Orthogonal' in df.columns: target_col = 'Quality_Raw_Orthogonal'
                    elif 'Quality_Orthogonal' in df.columns: target_col = 'Quality_Orthogonal'
                    else: continue
                else:
                    continue

            mu = stats[f].get('median', 0)
            sigma = stats[f].get('mad', 1)
            if sigma == 0: sigma = 1e-6

            z_col = f"{f}_Z"
            
            def calc_z(val):
                if pd.isna(val): return 0.0 
                z = (val - mu) / sigma
                
                # サイズとInvestmentの反転ロジック
                # Size: 小さいほどプラス (小型株効果)
                # Investment: 資産拡大が小さい(Conservative)ほどプラス
                if f == 'Size' or f == 'Investment': 
                    z = -z 
                
                # クリップ処理
                if z > 3.0: z = 3.0
                if z < -3.0: z = -3.0
                return z
            
            df[z_col] = df[target_col].apply(calc_z)
            
        return df, r_squared_map

    @staticmethod
    def generate_insights(z_scores):
        """インサイト生成 (Step 4準拠)"""
        insights = []
        
        z_size = z_scores.get('Size', 0)
        z_val  = z_scores.get('Value', 0)
        z_qual = z_scores.get('Quality', 0)
        z_mom  = z_scores.get('Momentum', 0)
        z_inv  = z_scores.get('Investment', 0)

        # 1. Size
        if z_size < -0.7:
            insights.append("🐘 **大型株中心**: 財務基盤が安定した大型株への配分が高く、市場変動に対する耐久性が期待できます。")
        elif z_size > 0.7:
            insights.append("🚀 **小型株効果**: 時価総額の小さい銘柄が多く、市場平均を上回る成長ポテンシャルを秘めています。")
        
        # 2. Value
        if z_val > 0.7:
            insights.append("💰 **バリュー投資**: 純資産に対して割安な銘柄が多く、下値リスクが限定的である可能性があります。")
        elif z_val < -0.7:
            insights.append("💎 **グロース寄り**: 将来の成長期待が高い銘柄が含まれており、割高でも買われている傾向があります。")

        # 3. Quality
        if z_qual > 0.7:
            insights.append("👑 **高クオリティ**: 収益性(ROE)が高く、経営効率の良い「質の高い」企業群です。")
            
        # 4. Momentum
        if z_mom > 0.7:
            insights.append("📈 **順張りトレンド**: 直近のパフォーマンスが良い銘柄に乗る「モメンタム重視」の構成です。")
        elif z_mom < -0.7:
            insights.append("🔄 **逆張り/出遅れ**: 直近で株価が軟調な銘柄が多く、反発（リバーサル）狙いの可能性があります。")

        # 5. Investment
        if z_inv > 0.7:
            insights.append("🛡️ **保守的経営**: 資産拡大を抑え、筋肉質な経営を行っている企業群です（CMA効果）。")
        elif z_inv < -0.7:
            insights.append("🏗️ **積極投資**: 設備投資や資産拡大に積極的な企業が含まれています（過剰投資リスクに注意）。")

        # 複合条件
        if z_qual > 0.5 and z_val > 0.5:
            insights.append("✨ **クオリティ・バリュー**: 質が高いのに割安に放置されている、理想的な銘柄群が含まれています。")
        if z_size > 0.5 and z_mom > 0.5:
            insights.append("🔥 **小型モメンタム**: 小型株かつ上昇トレンドにある、爆発力のある構成です。")

        if not insights:
            insights.append("⚖️ **市場中立 (バランス型)**: 特定のファクターへの偏りが少なく、インデックス（市場平均）に近い安定した構成です。")
            
        return insights

import pandas as pd
import numpy as np
from scipy.stats import linregress

class QuantEngine:
    """
    ポートフォリオの数値計算、スコアリング、インサイト生成を担当するエンジン
    【修正版 Step 4】BS不要化とInvestment（Growth）の統合、異常値のNaN除外強化
    【工程3】引き渡しミスの徹底排除と診断スピードの極限化
    【完了版 Step 6】ZスコアのNone撲滅と5ファクター完全保証ロジックの追加
    """
    
    @staticmethod
    def calculate_beta(df_fund, df_hist, benchmark_ticker="1321.T"):
        """時系列データからBetaのみを計算（異常値はNaN化）"""
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
            if 'Beta_Raw' not in df.columns: df['Beta_Raw'] = np.nan
            return df

        # 計算ロジック
        try:
            # Pandasのバージョンによるエラーを回避するため、引数をシンプル化
            rets = df_hist.pct_change().dropna()
        except Exception:
            df['Beta_Raw'] = np.nan
            return df
        
        if benchmark_ticker not in rets.columns:
            df['Beta_Raw'] = np.nan
            return df

        bench_ret = rets[benchmark_ticker]
        bench_var = bench_ret.var()

        betas = {}

        for t in df['Ticker']:
            if t in rets.columns:
                try:
                    cov = rets[t].cov(bench_ret)
                    # エラー値や極端な分散の場合はNaNとして除外
                    betas[t] = cov / bench_var if bench_var > 1e-8 else np.nan
                except:
                    betas[t] = np.nan
            else:
                betas[t] = np.nan
        
        df['Beta_Raw'] = df['Ticker'].map(betas)
        return df

    @staticmethod
    def process_raw_factors(df):
        """
        生データをファクター分析用の形式に加工
        【修正】BS（Total_Assets）依存を完全廃止し、Growthカラムを採用
        """
        # Value (PBR逆数)
        if 'PBR' in df.columns:
            df['Value_Raw'] = df['PBR'].apply(lambda x: 1/x if (pd.notnull(x) and x > 0) else np.nan)
        
        # Size (時価総額対数)
        if 'Size_Raw' in df.columns:
            df['Size_Log'] = np.log(pd.to_numeric(df['Size_Raw'], errors='coerce').replace(0, np.nan))
            # app.pyの表示ロジックに合わせて 'MarketCap' カラムを明示的に作成
            df['MarketCap'] = pd.to_numeric(df['Size_Raw'], errors='coerce')
        
        # Quality (ROE)
        if 'ROE' in df.columns:
            df['Quality_Raw'] = df['ROE']
        
        # Investment (資産成長率)
        # 【修正】BSの読み込みを廃止したため、DataProviderで取得した 'Growth' 
        # (FMPのAsset Growth または infoのRevenue Growth) を直接代入して計算負荷をゼロにする
        try:
            if 'Growth' in df.columns:
                df['Investment_Raw'] = pd.to_numeric(df['Growth'], errors='coerce')
            else:
                df['Investment_Raw'] = np.nan
        except Exception:
            df['Investment_Raw'] = np.nan
            
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
                df_out[col_name] = df_out.get(y_col, np.nan)
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
                df_out[col_name] = df_out.get(y_col, np.nan)
            return df_out, params

    @staticmethod
    def compute_z_scores(df_target, stats):
        """
        Zスコア計算 (市場全体の直交化パラメータを適用し、引き渡しミスを根絶)
        """
        df = df_target.copy()
        r_squared_map = {} 
        
        # 市場全体（ベンチマーク）で算出した回帰係数を取得
        slope = stats.get('ortho_slope', 0)
        intercept = stats.get('ortho_intercept', 0)
        
        def apply_ortho(row):
            try:
                q = row.get('Quality_Raw', np.nan)
                i = row.get('Investment_Raw', np.nan)
                if pd.isna(q): return np.nan
                if pd.isna(i): return q
                # 市場全体の「基準」を使って、ユーザーの銘柄のQualityからInvestmentの影響を除く
                return q - (slope * i + intercept)
            except Exception:
                return np.nan
            
        if 'Quality_Raw' in df.columns:
            df['Quality_Raw_Orthogonal'] = df.apply(apply_ortho, axis=1)
            df['Quality_Orthogonal'] = df['Quality_Raw_Orthogonal']

        # 評価対象の5ファクターを定義
        factors = ['Beta', 'Value', 'Size', 'Quality', 'Investment']

        # 【フォールバック用辞書】 カラム名が多少ズレても、関連する生データを使って計算を完遂させる
        fallback_cols = {
            'Quality': ['Quality_Raw_Orthogonal', 'Quality_Orthogonal', 'Quality_Raw', 'ROE'],
            'Value': ['Value_Raw', 'PBR'],
            'Size': ['Size_Log', 'Size_Raw', 'MarketCap'],
            'Investment': ['Investment_Raw', 'Growth'],
            'Beta': ['Beta_Raw']
        }

        for f in factors:
            z_col = f"{f}_Z"
            
            # stats（基準）が存在しない場合は 0.0 で初期化してスキップ
            if f not in stats: 
                df[z_col] = 0.0
                continue
            
            target_col = stats[f].get('col', None)
            
            # 引き渡しミス防止：target_colがユーザーDFにない場合の救済措置
            if target_col not in df.columns:
                found_col = None
                for candidate in fallback_cols.get(f, []):
                    if candidate in df.columns:
                        found_col = candidate
                        break
                
                if not found_col:
                    # データが全くない場合は、市場平均(0.0)として扱いエラーを防ぐ
                    df[z_col] = 0.0
                    continue 
                target_col = found_col

            mu = stats[f].get('median', 0)
            sigma = stats[f].get('mad', 1)
            if sigma == 0: sigma = 1e-6

            def calc_z(val):
                # 【重要】欠損値や無限大は「市場平均（ゼロ）」として扱い、Noneを根絶する
                if pd.isna(val) or np.isinf(val): return 0.0 
                z = (val - mu) / sigma
                
                # サイズとInvestmentの反転ロジック
                # Size: 小さいほどプラス (小型株効果)
                # Investment: 資産拡大が小さい(Conservative)ほどプラス
                if f == 'Size' or f == 'Investment': 
                    z = -z 
                
                # クリップ処理 (異常値のWinsorization)
                if z > 3.0: z = 3.0
                if z < -3.0: z = -3.0
                return z
            
            # 安全に計算するため、強制的に数値型へ変換してから適用
            numeric_series = pd.to_numeric(df[target_col], errors='coerce')
            df[z_col] = numeric_series.apply(calc_z)
            
        # 【最終防衛線】 5ファクターの列が確実に存在することを保証する
        for f in factors:
            z_col = f"{f}_Z"
            if z_col not in df.columns:
                df[z_col] = 0.0
            else:
                # 最終的に NaN が残ってしまった場合も 0.0 で埋める
                df[z_col] = df[z_col].fillna(0.0)
            
        return df, r_squared_map

    @staticmethod
    def generate_insights(z_scores):
        """インサイト生成 (5ファクター対応版)"""
        insights = []
        
        z_size = z_scores.get('Size', 0)
        z_val  = z_scores.get('Value', 0)
        z_qual = z_scores.get('Quality', 0)
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

        # 4. Investment
        if z_inv > 0.7:
            insights.append("🛡️ **保守的経営**: 資産拡大を抑え、筋肉質な経営を行っている企業群です（CMA効果）。")
        elif z_inv < -0.7:
            insights.append("🏗️ **積極投資**: 設備投資や資産拡大に積極的な企業が含まれています（過剰投資リスクに注意）。")

        # 複合条件
        if z_qual > 0.5 and z_val > 0.5:
            insights.append("✨ **クオリティ・バリュー**: 質が高いのに割安に放置されている、理想的な銘柄群が含まれています。")

        if not insights:
            insights.append("⚖️ **市場中立 (バランス型)**: 特定のファクターへの偏りが少なく、インデックス（市場平均）に近い安定した構成です。")
            
        return insights

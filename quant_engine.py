import pandas as pd
import numpy as np
from scipy.stats import linregress

class QuantEngine:
    """
    ポートフォリオの数値計算、スコアリング、インサイト生成を担当するエンジン
    【修正版 Step 4】BS不要化とInvestment（Growth）の統合、異常値のNaN除外強化
    【工程3】引き渡しミスの徹底排除と診断スピードの極限化
    【完了版 Step 6】ZスコアのNone撲滅、自己補完フォールバックによる0.00病の根絶
    【NEW版 Step 7】Rm-RfベースのBeta計算、時価総額加重(MCW)ロジック、ファクター相関行列の追加
    【最新修正版】Zスコア上限撤廃、Value/Sizeの厳密な対数化、ウェイト入力のガードレール強化
    """
    
    @staticmethod
    def calculate_beta(df_fund, df_hist, df_market=None, benchmark_ticker="1321.T"):
        """
        時系列データからBetaを計算（Rm-Rfの市場プレミアム対応版）
        """
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
            rets = df_hist.pct_change().dropna()
        except Exception:
            df['Beta_Raw'] = np.nan
            return df
        
        betas = {}

        # 【追加実装】 Rm と Rf を使った理論的Betaの計算ルート
        if df_market is not None and not df_market.empty and 'Rm' in df_market.columns and 'Rf' in df_market.columns:
            try:
                # Rmは日経平均などの日次リターン、Rfは日次換算済みの無リスク利子率
                rm_ret = df_market['Rm'].pct_change().dropna()
                rf_daily = df_market['Rf'].reindex(rm_ret.index).ffill()
                
                # 市場プレミアム (Rm - Rf)
                market_premium = rm_ret - rf_daily
                bench_var = market_premium.var()

                for t in df['Ticker']:
                    if t in rets.columns:
                        try:
                            asset_ret = rets[t]
                            # インデックスを揃えて欠損値を排除
                            aligned_data = pd.concat([asset_ret, market_premium, rf_daily], axis=1, join='inner').dropna()
                            aligned_data.columns = ['Asset', 'MarketPremium', 'Rf']
                            
                            # 個別銘柄の超過リターン (Ri - Rf)
                            asset_premium = aligned_data['Asset'] - aligned_data['Rf']
                            
                            # 共分散 / 分散
                            cov = asset_premium.cov(aligned_data['MarketPremium'])
                            betas[t] = cov / bench_var if bench_var > 1e-8 else np.nan
                        except:
                            betas[t] = np.nan
                    else:
                        betas[t] = np.nan
            except Exception:
                # 失敗時は通常のベンチマーク計算へフォールバック
                df_market = None 

        # 【既存ロジック】 df_marketがない場合のフォールバック（通常のベンチマークリターン）
        if df_market is None or df_market.empty:
            if benchmark_ticker not in rets.columns:
                df['Beta_Raw'] = np.nan
                return df

            bench_ret = rets[benchmark_ticker]
            bench_var = bench_ret.var()

            for t in df['Ticker']:
                if t in rets.columns:
                    try:
                        cov = rets[t].cov(bench_ret)
                        betas[t] = cov / bench_var if bench_var > 1e-8 else np.nan
                    except:
                        betas[t] = np.nan
                else:
                    betas[t] = np.nan
        
        df['Beta_Raw'] = df['Ticker'].map(betas)
        return df

    # =========================================================================
    # 【修正】 重みの決定ロジック ガードレール設置
    # =========================================================================
    @staticmethod
    def calculate_portfolio_weights(df, user_weights_provided=False):
        """
        ポートフォリオの重みを算出。
        手入力ウェイトの不整合（数が合わない、文字が混ざっている等）による
        計算のショート（全値0化）を防ぐ強力なガードレールを追加。
        """
        df_out = df.copy()
        
        # 1. ユーザー指定ルート
        if user_weights_provided and 'Weight' in df_out.columns:
            # 強制的に数値化、変換できない文字は0に
            df_out['Weight'] = pd.to_numeric(df_out['Weight'], errors='coerce').fillna(0)
            
            # 【ガードレール】有効な数値が入力されているかチェック
            valid_weights_count = (df_out['Weight'] > 0).sum()
            total_weight = df_out['Weight'].sum()
            
            # 合計が0より大きく、かつ入力されたウェイトの数が銘柄数と一致しているか
            if total_weight > 0 and valid_weights_count == len(df_out):
                df_out['Weight'] = df_out['Weight'] / total_weight
                return df_out
            else:
                # 数が合わない、あるいは不正な値により0になってしまった場合は「等金額加重」に強制フォールバック
                df_out['Weight'] = 1.0 / len(df_out)
                return df_out
        
        # 2. 自動計算ルート (時価総額加重)
        if 'MarketCap' in df_out.columns:
            valid_mc = pd.to_numeric(df_out['MarketCap'], errors='coerce').fillna(0)
            mc_sum = valid_mc.sum()
            if mc_sum > 0:
                df_out['Weight'] = valid_mc / mc_sum
            else:
                # 時価総額が取れなかった場合
                df_out['Weight'] = 1.0 / len(df_out)
        else:
            df_out['Weight'] = 1.0 / len(df_out)
            
        return df_out

    # =========================================================================
    # ファクター相関行列の算出
    # =========================================================================
    @staticmethod
    def calculate_factor_correlation(df):
        factors = ['Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z']
        existing_factors = [f for f in factors if f in df.columns]
        
        if len(existing_factors) > 1:
            corr_matrix = df[existing_factors].corr().fillna(0)
            return corr_matrix
        return pd.DataFrame()

    @staticmethod
    def process_raw_factors(df):
        """
        生データをファクター分析用の形式に加工
        【修正】ValueとSizeに確実な対数(Log)処理を適用し、数値の爆発を抑制
        """
        # Value (PBR逆数の対数化)
        if 'PBR' in df.columns:
            # PBRが0以下などの異常値はNaNにし、正常値のみ np.log(1/PBR) を適用
            df['Value_Raw'] = df['PBR'].apply(
                lambda x: np.log(1/x) if (pd.notnull(x) and x > 0) else np.nan
            )
        
        # Size (時価総額対数)
        if 'Size_Raw' in df.columns:
            # エラー文字を除去し数値化
            raw_size = pd.to_numeric(df['Size_Raw'], errors='coerce')
            # 確実に対数化（マイナスやゼロはNaNにして0.00病を防ぐ）
            df['Size_Log'] = raw_size.apply(
                lambda x: np.log(x) if (pd.notnull(x) and x > 0) else np.nan
            )
            df['MarketCap'] = raw_size
        
        # Quality (ROE)
        if 'ROE' in df.columns:
            df['Quality_Raw'] = pd.to_numeric(df['ROE'], errors='coerce')
        
        # Investment (資産成長率)
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
        Zスコア計算 
        【修正】3.0のクリップ（リミッター）を解除し、順位を視覚化できるように変更。
        また、0.00病を防ぐため、MAD(中央値絶対偏差)を標準偏差に近似させるスケーリングを追加。
        """
        df = df_target.copy()
        r_squared_map = {} 
        
        slope = stats.get('ortho_slope', 0) if isinstance(stats, dict) else 0
        intercept = stats.get('ortho_intercept', 0) if isinstance(stats, dict) else 0
        
        def apply_ortho(row):
            try:
                q = row.get('Quality_Raw', np.nan)
                i = row.get('Investment_Raw', np.nan)
                if pd.isna(q): return np.nan
                if pd.isna(i): return q
                return q - (slope * i + intercept)
            except Exception:
                return np.nan
            
        if 'Quality_Raw' in df.columns:
            df['Quality_Raw_Orthogonal'] = df.apply(apply_ortho, axis=1)
            df['Quality_Orthogonal'] = df['Quality_Raw_Orthogonal']

        factors = ['Beta', 'Value', 'Size', 'Quality', 'Investment']

        fallback_cols = {
            'Quality': ['Quality_Raw_Orthogonal', 'Quality_Orthogonal', 'Quality_Raw', 'ROE'],
            'Value': ['Value_Raw', 'PBR'],
            'Size': ['Size_Log', 'Size_Raw', 'MarketCap'],
            'Investment': ['Investment_Raw', 'Growth'],
            'Beta': ['Beta_Raw']
        }

        for f in factors:
            z_col = f"{f}_Z"
            
            target_col = stats[f].get('col') if isinstance(stats, dict) and f in stats else None
            
            if not target_col or target_col not in df.columns:
                found_col = None
                for candidate in fallback_cols.get(f, []):
                    if candidate in df.columns:
                        found_col = candidate
                        break
                
                if not found_col:
                    df[z_col] = 0.0
                    continue 
                target_col = found_col

            # 強制的に数値型へ変換、無限大はNaNにして0.00病を防止
            numeric_series = pd.to_numeric(df[target_col], errors='coerce').replace([np.inf, -np.inf], np.nan)

            if isinstance(stats, dict) and f in stats and 'median' in stats[f]:
                mu = stats[f]['median']
                # 市場MADに1.4826を掛けて標準偏差(SD)相当にスケールアップし、Zスコアの異常な肥大化を防ぐ
                raw_sigma = stats[f].get('mad', 1e-6)
                sigma = (raw_sigma * 1.4826) if pd.notna(raw_sigma) and raw_sigma > 1e-6 else 1e-6
            else:
                mu = numeric_series.median()
                mad = (numeric_series - mu).abs().median()
                # 自己計算MADにも同様に1.4826を掛ける
                sigma = (mad * 1.4826) if pd.notna(mad) and mad > 0 else 1e-6
                
                if pd.isna(mu):
                    df[z_col] = 0.0
                    continue

            if sigma == 0: sigma = 1e-6

            def calc_z(val):
                if pd.isna(val) or np.isinf(val): return np.nan 
                
                z = (val - mu) / sigma
                
                if f == 'Size' or f == 'Investment': 
                    z = -z 
                
                # 【修正: リミッター解除】
                # if z > 3.0: z = 3.0
                # if z < -3.0: z = -3.0
                # 上記を完全に削除しました。これにより突出したスコアもそのまま出力されます。
                
                return z
            
            df[z_col] = numeric_series.apply(calc_z)
            
        for f in factors:
            z_col = f"{f}_Z"
            if z_col not in df.columns:
                df[z_col] = 0.0
            else:
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

        if z_qual > 0.5 and z_val > 0.5:
            insights.append("✨ **クオリティ・バリュー**: 質が高いのに割安に放置されている、理想的な銘柄群が含まれています。")

        if not insights:
            insights.append("⚖️ **市場中立 (バランス型)**: 特定のファクターへの偏りが少なく、インデックス（市場平均）に近い安定した構成です。")
            
        return insights

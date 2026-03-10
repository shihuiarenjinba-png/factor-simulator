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
    【V17.1追加】特性ベータ(Sensitivity Beta)の逆算ロジック、スマートウェイトの完全化、極性正常化
    【第2工程パッチ】分母ゼロ爆発の防止(Sigma Floor)、対数化前のClip処理、Betaフォールバック強化
    """
    
    @staticmethod
    def calculate_beta(df_fund, df_hist, df_market=None, benchmark_ticker="1321.T"):
        """
        時系列データからBetaを計算（Rm-Rfの市場プレミアム対応版）
        ※この値は五角形レーダーチャート用の「回帰ベータ(Regression Beta)」として使用されます。
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
                            # 【ガードレール】分散が極端に小さい場合はNaNにして0.00病を防ぐ
                            betas[t] = cov / bench_var if bench_var > 1e-8 else np.nan
                        except:
                            betas[t] = np.nan
                    else:
                        betas[t] = np.nan
            except Exception:
                # 失敗時は通常のベンチマーク計算へフォールバック
                df_market = None 

        # 【既存ロジック・修正】 df_marketがない場合のフォールバック（通常のベンチマークリターン）
        if df_market is None or df_market.empty:
            # 2重フォールバック：指定ETFがなければ代替を探す
            if benchmark_ticker not in rets.columns:
                if "1306.T" in rets.columns:
                    benchmark_ticker = "1306.T"
                elif "^N225" in rets.columns:
                    benchmark_ticker = "^N225"
                else:
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
    # スマート・ウェイト・エンジン
    # =========================================================================
    @staticmethod
    def calculate_portfolio_weights(df, user_weights_provided=False):
        """
        ポートフォリオの重みを算出。
        手入力ウェイトがない場合、自動的に時価総額（Market Cap）加重計算へ移行します。
        """
        df_out = df.copy()
        
        # 1. ユーザー指定ルート (ウェイト入力がある場合)
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
                # 数が合わない場合は自動的に下の時価総額加重（または等金額）へ流すため pass する
                pass
        
        # 2. 自動計算ルート (時価総額加重へ完全移行)
        if 'MarketCap' in df_out.columns:
            valid_mc = pd.to_numeric(df_out['MarketCap'], errors='coerce').fillna(0)
            mc_sum = valid_mc.sum()
            if mc_sum > 0:
                df_out['Weight'] = valid_mc / mc_sum
            else:
                # 時価総額の合計が0になってしまった場合の最終手段
                df_out['Weight'] = 1.0 / len(df_out)
        else:
            # MarketCap カラム自体が存在しない場合の最終手段
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
        【修正】入力値に対する clip（上限・下限設定）を適用し、対数化による外れ値の暴走を防ぐ
        """
        # Value (PBR逆数の対数化)
        if 'PBR' in df.columns:
            # PBRが極端に低い(0.01等)または高い場合をクリップ (0.1 ~ 100倍の範囲)
            clipped_pbr = pd.to_numeric(df['PBR'], errors='coerce').clip(lower=0.1, upper=100.0)
            # 1/PBR にすることで、「PBRが低い（割安）ほど数値が大きくなる」ように事前反転
            df['Value_Raw'] = clipped_pbr.apply(
                lambda x: np.log(1/x) if (pd.notnull(x) and x > 0) else np.nan
            )
        
        # Size (時価総額対数)
        if 'Size_Raw' in df.columns:
            # エラー文字を除去し数値化、時価総額の最低ラインを1億円(1e8)にクリップしマイナス・ゼロを排除
            raw_size = pd.to_numeric(df['Size_Raw'], errors='coerce')
            clipped_size = raw_size.clip(lower=1e8)
            
            df['Size_Log'] = clipped_size.apply(
                lambda x: np.log(x) if (pd.notnull(x) and x > 0) else np.nan
            )
            df['MarketCap'] = raw_size # MarketCap自体は生の数値を使用（加重平均用）
        
        # Quality (ROE)
        if 'ROE' in df.columns:
            # 異常なROE（数千%など）を排除するため上下限設定 (-200% ~ +200%程度に収める)
            df['Quality_Raw'] = pd.to_numeric(df['ROE'], errors='coerce').clip(lower=-2.0, upper=2.0)
        
        # Investment (資産成長率)
        try:
            if 'Growth' in df.columns:
                df['Investment_Raw'] = pd.to_numeric(df['Growth'], errors='coerce').clip(lower=-1.0, upper=3.0)
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
        【修正】分母(Sigma)のゼロ爆発防止Floor実装、極性の正常化
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

        # 各ファクターについて処理
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

            numeric_series = pd.to_numeric(df[target_col], errors='coerce').replace([np.inf, -np.inf], np.nan)

            # ユニバース(stats)がある場合と、ローカル計算のみの場合で分岐
            if isinstance(stats, dict) and f in stats and 'median' in stats[f]:
                mu = stats[f]['median']
                raw_sigma = stats[f].get('mad', 1e-6)
                sigma = (raw_sigma * 1.4826) if pd.notna(raw_sigma) and raw_sigma > 1e-6 else 1e-6
            else:
                mu = numeric_series.median()
                mad = (numeric_series - mu).abs().median()
                sigma = (mad * 1.4826) if pd.notna(mad) and mad > 0 else 1e-6
                
            if pd.isna(mu):
                df[z_col] = 0.0
                continue

            # 【ガードレール】分母（sigma）に最小閾値（0.01）を設け、Zスコアの異常な爆発を防ぐ
            sigma = max(sigma, 0.01)

            def calc_z(val):
                if pd.isna(val) or np.isinf(val): return np.nan 
                
                z = (val - mu) / sigma
                
                # 【極性の正常化】
                if f == 'Size':
                    # 時価総額が小さいほどプラス（小型株効果）
                    z = -z 
                elif f == 'Investment':
                    # 資産成長率が低い（保守的）ほどプラス
                    z = -z 
                # ※ Valueは process_raw_factors で np.log(1/PBR) に変換済みのためそのまま
                
                return z
            
            df[z_col] = numeric_series.apply(calc_z)
            
        # 最終的にNaNがあれば0.0に置換して0.00病から脱却するが、計算自体は通す
        for f in factors:
            z_col = f"{f}_Z"
            if z_col not in df.columns:
                df[z_col] = 0.0
            else:
                df[z_col] = df[z_col].fillna(0.0)
            
        return df, r_squared_map

    # =========================================================================
    # 特性ベータ（Sensitivity Beta）の逆算ロジック
    # =========================================================================
    @staticmethod
    def calculate_sensitivity_beta(df, market_sensitivities=None):
        """
        Zスコアからファクター感応度を加重平均し、特性ベータ（Sensitivity Beta）を算出する。
        下部の寄与度グラフ用に使用されます。
        """
        if market_sensitivities is None:
            # デフォルトの市場感応度係数
            market_sensitivities = {
                'Size_Z': 0.25,       # 小型株ほど市場変動の影響を受けやすい
                'Value_Z': -0.15,     # 割安株は下値が固く、連動性が低い
                'Quality_Z': -0.20,   # 高収益企業は独自の値動きをしやすく連動性が低い
                'Investment_Z': -0.10 # 保守的な企業は変動がマイルド
            }

        df_out = df.copy()
        sensitivity_sum = pd.Series(0.0, index=df_out.index)
        
        # 係数の絶対値合計で割ることで、Zスコアのスケール感（±3.0程度）を維持する
        total_weight = sum(abs(v) for v in market_sensitivities.values())
        
        for factor_col, weight in market_sensitivities.items():
            if factor_col in df_out.columns:
                sensitivity_sum += df_out[factor_col].fillna(0.0) * weight
        
        if total_weight > 0:
            df_out['Sensitivity_Beta_Z'] = sensitivity_sum / total_weight
        else:
            df_out['Sensitivity_Beta_Z'] = 0.0
            
        return df_out

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

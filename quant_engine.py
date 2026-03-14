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
    【第3工程パッチ】Beta行列演算化、動的直交化、動的感応度、最小銘柄数バリデーション追加、Insight英語化
    【SyntaxError修正パッチ】Zスコア計算時のインデントエラーと不可視文字を完全排除
    【重要修正1】動的な銘柄除外: 欠落データを計算から外し、残存銘柄でウェイトを再配分 (リウェイト)
    【重要修正2】妥当性チェック: ベータ値が完全に「1.0」や「0.0」に張り付く偽装データを検出・排除
    """
    
    @staticmethod
    def calculate_beta(df_fund, df_hist, df_market=None, benchmark_ticker="1321.T"):
        """
        時系列データからBetaを計算（Rm-Rfの市場プレミアム対応版）
        ※行列演算（Vectorized）によりforループのボトルネックを解消
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

        # 【修正】 Rm と Rf を使った理論的Betaの計算ルート（行列演算化）
        if df_market is not None and not df_market.empty and 'Rm' in df_market.columns and 'Rf' in df_market.columns:
            try:
                rm_ret = df_market['Rm'].pct_change().dropna()
                rf_daily = df_market['Rf'].reindex(rm_ret.index).ffill()
                market_premium = rm_ret - rf_daily
                
                # 全銘柄一括でインデックスを揃える
                aligned_data = pd.concat([rets, market_premium.rename('MarketPremium'), rf_daily.rename('Rf')], axis=1, join='inner').dropna()
                
                if not aligned_data.empty:
                    market_prem_aligned = aligned_data['MarketPremium']
                    bench_var = market_prem_aligned.var()

                    if bench_var > 1e-8:
                        # 全銘柄の超過リターンを一括計算
                        rf_aligned = aligned_data['Rf']
                        asset_premiums = aligned_data[rets.columns].sub(rf_aligned, axis=0)
                        
                        # 行列演算による共分散の一括計算
                        covariances = asset_premiums.cov(market_prem_aligned)
                        betas = (covariances / bench_var).to_dict()
                        
                        # 【妥当性チェック】 ベータが完全に「1.0」や「0.0」に張り付く場合は、ベンチマークが流用されたか無効データとみなして除外
                        for k in list(betas.keys()):
                            if abs(betas[k] - 1.0) < 1e-6 or abs(betas[k]) < 1e-6:
                                betas[k] = np.nan
            except Exception:
                # 失敗時は通常のベンチマーク計算へフォールバック
                df_market = None 

        # 【既存ロジック・修正】 df_marketがない場合のフォールバック（行列演算化）
        if df_market is None or df_market.empty:
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

            if bench_var > 1e-8:
                # 行列演算による共分散の一括計算
                covariances = rets.cov(bench_ret)
                betas = (covariances / bench_var).to_dict()
                
                # 【妥当性チェック】 対象がベンチマーク自身でないのにベータが「1.0」に張り付く場合は除外
                for k in list(betas.keys()):
                    if k != benchmark_ticker:
                        if abs(betas[k] - 1.0) < 1e-6 or abs(betas[k]) < 1e-6:
                            betas[k] = np.nan
        
        df['Beta_Raw'] = df['Ticker'].map(betas).astype(float)
        return df

    # =========================================================================
    # スマート・ウェイト・エンジン (動的リウェイト機能搭載)
    # =========================================================================
    @staticmethod
    def calculate_portfolio_weights(df, user_weights_provided=False):
        df_out = df.copy()
        
        # 【動的な銘柄除外】 Beta_Raw が計算不能 (NaN) になった銘柄を「死んだ銘柄」としてマークする
        valid_mask = pd.Series(True, index=df_out.index)
        if 'Beta_Raw' in df_out.columns:
            valid_mask = df_out['Beta_Raw'].notna()
        
        if user_weights_provided and 'Weight' in df_out.columns:
            df_out['Weight'] = pd.to_numeric(df_out['Weight'], errors='coerce').fillna(0)
            
            # 死んだ銘柄のウェイトを強制的に0にする
            df_out.loc[~valid_mask, 'Weight'] = 0.0
            
            valid_weights_count = (df_out['Weight'] > 0).sum()
            total_weight = df_out['Weight'].sum()
            
            if total_weight > 0 and valid_weights_count > 0:
                # 残った健全な銘柄だけで合計が1(100%)になるように再配分(リウェイト)
                df_out['Weight'] = df_out['Weight'] / total_weight
                return df_out
            else:
                pass
        
        if 'MarketCap' in df_out.columns:
            valid_mc = pd.to_numeric(df_out['MarketCap'], errors='coerce').fillna(0)
            # 死んだ銘柄は時価総額0として扱い、ウェイト計算から排除
            valid_mc.loc[~valid_mask] = 0.0
            mc_sum = valid_mc.sum()
            
            if mc_sum > 0:
                df_out['Weight'] = valid_mc / mc_sum
            else:
                # すべての時価総額が不明だが、有効な銘柄が残っている場合は等金額配分
                valid_count = valid_mask.sum()
                if valid_count > 0:
                    df_out['Weight'] = np.where(valid_mask, 1.0 / valid_count, 0.0)
                else:
                    df_out['Weight'] = 1.0 / len(df_out)
        else:
            # 時価総額情報がない場合も、有効な銘柄だけで等金額配分
            valid_count = valid_mask.sum()
            if valid_count > 0:
                df_out['Weight'] = np.where(valid_mask, 1.0 / valid_count, 0.0)
            else:
                df_out['Weight'] = 1.0 / len(df_out)
            
        return df_out

    # =========================================================================
    # ファクター相関行列の算出 (変更なし)
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
        # Value (PBR逆数の対数化)
        if 'PBR' in df.columns:
            clipped_pbr = pd.to_numeric(df['PBR'], errors='coerce').clip(lower=0.1, upper=100.0)
            df['Value_Raw'] = clipped_pbr.apply(
                lambda x: np.log(1/x) if (pd.notnull(x) and x > 0) else np.nan
            )
        
        # Size (時価総額対数)
        if 'Size_Raw' in df.columns:
            raw_size = pd.to_numeric(df['Size_Raw'], errors='coerce')
            clipped_size = raw_size.clip(lower=1e8)
            df['Size_Log'] = clipped_size.apply(
                lambda x: np.log(x) if (pd.notnull(x) and x > 0) else np.nan
            )
            df['MarketCap'] = raw_size
        
        # Quality (ROE)
        if 'ROE' in df.columns:
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
        """直交化メソッド (変更なし)"""
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
    def compute_z_scores(df_target, stats, ortho_pairs=None):
        """
        Zスコア計算 
        ortho_pairs: 直交化したいペアのリスト。例: [('Quality', 'Investment'), ('Value', 'Size')]
                     Noneの場合は後方互換性のため stats 内のパラメータから推論します。
        """
        df = df_target.copy()
        r_squared_map = {} 
        
        # 動的直交化の適用
        if ortho_pairs is None:
            ortho_pairs = [('Quality', 'Investment')] if isinstance(stats, dict) and 'ortho_slope' in stats else []

        for target_factor, predictor_factor in ortho_pairs:
            target_col = f"{target_factor}_Raw"
            predictor_col = f"{predictor_factor}_Raw"
            pair_key = f"ortho_{target_factor}_{predictor_factor}"
            
            # statsから傾き・切片を取得（存在しない場合はクロスセクションで簡易計算）
            if isinstance(stats, dict) and pair_key in stats:
                slope = stats[pair_key].get('slope', 0)
                intercept = stats[pair_key].get('intercept', 0)
            elif isinstance(stats, dict) and 'ortho_slope' in stats and target_factor == 'Quality':
                slope = stats.get('ortho_slope', 0)
                intercept = stats.get('ortho_intercept', 0)
            else:
                try:
                    if target_col in df.columns and predictor_col in df.columns:
                        valid_data = df[[predictor_col, target_col]].dropna()
                        if len(valid_data) >= 3:
                            slope, intercept, _, _, _ = linregress(valid_data[predictor_col], valid_data[target_col])
                        else:
                            slope, intercept = 0, 0
                    else:
                        slope, intercept = 0, 0
                except:
                    slope, intercept = 0, 0
            
            ortho_col_name = f"{target_factor}_Raw_Orthogonal"
            
            def apply_ortho(row):
                try:
                    y = row.get(target_col, np.nan)
                    x = row.get(predictor_col, np.nan)
                    if pd.isna(y): return np.nan
                    if pd.isna(x): return y
                    return y - (slope * x + intercept)
                except Exception:
                    return np.nan
            
            if target_col in df.columns:
                df[ortho_col_name] = df.apply(apply_ortho, axis=1)
                df[f"{target_factor}_Orthogonal"] = df[ortho_col_name]

        factors = ['Beta', 'Value', 'Size', 'Quality', 'Investment']

        fallback_cols = {
            'Quality': ['Quality_Raw_Orthogonal', 'Quality_Orthogonal', 'Quality_Raw', 'ROE'],
            'Value': ['Value_Raw_Orthogonal', 'Value_Orthogonal', 'Value_Raw', 'PBR'],
            'Size': ['Size_Raw_Orthogonal', 'Size_Log', 'Size_Raw', 'MarketCap'],
            'Investment': ['Investment_Raw_Orthogonal', 'Investment_Raw', 'Growth'],
            'Beta': ['Beta_Raw_Orthogonal', 'Beta_Raw']
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

            # 【追加バリデーション】ユニバースが極端に少ない場合のガードレール
            valid_count = numeric_series.notna().sum()
            if valid_count < 3:
                df[z_col] = 0.0
                continue

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
                
                if f == 'Size':
                    z = -z 
                elif f == 'Investment':
                    z = -z 
                
                return z
            
            df[z_col] = numeric_series.apply(calc_z)
            
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
    def calculate_sensitivity_beta(df, market_sensitivities='dynamic'):
        """
        Zスコアからファクター感応度を加重平均し、特性ベータ（Sensitivity Beta）を算出。
        引数に'dynamic'を渡すことで、ユニバース内のBetaとの相関から動的に係数を算出。
        """
        df_out = df.copy()

        if market_sensitivities == 'dynamic':
            market_sensitivities = {}
            factors_to_check = ['Size_Z', 'Value_Z', 'Quality_Z', 'Investment_Z']
            
            # Beta_Zが存在し、かつサンプルサイズが最低限ある場合のみ動的計算
            if 'Beta_Z' in df_out.columns and df_out['Beta_Z'].notna().sum() >= 5:
                for f in factors_to_check:
                    if f in df_out.columns:
                        corr = df_out['Beta_Z'].corr(df_out[f])
                        market_sensitivities[f] = corr if pd.notna(corr) else 0.0
            else:
                # フォールバック (デフォルトの市場感応度係数)
                market_sensitivities = {
                    'Size_Z': 0.25, 
                    'Value_Z': -0.15, 
                    'Quality_Z': -0.20, 
                    'Investment_Z': -0.10 
                }
        elif market_sensitivities is None:
            market_sensitivities = {
                'Size_Z': 0.25, 'Value_Z': -0.15, 'Quality_Z': -0.20, 'Investment_Z': -0.10
            }

        sensitivity_sum = pd.Series(0.0, index=df_out.index)
        
        # 係数の絶対値合計で割ることで、Zスコアのスケール感を維持する
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
        """
        インサイト生成 (5ファクター対応版)
        出力テキストを純粋な英語（絵文字・全角文字なし）に統一。
        """
        insights = []
        
        z_size = z_scores.get('Size', 0)
        z_val  = z_scores.get('Value', 0)
        z_qual = z_scores.get('Quality', 0)
        z_inv  = z_scores.get('Investment', 0)

        # 1. Size
        if z_size < -0.7:
            insights.append("Large Cap Focus: High allocation to large-cap stocks with stable financial foundations, providing resilience against market volatility.")
        elif z_size > 0.7:
            insights.append("Small Cap Effect: Weighted towards smaller market capitalization stocks, offering potential to outperform the market average.")
        
        # 2. Value
        if z_val > 0.7:
            insights.append("Value Investing: Consists of stocks trading at a discount to their book value, potentially limiting downside risk.")
        elif z_val < -0.7:
            insights.append("Growth Tilt: Includes stocks with high future growth expectations, typically trading at premium valuations.")

        # 3. Quality
        if z_qual > 0.7:
            insights.append("High Quality: Dominated by high-quality companies characterized by strong profitability (ROE) and operational efficiency.")

        # 4. Investment
        if z_inv > 0.7:
            insights.append("Conservative Management: Companies maintaining disciplined asset growth and lean operations (CMA effect).")
        elif z_inv < -0.7:
            insights.append("Aggressive Investment: Includes companies aggressively expanding capital expenditures and assets (monitor for over-investment risks).")

        if z_qual > 0.5 and z_val > 0.5:
            insights.append("Quality Value: An ideal mix of high-quality companies that are currently undervalued by the market.")

        if not insights:
            insights.append("Market Neutral (Balanced): Minimal tilt towards specific factors, representing a stable composition closely mirroring the market average.")
            
        return insights

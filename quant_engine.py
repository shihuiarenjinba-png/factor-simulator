import pandas as pd
import numpy as np
from scipy.stats import linregress

# 多変量回帰分析用ライブラリ
try:
    import statsmodels.api as sm
except ImportError:
    sm = None

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
    【重要修正3】多変量回帰分析 (Time-series Regression) の実装と統計的有意性(t値, R^2)の算出
    【重要修正4】加重平均スコアリング: 銘柄ごとの固有ファクタースコアのウェイト寄与度分解
    【最重要修正 V18.0】回帰分析の目的変数を純粋なリターンから「超過リターン(Ri - Rf)」に厳密化
    【最重要修正 V18.1】ファクター同期の厳格化 (Inner Joinと正規化の徹底)、サンプル数バリデーション強化
    【最重要修正 V19.0】二段構え回帰(共通期間一括 vs 個別加重平均)の実装、Adj R2とN数の出力強化
    【最重要修正 V19.1】自動ウェイト計算(時価総額加重)の機能強化とフェールセーフ実装
    【修正 V19.2】process_raw_factors の SettingWithCopyWarning 対策 (.copy() + df.loc[] 使用)
    """
    
    # =========================================================================
    # 時系列多変量回帰分析 (二段構え: 共通期間一括 or 個別加重平均)
    # =========================================================================
    @staticmethod
    def build_individual_regression_table(df_hist_ret, df_weights, df_ff5, min_n_obs=24):
        if df_hist_ret.empty or df_ff5.empty or df_weights.empty or sm is None:
            return pd.DataFrame()

        try:
            hist = df_hist_ret.copy()
            ff5 = df_ff5.copy()

            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            hist.index = pd.to_datetime(hist.index).normalize()

            if ff5.index.tz is not None:
                ff5.index = ff5.index.tz_localize(None)
            ff5.index = pd.to_datetime(ff5.index).normalize()

            valid_tickers = [t for t in df_weights['Ticker'] if t in hist.columns]
            if not valid_tickers:
                return pd.DataFrame()

            weight_map = dict(zip(df_weights['Ticker'], df_weights['Weight']))
            rows = []
            for ticker in valid_tickers:
                ticker_ret = hist[ticker].dropna()
                aligned = pd.concat([ticker_ret.rename('Ret'), ff5], axis=1, join='inner').dropna()
                n_obs = len(aligned)
                if n_obs < min_n_obs:
                    continue

                y = aligned['Ret'] - aligned['rf']
                X = aligned[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']]
                X = sm.add_constant(X)
                try:
                    model = sm.OLS(y, X).fit()
                    rows.append(
                        {
                            'Ticker': ticker,
                            'Weight': float(weight_map.get(ticker, 0.0)),
                            'N': int(n_obs),
                            'Beta': float(model.params.get('mkt_rf', 0.0)),
                            'Size': float(model.params.get('smb', 0.0)),
                            'Value': float(model.params.get('hml', 0.0)),
                            'Quality': float(model.params.get('rmw', 0.0)),
                            'Investment': float(model.params.get('cma', 0.0)),
                            'R_squared': float(model.rsquared),
                            'Adjusted_R_squared': float(model.rsquared_adj),
                            'Alpha': float(model.params.get('const', 0.0)),
                        }
                    )
                except Exception:
                    continue

            if not rows:
                return pd.DataFrame()

            out = pd.DataFrame(rows).sort_values(['Weight', 'N'], ascending=[False, False]).reset_index(drop=True)
            return out
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def run_5factor_regression(df_hist_ret, df_weights, df_ff5, min_n_obs=24, force_individual=False):
        """
        df_hist_ret: 既にリターン化されたヒストリカルデータ (月次を想定)
        df_weights: 'Ticker' と 'Weight' を持つ DataFrame
        df_ff5: Fama-French 5要素 + RF (月次を想定)
        min_n_obs: プランA(一括回帰)を許容する最小共通サンプル数 (デフォルト24ヶ月)
        force_individual: 強制的にプランB(個別回帰の積み上げ)を実行するか
        """
        if df_hist_ret.empty or df_ff5.empty or df_weights.empty or sm is None:
            print("[Regression] 初期データ不足、または statsmodels が未インストールです。")
            return None

        try:
            # インデックスの厳密な同期処理
            if df_hist_ret.index.tz is not None:
                df_hist_ret.index = df_hist_ret.index.tz_localize(None)
            df_hist_ret.index = pd.to_datetime(df_hist_ret.index).normalize()
            
            if df_ff5.index.tz is not None:
                df_ff5.index = df_ff5.index.tz_localize(None)
            df_ff5.index = pd.to_datetime(df_ff5.index).normalize()
            
            valid_tickers = [t for t in df_weights['Ticker'] if t in df_hist_ret.columns]
            if not valid_tickers: 
                print("[Regression] 有効なティッカーが履歴データに存在しません。")
                return None
            
            w_dict = dict(zip(df_weights['Ticker'], df_weights['Weight']))
            w_series = pd.Series({t: w_dict[t] for t in valid_tickers})
            if w_series.sum() == 0: return None
            w_series = w_series / w_series.sum()

            # -------------------------------------------------------------
            # プランA: 共通期間による一括回帰 (Portfolio Regression)
            # -------------------------------------------------------------
            if not force_individual:
                # 全銘柄が欠損していない行だけを残す (完全な共通期間)
                common_rets = df_hist_ret[valid_tickers].dropna()
                
                # ポートフォリオの加重平均リターン
                port_ret = (common_rets * w_series).sum(axis=1)
                
                # ファクターと Inner Join
                aligned = pd.concat([port_ret.rename('Port_Ret'), df_ff5], axis=1, join='inner').dropna()
                n_obs = len(aligned)
                
                if n_obs >= min_n_obs:
                    print(f"[Regression Plan A] 共通期間({n_obs}ヶ月)での一括回帰を実行します。")
                    y = aligned['Port_Ret'] - aligned['rf']
                    X = aligned[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']]
                    X = sm.add_constant(X)
                    
                    model = sm.OLS(y, X).fit()
                    
                    return {
                        'Method': 'Portfolio',
                        'N_Observations': n_obs,
                        'Alpha': model.params.get('const', 0),
                        'Beta': model.params.get('mkt_rf', 0),
                        'Size': model.params.get('smb', 0),
                        'Value': model.params.get('hml', 0),
                        'Quality': model.params.get('rmw', 0),
                        'Investment': model.params.get('cma', 0),
                        'R_squared': model.rsquared,
                        'Adjusted_R_squared': model.rsquared_adj,
                        'p_values': {
                            'Alpha': model.pvalues.get('const', 1),
                            'Beta': model.pvalues.get('mkt_rf', 1)
                        }
                    }
                else:
                    print(f"[Regression] 共通期間が {n_obs} ヶ月しかありません。プランB(個別回帰)へ移行します。")

            # -------------------------------------------------------------
            # プランB: 個別回帰の加重平均 (Individual Aggregation)
            # -------------------------------------------------------------
            print("[Regression Plan B] 各銘柄の個別回帰による加重平均を実行します。")
            indiv_results = []
            total_n_obs = 0
            valid_count = 0
            
            for ticker in valid_tickers:
                ticker_ret = df_hist_ret[ticker].dropna()
                aligned_indiv = pd.concat([ticker_ret.rename('Ret'), df_ff5], axis=1, join='inner').dropna()
                
                n_indiv_obs = len(aligned_indiv)
                if n_indiv_obs < min_n_obs:
                    continue
                    
                y_i = aligned_indiv['Ret'] - aligned_indiv['rf']
                X_i = aligned_indiv[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']]
                X_i = sm.add_constant(X_i)
                
                try:
                    mod_i = sm.OLS(y_i, X_i).fit()
                    indiv_results.append({
                        'Ticker': ticker,
                        'Weight': w_series[ticker],
                        'N': n_indiv_obs,
                        'Alpha': mod_i.params.get('const', 0),
                        'Beta': mod_i.params.get('mkt_rf', 0),
                        'Size': mod_i.params.get('smb', 0),
                        'Value': mod_i.params.get('hml', 0),
                        'Quality': mod_i.params.get('rmw', 0),
                        'Investment': mod_i.params.get('cma', 0),
                        'R_squared': mod_i.rsquared,
                        'Adjusted_R_squared': mod_i.rsquared_adj,
                        'p_Beta': mod_i.pvalues.get('mkt_rf', 1)
                    })
                    total_n_obs += n_indiv_obs
                    valid_count += 1
                except Exception:
                    pass
                    
            if not indiv_results:
                print("[Regression] 有効なデータを持つ銘柄が一つもありませんでした。")
                return None
                
            df_res = pd.DataFrame(indiv_results)
            df_res['Weight'] = df_res['Weight'] / df_res['Weight'].sum()
            
            avg_n_obs = int(total_n_obs / valid_count)
            
            agg_alpha = (df_res['Alpha'] * df_res['Weight']).sum()
            agg_beta = (df_res['Beta'] * df_res['Weight']).sum()
            agg_size = (df_res['Size'] * df_res['Weight']).sum()
            agg_value = (df_res['Value'] * df_res['Weight']).sum()
            agg_quality = (df_res['Quality'] * df_res['Weight']).sum()
            agg_invest = (df_res['Investment'] * df_res['Weight']).sum()
            
            agg_r2 = (df_res['R_squared'] * df_res['Weight']).sum()
            agg_adj_r2 = (df_res['Adjusted_R_squared'] * df_res['Weight']).sum()
            agg_p_beta = (df_res['p_Beta'] * df_res['Weight']).sum()

            return {
                'Method': 'Individual Aggregation',
                'N_Observations': avg_n_obs,
                'Alpha': agg_alpha,
                'Beta': agg_beta,
                'Size': agg_size,
                'Value': agg_value,
                'Quality': agg_quality,
                'Investment': agg_invest,
                'R_squared': agg_r2,
                'Adjusted_R_squared': agg_adj_r2,
                'p_values': {
                    'Alpha': 1.0, 
                    'Beta': agg_p_beta
                }
            }

        except Exception as e:
            print(f"Regression Error: {e}")
            return None

    # =========================================================================
    # 加重平均スコアリングロジック (特性分解・棒グラフ用)
    # =========================================================================
    @staticmethod
    def calculate_weighted_factor_contributions(df_port_scored):
        """
        各銘柄の「回帰前（固有）ファクタースコア（Value Z, Quality Z等）」に対して、
        ポートフォリオ内のウェイトを掛け合わせた「加重平均寄与度」を算出する。
        """
        factors = ['Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z']
        df_out = df_port_scored.copy()
        
        if 'Weight' not in df_out.columns:
            df_out['Weight'] = 1.0 / len(df_out) if len(df_out) > 0 else 0
            
        for f in factors:
            if f in df_out.columns:
                contrib_col = f"{f}_Contrib"
                df_out[contrib_col] = df_out[f].fillna(0) * df_out['Weight']
            else:
                df_out[f"{f}_Contrib"] = 0.0
                
        portfolio_summary = {}
        for f in factors:
            contrib_col = f"{f}_Contrib"
            portfolio_summary[f] = df_out[contrib_col].sum()
            
        return df_out, portfolio_summary

    # =========================================================================
    # 既存のBeta計算ロジック（ユニバース比較等に使用）
    # =========================================================================
    @staticmethod
    def calculate_beta(df_fund, df_hist, df_market=None, benchmark_ticker="1321.T"):
        """
        時系列データからBetaを計算（Rm-Rfの市場プレミアム対応版）
        ※行列演算（Vectorized）によりforループのボトルネックを解消
        """
        if not isinstance(df_fund, pd.DataFrame):
            try:
                df = pd.DataFrame(df_fund)
                if 'Ticker' not in df.columns and 0 in df.columns:
                    df.rename(columns={0: 'Ticker'}, inplace=True)
            except:
                return pd.DataFrame()
        else:
            df = df_fund.copy()

        if not isinstance(df_hist, pd.DataFrame) or df_hist.empty:
            if 'Beta_Raw' not in df.columns: df['Beta_Raw'] = np.nan
            return df

        try:
            rets = df_hist.pct_change().dropna(how='all')
        except Exception:
            df['Beta_Raw'] = np.nan
            return df
        
        betas = {}

        if df_market is not None and not df_market.empty and 'Rm' in df_market.columns and 'Rf' in df_market.columns:
            try:
                if df_market['Rm'].max() > 10.0:
                    rm_ret = df_market['Rm'].pct_change().dropna()
                else:
                    rm_ret = df_market['Rm'].dropna()

                rf_daily = df_market['Rf'].reindex(rm_ret.index).ffill()
                market_premium = rm_ret - rf_daily
                
                aligned_data = pd.concat([rets, market_premium.rename('MarketPremium'), rf_daily.rename('Rf')], axis=1, join='inner').dropna(subset=['MarketPremium', 'Rf'])
                
                if not aligned_data.empty:
                    market_prem_aligned = aligned_data['MarketPremium']
                    bench_var = market_prem_aligned.var()

                    if bench_var > 1e-8:
                        rf_aligned = aligned_data['Rf']
                        asset_premiums = aligned_data[rets.columns].sub(rf_aligned, axis=0)
                        
                        covariances = asset_premiums.cov(market_prem_aligned)
                        betas = (covariances / bench_var).to_dict()
                        
                        for k in list(betas.keys()):
                            if abs(betas[k] - 1.0) < 1e-6 or abs(betas[k]) < 1e-6:
                                betas[k] = np.nan
            except Exception:
                df_market = None 

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
                covariances = rets.cov(bench_ret)
                betas = (covariances / bench_var).to_dict()
                
                for k in list(betas.keys()):
                    if k != benchmark_ticker:
                        if abs(betas[k] - 1.0) < 1e-6 or abs(betas[k]) < 1e-6:
                            betas[k] = np.nan
        
        df['Beta_Raw'] = df['Ticker'].map(betas).astype(float)
        return df

    # =========================================================================
    # 【強化】スマート・ウェイト・エンジン (時価総額加重の自動計算)
    # =========================================================================
    @staticmethod
    def calculate_portfolio_weights(df, user_weights_provided=False):
        """
        ウェイトが指定されていない場合、自動的に時価総額加重(Market Cap Weighting)を行う。
        死んだ銘柄(Beta算出不可等)はウェイトから除外し、残りの銘柄で100%になるよう再配分する。
        """
        df_out = df.copy()
        
        valid_mask = pd.Series(True, index=df_out.index)
        if 'Beta_Raw' in df_out.columns:
            valid_mask = df_out['Beta_Raw'].notna()
        
        if user_weights_provided and 'Weight' in df_out.columns:
            df_out['Weight'] = pd.to_numeric(df_out['Weight'], errors='coerce').fillna(0)
            df_out.loc[~valid_mask, 'Weight'] = 0.0
            
            valid_weights_count = (df_out['Weight'] > 0).sum()
            total_weight = df_out['Weight'].sum()
            
            if total_weight > 0 and valid_weights_count > 0:
                df_out['Weight'] = df_out['Weight'] / total_weight
                return df_out
        
        if 'MarketCap' in df_out.columns:
            valid_mc = pd.to_numeric(df_out['MarketCap'], errors='coerce').fillna(0)
            valid_mc.loc[~valid_mask] = 0.0 
            mc_sum = valid_mc.sum()
            
            if mc_sum > 0:
                df_out['Weight'] = valid_mc / mc_sum
            else:
                valid_count = valid_mask.sum()
                if valid_count > 0:
                    df_out['Weight'] = np.where(valid_mask, 1.0 / valid_count, 0.0)
                else:
                    df_out['Weight'] = 1.0 / len(df_out)
        else:
            valid_count = valid_mask.sum()
            if valid_count > 0:
                df_out['Weight'] = np.where(valid_mask, 1.0 / valid_count, 0.0)
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
        # 【修正 V19.2】SettingWithCopyWarning対策: .copy()でスライスのコピーを明示し、df.loc[]で代入
        df = df.copy()
        if 'PBR' in df.columns:
            clipped_pbr = pd.to_numeric(df['PBR'], errors='coerce').clip(lower=0.1, upper=100.0)
            df.loc[:, 'Value_Raw'] = clipped_pbr.apply(
                lambda x: np.log(1/x) if (pd.notnull(x) and x > 0) else np.nan
            )
        
        if 'Size_Raw' in df.columns:
            raw_size = pd.to_numeric(df['Size_Raw'], errors='coerce')
            clipped_size = raw_size.clip(lower=1e8)
            df.loc[:, 'Size_Log'] = clipped_size.apply(
                lambda x: np.log(x) if (pd.notnull(x) and x > 0) else np.nan
            )
            df.loc[:, 'MarketCap'] = raw_size
        
        if 'ROE' in df.columns:
            df.loc[:, 'Quality_Raw'] = pd.to_numeric(df['ROE'], errors='coerce').clip(lower=-2.0, upper=2.0)
        
        try:
            if 'Growth' in df.columns:
                df.loc[:, 'Investment_Raw'] = pd.to_numeric(df['Growth'], errors='coerce').clip(lower=-1.0, upper=3.0)
            else:
                df.loc[:, 'Investment_Raw'] = np.nan
        except Exception:
            df.loc[:, 'Investment_Raw'] = np.nan
            
        return df

    @staticmethod
    def calculate_orthogonalization(df, x_col, y_col):
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
        df = df_target.copy()
        r_squared_map = {} 
        
        if ortho_pairs is None:
            ortho_pairs = [('Quality', 'Investment')] if isinstance(stats, dict) and 'ortho_slope' in stats else []

        for target_factor, predictor_factor in ortho_pairs:
            target_col = f"{target_factor}_Raw"
            predictor_col = f"{predictor_factor}_Raw"
            pair_key = f"ortho_{target_factor}_{predictor_factor}"
            
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
        df_out = df.copy()

        if market_sensitivities == 'dynamic':
            market_sensitivities = {}
            factors_to_check = ['Size_Z', 'Value_Z', 'Quality_Z', 'Investment_Z']
            
            if 'Beta_Z' in df_out.columns and df_out['Beta_Z'].notna().sum() >= 5:
                for f in factors_to_check:
                    if f in df_out.columns:
                        corr = df_out['Beta_Z'].corr(df_out[f])
                        market_sensitivities[f] = corr if pd.notna(corr) else 0.0
            else:
                market_sensitivities = {
                    'Size_Z': 0.25, 'Value_Z': -0.15, 'Quality_Z': -0.20, 'Investment_Z': -0.10 
                }
        elif market_sensitivities is None:
            market_sensitivities = {
                'Size_Z': 0.25, 'Value_Z': -0.15, 'Quality_Z': -0.20, 'Investment_Z': -0.10
            }

        sensitivity_sum = pd.Series(0.0, index=df_out.index)
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
        insights = []
        
        z_size = z_scores.get('Size', 0)
        z_val  = z_scores.get('Value', 0)
        z_qual = z_scores.get('Quality', 0)
        z_inv  = z_scores.get('Investment', 0)

        if z_size < -0.7:
            insights.append("Large Cap Focus: High allocation to large-cap stocks with stable financial foundations, providing resilience against market volatility.")
        elif z_size > 0.7:
            insights.append("Small Cap Effect: Weighted towards smaller market capitalization stocks, offering potential to outperform the market average.")
        
        if z_val > 0.7:
            insights.append("Value Investing: Consists of stocks trading at a discount to their book value, potentially limiting downside risk.")
        elif z_val < -0.7:
            insights.append("Growth Tilt: Includes stocks with high future growth expectations, typically trading at premium valuations.")

        if z_qual > 0.7:
            insights.append("High Quality: Dominated by high-quality companies characterized by strong profitability (ROE) and operational efficiency.")

        if z_inv > 0.7:
            insights.append("Conservative Management: Companies maintaining disciplined asset growth and lean operations (CMA effect).")
        elif z_inv < -0.7:
            insights.append("Aggressive Investment: Includes companies aggressively expanding capital expenditures and assets (monitor for over-investment risks).")

        if z_qual > 0.5 and z_val > 0.5:
            insights.append("Quality Value: An ideal mix of high-quality companies that are currently undervalued by the market.")

        if not insights:
            insights.append("Market Neutral (Balanced): Minimal tilt towards specific factors, representing a stable composition closely mirroring the market average.")
            
        return insights

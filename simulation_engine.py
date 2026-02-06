import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.decomposition import PCA
import pandas_datareader.data as web
from datetime import datetime

# =========================================================
# 🛠️ Class Definitions (Structure Fixed for Import Safety)
# =========================================================

class MarketDataEngine:
    """Manages market data, factors, and benchmarks."""
    def __init__(self):
        self.start_date = "2000-01-01"
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.usdjpy_cache = None

    def validate_tickers(self, input_dict):
        """Check if tickers exist."""
        valid_data = {}
        invalid_tickers = []
        status_text = st.empty()
        
        for ticker, weight in input_dict.items():
            try:
                # Check via yfinance
                tick = yf.Ticker(ticker)
                hist = tick.history(period="5d")
                if not hist.empty:
                    valid_data[ticker] = {'name': ticker, 'weight': weight}
                    status_text.text(f"✅ OK: {ticker}")
                else:
                    invalid_tickers.append(ticker)
            except:
                invalid_tickers.append(ticker)
        
        status_text.empty()
        return valid_data, invalid_tickers

    def _get_usdjpy(self):
        """Fetch JPY rate with cache."""
        if self.usdjpy_cache is not None:
            return self.usdjpy_cache
        try:
            raw = yf.download("JPY=X", start=self.start_date, end=self.end_date, interval="1mo", auto_adjust=True, progress=False)
            
            if isinstance(raw, pd.DataFrame):
                if 'Close' in raw.columns:
                    usdjpy = raw['Close']
                else:
                    usdjpy = raw.iloc[:, 0]
            else:
                usdjpy = raw

            if isinstance(usdjpy, pd.DataFrame):
                usdjpy = usdjpy.iloc[:, 0]

            # 修正: 'M' -> 'ME' (FutureWarning対策)
            usdjpy = usdjpy.resample('ME').last().ffill()
            if usdjpy.index.tz is not None: 
                usdjpy.index = usdjpy.index.tz_localize(None)
            
            self.usdjpy_cache = usdjpy
            return usdjpy
        except Exception:
            return pd.Series(dtype=float)

    @st.cache_data(ttl=3600*24*7)
    def fetch_french_factors(_self, region='US'):
        """Fetch Fama-French Factors (Robust Fallback)."""
        try:
            name = 'F-F_Research_Data_Factors'
            if region == 'Japan': 
                name = 'Japan_3_Factors'
            elif region == 'Global': 
                name = 'Global_3_Factors'

            # Attempt to fetch data
            ff_data = web.DataReader(name, 'famafrench', start=_self.start_date, end=_self.end_date)[0]
            
            # Process data if successful
            ff_data = ff_data / 100.0
            # 修正: 'M' -> 'ME'
            ff_data.index = ff_data.index.to_timestamp(freq='ME')
            
            if ff_data.index.tz is not None: 
                ff_data.index = ff_data.index.tz_localize(None)
            
            return ff_data
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600*24)
    def fetch_historical_prices(_self, tickers):
        """Fetch stock prices."""
        try:
            raw_data = yf.download(tickers, start=_self.start_date, end=_self.end_date, interval="1mo", auto_adjust=True, progress=False)
            data = pd.DataFrame()

            if len(tickers) == 1:
                ticker = tickers[0]
                if isinstance(raw_data, pd.Series):
                    data[ticker] = raw_data
                elif isinstance(raw_data, pd.DataFrame):
                    if 'Close' in raw_data.columns:
                        data[ticker] = raw_data['Close']
                    else:
                        data[ticker] = raw_data.iloc[:, 0]
            else:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    try:
                        data = raw_data.xs('Close', axis=1, level=0, drop_level=True)
                    except KeyError:
                        try:
                            data = raw_data.xs('Adj Close', axis=1, level=0, drop_level=True)
                        except:
                            data = raw_data.iloc[:, :len(tickers)]
                            data.columns = tickers
                else:
                    data = raw_data

            # 修正: 'M' -> 'ME'
            data = data.resample('ME').last().ffill()
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            usdjpy = _self._get_usdjpy()
            if not usdjpy.empty:
                usdjpy = usdjpy.reindex(data.index, method='ffill')
                data_jpy = data.copy()
                for col in data.columns:
                    # Do not convert Japanese assets or indices
                    is_japan = str(col).endswith(".T") or str(col) in ["^N225", "^TPX", "1306.T"]
                    if not is_japan:
                        data_jpy[col] = data[col] * usdjpy
            else:
                data_jpy = data

            returns = data_jpy.pct_change().dropna(how='all').dropna()
            
            valid_cols = [c for c in returns.columns if c in tickers]
            if valid_cols:
                returns = returns[valid_cols]
            
            return returns
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600*24)
    def fetch_benchmark_data(_self, ticker, is_jpy_asset=False):
        """Fetch benchmark."""
        try:
            raw_data = yf.download(ticker, start=_self.start_date, end=_self.end_date, interval="1mo", auto_adjust=True, progress=False)
            data = pd.Series(dtype=float)
            if isinstance(raw_data, pd.DataFrame):
                if 'Close' in raw_data.columns:
                    data = raw_data['Close']
                elif isinstance(raw_data.columns, pd.MultiIndex):
                     try: data = raw_data.xs('Close', axis=1, level=0, drop_level=True)
                     except: data = raw_data.iloc[:, 0]
                else:
                    data = raw_data.iloc[:, 0]
            else:
                data = raw_data

            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]

            # 修正: 'M' -> 'ME'
            data = data.resample('ME').last().ffill()
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            if not is_jpy_asset:
                usdjpy = _self._get_usdjpy()
                if not usdjpy.empty:
                    usdjpy = usdjpy.reindex(data.index, method='ffill')
                    data = data * usdjpy
            
            return data.pct_change().dropna()
        except:
            return pd.Series(dtype=float)

class PortfolioAnalyzer:
    
    @staticmethod
    def create_synthetic_history(returns_df, weights_dict):
        valid_tickers = [t for t in weights_dict.keys() if t in returns_df.columns]
        if not valid_tickers:
            return pd.Series(dtype=float), {}

        filtered_weights = {k: weights_dict[k] for k in valid_tickers}
        total_weight = sum(filtered_weights.values())
        norm_weights = {k: v/total_weight for k, v in filtered_weights.items()}
        
        weighted_returns = pd.DataFrame()
        for ticker, w in norm_weights.items():
            weighted_returns[ticker] = returns_df[ticker] * w
            
        port_ret = weighted_returns.sum(axis=1)
        return port_ret, norm_weights

    @staticmethod
    def calculate_correlation_matrix(returns_df):
        if returns_df.empty:
            return pd.DataFrame()
        return returns_df.corr()

    @staticmethod
    def perform_factor_regression(port_ret, factor_df):
        if port_ret.empty or factor_df is None or factor_df.empty:
            return None, None

        df_y = port_ret.to_frame(name='y')
        df_y['period'] = df_y.index.to_period('M') 
        df_x = factor_df.copy()
        df_x['period'] = df_x.index.to_period('M') 
        
        merged = pd.merge(df_y, df_x, on='period', how='inner').dropna()
        if merged.empty: return None, None
        
        y = merged['y']
        X_cols = [c for c in merged.columns if c in ['Mkt-RF', 'SMB', 'HML']]
        if not X_cols: return None, None
        
        X = merged[X_cols]
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X)
            results = model.fit()
            return results.params, results.rsquared
        except:
            return None, None

    @staticmethod
    def run_monte_carlo_simulation(port_ret, n_years=20, n_simulations=5000, initial_investment=1000000):
        """
        修正: デフォルト回数を7500 -> 5000に変更
        """
        if port_ret.empty:
            return None, None

        mu_monthly = port_ret.mean()
        sigma_monthly = port_ret.std()
        
        n_months = n_years * 12
        drift = (mu_monthly - 0.5 * sigma_monthly**2)
        
        df_t = 6
        Z = np.random.standard_t(df_t, (n_months, n_simulations))
        
        daily_returns = np.exp(drift + sigma_monthly * Z)
        
        price_paths = np.zeros((n_months + 1, n_simulations))
        price_paths[0] = initial_investment
        price_paths[1:] = initial_investment * np.cumprod(daily_returns, axis=0)
        
        last_date = port_ret.index[-1]
        # 修正: 'M' -> 'ME'
        future_dates = pd.date_range(start=last_date, periods=n_months + 1, freq='ME')
        
        percentiles = [10, 50, 90]
        stats_data = np.percentile(price_paths, percentiles, axis=1)
        df_stats = pd.DataFrame(stats_data.T, index=future_dates, columns=['p10', 'p50', 'p90'])
        
        final_values = price_paths[-1, :]
        
        return df_stats, final_values

    @staticmethod
    def calculate_calmar_ratio(port_ret):
        if port_ret.empty: return np.nan
        cum_ret = (1 + port_ret).cumprod()
        if len(port_ret) < 12: return np.nan
        cagr = (cum_ret.iloc[-1])**(12/len(port_ret)) - 1
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()
        if max_dd == 0: return np.nan
        return cagr / abs(max_dd)

    @staticmethod
    def calculate_omega_ratio(port_ret, threshold=0.0):
        if port_ret.empty: return np.nan
        gains = port_ret[port_ret > threshold] - threshold
        losses = threshold - port_ret[port_ret < threshold]
        sum_gains = gains.sum()
        sum_losses = losses.sum()
        if sum_losses == 0: return np.inf
        return sum_gains / sum_losses

    @staticmethod
    def calculate_information_ratio(port_ret, bench_ret):
        if port_ret.empty or bench_ret.empty: return np.nan, np.nan
        
        p_df = port_ret.to_frame(name='p')
        b_df = bench_ret.to_frame(name='b')
        p_df['period'] = p_df.index.to_period('M')
        b_df['period'] = b_df.index.to_period('M')
        
        merged = pd.merge(p_df, b_df, on='period', how='inner').dropna()
        
        if len(merged) < 12: return np.nan, np.nan
        
        active_ret = merged['p'] - merged['b']
        mean_active = active_ret.mean() * 12
        tracking_error = active_ret.std() * np.sqrt(12)
        if tracking_error == 0: return np.nan, 0.0
        return mean_active / tracking_error, tracking_error

    @staticmethod
    def perform_pca(returns_df):
        if returns_df.shape[1] < 2: 
            return 1.0, None
        
        pca = PCA(n_components=2)
        pca.fit(returns_df)
        
        loadings = pd.DataFrame(
            pca.components_.T, 
            index=returns_df.columns, 
            columns=['PC1', 'PC2']
        )
        
        return pca.explained_variance_ratio_[0], loadings

    @staticmethod
    def rolling_beta_analysis(port_ret, factor_df, window=24):
        if factor_df is None or factor_df.empty or port_ret.empty:
            return pd.DataFrame()

        df_y = port_ret.to_frame(name='y')
        df_y['period'] = df_y.index.to_period('M') 
        df_x = factor_df.copy()
        df_x['period'] = df_x.index.to_period('M') 
        
        merged = pd.merge(df_y, df_x, on='period', how='inner').dropna()
        if merged.empty: return pd.DataFrame()
        
        y = merged['y']
        X_cols = [c for c in merged.columns if c not in ['y', 'period']]
        X = merged[X_cols]
        
        data_len = len(y)
        if data_len < window:
            window = max(6, int(data_len / 2))
        if data_len < window:
            return pd.DataFrame()

        try:
            X_const = sm.add_constant(X)
            model = RollingOLS(y, X_const, window=window)
            rres = model.fit()
            params = rres.params.copy()
            if 'const' in params.columns:
                params = params.drop(columns=['const'])
            return params.dropna()
        except:
            return pd.DataFrame()

    @staticmethod
    def cost_drag_simulation(port_ret, cost_tier):
        """
        Calculates impact of costs.
        FIXED: Returns exactly 4 values to match app.py expectation.
        Returns: Gross Curve, Net Curve, Loss Amount, Loss Percentage
        """
        if port_ret.empty: return pd.Series(), pd.Series(), 0, 0
        
        cost_map = {'Low': 0.001, 'Medium': 0.006, 'High': 0.020}
        annual_cost = cost_map.get(cost_tier, 0.006)
        monthly_cost = (1 + annual_cost)**(1/12) - 1
        
        net_ret = port_ret - monthly_cost
        gross_cum = (1 + port_ret).cumprod()
        net_cum = (1 + net_ret).cumprod()
        
        final_gross = gross_cum.iloc[-1]
        final_net = net_cum.iloc[-1]
        
        diff_val = final_gross - final_net
        lost_pct = 1 - (final_net / final_gross) 
        
        # 修正: 戻り値を4つに統一 (unpacking error回避)
        return gross_cum, net_cum, diff_val, lost_pct

    @staticmethod
    def calculate_strict_attribution(returns_df, weights_dict):
        assets = list(weights_dict.keys())
        available_assets = [a for a in assets if a in returns_df.columns]
        if not available_assets: return pd.Series(dtype=float)
            
        w_series = pd.Series(weights_dict)
        total_w = w_series[available_assets].sum()
        initial_w = w_series[available_assets] / total_w
        
        r_df = returns_df[available_assets].copy()
        
        cum_r_index = (1 + r_df).cumprod()
        asset_values = cum_r_index.multiply(initial_w, axis=1)
        port_values = asset_values.sum(axis=1)
        
        weights_df = asset_values.div(port_values, axis=0).shift(1)
        weights_df.iloc[0] = initial_w
        
        port_ret = (weights_df * r_df).sum(axis=1)
        total_cum_ret = (1 + port_ret).prod() - 1
        
        log_return = np.log(1 + total_cum_ret)
        k = log_return / total_cum_ret if total_cum_ret != 0 else 1.0
            
        kt = np.log(1 + port_ret) / port_ret
        kt = kt.fillna(1.0)
        
        term = weights_df * r_df
        smoothed_term = term.multiply(kt, axis=0)
        
        final_attribution = smoothed_term.sum() / k
        
        return final_attribution.sort_values(ascending=True)

    @staticmethod
    def calculate_risk_contribution(returns_df, weights_dict):
        """
        Calculates Marginal Risk Contribution (MRC).
        """
        assets = list(weights_dict.keys())
        valid_assets = [a for a in assets if a in returns_df.columns]
        if not valid_assets:
            return pd.Series(dtype=float)

        # Filter and Normalize weights
        w_series = pd.Series({k: weights_dict[k] for k in valid_assets})
        w_series = w_series / w_series.sum() 
        
        # Covariance Matrix (Annualized)
        cov_matrix = returns_df[valid_assets].cov() * 12 
        
        # Portfolio Volatility
        port_vol = np.sqrt(w_series.T @ cov_matrix @ w_series)
        
        # Marginal Risk Contribution: (Cov * w) / PortVol
        mrc = cov_matrix @ w_series / port_vol
        
        # Risk Contribution: w * MRC
        rc = w_series * mrc
        
        # Percent Contribution: RC / PortVol
        rc_pct = rc / port_vol
        
        # 修正: sort_valuesを削除し、投資比率と同じ並び順を維持
        return rc_pct

    @staticmethod
    def calculate_label_offsets(values, min_dist=0.08, base_y=1.05):
        """
        Calculates Y-axis offsets for histogram labels to prevent overlap.
        """
        if not values: return []
        
        # Create a list of (index, value)
        indexed_values = sorted(enumerate(values), key=lambda x: x[1])
        
        y_offsets = [base_y] * len(values)
        
        # Determine value range to normalize distance check
        val_range = max(values) - min(values)
        if val_range == 0: val_range = 1.0
        
        # Iterate through sorted values and stack levels if too close
        levels = [base_y] * len(values) # temporary storage for sorted
        current_level = base_y
        
        for i in range(1, len(indexed_values)):
            curr_val = indexed_values[i][1]
            prev_val = indexed_values[i-1][1]
            
            # Check normalized distance
            dist = (curr_val - prev_val) / val_range
            
            if dist < min_dist:
                # If close to previous, bump up level
                prev_level = levels[i-1]
                if prev_level == base_y:
                    current_level = base_y + 0.15
                elif prev_level == base_y + 0.15:
                    current_level = base_y + 0.3
                else:
                    current_level = base_y # Reset if stack gets too high
            else:
                current_level = base_y
            
            levels[i] = current_level
            
        # Map back to original indices
        final_offsets = [0.0] * len(values)
        for i, (orig_idx, _) in enumerate(indexed_values):
            final_offsets[orig_idx] = levels[i]
            
        return final_offsets

class PortfolioDiagnosticEngine:
    @staticmethod
    def generate_report(weights_dict, pca_ratio, port_ret, benchmark_ret=None):
        report = {
            "type": "",
            "risk_comment": "",
            "diversification_comment": "",
            "action_plan": ""
        }
        
        num_assets = len(weights_dict)
        
        if num_assets == 1:
            report["type"] = "🏹 集中投資 (スナイパー型)"
            report["diversification_comment"] = "分散効果はゼロです。すべての卵を一つのカゴに入れています。"
            report["risk_comment"] = "⚠️ 個別銘柄リスクを最大限に負っています。"
            report["action_plan"] = "少なくとも3〜5つの相関の低い資産に分散することを推奨します。"
        else:
            if pca_ratio >= 0.85:
                report["type"] = "⚠️ 見せかけの分散 (フェイク・ダイバーシフィケーション)"
                report["diversification_comment"] = f"変動の{pca_ratio*100:.1f}%が単一の要因（市場全体など）で説明されてしまいます。"
                report["risk_comment"] = "市場暴落時に、保有資産すべてが同時に下落するリスクが高い状態です。"
                report["action_plan"] = "株式以外の資産（債券、ゴールドなど）を追加し、リスク要因を分散してください。"
            elif pca_ratio <= 0.60:
                report["type"] = "🏰 要塞型 (フォートレス)"
                report["diversification_comment"] = f"メイン要因による説明率は{pca_ratio*100:.1f}%に留まり、独自の動きをする資産が組み込まれています。"
                report["risk_comment"] = "無駄なリスクが効果的に分散され、防御力が高いポートフォリオです。"
                report["action_plan"] = "現在のバランスは非常に良好です。リバランスを行い維持しましょう。"
            else:
                report["type"] = "⚖️ バランス型"
                report["diversification_comment"] = f"市場連動性は{pca_ratio*100:.1f}%で、標準的な分散レベルです。"
                report["risk_comment"] = "市場平均と同程度のリスク・リターン特性を持つ可能性が高いです。"
                report["action_plan"] = "より防御力を高めるなら、債券比率の調整やオルタナティブ資産の検討が有効です。"

        return report

    @staticmethod
    def get_skew_kurt_desc(port_ret):
        if port_ret.empty: return "データ不足です。"
        skew = port_ret.skew()
        kurt = port_ret.kurt()
        desc = []
        if skew < -0.5: desc.append("⚠️ 負の歪度: 通常時は安定していますが、稀に大きな急落が起きるリスクがあります（コツコツドカン型）。")
        elif skew > 0.5: desc.append("✅ 正の歪度: 損失は限定的ですが、稀に大きな利益が出る可能性があります。")
        
        if kurt > 2.0: desc.append("⚠️ ファットテール: 正規分布に比べて「極端な事象（暴騰・暴落）」が発生する確率が高い状態です。")
        
        return " ".join(desc) if desc else "統計的に標準的な分布（正規分布に近い）です。"

    @staticmethod
    def generate_factor_report(params):
        """Translate Factor Analysis."""
        if params is None: return "データなし"
        
        comments = []
        
        # 1. HML
        hml = params.get('HML', 0)
        if hml > 0.15:
            comments.append("✅ **バリュー株選好:** 割安株や高配当株との連動性が高いです。")
        elif hml < -0.15:
            comments.append("🚀 **グロース株選好:** 成長株やハイテク株との連動性が高いです。")
        else:
            comments.append("⚖️ **スタイル中立:** バリューとグロースのバランスが取れています。")

        # 2. SMB
        smb = params.get('SMB', 0)
        if smb > 0.15:
            comments.append("🐣 **小型株バイアス:** 変動は大きいですが、将来の成長余地を取りに行っています。")
        elif smb < -0.15:
            comments.append("🐘 **大型株バイアス:** 安定した大企業中心の構成です。")
        
        # 3. Mkt-RF
        mkt = params.get('Mkt-RF', 1.0)
        if mkt > 1.1:
            comments.append("🎢 **ハイベータ（積極運用）:** 市場平均よりも大きく動く、攻撃的な構成です。")
        elif mkt < 0.9:
            comments.append("🛡️ **ローベータ（守りの運用）:** 市場下落時にも比較的ダメージを受けにくい構成です。")

        return "\n".join(comments)

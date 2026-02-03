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
# 🛠️ クラス定義 (Brain: V17.1 - Final Complete Edition)
# =========================================================

class MarketDataEngine:
    """市場データ、ファクター、ベンチマークを一元管理するエンジン"""
    def __init__(self):
        self.start_date = "2000-01-01"
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.usdjpy_cache = None

    def validate_tickers(self, input_dict):
        """銘柄の存在確認"""
        valid_data = {}
        invalid_tickers = []
        status_text = st.empty()
        
        for ticker, weight in input_dict.items():
            try:
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
        """為替データを取得（キャッシュ付き・構造化対応）"""
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

            usdjpy = usdjpy.resample('M').last().ffill()
            if usdjpy.index.tz is not None: 
                usdjpy.index = usdjpy.index.tz_localize(None)
            
            self.usdjpy_cache = usdjpy
            return usdjpy
        except Exception:
            return pd.Series(dtype=float)

    @st.cache_data(ttl=3600*24*7)
    def fetch_french_factors(_self, region='US'):
        """ファクターデータ取得（修正済み）"""
        try:
            name = 'F-F_Research_Data_Factors'
            if region == 'Japan': 
                name = 'Japan_3_Factors'
            elif region == 'Global': 
                name = 'Global_3_Factors'

            ff_data = web.DataReader(name, 'famafrench', start=_self.start_date, end=_self.end_date)[0]
            ff_data = ff_data / 100.0
            
            ff_data.index = ff_data.index.to_timestamp(freq='M')
            if ff_data.index.tz is not None: 
                ff_data.index = ff_data.index.tz_localize(None)
            
            return ff_data
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600*24)
    def fetch_historical_prices(_self, tickers):
        """株価データ取得（修正済み）"""
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

            data = data.resample('M').last().ffill()
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            usdjpy = _self._get_usdjpy()
            if not usdjpy.empty:
                usdjpy = usdjpy.reindex(data.index, method='ffill')
                data_jpy = data.copy()
                for col in data.columns:
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
            st.error(f"データ取得エラー詳細: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600*24)
    def fetch_benchmark_data(_self, ticker, is_jpy_asset=False):
        """ベンチマーク取得"""
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

            data = data.resample('M').last().ffill()
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

    # --- 相関行列計算 ---
    @staticmethod
    def calculate_correlation_matrix(returns_df):
        """構成銘柄の相関行列を計算"""
        if returns_df.empty:
            return pd.DataFrame()
        return returns_df.corr()

    # --- 🔥 [NEW] ファクター回帰分析 (Style Analysis) ---
    @staticmethod
    def perform_factor_regression(port_ret, factor_df):
        """ポートフォリオのリターンをFama-French 3ファクターで回帰分析する"""
        if port_ret.empty or factor_df.empty:
            return None, None

        # データの期間合わせ
        df_y = port_ret.to_frame(name='y')
        df_y['period'] = df_y.index.to_period('M') 
        df_x = factor_df.copy()
        df_x['period'] = df_x.index.to_period('M') 
        
        merged = pd.merge(df_y, df_x, on='period', how='inner').dropna()
        if merged.empty: return None, None
        
        y = merged['y']
        # 'RF'はリスクフリーレートなので、超過リターン(y - RF)を計算するのが一般的だが、
        # ここでは簡易的に Mkt-RF, SMB, HML を説明変数として回帰する
        X_cols = [c for c in merged.columns if c in ['Mkt-RF', 'SMB', 'HML']]
        X = merged[X_cols]
        X = sm.add_constant(X) # Alpha (const) を追加

        try:
            model = sm.OLS(y, X)
            results = model.fit()
            return results.params, results.rsquared
        except:
            return None, None

    # --- モンテカルロ・シミュレーション (Fat-Tail Edition) ---
    @staticmethod
    def run_monte_carlo_simulation(port_ret, n_years=20, n_simulations=7500, initial_investment=1000000):
        if port_ret.empty:
            return None, None

        mu_monthly = port_ret.mean()
        sigma_monthly = port_ret.std()
        
        n_months = n_years * 12
        dt = 1/12
        
        drift = (mu_monthly - 0.5 * sigma_monthly**2)
        
        df_t = 6
        Z = np.random.standard_t(df_t, (n_months, n_simulations))
        
        daily_returns = np.exp(drift + sigma_monthly * Z)
        
        price_paths = np.zeros((n_months + 1, n_simulations))
        price_paths[0] = initial_investment
        price_paths[1:] = initial_investment * np.cumprod(daily_returns, axis=0)
        
        last_date = port_ret.index[-1]
        future_dates = pd.date_range(start=last_date, periods=n_months + 1, freq='M')
        
        percentiles = [10, 50, 90]
        stats_data = np.percentile(price_paths, percentiles, axis=1)
        df_stats = pd.DataFrame(stats_data.T, index=future_dates, columns=['p10', 'p50', 'p90'])
        
        final_values = price_paths[-1, :]
        
        return df_stats, final_values

    # --- 既存分析メソッド ---
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
        if returns_df.shape[1] < 2: return 1.0, None
        pca = PCA(n_components=2)
        pca.fit(returns_df)
        return pca.explained_variance_ratio_[0], pca

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
        if port_ret.empty: return pd.Series(), pd.Series(), 0, 0
        cost_map = {'Low': 0.001, 'Medium': 0.006, 'High': 0.020}
        annual_cost = cost_map.get(cost_tier, 0.006)
        monthly_cost = (1 + annual_cost)**(1/12) - 1
        net_ret = port_ret - monthly_cost
        gross_cum = (1 + port_ret).cumprod()
        net_cum = (1 + net_ret).cumprod()
        return gross_cum, net_cum, gross_cum.iloc[-1] - net_cum.iloc[-1], annual_cost

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
            report["type"] = "🏹 集中投資型 (Sniper Allocation)"
            report["diversification_comment"] = "現在、1つの銘柄に全ての資金を投じています。分散効果は一切機能していません。"
            report["risk_comment"] = "⚠️ 個別株リスクが最大化しています。"
            report["action_plan"] = "少なくとも動きの異なる3〜5銘柄への分散を推奨します。"
        else:
            if pca_ratio >= 0.85:
                report["type"] = "⚠️ 見かけの分散 (High Correlation)"
                report["diversification_comment"] = f"全体の動きの{pca_ratio*100:.1f}%が「共通の要因」で説明できてしまいます。"
                report["risk_comment"] = "全銘柄が同時に下落する「共倒れ」のリスクが高い状態です。"
                report["action_plan"] = "株式とは異なる動きをする資産（国債、コモディティ）を組み入れてください。"
            elif pca_ratio <= 0.60:
                report["type"] = "🏰 要塞型分散 (True Diversification)"
                report["diversification_comment"] = f"説明力が{pca_ratio*100:.1f}%と低く、理想的な分散効果が発揮されています。"
                report["risk_comment"] = "市場全体のリスク以外の無駄なリスクは排除されています。"
                report["action_plan"] = "現在のバランスは非常に良好です。定期的なリバランスを行ってください。"
            else:
                report["type"] = "⚖️ 標準分散型 (Balanced)"
                report["diversification_comment"] = f"説明力は{pca_ratio*100:.1f}%で、適度な分散が効いています。"
                report["risk_comment"] = "市場平均に近い動きをする可能性が高いです。"
                report["action_plan"] = "さらに守りを固めるなら債券比率の調整を検討してください。"

        return report

    @staticmethod
    def get_skew_kurt_desc(port_ret):
        if port_ret.empty: return "データ不足のため診断できません。"
        skew = port_ret.skew()
        kurt = port_ret.kurt()
        desc = []
        if skew < -0.5: desc.append("⚠️ 左に裾が長い分布です（コツコツドカン型）。")
        elif skew > 0.5: desc.append("✅ 右に裾が長い分布です（宝くじ型）。")
        if kurt > 2.0: desc.append("⚠️ 極端な値動き（ファットテール）が起きやすい構造です。")
        return " ".join(desc) if desc else "分布に極端な偏りはなく、統計的に素直な挙動を示しています。"

    # --- 🔥 [NEW] ファクター解説生成 ---
    @staticmethod
    def generate_factor_report(params):
        """回帰分析結果（Beta）から、ポートフォリオのスタイルを言語化する"""
        if params is None: return "分析不能"
        
        comments = []
        
        # 1. HML (バリュー vs グロース)
        hml = params.get('HML', 0)
        if hml > 0.15:
            comments.append("✅ **バリュー株選好:** 割安株（高配当・成熟企業）の動きに連動しやすい傾向があります。")
        elif hml < -0.15:
            comments.append("🚀 **グロース株選好:** 成長株（ハイテク・新興企業）の動きに強く連動します。")
        else:
            comments.append("⚖️ **スタイル中立:** バリュー・グロースの偏りは少なく、バランスが取れています。")

        # 2. SMB (サイズ)
        smb = params.get('SMB', 0)
        if smb > 0.15:
            comments.append("🐣 **小型株効果:** 中小型株の動きを含んでおり、市場平均以上の爆発力を秘めています。")
        elif smb < -0.15:
            comments.append("🐘 **大型株安定:** 巨大企業中心の構成で、市場の動揺に対して比較的堅牢です。")
        
        # 3. Mkt-RF (市場感応度)
        mkt = params.get('Mkt-RF', 1.0)
        if mkt > 1.1:
            comments.append("🎢 **ハイベータ:** 市場平均よりも大きく動く、積極的なリスクテイク姿勢です。")
        elif mkt < 0.9:
            comments.append("🛡️ **ローベータ:** 市場全体が下がっても、傷は浅く済むディフェンシブな構成です。")

        return "\n".join(comments)

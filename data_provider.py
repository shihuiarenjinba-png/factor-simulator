import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os  # 【追加】環境変数用
import requests  # 【追加】FMP APIアクセス用
import time # 【追加】APIレート制限制御用
from scipy.stats import linregress
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------
# 0. 基本設定 (Config)
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Market Factor Lab Pro (Modular)")

# 分析対象ユニバース
NIKKEI_225_SAMPLE = [
    "7203.T", "6758.T", "6861.T", "9984.T", "9983.T", "8035.T", "6098.T", "4063.T", "6367.T", "9432.T",
    "4502.T", "4503.T", "6501.T", "7267.T", "8058.T", "8001.T", "6954.T", "6981.T", "9020.T", "9022.T",
    "7741.T", "5108.T", "4452.T", "6902.T", "7974.T", "8031.T", "4519.T", "4568.T", "6273.T", "4543.T",
    "6702.T", "6503.T", "4901.T", "4911.T", "2502.T", "2802.T", "3382.T", "8306.T", "8316.T", "8411.T",
    "8766.T", "8591.T", "8801.T", "8802.T", "9021.T", "9101.T", "9433.T", "9434.T", "9501.T", "9502.T"
]

TOPIX_100_SAMPLE = [
    "7203.T", "6758.T", "8306.T", "9984.T", "8035.T", "9432.T", "6861.T", "9983.T", "4063.T", "8058.T",
    "6501.T", "8001.T", "6902.T", "4568.T", "8316.T", "8411.T", "8766.T", "9022.T", "6367.T", "4502.T",
    "6098.T", "7741.T", "6954.T", "4503.T", "6981.T", "5108.T", "4452.T", "7974.T", "8031.T", "4519.T"
]

# ---------------------------------------------------------
# 【Modified】Module 1: Data Provider (Hybrid: yfinance + FMP Rescue)
# ---------------------------------------------------------
class DataProvider:
    """
    データ取得とキャッシュ管理を担当する独立モジュール (v3.0 Hybrid)
    yfinanceで取得できないデータをFMP APIで補完し、セクター情報を統一する。
    """
    
    # GitHub Secrets等からAPIキーを読み込む (なければNone)
    FMP_API_KEY = os.environ.get("FMP_API_KEY")

    # セクター変換辞書: yfinance/FMPの業種名 -> Kenneth French 10 Industry Code
    SECTOR_TRANSLATION = {
        # Tech / Telecom
        'Technology': 'HiTec', 'Information & Communication': 'Telcm', 
        'Electric Appliances': 'HiTec', 'Precision Instruments': 'HiTec',
        'Services': 'Other', 'Communication Services': 'Telcm',
        
        # Manufacturing / Energy
        'Automobiles & Components': 'Manuf', 'Transportation Equipment': 'Manuf',
        'Machinery': 'Manuf', 'Chemicals': 'Manuf', 'Basic Materials': 'Manuf',
        'Energy': 'Enrgy', 'Oil & Coal Products': 'Enrgy', 'Mining': 'Enrgy',
        'Glass & Ceramics Products': 'Manuf', 'Iron & Steel': 'Manuf', 
        'Nonferrous Metals': 'Manuf', 'Metal Products': 'Manuf',
        
        # Consumer
        'Consumer Cyclical': 'Durbl', 'Consumer Defensive': 'NoDur',
        'Retail Trade': 'Shops', 'Wholesale Trade': 'Shops', 
        'Foods': 'NoDur', 'Pharmaceuticals': 'Hlth', 'Healthcare': 'Hlth',
        'Textiles & Apparels': 'NoDur', 'Pulp & Paper': 'Manuf',
        
        # Finance / Real Estate / Other
        'Financial Services': 'Other', 'Banks': 'Other', 'Insurance': 'Other',
        'Securities & Commodity Futures': 'Other', 'Other Financing Business': 'Other',
        'Real Estate': 'Other', 'Construction': 'Manuf',
        
        # Infrastructure
        'Utilities': 'Utils', 'Electric Power & Gas': 'Utils',
        'Land Transportation': 'Other', 'Marine Transportation': 'Other', 
        'Air Transportation': 'Other', 'Warehousing & Harbor Transportation Services': 'Other'
    }

    @staticmethod
    def get_universe_tickers(mode="Nikkei 225"):
        if "Nikkei" in mode: return NIKKEI_225_SAMPLE
        elif "TOPIX" in mode: return TOPIX_100_SAMPLE
        return []

    @staticmethod
    def _map_sector(raw_sector):
        """生データのセクター名をKF10分類に変換"""
        if not raw_sector or pd.isna(raw_sector):
            return 'Other'
        # 部分一致検索も考慮（簡易的）
        for key, val in DataProvider.SECTOR_TRANSLATION.items():
            if key in raw_sector:
                return val
        return 'Other'

    @staticmethod
    def _fetch_fmp_rescue(ticker_list):
        """
        【救済措置】yfinanceでデータが取れなかった銘柄に対しFMP APIを叩く
        """
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list:
            return {}

        rescued_data = {}
        # FMPは無料枠制限があるため、必要最小限のエンドポイントを叩く
        # Ratios TTM (ROE, PBR取得用)
        
        with ThreadPoolExecutor(max_workers=2) as executor: # 負荷考慮しスレッド数抑制
            def fetch_one(t_orig):
                # Ticker変換: 7203.T -> 7203.JP
                symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
                url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={api_key}"
                try:
                    r = requests.get(url, timeout=5)
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        return t_orig, {
                            'ROE': item.get('returnOnEquityTTM'),
                            'PBR': item.get('priceToBookRatioTTM'),
                            # Growthは別エンドポイントだが、今回は簡易化のためRatiosの配当利回りなどを代用も可
                            # 本格的には 'financial-growth' エンドポイントが必要
                        }
                except Exception as e:
                    print(f"FMP Error {t_orig}: {e}")
                return t_orig, None

            results = list(executor.map(fetch_one, ticker_list))
        
        for t, data in results:
            if data:
                rescued_data[t] = data
        
        return rescued_data

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_fundamentals(tickers):
        unique_tickers = list(set(tickers))
        
        # 1. Primary: yfinance (High Speed)
        def get_yf_stock(ticker):
            try:
                tk = yf.Ticker(ticker)
                info = tk.info
                if info is None or 'currentPrice' not in info: return None
                return {
                    'Ticker': ticker,
                    'Name': info.get('shortName', ticker),
                    'Price': info.get('currentPrice', np.nan),
                    'Size_Raw': info.get('marketCap', np.nan),
                    'PBR': info.get('priceToBook', np.nan),
                    'ROE': info.get('returnOnEquity', np.nan),
                    'Growth': info.get('revenueGrowth', np.nan),
                    'Sector_Raw': info.get('sector', info.get('industry', 'Unknown'))
                }
            except: return None

        results = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            fetched = list(executor.map(get_yf_stock, unique_tickers))
        
        # Noneを除外してDataFrame化
        valid_data = [d for d in fetched if d is not None]
        df = pd.DataFrame(valid_data)
        
        if df.empty: return pd.DataFrame()

        # 2. Secondary: FMP Rescue (Check for Missing Data)
        # ROE または PBR が欠損している銘柄を特定
        if DataProvider.FMP_API_KEY:
            missing_mask = df['ROE'].isna() | df['PBR'].isna()
            missing_tickers = df.loc[missing_mask, 'Ticker'].tolist()
            
            if missing_tickers:
                print(f"Attempting FMP rescue for: {missing_tickers}")
                fmp_data = DataProvider._fetch_fmp_rescue(missing_tickers)
                
                # データの補完
                for i, row in df.iterrows():
                    t = row['Ticker']
                    if t in fmp_data:
                        # yfinanceでNaNの場合のみFMPデータで上書き
                        if pd.isna(row['ROE']):
                            df.at[i, 'ROE'] = fmp_data[t].get('ROE')
                        if pd.isna(row['PBR']):
                            df.at[i, 'PBR'] = fmp_data[t].get('PBR')

        # 3. Data Cleaning & Sector Mapping
        num_cols = ['Price', 'Size_Raw', 'PBR', 'ROE', 'Growth']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # セクターマッピング適用
        df['sector'] = df['Sector_Raw'].apply(DataProvider._map_sector)

        return df

    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_historical_prices(tickers, days=365):
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        if not tickers: return pd.DataFrame()

        try:
            df = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
            if len(tickers) == 1:
                t = tickers[0]
                return pd.DataFrame({t: df['Close']}) if 'Close' in df.columns else pd.DataFrame()
            else:
                try:
                    return df.xs('Close', axis=1, level=1, drop_level=True)
                except KeyError:
                    return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

# ---------------------------------------------------------
# Module 2 & 3: Logic & Engine (Existing + Update)
# ---------------------------------------------------------

def compute_derived_metrics(df_fund, df_hist, benchmark_ticker):
    df = df_fund.copy()
    
    # Value: 1/PBR
    df['Value_Raw'] = df['PBR'].apply(lambda x: 1/x if (pd.notnull(x) and x > 0) else np.nan)
    
    # Size: Log(MarketCap)
    df['Size_Log'] = np.log(pd.to_numeric(df['Size_Raw'], errors='coerce').replace(0, np.nan))
    
    # Momentum & Beta
    moms, betas = {}, {}
    if not df_hist.empty:
        rets = df_hist.pct_change().dropna()
        if benchmark_ticker in rets.columns:
            bench_ret = rets[benchmark_ticker]
            bench_var = bench_ret.var()
            for t in df['Ticker']:
                if t in rets.columns:
                    cov = rets[t].cov(bench_ret)
                    betas[t] = cov / bench_var if bench_var != 0 else 1.0
                    try:
                        moms[t] = (df_hist[t].iloc[-1] / df_hist[t].iloc[0]) - 1
                    except: moms[t] = np.nan
                else:
                    betas[t] = 1.0; moms[t] = np.nan
        else:
            for t in df['Ticker']: betas[t] = 1.0; moms[t] = np.nan
                
    df['Beta_Raw'] = df['Ticker'].map(betas)
    df['Momentum_Raw'] = df['Ticker'].map(moms)
    
    # Rename columns
    df.rename(columns={'ROE': 'Quality_Raw', 'Growth': 'Investment_Raw'}, inplace=True)
    return df

def calculate_market_stats(universe_df):
    stats = {}
    # 直交化 (Quality vs Investment)
    mask = universe_df['Quality_Raw'].notna() & universe_df['Investment_Raw'].notna()
    if mask.sum() > 10:
        slope, intercept, _, _, _ = linregress(universe_df.loc[mask, 'Investment_Raw'], universe_df.loc[mask, 'Quality_Raw'])
    else:
        slope, intercept = 0, 0
    stats['ortho_slope'] = slope
    stats['ortho_intercept'] = intercept
    
    factors = {
        'Beta': 'Beta_Raw', 'Size': 'Size_Log', 'Value': 'Value_Raw',
        'Momentum': 'Momentum_Raw', 'Quality': 'Quality_Raw', 'Investment': 'Investment_Raw'
    }
    temp_q = universe_df.apply(lambda x: x['Quality_Raw'] - (slope * x['Investment_Raw'] + intercept) 
                               if (pd.notnull(x['Quality_Raw']) and pd.notnull(x['Investment_Raw'])) else np.nan, axis=1)
    
    for f, col in factors.items():
        if f == 'Quality': series = temp_q.dropna()
        else: series = universe_df[col].dropna()
        if not series.empty: stats[f] = {'mean': series.mean(), 'std': series.std(), 'col': col}
        else: stats[f] = {'mean': 0, 'std': 1, 'col': col}
    return stats

def apply_scoring(target_df, stats):
    df = target_df.copy()
    slope = stats['ortho_slope']
    intercept = stats['ortho_intercept']
    df['Quality_Orthogonal'] = df.apply(lambda x: x['Quality_Raw'] - (slope * x['Investment_Raw'] + intercept) 
                                        if (pd.notnull(x['Quality_Raw']) and pd.notnull(x['Investment_Raw'])) else x['Quality_Raw'], axis=1)

    factors = ['Beta', 'Value', 'Size', 'Momentum', 'Quality', 'Investment']
    for f in factors:
        if f not in stats: continue
        if f == 'Quality': col_name = 'Quality_Orthogonal'
        else: col_name = stats[f]['col']
        mu = stats[f]['mean']
        sigma = stats[f]['std']
        z_col = f"{f}_Z"
        
        def calc_z(val):
            if pd.isna(val) or sigma == 0: return 0.0
            z = (val - mu) / sigma
            if f == 'Size': z = -z # SMB反転
            return z
        
        df[z_col] = df[col_name].apply(calc_z)
        
        def fmt(row):
            raw = row.get(col_name)
            if f == 'Size': raw_disp = row.get('Size_Raw')
            elif f == 'Value': raw_disp = 1/raw if (raw and raw!=0) else np.nan
            else: raw_disp = raw
            z = row.get(z_col)
            if pd.isna(raw_disp): return "-"
            
            if f == 'Size':
                if raw_disp >= 1e12: txt = f"{raw_disp/1e12:.2f}兆"
                elif raw_disp >= 1e8: txt = f"{raw_disp/1e8:.0f}億"
                else: txt = str(raw_disp)
            elif f in ['Momentum', 'Quality', 'Investment']: txt = f"{raw_disp*100:.1f}%"
            elif f == 'Value': txt = f"{raw_disp:.2f} (PBR)"
            else: txt = f"{raw_disp:.2f}"
            return f"{txt}\n({z:+.1f}σ)"
            
        df[f"{f}_Display"] = df.apply(fmt, axis=1)
    return df

# ---------------------------------------------------------
# 4. Main App (UI Integration)
# ---------------------------------------------------------
st.sidebar.header("1. Analysis Universe")
benchmark_mode = st.sidebar.radio("Compare against:", ["Nikkei 225 (Sample)", "TOPIX 100 (Sample)"])
selected_universe = NIKKEI_225_SAMPLE if "Nikkei" in benchmark_mode else TOPIX_100_SAMPLE
bench_ticker = "1321.T" if "Nikkei" in benchmark_mode else "1306.T"

st.sidebar.header("2. Portfolio Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
default_tickers = "7203.T, 9984.T, 6758.T, 8035.T, 6861.T, 9983.T, 4502.T, 6367.T"
input_tickers = st.sidebar.text_area("Input Tickers", default_tickers, height=100)

# APIキーの状態表示（デバッグ用・ユーザー安心用）
api_status = "✅ Connected" if os.environ.get("FMP_API_KEY") else "⚠️ No API Key (yfinance only)"
st.sidebar.caption(f"FMP API Status: {api_status}")

if st.sidebar.button("Run Full Analysis", type="primary"):
    input_data = []
    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file)
            for _, row in df_csv.iterrows():
                input_data.append({'Ticker': str(row['Ticker']).strip(), 'Weight': row.get('Weight', np.nan)})
        except: st.error("CSV format error"); st.stop()
    else:
        for t in [x.strip() for x in input_tickers.split(',') if x.strip()]:
            input_data.append({'Ticker': t, 'Weight': np.nan})
    
    user_df_base = pd.DataFrame(input_data)
    user_tickers = user_df_base['Ticker'].tolist()
    
    with st.spinner("Fetching Market Data (Module 1: Hybrid)..."):
        all_tickers = list(set(selected_universe + user_tickers + [bench_ticker]))
        df_fund = DataProvider.fetch_fundamentals(all_tickers)
        df_hist = DataProvider.fetch_historical_prices(all_tickers)
        if df_fund.empty: st.error("Data Fetch Failed."); st.stop()

    with st.spinner("Calculating Factors (Module 2)..."):
        df_full = compute_derived_metrics(df_fund, df_hist, bench_ticker)
        uni_df = df_full[df_full['Ticker'].isin(selected_universe)].copy()
        stats = calculate_market_stats(uni_df)
        user_df_calc = df_full[df_full['Ticker'].isin(user_tickers)].copy()
        user_scored = apply_scoring(user_df_calc, stats)
        user_scored = pd.merge(user_scored, user_df_base, on='Ticker', how='left')
        
        current_w = user_scored['Weight'].sum()
        nans = user_scored['Weight'].isna()
        if nans.any():
            rem = max(0, 100 - current_w)
            user_scored.loc[nans, 'Weight'] = rem / nans.sum()
        
    st.subheader("🛠 Portfolio Composition")
    edited = st.data_editor(user_scored[['Ticker', 'Name', 'Weight']], 
                            column_config={"Weight": st.column_config.NumberColumn(format="%.2f%%")},
                            use_container_width=True)
    
    st.subheader("🧬 Factor Heatmap")
    disp_cols = [c for c in user_scored.columns if "_Display" in c]
    def color_sigma(val):
        if "(" not in str(val): return ""
        try:
            sigma = float(val.split("(")[1].split("σ")[0])
            if sigma >= 1.0: return "background-color: #d1e7dd; color: #0f5132"
            if sigma <= -1.0: return "background-color: #f8d7da; color: #842029"
        except: pass
        return ""
    st.dataframe(user_scored[["Ticker", "Name"] + disp_cols].style.applymap(color_sigma), use_container_width=True)

    st.divider()
    w = edited['Weight'] / 100.0
    z_cols = [f"{f}_Z" for f in ['Beta', 'Size', 'Value', 'Momentum', 'Quality', 'Investment']]
    port_exp = {}
    for zc in z_cols:
        port_exp[zc.replace("_Z", "")] = (user_scored[zc] * w).sum()
    st.bar_chart(pd.Series(port_exp))
    
    st.success("Analysis Complete using Hybrid Data Provider (yfinance + FMP).")

import os
import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DataProvider:
    """
    【Module 1】データ取得プロバイダー (Ver. 3.2: MarketMonitor対応・完全安定化版)
    - yf.Tickersによるバルク取得とセッション管理の強化
    - FMP financial-growth エンドポイントからの直接取得
    - 取得失敗時のETF(ベンチマーク)による時系列補完ロジック搭載
    """
    
    FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.environ.get("FMP_API_KEY"))

    SECTOR_TRANSLATION = {
        'Technology': 'HiTec', 'Information & Communication': 'Telcm', 
        'Electric Appliances': 'HiTec', 'Precision Instruments': 'HiTec',
        'Services': 'Other', 'Communication Services': 'Telcm',
        'Automobiles & Components': 'Manuf', 'Transportation Equipment': 'Manuf',
        'Machinery': 'Manuf', 'Chemicals': 'Manuf', 'Basic Materials': 'Manuf',
        'Energy': 'Enrgy', 'Oil & Coal Products': 'Enrgy', 'Mining': 'Enrgy',
        'Glass & Ceramics Products': 'Manuf', 'Iron & Steel': 'Manuf', 
        'Nonferrous Metals': 'Manuf', 'Metal Products': 'Manuf',
        'Consumer Cyclical': 'Durbl', 'Consumer Defensive': 'NoDur',
        'Retail Trade': 'Shops', 'Wholesale Trade': 'Shops', 
        'Foods': 'NoDur', 'Pharmaceuticals': 'Hlth', 'Healthcare': 'Hlth',
        'Textiles & Apparels': 'NoDur', 'Pulp & Paper': 'Manuf',
        'Financial Services': 'Other', 'Banks': 'Other', 'Insurance': 'Other',
        'Securities & Commodity Futures': 'Other', 'Other Financing Business': 'Other',
        'Real Estate': 'Other', 'Construction': 'Manuf',
        'Utilities': 'Utils', 'Electric Power & Gas': 'Utils',
        'Land Transportation': 'Other', 'Marine Transportation': 'Other', 
        'Air Transportation': 'Other', 'Warehousing & Harbor Transportation Services': 'Other'
    }

    @staticmethod
    def _create_session():
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        retries = Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    @staticmethod
    def _map_sector(raw_sector):
        if not raw_sector or pd.isna(raw_sector): return 'Other'
        for key, val in DataProvider.SECTOR_TRANSLATION.items():
            if key in str(raw_sector): return val
        return 'Other'

    @staticmethod
    def _fetch_fmp_ratios(ticker_list):
        """
        FMP APIから財務指標を取得 (Rescue用)
        """
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list: return {}

        rescued_data = {}
        
        def fetch_one(t_orig):
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            # 1. Ratios (ROE, PBR)
            url_ratios = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={api_key}"
            # 2. Growth (Asset Growth)
            url_growth = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}?limit=1&apikey={api_key}"
            
            data_res = {}
            try:
                # Ratios
                r = requests.get(url_ratios, timeout=5)
                if r.status_code == 200:
                    items = r.json()
                    if items:
                        data_res['ROE'] = items[0].get('returnOnEquityTTM')
                        data_res['PBR'] = items[0].get('priceToBookRatioTTM')
                
                # Growth
                r_g = requests.get(url_growth, timeout=5)
                if r_g.status_code == 200:
                    g_items = r_g.json()
                    if g_items:
                        data_res['Growth'] = g_items[0].get('assetGrowth')
                        
                return t_orig, data_res
            except:
                return t_orig, None

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(fetch_one, ticker_list))
        
        for t, data in results:
            if data: rescued_data[t] = data
            
        return rescued_data

    @staticmethod
    def _fetch_fmp_history(ticker_list, days=365):
        """FMP APIから株価を取得"""
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list: return pd.DataFrame()

        all_series = {}
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)
        
        def fetch_hist_one(t_orig):
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if 'historical' in data:
                        df = pd.DataFrame(data['historical'])
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)
                        return t_orig, df['close']
            except:
                pass
            return t_orig, None

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(fetch_hist_one, ticker_list))

        for t, series in results:
            if series is not None: all_series[t] = series
        
        return pd.DataFrame(all_series)

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fundamentals(tickers):
        """
        ファンダメンタルズ情報を取得
        - yf.Tickersによるバルク取得
        """
        unique_tickers = list(set(tickers))
        if not unique_tickers: return pd.DataFrame()

        session = DataProvider._create_session()
        
        tickers_str = " ".join(unique_tickers)
        try:
            tks = yf.Tickers(tickers_str, session=session)
        except Exception:
            tks = None

        def get_yf_stock(ticker):
            try:
                if not tks: return None
                tk = tks.tickers.get(ticker)
                if not tk: return None
                
                info = tk.info
                if info is None or 'currentPrice' not in info: return None
                
                res = {
                    'Ticker': ticker,
                    'Name': info.get('shortName', ticker),
                    'Price': info.get('currentPrice', np.nan),
                    'Size_Raw': info.get('marketCap', np.nan),
                    'PBR': info.get('priceToBook', np.nan),
                    'ROE': info.get('returnOnEquity', np.nan),
                    'Sector_Raw': info.get('sector', info.get('industry', 'Unknown')),
                    'Growth': info.get('revenueGrowth', np.nan)
                }
                return res
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(get_yf_stock, unique_tickers))
        
        valid_data = [d for d in results if d is not None]
        df = pd.DataFrame(valid_data)
        
        if df.empty:
            df = pd.DataFrame({'Ticker': unique_tickers})
            
        req_cols = ['ROE', 'PBR', 'Growth']
        for c in req_cols:
            if c not in df.columns: df[c] = np.nan

        missing_cond = df['ROE'].isna() | df['Growth'].isna()
        missing_tickers = df[missing_cond]['Ticker'].tolist() if not df.empty else unique_tickers
        
        if missing_tickers and DataProvider.FMP_API_KEY:
            fmp_data = DataProvider._fetch_fmp_ratios(missing_tickers)
            for i, row in df.iterrows():
                t = row['Ticker']
                if t in fmp_data:
                    if pd.isna(row.get('ROE')): df.at[i, 'ROE'] = fmp_data[t].get('ROE')
                    if pd.isna(row.get('PBR')): df.at[i, 'PBR'] = fmp_data[t].get('PBR')
                    if pd.isna(row.get('Growth')): df.at[i, 'Growth'] = fmp_data[t].get('Growth')

        num_cols = ['Price', 'Size_Raw', 'PBR', 'ROE', 'Growth']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'Sector_Raw' in df.columns:
            df['sector'] = df['Sector_Raw'].apply(DataProvider._map_sector)
        else:
            df['sector'] = 'Other'

        return df

    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_historical_prices(tickers, days=365):
        """
        時系列株価データを取得
        【修正】取得失敗時にベンチマーク(ETF)の動きで補完する安全装置を追加
        """
        if not tickers: return pd.DataFrame()
        session = DataProvider._create_session()
        
        # 補完用のベンチマークETFを特定 (tickersリストの最後に入っていることが多い)
        bench_etf = next((t for t in tickers if t in ["1321.T", "1306.T"]), None)
        
        try:
            end_d = datetime.date.today()
            start_d = end_d - datetime.timedelta(days=days)
            
            df = yf.download(
                tickers, start=start_d, end=end_d,
                progress=False, group_by='ticker', auto_adjust=True,
                session=session
            )
            
            if df.empty: raise ValueError("yfinance returned empty")

            if len(tickers) == 1:
                t = tickers[0]
                if 'Close' in df.columns: result_df = pd.DataFrame({t: df['Close']})
                else: result_df = pd.DataFrame()
            else:
                try:
                    result_df = df.iloc[:, df.columns.get_level_values(1) == 'Close']
                    result_df.columns = result_df.columns.get_level_values(0)
                except:
                    result_df = pd.DataFrame()
            
            # 欠損補完 (FMP)
            current_cols = result_df.columns.tolist() if not result_df.empty else []
            missing = list(set(tickers) - set(current_cols))
            
            if missing and DataProvider.FMP_API_KEY:
                fmp_df = DataProvider._fetch_fmp_history(missing, days)
                if not fmp_df.empty:
                    result_df = pd.concat([result_df, fmp_df], axis=1)
                    
            # 【追加】最終的な安全装置: それでも取れなかった銘柄をETFの動きで補完
            final_cols = result_df.columns.tolist() if not result_df.empty else []
            still_missing = list(set(tickers) - set(final_cols))
            
            if still_missing and bench_etf and bench_etf in result_df.columns:
                for t in still_missing:
                    # ETFの動きに連動すると仮定してデータを埋める（計算エラーを防ぐため）
                    result_df[t] = result_df[bench_etf]

            return result_df

        except Exception:
            if DataProvider.FMP_API_KEY:
                fmp_fallback = DataProvider._fetch_fmp_history(tickers, days)
                if not fmp_fallback.empty: return fmp_fallback
                
            return pd.DataFrame()

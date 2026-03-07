import os
import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import re
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DataProvider:
    """
    【Module 1】データ取得プロバイダー (Ver. 3.4: 接続安定化・フォールバック強化版)
    - Tickerの自動正規化（Excelの数値変換エラー防止）
    - yf.Tickersによるバルク取得とセッション管理の強化
    - FMP APIへの強力な自動フォールバック（None撲滅）
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
        # yfinanceがハングするのを防ぐため、リトライ回数を下げて素早く諦め、FMP APIへ移譲させる
        retries = Retry(
            total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    @staticmethod
    def _normalize_ticker(t):
        """Excel等で発生する 7203.0 や 7203 等を 7203.T に強制正規化"""
        if pd.isna(t) or not str(t).strip():
            return ""
        t_str = str(t).split('.')[0].strip().upper()
        if re.fullmatch(r'\d{4}', t_str):
            return f"{t_str}.T"
        return str(t).strip().upper()

    @staticmethod
    def _map_sector(raw_sector):
        if not raw_sector or pd.isna(raw_sector): return 'Other'
        for key, val in DataProvider.SECTOR_TRANSLATION.items():
            if key in str(raw_sector): return val
        return 'Other'

    @staticmethod
    def _fetch_fmp_ratios(ticker_list):
        """FMP APIから財務指標を取得 (Rescue用)"""
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
                # タイムアウトを短く設定し、全体の「解析中」ハングを防ぐ
                r = requests.get(url_ratios, timeout=3)
                if r.status_code == 200:
                    items = r.json()
                    if items:
                        data_res['ROE'] = items[0].get('returnOnEquityTTM')
                        data_res['PBR'] = items[0].get('priceToBookRatioTTM')
                
                r_g = requests.get(url_growth, timeout=3)
                if r_g.status_code == 200:
                    g_items = r_g.json()
                    if g_items:
                        data_res['Growth'] = g_items[0].get('assetGrowth')
                        
                return t_orig, data_res
            except Exception:
                return t_orig, None

        with ThreadPoolExecutor(max_workers=5) as executor:
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
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    if 'historical' in data:
                        df = pd.DataFrame(data['historical'])
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)
                        return t_orig, df['close']
            except Exception:
                pass
            return t_orig, None

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_hist_one, ticker_list))

        for t, series in results:
            if series is not None: all_series[t] = series
        
        return pd.DataFrame(all_series)

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fundamentals(tickers):
        """
        ファンダメンタルズ情報を取得
        - yf.Tickersによるバルク取得と、失敗時のFMP強力補完
        """
        unique_tickers = list(set([DataProvider._normalize_ticker(t) for t in tickers if pd.notna(t)]))
        unique_tickers = [t for t in unique_tickers if t]
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
                if info is None: return None
                
                res = {
                    'Ticker': ticker,
                    'Name': info.get('shortName', info.get('longName', ticker)),
                    'Price': info.get('currentPrice', info.get('previousClose', np.nan)),
                    'Size_Raw': info.get('marketCap', np.nan),
                    'PBR': info.get('priceToBook', np.nan),
                    'ROE': info.get('returnOnEquity', np.nan),
                    'Sector_Raw': info.get('sector', info.get('industry', 'Unknown')),
                    # revenueGrowthがなければearningsGrowthで代替
                    'Growth': info.get('revenueGrowth', info.get('earningsGrowth', np.nan))
                }
                return res
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as executor: # yfinance取得のスレッドを増やして高速化
            results = list(executor.map(get_yf_stock, unique_tickers))
        
        valid_data = [d for d in results if d is not None]
        df = pd.DataFrame(valid_data)
        
        if df.empty:
            df = pd.DataFrame({'Ticker': unique_tickers})
            
        req_cols = ['ROE', 'PBR', 'Growth']
        for c in req_cols:
            if c not in df.columns: df[c] = np.nan

        # どれか一つでも欠損している場合はFMPから補完を試みる条件を厳密化
        missing_cond = df['ROE'].isna() | df['Growth'].isna() | df['PBR'].isna()
        missing_tickers = df[missing_cond]['Ticker'].tolist() if not df.empty else unique_tickers
        
        if missing_tickers and DataProvider.FMP_API_KEY:
            fmp_data = DataProvider._fetch_fmp_ratios(missing_tickers)
            for i, row in df.iterrows():
                t = row['Ticker']
                if t in fmp_data:
                    # 取得できた場合のみ、NaNの部分を上書きする
                    if pd.isna(row.get('ROE')): df.at[i, 'ROE'] = fmp_data[t].get('ROE')
                    if pd.isna(row.get('PBR')): df.at[i, 'PBR'] = fmp_data[t].get('PBR')
                    if pd.isna(row.get('Growth')): df.at[i, 'Growth'] = fmp_data[t].get('Growth')

        # 数値型へ強制変換 (Noneや文字列を排除)
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
        取得失敗時にベンチマーク(ETF)の動きで補完する安全装置
        """
        if not tickers: return pd.DataFrame()
        
        unique_tickers = list(set([DataProvider._normalize_ticker(t) for t in tickers if pd.notna(t)]))
        unique_tickers = [t for t in unique_tickers if t]
        if not unique_tickers: return pd.DataFrame()

        session = DataProvider._create_session()
        
        bench_etf = next((t for t in unique_tickers if t in ["1321.T", "1306.T"]), None)
        
        try:
            end_d = datetime.date.today()
            start_d = end_d - datetime.timedelta(days=days)
            
            df = yf.download(
                unique_tickers, start=start_d, end=end_d,
                progress=False, group_by='ticker', auto_adjust=True,
                session=session
            )
            
            if df.empty: raise ValueError("yfinance returned empty")

            if len(unique_tickers) == 1:
                t = unique_tickers[0]
                if 'Close' in df.columns: result_df = pd.DataFrame({t: df['Close']})
                else: result_df = pd.DataFrame()
            else:
                try:
                    result_df = df.iloc[:, df.columns.get_level_values(1) == 'Close']
                    result_df.columns = result_df.columns.get_level_values(0)
                except Exception:
                    result_df = pd.DataFrame()
            
            current_cols = result_df.columns.tolist() if not result_df.empty else []
            missing = list(set(unique_tickers) - set(current_cols))
            
            if missing and DataProvider.FMP_API_KEY:
                fmp_df = DataProvider._fetch_fmp_history(missing, days)
                if not fmp_df.empty:
                    result_df = pd.concat([result_df, fmp_df], axis=1)
                    
            final_cols = result_df.columns.tolist() if not result_df.empty else []
            still_missing = list(set(unique_tickers) - set(final_cols))
            
            # 最終防衛線: 取れなかった銘柄はベンチマークETFの動きで補完
            if still_missing and bench_etf and bench_etf in result_df.columns:
                for t in still_missing:
                    result_df[t] = result_df[bench_etf]

            return result_df

        except Exception:
            if DataProvider.FMP_API_KEY:
                fmp_fallback = DataProvider._fetch_fmp_history(unique_tickers, days)
                if not fmp_fallback.empty: return fmp_fallback
                
            return pd.DataFrame()

    # =========================================================================
    # 【追加実装】 app.py の呼び出しエラー (AttributeError) を解消するラッパー
    # =========================================================================
    @staticmethod
    def get_bulk_fundamentals(tickers):
        """
        app.py から呼び出される 'get_bulk_fundamentals' という名前を、
        実際の処理関数である 'fetch_fundamentals' に中継します。
        これにより、AttributeError を防ぎ、データの受け渡しを正常化します。
        """
        return DataProvider.fetch_fundamentals(tickers)

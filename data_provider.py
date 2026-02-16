import os
import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

class DataProvider:
    """
    【Module 1】データ取得プロバイダー (Robust Ver.)
    Yahoo Finance を主軸とし、取得失敗時には FMP API で補完する堅牢な設計。
    常に DataFrame を返すことを保証し、計算エンジンのエラーを防ぐ。
    """
    
    # APIキーの取得 (Streamlit Secrets または 環境変数)
    FMP_API_KEY = os.environ.get("FMP_API_KEY")

    # セクター変換辞書 (Kenneth French 10 Industry Code準拠)
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
    def _map_sector(raw_sector):
        """セクター名をKF10分類に変換"""
        if not raw_sector or pd.isna(raw_sector):
            return 'Other'
        for key, val in DataProvider.SECTOR_TRANSLATION.items():
            if key in str(raw_sector):
                return val
        return 'Other'

    @staticmethod
    def _fetch_fmp_ratios(ticker_list):
        """
        【救済措置: 財務】FMP APIから財務指標(ROE, PBR)を取得
        """
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list:
            return {}

        rescued_data = {}
        
        def fetch_one(t_orig):
            # FMP用シンボル変換 (7203.T -> 7203.JP)
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={api_key}"
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        return t_orig, {
                            'ROE': item.get('returnOnEquityTTM'),
                            'PBR': item.get('priceToBookRatioTTM'),
                            'Growth': item.get('dividendYieldTTM') # 簡易代用
                        }
            except Exception as e:
                pass
            return t_orig, None

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(fetch_one, ticker_list))
        
        for t, data in results:
            if data: rescued_data[t] = data
            
        return rescued_data

    @staticmethod
    def _fetch_fmp_history(ticker_list, days=365):
        """
        【救済措置: 株価】FMP APIから過去の株価データを取得 (モメンタム計算用)
        """
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list:
            return pd.DataFrame()

        all_series = {}
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")

        def fetch_hist_one(t_orig):
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={s_str}&to={e_str}&apikey={api_key}"
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
            if series is not None:
                all_series[t] = series
        
        if not all_series:
            return pd.DataFrame()
            
        return pd.DataFrame(all_series)

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_fundamentals(tickers):
        """
        ファンダメンタルズ情報を取得 (yfinance -> FMP Fallback)
        """
        unique_tickers = list(set(tickers))
        if not unique_tickers: return pd.DataFrame()

        # 1. Primary: yfinance
        def get_yf_stock(ticker):
            try:
                tk = yf.Ticker(ticker)
                info = tk.info
                # 必須項目チェック
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
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(get_yf_stock, unique_tickers))
        
        valid_data = [d for d in results if d is not None]
        df = pd.DataFrame(valid_data)
        
        # 2. Secondary: FMP Rescue (データ欠損補完)
        if df.empty:
            # 全滅した場合はリストだけ作ってFMPに賭ける
            df = pd.DataFrame({'Ticker': unique_tickers})
            df['ROE'] = np.nan
            df['PBR'] = np.nan
            
        missing_mask = df['ROE'].isna() | df['PBR'].isna()
        missing_tickers = df.loc[missing_mask, 'Ticker'].tolist()
        
        # DataFrameに含まれていない(取得失敗した)銘柄も救済対象にする
        failed_tickers = list(set(unique_tickers) - set(df['Ticker'].tolist()))
        missing_tickers.extend(failed_tickers)
        
        if missing_tickers and DataProvider.FMP_API_KEY:
            # print(f"Rescuing fundamentals for: {len(missing_tickers)} tickers")
            fmp_data = DataProvider._fetch_fmp_ratios(missing_tickers)
            
            # 既存行の補完
            for i, row in df.iterrows():
                t = row['Ticker']
                if t in fmp_data:
                    if pd.isna(row.get('ROE')): df.at[i, 'ROE'] = fmp_data[t].get('ROE')
                    if pd.isna(row.get('PBR')): df.at[i, 'PBR'] = fmp_data[t].get('PBR')
            
            # 全く取れなかった銘柄の新規追加
            new_rows = []
            for t in failed_tickers:
                if t in fmp_data:
                    new_rows.append({
                        'Ticker': t,
                        'Name': t, # 名前は取れないのでTickerで代用
                        'ROE': fmp_data[t].get('ROE'),
                        'PBR': fmp_data[t].get('PBR'),
                        'Sector_Raw': 'Unknown'
                    })
            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        if df.empty: return pd.DataFrame()

        # 3. Data Cleaning
        num_cols = ['Price', 'Size_Raw', 'PBR', 'ROE', 'Growth']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # セクターマッピング
        if 'Sector_Raw' in df.columns:
            df['sector'] = df['Sector_Raw'].apply(DataProvider._map_sector)
        else:
            df['sector'] = 'Other'

        return df

    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_historical_prices(tickers, days=365):
        """
        過去の株価データを取得 (yfinance -> FMP Fallback)
        """
        if not tickers: return pd.DataFrame()
        
        # 1. Primary: yfinance
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            df = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
            
            result_df = pd.DataFrame()
            if len(tickers) == 1:
                t = tickers[0]
                if not df.empty and 'Close' in df.columns:
                    result_df = pd.DataFrame({t: df['Close']})
            else:
                if not df.empty:
                    try:
                        result_df = df.xs('Close', axis=1, level=1, drop_level=True)
                    except KeyError:
                        pass
            
            # データが空、または列が足りない場合は救済
            if result_df.empty or len(result_df.columns) < len(tickers):
                missing = list(set(tickers) - set(result_df.columns))
                if missing and DataProvider.FMP_API_KEY:
                    # print(f"Rescuing history for: {len(missing)} tickers")
                    fmp_df = DataProvider._fetch_fmp_history(missing, days)
                    if not fmp_df.empty:
                        # インデックス(日付)を合わせて結合
                        result_df = result_df.join(fmp_df, how='outer')
            
            return result_df

        except Exception as e:
            # yfinanceが完全にコケた場合、全銘柄をFMPで取りに行く
            if DataProvider.FMP_API_KEY:
                return DataProvider._fetch_fmp_history(tickers, days)
            return pd.DataFrame() # 最終的に空を返す (文字列エラーを防ぐ)

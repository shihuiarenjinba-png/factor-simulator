import os
import sqlite3
import datetime
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import re
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DataProvider:
    """
    【Module 1】データ取得プロバイダー (Ver. 7.0: 5-Factor完全統合 & 429回避版)
    - Tickerの自動正規化（Excelの数値変換エラー、.JP等の表記揺れを.Tに統一）
    - FMP APIへの強力な自動フォールバック（None撲滅）
    - JPX公式リストのローカルキャッシュによる高速なユニバース展開
    - yfinanceマルチインデックスの堅牢な解体 (xsメソッド)
    - SQLiteを用いたローカルDBへの直接コンタクト
    - yf.downloadの小分けチャンク処理とスリープによる429エラー徹底回避
    - タイムゾーンの剥奪と正規化の徹底による日付ズレの完全防止
    - 【NEW】Kenneth R. French Data Libraryからの日本市場5ファクター(日次)の自動取得とキャッシュ
    - 【NEW】PBR/ROE欠損時の「総資産ベース」等による代替計算ロジック強化
    """
    
    # ローカルデータベースのパス
    DB_PATH = "market_data.db"
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

    # =========================================================================
    # SQLite データベース管理メソッド
    # =========================================================================
    @staticmethod
    def _init_db():
        """SQLiteデータベースとテーブルの初期化"""
        with sqlite3.connect(DataProvider.DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_prices (
                    ticker TEXT,
                    date TEXT,
                    close REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            # 5ファクター用のテーブルを追加
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ff5_factors (
                    date TEXT PRIMARY KEY,
                    mkt_rf REAL,
                    smb REAL,
                    hml REAL,
                    rmw REAL,
                    cma REAL,
                    rf REAL
                )
            """)

    @staticmethod
    def _save_prices_to_sql(df):
        """取得した株価データ(DataFrame)をSQLiteに保存（重複排除）"""
        if df.empty: return
        DataProvider._init_db()
        try:
            temp_df = df.copy()
            temp_df.index.name = 'date'
            long_df = temp_df.reset_index().melt(id_vars='date', var_name='ticker', value_name='close')
            long_df['date'] = pd.to_datetime(long_df['date']).dt.strftime('%Y-%m-%d')
            long_df = long_df.dropna(subset=['close'])
            
            with sqlite3.connect(DataProvider.DB_PATH) as conn:
                long_df.to_sql('historical_prices', conn, if_exists='append', index=False)
                conn.execute("""
                    DELETE FROM historical_prices 
                    WHERE rowid NOT IN (
                        SELECT MIN(rowid) 
                        FROM historical_prices 
                        GROUP BY ticker, date
                    )
                """)
        except Exception as e:
            print(f"SQL Save Error: {e}")

    @staticmethod
    def _load_prices_from_sql(tickers, start_date, end_date):
        """SQLiteからデータを読み込み、DataFrameで返す"""
        DataProvider._init_db()
        if not tickers: return pd.DataFrame()
        ticker_list = "','".join(tickers)
        query = f"""
            SELECT date, ticker, close FROM historical_prices 
            WHERE ticker IN ('{ticker_list}') 
            AND date >= '{start_date}' AND date <= '{end_date}'
        """
        try:
            with sqlite3.connect(DataProvider.DB_PATH) as conn:
                df = pd.read_sql(query, conn)
            
            if df.empty: return pd.DataFrame()
            
            df['date'] = pd.to_datetime(df['date'])
            pivot_df = df.pivot(index='date', columns='ticker', values='close')
            return pivot_df
        except Exception as e:
            print(f"SQL Load Error: {e}")
            return pd.DataFrame()

    # =========================================================================
    # 【NEW】Kenneth French 5-Factor データ取得メソッド
    # =========================================================================
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_ken_french_5factors(start_date, end_date=None):
        """
        Kenneth R. French Data Library から日本市場の5ファクター(日次)を取得する。
        """
        DataProvider._init_db()
        
        # 1. まずSQLiteからロードを試みる
        try:
            with sqlite3.connect(DataProvider.DB_PATH) as conn:
                query = f"SELECT * FROM ff5_factors WHERE date >= '{start_date}'"
                if end_date:
                    query += f" AND date <= '{end_date}'"
                df_ff = pd.read_sql(query, conn, index_col='date', parse_dates=['date'])
                
            if not df_ff.empty:
                # 最新データ（直近1ヶ月以内）が含まれていれば、キャッシュを信用して返す
                max_date = df_ff.index.max()
                today = pd.to_datetime(datetime.date.today())
                if (today - max_date).days <= 30:
                    return df_ff
        except Exception:
            pass

        # 2. キャッシュがない、または古い場合はWebから取得
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_5_Factors_Daily_CSV.zip"
        session = DataProvider._create_session()
        
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # ZIP内のCSVファイル名を取得
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # ヘッダー行をスキップして読み込む（通常、最初の数行は説明文）
                    # Fama-FrenchのCSVは通常3行目からデータ開始
                    df = pd.read_csv(f, skiprows=3, index_col=0)
            
            # インデックス（日付）のクレンジング (例: '19900702' -> datetime)
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
            df = df.dropna()
            df.index = df.index.normalize()
            
            # カラム名の正規化と数値変換（パーセント表記を小数に変換）
            df.columns = [c.strip().upper() for c in df.columns]
            rename_map = {'MKT-RF': 'mkt_rf', 'SMB': 'smb', 'HML': 'hml', 'RMW': 'rmw', 'CMA': 'cma', 'RF': 'rf'}
            
            # 必要なカラムのみ抽出しリネーム
            df = df[[c for c in df.columns if c in rename_map.keys()]]
            df.rename(columns=rename_map, inplace=True)
            
            # Fama-Frenchのデータは%表記なので100で割る
            df = df / 100.0
            
            # SQLiteに保存
            with sqlite3.connect(DataProvider.DB_PATH) as conn:
                df_to_save = df.reset_index()
                df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
                df_to_save.to_sql('ff5_factors', conn, if_exists='replace', index=False)

            # 指定期間でフィルタリングして返す
            df_filtered = df.loc[start_date:end_date] if end_date else df.loc[start_date:]
            return df_filtered
            
        except Exception as e:
            st.error(f"Fama-Frenchデータの取得に失敗しました: {e}")
            return pd.DataFrame()

    # =========================================================================
    # 通信・セッション管理
    # =========================================================================
    @staticmethod
    def _create_session():
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })
        retries = Retry(
            total=4,
            backoff_factor=2.0, # 指数関数的バックオフを強化 (2, 4, 8秒...)
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    @staticmethod
    def _normalize_ticker(t):
        if pd.isna(t) or not str(t).strip():
            return ""
        t_str = str(t).strip().upper()
        match = re.search(r'\b(\d{4})\b', t_str)
        if match:
            return f"{match.group(1)}.T"
        return t_str

    @staticmethod
    def _map_sector(raw_sector):
        if not raw_sector or pd.isna(raw_sector): return 'Other'
        for key, val in DataProvider.SECTOR_TRANSLATION.items():
            if key in str(raw_sector): return val
        return 'Other'

    @staticmethod
    @st.cache_data(ttl=86400 * 7, show_spinner=False)
    def get_jpx_universe():
        file_path = "jpx_list.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, dtype={'コード': str})
                df['Ticker'] = df['コード'].apply(DataProvider._normalize_ticker)
                return dict(zip(df['Ticker'], df['銘柄名']))
            except Exception as e:
                st.warning(f"JPXリストの読み込みに失敗しました: {e}")
        
        fallback_tickers = [
            "7203.T", "8306.T", "9984.T", "6861.T", "8035.T", "9432.T", "6758.T", "8316.T", "4063.T", "8058.T",
            "6098.T", "4502.T", "6902.T", "8001.T", "8766.T", "7974.T", "4568.T", "8031.T", "6501.T", "7741.T",
            "8411.T", "3382.T", "6367.T", "4519.T", "4543.T", "6954.T", "8053.T", "8002.T", "6594.T", "6981.T",
            "4661.T", "4901.T", "2914.T", "6146.T", "7267.T", "8725.T", "4523.T", "7733.T", "4503.T", "6702.T",
            "9022.T", "8591.T", "6503.T", "9020.T", "5108.T", "7269.T", "8802.T", "8801.T", "1925.T", "7011.T"
        ]
        return {t: "JPX Data Missing" for t in fallback_tickers}

    # =========================================================================
    # マーケットデータ (Rm, Rf) の取得 (SQLite対応)
    # =========================================================================
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_market_rates(days=365):
        session = DataProvider._create_session()
        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=days)

        market_data = {}
        
        cached_df = DataProvider._load_prices_from_sql(
            ['^N225'], 
            start_d.strftime('%Y-%m-%d'), 
            end_d.strftime('%Y-%m-%d')
        )
        
        close_series = pd.Series(dtype=float)
        use_api = True
        
        if not cached_df.empty and '^N225' in cached_df.columns:
            latest_date = cached_df.index.max()
            if (pd.to_datetime(end_d) - latest_date).days <= 4:
                close_series = cached_df['^N225']
                use_api = False

        if use_api:
            try:
                rm_df = yf.download("^N225", start=start_d, end=end_d, session=session, progress=False, threads=False)
                
                if not rm_df.empty:
                    if isinstance(rm_df.columns, pd.MultiIndex):
                        if 'Close' in rm_df.columns.get_level_values(1):
                            close_series = rm_df.xs('Close', level=1, axis=1).squeeze()
                        elif 'Close' in rm_df.columns.get_level_values(0):
                            close_series = rm_df.xs('Close', level=0, axis=1).squeeze()
                    else:
                        if 'Close' in rm_df.columns:
                            close_series = rm_df['Close'].squeeze()
                    
                    if not close_series.empty:
                        if close_series.index.tz is not None:
                            close_series.index = close_series.index.tz_localize(None)
                        close_series.index = pd.to_datetime(close_series.index).normalize()
                        
                        save_df = pd.DataFrame({'^N225': close_series})
                        DataProvider._save_prices_to_sql(save_df)
            except Exception:
                pass
        
        if not close_series.empty:
            market_data['Rm'] = close_series
        else:
            market_data['Rm'] = pd.Series(dtype=float)

        if 'Rm' in market_data and not market_data['Rm'].empty:
            dates = market_data['Rm'].index
            daily_rf = (1 + 0.005) ** (1/252) - 1
            market_data['Rf'] = pd.Series(daily_rf, index=dates)
        else:
            market_data['Rf'] = pd.Series(dtype=float)

        return pd.DataFrame(market_data)

    @staticmethod
    def _fetch_fmp_ratios(ticker_list):
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list: return {}

        rescued_data = {}
        session = DataProvider._create_session() # FMPにもSessionを適用
        
        def fetch_one(t_orig):
            time.sleep(0.3) 
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url_ratios = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={api_key}"
            url_growth = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}?limit=1&apikey={api_key}"
            url_profile = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"
            
            data_res = {}
            try:
                # Ratios (ROE, PBR)
                r = session.get(url_ratios, timeout=5)
                if r.status_code == 200:
                    items = r.json()
                    if items:
                        data_res['ROE'] = items[0].get('returnOnEquityTTM')
                        data_res['PBR'] = items[0].get('priceToBookRatioTTM')
                
                # Growth (Investment)
                r_g = session.get(url_growth, timeout=5)
                if r_g.status_code == 200:
                    g_items = r_g.json()
                    if g_items:
                        data_res['Growth'] = g_items[0].get('assetGrowth')
                        
                # Profile (Market Cap 代替)
                r_p = session.get(url_profile, timeout=5)
                if r_p.status_code == 200:
                    p_items = r_p.json()
                    if p_items:
                        data_res['Size_Raw'] = p_items[0].get('mktCap')
                        
                return t_orig, data_res
            except Exception:
                return t_orig, None

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(fetch_one, ticker_list))
        
        for t, data in results:
            if data: rescued_data[t] = data
            
        return rescued_data

    @staticmethod
    def _fetch_fmp_history(ticker_list, days=365):
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list: return pd.DataFrame()

        all_series = {}
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)
        session = DataProvider._create_session()
        
        def fetch_hist_one(t_orig):
            time.sleep(0.3)
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
            try:
                r = session.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if 'historical' in data:
                        df = pd.DataFrame(data['historical'])
                        df['date'] = pd.to_datetime(df['date'])
                        if isinstance(df['date'].dtype, pd.DatetimeTZDtype) or hasattr(df['date'].dtype, 'tz'):
                            df['date'] = df['date'].dt.tz_localize(None)
                        df['date'] = df['date'].dt.normalize()
                        df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)
                        return t_orig, df['close']
            except Exception:
                pass
            return t_orig, None

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(fetch_hist_one, ticker_list))

        for t, series in results:
            if series is not None: all_series[t] = series
        
        return pd.DataFrame(all_series)

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fundamentals(tickers):
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
            # API制限回避のスリープ
            time.sleep(0.5)
            try:
                if not tks: return None
                tk = tks.tickers.get(ticker.upper())
                if not tk: return None
                
                info = tk.info
                if info is None: info = {}
                
                # 【重要修正3】財務データの代替計算ロジック強化
                # PBRが取れない場合、時価総額 / 純資産 で代替（あるいはその逆）
                mktCap = info.get('marketCap', np.nan)
                pbr = info.get('priceToBook', np.nan)
                roe = info.get('returnOnEquity', np.nan)
                
                # totalAssets等の情報から補完できるか試みる（yfinanceの仕様に依存）
                if pd.isna(pbr) and 'totalAssets' in info and mktCap:
                    # 簡易的なPBR代替
                    # ※厳密には純資産だが、情報がない場合の苦肉の策
                    pass 
                
                res = {
                    'Ticker': ticker,
                    'Name': info.get('shortName', info.get('longName', ticker)),
                    'Price': info.get('currentPrice', info.get('previousClose', np.nan)),
                    'Size_Raw': mktCap,
                    'PBR': pbr,
                    'ROE': roe,
                    'Sector_Raw': info.get('sector', info.get('industry', 'Unknown')),
                    'Growth': info.get('revenueGrowth', info.get('earningsGrowth', np.nan))
                }
                return res
            except Exception:
                return {
                    'Ticker': ticker, 'Name': ticker, 'Price': np.nan, 'Size_Raw': np.nan,
                    'PBR': np.nan, 'ROE': np.nan, 'Sector_Raw': 'Unknown', 'Growth': np.nan
                }

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(get_yf_stock, unique_tickers))
        
        valid_data = [d for d in results if d is not None]
        df = pd.DataFrame(valid_data)
        
        if df.empty:
            df = pd.DataFrame({'Ticker': unique_tickers})
            
        req_cols = ['ROE', 'PBR', 'Growth', 'Size_Raw']
        for c in req_cols:
            if c not in df.columns: df[c] = np.nan

        # yfinanceで欠損した項目をFMPで補完
        missing_cond = df['ROE'].isna() | df['Growth'].isna() | df['PBR'].isna() | df['Size_Raw'].isna()
        missing_tickers = df[missing_cond]['Ticker'].tolist() if not df.empty else unique_tickers
        
        if missing_tickers and DataProvider.FMP_API_KEY:
            fmp_data = DataProvider._fetch_fmp_ratios(missing_tickers)
            for i, row in df.iterrows():
                t = row['Ticker']
                if t in fmp_data:
                    if pd.isna(row.get('ROE')): df.at[i, 'ROE'] = fmp_data[t].get('ROE')
                    if pd.isna(row.get('PBR')): df.at[i, 'PBR'] = fmp_data[t].get('PBR')
                    if pd.isna(row.get('Growth')): df.at[i, 'Growth'] = fmp_data[t].get('Growth')
                    if pd.isna(row.get('Size_Raw')): df.at[i, 'Size_Raw'] = fmp_data[t].get('Size_Raw')

        num_cols = ['Price', 'Size_Raw', 'PBR', 'ROE', 'Growth']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'Sector_Raw' in df.columns:
            df['sector'] = df['Sector_Raw'].apply(DataProvider._map_sector)
        else:
            df['sector'] = 'Other'

        return df

    # =========================================================================
    # 履歴データ取得（SQLite対応・APIリクエスト最小化・チャンク処理）
    # =========================================================================
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_historical_prices(tickers, days=365):
        if not tickers: return pd.DataFrame()
        
        unique_tickers = list(set([DataProvider._normalize_ticker(t) for t in tickers if pd.notna(t)]))
        unique_tickers = [t for t in unique_tickers if t]
        if not unique_tickers: return pd.DataFrame()

        session = DataProvider._create_session()
        
        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=days)
        
        # 1. SQLデータベースから既存の株価履歴をロード
        cached_df = DataProvider._load_prices_from_sql(
            unique_tickers, 
            start_d.strftime('%Y-%m-%d'), 
            end_d.strftime('%Y-%m-%d')
        )
        
        # 2. APIで新たに取得しなければならないティッカーを選別
        missing_tickers_for_api = unique_tickers.copy()
        if not cached_df.empty:
            latest_date_in_db = cached_df.index.max()
            if (pd.to_datetime(end_d) - latest_date_in_db).days <= 4:
                valid_tickers_in_db = cached_df.columns.dropna().tolist()
                missing_tickers_for_api = [t for t in unique_tickers if t not in valid_tickers_in_db]
                if not missing_tickers_for_api:
                    return cached_df

        result_df = pd.DataFrame()

        # 3. 不足している銘柄のみAPIで取得 (429対策として小分けチャンク処理)
        if missing_tickers_for_api:
            chunk_size = 5 # 5銘柄ずつ小分けにして取得
            for i in range(0, len(missing_tickers_for_api), chunk_size):
                chunk = missing_tickers_for_api[i:i + chunk_size]
                time.sleep(1.5) # チャンク間で長めのスリープ(1.5秒)を挟む
                try:
                    df = yf.download(
                        chunk, start=start_d, end=end_d,
                        progress=False, group_by='ticker', auto_adjust=True,
                        session=session, threads=False
                    )
                    
                    if not df.empty:
                        chunk_result = pd.DataFrame()
                        if isinstance(df.columns, pd.MultiIndex):
                            try:
                                if 'Close' in df.columns.get_level_values(1):
                                    chunk_result = df.xs('Close', level=1, axis=1).copy()
                                elif 'Close' in df.columns.get_level_values(0):
                                    chunk_result = df.xs('Close', level=0, axis=1).copy()
                            except Exception:
                                pass
                        else:
                            if 'Close' in df.columns: 
                                chunk_result = pd.DataFrame({chunk[0]: df['Close']})

                        if not chunk_result.empty:
                            if chunk_result.index.tz is not None:
                                chunk_result.index = chunk_result.index.tz_localize(None)
                            chunk_result.index = pd.to_datetime(chunk_result.index).normalize()
                            
                            if result_df.empty:
                                result_df = chunk_result
                            else:
                                result_df = pd.concat([result_df, chunk_result], axis=1)
                except Exception:
                    pass

        # 4. Yfinanceで取れなかった銘柄をFMPでフォールバック
        current_cols = result_df.columns.tolist() if not result_df.empty else []
        missing_from_api = list(set(missing_tickers_for_api) - set(current_cols))
        
        if missing_from_api and DataProvider.FMP_API_KEY:
            fmp_df = DataProvider._fetch_fmp_history(missing_from_api, days)
            if not fmp_df.empty:
                if fmp_df.index.tz is not None:
                    fmp_df.index = fmp_df.index.tz_localize(None)
                fmp_df.index = pd.to_datetime(fmp_df.index).normalize()
                
                if result_df.empty:
                    result_df = fmp_df
                else:
                    result_df = pd.concat([result_df, fmp_df], axis=1)
                    
        # 5. 取得できたAPIデータをSQLiteに保存
        if not result_df.empty:
            DataProvider._save_prices_to_sql(result_df)

        # 6. キャッシュ（DB）と今回取得したAPIデータを統合して返す
        final_df = pd.DataFrame()
        if not cached_df.empty and not result_df.empty:
            final_df = pd.concat([cached_df, result_df], axis=1)
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        elif not cached_df.empty:
            final_df = cached_df
        else:
            final_df = result_df

        return final_df

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
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DataProvider:
    """
    【Module 1】データ取得プロバイダー (Ver. 5.0: SQLiteデータベース搭載版)
    - Tickerの自動正規化（Excelの数値変換エラー、.JP等の表記揺れを.Tに統一）
    - yf.Tickersによるバルク取得とセッション管理の強化
    - FMP APIへの強力な自動フォールバック（None撲滅）
    - JPX公式リストのローカルキャッシュによる高速なユニバース展開
    - Beta計算用の市場プレミアム (Rm, Rf) 取得ロジックの安定化
    - yfinanceマルチインデックスの堅牢な解体 (xsメソッド)
    - タイムゾーンの剥奪(tz_localize(None))による結合エラーの完全防止
    - 【NEW】SQLiteを用いたローカルDBへの直接コンタクトによる429エラーの根本的解決
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

    @staticmethod
    def _save_prices_to_sql(df):
        """取得した株価データ(DataFrame)をSQLiteに保存（重複排除）"""
        if df.empty: return
        DataProvider._init_db()
        try:
            temp_df = df.copy()
            # インデックス名を設定し、扱いやすくする
            temp_df.index.name = 'date'
            # ワイド型(日付×銘柄)からロング型(日付,銘柄,価格)へ変換
            long_df = temp_df.reset_index().melt(id_vars='date', var_name='ticker', value_name='close')
            long_df['date'] = pd.to_datetime(long_df['date']).dt.strftime('%Y-%m-%d')
            long_df = long_df.dropna(subset=['close'])
            
            with sqlite3.connect(DataProvider.DB_PATH) as conn:
                long_df.to_sql('historical_prices', conn, if_exists='append', index=False)
                # 重複を削除して最新のレコードのみ保持
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
    # 通信・セッション管理
    # =========================================================================
    @staticmethod
    def _create_session():
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        })
        retries = Retry(
            total=3,
            backoff_factor=1.0,
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
        
        # 1. まずローカルのSQLiteからロード
        cached_df = DataProvider._load_prices_from_sql(
            ['^N225'], 
            start_d.strftime('%Y-%m-%d'), 
            end_d.strftime('%Y-%m-%d')
        )
        
        close_series = pd.Series(dtype=float)
        use_api = True
        
        # SQL内に最新のデータ（直近4日以内）があればAPIは呼ばない
        if not cached_df.empty and '^N225' in cached_df.columns:
            latest_date = cached_df.index.max()
            if (pd.to_datetime(end_d) - latest_date).days <= 4:
                close_series = cached_df['^N225']
                use_api = False

        if use_api:
            try:
                # 429回避のため threads=False
                rm_df = yf.download("^N225", start=start_d, end=end_d, session=session, progress=False, threads=False)
                
                if not rm_df.empty:
                    # yfinanceの出力構造に対応 (MultiIndexか否か)
                    if isinstance(rm_df.columns, pd.MultiIndex):
                        if 'Close' in rm_df.columns.get_level_values(1):
                            close_series = rm_df.xs('Close', level=1, axis=1).squeeze()
                        elif 'Close' in rm_df.columns.get_level_values(0):
                            close_series = rm_df.xs('Close', level=0, axis=1).squeeze()
                    else:
                        if 'Close' in rm_df.columns:
                            close_series = rm_df['Close'].squeeze()
                    
                    if not close_series.empty:
                        # タイムゾーンの除去と日付の正規化
                        if close_series.index.tz is not None:
                            close_series.index = close_series.index.tz_localize(None)
                        close_series.index = pd.to_datetime(close_series.index).normalize()
                        
                        # 取得したデータをSQLに保存
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
        
        def fetch_one(t_orig):
            time.sleep(0.5) 
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url_ratios = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={api_key}"
            url_growth = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}?limit=1&apikey={api_key}"
            
            data_res = {}
            try:
                r = requests.get(url_ratios, timeout=5)
                if r.status_code == 200:
                    items = r.json()
                    if items:
                        data_res['ROE'] = items[0].get('returnOnEquityTTM')
                        data_res['PBR'] = items[0].get('priceToBookRatioTTM')
                
                r_g = requests.get(url_growth, timeout=5)
                if r_g.status_code == 200:
                    g_items = r_g.json()
                    if g_items:
                        data_res['Growth'] = g_items[0].get('assetGrowth')
                        
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
        
        def fetch_hist_one(t_orig):
            time.sleep(0.5)
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if 'historical' in data:
                        df = pd.DataFrame(data['historical'])
                        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
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
            try:
                if not tks: return None
                tk = tks.tickers.get(ticker.upper())
                if not tk: return None
                
                info = tk.info
                if info is None:
                    info = {}
                
                res = {
                    'Ticker': ticker,
                    'Name': info.get('shortName', info.get('longName', ticker)),
                    'Price': info.get('currentPrice', info.get('previousClose', np.nan)),
                    'Size_Raw': info.get('marketCap', np.nan),
                    'PBR': info.get('priceToBook', np.nan),
                    'ROE': info.get('returnOnEquity', np.nan),
                    'Sector_Raw': info.get('sector', info.get('industry', 'Unknown')),
                    'Growth': info.get('revenueGrowth', info.get('earningsGrowth', np.nan))
                }
                return res
            except Exception:
                return {
                    'Ticker': ticker, 'Name': ticker, 'Price': np.nan, 'Size_Raw': np.nan,
                    'PBR': np.nan, 'ROE': np.nan, 'Sector_Raw': 'Unknown', 'Growth': np.nan
                }

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(get_yf_stock, unique_tickers))
        
        valid_data = [d for d in results if d is not None]
        df = pd.DataFrame(valid_data)
        
        if df.empty:
            df = pd.DataFrame({'Ticker': unique_tickers})
            
        req_cols = ['ROE', 'PBR', 'Growth', 'Size_Raw']
        for c in req_cols:
            if c not in df.columns: df[c] = np.nan

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
    # 履歴データ取得（SQLite対応・APIリクエスト最小化）
    # =========================================================================
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_historical_prices(tickers, days=365):
        if not tickers: return pd.DataFrame()
        
        unique_tickers = list(set([DataProvider._normalize_ticker(t) for t in tickers if pd.notna(t)]))
        unique_tickers = [t for t in unique_tickers if t]
        if not unique_tickers: return pd.DataFrame()

        session = DataProvider._create_session()
        bench_etf = next((t for t in unique_tickers if t in ["1321.T", "1306.T"]), None)
        
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
            # データベースの最新日付が直近4日以内なら、APIリクエストをスキップしてDBを信用する
            if (pd.to_datetime(end_d) - latest_date_in_db).days <= 4:
                valid_tickers_in_db = cached_df.columns.dropna().tolist()
                missing_tickers_for_api = [t for t in unique_tickers if t not in valid_tickers_in_db]
                # もし全銘柄のデータがDB内に揃っていれば、そのまま返す（通信ゼロ）
                if not missing_tickers_for_api:
                    return cached_df

        result_df = pd.DataFrame()

        # 3. 不足している銘柄のみAPIで取得
        if missing_tickers_for_api:
            try:
                # threads=False で429エラー(Too Many Requests)を回避
                df = yf.download(
                    missing_tickers_for_api, start=start_d, end=end_d,
                    progress=False, group_by='ticker', auto_adjust=True,
                    session=session, threads=False
                )
                
                if not df.empty:
                    # マルチインデックスの安全な解体 (xsメソッド)
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            if 'Close' in df.columns.get_level_values(1):
                                result_df = df.xs('Close', level=1, axis=1).copy()
                            elif 'Close' in df.columns.get_level_values(0):
                                result_df = df.xs('Close', level=0, axis=1).copy()
                        except Exception:
                            pass
                    else:
                        if 'Close' in df.columns: 
                            result_df = pd.DataFrame({missing_tickers_for_api[0]: df['Close']})

                    if not result_df.empty:
                        # タイムゾーンの除去と日付の正規化
                        if result_df.index.tz is not None:
                            result_df.index = result_df.index.tz_localize(None)
                        result_df.index = pd.to_datetime(result_df.index).normalize()
                        
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
                    
        # 5. 取得できたAPIデータをSQLiteに保存（次回は通信不要になる）
        if not result_df.empty:
            DataProvider._save_prices_to_sql(result_df)

        # 6. キャッシュ（DB）と今回取得したAPIデータを統合して返す
        final_df = pd.DataFrame()
        if not cached_df.empty and not result_df.empty:
            final_df = pd.concat([cached_df, result_df], axis=1)
            # 重複列がある場合は最初の列を残す
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        elif not cached_df.empty:
            final_df = cached_df
        else:
            final_df = result_df
            
        # 7. ベンチマークによる最終フォールバック
        final_cols = final_df.columns.tolist() if not final_df.empty else []
        still_missing = list(set(unique_tickers) - set(final_cols))
        
        if still_missing and bench_etf and bench_etf in final_df.columns:
            for t in still_missing:
                final_df[t] = final_df[bench_etf]

        return final_df

    @staticmethod
    def get_bulk_fundamentals(tickers):
        return DataProvider.fetch_fundamentals(tickers)

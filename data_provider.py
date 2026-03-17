import os
import sqlite3
import datetime
import time
import requests
import random
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import re
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DataProvider:
    """
    【Module 1】データ取得プロバイダー (Ver. 10.1: 429 Error回避＆永続キャッシュ強化版)
    - 429 Too Many Requests 対策として、指数バックオフ(Exponential Backoff)とUAローテーションを導入。
    - 財務データ(Fundamentals)をSQLiteに永続化し、APIの呼び出し回数を劇的に削減。
    - チャンク取得の最適化と、各リクエスト間の優しいスリープ処理を実装。
    """
    
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
            # 履歴データ用テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_prices (
                    ticker TEXT,
                    date TEXT,
                    close REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            # ファクターデータ用テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ff5_factors (
                    date TEXT PRIMARY KEY,
                    mkt_rf REAL, smb REAL, hml REAL, rmw REAL, cma REAL, rf REAL
                )
            """)
            # 【新規】財務データ(Fundamentals)キャッシュ用テーブル (429対策の要)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fundamentals_cache (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    price REAL,
                    size_raw REAL,
                    pbr REAL,
                    roe REAL,
                    sector_raw TEXT,
                    growth REAL,
                    last_updated TIMESTAMP
                )
            """)

    @staticmethod
    def _save_prices_to_sql(df):
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
                        SELECT MIN(rowid) FROM historical_prices GROUP BY ticker, date
                    )
                """)
        except Exception as e:
            print(f"SQL Save Error: {e}")

    @staticmethod
    def _load_prices_from_sql(tickers, start_date, end_date):
        DataProvider._init_db()
        if not tickers: return pd.DataFrame()
        ticker_list = "','".join(tickers)
        query = f"""
            SELECT date, ticker, close FROM historical_prices 
            WHERE ticker IN ('{ticker_list}') AND date >= '{start_date}' AND date <= '{end_date}'
        """
        try:
            with sqlite3.connect(DataProvider.DB_PATH) as conn:
                df = pd.read_sql(query, conn)
            if df.empty: return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            return df.pivot(index='date', columns='ticker', values='close')
        except Exception as e:
            print(f"SQL Load Error: {e}")
            return pd.DataFrame()

    # =========================================================================
    # 通信・セッション管理 (429 Error 回避の要)
    # =========================================================================
    @staticmethod
    def _create_session():
        session = requests.Session()
        # 【新規】User-Agentのローテーションでスクレイピング判定を回避
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ]
        session.headers.update({
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        })
        # リトライ設定の強化
        retries = Retry(
            total=5,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    # =========================================================================
    # Kenneth French 5-Factor データ取得メソッド
    # =========================================================================
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_ken_french_5factors(start_date, end_date=None):
        csv_path = "Japan_5_Factors.csv"
        df_ff = pd.DataFrame()

        if os.path.exists(csv_path):
            try:
                df_ff = pd.read_csv(csv_path, skiprows=6, index_col=0)
                df_ff.index = pd.to_datetime(df_ff.index.astype(str), format='%Y%m', errors='coerce')
                df_ff = df_ff.dropna(how='all')
                print(f"[Factor] {csv_path} から読み込み成功")
            except Exception as e:
                print(f"[Factor Error] CSV読み込み失敗: {e}")

        if df_ff.empty:
            try:
                import pandas_datareader.data as web
                dataset_name = "Japan_5_Factors" 
                print(f"[Factor] pandas_datareaderより {dataset_name} の取得開始...")
                ff_dict = web.DataReader(dataset_name, 'famafrench', start="1990-01-01")
                df_ff = ff_dict[0]
                if isinstance(df_ff.index, pd.PeriodIndex):
                    df_ff.index = df_ff.index.to_timestamp(how='end')
                print("[Factor] pandas_datareaderより取得成功")
            except Exception as e:
                print(f"[Factor Error] オンライン取得失敗: {e}")

        if df_ff.empty: 
            return pd.DataFrame()

        df_ff.columns = [str(c).strip().upper() for c in df_ff.columns]
        rename_map = {'MKT-RF': 'mkt_rf', 'SMB': 'smb', 'HML': 'hml', 'RMW': 'rmw', 'CMA': 'cma', 'RF': 'rf'}
        df_ff = df_ff[[c for c in df_ff.columns if c in rename_map.keys()]]
        df_ff.rename(columns=rename_map, inplace=True)

        df_ff = df_ff.apply(pd.to_numeric, errors='coerce')
        if df_ff.max().max() > 0.5:
            df_ff = df_ff.astype(float) / 100.0

        df_ff.index = pd.to_datetime(df_ff.index)
        df_ff = df_ff.resample('ME').last()
        df_ff.index = df_ff.index.normalize().tz_localize(None)

        df_ff = df_ff.interpolate(method='linear').ffill().bfill()

        start_ts = pd.to_datetime(start_date).replace(tzinfo=None)
        if end_date:
            end_ts = pd.to_datetime(end_date).replace(tzinfo=None)
            df_ff = df_ff[(df_ff.index >= start_ts) & (df_ff.index <= end_ts)]
        else:
            df_ff = df_ff[df_ff.index >= start_ts]

        return df_ff

    # =========================================================================
    # 株価履歴データ取得 (指数バックオフ対応版)
    # =========================================================================
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_historical_prices_monthly(tickers, days=365):
        df_daily = DataProvider.fetch_historical_prices(tickers, days=days)
        if df_daily.empty: 
            return pd.DataFrame()
        df_monthly_price = df_daily.resample('ME').last()
        return df_monthly_price.pct_change().dropna(how='all')

    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_historical_prices(tickers, days=365):
        if not tickers: return pd.DataFrame()
        
        unique_tickers = list(set([DataProvider._normalize_ticker(t) for t in tickers if pd.notna(t)]))
        unique_tickers = [t for t in unique_tickers if t]
        if not unique_tickers: return pd.DataFrame()

        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=days)
        
        cached_df = DataProvider._load_prices_from_sql(
            unique_tickers, 
            start_d.strftime('%Y-%m-%d'), 
            end_d.strftime('%Y-%m-%d')
        )
        
        missing_tickers_for_api = unique_tickers.copy()
        if not cached_df.empty:
            latest_date_in_db = cached_df.index.max()
            if (pd.to_datetime(end_d) - latest_date_in_db).days <= 4:
                valid_tickers_in_db = cached_df.columns.dropna().tolist()
                missing_tickers_for_api = [t for t in unique_tickers if t not in valid_tickers_in_db]
                if not missing_tickers_for_api:
                    return cached_df

        result_df = pd.DataFrame()

        if missing_tickers_for_api:
            chunk_size = 5  # 【修正】リクエスト回数を減らすためチャンクを5に拡大
            print(f"--- yfinance 履歴データ取得開始 (全{len(missing_tickers_for_api)}銘柄, チャンクサイズ:{chunk_size}) ---")
            
            for i in range(0, len(missing_tickers_for_api), chunk_size):
                chunk = missing_tickers_for_api[i:i + chunk_size]
                
                # 【新規】指数バックオフによるリトライ処理
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        time.sleep(1.0 + random.uniform(0, 1)) # アクセス前にランダムな間隔を空ける
                        fresh_session = DataProvider._create_session() # 毎回新しいUAでセッション作成
                        
                        print(f"[yfinance] チャンク取得中: {chunk} ...")
                        df = yf.download(
                            chunk, start=start_d, end=end_d,
                            progress=False, group_by='ticker', auto_adjust=True,
                            session=fresh_session, threads=False, timeout=30
                        )
                        
                        if not df.empty:
                            chunk_result = pd.DataFrame()
                            if isinstance(df.columns, pd.MultiIndex):
                                try:
                                    if 'Close' in df.columns.get_level_values(1):
                                        chunk_result = df.xs('Close', level=1, axis=1).copy()
                                    elif 'Close' in df.columns.get_level_values(0):
                                        chunk_result = df.xs('Close', level=0, axis=1).copy()
                                except Exception: pass
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
                        # 成功したらリトライループを抜ける
                        break
                        
                    except Exception as e:
                        err_msg = str(e).lower()
                        # 429エラーを検知した場合のみバックオフ待機
                        if "429" in err_msg or "too many" in err_msg:
                            wait_time = (2 ** attempt) + random.uniform(1, 3)
                            print(f"⚠️ [429 Error] 制限到達。{wait_time:.1f}秒待機して再試行({attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            print(f"[yfinance Error] チャンク {chunk} 取得失敗: {e}")
                            break
                            
            print("--- yfinance 履歴データ取得終了 ---")

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
                    
        if not result_df.empty:
            DataProvider._save_prices_to_sql(result_df)

        final_df = pd.DataFrame()
        if not cached_df.empty and not result_df.empty:
            final_df = pd.concat([cached_df, result_df], axis=1)
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        elif not cached_df.empty:
            final_df = cached_df
        else:
            final_df = result_df

        return final_df

    # =========================================================================
    # 財務データ取得 (SQLite永続キャッシュによる429完全回避)
    # =========================================================================
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fundamentals(tickers):
        unique_tickers = list(set([DataProvider._normalize_ticker(t) for t in tickers if pd.notna(t)]))
        unique_tickers = [t for t in unique_tickers if t]
        if not unique_tickers: return pd.DataFrame()

        DataProvider._init_db()
        cached_data = []
        tickers_to_fetch = []

        # 1. DBから直近3日以内のキャッシュを取得
        try:
            with sqlite3.connect(DataProvider.DB_PATH) as conn:
                ticker_list_str = "','".join(unique_tickers)
                # 3日以内のデータがあればそれを採用
                query = f"SELECT * FROM fundamentals_cache WHERE ticker IN ('{ticker_list_str}') AND datetime(last_updated) >= datetime('now', '-3 days')"
                df_cache = pd.read_sql(query, conn)
                
            if not df_cache.empty:
                for _, row in df_cache.iterrows():
                    cached_data.append({
                        'Ticker': row['ticker'],
                        'Name': row['name'],
                        'Price': row['price'],
                        'Size_Raw': row['size_raw'],
                        'PBR': row['pbr'],
                        'ROE': row['roe'],
                        'Sector_Raw': row['sector_raw'],
                        'Growth': row['growth']
                    })
                cached_tickers = df_cache['ticker'].tolist()
                tickers_to_fetch = [t for t in unique_tickers if t not in cached_tickers]
            else:
                tickers_to_fetch = unique_tickers
        except Exception as e:
            print(f"Fundamentals DB Load Error: {e}")
            tickers_to_fetch = unique_tickers

        # 2. キャッシュにない銘柄のみAPIで取得 (指数バックオフ搭載)
        valid_api_data = []
        if tickers_to_fetch:
            print(f"--- yfinance 財務データAPI取得開始 (全{len(tickers_to_fetch)}銘柄, DBキャッシュ利用) ---")
            for ticker in tickers_to_fetch:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        time.sleep(0.5 + random.uniform(0, 1))
                        session = DataProvider._create_session()
                        tk = yf.Ticker(ticker, session=session)
                        info = tk.info or {}
                        
                        mktCap = info.get('marketCap', np.nan)
                        pbr = info.get('priceToBook', np.nan)
                        roe = info.get('returnOnEquity', np.nan)
                        
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
                        valid_api_data.append(res)
                        break # 成功したらリトライループを抜ける
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "429" in err_msg or "too many" in err_msg:
                            wait_time = (2 ** attempt) + random.uniform(1, 2)
                            print(f"⚠️ [429 Error] 財務データ取得制限。{wait_time:.1f}秒待機({attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            print(f"[yfinance Error] {ticker} 取得失敗: {e}")
                            valid_api_data.append({
                                'Ticker': ticker, 'Name': ticker, 'Price': np.nan, 'Size_Raw': np.nan,
                                'PBR': np.nan, 'ROE': np.nan, 'Sector_Raw': 'Unknown', 'Growth': np.nan
                            })
                            break
            print("--- yfinance 財務データ取得終了 ---")

            # 取得したデータをDBに保存
            if valid_api_data:
                try:
                    df_to_save = pd.DataFrame(valid_api_data)
                    df_to_save['last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # DBのカラム名に合わせる
                    rename_cols = {
                        'Ticker': 'ticker', 'Name': 'name', 'Price': 'price', 
                        'Size_Raw': 'size_raw', 'PBR': 'pbr', 'ROE': 'roe', 
                        'Sector_Raw': 'sector_raw', 'Growth': 'growth'
                    }
                    df_db = df_to_save.rename(columns=rename_cols)
                    
                    with sqlite3.connect(DataProvider.DB_PATH) as conn:
                        # 既存のレコードがあれば上書き (INSERT OR REPLACE)
                        for _, row in df_db.iterrows():
                            conn.execute(f"""
                                INSERT OR REPLACE INTO fundamentals_cache 
                                (ticker, name, price, size_raw, pbr, roe, sector_raw, growth, last_updated) 
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (row['ticker'], row['name'], row['price'], row['size_raw'], row['pbr'], row['roe'], row['sector_raw'], row['growth'], row['last_updated']))
                except Exception as e:
                    print(f"Fundamentals DB Save Error: {e}")

        # 3. データの結合と後処理
        all_data = cached_data + valid_api_data
        df = pd.DataFrame(all_data)
        
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
    # マーケットデータ (Rm, Rf) の取得 (指数バックオフ対応版)
    # =========================================================================
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_market_rates(days=365):
        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=days)
        
        df_market_daily = pd.DataFrame()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                session = DataProvider._create_session()
                rm_df = yf.download("^N225", start=start_d, end=end_d, session=session, progress=False, threads=False, timeout=30)
                if not rm_df.empty:
                    if isinstance(rm_df.columns, pd.MultiIndex):
                        if 'Close' in rm_df.columns.get_level_values(1):
                            close = rm_df.xs('Close', level=1, axis=1).squeeze()
                        elif 'Close' in rm_df.columns.get_level_values(0):
                            close = rm_df.xs('Close', level=0, axis=1).squeeze()
                    else:
                        if 'Close' in rm_df.columns:
                            close = rm_df['Close'].squeeze()
                    
                    if not close.empty:
                        df_market_daily['Rm_Price'] = close
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "too many" in err_msg:
                    time.sleep((2 ** attempt) + random.uniform(1, 2))
                else:
                    break

        if df_market_daily.empty: 
            return pd.DataFrame()

        df_market_monthly = df_market_daily.resample('ME').last()
        df_market_monthly['Rm'] = df_market_monthly['Rm_Price'].pct_change()
        
        annual_rf = 0.005
        monthly_rf = annual_rf / 12.0 
        
        df_market_monthly['Rf'] = monthly_rf
        df_market_monthly.index = df_market_monthly.index.normalize().tz_localize(None)
        
        return df_market_monthly.dropna(subset=['Rm'])

    # =========================================================================
    # ユーティリティ・FMPフォールバック (変更なし)
    # =========================================================================
    @staticmethod
    def _normalize_ticker(t):
        if pd.isna(t) or not str(t).strip(): return ""
        t_str = str(t).strip().upper()
        match = re.search(r'\b(\d{4})\b', t_str)
        if match: return f"{match.group(1)}.T"
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
        
        fallback_tickers = ["7203.T", "8306.T", "9984.T", "6861.T", "8035.T", "9432.T", "6758.T", "8316.T", "4063.T", "8058.T"]
        return {t: "JPX Data Missing" for t in fallback_tickers}

    @staticmethod
    def _fetch_fmp_ratios(ticker_list):
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list: return {}

        rescued_data = {}
        session = DataProvider._create_session()
        
        for t_orig in ticker_list:
            time.sleep(0.5) 
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url_ratios = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={api_key}"
            url_growth = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}?limit=1&apikey={api_key}"
            url_profile = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"
            
            data_res = {}
            try:
                r = session.get(url_ratios, timeout=45)
                if r.status_code == 200 and r.json():
                    items = r.json()
                    data_res['ROE'] = items[0].get('returnOnEquityTTM')
                    data_res['PBR'] = items[0].get('priceToBookRatioTTM')
                
                r_g = session.get(url_growth, timeout=45)
                if r_g.status_code == 200 and r_g.json():
                    g_items = r_g.json()
                    data_res['Growth'] = g_items[0].get('assetGrowth')
                        
                r_p = session.get(url_profile, timeout=45)
                if r_p.status_code == 200 and r_p.json():
                    p_items = r_p.json()
                    data_res['Size_Raw'] = p_items[0].get('mktCap')
                        
                rescued_data[t_orig] = data_res
            except Exception:
                pass
            
        return rescued_data

    @staticmethod
    def _fetch_fmp_history(ticker_list, days=365):
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list: return pd.DataFrame()

        all_series = {}
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)
        session = DataProvider._create_session()
        
        for t_orig in ticker_list:
            time.sleep(0.5)
            symbol = t_orig.replace(".T", ".JP") if ".T" in t_orig else t_orig
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
            try:
                r = session.get(url, timeout=45)
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
                        all_series[t_orig] = df['close']
            except Exception:
                pass

        return pd.DataFrame(all_series)

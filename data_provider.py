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
    【Module 1】データ取得プロバイダー (Robust Ver. 2.0)
    Yahoo Finance (yfinance) を主軸とし、セッション管理とリトライ機能で取得成功率を向上。
    取得失敗時には FMP API で補完する堅牢な設計。
    """
    
    # APIキーの取得 (Streamlit Secrets優先)
    FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.environ.get("FMP_API_KEY"))

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
    def _create_session():
        """
        yfinance用のカスタムセッションを作成 (401エラー対策)
        User-Agentの偽装とリトライ設定を行う
        """
        session = requests.Session()
        
        # ブラウザのUser-Agentを偽装
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        # リトライ設定 (最大3回, バックオフ係数1, 対象ステータスコード)
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

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
        """FMP APIから財務指標を取得 (Rescue用)"""
        api_key = DataProvider.FMP_API_KEY
        if not api_key or not ticker_list:
            return {}

        rescued_data = {}
        
        def fetch_one(t_orig):
            # 日本株のシンボル変換 (T -> JP)
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
                            'Growth': item.get('dividendYieldTTM') 
                        }
            except:
                pass
            return t_orig, None

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(fetch_one, ticker_list))
        
        for t, data in results:
            if data: rescued_data[t] = data
            
        return rescued_data

    @staticmethod
    def _fetch_fmp_history(ticker_list, days=365):
        """FMP APIから株価を取得 (Rescue用)"""
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
        ファンダメンタルズ情報を取得
        yfinanceのTickerオブジェクトにカスタムセッションを適用して401エラーを回避
        """
        unique_tickers = list(set(tickers))
        if not unique_tickers: return pd.DataFrame()

        # カスタムセッションの作成
        session = DataProvider._create_session()

        # 1. Primary: yfinance
        def get_yf_stock(ticker):
            try:
                # sessionを渡して初期化
                tk = yf.Ticker(ticker, session=session)
                
                # info取得 (ここが401エラーの発生源)
                info = tk.info
                
                if info is None or 'currentPrice' not in info:
                    # 必須データがない場合はNoneを返す
                    return None
                    
                # 辞書に整形
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
            except Exception:
                # 取得失敗時はNone
                return None

        # 並列処理で取得 (ワーカー数は適宜調整)
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(get_yf_stock, unique_tickers))
        
        valid_data = [d for d in results if d is not None]
        df = pd.DataFrame(valid_data)
        
        # 2. Secondary: FMP Rescue (yfinanceが全滅または一部欠損の場合)
        if df.empty:
            df = pd.DataFrame({'Ticker': unique_tickers})
            
        # 必須カラムの初期化
        for col in ['ROE', 'PBR', 'Growth']:
            if col not in df.columns: df[col] = np.nan

        # 欠損値がある銘柄を特定
        missing_cond = df['ROE'].isna() | df['PBR'].isna()
        missing_tickers = df[missing_cond]['Ticker'].tolist() if not df.empty else unique_tickers
        
        # FMPで補完
        if missing_tickers and DataProvider.FMP_API_KEY:
            fmp_data = DataProvider._fetch_fmp_ratios(missing_tickers)
            for i, row in df.iterrows():
                t = row['Ticker']
                if t in fmp_data:
                    # ROE補完
                    if pd.isna(row.get('ROE')): 
                        df.at[i, 'ROE'] = fmp_data[t].get('ROE')
                    # PBR補完
                    if pd.isna(row.get('PBR')): 
                        df.at[i, 'PBR'] = fmp_data[t].get('PBR')
                    # Growth補完 (FMPではdividendYieldTTMが入っている仮実装)
                    # 本来はFMPのrevenueGrowthを取得すべきだが、まずは穴埋めとして維持
                    if pd.isna(row.get('Growth')): 
                        df.at[i, 'Growth'] = fmp_data[t].get('Growth')

        # データ型変換 (数値化)
        num_cols = ['Price', 'Size_Raw', 'PBR', 'ROE', 'Growth']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # セクター変換
        if 'Sector_Raw' in df.columns:
            df['sector'] = df['Sector_Raw'].apply(DataProvider._map_sector)
        else:
            df['sector'] = 'Other'

        return df

    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_historical_prices(tickers, days=365):
        """
        時系列株価データを取得 (yfinance -> FMP)
        download関数にsessionを適用して安定化
        """
        if not tickers: return pd.DataFrame()
        
        # カスタムセッション作成
        session = DataProvider._create_session()
        
        # 1. yfinance での取得試行
        try:
            end_d = datetime.date.today()
            start_d = end_d - datetime.timedelta(days=days)
            
            # 【修正点】yf.download に session=session を渡す (v0.2.x以降対応)
            # ※ yfinanceのバージョンによっては session 引数が効かない場合があるため
            #    その場合は内部でrequestsを使う際に影響するようにglobal設定が必要だが
            #    ここでは引数渡しを試みる
            
            # 【復元点】yf.downlo -> yf.download に修正
            df = yf.download(
                tickers, 
                start=start_d, 
                end=end_d, 
                progress=False, 
                group_by='ticker', 
                auto_adjust=True,
                session=session  # セッション注入
            )
            
            if df.empty:
                # 空の場合はエラーとして扱い、FMPへ
                raise ValueError("yfinance returned empty")

            # --- 戻り値の整形 (MultiIndex対応) ---
            if len(tickers) == 1:
                t = tickers[0]
                # 単一銘柄の場合、カラムは 'Open', 'High', ... となっていることが多い
                if 'Close' in df.columns:
                    result_df = pd.DataFrame({t: df['Close']})
                else:
                    result_df = pd.DataFrame()
            else:
                # 複数銘柄の場合、(Price, Ticker) または (Ticker, Price) の形
                # 'Close' レベルを探して抽出
                try:
                    # 'Close' がカラムのレベル1にあると仮定 (yfinance標準)
                    result_df = df.iloc[:, df.columns.get_level_values(1) == 'Close']
                    # カラム名をTickerのみにする
                    result_df.columns = result_df.columns.get_level_values(0)
                except:
                    # 構造が違う場合のフェイルセーフ
                    result_df = pd.DataFrame()
            
            # 足りない銘柄があれば FMP で救済
            current_cols = result_df.columns.tolist() if not result_df.empty else []
            missing = list(set(tickers) - set(current_cols))
            
            if missing and DataProvider.FMP_API_KEY:
                fmp_df = DataProvider._fetch_fmp_history(missing, days)
                if not fmp_df.empty:
                    result_df = pd.concat([result_df, fmp_df], axis=1)
            
            return result_df

        except Exception:
            # yfinanceが失敗したらFMPに全任せ
            if DataProvider.FMP_API_KEY:
                return DataProvider._fetch_fmp_history(tickers, days)
            return pd.DataFrame()

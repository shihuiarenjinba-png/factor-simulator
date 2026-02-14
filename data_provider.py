import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from scipy.stats import linregress
from concurrent.futures import ThreadPoolExecutor  # 【追加】並列処理用

# ---------------------------------------------------------
# 0. 基本設定 (Config)
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Market Factor Lab Pro (Modular)")

# 分析対象ユニバース（既存のロジックを維持するため残しています）
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
# 【Modified】Module 1: Data Provider (並列処理・高速化・堅牢版)
# ---------------------------------------------------------
class DataProvider:
    """
    データ取得とキャッシュ管理を担当する独立モジュール (v2.0 High-Speed)
    """

    @staticmethod
    def get_universe_tickers(mode="Nikkei 225"):
        """
        ベンチマークに応じた銘柄リストを動的に返す（将来的な拡張用）
        """
        if "Nikkei" in mode:
            # 必要に応じて全銘柄リストなどをここに記述
            return NIKKEI_225_SAMPLE
        elif "TOPIX" in mode:
            return TOPIX_100_SAMPLE
        return []

    @staticmethod
    @st.cache_data(ttl=3600)  # 1時間はキャッシュを保持
    def fetch_fundamentals(tickers):
        """
        ファンダメンタルズ情報（時価総額、ROE、PBRなど）を並列処理で高速取得
        """
        # 重複削除
        unique_tickers = list(set(tickers))
        
        # 内部関数: 1銘柄ごとの取得ロジック
        def get_single_stock(ticker):
            try:
                tk = yf.Ticker(ticker)
                # fast_infoではなく、より詳細な info を使用
                info = tk.info
                
                # 【バリデーション】 必須データ欠損チェック
                # 価格情報が取れないものは、上場廃止やエラーの可能性が高いため除外
                if info is None or 'currentPrice' not in info or info['currentPrice'] is None:
                    return None
                
                return {
                    'Ticker': ticker,
                    'Name': info.get('shortName', ticker),
                    'Price': info.get('currentPrice', np.nan),
                    'Size_Raw': info.get('marketCap', np.nan),
                    'PBR': info.get('priceToBook', np.nan),     # Value用
                    'ROE': info.get('returnOnEquity', np.nan),  # Quality用
                    'Growth': info.get('revenueGrowth', np.nan) # Investment用
                }
            except Exception:
                # 取得失敗時はNoneを返し、後でリストから除外する
                return None

        # 並列処理の実行 (最大20スレッドで同時取得)
        results = []
        # UX向上のためspinnerは呼び出し元で制御推奨だが、ここではバックグラウンド処理
        with ThreadPoolExecutor(max_workers=20) as executor:
            # mapは入力順序を保持する
            fetched_data = list(executor.map(get_single_stock, unique_tickers))
        
        # None (取得失敗) をリストから除外
        results = [d for d in fetched_data if d is not None]

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        
        # 数値型の強制変換（念のためエラー回避）
        num_cols = ['Price', 'Size_Raw', 'PBR', 'ROE', 'Growth']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df

    @staticmethod
    @st.cache_data(ttl=86400) # ヒストリカルデータは1日保持
    def fetch_historical_prices(tickers, days=365):
        """
        ヒストリカルデータ（株価推移）を一括取得 (v2.0 Robust)
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        if not tickers:
            return pd.DataFrame()

        try:
            # yf.downloadは内部でマルチスレッド化されているため高速
            # group_by='ticker' で構造を固定化
            df = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
            
            # DataFrame構造の正規化
            if len(tickers) == 1:
                # 1銘柄の場合: Index=Date, Columns=[Open, High, Low, Close...]
                ticker = tickers[0]
                # 'Close'列が存在するか確認
                if 'Close' in df.columns:
                    return pd.DataFrame({ticker: df['Close']})
                else:
                    return pd.DataFrame() # データなし
            else:
                # 複数銘柄の場合: MultiIndex (Ticker, OHLC) -> (Ticker, Close) を抽出
                try:
                    # xsを使って 'Close' レベルを抽出 (axis=1)
                    df_close = df.xs('Close', axis=1, level=1, drop_level=True)
                    return df_close
                except KeyError:
                    # まれに構造が違う場合やデータ欠損への対策
                    return pd.DataFrame()
                    
        except Exception as e:
            st.error(f"Historical Data Error: {e}")
            return pd.DataFrame()

# ---------------------------------------------------------
# Module 2 & 3: Logic & Engine (既存ロジックの移植)
# ---------------------------------------------------------

def compute_derived_metrics(df_fund, df_hist, benchmark_ticker):
    """
    生データから Beta, Momentum, Value(逆数), Size(対数) を計算
    """
    df = df_fund.copy()
    
    # 1. Value (PBRの逆数)
    df['Value_Raw'] = df['PBR'].apply(lambda x: 1/x if (pd.notnull(x) and x > 0) else np.nan)
    
    # 2. Size (対数正規化)
    df['Size_Log'] = np.log(pd.to_numeric(df['Size_Raw'], errors='coerce').replace(0, np.nan))
    
    # 3. Momentum & Beta
    moms = {}
    betas = {}
    
    if not df_hist.empty:
        # リターン計算
        rets = df_hist.pct_change().dropna()
        
        if benchmark_ticker in rets.columns:
            bench_ret = rets[benchmark_ticker]
            bench_var = bench_ret.var()
            
            # Beta Loop
            for t in df['Ticker']:
                if t in rets.columns:
                    # Beta
                    cov = rets[t].cov(bench_ret)
                    betas[t] = cov / bench_var if bench_var != 0 else 1.0
                    
                    # Momentum (過去1年のリターン累積)
                    # 簡易的に (最終価格 / 最初価格) - 1
                    try:
                        p_end = df_hist[t].iloc[-1]
                        p_start = df_hist[t].iloc[0]
                        moms[t] = (p_end / p_start) - 1
                    except:
                        moms[t] = np.nan
                else:
                    betas[t] = 1.0
                    moms[t] = np.nan
        else:
            # ベンチマークデータがない場合
            for t in df['Ticker']:
                betas[t] = 1.0
                moms[t] = np.nan
                
    df['Beta_Raw'] = df['Ticker'].map(betas)
    df['Momentum_Raw'] = df['Ticker'].map(moms)
    
    # Rename columns to match logic
    df.rename(columns={'ROE': 'Quality_Raw', 'Growth': 'Investment_Raw'}, inplace=True)
    
    return df

def calculate_market_stats(universe_df):
    """
    ユニバース全体の平均(mu)と標準偏差(sigma)を計算 + 直交化パラメータ
    """
    stats = {}
    
    # 直交化 (Quality vs Investment)
    mask = universe_df['Quality_Raw'].notna() & universe_df['Investment_Raw'].notna()
    if mask.sum() > 10:
        slope, intercept, _, _, _ = linregress(universe_df.loc[mask, 'Investment_Raw'], universe_df.loc[mask, 'Quality_Raw'])
    else:
        slope, intercept = 0, 0
    
    stats['ortho_slope'] = slope
    stats['ortho_intercept'] = intercept
    
    # 各ファクターの統計量
    factors = {
        'Beta': 'Beta_Raw',
        'Size': 'Size_Log',
        'Value': 'Value_Raw',
        'Momentum': 'Momentum_Raw',
        'Quality': 'Quality_Raw', # ※計算時に直交化するが、元データの統計も一応保持
        'Investment': 'Investment_Raw'
    }
    
    # Qualityの直交化済みデータの一時作成（統計量計算用）
    temp_q = universe_df.apply(lambda x: x['Quality_Raw'] - (slope * x['Investment_Raw'] + intercept) 
                               if (pd.notnull(x['Quality_Raw']) and pd.notnull(x['Investment_Raw'])) else np.nan, axis=1)
    
    for f, col in factors.items():
        if f == 'Quality':
            series = temp_q.dropna()
        else:
            series = universe_df[col].dropna()
            
        if not series.empty:
            stats[f] = {'mean': series.mean(), 'std': series.std(), 'col': col}
        else:
            stats[f] = {'mean': 0, 'std': 1, 'col': col}
            
    return stats

def apply_scoring(target_df, stats):
    """
    Zスコア計算 & SMB反転 & フォーマット
    """
    df = target_df.copy()
    
    # 直交化適用
    slope = stats['ortho_slope']
    intercept = stats['ortho_intercept']
    df['Quality_Orthogonal'] = df.apply(lambda x: x['Quality_Raw'] - (slope * x['Investment_Raw'] + intercept) 
                                        if (pd.notnull(x['Quality_Raw']) and pd.notnull(x['Investment_Raw'])) else x['Quality_Raw'], axis=1)

    factors = ['Beta', 'Value', 'Size', 'Momentum', 'Quality', 'Investment']
    
    for f in factors:
        if f not in stats: continue
        
        # 参照カラムの決定
        if f == 'Quality': col_name = 'Quality_Orthogonal'
        else: col_name = stats[f]['col']
            
        mu = stats[f]['mean']
        sigma = stats[f]['std']
        z_col = f"{f}_Z"
        
        # Zスコア計算
        def calc_z(val):
            if pd.isna(val): return 0.0
            if sigma == 0: return 0.0
            z = (val - mu) / sigma
            
            # 【重要】SMB反転: 大型株(Size大)ほどマイナススコアにする
            if f == 'Size':
                z = -z
            return z
            
        df[z_col] = df[col_name].apply(calc_z)
        
        # 表示用フォーマット
        def fmt(row):
            raw = row.get(col_name)
            # 生データの表示用には元のRaw値を使う場合もある
            if f == 'Size': raw_disp = row.get('Size_Raw')
            elif f == 'Value': raw_disp = 1/raw if (raw and raw!=0) else np.nan # PBRに戻して表示
            else: raw_disp = raw
            
            z = row.get(z_col)
            
            if pd.isna(raw_disp): return "-"
            
            if f == 'Size':
                if raw_disp >= 1e12: txt = f"{raw_disp/1e12:.2f}兆"
                elif raw_disp >= 1e8: txt = f"{raw_disp/1e8:.0f}億"
                else: txt = str(raw_disp)
            elif f in ['Momentum', 'Quality', 'Investment']:
                txt = f"{raw_disp*100:.1f}%"
            elif f == 'Value':
                txt = f"{raw_disp:.2f} (PBR)"
            else:
                txt = f"{raw_disp:.2f}"
                
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
st.sidebar.write("CSVアップロード または テキスト入力")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
default_tickers = "7203.T, 9984.T, 6758.T, 8035.T, 6861.T, 9983.T, 4502.T, 6367.T"
input_tickers = st.sidebar.text_area("Input Tickers", default_tickers, height=100)

if st.sidebar.button("Run Full Analysis", type="primary"):
    
    # A. 入力解析
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
    
    # B. Module 1: データ取得 (High-Speed & Parallel)
    with st.spinner("Fetching Market Data (Module 1)..."):
        # ユニバース + ユーザー銘柄 + ベンチマークETF
        all_tickers = list(set(selected_universe + user_tickers + [bench_ticker]))
        
        # 1. ファンダメンタルズ取得 (並列処理)
        df_fund = DataProvider.fetch_fundamentals(all_tickers)
        
        # 2. ヒストリカル取得
        df_hist = DataProvider.fetch_historical_prices(all_tickers)
        
        if df_fund.empty:
            st.error("Data Fetch Failed.")
            st.stop()

    # C. Module 2: 計算ロジック
    with st.spinner("Calculating Factors (Module 2)..."):
        # 指標計算 (Beta, LogSize, etc)
        df_full = compute_derived_metrics(df_fund, df_hist, bench_ticker)
        
        # ユニバース統計量の算出
        uni_df = df_full[df_full['Ticker'].isin(selected_universe)].copy()
        stats = calculate_market_stats(uni_df)
        
        # ユーザ銘柄のスコアリング
        user_df_calc = df_full[df_full['Ticker'].isin(user_tickers)].copy()
        user_scored = apply_scoring(user_df_calc, stats)
        
        # マージしてウェイト復元
        user_scored = pd.merge(user_scored, user_df_base, on='Ticker', how='left')
        
        # ウェイト自動補完
        current_w = user_scored['Weight'].sum()
        nans = user_scored['Weight'].isna()
        if nans.any():
            rem = max(0, 100 - current_w)
            user_scored.loc[nans, 'Weight'] = rem / nans.sum()
        
    # D. 結果表示 (UI)
    st.subheader("🛠 Portfolio Composition")
    edited = st.data_editor(user_scored[['Ticker', 'Name', 'Weight']], 
                            column_config={"Weight": st.column_config.NumberColumn(format="%.2f%%")},
                            use_container_width=True)
    
    # ヒートマップ
    st.subheader("🧬 Factor Heatmap")
    disp_cols = [c for c in user_scored.columns if "_Display" in c]
    
    def color_sigma(val):
        if "(" not in str(val): return ""
        try:
            sigma = float(val.split("(")[1].split("σ")[0])
            if sigma >= 1.0: return "background-color: #d1e7dd; color: #0f5132" # Green
            if sigma <= -1.0: return "background-color: #f8d7da; color: #842029" # Red
        except: pass
        return ""

    st.dataframe(user_scored[["Ticker", "Name"] + disp_cols].style.applymap(color_sigma), use_container_width=True)

    # グラフ表示
    st.divider()
    w = edited['Weight'] / 100.0
    z_cols = [f"{f}_Z" for f in ['Beta', 'Size', 'Value', 'Momentum', 'Quality', 'Investment']]
    
    # ポートフォリオの加重平均Zスコア
    port_exp = {}
    for zc in z_cols:
        port_exp[zc.replace("_Z", "")] = (user_scored[zc] * w).sum()
        
    st.bar_chart(pd.Series(port_exp))
    
    st.success("Analysis Complete using Modular Data Provider.")

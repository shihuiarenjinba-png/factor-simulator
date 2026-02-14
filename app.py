import streamlit as st
import pandas as pd
import numpy as np
import datetime

# 作成した3つのモジュールを読み込み
# ※同じフォルダにファイルがあることを前提としています
try:
    from data_provider import DataProvider
    from quant_engine import QuantEngine
    from universe_manager import UniverseManager
except ImportError as e:
    st.error(f"【重要】モジュールが見つかりません: {e}")
    st.info("app.py と同じ場所に data_provider.py, quant_engine.py, universe_manager.py があるか確認してください。")
    st.stop()

# ---------------------------------------------------------
# 0. アプリ設定 & 定数定義
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Market Factor Lab (Modular Ver.)")

# ベンチマーク定義（日経225採用銘柄の一部サンプル + 代表的なETF）
# 本番では全銘柄リストを入れるとより精度が上がります
NIKKEI_225_SAMPLE = [
    "7203.T", "6758.T", "8035.T", "9984.T", "9983.T", "6098.T", "4063.T", "6367.T", "9432.T", "4502.T",
    "4503.T", "6501.T", "7267.T", "8058.T", "8001.T", "6954.T", "6981.T", "9020.T", "9022.T", "7741.T",
    "5108.T", "4452.T", "6902.T", "7974.T", "8031.T", "4519.T", "4568.T", "6273.T", "4543.T", "6702.T",
    "6503.T", "4901.T", "4911.T", "2502.T", "2802.T", "3382.T", "8306.T", "8316.T", "8411.T", "8766.T",
    "8591.T", "8801.T", "8802.T", "9021.T", "9101.T", "9433.T", "9434.T", "9501.T", "9502.T"
]

# ---------------------------------------------------------
# 1. ヘルパー関数 (Beta計算など、アプリ固有の処理)
# ---------------------------------------------------------
def calculate_beta_momentum(tickers, benchmark_ticker="1321.T"):
    """
    アプリ側で実行する時系列計算（BetaとMomentum）。
    DataProviderからヒストリカルデータを取得して計算する。
    """
    # 全銘柄 + ベンチマークのデータを取得
    needed_tickers = list(set(tickers + [benchmark_ticker]))
    df_hist = DataProvider.fetch_historical_prices(needed_tickers, days=365)
    
    betas = {}
    momenta = {}
    
    if df_hist.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # リターン計算
    rets = df_hist.pct_change().dropna()
    
    if benchmark_ticker not in rets.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    bench_ret = rets[benchmark_ticker]
    bench_var = bench_ret.var()

    for t in tickers:
        if t in rets.columns:
            # Beta: 共分散 / 分散
            try:
                cov = rets[t].cov(bench_ret)
                if bench_var > 0:
                    betas[t] = cov / bench_var
                else:
                    betas[t] = 1.0
            except:
                betas[t] = 1.0
            
            # Momentum: 過去1年のリターン (簡易版)
            try:
                p_start = df_hist[t].iloc[0]
                p_end = df_hist[t].iloc[-1]
                if p_start > 0:
                    momenta[t] = (p_end / p_start) - 1
                else:
                    momenta[t] = 0.0
            except:
                momenta[t] = 0.0
        else:
            betas[t] = np.nan
            momenta[t] = np.nan
            
    return pd.Series(betas), pd.Series(momenta)

# ---------------------------------------------------------
# 2. UI レイアウト & 入力
# ---------------------------------------------------------
st.sidebar.header("📊 Settings")

# ベンチマーク選択
bench_mode = st.sidebar.selectbox("Benchmark Universe", ["Nikkei 225 (Sample)", "TOPIX 100 (Sample)"])
universe_tickers = NIKKEI_225_SAMPLE # デモ用簡易切り替え
benchmark_etf = "1321.T" # 日経225連動ETF

# ポートフォリオ入力
st.sidebar.subheader("My Portfolio")
default_input = "7203.T, 9984.T, 6758.T, 8035.T, 6861.T"
input_text = st.sidebar.text_area("Tickers (comma separated)", default_input, height=100)
uploaded_file = st.sidebar.file_uploader("Or upload CSV", type=['csv'])

# 実行ボタン
run_btn = st.sidebar.button("Run Analysis", type="primary")

# ---------------------------------------------------------
# 3. メイン処理フロー
# ---------------------------------------------------------
if run_btn:
    st.title("🛡️ Modular Portfolio Analysis Result")
    
    # [Step 1] ユーザー入力の解析
    user_tickers = []
    user_weights = {}
    
    if uploaded_file:
        try:
            df_in = pd.read_csv(uploaded_file)
            # Tickerカラムを探す
            ticker_col = next((c for c in df_in.columns if 'ticker' in c.lower()), None)
            if ticker_col:
                user_tickers = df_in[ticker_col].astype(str).tolist()
                # Weightカラムがあれば取得
                weight_col = next((c for c in df_in.columns if 'weight' in c.lower()), None)
                if weight_col:
                    for idx, row in df_in.iterrows():
                        user_weights[row[ticker_col]] = row[weight_col]
        except Exception as e:
            st.error(f"CSV読込エラー: {e}")
            st.stop()
    else:
        raw_list = [x.strip() for x in input_text.split(',') if x.strip()]
        user_tickers = raw_list

    if not user_tickers:
        st.warning("銘柄を入力してください。")
        st.stop()

    # -----------------------------------------------------
    # [Step 2] 市場データの基準作成 (The "Ruler")
    # -----------------------------------------------------
    with st.status("🏗️ Building Market Universe...", expanded=True) as status:
        st.write("Fetching Benchmark Data (Module 1)...")
        # 1. データ取得
        df_bench_fund = DataProvider.fetch_fundamentals(universe_tickers)
        
        # 2. ベータ計算用のヒストリカルデータ
        # (市場平均のBetaも計算に含めるため取得)
        s_beta_bench, s_mom_bench = calculate_beta_momentum(universe_tickers, benchmark_etf)
        
        # DataFrameに結合
        df_bench_fund['Beta_Raw'] = df_bench_fund['Ticker'].map(s_beta_bench)
        df_bench_fund['Momentum_Raw'] = df_bench_fund['Ticker'].map(s_mom_bench)
        
        st.write("Calculating Market Statistics (Module 3)...")
        # 3. 統計量(Stats)の生成
        # ここで「外れ値処理」と「直交化」が行われ、きれいな平均・標準偏差が返ってくる
        market_stats, df_bench_processed = UniverseManager.generate_market_stats(df_bench_fund)
        
        status.update(label="Market Universe Ready!", state="complete", expanded=False)

    # -----------------------------------------------------
    # [Step 3] ユーザーポートフォリオの評価 (The "Measurement")
    # -----------------------------------------------------
    with st.spinner("🔬 Analyzing Your Portfolio..."):
        # 1. ユーザー銘柄のデータ取得
        # (ベンチマークと重複している銘柄はキャッシュから即座に返る)
        df_user_fund = DataProvider.fetch_fundamentals(user_tickers)
        
        # 2. Beta / Momentum 計算
        s_beta_user, s_mom_user = calculate_beta_momentum(user_tickers, benchmark_etf)
        df_user_fund['Beta_Raw'] = df_user_fund['Ticker'].map(s_beta_user)
        df_user_fund['Momentum_Raw'] = df_user_fund['Ticker'].map(s_mom_user)
        
        # 3. 生データの加工 (Log化など)
        # Module 2 のロジックを使って、市場データと同じ基準で加工する
        df_user_proc = QuantEngine.process_raw_factors(df_user_fund)
        
        # 4. 直交化の適用 (市場のパラメータを使って、ユーザー銘柄を補正)
        slope = market_stats['ortho_slope']
        intercept = market_stats['ortho_intercept']
        
        def apply_ortho(row):
            q = row.get('Quality_Metric', np.nan)
            i = row.get('Investment_Metric', np.nan)
            if pd.isna(q): return np.nan
            if pd.isna(i): return q
            return q - (slope * i + intercept)
            
        df_user_proc['Quality_Orthogonal'] = df_user_proc.apply(apply_ortho, axis=1)

        # 5. Zスコア計算 & SMB反転
        # ここで Module 2 が「サイズが大きいほどマイナス」にする処理を実行
        df_scored = QuantEngine.compute_z_scores(df_user_proc, market_stats)
        
        # ウェイト情報の結合
        if user_weights:
            df_scored['Weight'] = df_scored['Ticker'].map(user_weights)
        else:
            # ウェイト指定がない場合は等ウェイト
            df_scored['Weight'] = 100.0 / len(df_scored)

    # -----------------------------------------------------
    # [Step 4] 結果表示 (Visualization)
    # -----------------------------------------------------
    
    # 1. データテーブル（ヒートマップ）
    st.subheader("🧬 Factor Heatmap (Z-Score)")
    
    # 表示用の列を選択
    display_cols = ['Ticker', 'Name', 'Weight'] + [c for c in df_scored.columns if 'Display' in c or '_Z' in c]
    # シンプルにするため、Zスコアと表示用Rawデータに絞る
    final_view = df_scored.copy()
    
    # スタイリング関数
    def style_z_score(v):
        try:
            val = float(v)
            if val > 1.0: return 'background-color: #d4edda; color: #155724' # Green
            if val < -1.0: return 'background-color: #f8d7da; color: #721c24' # Red
            return ''
        except:
            return ''

    # Zスコア列だけを抽出して表示
    z_cols = [c for c in final_view.columns if c.endswith('_Z')]
    st.dataframe(
        final_view[['Ticker', 'Name', 'Weight'] + z_cols].style.applymap(style_z_score, subset=z_cols),
        use_container_width=True
    )

    # 2. ポートフォリオ全体のエクスポージャー
    st.subheader("📊 Portfolio Total Exposure")
    
    # 加重平均Zスコアの計算
    total_weight = final_view['Weight'].sum()
    if total_weight == 0: total_weight = 1.0
    
    exposure = {}
    for col in z_cols:
        factor_name = col.replace('_Z', '')
        # (Zスコア * ウェイト) の総和 / 総ウェイト
        w_avg = (final_view[col] * final_view['Weight']).sum() / total_weight
        exposure[factor_name] = w_avg
        
    exp_df = pd.Series(exposure, name="Z-Score")
    st.bar_chart(exp_df)
    
    st.success("Analysis Completed Successfully.")
    
    # デバッグ情報（確認用）
    with st.expander("Show Market Statistics (Debug)"):
        st.write("Calculated Market Parameters (used for Z-score):")
        st.json(market_stats)

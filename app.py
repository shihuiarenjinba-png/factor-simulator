import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re  # 正規表現モジュール
import unicodedata  # サニタイズ用
import time
import datetime  # 日付処理用のライブラリ

# カスタムモジュールの読み込み
try:
    from data_provider import DataProvider
    from quant_engine import QuantEngine
    from universe_manager import MarketMonitor, UniverseManager
    from visualizer import Visualizer  
except ModuleNotFoundError as e:
    st.error(f"起動エラー: モジュールが見つかりません ({e})")
    st.stop()

# ---------------------------------------------------------
# 0. ページ設定 & デザイン定義
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Market Factor Lab (Pro)")

st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #333; }
    .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
    .summary-box { background-color: #e8f4f8; padding: 15px; border-radius: 8px; border-left: 5px solid #1f77b4; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

st.title("Factor Simulator V18.1 - 5-Factor Regression Analysis")
st.markdown("Kenneth R. French の日本市場5ファクターに基づく厳密な時系列回帰分析と、銘柄ごとの加重平均寄与度を可視化します。")

# ---------------------------------------------------------
# キャッシュラッパー & 詳細進捗表示
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_market_data(tickers):
    """
    【重要修正】yfinance由来のデータ（株価・財務）のみをここで取得。
    ファクターデータ(Fama-French)はエラー切り分けのためこの関数の外で取得する。
    """
    df_fund = DataProvider.fetch_fundamentals(tickers)
    df_hist = DataProvider.fetch_historical_prices(tickers, days=365*2)
    df_market = DataProvider.fetch_market_rates(days=365*2)
    
    return df_fund, df_hist, df_market

# ---------------------------------------------------------
# 1. サイドバー: 入力
# ---------------------------------------------------------
st.sidebar.header("📊 分析設定")

port_input = st.sidebar.text_area(
    "ポートフォリオ銘柄 (カンマまたは改行区切り)",
    value="7203, 8306, 9984, 6758, 8035",
    help="証券コード（4桁）を入力してください。"
)

weight_input = st.sidebar.text_area(
    "ウェイト入力 (カンマ区切り・空欄で自動配分)",
    value="",
    help="銘柄と同じ順番で数値を入力してください。空欄時は等金額または時価総額加重になります。"
)

st.sidebar.markdown("---")
bench_mode = st.sidebar.radio("ベンチマークユニバース", ("TOPIX Core 30", "Nikkei 225"))
uploaded_file = st.sidebar.file_uploader("公式構成銘柄ファイル (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

run_button = st.sidebar.button("回帰分析を実行", type="primary")

# ---------------------------------------------------------
# 2. メイン処理ロジック
# ---------------------------------------------------------
if run_button:
    progress_bar = st.progress(0, text="[1/5] 初期化中...")
    
    # --- Step 0: 入力サニタイズ ---
    sanitized_port = unicodedata.normalize('NFKC', port_input)
    raw_codes = re.findall(r'\b\d{4}\b', sanitized_port)
    port_tickers = [f"{code}.T" for code in list(dict.fromkeys(raw_codes))]
    
    if not port_tickers:
        st.error("有効な銘柄コードが見つかりませんでした。")
        st.stop()

    clean_w_str = unicodedata.normalize('NFKC', weight_input).replace(" ", "").replace("\n", "")
    weight_list = []
    if clean_w_str:
        sanitized_w = re.sub(r'[^0-9.,]', '', clean_w_str)
        if sanitized_w:
            try: weight_list = [float(w) for w in sanitized_w.split(',') if w]
            except: pass 

    # --- Step A: ユニバース準備 ---
    progress_bar.progress(10, text="[2/5] ユニバース（比較基準）の準備中...")
    custom_list = MarketMonitor.load_tickers_from_file(uploaded_file) if uploaded_file else None
    bench_tickers = MarketMonitor.get_latest_tickers(bench_mode, custom_list)
    all_target_tickers = list(set(bench_tickers + port_tickers))

    # --- Step B: データ取得 (Fama-French含む) ---
    progress_bar.progress(20, text=f"[3/5] 市場データ(株価・財務)の取得中... (最大1分程度)")
    
    # 1. yfinanceからのデータ取得
    df_all_fund, df_hist, df_market = get_cached_market_data(all_target_tickers)
    
    if df_all_fund.empty or df_hist.empty:
        # yfinanceが原因のエラーであることがここで確定する
        st.error("❌ 株価・財務データの取得に失敗しました。Yahoo Financeの制限（429エラー）の可能性があります。10分ほど時間を置いて再試行してください。")
        st.stop()

    progress_bar.progress(40, text=f"[3/5] マクロファクターデータの同期中...")
    
    # 2. 【重要修正】ケネス・フレンチデータの独立取得
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=365*2)).strftime('%Y-%m-%d')
    df_ff5 = DataProvider.fetch_ken_french_5factors(start_date=start_date, end_date=end_date)

    if df_ff5.empty:
        # ファクターデータが原因のエラーであることがここで確定する
        st.warning("⚠️ ケネス・フレンチの5ファクターデータの取得に失敗しました。サイトの仕様変更または一時的なアクセス制限の可能性があります。")

    # --- 取得結果のレポート ---
    fetched_tickers = df_all_fund['Ticker'].tolist()
    missing_api = [t for t in port_tickers if t not in fetched_tickers]
    port_tickers_valid = [t for t in port_tickers if t not in missing_api]

    if missing_api:
        st.warning(f"⚠️ 以下の銘柄は株価取得エラー(429等)のため除外されました: {', '.join(missing_api)}")

    if not port_tickers_valid:
        st.error("計算可能な銘柄がありません。")
        st.stop()

    # --- Step C: ウェイトの初期計算 ---
    df_port_initial = df_all_fund[df_all_fund['Ticker'].isin(port_tickers_valid)].copy()
    if weight_list and len(weight_list) == len(port_tickers):
        weight_map = dict(zip(port_tickers, weight_list))
        df_port_initial['Weight'] = df_port_initial['Ticker'].map(weight_map)
        df_port_initial = QuantEngine.calculate_portfolio_weights(df_port_initial, user_weights_provided=True)
    else:
        df_port_initial = QuantEngine.calculate_portfolio_weights(df_port_initial, user_weights_provided=False)

    # --- Step D: 【重要】時系列多変量回帰分析の実行 ---
    progress_bar.progress(60, text="[4/5] 時系列多変量回帰分析 (Time-series Regression) 実行中...")
    
    regression_results = None
    if not df_ff5.empty:
        regression_results = QuantEngine.run_5factor_regression(df_hist, df_port_initial[['Ticker', 'Weight']], df_ff5)
    
    # 回帰結果がない場合のフォールバック用
    portfolio_z_radar = {}
    regression_success = False

    if regression_results and regression_results.get('R_squared') is not None:
        regression_success = True
        portfolio_z_radar = {
            'Beta': regression_results.get('Beta', 0),
            'Size': regression_results.get('Size', 0),
            'Value': regression_results.get('Value', 0),
            'Quality': regression_results.get('Quality', 0),
            'Investment': regression_results.get('Investment', 0)
        }
    else:
        st.warning("⚠️ 回帰分析に必要なデータ(十分な履歴またはファクターデータ)が揃わなかったため、一部の分析をスキップしました。")
        portfolio_z_radar = {'Beta': 1.0, 'Value': 0, 'Size': 0, 'Quality': 0, 'Investment': 0}

    # --- Step E: 銘柄固有（回帰前）Zスコアの計算 ---
    progress_bar.progress(80, text="[5/5] 銘柄固有スコアと加重平均寄与度の計算中...")
    df_bench = df_all_fund[df_all_fund['Ticker'].isin(bench_tickers)]
    market_stats, _ = UniverseManager.generate_market_stats(df_bench)

    df_port_proc = QuantEngine.process_raw_factors(df_port_initial)
    df_port_scored, _ = QuantEngine.compute_z_scores(df_port_proc, market_stats)
    
    # --- Step F: 加重平均寄与度の計算 ---
    df_port_scored, _ = QuantEngine.calculate_weighted_factor_contributions(df_port_scored)

    progress_bar.progress(100, text="分析完了！")
    time.sleep(0.4)
    progress_bar.empty()

    # ---------------------------------------------------------
    # 3. 分析結果の表示 UI
    # ---------------------------------------------------------
    
    if regression_success:
        r2 = regression_results['R_squared']
        p_val = regression_results['p_values']['Beta']
        st.markdown(f"""
        <div class="summary-box">
            <h4>📈 回帰分析サマリー (Kenneth French 5-Factor Model)</h4>
            <p>このポートフォリオの日次超過リターン変動のうち、<b>{r2*100:.1f}%</b> は5つのマクロファクターによって説明可能です。(決定係数 R² = {r2:.3f})<br>
            ※市場ベータの統計的有意性 (p値): {p_val:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        suffix = f"(R²={regression_results['R_squared']:.2f})" if regression_success else "(Fallback)"
        fig_radar = Visualizer.plot_radar_chart(portfolio_z_radar, title_suffix=suffix)
        st.plotly_chart(fig_radar, use_container_width=True)
    with col2:
        fig_bar = Visualizer.plot_contribution_bar_chart(df_port_scored)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------------
    # 4. 動的ソート機能付き 詳細データテーブル
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 銘柄別 固有ファクタースコア & 加重平均寄与度")
    
    base_cols = ['Ticker', 'Weight', 'Name'] if 'Name' in df_port_scored.columns else ['Ticker', 'Weight']
    score_cols = ['Value_Z', 'Quality_Z', 'Investment_Z', 'Size_Z']
    contrib_cols = ['Value_Z_Contrib', 'Quality_Z_Contrib', 'Investment_Z_Contrib', 'Size_Z_Contrib']
    
    avail_cols = base_cols + [c for c in score_cols + contrib_cols if c in df_port_scored.columns]
    df_display = df_port_scored[avail_cols].copy()
    
    rename_dict = {
        'Value_Z': 'Value (固有)', 'Quality_Z': 'Quality (固有)', 
        'Investment_Z': 'Investment (Asset Growth) (固有)', 'Size_Z': 'Size (固有)',
        'Value_Z_Contrib': 'Value (寄与)', 'Quality_Z_Contrib': 'Quality (寄与)', 
        'Investment_Z_Contrib': 'Investment (Asset Growth) (寄与)', 'Size_Z_Contrib': 'Size (寄与)',
        'Weight': 'ウェイト'
    }
    df_display.rename(columns=rename_dict, inplace=True)
    
    sort_options = ['ウェイト'] + [v for v in rename_dict.values() if v in df_display.columns]
    
    col_sort1, col_sort2 = st.columns([1, 3])
    with col_sort1:
        sort_by = st.selectbox("並べ替え基準", options=sort_options, index=0)
    with col_sort2:
        st.markdown("<br>", unsafe_allow_html=True) 
        sort_asc = st.checkbox("昇順で並べ替え", value=False)
    
    if sort_by in df_display.columns:
        df_display = df_display.sort_values(by=sort_by, ascending=sort_asc)
    
    format_dict = {col: "{:.2f}" for col in df_display.columns if col not in ['Ticker', 'Name']}
    
    st.dataframe(
        df_display.style.format(format_dict)
                        .background_gradient(subset=[c for c in df_display.columns if '寄与' in c], cmap='RdBu', vmin=-0.5, vmax=0.5),
        use_container_width=True,
        hide_index=True
    )

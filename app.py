import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# カスタムモジュールの読み込み
try:
    from data_provider import DataProvider
    from quant_engine import QuantEngine
    from universe_manager import UniverseManager
except ImportError as e:
    st.error(f"起動エラー: モジュールが見つかりません ({e})")
    st.info("app.py と同じフォルダに data_provider.py, quant_engine.py, universe_manager.py があるか確認してください。")
    st.stop()

# ---------------------------------------------------------
# 0. ページ設定 & デザイン定義
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Market Factor Lab (Pro)")

# カスタムCSS
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
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 14px;
        font-weight: bold;
        color: #555;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #007bff;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    /* テーブル内の文字サイズ調整 */
    .stDataFrame { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. 定数定義 (ベンチマーク用ユニバース)
# ---------------------------------------------------------
NIKKEI_225_SAMPLE = [
    "7203.T", "6758.T", "8035.T", "9984.T", "9983.T", "6098.T", "4063.T", "6367.T", "9432.T", "4502.T",
    "4503.T", "6501.T", "7267.T", "8058.T", "8001.T", "6954.T", "6981.T", "9020.T", "9022.T", "7741.T",
    "5108.T", "4452.T", "6902.T", "7974.T", "8031.T", "4519.T", "4568.T", "6273.T", "4543.T", "6702.T",
    "6503.T", "4901.T", "4911.T", "2502.T", "2802.T", "3382.T", "8306.T", "8316.T", "8411.T", "8766.T",
    "8591.T", "8801.T", "8802.T", "9021.T", "9101.T", "9433.T", "9434.T", "9501.T", "9502.T"
]

TOPIX_CORE_30 = [
    "7203.T", "6758.T", "8306.T", "9984.T", "9432.T", "6861.T", "8035.T", "6098.T", "8316.T", "4063.T",
    "9983.T", "6367.T", "4502.T", "7974.T", "8058.T", "8001.T", "2914.T", "6501.T", "7267.T", "8411.T",
    "6954.T", "6902.T", "7741.T", "9020.T", "9022.T", "4452.T", "5108.T", "8801.T", "6752.T", "6273.T"
]

# ---------------------------------------------------------
# 2. ヘルパー関数
# ---------------------------------------------------------
def parse_portfolio_input(input_text):
    weights = {}
    raw_items = [x.strip() for x in input_text.replace('\n', ',').split(',') if x.strip()]
    if not raw_items: return {}

    is_weighted = any(':' in item for item in raw_items)
    if is_weighted:
        for item in raw_items:
            if ':' in item:
                parts = item.split(':')
                ticker = parts[0].strip()
                try: w = float(parts[1])
                except ValueError: w = 0.0
                weights[ticker] = w
            else:
                weights[item] = 0.0
    else:
        count = len(raw_items)
        for item in raw_items:
            weights[item] = 1.0 / count
            
    total_w = sum(weights.values())
    if total_w > 0:
        for k in weights: weights[k] = weights[k] / total_w
            
    return weights

def parse_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
        else: df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return {}
    
    df.columns = [c.strip().lower() for c in df.columns]
    
    ticker_col = next((c for c in ['ticker', 'code', 'symbol', 'stock', '銘柄コード', 'コード'] if c in df.columns), None)
    if not ticker_col:
        st.error("CSV/Excelに「Ticker」または「Code」列が見つかりません。")
        return {}
    
    weight_col = next((c for c in ['weight', 'ratio', 'share', 'portfolio%', '比率', 'ウェイト'] if c in df.columns), None)
    
    weights = {}
    count = len(df)
    
    for _, row in df.iterrows():
        t = str(row[ticker_col]).strip()
        try: w = float(row[weight_col]) if weight_col else 1.0 / count
        except: w = 0.0
        weights[t] = w
        
    total_w = sum(weights.values())
    if total_w > 0:
        for k in weights: weights[k] = weights[k] / total_w
            
    return weights

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_market_data(tickers, bench_etf):
    df_fund = DataProvider.fetch_fundamentals(tickers)
    df_hist = DataProvider.fetch_historical_prices(tickers + [bench_etf])
    return df_fund, df_hist

# ---------------------------------------------------------
# 3. UI レイアウト & 入力
# ---------------------------------------------------------
st.sidebar.header("📊 Settings")

bench_mode = st.sidebar.selectbox("Benchmark Index", ["Nikkei 225", "TOPIX Core 30"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Display Options")
sort_key = st.sidebar.selectbox(
    "Sort Table By",
    ["Ticker", "Value (PBR)", "Quality (ROE)", "Investment (Asset Growth)", "Size", "Weight"]
)

if bench_mode == "Nikkei 225":
    benchmark_etf = "1321.T"
    universe_tickers = NIKKEI_225_SAMPLE
else:
    benchmark_etf = "1306.T"
    universe_tickers = TOPIX_CORE_30

st.sidebar.markdown("---")
st.sidebar.subheader("My Portfolio")

input_mode = st.sidebar.radio("Input Mode", ["Manual Input", "File Upload"], horizontal=True)

if input_mode == "Manual Input":
    st.sidebar.caption("Format: `Ticker` or `Ticker:Weight`")
    default_input = "7203.T: 40, 6758.T: 30, 9984.T: 30"
    input_text = st.sidebar.text_area("Input", default_input, height=120)
    uploaded_file = None
else:
    st.sidebar.caption("Support: CSV, Excel (Columns: Ticker, Weight)")
    uploaded_file = st.sidebar.file_uploader("Upload Portfolio", type=['csv', 'xlsx'])
    input_text = ""

run_btn = st.sidebar.button("Run Analysis", type="primary")

# ---------------------------------------------------------
# 4. メイン処理フロー
# ---------------------------------------------------------
if run_btn:
    st.title("🛡️ Market Factor Lab (Pro)")
    
    if input_mode == "Manual Input":
        portfolio_dict = parse_portfolio_input(input_text)
    else:
        if uploaded_file is not None:
            portfolio_dict = parse_uploaded_file(uploaded_file)
        else:
            st.warning("ファイルをアップロードしてください。")
            st.stop()
            
    user_tickers = list(portfolio_dict.keys())
    
    if not user_tickers:
        st.warning("有効な銘柄が見つかりませんでした。入力形式を確認してください。")
        st.stop()
        
    with st.status("Running Analysis...", expanded=True) as status:
        st.write(f"1. Fetching Market Data ({bench_mode})...")
        df_bench_fund, df_bench_hist = get_cached_market_data(universe_tickers, benchmark_etf)
        
        st.write("2. Calculating Market Beta...")
        df_bench_fund = QuantEngine.calculate_beta(df_bench_fund, df_bench_hist, benchmark_etf)
        
        st.write("3. Generating Robust Statistics (Universe Manager)...")
        market_stats, df_bench_processed = UniverseManager.generate_market_stats(df_bench_fund)
        
        st.write("4. Analyzing Your Portfolio...")
        df_user_fund, df_user_hist = get_cached_market_data(user_tickers, benchmark_etf)
        df_user_fund = QuantEngine.calculate_beta(df_user_fund, df_user_hist, benchmark_etf)
        
        df_user_proc = QuantEngine.process_raw_factors(df_user_fund)
        df_scored, r_squared_map = QuantEngine.compute_z_scores(df_user_proc, market_stats)
        
        df_scored['Weight'] = df_scored['Ticker'].map(portfolio_dict)
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # -----------------------------------------------------
    # 結果表示 (Phase 4: Safety & Polish)
    # -----------------------------------------------------
    if df_scored.empty:
        st.error("❌ データの取得に完全に失敗しました。")
        st.info("💡 ヒント: Yahoo Financeのアクセス制限（429 Error）に到達している可能性があります。しばらく時間を置くか、入力する銘柄数を減らして再試行してください。")
        st.stop()

    missing_tickers = [t for t in user_tickers if t not in df_scored['Ticker'].values]
    if missing_tickers:
        st.warning(f"⚠️ 以下の銘柄はデータが取得できなかったためスキップされました（上場廃止やティッカーミス、またはAPI制限の可能性があります）:\n`{', '.join(missing_tickers)}`")

    valid_cols = [c for c in df_scored.columns if c.endswith('_Z')]
    valid_count = df_scored.dropna(subset=valid_cols).shape[0]
    total_count = len(df_scored)
    
    if valid_count < total_count:
        st.info(f"ℹ️ 一部のデータが取得できなかった銘柄があります (完全なデータ: {valid_count}/{total_count} 銘柄)。API制限（429 Error）の影響で N/A が含まれる場合があります。")

    z_cols = [c for c in df_scored.columns if c.endswith('_Z')]
    portfolio_exposure = {}
    
    for col in z_cols:
        valid_rows = df_scored.dropna(subset=[col, 'Weight'])
        if not valid_rows.empty:
            w_avg = np.average(valid_rows[col], weights=valid_rows['Weight'])
            portfolio_exposure[col.replace('_Z', '')] = w_avg
        else:
            portfolio_exposure[col.replace('_Z', '')] = 0.0

    # --- KPI Cards ---
    st.subheader(f"📊 Portfolio Diagnostic (vs {bench_mode})")
    col1, col2, col3 = st.columns(3)
    
    valid_beta = df_user_fund.dropna(subset=['Beta_Raw']).copy()
    valid_beta['Weight'] = valid_beta['Ticker'].map(portfolio_dict)
    avg_beta = np.average(valid_beta['Beta_Raw'], weights=valid_beta['Weight']) if not valid_beta.empty else 0.0

    col1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Weighted Beta</div>
        <div class="metric-value">{avg_beta:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    valid_roe = df_scored.dropna(subset=['Quality_Raw', 'Weight']).copy()
    roe_display = f"{np.average(valid_roe['Quality_Raw'], weights=valid_roe['Weight']):.1f}%" if not valid_roe.empty else "N/A"
        
    col2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg ROE (Profitability)</div>
        <div class="metric-value" style="color: #007bff;">{roe_display}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Holdings</div>
        <div class="metric-value">{len(user_tickers) - len(missing_tickers)} / {len(user_tickers)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- Charts ---
    c_chart, c_insight = st.columns([2, 1])
    
    with c_chart:
        st.subheader("Factor Exposure (Weighted)")
        factors = list(portfolio_exposure.keys())
        scores = list(portfolio_exposure.values())
        y_labels = [f"{f}" for f in factors]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scores, y=y_labels, orientation='h',
            marker=dict(color=scores, colorscale='RdBu', cmin=-2, cmax=2),
            text=[f"{s:.2f}" for s in scores], textposition='auto'
        ))
        fig.update_layout(
            title=f"Weighted Z-Scores (0 = {bench_mode})",
            xaxis_title="Standard Deviation (σ)",
            yaxis=dict(autorange="reversed"),
            height=400, margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
        
        # 【修正】Streamlitの警告ログ解消のため、パラメータをスッキリ修正
        st.plotly_chart(fig)

    with c_insight:
        st.subheader("AI Insight")
        insights = QuantEngine.generate_insights(portfolio_exposure)
        for msg in insights:
            st.markdown(f'<div class="insight-box">{msg}</div>', unsafe_allow_html=True)
        st.info("※ SizeとInvestmentは反転しています（＋方向 = 小型株 / 保守的経営）")

    # --- Data Table ---
    with st.expander("Show Detailed Factor Data", expanded=True):
        
        df_display = df_scored.copy()

        if "Value" in sort_key and 'Value_Z' in df_display.columns: df_display = df_display.sort_values('Value_Z', ascending=False)
        elif "Quality" in sort_key and 'Quality_Z' in df_display.columns: df_display = df_display.sort_values('Quality_Z', ascending=False)
        elif "Investment" in sort_key and 'Investment_Z' in df_display.columns: df_display = df_display.sort_values('Investment_Z', ascending=False)
        elif "Size" in sort_key and 'Size_Z' in df_display.columns: df_display = df_display.sort_values('Size_Z', ascending=False)
        elif "Weight" in sort_key: df_display = df_display.sort_values('Weight', ascending=False)
        else: df_display = df_display.sort_values('Ticker', ascending=True)

        def format_col(row, raw_col, z_col, unit="", is_percent=False):
            raw_val = row.get(raw_col, np.nan)
            z_val = row.get(z_col, np.nan)
            
            if pd.isna(raw_val): return "N/A"
            z_str = f"{z_val:.2f}" if pd.notna(z_val) else "N/A"
            
            if is_percent: val_str = f"{raw_val*100:.1f}%"
            else: val_str = f"{raw_val:.2f}{unit}"
                
            return f"{val_str} (Z: {z_str})"

        if 'PBR' in df_display.columns and 'Value_Z' in df_display.columns:
            df_display['Value (PBR)'] = df_display.apply(lambda x: format_col(x, 'PBR', 'Value_Z', unit="x"), axis=1)
        
        if 'Quality_Raw' in df_display.columns and 'Quality_Z' in df_display.columns:
             df_display['Quality (ROE)'] = df_display.apply(lambda x: format_col(x, 'Quality_Raw', 'Quality_Z', is_percent=True), axis=1)
             
        if 'Investment_Raw' in df_display.columns and 'Investment_Z' in df_display.columns:
             df_display['Investment (Asset Growth)'] = df_display.apply(lambda x: format_col(x, 'Investment_Raw', 'Investment_Z', is_percent=True), axis=1)

        if 'Size_Z' in df_display.columns or 'MarketCap' in df_display.columns:
            if 'MarketCap' in df_display.columns:
                 def format_size(x):
                     mcap = x.get('MarketCap', np.nan)
                     z_val = x.get('Size_Z', np.nan)
                     if pd.isna(mcap): return "N/A"
                     z_str = f"{z_val:.2f}" if pd.notna(z_val) else "N/A"
                     return f"{mcap/1e9:.0f}B (Z: {z_str})"
                 df_display['Size (MktCap)'] = df_display.apply(format_size, axis=1)
            else:
                 df_display['Size (Log)'] = df_display.apply(lambda x: format_col(x, 'Size_Log', 'Size_Z'), axis=1)

        base_cols = ['Ticker']
        if 'Name' in df_display.columns: base_cols.append('Name')
        base_cols.append('Weight')
        
        custom_cols = [c for c in ['Value (PBR)', 'Quality (ROE)', 'Investment (Asset Growth)', 'Size (MktCap)', 'Size (Log)'] if c in df_display.columns]
        final_cols = base_cols + custom_cols
        
        # 【修正】Streamlitの警告を解消するため use_container_width の指定を削除し、最新の書き方に準拠
        st.dataframe(
            df_display[final_cols].style.format({'Weight': '{:.1%}'}),
            hide_index=True
        )

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re  # 正規表現モジュール（4桁数字の判定用）

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

# カスタムCSS (レスポンシブ対応追加)
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
    
    /* モバイル・小画面用レスポンシブデザイン */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 20px;
        }
        .metric-label {
            font-size: 12px;
        }
    }
    
    /* ダウンロードボタンの余白 */
    .stDownloadButton {
        margin-top: 10px;
        margin-bottom: 10px;
    }
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

# ティッカーを自動補正する関数 (例: "7203" -> "7203.T")
def format_ticker(t):
    t_str = str(t).strip().upper()
    # 4桁の数字のみで構成されている場合は '.T' を付与
    if re.fullmatch(r'\d{4}', t_str):
        return f"{t_str}.T"
    return t_str

def parse_portfolio_input(input_text):
    weights = {}
    raw_items = [x.strip() for x in input_text.replace('\n', ',').split(',') if x.strip()]
    if not raw_items: return {}

    is_weighted = any(':' in item for item in raw_items)
    if is_weighted:
        for item in raw_items:
            if ':' in item:
                parts = item.split(':')
                # ティッカーを補正関数に通す
                ticker = format_ticker(parts[0])
                try: w = float(parts[1])
                except ValueError: w = 0.0
                weights[ticker] = w
            else:
                weights[format_ticker(item)] = 0.0
    else:
        count = len(raw_items)
        for item in raw_items:
            # ティッカーを補正関数に通す
            weights[format_ticker(item)] = 1.0 / count
            
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
    
    # 列名をすべて文字列にし、空白を消し、小文字化（揺らぎ吸収）
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 認識できる列名のバリエーションを大幅増加
    possible_ticker_cols = ['ticker', 'code', 'symbol', 'stock', '銘柄コード', 'コード', '銘柄']
    possible_weight_cols = ['weight', 'ratio', 'share', 'portfolio%', '比率', 'ウェイト', '割合', '%']

    ticker_col = next((c for c in possible_ticker_cols if c in df.columns), None)
    if not ticker_col:
        st.error(f"CSV/Excelに銘柄を特定する列が見つかりません。以下のいずれかの列名を使用してください:\n{', '.join(possible_ticker_cols)}")
        return {}
    
    weight_col = next((c for c in possible_weight_cols if c in df.columns), None)
    
    weights = {}
    count = len(df)
    
    for _, row in df.iterrows():
        # ティッカーを補正関数に通す
        t = format_ticker(row[ticker_col])
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

# データフレームをCSVダウンロード用データに変換するヘルパー関数
@st.cache_data
def convert_df_to_csv(df):
    # 日本語の文字化けを防ぐためにutf-8-sigを使用
    return df.to_csv(index=False).encode('utf-8-sig')

# --- 表示用データフレーム整形ヘルパー ---
def create_display_dataframe(df_to_format, sort_val, is_portfolio=True):
    df_disp = df_to_format.copy()

    # ソート処理
    if "Value" in sort_val and 'Value_Z' in df_disp.columns: df_disp = df_disp.sort_values('Value_Z', ascending=False)
    elif "Quality" in sort_val and 'Quality_Z' in df_disp.columns: df_disp = df_disp.sort_values('Quality_Z', ascending=False)
    elif "Investment" in sort_val and 'Investment_Z' in df_disp.columns: df_disp = df_disp.sort_values('Investment_Z', ascending=False)
    elif "Size" in sort_val and 'Size_Z' in df_disp.columns: df_disp = df_disp.sort_values('Size_Z', ascending=False)
    elif "Weight" in sort_val and 'Weight' in df_disp.columns: df_disp = df_disp.sort_values('Weight', ascending=False)
    else: df_disp = df_disp.sort_values('Ticker', ascending=True)

    # フォーマット関数
    def fmt(row, raw_col, z_col, unit="", is_percent=False):
        raw_val = row.get(raw_col, np.nan)
        z_val = row.get(z_col, np.nan)
        if pd.isna(raw_val): return "N/A"
        z_str = f"{z_val:.2f}" if pd.notna(z_val) else "N/A"
        if is_percent: val_str = f"{raw_val*100:.1f}%"
        else: val_str = f"{raw_val:.2f}{unit}"
        return f"{val_str} (Z: {z_str})"

    if 'PBR' in df_disp.columns and 'Value_Z' in df_disp.columns:
        df_disp['Value (PBR)'] = df_disp.apply(lambda x: fmt(x, 'PBR', 'Value_Z', unit="x"), axis=1)
    if 'Quality_Raw' in df_disp.columns and 'Quality_Z' in df_disp.columns:
        df_disp['Quality (ROE)'] = df_disp.apply(lambda x: fmt(x, 'Quality_Raw', 'Quality_Z', is_percent=True), axis=1)
    if 'Investment_Raw' in df_disp.columns and 'Investment_Z' in df_disp.columns:
        df_disp['Investment (Asset Growth)'] = df_disp.apply(lambda x: fmt(x, 'Investment_Raw', 'Investment_Z', is_percent=True), axis=1)

    if 'MarketCap' in df_disp.columns:
        def format_size(x):
            mcap = x.get('MarketCap', np.nan)
            z_val = x.get('Size_Z', np.nan)
            if pd.isna(mcap): return "N/A"
            z_str = f"{z_val:.2f}" if pd.notna(z_val) else "N/A"
            return f"{mcap/1e9:.0f}B (Z: {z_str})"
        df_disp['Size (MktCap)'] = df_disp.apply(format_size, axis=1)
    elif 'Size_Log' in df_disp.columns:
        df_disp['Size (Log)'] = df_disp.apply(lambda x: fmt(x, 'Size_Log', 'Size_Z'), axis=1)

    base_cols = ['Ticker']
    if 'Name' in df_disp.columns: base_cols.append('Name')
    if is_portfolio and 'Weight' in df_disp.columns: base_cols.append('Weight')
    
    custom_cols = [c for c in ['Value (PBR)', 'Quality (ROE)', 'Investment (Asset Growth)', 'Size (MktCap)', 'Size (Log)'] if c in df_disp.columns]
    return df_disp[base_cols + custom_cols]

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
    # 数字だけの入力を許容する旨をキャプションに追加
    st.sidebar.caption("Format: `7203` or `7203:40` (.T is auto-added)")
    default_input = "7203: 40, 6758: 30, 9984: 30"
    input_text = st.sidebar.text_area("Input", default_input, height=120)
    uploaded_file = None
else:
    st.sidebar.caption("Support: CSV, Excel (Columns: 銘柄コード, 比率等)")
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
        st.warning("有効な銘柄が見つかりませんでした。入力形式またはファイルの中身を確認してください。")
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
    # 結果表示 (Tabsによる拡張)
    # -----------------------------------------------------
    if df_scored.empty:
        st.error("❌ データの取得に完全に失敗しました。")
        st.info("💡 ヒント: API制限（429 Error）の可能性があります。少し待ってから再試行してください。")
        st.stop()

    missing_tickers = [t for t in user_tickers if t not in df_scored['Ticker'].values]
    if missing_tickers:
        st.warning(f"⚠️ データ取得スキップ銘柄:\n`{', '.join(missing_tickers)}`")

    tab_port, tab_bench = st.tabs(["💼 My Portfolio", "🌐 Market Universe (All Stocks)"])

    # ==========================================
    # Tab 1: My Portfolio
    # ==========================================
    with tab_port:
        z_cols = [c for c in df_scored.columns if c.endswith('_Z')]
        portfolio_exposure = {}
        
        for col in z_cols:
            valid_rows = df_scored.dropna(subset=[col, 'Weight'])
            if not valid_rows.empty:
                w_avg = np.average(valid_rows[col], weights=valid_rows['Weight'])
                portfolio_exposure[col.replace('_Z', '')] = w_avg
            else:
                portfolio_exposure[col.replace('_Z', '')] = 0.0

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
            # レスポンシブ対応: use_container_width=True
            st.plotly_chart(fig, use_container_width=True)

        with c_insight:
            st.subheader("AI Insight")
            insights = QuantEngine.generate_insights(portfolio_exposure)
            for msg in insights:
                st.markdown(f'<div class="insight-box">{msg}</div>', unsafe_allow_html=True)
            st.info("※ SizeとInvestmentは反転しています（＋方向 = 小型株 / 保守的経営）")

        with st.expander("Show Detailed Factor Data", expanded=True):
            df_port_disp = create_display_dataframe(df_scored, sort_key, is_portfolio=True)
            
            # CSVダウンロード機能の追加
            csv_port = convert_df_to_csv(df_port_disp)
            st.download_button(
                label="📥 Download Portfolio Data (CSV)",
                data=csv_port,
                file_name='portfolio_analysis.csv',
                mime='text/csv',
            )
            
            # 表を画面幅に合わせる
            st.dataframe(df_port_disp.style.format({'Weight': '{:.1%}'}), hide_index=True, use_container_width=True)

    # ==========================================
    # Tab 2: Market Universe (全銘柄モニタリング)
    # ==========================================
    with tab_bench:
        st.subheader(f"🌐 {bench_mode} - Constituent Factors")
        st.markdown("ベンチマークを構成する全銘柄の現在のファクター状態（Zスコア）です。サイドバーの「Sort Table By」でランキング形式で確認できます。")
        
        df_bench_scored, _ = QuantEngine.compute_z_scores(df_bench_processed, market_stats)
        
        df_bench_disp = create_display_dataframe(df_bench_scored, sort_key, is_portfolio=False)
        
        # CSVダウンロード機能の追加
        csv_bench = convert_df_to_csv(df_bench_disp)
        st.download_button(
            label="📥 Download Universe Data (CSV)",
            data=csv_bench,
            file_name=f"{bench_mode.replace(' ', '_').lower()}_universe.csv",
            mime='text/csv',
        )
        
        # 表を画面幅に合わせる
        st.dataframe(df_bench_disp, hide_index=True, use_container_width=True)

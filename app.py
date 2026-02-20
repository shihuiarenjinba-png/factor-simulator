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
# 日経225 (サンプル)
NIKKEI_225_SAMPLE = [
    "7203.T", "6758.T", "8035.T", "9984.T", "9983.T", "6098.T", "4063.T", "6367.T", "9432.T", "4502.T",
    "4503.T", "6501.T", "7267.T", "8058.T", "8001.T", "6954.T", "6981.T", "9020.T", "9022.T", "7741.T",
    "5108.T", "4452.T", "6902.T", "7974.T", "8031.T", "4519.T", "4568.T", "6273.T", "4543.T", "6702.T",
    "6503.T", "4901.T", "4911.T", "2502.T", "2802.T", "3382.T", "8306.T", "8316.T", "8411.T", "8766.T",
    "8591.T", "8801.T", "8802.T", "9021.T", "9101.T", "9433.T", "9434.T", "9501.T", "9502.T"
]

# TOPIX Core 30 (サンプル: 日本を代表する超大型株)
TOPIX_CORE_30 = [
    "7203.T", "6758.T", "8306.T", "9984.T", "9432.T", "6861.T", "8035.T", "6098.T", "8316.T", "4063.T",
    "9983.T", "6367.T", "4502.T", "7974.T", "8058.T", "8001.T", "2914.T", "6501.T", "7267.T", "8411.T",
    "6954.T", "6902.T", "7741.T", "9020.T", "9022.T", "4452.T", "5108.T", "8801.T", "6752.T", "6273.T"
]

# ---------------------------------------------------------
# 2. ヘルパー関数
# ---------------------------------------------------------
def parse_portfolio_input(input_text):
    """入力テキストを解析し、{Ticker: Weight} の辞書を返す"""
    weights = {}
    raw_items = [x.strip() for x in input_text.replace('\n', ',').split(',') if x.strip()]
    
    if not raw_items:
        return {}

    is_weighted = any(':' in item for item in raw_items)
    
    if is_weighted:
        for item in raw_items:
            if ':' in item:
                parts = item.split(':')
                ticker = parts[0].strip()
                try:
                    w = float(parts[1])
                except ValueError:
                    w = 0.0
                weights[ticker] = w
            else:
                weights[item] = 0.0
    else:
        count = len(raw_items)
        for item in raw_items:
            weights[item] = 1.0 / count
            
    # 重みの正規化
    total_w = sum(weights.values())
    if total_w > 0:
        for k in weights:
            weights[k] = weights[k] / total_w
            
    return weights

def parse_uploaded_file(uploaded_file):
    """アップロードされたファイルを解析して {Ticker: Weight} を返す"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return {}
    
    # カラム名のゆらぎ吸収 (大文字小文字無視)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Tickerカラムの特定
    ticker_col = None
    possible_ticker_cols = ['ticker', 'code', 'symbol', 'stock', '銘柄コード', 'コード']
    for c in possible_ticker_cols:
        if c in df.columns:
            ticker_col = c
            break
            
    if not ticker_col:
        st.error("CSV/Excelに「Ticker」または「Code」列が見つかりません。")
        return {}
    
    # Weightカラムの特定 (なければ均等)
    weight_col = None
    possible_weight_cols = ['weight', 'ratio', 'share', 'portfolio%', '比率', 'ウェイト']
    for c in possible_weight_cols:
        if c in df.columns:
            weight_col = c
            break
            
    weights = {}
    count = len(df)
    
    for _, row in df.iterrows():
        t = str(row[ticker_col]).strip()
        if weight_col:
            try:
                w = float(row[weight_col])
            except:
                w = 0.0
        else:
            w = 1.0 / count
        weights[t] = w
        
    # 重みの正規化
    total_w = sum(weights.values())
    if total_w > 0:
        for k in weights:
            weights[k] = weights[k] / total_w
            
    return weights

# 【修正】ステップ1：キャッシュ機能の導入
@st.cache_data(ttl=3600)
def get_cached_market_data(tickers, bench_etf):
    """データ取得をキャッシュ化し、重い通信処理をスキップする"""
    df_fund = DataProvider.fetch_fundamentals(tickers)
    df_hist = DataProvider.fetch_historical_prices(tickers + [bench_etf])
    return df_fund, df_hist

# ---------------------------------------------------------
# 3. UI レイアウト & 入力
# ---------------------------------------------------------
st.sidebar.header("📊 Settings")

bench_mode = st.sidebar.selectbox("Benchmark Index", ["Nikkei 225", "TOPIX Core 30"])

# ソート対象を Asset Growth に変更
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Display Options")
sort_key = st.sidebar.selectbox(
    "Sort Table By",
    ["Ticker", "Value (PBR)", "Quality (ROE)", "Momentum (Return)", "Investment (Asset Growth)", "Size", "Weight"]
)

if bench_mode == "Nikkei 225":
    benchmark_etf = "1321.T"
    universe_tickers = NIKKEI_225_SAMPLE
else:
    benchmark_etf = "1306.T"
    universe_tickers = TOPIX_CORE_30

st.sidebar.markdown("---")

st.sidebar.subheader("My Portfolio")

# 入力モード選択
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
    
    # [Step 1] 入力解析 (モード分岐)
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
        
    # [Step 2] データ取得 & 市場統計作成
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. ベンチマークデータの取得
    status_text.text(f"Fetching Market Data ({bench_mode})...")
    
    # 【修正】ステップ1：キャッシュ関数の呼び出しに変更
    df_bench_fund, df_bench_hist = get_cached_market_data(universe_tickers, benchmark_etf)
    
    progress_bar.progress(20)
    
    # 2. ベンチマークの計算
    status_text.text("Calculating Market Beta & Momentum...")
    
    df_bench_fund = QuantEngine.calculate_beta_momentum(df_bench_fund, df_bench_hist, benchmark_etf)
    
    progress_bar.progress(40)
    
    status_text.text("Generating Robust Statistics (Universe Manager)...")
    market_stats, df_bench_processed = UniverseManager.generate_market_stats(df_bench_fund)
    progress_bar.progress(60)

    # [Step 3] ユーザーポートフォリオ評価
    status_text.text("Analyzing Your Portfolio...")
    
    # 3. ユーザーデータの取得
    # 【修正】ステップ1：ユーザーデータにもキャッシュ関数を利用
    df_user_fund, df_user_hist = get_cached_market_data(user_tickers, benchmark_etf)
    
    # 4. ユーザーデータの計算
    df_user_fund = QuantEngine.calculate_beta_momentum(df_user_fund, df_user_hist, benchmark_etf)
    
    # 生データ加工
    df_user_proc = QuantEngine.process_raw_factors(df_user_fund)
    
    # Zスコア計算 (ここで内部的に直交化も実行されます)
    df_scored, r_squared_map = QuantEngine.compute_z_scores(df_user_proc, market_stats)
    
    # ウェイト情報をマージ
    df_scored['Weight'] = df_scored['Ticker'].map(portfolio_dict)
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    # -----------------------------------------------------
    # [Step 4] 結果表示 (Phase 4: Safety & Polish)
    # -----------------------------------------------------
    
    # [Phase 4] 安全装置: データが空の場合は停止
    if df_scored.empty:
        st.error("データの取得に失敗しました。時間をおいて再試行するか、銘柄コードを確認してください。")
        st.stop()

    # [Phase 4] データ健全性のチェック
    valid_cols = [c for c in df_scored.columns if c.endswith('_Z')]
    valid_count = df_scored.dropna(subset=valid_cols).shape[0]
    total_count = len(df_scored)
    
    if valid_count < total_count:
        st.warning(f"⚠️ 一部のデータが取得できませんでした (完全なデータ: {valid_count}/{total_count} 銘柄)。N/A の項目が含まれる可能性があります。")

    # 加重平均Zスコアの算出
    z_cols = [c for c in df_scored.columns if c.endswith('_Z')]
    portfolio_exposure = {}
    
    for col in z_cols:
        valid_rows = df_scored.dropna(subset=[col, 'Weight'])
        if not valid_rows.empty:
            w_avg = np.average(valid_rows[col], weights=valid_rows['Weight'])
            factor_name = col.replace('_Z', '')
            portfolio_exposure[factor_name] = w_avg
        else:
            portfolio_exposure[col.replace('_Z', '')] = 0.0

    # --- KPI Cards ---
    st.subheader(f"📊 Portfolio Diagnostic (vs {bench_mode})")
    col1, col2, col3 = st.columns(3)
    
    # Weighted Beta
    valid_beta = df_user_fund.dropna(subset=['Beta_Raw']).copy()
    valid_beta['Weight'] = valid_beta['Ticker'].map(portfolio_dict)
    if not valid_beta.empty:
        avg_beta = np.average(valid_beta['Beta_Raw'], weights=valid_beta['Weight'])
    else:
        avg_beta = 0.0

    col1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Weighted Beta</div>
        <div class="metric-value">{avg_beta:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Avg ROE (Profitability)
    valid_roe = df_scored.dropna(subset=['Quality_Raw', 'Weight']).copy()
    if not valid_roe.empty:
        avg_roe = np.average(valid_roe['Quality_Raw'], weights=valid_roe['Weight'])
        roe_display = f"{avg_roe:.1f}%"
    else:
        roe_display = "N/A"
        
    col2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg ROE (Profitability)</div>
        <div class="metric-value" style="color: #007bff;">{roe_display}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Holdings
    col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Holdings</div>
        <div class="metric-value">{len(user_tickers)}</div>
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
        st.plotly_chart(fig, use_container_width=True)

    with c_insight:
        st.subheader("AI Insight")
        insights = QuantEngine.generate_insights(portfolio_exposure)
        for msg in insights:
            st.markdown(f'<div class="insight-box">{msg}</div>', unsafe_allow_html=True)
        st.info("※ SizeとInvestmentは反転しています（＋方向 = 小型株 / 保守的経営）")

    # --- Data Table ---
    with st.expander("Show Detailed Factor Data", expanded=True):
        
        df_display = df_scored.copy()

        # 並び替えロジック
        if "Value" in sort_key:
            if 'Value_Z' in df_display.columns: df_display = df_display.sort_values('Value_Z', ascending=False)
        elif "Quality" in sort_key:
            if 'Quality_Z' in df_display.columns: df_display = df_display.sort_values('Quality_Z', ascending=False)
        elif "Momentum" in sort_key:
            if 'Momentum_Z' in df_display.columns: df_display = df_display.sort_values('Momentum_Z', ascending=False)
        elif "Investment" in sort_key:
            if 'Investment_Z' in df_display.columns: df_display = df_display.sort_values('Investment_Z', ascending=False)
        elif "Size" in sort_key:
            if 'Size_Z' in df_display.columns: df_display = df_display.sort_values('Size_Z', ascending=False)
        elif "Weight" in sort_key:
            df_display = df_display.sort_values('Weight', ascending=False)
        else:
            df_display = df_display.sort_values('Ticker', ascending=True)

        # 【修正】表示用カラムの生成関数 (Zスコアが無くても生データを表示するようガードを緩和)
        def format_col(row, raw_col, z_col, unit="", is_percent=False):
            raw_val = row.get(raw_col, np.nan)
            z_val = row.get(z_col, np.nan)
            
            # 生データ自体が無い場合は N/A
            if pd.isna(raw_val):
                return "N/A"
            
            # 生データがあればZスコアがNaNでも表示する
            z_str = f"{z_val:.2f}" if pd.notna(z_val) else "N/A"
            
            if is_percent:
                val_str = f"{raw_val*100:.1f}%"
            else:
                val_str = f"{raw_val:.2f}{unit}"
                
            return f"{val_str} (Z: {z_str})"

        # 1. Value (PBR)
        if 'PBR' in df_display.columns and 'Value_Z' in df_display.columns:
            df_display['Value (PBR)'] = df_display.apply(
                lambda x: format_col(x, 'PBR', 'Value_Z', unit="x"), axis=1
            )
        
        # 2. Quality (ROE)
        if 'Quality_Raw' in df_display.columns and 'Quality_Z' in df_display.columns:
             df_display['Quality (ROE)'] = df_display.apply(
                lambda x: format_col(x, 'Quality_Raw', 'Quality_Z', is_percent=True), axis=1
            )
             
        # 3. Momentum (Return)
        if 'Momentum_Raw' in df_display.columns and 'Momentum_Z' in df_display.columns:
             df_display['Momentum (Return)'] = df_display.apply(
                lambda x: format_col(x, 'Momentum_Raw', 'Momentum_Z', is_percent=True), axis=1
            )
        
        # 4. Investment (Asset Growth)
        if 'Investment_Raw' in df_display.columns and 'Investment_Z' in df_display.columns:
             df_display['Investment (Asset Growth)'] = df_display.apply(
                lambda x: format_col(x, 'Investment_Raw', 'Investment_Z', is_percent=True), axis=1
            )

        # 【修正】5. Size (Market Cap)
        # Zスコアが欠損していてもMarketCapがあれば表示させる
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
                 df_display['Size (Log)'] = df_display.apply(
                    lambda x: format_col(x, 'Size_Log', 'Size_Z'), axis=1
                )

        # 表示カラムの選定
        base_cols = ['Ticker']
        if 'Name' in df_display.columns:
            base_cols.append('Name')
        base_cols.append('Weight')
        
        # 生成したカスタムカラムを追加
        custom_cols = []
        if 'Value (PBR)' in df_display.columns: custom_cols.append('Value (PBR)')
        if 'Quality (ROE)' in df_display.columns: custom_cols.append('Quality (ROE)')
        if 'Momentum (Return)' in df_display.columns: custom_cols.append('Momentum (Return)')
        if 'Investment (Asset Growth)' in df_display.columns: custom_cols.append('Investment (Asset Growth)')
        
        if 'Size (MktCap)' in df_display.columns: custom_cols.append('Size (MktCap)')
        elif 'Size (Log)' in df_display.columns: custom_cols.append('Size (Log)')
        
        # 最終表示
        final_cols = base_cols + custom_cols
        
        # 【修正】将来のStreamlitバージョンエラーを防ぐため use_container_width=True を指定
        st.dataframe(
            df_display[final_cols].style.format({'Weight': '{:.1%}'}),
            use_container_width=True,
            hide_index=True
        )

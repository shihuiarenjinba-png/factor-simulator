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
        color: #666;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #007bff;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
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
    """
    入力テキストを解析し、{Ticker: Weight} の辞書を返す
    """
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

# Note: generate_insights 関数は削除し、QuantEngineのメソッドを使用します

# ---------------------------------------------------------
# 3. UI レイアウト & 入力
# ---------------------------------------------------------
st.sidebar.header("📊 Settings")

bench_mode = st.sidebar.selectbox("Benchmark Index", ["Nikkei 225", "TOPIX Core 30"])

if bench_mode == "Nikkei 225":
    benchmark_etf = "1321.T"
    universe_tickers = NIKKEI_225_SAMPLE
else:
    benchmark_etf = "1306.T"
    universe_tickers = TOPIX_CORE_30

st.sidebar.markdown("---")

st.sidebar.subheader("My Portfolio")
st.sidebar.caption("Format: `Ticker` or `Ticker:Weight`")

default_input = "7203.T: 40, 6758.T: 30, 9984.T: 30"
input_text = st.sidebar.text_area("Input", default_input, height=120)

run_btn = st.sidebar.button("Run Analysis", type="primary")

# ---------------------------------------------------------
# 4. メイン処理フロー
# ---------------------------------------------------------
if run_btn:
    st.title("🛡️ Market Factor Lab (Pro)")
    
    # [Step 1] 入力解析
    portfolio_dict = parse_portfolio_input(input_text)
    user_tickers = list(portfolio_dict.keys())
    
    if not user_tickers:
        st.warning("銘柄を入力してください。")
        st.stop()
        
    # [Step 2] データ取得 & 市場統計作成
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. ベンチマークデータの取得
    status_text.text(f"Fetching Market Data ({bench_mode})...")
    
    df_bench_fund = DataProvider.fetch_fundamentals(universe_tickers)
    df_bench_hist = DataProvider.fetch_historical_prices(universe_tickers + [benchmark_etf])
    
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
    df_user_fund = DataProvider.fetch_fundamentals(user_tickers)
    df_user_hist = DataProvider.fetch_historical_prices(user_tickers + [benchmark_etf])
    
    # 4. ユーザーデータの計算
    df_user_fund = QuantEngine.calculate_beta_momentum(df_user_fund, df_user_hist, benchmark_etf)
    
    # 生データ加工
    df_user_proc = QuantEngine.process_raw_factors(df_user_fund)
    
    # --- 【修正】直交化ロジックの削除 ---
    # 以前ここにあった apply_ortho 関数とその適用処理は、
    # 次の compute_z_scores 内で自動的に行われるため不要になりました。
    
    # Zスコア計算 (ここで内部的に直交化も実行されます)
    df_scored, r_squared_map = QuantEngine.compute_z_scores(df_user_proc, market_stats)
    
    # ウェイト情報をマージ
    df_scored['Weight'] = df_scored['Ticker'].map(portfolio_dict)
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    # -----------------------------------------------------
    # [Step 4] 結果表示
    # -----------------------------------------------------
    
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
    valid_beta = df_user_fund.dropna(subset=['Beta_Raw'])
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
    
    # Quality Score
    qual_score = portfolio_exposure.get('Quality', 0)
    q_color = "green" if qual_score > 0 else "red"
    col2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Quality Score</div>
        <div class="metric-value" style="color:{q_color}">{qual_score:.2f} σ</div>
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
        # 【修正】QuantEngineのメソッドを使用
        insights = QuantEngine.generate_insights(portfolio_exposure)
        for msg in insights:
            st.markdown(f'<div class="insight-box">{msg}</div>', unsafe_allow_html=True)
        st.info("※ Sizeは反転しています (＋方向 = 小型株効果)")

    # --- Data Table ---
    with st.expander("Show Detailed Factor Data", expanded=True):
        disp_cols = ['Ticker', 'Name', 'Weight'] + z_cols
        # 表示用フォーマット
        format_dict = {col: "{:.2f}" for col in z_cols}
        format_dict['Weight'] = "{:.1%}"
        
        def color_z(val):
            try:
                v = float(val)
                if v > 1.0: return 'background-color: #d4edda; color: black'
                if v < -1.0: return 'background-color: #f8d7da; color: black'
            except: pass
            return ''
            
        st.dataframe(
            df_scored[disp_cols].style.applymap(color_z, subset=z_cols).format(format_dict),
            use_container_width=True
        )

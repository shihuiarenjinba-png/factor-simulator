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

# カスタムCSS (カードデザインとフォント調整)
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

# ベンチマーク定義 (日経225採用銘柄の一部 + ETF)
NIKKEI_225_SAMPLE = [
    "7203.T", "6758.T", "8035.T", "9984.T", "9983.T", "6098.T", "4063.T", "6367.T", "9432.T", "4502.T",
    "4503.T", "6501.T", "7267.T", "8058.T", "8001.T", "6954.T", "6981.T", "9020.T", "9022.T", "7741.T",
    "5108.T", "4452.T", "6902.T", "7974.T", "8031.T", "4519.T", "4568.T", "6273.T", "4543.T", "6702.T",
    "6503.T", "4901.T", "4911.T", "2502.T", "2802.T", "3382.T", "8306.T", "8316.T", "8411.T", "8766.T",
    "8591.T", "8801.T", "8802.T", "9021.T", "9101.T", "9433.T", "9434.T", "9501.T", "9502.T"
]

# ---------------------------------------------------------
# 1. ヘルパー関数 (Beta計算 & インサイト生成)
# ---------------------------------------------------------
def calculate_beta_momentum(tickers, benchmark_ticker="1321.T"):
    """
    時系列データからBetaとMomentumを計算する
    """
    needed_tickers = list(set(tickers + [benchmark_ticker]))
    df_hist = DataProvider.fetch_historical_prices(needed_tickers, days=365)
    
    betas = {}
    momenta = {}
    
    if df_hist.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    rets = df_hist.pct_change().dropna()
    if benchmark_ticker not in rets.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    bench_ret = rets[benchmark_ticker]
    bench_var = bench_ret.var()

    for t in tickers:
        if t in rets.columns:
            # Beta
            try:
                cov = rets[t].cov(bench_ret)
                betas[t] = cov / bench_var if bench_var > 0 else 1.0
            except:
                betas[t] = 1.0
            # Momentum (12M Return)
            try:
                p_start = df_hist[t].iloc[0]
                p_end = df_hist[t].iloc[-1]
                momenta[t] = (p_end / p_start) - 1 if p_start > 0 else 0.0
            except:
                momenta[t] = 0.0
        else:
            betas[t] = np.nan
            momenta[t] = np.nan
            
    return pd.Series(betas), pd.Series(momenta)

def generate_insights(z_scores):
    """
    Zスコアに基づいて日本語の診断メッセージを生成する
    """
    insights = []
    
    # 1. Size (大型 vs 小型)
    # 反転済み: プラス=小型, マイナス=大型
    size_z = z_scores.get('Size', 0)
    if size_z < -1.0:
        insights.append("✅ **大型株中心**: 財務基盤が安定した大型株への配分が高く、市場変動に対する耐久性が期待できます。")
    elif size_z > 1.0:
        insights.append("🚀 **小型株効果**: 時価総額の小さい銘柄が多く、市場平均を上回る成長ポテンシャルを秘めています。")
        
    # 2. Value (割安 vs 割高)
    value_z = z_scores.get('Value', 0)
    if value_z > 1.0:
        insights.append("💰 **バリュー投資**: 純資産に対して割安な銘柄が多く、下値リスクが限定的である可能性があります。")
        
    # 3. Quality (高収益 vs 低収益)
    qual_z = z_scores.get('Quality', 0)
    if qual_z > 1.0:
        insights.append("💎 **高クオリティ**: ROE等の収益性が市場平均より高く、経営効率の良い企業群です。")
        
    # 4. Momentum (順張り vs 逆張り)
    mom_z = z_scores.get('Momentum', 0)
    if mom_z < -1.0:
        insights.append("🔄 **リバーサル狙い**: 直近で株価が出遅れている銘柄が多く、反発（見直し買い）を狙う構成です。")
    elif mom_z > 1.0:
        insights.append("📈 **モメンタム重視**: 直近の株価パフォーマンスが良い銘柄に乗る「順張り」の傾向があります。")

    if not insights:
        insights.append("⚖️ **市場中立**: 特定のファクターへの極端な偏りがなく、市場全体（インデックス）に近いバランスです。")
        
    return insights

# ---------------------------------------------------------
# 2. UI レイアウト & 入力
# ---------------------------------------------------------
st.sidebar.header("📊 Settings")

# ベンチマーク
benchmark_etf = "1321.T"
universe_tickers = NIKKEI_225_SAMPLE

# ポートフォリオ入力
st.sidebar.subheader("My Portfolio")
default_input = "7203.T, 9984.T, 6758.T, 8035.T"
input_text = st.sidebar.text_area("Tickers (comma separated)", default_input, height=100)

run_btn = st.sidebar.button("Run Analysis", type="primary")

# ---------------------------------------------------------
# 3. メイン処理フロー
# ---------------------------------------------------------
if run_btn:
    st.title("🛡️ Market Factor Lab (Pro)")
    
    # [Step 1] 入力解析
    user_tickers = [x.strip() for x in input_text.split(',') if x.strip()]
    if not user_tickers:
        st.warning("銘柄を入力してください。")
        st.stop()

    # [Step 2] データ取得 & 市場統計作成 (Benchmark Construction)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Fetching Market Data...")
    df_bench_fund = DataProvider.fetch_fundamentals(universe_tickers)
    progress_bar.progress(20)
    
    status_text.text("Calculating Market Beta & Momentum...")
    s_beta_bench, s_mom_bench = calculate_beta_momentum(universe_tickers, benchmark_etf)
    df_bench_fund['Beta_Raw'] = df_bench_fund['Ticker'].map(s_beta_bench)
    df_bench_fund['Momentum_Raw'] = df_bench_fund['Ticker'].map(s_mom_bench)
    progress_bar.progress(40)
    
    status_text.text("Generating Robust Statistics (Universe Manager)...")
    market_stats, df_bench_processed = UniverseManager.generate_market_stats(df_bench_fund)
    progress_bar.progress(60)

    # [Step 3] ユーザーポートフォリオ評価 (User Scoring)
    status_text.text("Analyzing Your Portfolio...")
    df_user_fund = DataProvider.fetch_fundamentals(user_tickers)
    s_beta_user, s_mom_user = calculate_beta_momentum(user_tickers, benchmark_etf)
    df_user_fund['Beta_Raw'] = df_user_fund['Ticker'].map(s_beta_user)
    df_user_fund['Momentum_Raw'] = df_user_fund['Ticker'].map(s_mom_user)
    
    # 生データ加工 (Log化など)
    df_user_proc = QuantEngine.process_raw_factors(df_user_fund)
    
    # 直交化 (ユーザーデータの補正)
    slope = market_stats['ortho_slope']
    intercept = market_stats['ortho_intercept']
    def apply_ortho(row):
        q = row.get('Quality_Metric', np.nan)
        i = row.get('Investment_Metric', np.nan)
        if pd.isna(q): return np.nan
        if pd.isna(i): return q
        return q - (slope * i + intercept)
    df_user_proc['Quality_Orthogonal'] = df_user_proc.apply(apply_ortho, axis=1)

    # Zスコア計算 (市場基準との比較)
    df_scored, r_squared_map = QuantEngine.compute_z_scores(df_user_proc, market_stats)
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    # -----------------------------------------------------
    # [Step 4] 結果表示 (Visualization)
    # -----------------------------------------------------
    
    # 全体ウェイト (現状は均等配分と仮定)
    total_weight = 1.0 / len(df_scored)
    
    # ポートフォリオ全体のZスコア平均を算出
    z_cols = [c for c in df_scored.columns if c.endswith('_Z')]
    portfolio_exposure = {}
    
    for col in z_cols:
        # Zスコアの単純平均 (本来はウェイト加重平均推奨)
        score = df_scored[col].mean()
        factor_name = col.replace('_Z', '')
        portfolio_exposure[factor_name] = score

    # --- Layout: Top KPI Cards ---
    st.subheader("📊 Portfolio Diagnostic")
    
    col1, col2, col3 = st.columns(3)
    
    # Beta (平均)
    avg_beta = df_user_fund['Beta_Raw'].mean()
    col1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Beta (Risk)</div>
        <div class="metric-value">{avg_beta:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quality Z-Score
    qual_score = portfolio_exposure.get('Quality', 0)
    q_color = "green" if qual_score > 0 else "red"
    col2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Quality Score</div>
        <div class="metric-value" style="color:{q_color}">{qual_score:.2f} σ</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Share (簡易: 銘柄数で表現)
    col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Holdings</div>
        <div class="metric-value">{len(user_tickers)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- Layout: Main Chart & Insights ---
    c_chart, c_insight = st.columns([2, 1])
    
    with c_chart:
        st.subheader("Factor Exposure (vs Market Natural)")
        
        # グラフ用データ作成
        factors = list(portfolio_exposure.keys())
        scores = list(portfolio_exposure.values())
        
        # R²の表示用テキスト作成
        y_labels = []
        for f in factors:
            r2 = r_squared_map.get(f)
            if r2 is not None:
                # ファクター名に R² を添える
                y_labels.append(f"{f} (R²: {r2:.2f})")
            else:
                y_labels.append(f)
        
        # Plotly Bar Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scores,
            y=y_labels,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='RdBu',
                cmin=-2, cmax=2
            ),
            text=[f"{s:.2f}" for s in scores],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Portfolio Z-Scores (0 = Market Benchmark)",
            xaxis_title="Standard Deviation (σ)",
            yaxis=dict(autorange="reversed"), # 上から順に表示
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # 基準線 (0)
        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)

    with c_insight:
        st.subheader("AI Insight")
        
        insights = generate_insights(portfolio_exposure)
        
        for msg in insights:
            st.markdown(f'<div class="insight-box">{msg}</div>', unsafe_allow_html=True)
            
        st.info("※ Sizeは反転しています (＋方向 = 小型株効果)")

    # --- Layout: Data Table ---
    with st.expander("Show Detailed Factor Data", expanded=True):
        # 表示用に整理
        disp_cols = ['Ticker', 'Name'] + z_cols
        
        # スタイリング
        def color_z(val):
            try:
                v = float(val)
                if v > 1.0: return 'background-color: #d4edda; color: black'
                if v < -1.0: return 'background-color: #f8d7da; color: black'
                return ''
            except:
                return ''
                
        st.dataframe(
            df_scored[disp_cols].style.applymap(color_z, subset=z_cols).format("{:.2f}", subset=z_cols),
            use_container_width=True
        )

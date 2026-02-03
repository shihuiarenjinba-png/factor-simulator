import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import requests
import io

# 🔗 Brain（計算エンジン）を読み込む
from simulation_engine import MarketDataEngine, PortfolioAnalyzer, PortfolioDiagnosticEngine

# =========================================================
# ⚙️ 自動セットアップ & 定数定義
# =========================================================
def ensure_japanese_font():
    """日本語フォントがなければ自動ダウンロードする関数"""
    font_filename = 'IPAexGothic.ttf'
    if not os.path.exists(font_filename):
        url = "https://github.com/minoryorg/ipaex-font/raw/master/ipaexg.ttf"
        try:
            with st.spinner('📥 初回セットアップ中: 日本語フォントをダウンロードしています...'):
                response = requests.get(url)
                with open(font_filename, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            st.warning(f"⚠️ フォント取得失敗 (英語モードで動作します): {e}")

ensure_japanese_font()

# 🎨 V17.1 プロフェッショナル・カラーパレット (Updated)
COLORS = {
    'main': '#00FFFF',      # ネオンシアン (線・強調用)
    'benchmark': '#FF69B4', # ホットピンク
    'principal': '#FFFFFF', # ホワイト
    'median': '#32CD32',    # ライムグリーン
    'mean': '#FFD700',      # ゴールド (新規: 平均値用)
    'p10': '#FF6347',       # 悲観シナリオ
    'p90': '#00BFFF',       # 楽観シナリオ
    'hist_bar': '#42A5F5',  # 教科書的な中間青 (新規: 視認性と美観のバランス)
    'cost_net': '#FF6347',  # トマトレッド
    'bg_fill': 'rgba(0, 255, 255, 0.1)'
}

# =========================================================
# ⚙️ システム設定
# =========================================================
st.set_page_config(page_title="Factor Simulator V17.1", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    .metric-card { background-color: #262730; border: 1px solid #444; padding: 15px; border-radius: 8px; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #00FFFF; color: black; font-weight: bold; }
    .report-box { border-left: 5px solid #00FFFF; padding-left: 15px; margin-top: 10px; background-color: rgba(0, 255, 255, 0.05); }
    .factor-box { border-left: 5px solid #FF69B4; padding-left: 15px; margin-top: 10px; background-color: rgba(255, 105, 180, 0.05); }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Factor & Stress Test Simulator V17.1")
st.caption("プロフェッショナル版: ポートフォリオ診断・モンテカルロ・リスク分析")

# =========================================================
# 🖥️ UI & メインロジック
# =========================================================

if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None

# --- サイドバー設定 ---
with st.sidebar:
    st.header("⚙️ 設定パネル")

    st.markdown("### 1. ポートフォリオ構成")
    
    # 🔥 [NEW] CSVアップロード機能
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'], help="列名: 'Ticker', 'Weight' のCSVを使用してください")
    
    default_input = "SPY: 40, VWO: 20, 7203.T: 20, GLD: 20"
    
    # CSVがアップロードされたら、それをテキストエリアの初期値として整形する
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            # 1列目と2列目を使用する（列名が何であれ）
            if df_upload.shape[1] >= 2:
                tickers_up = df_upload.iloc[:, 0].astype(str)
                weights_up = df_upload.iloc[:, 1].astype(str)
                # "Ticker: Weight" の形式に変換
                formatted_list = [f"{t}: {w}" for t, w in zip(tickers_up, weights_up)]
                default_input = ", ".join(formatted_list)
                st.success("✅ CSV読み込み完了")
            else:
                st.error("CSVは少なくとも2列（銘柄, 比率）必要です。")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

    input_text = st.text_area("Ticker: Weight (直接入力可)", value=default_input, height=150)

    st.markdown("### 2. 分析モデル & ベンチマーク")
    target_region = st.selectbox("分析リージョン", ["US (米国)", "Japan (日本)", "Global (全世界)"], index=0)

    region_code = target_region.split()[0]
    bench_options = {
        'US': {'S&P 500 (^GSPC)': '^GSPC', 'NASDAQ 100 (^NDX)': '^NDX'},
        'Japan': {'TOPIX (1306 ETF)': '1306.T', '日経平均 (^N225)': '^N225'},
        'Global': {'VT (全世界株ETF)': 'VT', 'MSCI ACWI (指数)': 'ACWI'}
    }

    selected_bench_label = st.selectbox("比較ベンチマーク", list(bench_options[region_code].keys()) + ["Custom (自由入力)"])

    if selected_bench_label == "Custom (自由入力)":
        bench_ticker = st.text_input("ベンチマークTicker", value="^GSPC")
    else:
        bench_ticker = bench_options[region_code][selected_bench_label]

    st.markdown("### 3. コスト設定")
    cost_tier = st.select_slider("運用コスト", options=["Low", "Medium", "High"], value="Medium")

    analyze_btn = st.button("🚀 分析開始", type="primary", use_container_width=True)

# --- メイン処理 ---
if analyze_btn:
    with st.spinner("⏳ データ取得・7500回の未来シミュレーション計算中..."):
        # 1. ポートフォリオ解析
        raw_items = [item.strip() for item in input_text.split(',')]
        parsed_dict = {}
        for item in raw_items:
            try:
                k, v = item.split(':')
                parsed_dict[k.strip()] = float(v.strip())
            except: pass

        if not parsed_dict: st.stop()

        # 🚀 Brainクラスを呼び出す
        engine = MarketDataEngine()
        valid_assets, _ = engine.validate_tickers(parsed_dict)
        if not valid_assets: st.stop()

        tickers = list(valid_assets.keys())
        hist_returns = engine.fetch_historical_prices(tickers)

        weights_clean = {k: v['weight'] for k, v in valid_assets.items()}
        port_series, final_weights = PortfolioAnalyzer.create_synthetic_history(hist_returns, weights_clean)

        # 2. ベンチマーク取得
        is_jpy_bench = True if bench_ticker in ['^TPX', '^N225', '1306.T'] or bench_ticker.endswith('.T') else False
        bench_series = engine.fetch_benchmark_data(bench_ticker, is_jpy_asset=is_jpy_bench)

        # 3. ファクター取得
        french_factors = engine.fetch_french_factors(region_code)

        st.session_state.portfolio_data = {
            'returns': port_series,
            'benchmark': bench_series,
            'components': hist_returns,
            'weights': final_weights,
            'factors': french_factors,
            'asset_info': valid_assets,
            'cost_tier': cost_tier,
            'bench_name': selected_bench_label
        }

if st.session_state.portfolio_data:
    data = st.session_state.portfolio_data
    analyzer = PortfolioAnalyzer()
    port_ret = data['returns']
    bench_ret = data['benchmark']

    # 基本指標
    total_ret_cum = (1 + port_ret).cumprod()
    cagr = (total_ret_cum.iloc[-1])**(12/len(port_ret)) - 1
    vol = port_ret.std() * np.sqrt(12)
    max_dd = (total_ret_cum / total_ret_cum.cummax() - 1).min()

    # プロ指標計算
    calmar = analyzer.calculate_calmar_ratio(port_ret)
    omega = analyzer.calculate_omega_ratio(port_ret, threshold=0.0) 
    info_ratio, track_err = analyzer.calculate_information_ratio(port_ret, bench_ret)

    # --- ダッシュボード表示 ---
    st.markdown("---")

    # メトリクス行
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR (年率)", f"{cagr:.2%}")
    c2.metric("Vol (リスク)", f"{vol:.2%}")
    c3.metric("Max DD", f"{max_dd:.2%}", delta_color="inverse")
    c4.metric("Calmar Ratio", f"{calmar:.2f}", help="年率リターン ÷ 最大DD。0.5以上で優秀。")
    c5.metric("Omega Ratio (0%)", f"{omega:.2f}", help="勝ちの面積 ÷ 負けの面積。1.0以上で勝ち越し。")

    if not np.isnan(info_ratio):
        st.caption(f"📊 vs {data['bench_name']} | Information Ratio: **{info_ratio:.2f}** (Tracking Error: {track_err:.2%})")

    # タブ構成
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🧬 DNA解析", "🌊 ファクター", "⏳ タイムマシン", "💸 コスト", "🏆 寄与度", "🔮 未来予測"])

    # -----------------------------------------------------
    # Tab 1: DNA解析
    # -----------------------------------------------------
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("分散の「質」を可視化")
            pca_ratio, _ = analyzer.perform_pca(data['components'])
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = pca_ratio * 100, 
                title = {'text': "第1主成分の支配率 (%)"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['main']},
                         'steps': [{'range': [0, 60], 'color': "#333"}, {'range': [60, 100], 'color': "#555"}],
                         'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("資産配分")
            fig_pie = px.pie(values=list(data['weights'].values()), names=list(data['weights'].keys()), hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("🩺 ポートフォリオ診断")
            report = PortfolioDiagnosticEngine.generate_report(data['weights'], pca_ratio, port_ret)
            st.markdown(f"""
            <div class="report-box">
                <h3 style="color: #00FFFF; margin-bottom:0px;">{report['type']}</h3>
                <hr style="margin-top:5px; margin-bottom:10px; border-color: #555;">
                <p><b>🧐 現状分析:</b><br>{report['diversification_comment']}</p>
                <p><b>⚠️ リスク警告:</b><br>{report['risk_comment']}</p>
                <p><b>💡 次のアクション:</b><br>{report['action_plan']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("🔥 相関ヒートマップ")
            corr_matrix = analyzer.calculate_correlation_matrix(data['components'])
            if not corr_matrix.empty:
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto='.2f', 
                                    aspect="auto", 
                                    color_continuous_scale='RdBu_r', 
                                    zmin=-1, zmax=1,
                                    title="銘柄間の相関係数 (-1: 逆相関, +1: 正相関)")
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption("ℹ️ **赤色 (+1.0)** は同じ動き、**青色 (-1.0)** は逆の動きをします。青色が混ざっているほど、リスク分散効果が高いことを示します。")

    # -----------------------------------------------------
    # Tab 2: ファクター動向
    # -----------------------------------------------------
    with tab2:
        if data['factors'].empty:
            st.error("🚫 ファクターデータの取得に失敗しました。")
        else:
            st.subheader("📊 ポートフォリオのスタイル診断 (回帰分析)")
            params, r_sq = analyzer.perform_factor_regression(port_ret, data['factors'])
            
            if params is not None:
                c1, c2 = st.columns([1, 1])
                with c1:
                    beta_df = params.drop('const') if 'const' in params else params
                    colors = ['#00CC96' if x > 0 else '#FF4B4B' for x in beta_df.values]
                    
                    fig_beta = go.Figure(go.Bar(
                        x=beta_df.values, y=beta_df.index, orientation='h', 
                        marker_color=colors, text=[f"{x:.2f}" for x in beta_df.values], textposition='auto'
                    ))
                    fig_beta.update_layout(title="ファクター感応度 (Beta)", xaxis_title="感応度 (正=順相関, 負=逆相関)", height=300)
                    st.plotly_chart(fig_beta, use_container_width=True)
                    st.caption(f"決定係数 (R²): {r_sq:.2%} (このモデルで動きの{r_sq*100:.0f}%を説明できます)")
                
                with c2:
                    commentary = PortfolioDiagnosticEngine.generate_factor_report(params)
                    st.markdown(f"""
                    <div class="factor-box">
                        <h4 style="color: #FF69B4; margin-bottom:10px;">🧠 AIスタイル分析</h4>
                        <div style="white-space: pre-wrap;">{commentary}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📈 市場性格の「変節」を追う (Rolling Beta)")
            
            rolling_betas = analyzer.rolling_beta_analysis(port_ret, data['factors'])
            if rolling_betas.empty:
                st.warning("⚠️ データ期間不足のため分析できません。")
            else:
                fig_roll = go.Figure()
                if 'Mkt-RF' in rolling_betas.columns: fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['Mkt-RF'], name='市場連動 (Beta)', line=dict(width=3, color=COLORS['main'])))
                if 'SMB' in rolling_betas.columns: fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['SMB'], name='サイズ (SMB)', line=dict(dash='dot', color='orange')))
                if 'HML' in rolling_betas.columns: fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['HML'], name='バリュー (HML)', line=dict(dash='dot', color='yellow')))
                st.plotly_chart(fig_roll, use_container_width=True)

    # -----------------------------------------------------
    # Tab 3: タイムマシン
    # -----------------------------------------------------
    with tab3:
        st.subheader("ヒストリカル・ストレステスト")
        cum_ret = (1 + port_ret).cumprod() * 10000

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=cum_ret.index, y=[10000]*len(cum_ret), mode='lines', name='元本 (10,000)', line=dict(color=COLORS['principal'], width=1, dash='dot')))

        if not bench_ret.empty:
            bench_cum = (1 + bench_ret).cumprod()
            common_idx = cum_ret.index.intersection(bench_cum.index)
            bench_cum = bench_cum.loc[common_idx]
            bench_cum = bench_cum / bench_cum.iloc[0] * 10000
            fig_hist.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, mode='lines', name=f"Benchmark ({data['bench_name']})", line=dict(color=COLORS['benchmark'], width=1.5)))

        fig_hist.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, fill='tozeroy', fillcolor=COLORS['bg_fill'], mode='lines', name='My Portfolio', line=dict(color=COLORS['main'], width=2.5)))
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 リターン分布解析 (ヒストグラム)")

        mu, std = port_ret.mean(), port_ret.std()
        fig_dist = go.Figure()
        
        # 🎨 [UPDATE] ヒストグラムの色を教科書的な中間青に変更
        fig_dist.add_trace(go.Histogram(x=port_ret, histnorm='probability density', name='実績分布', marker_color=COLORS['hist_bar'], opacity=0.8, nbinsx=50))

        x_range = np.linspace(port_ret.min(), port_ret.max(), 100)
        y_norm = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x_range - mu) / std) ** 2)
        fig_dist.add_trace(go.Scatter(x=x_range, y=y_norm, mode='lines', name='正規分布 (理論値)', line=dict(color='white', dash='dash', width=2)))

        fig_dist.update_layout(xaxis_title="月次リターン", yaxis_title="確率密度", hovermode="x", barmode='overlay', margin=dict(t=30, b=30), height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.info(PortfolioDiagnosticEngine.get_skew_kurt_desc(port_ret))

    # -----------------------------------------------------
    # Tab 4: コスト診断
    # -----------------------------------------------------
    with tab4:
        st.subheader("コストドラッグ診断")
        gross, net, loss, cost_pct = analyzer.cost_drag_simulation(port_ret, data['cost_tier'])
        loss_amount = 1000000 * loss
        final_amount_net = 1000000 * net.iloc[-1]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(x=gross.index, y=gross, name='コストなし (理想)', line=dict(color='gray', dash='dot')))
            fig_cost.add_trace(go.Scatter(x=net.index, y=net, name=f'コストあり (現実)', fill='tonexty', line=dict(color=COLORS['cost_net'])))
            st.plotly_chart(fig_cost, use_container_width=True)
        with c2:
            st.error(f"💸 累積損失インパクト: ▲{loss_amount:,.0f} 円")
            st.markdown(f"100万円投資時の最終評価額: **{final_amount_net:,.0f} 円**")

    # -----------------------------------------------------
    # Tab 5: 寄与度分析
    # -----------------------------------------------------
    with tab5:
        st.subheader("銘柄別 厳密寄与度分析")
        attrib = analyzer.calculate_strict_attribution(data['components'], data['weights'])
        if not attrib.empty:
            colors = ['#FF4B4B' if x < 0 else '#00CC96' for x in attrib.values]
            fig_attr = go.Figure(go.Bar(
                x=attrib.values, y=attrib.index, orientation='h', marker_color=colors,
                text=[f"{x:.2%}" for x in attrib.values], textposition='auto'
            ))
            fig_attr.update_layout(xaxis_title="寄与度", yaxis_title="銘柄")
            st.plotly_chart(fig_attr, use_container_width=True)

    # -----------------------------------------------------
    # Tab 6: 🔮 未来予測 (モンテカルロ・7500回・Fat-Tail)
    # -----------------------------------------------------
    with tab6:
        st.subheader("🎲 モンテカルロ・シミュレーション (7,500回 / Fat-Tail Model)")
        st.caption("正規分布よりも極端な値動きが発生しやすい「t分布（自由度6）」を用いた、プロ仕様の厳格なストレステストです。")

        # シミュレーション実行 (Brain内で7500回計算済み)
        sim_years = 20
        init_inv = 1000000
        df_stats, final_values = analyzer.run_monte_carlo_simulation(port_ret, n_years=sim_years, n_simulations=7500, initial_investment=init_inv)

        if df_stats is not None:
            # 1. コーンチャート
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p50'], mode='lines', name='中央値 (標準)', line=dict(color=COLORS['median'], width=3)))
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p10'], mode='lines', name='下位10% (悲観)', line=dict(color=COLORS['p10'], width=1, dash='dot')))
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p90'], mode='lines', name='上位10% (楽観)', line=dict(color=COLORS['p90'], width=1, dash='dot')))
            fig_mc.update_layout(title=f"今後{sim_years}年間の資産推移予測 (元本: {init_inv:,}円)", yaxis_title="評価額 (円)", height=500)
            st.plotly_chart(fig_mc, use_container_width=True)

            # 2. 最終結果のヒストグラム & 統計
            st.markdown("### 🏁 20年後の資産分布 (ヒストグラム)")
            
            # 統計量の計算
            final_median = np.median(final_values)
            final_mean = np.mean(final_values)
            final_p10 = np.percentile(final_values, 10)
            final_p90 = np.percentile(final_values, 90)
            
            # グラフが見やすいように表示範囲を計算 (98パーセンタイルまで)
            x_max_view = np.percentile(final_values, 98)

            # ヒストグラムの最大頻度を計算 (ラベルの高さ調整の基準にするため)
            counts, _ = np.histogram(final_values, bins=100)
            y_max_freq = counts.max()

            # カラムで統計表示
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("下位10% (悲観)", f"{final_p10:,.0f} 円", delta_color="inverse")
            mc2.metric("中央値 (Most Likely)", f"{final_median:,.0f} 円")
            mc3.metric("平均値 (Expected)", f"{final_mean:,.0f} 円")
            mc4.metric("上位10% (楽観)", f"{final_p90:,.0f} 円")

            # ヒストグラム描画
            fig_mc_hist = go.Figure()
            
            # 🎨 [UPDATE] ヒストグラムの色を教科書的な中間青に変更
            fig_mc_hist.add_trace(go.Histogram(
                x=final_values, nbinsx=100, name='頻度', 
                marker_color=COLORS['hist_bar'], opacity=0.85
            ))
            
            # 🎨 [UPDATE] 指標ラベルに高低差をつける設定
            # 構成: (値, 色, ラベル, 高さ倍率, 線種, 太さ)
            # MedianとMeanが近い場合でも重ならないよう、高さを変えています
            lines_config = [
                (final_p10, COLORS['p10'], "P10", 1.05, "dash", 2),
                (final_median, COLORS['median'], "Median", 1.15, "solid", 3), # 中央値は少し高く
                (final_mean, COLORS['mean'], "Mean", 1.25, "dash", 3),        # 平均値はさらに高く
                (final_p90, COLORS['p90'], "P90", 1.05, "dash", 2),
            ]

            for val, color, label, h_rate, dash, width in lines_config:
                # 垂直線を描画
                fig_mc_hist.add_vline(x=val, line_width=width, line_dash=dash, line_color=color)
                
                # ラベルを配置 (add_vlineのannotation機能ではなく、座標指定で高さをコントロール)
                fig_mc_hist.add_annotation(
                    x=val, y=y_max_freq * h_rate,
                    text=label, showarrow=False,
                    font=dict(color=color, size=13, weight='bold'),
                    xanchor='left', yanchor='bottom'
                )

            # レイアウト調整 (ラベルが見切れないようY軸の上限を拡張)
            fig_mc_hist.update_layout(
                xaxis_title="最終資産額 (円)", 
                yaxis_title="発生回数", 
                showlegend=False,
                xaxis=dict(range=[0, x_max_view]),
                yaxis=dict(range=[0, y_max_freq * 1.4]) # 上部に十分な余白を確保
            )
            st.plotly_chart(fig_mc_hist, use_container_width=True)
            
            st.success(f"✅ 計算完了: 業界標準(5,000回)を超える **7,500回** のシナリオ生成に成功しました。")

else:
    st.info("👈 サイドバーから設定を行い、分析を開始してください")

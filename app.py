import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import unicodedata  # 追加: 全角・半角変換用
from sklearn.decomposition import PCA

# 将来の警告を無視する設定
warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================================================
# 🔗 モジュール読み込みチェック
# =========================================================
try:
    from logic_engine import MarketDataEngine, PortfolioAnalyzer, PortfolioDiagnosticEngine
    from pdf_generator import create_pdf_report
except ImportError as e:
    st.error(f"❌ 重要ファイルが見つかりません: {e}")
    st.info("app.py と同じフォルダに 'simulation_engine.py' と 'pdf_generator.py' があるか確認してください。")
    st.stop()

# =========================================================
# ⚙️ 定数・設定
# =========================================================

# 🎨 カラーパレット
COLORS = {
    'main': '#00FFFF',      # Neon Cyan
    'benchmark': '#FF69B4', # Hot Pink
    'principal': '#FFFFFF', # White
    'median': '#32CD32',    # Lime Green
    'mean': '#FFD700',      # Gold
    'p10': '#FF6347',       # Pessimistic
    'p90': '#00BFFF',       # Optimistic
    'hist_bar': '#42A5F5',  # Mid Blue
    'cost_net': '#FF6347',  # Tomato Red
    'bg_fill': 'rgba(0, 255, 255, 0.1)'
}

st.set_page_config(page_title="Factor Simulator V18.1 JP", layout="wide", page_icon="🧬")

# CSSスタイリング
st.markdown("""
<style>
    .metric-card { background-color: #262730; border: 1px solid #444; padding: 15px; border-radius: 8px; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #00FFFF; color: black; font-weight: bold; }
    .report-box { border-left: 5px solid #00FFFF; padding-left: 15px; margin-top: 10px; background-color: rgba(0, 255, 255, 0.05); }
    .factor-box { border-left: 5px solid #FF69B4; padding-left: 15px; margin-top: 10px; background-color: rgba(255, 105, 180, 0.05); }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    h1, h2, h3 { color: #E0E0E0; font-family: 'Hiragino Kaku Gothic Pro', 'Meiryo', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Factor & Stress Test Simulator V18.1")
st.caption("Professional Edition: ポートフォリオ診断・モンテカルロ分析・リスク管理 (日本語版)")

# =========================================================
# 🛠️ セッション状態の初期化
# =========================================================
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None
if 'payload' not in st.session_state:
    st.session_state.payload = None
if 'figs' not in st.session_state:
    st.session_state.figs = {}

# =========================================================
# 🏗️ サイドバー: ポートフォリオ設定
# =========================================================
with st.sidebar:
    st.header("⚙️ 設定パネル")

    st.markdown("### 1. ポートフォリオ構成")
    
    uploaded_file = st.file_uploader("CSVをアップロード", type=['csv'], help="必須列: 'Ticker', 'Weight'")
    
    default_input = "SPY: 40, VWO: 20, 7203.T: 20, GLD: 20"
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if df_upload.shape[1] >= 2:
                tickers_up = df_upload.iloc[:, 0].astype(str)
                weights_up = df_upload.iloc[:, 1].astype(str)
                formatted_list = [f"{t}: {w}" for t, w in zip(tickers_up, weights_up)]
                default_input = ", ".join(formatted_list)
                st.success("✅ CSV読み込み完了")
            else:
                st.error("CSVには少なくとも2列（ティッカー, 比率）が必要です。")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

    input_text = st.text_area("ティッカー: 比率 (入力)", value=default_input, height=100)

    st.markdown("### 2. 分析モデル & ベンチマーク")
    
    # ユーザーがリージョンを変更すると、Streamlitは再描画し、下のbench_optionsからデフォルト（index=0）を取得します
    target_region = st.selectbox("分析対象地域", ["US (米国)", "Japan (日本)", "Global (全世界)"], index=0)
    region_code = target_region.split()[0]
    
    # ベンチマーク辞書の定義
    bench_options = {
        'US': {'S&P 500 (^GSPC)': '^GSPC', 'NASDAQ 100 (^NDX)': '^NDX'},
        'Japan': {'TOPIX (1306 ETF)': '1306.T', '日経平均 (^N225)': '^N225'},
        'Global': {'VT (全世界株式)': 'VT', 'MSCI ACWI (指数)': 'ACWI'}
    }
    
    # リージョンに応じた選択肢リストを取得
    current_bench_options = list(bench_options[region_code].keys()) + ["Custom"]
    
    # index=0を指定することで、リージョン変更時にリストの先頭（標準ベンチマーク）に自動で切り替わるようにします
    selected_bench_label = st.selectbox("比較対象ベンチマーク", current_bench_options, index=0)

    if selected_bench_label == "Custom":
        bench_ticker = st.text_input("ベンチマークのティッカー", value="^GSPC")
    else:
        bench_ticker = bench_options[region_code][selected_bench_label]

    st.markdown("### 3. コスト設定")
    cost_tier = st.select_slider("信託報酬・管理コスト", options=["Low", "Medium", "High"], value="Medium")

    st.markdown("### 4. アドバイザーコメント")
    st.caption("✍️ PDFレポートの冒頭に掲載されるメッセージです。")
    
    default_note = "今回の分析結果に基づき、成長と安定のバランスを重視したこの配分を推奨します。リスク許容度に合わせて定期的なリバランスを行ってください。"
    advisor_note = st.text_area("クライアントへのメッセージ:", 
                                value=default_note,
                                height=100)

    st.markdown("---")
    analyze_btn = st.button("🚀 分析を開始する", type="primary", use_container_width=True)


# =========================================================
# 🚀 メインロジック (計算実行)
# =========================================================

if analyze_btn:
    # メモリ保護のため回数を5,000回に調整
    n_sims = 5000
    with st.spinner(f"⏳ データを取得し、{n_sims:,}回のシミュレーションを実行中..."):
        try:
            # 1. 入力解析 (堅牢化: 全角→半角変換、改行対応)
            # 正規化 (NFKC) で全角英数を半角に変換
            normalized_text = unicodedata.normalize('NFKC', input_text)
            # 改行をカンマに置換して、改行区切りでも動くようにする
            normalized_text = normalized_text.replace('\n', ',')
            
            raw_items = [item.strip() for item in normalized_text.split(',') if item.strip()]
            parsed_dict = {}
            error_lines = []
            
            for item in raw_items:
                try:
                    if ':' in item:
                        k, v = item.split(':')
                        parsed_dict[k.strip()] = float(v.strip())
                    elif ' ' in item: # コロンがない場合、スペース区切りも試行
                        parts = item.split()
                        if len(parts) >= 2:
                            parsed_dict[parts[0].strip()] = float(parts[1].strip())
                except:
                    error_lines.append(item)

            if error_lines:
                st.warning(f"⚠️ 読み取れなかった行があります (スキップしました): {', '.join(error_lines)}")

            if not parsed_dict:
                st.error("有効なデータが見つかりません。「ティッカー: 比率」の形式で入力してください。")
                st.stop()

            # 🚀 Engine 呼び出し
            engine = MarketDataEngine()
            valid_assets, _ = engine.validate_tickers(parsed_dict)
            if not valid_assets:
                st.error("有効なティッカーが1つも見つかりませんでした。入力コードを確認してください。")
                st.stop()

            tickers = list(valid_assets.keys())
            hist_returns = engine.fetch_historical_prices(tickers)

            if hist_returns.empty:
                 st.error("価格データの取得に失敗しました。")
                 st.stop()

            weights_clean = {k: v['weight'] for k, v in valid_assets.items()}
            port_series, final_weights = PortfolioAnalyzer.create_synthetic_history(hist_returns, weights_clean)

            # 2. ベンチマーク取得
            is_jpy_bench = True if bench_ticker in ['^TPX', '^N225', '1306.T'] or bench_ticker.endswith('.T') else False
            bench_series = engine.fetch_benchmark_data(bench_ticker, is_jpy_asset=is_jpy_bench)

            # 3. ファクター取得
            french_factors = engine.fetch_french_factors(region_code)

            # データ保存
            st.session_state.portfolio_data = {
                'returns': port_series,
                'benchmark': bench_series,
                'components': hist_returns,
                'weights': final_weights,
                'factors': french_factors,
                'asset_info': valid_assets,
                'cost_tier': cost_tier,
                'bench_name': selected_bench_label,
            }
            
            # 再計算時にキャッシュをクリア
            st.session_state.pdf_bytes = None
            st.session_state.analysis_done = False

        except Exception as e:
            st.error(f"分析エラーが発生しました: {e}")
            st.stop()


# =========================================================
# 📊 ダッシュボード表示 & PDF用データ準備
# =========================================================

if st.session_state.portfolio_data:
    data = st.session_state.portfolio_data
    analyzer = PortfolioAnalyzer()
    port_ret = data['returns']
    bench_ret = data['benchmark']

    # --- 1. 基本指標 ---
    total_ret_cum = (1 + port_ret).cumprod()
    cagr = (total_ret_cum.iloc[-1])**(12/len(port_ret)) - 1
    vol = port_ret.std() * np.sqrt(12)
    max_dd = (total_ret_cum / total_ret_cum.cummax() - 1).min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    try:
        omega = analyzer.calculate_omega_ratio(port_ret, threshold=0.0)
    except:
        omega = 0.0
        
    try:
        info_ratio, track_err = analyzer.calculate_information_ratio(port_ret, bench_ret)
    except:
        info_ratio, track_err = np.nan, np.nan

    sharpe_ratio = (cagr - 0.02) / vol # Simplified Sharpe

    # --- 2. 高度計算 ---
    params, r_sq = analyzer.perform_factor_regression(port_ret, data['factors'])
    if params is not None:
        factor_comment = PortfolioDiagnosticEngine.generate_factor_report(params)
    else:
        factor_comment = "ファクターデータが不足しており分析できません。"

    # モンテカルロ (クラウド環境用に5000回に設定)
    sim_years = 20
    init_inv = 1000000
    n_sims = 5000 
    df_stats, final_values = analyzer.run_monte_carlo_simulation(port_ret, n_years=sim_years, n_simulations=n_sims, initial_investment=init_inv)
    
    final_median = np.median(final_values)
    final_p10 = np.percentile(final_values, 10)
    final_p90 = np.percentile(final_values, 90)
    
    # 相関行列
    corr_matrix = analyzer.calculate_correlation_matrix(data['components'])
    fig_corr_report = None
    if not corr_matrix.empty:
        fig_corr_report = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)

    # AI診断 & PCA
    pca_ratio, _ = analyzer.perform_pca(data['components'])
    report = PortfolioDiagnosticEngine.generate_report(data['weights'], pca_ratio, port_ret)

    # ▼▼▼ 詳細レビュー生成 (日本語版) ▼▼▼
    detailed_review = []
    
    # 効率性評価
    if sharpe_ratio > 1.0:
        detailed_review.append(f"✅ 効率性: 非常に優れたリスク調整後リターン (Sharpe: {sharpe_ratio:.2f}) を示しています。取ったリスクに対して十分なリターンが得られています。")
    elif sharpe_ratio > 0.6:
        detailed_review.append(f"ℹ️ 効率性: リスクとリターンのバランスは良好です (Sharpe: {sharpe_ratio:.2f})。分散された株式ポートフォリオとして標準的な水準です。")
    else:
        detailed_review.append(f"⚠️ 効率性: リスクに対するリターンがやや低めです (Sharpe: {sharpe_ratio:.2f})。分散投資の強化や、高ボラティリティ資産の比率見直しを検討してください。")

    # ボラティリティ評価
    if vol < 0.12:
        detailed_review.append(f"🛡️ 安定性: 変動率（ボラティリティ）は低く ({vol:.2%})、資産保全に適したディフェンシブな構成です。")
    elif vol < 0.18:
        detailed_review.append(f"⚖️ 安定性: 変動率は中程度 ({vol:.2%}) であり、市場平均並みの値動きが予想されます。")
    else:
        detailed_review.append(f"🔥 安定性: 変動率が高くなっています ({vol:.2%})。大きな価格変動に耐えられるリスク許容度が必要です。")

    # ドローダウン評価
    detailed_review.append(f"📉 耐性テスト: 過去の最大下落率（Max Drawdown）は {max_dd:.2%} でした。将来の弱気相場でも同程度の一時的な資産減少を覚悟する必要があります。")

    detailed_review_str = "\n".join(detailed_review)

    # --- 3. Payload 作成 ---
    analysis_payload = {
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'metrics': {
            'CAGR': f"{cagr:.2%}",
            'Vol': f"{vol:.2%}",
            'MaxDD': f"{max_dd:.2%}",
            'Sharpe': f"{sharpe_ratio:.2f}",
            'Calmar Ratio': f"{calmar:.2f}",
            'Information Ratio': f"{info_ratio:.2f}" if not np.isnan(info_ratio) else "N/A"
        },
        'factor_comment': factor_comment,
        'diagnosis': {
            'type': report['type'],
            'diversification_comment': report['diversification_comment'],
            'risk_comment': report['risk_comment'],
            'action_plan': report['action_plan']
        },
        'detailed_review': detailed_review_str,
        'mc_stats': f"中央値シナリオ: {final_median:,.0f}円 | "
                    f"悲観シナリオ(10%): {final_p10:,.0f}円 | "
                    f"楽観シナリオ(90%): {final_p90:,.0f}円"
    }

    # PDF用にグラフを格納
    figs_for_report = {}
    if fig_corr_report:
        figs_for_report['correlation'] = fig_corr_report

    # --- 4. ビジュアライゼーション表示 ---
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("年平均成長率 (CAGR)", f"{cagr:.2%}")
    c2.metric("リスク (Vol)", f"{vol:.2%}")
    c3.metric("最大下落率 (Max DD)", f"{max_dd:.2%}", delta_color="inverse")
    c4.metric("シャープレシオ", f"{sharpe_ratio:.2f}")
    c5.metric("オメガレシオ", f"{omega:.2f}")

    if not np.isnan(info_ratio):
        st.caption(f"📊 対ベンチマーク ({data['bench_name']}) | インフォメーションレシオ: **{info_ratio:.2f}** (トラッキングエラー: {track_err:.2%})")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🧬 構成", "🌊 要因", "⏳ 過去", "💸 コスト", "🏆 寄与度", "🔮 将来"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("分散の質 (PCA分析)")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = pca_ratio * 100, 
                title = {'text': "第1主成分の寄与率 (%)"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['main']},
                         'steps': [{'range': [0, 60], 'color': "#333"}, {'range': [60, 100], 'color': "#555"}],
                         'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("#### 🧭 資産クラスターマップ (PCA)")
            try:
                comp_clean = data['components'].dropna()
                if not comp_clean.empty and comp_clean.shape[1] > 1:
                    pca = PCA(n_components=2)
                    pca_coords = pca.fit_transform(comp_clean.T)
                    labels = comp_clean.columns
                    
                    fig_pca = px.scatter(x=pca_coords[:, 0], y=pca_coords[:, 1], text=labels, 
                                         color=labels, title="資産の類似性マップ")
                    fig_pca.update_traces(textposition='top center', marker=dict(size=12))
                    fig_pca.update_layout(xaxis_title="第1成分", yaxis_title="第2成分", showlegend=False)
                    st.plotly_chart(fig_pca, use_container_width=True)
            except Exception as e:
                st.warning(f"PCA散布図の描画エラー: {e}")

        with c2:
            st.subheader("資産配分")
            fig_pie = px.pie(values=list(data['weights'].values()), names=list(data['weights'].keys()), hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            figs_for_report['allocation'] = fig_pie
            
            st.markdown("---")
            st.subheader("🩺 AIポートフォリオ診断")
            st.markdown(f"""
            <div class="report-box">
                <h3 style="color: #00FFFF; margin-bottom:0px;">{report['type']}</h3>
                <hr style="margin-top:5px; margin-bottom:10px; border-color: #555;">
                <p><b>🧐 診断:</b><br>{report['diversification_comment']}</p>
                <p><b>⚠️ リスク警告:</b><br>{report['risk_comment']}</p>
                <p><b>💡 アクションプラン:</b><br>{report['action_plan']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if fig_corr_report:
                st.markdown("#### 🔥 相関マトリックス")
                st.plotly_chart(fig_corr_report, use_container_width=True)

    with tab2:
        if data['factors'].empty:
            st.error("🚫 ファクターデータの取得に失敗しました。")
        else:
            st.subheader("📊 スタイル分析 (回帰分析)")
            if params is not None:
                c1, c2 = st.columns([1, 1])
                with c1:
                    beta_df = params.drop('const') if 'const' in params else params
                    colors = ['#00CC96' if x > 0 else '#FF4B4B' for x in beta_df.values]
                    fig_beta = go.Figure(go.Bar(
                        x=beta_df.values, y=beta_df.index, orientation='h', 
                        marker_color=colors, text=[f"{x:.2f}" for x in beta_df.values], textposition='auto'
                    ))
                    fig_beta.update_layout(title="ファクター感応度 (Beta)", xaxis_title="感応度", height=300)
                    st.plotly_chart(fig_beta, use_container_width=True)
                    st.caption(f"決定係数 (R²): {r_sq:.2%} (モデル説明力)")
                    figs_for_report['factors'] = fig_beta
                
                with c2:
                    st.markdown(f"""
                    <div class="factor-box">
                        <h4 style="color: #FF69B4; margin-bottom:10px;">🧠 AIスタイル分析</h4>
                        <div style="white-space: pre-wrap;">{factor_comment}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📈 ファクター推移 (ローリング分析)")
            rolling_betas = analyzer.rolling_beta_analysis(port_ret, data['factors'])
            
            if not rolling_betas.empty:
                fig_roll = go.Figure()
                cols = rolling_betas.columns
                if 'Mkt-RF' in cols: 
                    fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['Mkt-RF'], name='市場感応度 (Beta)', line=dict(width=3, color=COLORS['main'])))
                if 'SMB' in cols: 
                    fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['SMB'], name='小型株効果 (SMB)', line=dict(dash='dot', color='orange')))
                if 'HML' in cols: 
                    fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['HML'], name='バリュー効果 (HML)', line=dict(dash='dot', color='yellow')))
                
                if not any(x in cols for x in ['Mkt-RF', 'SMB', 'HML']):
                    for c in cols:
                        fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas[c], name=c))

                fig_roll.update_layout(title="過去12ヶ月のファクター感応度推移", yaxis_title="Beta", height=400)
                st.plotly_chart(fig_roll, use_container_width=True)
            else:
                st.info("ローリング分析には少なくとも12ヶ月以上のデータが必要です。")

    with tab3:
        st.subheader("過去データによるストレステスト")
        cum_ret = (1 + port_ret).cumprod() * 10000
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=cum_ret.index, y=[10000]*len(cum_ret), mode='lines', name='元本 (10,000)', line=dict(color=COLORS['principal'], width=1, dash='dot')))

        if not bench_ret.empty:
            bench_cum = (1 + bench_ret).cumprod()
            common_idx = cum_ret.index.intersection(bench_cum.index)
            bench_cum = bench_cum.loc[common_idx]
            bench_cum = bench_cum / bench_cum.iloc[0] * 10000
            fig_hist.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, mode='lines', name=f"ベンチマーク ({data['bench_name']})", line=dict(color=COLORS['benchmark'], width=1.5)))

        fig_hist.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, fill='tozeroy', fillcolor=COLORS['bg_fill'], mode='lines', name='ポートフォリオ', line=dict(color=COLORS['main'], width=2.5)))
        st.plotly_chart(fig_hist, use_container_width=True)
        figs_for_report['cumulative'] = fig_hist

        fig_dd = go.Figure()
        dd_series = (cum_ret / cum_ret.cummax() - 1)
        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', name='Drawdown', line=dict(color='red')))
        fig_dd.update_layout(title="ドローダウン推移")
        st.plotly_chart(fig_dd, use_container_width=True)
        figs_for_report['drawdown'] = fig_dd

        st.markdown("---")
        st.subheader("📊 リターン分布ヒストグラム")
        mu, std = port_ret.mean(), port_ret.std()
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=port_ret, 
            histnorm='probability density', 
            name='実績リターン', 
            marker_color=COLORS['hist_bar'], 
            opacity=0.75, 
            nbinsx=60
        ))
        
        if not np.isnan(std) and std > 0:
            x_range = np.linspace(port_ret.min(), port_ret.max(), 100)
            y_norm = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x_range - mu) / std) ** 2)
            fig_dist.add_trace(go.Scatter(x=x_range, y=y_norm, mode='lines', name='正規分布 (理論値)', line=dict(color='white', dash='dash', width=2)))
        
        fig_dist.update_layout(title="月次リターンの分布 vs 正規分布", xaxis_title="月次リターン", yaxis_title="密度", height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab4:
        st.subheader("コストによるリターン低下分析 (20年シミュレーション)")
        
        # 修正: エンジンの戻り値4つに対応 (gross, net, loss, cost_pct)
        sim_res = analyzer.cost_drag_simulation(port_ret, data['cost_tier'])
        if len(sim_res) == 4:
            gross, net, loss, cost_pct = sim_res
        else:
            gross, net, loss = sim_res
            cost_pct = 0.0 # fallback
        
        loss_amount = 1000000 * loss
        final_amount_net = 1000000 * net.iloc[-1]
        
        c1, c2 = st.columns([3, 1])
        with c1:
            # 改善: 積層面積グラフ (Stacked Area) に変更して「失われた部分」を強調
            fig_cost = go.Figure()
            # 下層: 実質リターン
            fig_cost.add_trace(go.Scatter(
                x=net.index, y=net, 
                mode='lines', 
                stackgroup='one', 
                name=f'実質資産 (コスト控除後)', 
                line=dict(color=COLORS['main'], width=2),
                fillcolor='rgba(0, 255, 255, 0.2)'
            ))
            # 上層: 失われたコスト (差分)
            loss_series = gross - net
            fig_cost.add_trace(go.Scatter(
                x=gross.index, y=loss_series, 
                mode='lines', 
                stackgroup='one', 
                name='コストによる損失', 
                line=dict(color='rgba(255, 99, 71, 0.5)', width=0),
                fillcolor='rgba(255, 99, 71, 0.3)'
            ))
            
            fig_cost.update_layout(title="資産成長とコストの浸食イメージ (元本=1.0)", xaxis_title="経過年数", yaxis_title="倍率")
            st.plotly_chart(fig_cost, use_container_width=True)
            
        with c2:
            st.error(f"💸 失われる価値: ▲{loss_amount:,.0f} 円")
            st.markdown(f"最終評価額 (100万円投資): **{final_amount_net:,.0f} 円**")
            st.info(f"推定コスト率: 年 {cost_pct:.2%}")

    with tab5:
        st.subheader("リスク寄与度 vs 投資配分")
        attrib = analyzer.calculate_strict_attribution(data['components'], data['weights'])
        
        if not attrib.empty:
            # 改善: 投資比率とリスク寄与度を比較するグループ化棒グラフ
            weights_series = pd.Series(data['weights'])
            # インデックスを合わせる
            common_idx = weights_series.index.intersection(attrib.index)
            w_aligned = weights_series[common_idx] * 100 # %表記に
            r_aligned = attrib[common_idx] * 100 # %表記に
            
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                y=w_aligned.index, x=w_aligned.values, 
                name='投資配分 (%)', orientation='h', 
                marker_color='rgba(200, 200, 200, 0.6)'
            ))
            fig_compare.add_trace(go.Bar(
                y=r_aligned.index, x=r_aligned.values, 
                name='リスク寄与 (%)', orientation='h', 
                marker_color=COLORS['hist_bar']
            ))
            
            fig_compare.update_layout(
                barmode='group', 
                title="「お金を置いている場所」と「リスクが発生している場所」のズレ",
                xaxis_title="パーセント (%)",
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            figs_for_report['attribution'] = fig_compare

    with tab6:
        st.subheader(f"🎲 モンテカルロ・シミュレーション ({n_sims:,}回 / ファットテール対応)")
        if df_stats is not None:
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p50'], mode='lines', name='中央値', line=dict(color=COLORS['median'], width=3)))
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p10'], mode='lines', name='下位 10% (悲観)', line=dict(color=COLORS['p10'], width=1, dash='dot')))
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p90'], mode='lines', name='上位 10% (楽観)', line=dict(color=COLORS['p90'], width=1, dash='dot')))
            fig_mc.update_layout(title=f"20年後の資産予測 (元本: {init_inv:,} 円)", yaxis_title="評価額 (円)", height=500)
            st.plotly_chart(fig_mc, use_container_width=True)
            figs_for_report['monte_carlo'] = fig_mc

            st.markdown("### 🏁 最終評価額の分布")
            final_mean = np.mean(final_values)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("悲観 (P10)", f"{final_p10:,.0f}", delta_color="inverse")
            mc2.metric("中央値", f"{final_median:,.0f}")
            mc3.metric("平均値", f"{final_mean:,.0f}")
            mc4.metric("楽観 (P90)", f"{final_p90:,.0f}")

            # ヒストグラムの改善: ラベルが重ならないように高さを調整
            fig_mc_hist = go.Figure()
            counts, _ = np.histogram(final_values, bins=100)
            y_max_freq = counts.max()
            x_max_view = np.percentile(final_values, 98)

            fig_mc_hist.add_trace(go.Histogram(
                x=final_values, nbinsx=100, name='頻度', 
                marker_color=COLORS['hist_bar'], opacity=0.85
            ))
            
            # 改善: ラベル位置のオフセット設定 (y_max_freq に対する倍率)
            lines_config = [
                (final_p10, COLORS['p10'], f"悲観10%:<br>{final_p10:,.0f}", 1.05, "dash", 2),
                (final_median, COLORS['median'], f"中央値:<br>{final_median:,.0f}", 1.25, "solid", 3), # 高さを変える
                (final_mean, COLORS['mean'], f"平均値:<br>{final_mean:,.0f}", 1.15, "dot", 2),      # 高さを変える
                (final_p90, COLORS['p90'], f"楽観10%:<br>{final_p90:,.0f}", 1.05, "dash", 2),
            ]
            
            for val, color, label, h_rate, dash, width in lines_config:
                # 垂直線
                fig_mc_hist.add_vline(x=val, line_width=width, line_dash=dash, line_color=color)
                # ラベル (y軸の位置を h_rate * y_max_freq に設定して重なり防止)
                fig_mc_hist.add_annotation(
                    x=val, y=y_max_freq * h_rate,
                    text=label, showarrow=False, font=dict(color=color)
                )

            fig_mc_hist.update_layout(
                xaxis_title="最終評価額 (円)", yaxis_title="頻度", showlegend=False,
                xaxis=dict(range=[0, x_max_view]), 
                # y軸の範囲を少し広げてラベルを表示させる
                yaxis=dict(range=[0, y_max_freq * 1.4])
            )
            st.plotly_chart(fig_mc_hist, use_container_width=True)
            
            st.success(f"✅ シミュレーション完了: **{n_sims:,} シナリオ** を生成しました。")

    # --- 5. データ保存 ---
    st.session_state.payload = analysis_payload
    st.session_state.figs = figs_for_report
    st.session_state.analysis_done = True


# =========================================================
# 📄 PDF ダウンロードセクション
# =========================================================
st.markdown("---")

if st.session_state.analysis_done:
    st.header("📄 レポート作成")
    st.caption("分析結果をPDFレポートとしてダウンロードできます。")

    col_gen, col_dl = st.columns([1, 1])

    with col_gen:
        if st.button("📥 PDFレポートを作成"):
            with st.spinner("📄 PDFを生成中..."):
                try:
                    final_payload = st.session_state.payload.copy()
                    
                    # サイドバーの入力値をキャプチャ
                    if 'advisor_note' in locals() or 'advisor_note' in globals():
                        final_payload['advisor_note'] = advisor_note
                    
                    if final_payload and st.session_state.figs:
                        # pdf_generator呼び出し
                        pdf_buffer = create_pdf_report(final_payload, st.session_state.figs)
                        
                        if pdf_buffer:
                            # 修正: BytesIOオブジェクトからバイト列を取り出す (.getvalue())
                            # これにより '_io.BytesIO has no len()' エラーを回避します
                            st.session_state.pdf_bytes = pdf_buffer.getvalue()
                            
                            st.success(f"✅ レポートの準備ができました! ({len(st.session_state.pdf_bytes):,} bytes)")
                        else:
                            st.error("⚠️ PDFデータの生成に失敗しました（データが空です）。")
                    else:
                        st.error("⚠️ シミュレーションデータが見つかりません。先に分析を実行してください。")
                        
                except Exception as e:
                    st.error(f"PDF生成エラー: {e}")

    with col_dl:
        if st.session_state.pdf_bytes is not None:
            st.download_button(
                label="⬇️ PDFファイルをダウンロード",
                data=st.session_state.pdf_bytes,
                file_name="Portfolio_Analysis_Report.pdf",
                mime="application/pdf",
                type="primary"
            )

else:
    st.info("ℹ️ PDFレポートを作成するには、まず「分析を開始する」ボタンを押してシミュレーションを実行してください。")

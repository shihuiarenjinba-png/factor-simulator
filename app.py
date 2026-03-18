import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import unicodedata
import time
import datetime
import io

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
    .summary-box { background-color: #e8f4f8; padding: 15px; border-radius: 8px; border-left: 5px solid #1f77b4; margin-bottom: 20px;}
    .warning-box { background-color: #fff3f3; padding: 15px; border-radius: 8px; border-left: 5px solid #ff4b4b; margin-bottom: 20px;}
    .success-box { background-color: #e6ffed; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; margin-bottom: 10px; font-size: 0.9em;}
</style>
""", unsafe_allow_html=True)

st.title("Factor Simulator V20.0 - Robust UI & Engine")
st.markdown("Kenneth R. French の日本市場5ファクターに基づく厳密な月次時系列回帰分析と、銘柄ごとの加重平均寄与度を可視化します。")

# ---------------------------------------------------------
# キャッシュラッパー
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_market_data(tickers, days=730):
    df_fund = DataProvider.fetch_fundamentals(tickers)
    df_hist = DataProvider.fetch_historical_prices_monthly(tickers, days=days)
    df_market = DataProvider.fetch_market_rates(days=days)
    return df_fund, df_hist, df_market

# ---------------------------------------------------------
# 1. サイドバー: UIの整理とCSV即時反映
# ---------------------------------------------------------
st.sidebar.header("📊 分析設定")

# エキスパンダー1: 基本設定
with st.sidebar.expander("⚙️ 基本設定 (期間・ベンチマーク)", expanded=True):
    analysis_mode = st.radio(
        "分析期間の設定 (Monthly Data)", 
        ["直近指定期間", "利用可能な最大期間 (Max Historical)"],
        help="「最大期間」を選ぶと1990年以降のデータを全て使用します。"
    )
    lookback_years = st.slider("直近指定期間 (年)", 1, 10, 5) if analysis_mode == "直近指定期間" else 30
    bench_mode = st.radio("ベンチマークユニバース", ("TOPIX Core 30", "Nikkei 225"))

# エキスパンダー2: ポートフォリオ入力
with st.sidebar.expander("📁 ポートフォリオ入力", expanded=True):
    input_method = st.radio("入力方法", ["手入力 (テキスト)", "CSV/Excel アップロード"])
    
    port_tickers = []
    weight_list = []
    
    if input_method == "手入力 (テキスト)":
        port_input = st.text_area("銘柄コード (カンマまたは改行区切り)", value="7203, 8306, 9984, 6758, 8035")
        weight_input = st.text_area("ウェイト (空欄で時価総額加重)", value="", help="銘柄と同じ順番で数値を入力。")
        
        # サニタイズ
        sanitized_port = unicodedata.normalize('NFKC', port_input)
        raw_codes = re.findall(r'\b\d{4}\b', sanitized_port)
        port_tickers = [f"{code}.T" for code in list(dict.fromkeys(raw_codes))]
        
        clean_w_str = unicodedata.normalize('NFKC', weight_input).replace(" ", "").replace("\n", "")
        if clean_w_str:
            sanitized_w = re.sub(r'[^0-9.,]', '', clean_w_str)
            if sanitized_w:
                try: weight_list = [float(w) for w in sanitized_w.split(',') if w]
                except: pass 
                
    else:
        # CSV即時反映ロジック
        uploaded_file = st.file_uploader("構成銘柄ファイルを選択", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_up = pd.read_csv(uploaded_file)
                else:
                    df_up = pd.read_excel(uploaded_file)
                
                # ティッカー列の探索
                ticker_col = next((c for c in df_up.columns if 'コード' in c or 'Ticker' in c or '銘柄' in c), None)
                if ticker_col:
                    raw_codes = df_up[ticker_col].astype(str).str.extract(r'(\d{4})')[0].dropna().tolist()
                    port_tickers = [f"{code}.T" for code in list(dict.fromkeys(raw_codes))]
                    
                    # ウェイト列の探索
                    weight_col = next((c for c in df_up.columns if c in ['Weight', 'ウェイト', '比率', '割合']), None)
                    if weight_col:
                        weight_list = pd.to_numeric(df_up[weight_col], errors='coerce').fillna(0).tolist()[:len(port_tickers)]
                        st.markdown(f"<div class='success-box'>✅ <b>{len(port_tickers)}</b> 銘柄とウェイト情報を認識しました。</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='success-box'>✅ <b>{len(port_tickers)}</b> 銘柄を認識しました。<br><small>※ウェイト列がないため、時価総額加重で計算します。</small></div>", unsafe_allow_html=True)
                else:
                    st.error("ファイル内に銘柄コードを示す列が見つかりません。")
            except Exception as e:
                st.error(f"ファイルの読み込みに失敗しました: {e}")

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 回帰分析を実行", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 2. メイン処理ロジック
# ---------------------------------------------------------
if run_button:
    if not port_tickers:
        st.error("有効な銘柄コードが見つかりません。設定を確認してください。")
        st.stop()

    # --- 日付設定 ---
    global_end_date = datetime.date.today()
    if analysis_mode == "利用可能な最大期間 (Max Historical)":
        global_start_date = datetime.date(1990, 1, 1)
        lookback_days = (global_end_date - global_start_date).days
    else:
        lookback_days = 365 * lookback_years
        global_start_date = global_end_date - datetime.timedelta(days=lookback_days)

    # --- 処理状況の可視化 (プログレスバー細分化) ---
    progress_bar = st.progress(0, text="[1/5] 分析の準備中...")
    
    progress_bar.progress(10, text="[2/5] ユニバース（ベンチマーク）データを展開中...")
    bench_tickers = MarketMonitor.get_latest_tickers(bench_mode)
    all_target_tickers = list(set(bench_tickers + port_tickers))

    progress_bar.progress(20, text=f"[3/5] 市場データ(株価・財務)を取得中... (対象: {len(all_target_tickers)}銘柄)")
    
    # データ取得実行 (エラーハンドリング付き)
    try:
        df_all_fund, df_hist, df_market = get_cached_market_data(all_target_tickers, days=lookback_days)
    except Exception as e:
        st.error(f"❌ データ取得中に致命的なエラーが発生しました: {e}")
        st.stop()

    if df_all_fund.empty or df_hist.empty:
        st.error("❌ 株価・財務データの取得に失敗しました。Yahoo Financeの制限（429エラー）の可能性があります。\n\n**【回避策】10分ほど時間を置いてから再試行してください。**")
        st.stop()

    progress_bar.progress(50, text="[3/5] マクロファクターデータを同期中...")
    df_ff5 = DataProvider.fetch_ken_french_5factors(
        start_date=global_start_date.strftime('%Y-%m-%d'), 
        end_date=global_end_date.strftime('%Y-%m-%d')
    )

    if df_ff5.empty:
        st.warning("⚠️ ケネス・フレンチの5ファクターデータの取得に失敗しました。参考値のみで計算を続行します。")

    # --- 欠落データのパージと部分成功の許容 ---
    fetched_tickers = df_all_fund['Ticker'].tolist()
    missing_api = [t for t in port_tickers if t not in fetched_tickers]
    port_tickers_valid = [t for t in port_tickers if t not in missing_api]

    if missing_api:
        st.warning(f"⚠️ 以下の銘柄はAPI制限等のため取得できず、除外して計算を続行します: {', '.join(missing_api)}")

    if not port_tickers_valid:
        st.error("計算可能なポートフォリオ銘柄が一つもありません。処理を中止します。")
        st.stop()

    # --- ウェイト計算 (時価総額加重フォールバック対応) ---
    progress_bar.progress(60, text="[4/5] ポートフォリオ・ウェイトを計算中...")
    df_port_initial = df_all_fund[df_all_fund['Ticker'].isin(port_tickers_valid)].copy()
    
    if weight_list and len(weight_list) == len(port_tickers):
        # 欠落した銘柄分を抜いてウェイトマップを作成
        valid_weight_list = [w for t, w in zip(port_tickers, weight_list) if t in port_tickers_valid]
        weight_map = dict(zip(port_tickers_valid, valid_weight_list))
        df_port_initial['Weight'] = df_port_initial['Ticker'].map(weight_map)
        df_port_initial = QuantEngine.calculate_portfolio_weights(df_port_initial, user_weights_provided=True)
    else:
        # CSV等でウェイト指定がない場合は自動的に時価総額加重(MCW)
        df_port_initial = QuantEngine.calculate_portfolio_weights(df_port_initial, user_weights_provided=False)

    # --- 回帰分析実行 ---
    progress_bar.progress(70, text="[4/5] 月次多変量回帰分析 (Time-series Regression) を実行中...")
    regression_results = None
    if not df_ff5.empty:
        regression_results = QuantEngine.run_5factor_regression(df_hist, df_port_initial[['Ticker', 'Weight']], df_ff5, min_n_obs=24)
    
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
        portfolio_z_radar = {'Beta': 1.0, 'Value': 0, 'Size': 0, 'Quality': 0, 'Investment': 0}

    # --- スコア・寄与度計算 ---
    progress_bar.progress(85, text="[5/5] 銘柄固有スコアと加重平均寄与度を計算中...")
    df_bench = df_all_fund[df_all_fund['Ticker'].isin(bench_tickers)]
    market_stats, _ = UniverseManager.generate_market_stats(df_bench)

    df_port_proc = QuantEngine.process_raw_factors(df_port_initial)
    df_port_scored, _ = QuantEngine.compute_z_scores(df_port_proc, market_stats)
    df_port_scored, _ = QuantEngine.calculate_weighted_factor_contributions(df_port_scored)

    progress_bar.progress(100, text="✨ 分析完了！")
    time.sleep(0.5)
    progress_bar.empty()

    # ---------------------------------------------------------
    # 3. 分析結果の表示 UI
    # ---------------------------------------------------------
    if regression_success:
        adj_r2 = regression_results.get('Adjusted_R_squared', 0)
        n_obs = regression_results.get('N_Observations', 'Unknown')
        method = regression_results.get('Method', 'Unknown')
        p_val = regression_results.get('p_values', {}).get('Beta', 1.0)
        
        mode_text = "全期間 (1990~)" if analysis_mode == "利用可能な最大期間 (Max Historical)" else f"直近 {lookback_years} 年間"
        
        st.markdown(f"""
        <div class="summary-box">
            <h4>📈 回帰分析サマリー ({mode_text} / 月次ベース)</h4>
            <p>このポートフォリオの月次超過リターン変動のうち、<b>{adj_r2*100:.1f}%</b> が5つのマクロファクターによって説明可能です。<br>
            <ul style="margin-top: 5px; margin-bottom: 5px; line-height: 1.6;">
                <li><b>分析手法 (Engine):</b> {method} 
                <span style="font-size:0.85em; color:#666;">(Portfolio=共通期間一括, Individual=個別回帰加重平均)</span></li>
                <li><b>分析サンプル数 (N):</b> {n_obs} ヶ月</li>
                <li><b>自由度調整済み決定係数 (Adj R²):</b> {adj_r2:.3f}</li>
                <li><b>市場ベータの有意確率 (p-value):</b> {p_val:.4f}</li>
            </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ 回帰分析の実行条件が揃いませんでした</h4>
            <p>ファクターデータの取得失敗、または株価データとの有効な月次結合数が不足(24ヶ月未満)しています。<br>
            表示されているレーダーチャートは回帰分析を伴わない<b>推定不能な参考値(Fallback)</b>です。</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        suffix = f"(Adj R²={regression_results.get('Adjusted_R_squared', 0):.2f}, N={regression_results.get('N_Observations', 0)})" if regression_success else "(推定不能：参考値)"
        p_text = f"{global_start_date.strftime('%Y/%m')} - {global_end_date.strftime('%Y/%m')}"
        fig_radar = Visualizer.plot_radar_chart(portfolio_z_radar, title_suffix=suffix, period_text=p_text)
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

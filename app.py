import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re  # 正規表現モジュール（4桁数字の判定用）
import unicodedata  # 全角/半角の文字正規化サニタイズ用
import time # プログレスバーの演出用

# カスタムモジュールの読み込み
try:
    from data_provider import DataProvider
    from quant_engine import QuantEngine
    from universe_manager import MarketMonitor, UniverseManager
    from visualizer import Visualizer  
except ModuleNotFoundError as e:
    st.error(f"起動エラー: モジュールが見つかりません ({e})")
    st.info("app.py と同じフォルダに data_provider.py, quant_engine.py, universe_manager.py, visualizer.py があるか確認してください。")
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
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Factor Simulator V17.2 - Portfolio Analysis")
st.markdown("5ファクター（Beta, Value, Size, Quality, Investment）に基づくポートフォリオの直交化解析と可視化を行います。")

# ---------------------------------------------------------
# 一括データ取得＆強力キャッシュラッパー
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_market_data(tickers):
    df_fund = DataProvider.fetch_fundamentals(tickers)
    df_hist = DataProvider.fetch_historical_prices(tickers)
    df_market = DataProvider.fetch_market_rates()
    return df_fund, df_hist, df_market

# ---------------------------------------------------------
# 1. サイドバー: 入力インターフェース
# ---------------------------------------------------------
st.sidebar.header("📊 分析設定")

# ターゲットポートフォリオの入力
port_input = st.sidebar.text_area(
    "ポートフォリオ銘柄 (カンマまたは改行区切り)",
    value="7203, 8306, 9984, 6758, 8035",
    help="証券コード（4桁）を入力してください。"
)

# ウェイトの入力インターフェース
weight_input = st.sidebar.text_area(
    "ウェイト入力 (カンマ区切り・空欄で時価総額加重に自動移行)",
    value="",
    help="銘柄と同じ順番で数値を入力してください。例: 0.2, 0.3, 0.1..."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 ユニバース（市場基準）設定")

bench_mode = st.sidebar.radio(
    "ベンチマークの選択",
    ("TOPIX Core 30", "Nikkei 225")
)

# カスタム構成銘柄ファイルのアップロード
uploaded_file = st.sidebar.file_uploader(
    "公式構成銘柄ファイル (CSV/Excel) - 推奨", 
    type=['csv', 'xlsx', 'xls'],
    help="JPX等からダウンロードした構成銘柄リストをアップロードすると、APIの遅延やサイト依存を回避できます。"
)

run_button = st.sidebar.button("分析を実行", type="primary")

# ---------------------------------------------------------
# 2. メイン処理ロジック
# ---------------------------------------------------------
if run_button:
    progress_bar = st.progress(0, text="初期化中...")
    status_text = st.empty()
    
    # -----------------------------------------------------
    # Step 0: 入力データのサニタイズ（全角→半角、不要文字の除去）
    # -----------------------------------------------------
    sanitized_port = unicodedata.normalize('NFKC', port_input)
    raw_codes = re.findall(r'\b\d{4}\b', sanitized_port)
    
    port_tickers = []
    for code in raw_codes:
        ticker = f"{code}.T"
        if ticker not in port_tickers:
            port_tickers.append(ticker)
    
    if not port_tickers:
        st.error("有効な銘柄コード（4桁の数字）が見つかりませんでした。")
        st.stop()

    clean_weight_str = unicodedata.normalize('NFKC', weight_input).replace(" ", "").replace("\n", "")
    weight_list = []
    if clean_weight_str:
        sanitized_w = re.sub(r'[^0-9.,]', '', clean_weight_str)
        if sanitized_w:
            try:
                weight_list = [float(w) for w in sanitized_w.split(',') if w]
            except ValueError:
                pass 

    # -----------------------------------------------------
    # Step A: ユニバース（母集団）データの準備
    # -----------------------------------------------------
    progress_bar.progress(10, text="ユニバース情報の構築中...")
    custom_list = None
    if uploaded_file is not None:
        custom_list = MarketMonitor.load_tickers_from_file(uploaded_file)
        if custom_list:
            st.sidebar.success(f"カスタムファイルを読み込みました ({len(custom_list)}銘柄)")
        else:
            st.sidebar.warning("ファイルから銘柄コードを抽出できませんでした。フォールバックを使用します。")
    
    bench_tickers = MarketMonitor.get_latest_tickers(bench_mode, custom_list)
    all_target_tickers = list(set(bench_tickers + port_tickers))

    # -----------------------------------------------------
    # Step B: データの取得（キャッシュラッパーを使用）
    # -----------------------------------------------------
    progress_bar.progress(30, text="市場データ（財務・価格）を取得/キャッシュから読み込み中...")
    
    df_all_fund, df_hist, df_market = get_cached_market_data(all_target_tickers)
    
    # 完全に空の場合の全体エラーハンドリング
    if df_all_fund.empty:
        st.error("❌ データの取得に完全に失敗しました。")
        st.info("💡 **ヒント**: 現在、Yahoo Finance側で一時的なアクセス制限（Too Many Requests）がかかっている可能性が高いです。数分〜数十分ほど待ってから再度お試しください。")
        st.stop()

    # -----------------------------------------------------
    # Step C: 回帰ベータの計算
    # -----------------------------------------------------
    progress_bar.progress(65, text="市場連動性（回帰ベータ）を計算中...")
    df_all_fund = QuantEngine.calculate_beta(df_all_fund, df_hist, df_market)

    # -----------------------------------------------------
    # 【修正箇所】データ品質の可視化とエラーの明示
    # -----------------------------------------------------
    fetched_tickers = df_all_fund['Ticker'].tolist()
    
    # 1. 物理的な取得失敗 (APIエラーなどで完全にデータがない)
    missing_api = [t for t in port_tickers if t not in fetched_tickers]
    
    # 2. 計算結果が異常 (Beta_Raw が NaN になったもの)
    invalid_data = []
    if 'Beta_Raw' in df_all_fund.columns:
        invalid_data = df_all_fund[
            (df_all_fund['Ticker'].isin(port_tickers)) & 
            (df_all_fund['Beta_Raw'].isna())
        ]['Ticker'].tolist()

    all_excluded = list(set(missing_api + invalid_data))
    
    with st.expander("📊 データ品質・取得レポート", expanded=bool(all_excluded)):
        st.markdown(f"**・リクエスト銘柄数**: {len(port_tickers)} 件")
        st.markdown(f"**・分析採用銘柄数**: {len(port_tickers) - len(all_excluded)} 件")
        
        if all_excluded:
            st.warning("⚠️ 以下の銘柄はデータ不足のため分析から除外されました：")
            
            if missing_api:
                st.error(f"❌ 取得失敗（API応答なし）: {', '.join(missing_api)}")
            if invalid_data:
                st.error(f"❌ 計算不能（データ異常・流動性不足）: {', '.join(invalid_data)}")
                
            st.markdown("※ 上記の銘柄はベンチマークへの同化による分析エラーを防ぐための安全措置として除外されました。残りの銘柄のみでポートフォリオを再構築し、ウェイトを自動配分します。")
        else:
            st.success("✅ 全ポートフォリオ銘柄のデータ取得・計算に成功しました。")
            
        # ユニバースの取得状況
        fetched_bench = [t for t in bench_tickers if t in fetched_tickers]
        st.markdown(f"**・ユニバース(比較対象)取得**: {len(fetched_bench)} / {len(bench_tickers)} 件")

    port_tickers = [t for t in port_tickers if t not in all_excluded]
    
    if not port_tickers:
        st.error("❌ 計算可能なポートフォリオ銘柄がなくなりました。分析を中止します。")
        st.stop()

    # -----------------------------------------------------
    # Step D: ユニバース統計量の算出とポートフォリオのZスコア化
    # -----------------------------------------------------
    progress_bar.progress(80, text="ファクター直交化とZスコアへの変換中...")
    df_bench = df_all_fund[df_all_fund['Ticker'].isin(bench_tickers)]
    market_stats, df_bench_proc = UniverseManager.generate_market_stats(df_bench)

    df_port = df_all_fund[df_all_fund['Ticker'].isin(port_tickers)].copy()
    df_port_proc = QuantEngine.process_raw_factors(df_port)
    df_port_scored, _ = QuantEngine.compute_z_scores(df_port_proc, market_stats)
    
    # -----------------------------------------------------
    # Step E: 特性ベータの逆算とスマートウェイトの適用
    # -----------------------------------------------------
    progress_bar.progress(90, text="特性ベータ(Sensitivity Beta)と加重ウェイトの計算中...")
    
    df_port_scored = QuantEngine.calculate_sensitivity_beta(df_port_scored)

    # -----------------------------------------------------
    # ウェイト入力の厳格なフィードバック
    # -----------------------------------------------------
    # 除外された銘柄がある場合、入力ウェイト数と合わなくなる可能性があるため再配分
    original_port_length = len(port_tickers) + len(all_excluded)
    if weight_list:
        if len(weight_list) == original_port_length:
            # 入力されたポートフォリオの順序と合わせてマップを作成
            sanitized_port = unicodedata.normalize('NFKC', port_input)
            raw_codes_temp = re.findall(r'\b\d{4}\b', sanitized_port)
            original_tickers_ordered = [f"{code}.T" for code in list(dict.fromkeys(raw_codes_temp))]
            
            weight_map = dict(zip(original_tickers_ordered, weight_list))
            df_port_scored['Weight'] = df_port_scored['Ticker'].map(weight_map)
            df_port_scored = QuantEngine.calculate_portfolio_weights(df_port_scored, user_weights_provided=True)
            st.success("✅ 指定されたウェイトを残存銘柄へ適用・再配分しました。")
        else:
            st.error(f"❌ ウェイト入力エラー: リクエストされた銘柄数（{original_port_length}件）に対し、入力されたウェイト数が（{len(weight_list)}件）です。時価総額加重（または等金額）で自動計算します。")
            df_port_scored = QuantEngine.calculate_portfolio_weights(df_port_scored, user_weights_provided=False)
    else:
        df_port_scored = QuantEngine.calculate_portfolio_weights(df_port_scored, user_weights_provided=False)

    # -----------------------------------------------------
    # Step F: レーダーチャート用データの集計
    # -----------------------------------------------------
    progress_bar.progress(95, text="ビジュアル・レンダリング準備中...")
    
    portfolio_z = {}
    radar_factors = ['Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z']
    radar_names = ['Beta', 'Value', 'Size', 'Quality', 'Investment']
    
    for f, name in zip(radar_factors, radar_names):
        if f in df_port_scored.columns:
            portfolio_z[name] = (df_port_scored[f] * df_port_scored['Weight']).sum()
        else:
            portfolio_z[name] = 0.0

    progress_bar.progress(100, text="分析完了！")
    time.sleep(0.5)
    progress_bar.empty()

    # ---------------------------------------------------------
    # 3. 分析結果の表示
    # ---------------------------------------------------------
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 5ファクター・エクスポージャー", help="【Regression Beta (回帰ベータ)】\n過去の価格データに基づき、市場ベンチマークに対する純粋な価格連動性（時系列回帰）を測定したものです。")
        
        max_z_val = 0
        available_factors = [f for f in radar_factors if f in df_port_scored.columns]
        if available_factors:
            max_z_val = df_port_scored[available_factors].abs().max().max()
        
        dynamic_range = max(3.0, float(np.ceil(max_z_val * 1.1)))
        
        fig_radar = Visualizer.plot_radar_chart(portfolio_z)
        fig_radar.update_polars(
            radialaxis=dict(
                range=[-dynamic_range, dynamic_range],
                autorange=False
            )
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col2:
        st.subheader("🧩 ファクター寄与度分解", help="【Sensitivity Beta (特性ベータ)】\n各ファクター（割安性やクオリティなど）の組み合わせから逆算された、その銘柄が本質的に持つ市場感応度の理論値です。")
        fig_bar = Visualizer.plot_contribution_bar_chart(df_port_scored)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 銘柄別 Z-Score 詳細データ")
    
    display_cols = ['Ticker', 'Weight', 'Beta_Z', 'Sensitivity_Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z']
    existing_cols = [col for col in display_cols if col in df_port_scored.columns]
    
    df_display = df_port_scored[existing_cols].copy()
    
    rename_dict = {
        'Beta_Z': 'Regression Beta',
        'Sensitivity_Beta_Z': 'Sensitivity Beta',
        'Value_Z': 'Value',
        'Size_Z': 'Size',
        'Quality_Z': 'Quality',
        'Investment_Z': 'Investment'
    }
    df_display.rename(columns=rename_dict, inplace=True)
    
    # 小数点以下2桁にフォーマットし、インデックスを隠して表示
    st.dataframe(df_display.style.format({col: "{:.2f}" for col in df_display.columns if col not in ['Ticker']}), use_container_width=True, hide_index=True)

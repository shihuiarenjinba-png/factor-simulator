import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re  # 正規表現モジュール（4桁数字の判定用）

# カスタムモジュールの読み込み
try:
    from data_provider import DataProvider
    from quant_engine import QuantEngine
    from universe_manager import MarketMonitor, UniverseManager
    from visualizer import Visualizer  # 新規追加モジュールの読み込み
except ModuleNotFoundError as e: # 【修正箇所1】ImportErrorから変更し、文法エラーを隠さないように改善
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

st.title("Factor Simulator V17.1 - Portfolio Analysis")
st.markdown("5ファクター（Beta, Value, Size, Quality, Investment）に基づくポートフォリオの直交化解析と可視化を行います。")

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

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 ユニバース（市場基準）設定")

bench_mode = st.sidebar.radio(
    "ベンチマークの選択",
    ("TOPIX Core 30", "Nikkei 225")
)

# 新規実装：カスタム構成銘柄ファイルのアップロード
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
    with st.spinner("市場データを解析中... (APIからデータを取得しています)"):
        
        # 銘柄コードのパースと正規化 (4桁数字の抽出)
        raw_codes = re.findall(r'\b\d{4}\b', port_input)
        port_tickers = list(set([f"{code}.T" for code in raw_codes]))
        
        if not port_tickers:
            st.error("有効な銘柄コード（4桁の数字）が見つかりませんでした。")
            st.stop()

        # -----------------------------------------------------
        # Step A: ユニバース（母集団）データの準備
        # -----------------------------------------------------
        custom_list = None
        if uploaded_file is not None:
            custom_list = MarketMonitor.load_tickers_from_file(uploaded_file)
            if custom_list:
                st.sidebar.success(f"カスタムファイルを読み込みました ({len(custom_list)}銘柄)")
            else:
                st.sidebar.warning("ファイルから銘柄コードを抽出できませんでした。フォールバックを使用します。")
        
        bench_tickers = MarketMonitor.get_latest_tickers(bench_mode, custom_list)
        
        # ポートフォリオ銘柄がユニバースに含まれていない場合の考慮（計算用に結合）
        all_target_tickers = list(set(bench_tickers + port_tickers))

        # -----------------------------------------------------
        # Step B: ファンダメンタルズの取得と市場統計量の算出
        # -----------------------------------------------------
        # DataProviderから一括取得
        df_all_fund = DataProvider.fetch_fundamentals(all_target_tickers)
        
        if df_all_fund.empty:
            st.error("データの取得に失敗しました。ネットワーク接続を確認してください。")
            st.stop()

        # ベンチマーク集団のみを抽出して市場統計量（中央値、MAD、直交化係数）を計算
        df_bench = df_all_fund[df_all_fund['Ticker'].isin(bench_tickers)]
        market_stats, df_bench_proc = UniverseManager.generate_market_stats(df_bench)

        # -----------------------------------------------------
        # Step C: ポートフォリオのZスコア算出
        # -----------------------------------------------------
        df_port = df_all_fund[df_all_fund['Ticker'].isin(port_tickers)].copy()
        
        # 【修正箇所2】堅牢なエンジン仕様に合わせてデータ加工とメソッド呼び出しを同期
        df_port_proc = QuantEngine.process_raw_factors(df_port)
        df_port_scored, _ = QuantEngine.compute_z_scores(df_port_proc, market_stats)
        
        # 簡易的に均等ウェイトを設定
        df_port_scored['Weight'] = 1.0 / len(df_port_scored)
        
        # ポートフォリオ全体のZスコア（ウェイト加重平均）
        portfolio_z = {}
        factors = ['Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z']
        display_names = ['Beta', 'Value', 'Size', 'Quality', 'Investment']
        
        for f, name in zip(factors, display_names):
            if f in df_port_scored.columns:
                portfolio_z[name] = (df_port_scored[f] * df_port_scored['Weight']).sum()
            else:
                portfolio_z[name] = 0.0

        # ---------------------------------------------------------
        # 3. 分析結果の表示 (Visualizerの活用)
        # ---------------------------------------------------------
        st.success("分析が完了しました。")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 5ファクター・エクスポージャー")
            # レーダーチャートの描画
            fig_radar = Visualizer.plot_radar_chart(portfolio_z)
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col2:
            st.subheader("🧩 ファクター寄与度分解")
            # 寄与度バーチャートの描画
            fig_bar = Visualizer.plot_contribution_bar_chart(df_port_scored)
            st.plotly_chart(fig_bar, use_container_width=True)

        # データテーブルの表示
        st.markdown("---")
        st.subheader("📋 銘柄別 Z-Score 詳細データ")
        
        # 表示用に列を整理
        display_cols = ['Ticker', 'Weight'] + [f for f in factors if f in df_port_scored.columns]
        df_display = df_port_scored[display_cols].copy()
        
        # 小数点以下2桁にフォーマット
        st.dataframe(df_display.style.format({col: "{:.2f}" for col in df_display.columns if col not in ['Ticker']}), use_container_width=True)

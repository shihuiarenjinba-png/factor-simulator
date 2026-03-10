import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re  # 正規表現モジュール（4桁数字の判定用）
import unicodedata  # 【追加】全角/半角の文字正規化サニタイズ用

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

# 【追加】ウェイトの入力インターフェース
weight_input = st.sidebar.text_area(
    "ウェイト入力 (カンマ区切り・空欄で等金額/時価総額加重)",
    value="",
    help="銘柄と同じ順番で数値を入力してください。例: 0.2, 0.3, 0.1..."
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
        
        # -----------------------------------------------------
        # 【修正】入力データのサニタイズ（全角→半角、不要文字の除去）
        # -----------------------------------------------------
        # 銘柄コードのパースと正規化
        sanitized_port = unicodedata.normalize('NFKC', port_input)
        raw_codes = re.findall(r'\b\d{4}\b', sanitized_port)
        
        # 入力順序を保持しつつ重複を排除（ウェイトとの紐付けを正確にするため）
        port_tickers = []
        for code in raw_codes:
            ticker = f"{code}.T"
            if ticker not in port_tickers:
                port_tickers.append(ticker)
        
        if not port_tickers:
            st.error("有効な銘柄コード（4桁の数字）が見つかりませんでした。")
            st.stop()

        # ウェイト文字列のサニタイズとリスト化
        clean_weight_str = unicodedata.normalize('NFKC', weight_input).replace(" ", "").replace("\n", "")
        weight_list = []
        if clean_weight_str:
            # 数字、ドット、カンマ以外を除外して安全にリスト化
            sanitized_w = re.sub(r'[^0-9.,]', '', clean_weight_str)
            if sanitized_w:
                try:
                    weight_list = [float(w) for w in sanitized_w.split(',') if w]
                except ValueError:
                    pass # 変換失敗時は空リストのまま進行（エンジン側で自動フォールバック）

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
        # Step C: ポートフォリオのZスコア算出とウェイト適用
        # -----------------------------------------------------
        df_port = df_all_fund[df_all_fund['Ticker'].isin(port_tickers)].copy()
        
        # 【修正箇所2】堅牢なエンジン仕様に合わせてデータ加工とメソッド呼び出しを同期
        df_port_proc = QuantEngine.process_raw_factors(df_port)
        df_port_scored, _ = QuantEngine.compute_z_scores(df_port_proc, market_stats)
        
        # -----------------------------------------------------
        # 【追加】ウェイトの紐付けとQuantEngineによるガードレール計算
        # -----------------------------------------------------
        if weight_list and len(weight_list) == len(port_tickers):
            # 銘柄とウェイトを辞書で紐付け、DataFrameに確実に追加
            weight_map = dict(zip(port_tickers, weight_list))
            df_port_scored['Weight'] = df_port_scored['Ticker'].map(weight_map)
            df_port_scored = QuantEngine.calculate_portfolio_weights(df_port_scored, user_weights_provided=True)
        else:
            if weight_list:
                st.warning("⚠️ 銘柄数とウェイトの数が一致しません。等金額（または時価総額）加重で自動計算します。")
            df_port_scored = QuantEngine.calculate_portfolio_weights(df_port_scored, user_weights_provided=False)

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
            
            # -----------------------------------------------------
            # 【追加】レーダーチャートの動的スケーリング
            # -----------------------------------------------------
            # 算出されたZスコアの絶対値の最大を探す
            max_z_val = 0
            available_factors = [f for f in factors if f in df_port_scored.columns]
            if available_factors:
                max_z_val = df_port_scored[available_factors].abs().max().max()
            
            # 基本は±3.0を維持しつつ、飛び抜けた数値があればそれに合わせて枠を広げる（1.1倍の余白）
            dynamic_range = max(3.0, float(np.ceil(max_z_val * 1.1)))
            
            # Plotlyの機能を使って、後からチャートの最大値を書き換える
            fig_radar.update_polars(
                radialaxis=dict(
                    range=[-dynamic_range, dynamic_range],
                    autorange=False
                )
            )
            
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

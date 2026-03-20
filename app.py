import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re  # 正規表現モジュール
import unicodedata  # サニタイズ用
import time
import datetime  # 日付処理用のライブラリ

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
    .metric-value { font-size: 24px; font-weight: bold; color: #333; }
    .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
    .summary-box { background-color: #e8f4f8; padding: 15px; border-radius: 8px; border-left: 5px solid #1f77b4; margin-bottom: 20px;}
    .warning-box { background-color: #fff3f3; padding: 15px; border-radius: 8px; border-left: 5px solid #ff4b4b; margin-bottom: 20px;}
    .success-box { background-color: #e6ffed; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; margin-bottom: 10px; font-size: 0.9em;}
</style>
""", unsafe_allow_html=True)

st.title("Factor Simulator V19.1 - Responsive UI & Robust Engine")
st.markdown("Kenneth R. French の日本市場5ファクターに基づく厳密な月次時系列回帰分析と、銘柄ごとの加重平均寄与度を可視化します。")

# ---------------------------------------------------------
# キャッシュラッパー
# ---------------------------------------------------------
# 【修正①】ベンチマーク銘柄は24時間キャッシュ（毎回225銘柄を取得しない）
@st.cache_data(ttl=86400, show_spinner=False)
def get_cached_benchmark_fundamentals(bench_tickers_tuple):
    """
    ベンチマーク銘柄の財務データを24時間キャッシュ。
    リスト→タプル変換でキャッシュキーを安定化。
    """
    bench_tickers = list(bench_tickers_tuple)
    return DataProvider.fetch_fundamentals(bench_tickers)

# 【修正①】ポートフォリオ銘柄は1時間キャッシュ（少数なので速い）
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_portfolio_data(port_tickers_tuple, days=730):
    """
    ポートフォリオ銘柄のデータのみ取得。
    月次分析に合わせ、履歴データは _monthly メソッドを呼び出す。
    """
    port_tickers = list(port_tickers_tuple)
    df_fund = DataProvider.fetch_fundamentals(port_tickers)
    df_hist = DataProvider.fetch_historical_prices_monthly(port_tickers, days=days)
    df_market = DataProvider.fetch_market_rates(days=days)  # 内側で月次処理済み
    return df_fund, df_hist, df_market

# ---------------------------------------------------------
# 1. サイドバー: 入力
# ---------------------------------------------------------
st.sidebar.header("📊 分析設定")

# ── 基本設定 ──────────────────────────────────────────────
with st.sidebar.expander("⚙️ 基本設定 (期間・ベンチマーク)", expanded=True):
    analysis_mode = st.radio(
        "分析期間の設定 (Monthly Data)", 
        ["直近指定期間", "利用可能な最大期間 (Max Historical)"],
        help="「最大期間」を選ぶと1990年以降のデータを全て使用し、精度の高い回帰分析を行います。"
    )
    lookback_years = st.slider("直近指定期間 (年)", 1, 10, 5) if analysis_mode == "直近指定期間" else 30
    bench_mode = st.radio("ベンチマークユニバース", ("TOPIX Core 30", "Nikkei 225"))

# ── session_state の初期化 ────────────────────────────────
# ポートフォリオテーブルの初期値（デフォルト5銘柄）
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame({
        "コード": ["7203", "8306", "9984", "6758", "8035"],
        "ウェイト": [20.0, 20.0, 20.0, 20.0, 20.0]
    })

# ── ポートフォリオ入力 ────────────────────────────────────
with st.sidebar.expander("📁 ポートフォリオ入力", expanded=True):

    # CSVアップロード（アップロードしたらテーブルに即反映）
    uploaded_file = st.file_uploader(
        "CSVをアップロード（任意）",
        type=["csv", "xlsx", "xls"],
        help="アップロードすると下のテーブルに自動反映されます。"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            else:
                df_up = pd.read_excel(uploaded_file)

            # ティッカー列の自動検出
            ticker_col = next((c for c in df_up.columns if any(k in c for k in [
                "コード", "Ticker", "ticker", "銘柄", "ティッカー", "Symbol", "symbol", "Code", "code"
            ])), None)
            if not ticker_col:
                best_col, best_count = None, 0
                for c in df_up.columns:
                    cnt = df_up[c].astype(str).str.contains(r"\b\d{4}\b").sum()
                    if cnt > best_count:
                        best_count, best_col = cnt, c
                if best_count > 0:
                    ticker_col = best_col

            # ウェイト列の自動検出
            weight_col = next((c for c in df_up.columns if any(k in c for k in [
                "Weight", "weight", "ウェイト", "比率", "割合", "保有", "Ratio", "ratio", "%"
            ])), None)

            if ticker_col:
                raw_codes = df_up[ticker_col].astype(str).str.extract(r"(\d{4})")[0].dropna().tolist()
                codes = list(dict.fromkeys(raw_codes))  # 重複排除・順序保持
                if weight_col:
                    weights = pd.to_numeric(df_up[weight_col], errors="coerce").fillna(0).tolist()[:len(codes)]
                else:
                    weights = [round(100.0 / len(codes), 1)] * len(codes)

                # テーブルに反映（session_stateを上書き）
                st.session_state.portfolio_df = pd.DataFrame({
                    "コード": codes,
                    "ウェイト": [float(w) for w in weights]
                })
                st.success(f"✅ {len(codes)} 銘柄を読み込みました。")
            else:
                st.error("銘柄コード列が見つかりません。")
        except Exception as e:
            st.error(f"読み込み失敗: {e}")

    # テーブル直接編集（CSVがなくてもここで追加・削除・編集可能）
    edited_df = st.data_editor(
        st.session_state.portfolio_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "コード":   st.column_config.TextColumn("証券コード (4桁)", help="例: 7203"),
            "ウェイト": st.column_config.NumberColumn("ウェイト", min_value=0.0, format="%.1f", help="合計が100になるよう設定"),
        }
    )

    # 編集内容を session_state に即座に反映
    st.session_state.portfolio_df = edited_df

    # ウェイト合計を表示（参考情報）
    total_w = edited_df["ウェイト"].sum() if "ウェイト" in edited_df.columns else 0
    if abs(total_w - 100.0) > 0.5:
        st.caption(f"⚠️ ウェイト合計: {total_w:.1f}（分析時に自動正規化されます）")
    else:
        st.caption(f"✅ ウェイト合計: {total_w:.1f}")

# ── テーブルからポートフォリオを組み立て ─────────────────
port_tickers = []
weight_list  = []
uploaded_file_ref = None  # 後続処理との互換性のため

for _, row in edited_df.dropna(subset=["コード"]).iterrows():
    raw = str(row["コード"]).strip()
    match = re.search(r"\b(\d{4})\b", unicodedata.normalize("NFKC", raw))
    if match:
        port_tickers.append(f"{match.group(1)}.T")
        w = row.get("ウェイト", 0)
        weight_list.append(float(w) if pd.notna(w) else 0.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 回帰分析を実行", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 2. メイン処理ロジック
# ---------------------------------------------------------
if run_button:
    if not port_tickers:
        st.error("有効な銘柄コードが見つかりません。設定を確認してください。")
        st.stop()

    progress_bar = st.progress(0, text="[1/5] 初期化中...")
    
    # 日付の動的計算
    global_end_date = datetime.date.today()
    if analysis_mode == "利用可能な最大期間 (Max Historical)":
        global_start_date = datetime.date(1990, 1, 1)
        lookback_days = (global_end_date - global_start_date).days
    else:
        lookback_days = 365 * lookback_years
        global_start_date = global_end_date - datetime.timedelta(days=lookback_days)

    # --- Step A: ユニバース準備 ---
    progress_bar.progress(10, text="[2/5] ユニバース（比較基準）の準備中...")
    custom_list = None  # data_editorベースに移行したためCSVファイル渡しは不要
    bench_tickers = MarketMonitor.get_latest_tickers(bench_mode, custom_list)

    # 【修正①】ベンチマーク財務データは24時間キャッシュから取得
    # タプルに変換してキャッシュキーを安定化（リストはハッシュ不可のため）
    progress_bar.progress(15, text="[2/5] ベンチマーク統計データの取得中（キャッシュ利用）...")
    df_all_fund_bench = get_cached_benchmark_fundamentals(tuple(sorted(bench_tickers)))

    # --- Step B: ポートフォリオのデータ取得のみ実行（件数が少ない） ---
    progress_bar.progress(20, text=f"[3/5] ポートフォリオデータ(株価・財務)の取得中... (対象: {len(port_tickers)}銘柄)")
    
    try:
        # 【修正①】ポートフォリオ銘柄のみ取得（旧: all_target_tickers=ベンチ+ポートで225銘柄以上）
        df_port_fund, df_hist, df_market = get_cached_portfolio_data(
            tuple(sorted(port_tickers)), days=lookback_days
        )
    except Exception as e:
        st.error(f"❌ データ取得中に致命的なエラーが発生しました: {e}")
        st.stop()
    
    if df_port_fund.empty or df_hist.empty:
        st.error("❌ 株価・財務データの取得に失敗しました。Yahoo Financeの制限（429エラー）の可能性があります。\n\n**【回避策】分析するポートフォリオ銘柄数を減らすか、10分ほど時間を置いてから再試行してください。**")
        st.stop()

    # 【修正①】ベンチマーク財務データとポートフォリオ財務データを結合
    # ベンチ取得に失敗してもポートフォリオ分析は継続できるようフェールセーフ
    if not df_all_fund_bench.empty:
        all_tickers_in_bench = df_all_fund_bench['Ticker'].tolist() if 'Ticker' in df_all_fund_bench.columns else []
        port_only_fund = df_port_fund[~df_port_fund['Ticker'].isin(all_tickers_in_bench)] if 'Ticker' in df_port_fund.columns else df_port_fund
        df_all_fund = pd.concat([df_all_fund_bench, port_only_fund], ignore_index=True)
    else:
        # ベンチマーク取得失敗時はポートフォリオデータのみで続行
        df_all_fund = df_port_fund

    progress_bar.progress(40, text=f"[3/5] マクロファクターデータの同期中...")
    
    df_ff5 = DataProvider.fetch_ken_french_5factors(
        start_date=global_start_date.strftime('%Y-%m-%d'), 
        end_date=global_end_date.strftime('%Y-%m-%d')
    )

    if df_ff5.empty:
        st.warning("⚠️ ケネス・フレンチの5ファクターデータの取得に失敗しました。ローカルCSVまたはオンライン接続を確認してください。")

    # --- 欠落銘柄のパージ ---
    fetched_tickers = df_port_fund['Ticker'].tolist() if 'Ticker' in df_port_fund.columns else []
    missing_api = [t for t in port_tickers if t not in fetched_tickers]
    port_tickers_valid = [t for t in port_tickers if t not in missing_api]

    if missing_api:
        st.warning(f"⚠️ 以下の銘柄はAPI制限等のため取得できず、除外して計算を続行します: {', '.join(missing_api)}")

    if not port_tickers_valid:
        st.error("計算可能な銘柄がありません。")
        st.stop()

    # --- Step C: ウェイト計算 ---
    progress_bar.progress(50, text="[4/5] ウェイトの初期計算中...")
    df_port_initial = df_all_fund[df_all_fund['Ticker'].isin(port_tickers_valid)].copy()
    
    if weight_list and len(weight_list) == len(port_tickers):
        # 欠落した銘柄分を抜いてウェイトマップを作成
        valid_weight_list = [w for t, w in zip(port_tickers, weight_list) if t in port_tickers_valid]
        weight_map = dict(zip(port_tickers_valid, valid_weight_list))
        df_port_initial['Weight'] = df_port_initial['Ticker'].map(weight_map)
        df_port_initial = QuantEngine.calculate_portfolio_weights(df_port_initial, user_weights_provided=True)
    else:
        df_port_initial = QuantEngine.calculate_portfolio_weights(df_port_initial, user_weights_provided=False)

    # --- Step D: 二段構え 時系列多変量回帰分析 (ポートフォリオ + ベンチマーク) ---
    progress_bar.progress(60, text="[4/5] 月次多変量回帰分析 (ポートフォリオ) 実行中...")

    regression_results = None
    bench_regression_results = None

    if not df_ff5.empty:
        # ポートフォリオの回帰
        regression_results = QuantEngine.run_5factor_regression(
            df_hist, df_port_initial[['Ticker', 'Weight']], df_ff5, min_n_obs=24
        )

        # ベンチマークの回帰（比較基準として）
        # ベンチマーク銘柄の株価履歴をキャッシュ済みdf_histから使える銘柄のみ抽出
        progress_bar.progress(65, text="[4/5] ベンチマーク回帰分析 (比較基準) 実行中...")
        bench_tickers_in_hist = [t for t in bench_tickers if t in (df_hist.columns if not df_hist.empty else [])]
        if bench_tickers_in_hist:
            df_bench_weights = pd.DataFrame({
                'Ticker': bench_tickers_in_hist,
                'Weight': [1.0 / len(bench_tickers_in_hist)] * len(bench_tickers_in_hist)
            })
            bench_regression_results = QuantEngine.run_5factor_regression(
                df_hist, df_bench_weights, df_ff5, min_n_obs=24
            )
            if bench_regression_results:
                print(f"[Benchmark Regression] ベンチマーク回帰完了: {bench_mode}")

    portfolio_z_radar = {}
    bench_z_radar = {}
    relative_z_radar = {}
    regression_success = False

    if regression_results and regression_results.get('R_squared') is not None:
        regression_success = True
        portfolio_z_radar = {
            'Beta':       regression_results.get('Beta', 0),
            'Size':       regression_results.get('Size', 0),
            'Value':      regression_results.get('Value', 0),
            'Quality':    regression_results.get('Quality', 0),
            'Investment': regression_results.get('Investment', 0)
        }
    else:
        portfolio_z_radar = {'Beta': 1.0, 'Value': 0, 'Size': 0, 'Quality': 0, 'Investment': 0}

    # ベンチマーク回帰結果の整理
    if bench_regression_results and bench_regression_results.get('R_squared') is not None:
        bench_z_radar = {
            'Beta':       bench_regression_results.get('Beta', 1.0),
            'Size':       bench_regression_results.get('Size', 0),
            'Value':      bench_regression_results.get('Value', 0),
            'Quality':    bench_regression_results.get('Quality', 0),
            'Investment': bench_regression_results.get('Investment', 0)
        }
    else:
        # ベンチマーク回帰が取れない場合はCAPMの理論値（市場ポートフォリオ）を使用
        bench_z_radar = {'Beta': 1.0, 'Value': 0, 'Size': 0, 'Quality': 0, 'Investment': 0}

    # ポートフォリオ vs ベンチマークの差分（超過エクスポージャー）
    relative_z_radar = {k: portfolio_z_radar.get(k, 0) - bench_z_radar.get(k, 0) for k in portfolio_z_radar}

    # --- Step E: スコア計算 ---
    progress_bar.progress(80, text="[5/5] 固有スコアと寄与度の計算中...")
    # 【修正①】ベンチマーク統計はベンチマーク銘柄のデータから算出（変更なし）
    df_bench = df_all_fund[df_all_fund['Ticker'].isin(bench_tickers)]
    market_stats, _ = UniverseManager.generate_market_stats(df_bench)

    df_port_proc = QuantEngine.process_raw_factors(df_port_initial)
    df_port_scored, _ = QuantEngine.compute_z_scores(df_port_proc, market_stats)
    
    # --- Step F: 寄与度計算 ---
    progress_bar.progress(90, text="[5/5] 加重平均寄与度の集計中...")
    df_port_scored, _ = QuantEngine.calculate_weighted_factor_contributions(df_port_scored)

    # 【並べ替え対応】分析結果をsession_stateに保存し、ボタンを押さなくても並べ替えが効くようにする
    st.session_state["df_port_scored"] = df_port_scored

    progress_bar.progress(100, text="✨ 分析完了！")
    time.sleep(0.4)
    progress_bar.empty()

    # ---------------------------------------------------------
    # 3. 分析結果の表示 UI (フィードバックの高度化)
    # ---------------------------------------------------------
    
    if regression_success:
        # QuantEngineから返ってくる各種指標を展開
        adj_r2 = regression_results.get('Adjusted_R_squared', 0)
        n_obs = regression_results.get('N_Observations', 'Unknown')
        method = regression_results.get('Method', 'Unknown')
        p_val = regression_results.get('p_values', {}).get('Beta', 1.0)
        
        mode_text = "全期間 (1990~)" if analysis_mode == "利用可能な最大期間 (Max Historical)" else f"直近 {lookback_years} 年間"
        
        # 専門的フィードバックの可視化
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

        try:
            fig_radar = Visualizer.plot_radar_chart_vs_benchmark(
                portfolio_z_radar, bench_z_radar, relative_z_radar,
                bench_label=bench_mode, title_suffix=suffix, period_text=p_text
            )
        except Exception:
            # Visualizerに新メソッドがない場合は既存メソッドにフォールバック
            try:
                fig_radar = Visualizer.plot_radar_chart(portfolio_z_radar, title_suffix=suffix, period_text=p_text)
            except TypeError:
                fig_radar = Visualizer.plot_radar_chart(portfolio_z_radar, title_suffix=suffix)

        st.plotly_chart(fig_radar, width='stretch')
    with col2:
        fig_bar = Visualizer.plot_contribution_bar_chart(df_port_scored)
        st.plotly_chart(fig_bar, width='stretch')

# ---------------------------------------------------------
# 4. 詳細データテーブル（session_stateから読み出し・並べ替えはリアルタイム）
# run_button ブロックの外に配置することで、selectbox/checkboxの操作だけで
# 即座に並べ替えが反映される。ボタンを押し直す必要はない。
# ---------------------------------------------------------
if "df_port_scored" in st.session_state:
    st.markdown("---")
    st.subheader("📋 銘柄別 固有ファクタースコア & 加重平均寄与度")
    st.caption("Zスコア = ベンチマーク（市場）平均からの乖離。(+1.00σ) = 市場平均より1標準偏差高い")

    df_port_scored = st.session_state["df_port_scored"]

    rename_dict = {
        'Value_Z': 'Value (固有)', 'Quality_Z': 'Quality (固有)',
        'Investment_Z': 'Investment (固有)', 'Size_Z': 'Size (固有)',
        'Value_Z_Contrib': 'Value (寄与)', 'Quality_Z_Contrib': 'Quality (寄与)',
        'Investment_Z_Contrib': 'Investment (寄与)', 'Size_Z_Contrib': 'Size (寄与)',
        'Weight': 'ウェイト'
    }

    base_cols    = ['Ticker', 'Weight', 'Name'] if 'Name' in df_port_scored.columns else ['Ticker', 'Weight']
    score_cols   = ['Value_Z', 'Quality_Z', 'Investment_Z', 'Size_Z']
    contrib_cols = ['Value_Z_Contrib', 'Quality_Z_Contrib', 'Investment_Z_Contrib', 'Size_Z_Contrib']

    avail_cols = base_cols + [c for c in score_cols + contrib_cols if c in df_port_scored.columns]
    df_display = df_port_scored[avail_cols].copy()
    df_display.rename(columns=rename_dict, inplace=True)

    # ── 市場乖離ラベルをZスコア列の隣にかっこ書きで追加 ──────────────
    # Zスコア列に対してそれぞれ「スコア (乖離ラベル)」の文字列列を生成
    def deviation_label(z):
        """Zスコアを +Xσ 形式の乖離ラベルに変換"""
        if pd.isna(z):
            return "(-)"
        sign = "+" if z >= 0 else ""
        if abs(z) >= 2.0:
            level = "★★"   # 市場から大きく乖離
        elif abs(z) >= 1.0:
            level = "★"    # やや乖離
        else:
            level = ""
        return f"({sign}{z:.2f}σ{level})"

    # Zスコア列ごとに「数値 (乖離)」の表示列を追加
    z_display_cols = {}
    for orig_col, display_name in rename_dict.items():
        if orig_col in score_cols and display_name in df_display.columns:
            label_col = display_name + " 乖離"
            df_display[label_col] = df_port_scored[orig_col].apply(deviation_label)
            z_display_cols[display_name] = label_col

    # 表示順を整理：スコア列の直後に乖離列を挿入
    ordered_cols = []
    for col in df_display.columns:
        if col not in z_display_cols.values():  # 乖離列は後でinsert
            ordered_cols.append(col)
            if col in z_display_cols:
                ordered_cols.append(z_display_cols[col])
    df_display = df_display[ordered_cols]

    sort_options = ['ウェイト'] + [v for v in rename_dict.values() if v in df_display.columns]

    col_sort1, col_sort2 = st.columns([1, 3])
    with col_sort1:
        sort_by = st.selectbox("並べ替え基準", options=sort_options, index=0, key="sort_by_key")
    with col_sort2:
        st.markdown("<br>", unsafe_allow_html=True)
        sort_asc = st.checkbox("昇順で並べ替え", value=False, key="sort_asc_key")

    if sort_by in df_display.columns:
        df_display = df_display.sort_values(by=sort_by, ascending=sort_asc)

    # 数値列のみフォーマット（文字列の乖離列はスキップ）
    format_dict = {
        col: "{:.2f}" for col in df_display.columns
        if col not in ['Ticker', 'Name'] and col not in z_display_cols.values()
        and pd.api.types.is_numeric_dtype(df_display[col])
    }

    st.dataframe(
        df_display.style.format(format_dict)
                        .background_gradient(
                            subset=[c for c in df_display.columns if '寄与' in c and c not in z_display_cols.values()],
                            cmap='RdBu', vmin=-0.5, vmax=0.5
                        ),
        width='stretch',
        hide_index=True
    )

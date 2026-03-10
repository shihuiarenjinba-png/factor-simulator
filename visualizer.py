import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

class Visualizer:
    """
    【修正版 V17.1】可視化・グラフ生成モジュール
    ダブル・ベータ方式（回帰ベータと特性ベータ）に対応し、
    エッジ（プラス寄与）とドラッグ（マイナス寄与）の視認性を高めたプロ仕様。
    """

    @staticmethod
    def plot_radar_chart(portfolio_z_scores):
        """
        ポートフォリオの5ファクター・エクスポージャー（Zスコア）をレーダーチャートで描画。
        頂点の1つを「Regression Beta」として明示し、市場指数への純粋な連動性を表現。
        
        Parameters:
        portfolio_z_scores (dict or pd.Series): 各ファクターのZスコア
            想定キー: 'Beta', 'Value', 'Size', 'Quality', 'Investment'
        """
        # グラフ上の表示カテゴリ名（Beta を Regression Beta に変更）
        display_categories = ['Regression Beta', 'Value', 'Size', 'Quality', 'Investment']
        
        # データの取得（エンジンからは 'Beta' というキーで渡ってくる想定）
        values = [
            portfolio_z_scores.get('Beta', 0.0),
            portfolio_z_scores.get('Value', 0.0),
            portfolio_z_scores.get('Size', 0.0),
            portfolio_z_scores.get('Quality', 0.0),
            portfolio_z_scores.get('Investment', 0.0)
        ]
        
        # レーダーチャートの図形を閉じるため、始点を末尾に追加
        values_closed = values + [values[0]]
        categories_closed = display_categories + [display_categories[0]]

        # ホバーテキスト用の簡易説明（なぜこの数値になっているか）
        hover_texts = [
            "<b>Regression Beta</b><br>市場指数に対する純粋な連動性 (時系列回帰)",
            "<b>Value</b><br>割安性 (エッジが外側ほど割安)",
            "<b>Size</b><br>小型株効果 (エッジが外側ほど小型)",
            "<b>Quality</b><br>収益性・財務健全性 (高ROE等)",
            "<b>Investment</b><br>保守的経営 (低資産成長率)",
        ]
        hover_texts_closed = hover_texts + [hover_texts[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Portfolio Target',
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.4)',
            marker=dict(size=8),
            text=hover_texts_closed,
            hovertemplate="%{text}<br>スコア: %{r:.2f}<extra></extra>"
        ))

        # Zスコアの特性上、大半が -3.0 〜 +3.0 に収まるためスケールを固定。
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-3.0, 3.0],
                    tickmode='linear',
                    tick0=-3,
                    dtick=1,
                    gridcolor='lightgrey'
                ),
                angularaxis=dict(
                    gridcolor='lightgrey'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False,
            title=dict(
                text="ポートフォリオのファクター・エクスポージャー (Z-Score)",
                font=dict(size=18, color='black')
            ),
            margin=dict(l=60, r=60, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig

    @staticmethod
    def plot_contribution_bar_chart(df_portfolio):
        """
        各銘柄のファクターへの寄与度（Zスコア × 保有ウェイト）を
        積み上げ棒グラフで可視化。
        ここでのベータは「逆算で算出した特性ベータ（Sensitivity Beta）」を使用。
        
        Parameters:
        df_portfolio (pd.DataFrame): 
            必須カラム: 'Ticker', 'Weight', 'Sensitivity_Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z'
        """
        # 必須カラムを「Sensitivity_Beta_Z」に変更
        factors = ['Sensitivity_Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z']
        missing_cols = [col for col in factors if col not in df_portfolio.columns]
        
        if missing_cols:
            # データ不足時は空の警告グラフを返す
            fig = go.Figure()
            fig.update_layout(
                title=f"グラフ生成エラー: 必要なデータが不足しています ({', '.join(missing_cols)})",
                xaxis_visible=False, yaxis_visible=False
            )
            return fig

        df_contrib = df_portfolio.copy()
        
        # ウェイトが提供されていない場合は等金額投資(均等ウェイト)と仮定
        if 'Weight' not in df_contrib.columns:
            df_contrib['Weight'] = 1.0 / len(df_contrib)

        # 寄与度の計算: Zスコア × ウェイト割合
        for f in factors:
            df_contrib[f + '_Contrib'] = df_contrib[f] * df_contrib['Weight']

        contrib_cols = [f + '_Contrib' for f in factors]
        
        # Plotly Expressで扱いやすいようにデータを縦持ち(Melt)に変換
        df_melt = df_contrib.melt(
            id_vars=['Ticker'], 
            value_vars=contrib_cols,
            var_name='Factor', 
            value_name='Contribution'
        )
        
        # グラフの軸ラベル用に表示名をクリーンアップ (例: Sensitivity_Beta_Z_Contrib -> Sensitivity Beta)
        df_melt['Factor'] = df_melt['Factor'].str.replace('_Z_Contrib', '').str.replace('_', ' ')

        # カスタム・ホバーテキストの作成（エッジとドラッグの判別用）
        def create_hover_text(row):
            val = row['Contribution']
            if val > 0:
                status = "🟢 プラス寄与 (ポートフォリオのエッジ)"
            elif val < 0:
                status = "🔴 マイナス寄与 (ポートフォリオのドラッグ)"
            else:
                status = "⚪ ニュートラル"
            return f"<b>銘柄: {row['Ticker']}</b><br>ファクター: {row['Factor']}<br>寄与度: {val:.3f}<br>{status}"

        df_melt['Hover_Text'] = df_melt.apply(create_hover_text, axis=1)

        # 正負の値を表現できる relative モードで積み上げ棒グラフを作成
        # color_discrete_sequence を Bold にすることで、色が鮮やかになり判別しやすくなります
        fig = px.bar(
            df_melt, 
            x='Factor', 
            y='Contribution', 
            color='Ticker',
            custom_data=['Hover_Text'],
            title="各銘柄のファクター寄与度分解 (Sensitivity & Factors)",
            labels={'Contribution': '全体への寄与 (Z-Score × Weight)', 'Factor': 'ファクター'},
            barmode='relative',
            color_discrete_sequence=px.colors.qualitative.Bold 
        )
        
        # カスタムホバーを適用
        fig.update_traces(hovertemplate="%{customdata[0]}")
        
        # ゼロライン（基準線）をさらに黒く太く強調し、プラス/マイナスの視覚的な境界を明確化
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2.5)

        fig.update_layout(
            xaxis_title="ファクター",
            yaxis_title="寄与度スコア",
            legend_title="構成銘柄",
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='rgba(245, 245, 245, 1)' # ドラッグ(マイナス側)を際立たせるため、背景をわずかにグレーに
        )
        return fig

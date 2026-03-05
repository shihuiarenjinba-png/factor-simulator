import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

class Visualizer:
    """
    【新規追加 Module】可視化・グラフ生成モジュール
    5ファクターの分析結果を受け取り、直感的なインサイトを提供する高精細グラフを描画する。
    """

    @staticmethod
    def plot_radar_chart(portfolio_z_scores):
        """
        ポートフォリオの5ファクター・エクスポージャー（Zスコア）をレーダーチャートで描画。
        
        Parameters:
        portfolio_z_scores (dict or pd.Series): 各ファクターのZスコア
            想定キー: 'Beta', 'Value', 'Size', 'Quality', 'Investment'
        """
        categories = ['Beta', 'Value', 'Size', 'Quality', 'Investment']
        
        # データの取得（存在しない場合は0.0で保護）
        values = [portfolio_z_scores.get(cat, 0.0) for cat in categories]
        
        # レーダーチャートの図形を閉じるため、始点を末尾に追加
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Portfolio Target',
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.4)',
            marker=dict(size=8)
        ))

        # Zスコアの特性上、大半が -3.0 〜 +3.0 に収まるためスケールを固定。
        # これにより、異なるポートフォリオ間でも形状だけで偏りを比較可能にする。
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
        積み上げ棒グラフで可視化し、ポートフォリオの構成要素を分解する。
        
        Parameters:
        df_portfolio (pd.DataFrame): 
            必須カラム: 'Ticker', 'Weight', 'Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z'
        """
        # 必須カラムの存在確認（エラーハンドリング）
        factors = ['Beta_Z', 'Value_Z', 'Size_Z', 'Quality_Z', 'Investment_Z']
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
        
        # グラフの軸ラベル用に表示名をクリーンアップ (例: Beta_Z_Contrib -> Beta)
        df_melt['Factor'] = df_melt['Factor'].str.replace('_Z_Contrib', '')

        # 正負の値を表現できる relative モードで積み上げ棒グラフを作成
        fig = px.bar(
            df_melt, 
            x='Factor', 
            y='Contribution', 
            color='Ticker',
            title="各銘柄のファクター寄与度分解 (Contribution)",
            labels={'Contribution': '全体への寄与 (Z-Score × Weight)', 'Factor': 'ファクター'},
            barmode='relative',
            color_discrete_sequence=px.colors.qualitative.Set2 # 視認性の高いカラーパレット
        )
        
        # ゼロライン（基準線）を強調
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1.5)

        fig.update_layout(
            xaxis_title="ファクター",
            yaxis_title="寄与度スコア",
            legend_title="構成銘柄",
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        return fig

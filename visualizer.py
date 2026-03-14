import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

class Visualizer:
    """
    【修正版 V18.0】可視化・グラフ生成モジュール
    多変量回帰分析結果（マクロ視点）のレーダーチャートと、
    銘柄固有スコアの加重平均寄与度（ミクロ視点）の棒グラフを統合したプロ仕様。
    """

    @staticmethod
    def plot_radar_chart(portfolio_z_scores, title_suffix=""):
        """
        ポートフォリオの5ファクター・エクスポージャー（Zスコア）をレーダーチャートで描画。
        （Kenneth French 5-Factorの回帰ベータをZスコア化したものを想定）
        
        Parameters:
        portfolio_z_scores (dict or pd.Series): 各ファクターのZスコア
            想定キー: 'Beta', 'Value', 'Size', 'Quality', 'Investment'
        title_suffix (str): タイトルに追加する補足テキスト (R^2など)
        """
        # グラフ上の表示カテゴリ名
        display_categories = ['Market Beta', 'Value (HML)', 'Size (SMB)', 'Quality (RMW)', 'Investment (CMA)']
        
        # データの取得（エンジンからのキー名に対応）
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

        # ホバーテキスト用の簡易説明
        hover_texts = [
            "<b>Market Beta</b><br>市場全体に対する感応度",
            "<b>Value (HML)</b><br>割安性効果へのエクスポージャー",
            "<b>Size (SMB)</b><br>小型株効果へのエクスポージャー",
            "<b>Quality (RMW)</b><br>高収益性へのエクスポージャー",
            "<b>Investment (CMA)</b><br>保守的投資(低資産成長)へのエクスポージャー",
        ]
        hover_texts_closed = hover_texts + [hover_texts[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Portfolio (Regression)',
            line_color='#FF4B4B', # Streamlitのプライマリーカラーに近い赤
            fillcolor='rgba(255, 75, 75, 0.4)',
            marker=dict(size=8),
            text=hover_texts_closed,
            hovertemplate="%{text}<br>スコア: %{r:.2f}<extra></extra>"
        ))

        # スケールの動的調整（基本は-3から+3だが、突出している場合は広げる）
        max_val = max([abs(v) for v in values])
        dynamic_range = max(3.0, float(np.ceil(max_val * 1.2)))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-dynamic_range, dynamic_range],
                    tickmode='linear',
                    tick0=0,
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
                text=f"多変量回帰 5ファクター・エクスポージャー {title_suffix}",
                font=dict(size=16, color='black')
            ),
            margin=dict(l=60, r=60, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig

    @staticmethod
    def plot_contribution_bar_chart(df_portfolio):
        """
        【重要修正】回帰前（固有）データの「加重平均寄与度」を積み上げ棒グラフで可視化。
        ポートフォリオ全体がなぜそのファクター特性を持っているのかを、構成銘柄の積み上げで説明する。
        
        Parameters:
        df_portfolio (pd.DataFrame): 
            必須カラム: 'Ticker', 'Beta_Z_Contrib', 'Value_Z_Contrib', 'Size_Z_Contrib', 'Quality_Z_Contrib', 'Investment_Z_Contrib'
        """
        # QuantEngineで算出済みの寄与度カラム
        contrib_cols = [
            'Beta_Z_Contrib', 
            'Value_Z_Contrib', 
            'Size_Z_Contrib', 
            'Quality_Z_Contrib', 
            'Investment_Z_Contrib'
        ]
        
        missing_cols = [col for col in contrib_cols if col not in df_portfolio.columns]
        if missing_cols:
            fig = go.Figure()
            fig.update_layout(
                title=f"グラフ生成エラー: 加重平均データが不足 ({', '.join(missing_cols)})",
                xaxis_visible=False, yaxis_visible=False
            )
            return fig

        df_contrib = df_portfolio.copy()
        
        # Plotly Expressで扱いやすいようにデータを縦持ち(Melt)に変換
        df_melt = df_contrib.melt(
            id_vars=['Ticker'], 
            value_vars=contrib_cols,
            var_name='Factor', 
            value_name='Contribution'
        )
        
        # 軸ラベルを分かりやすくリネーム
        rename_map = {
            'Beta_Z_Contrib': '固有 Beta',
            'Value_Z_Contrib': '固有 Value',
            'Size_Z_Contrib': '固有 Size',
            'Quality_Z_Contrib': '固有 Quality',
            'Investment_Z_Contrib': '固有 Investment'
        }
        df_melt['Factor'] = df_melt['Factor'].map(rename_map)

        # カスタム・ホバーテキストの作成
        def create_hover_text(row):
            val = row['Contribution']
            if val > 0:
                status = "🟢 全体スコアを押し上げ"
            elif val < 0:
                status = "🔴 全体スコアを引き下げ"
            else:
                status = "⚪ ニュートラル"
            return f"<b>銘柄: {row['Ticker']}</b><br>ファクター: {row['Factor']}<br>加重寄与度: {val:.3f}<br>{status}"

        df_melt['Hover_Text'] = df_melt.apply(create_hover_text, axis=1)

        # 正負の値を表現できる relative モードで積み上げ棒グラフを作成
        fig = px.bar(
            df_melt, 
            x='Factor', 
            y='Contribution', 
            color='Ticker',
            custom_data=['Hover_Text'],
            title="銘柄固有スコアの加重平均寄与度分解",
            labels={'Contribution': 'ポートフォリオへの寄与 (固有Zスコア × ウェイト)', 'Factor': 'ファクター'},
            barmode='relative',
            color_discrete_sequence=px.colors.qualitative.Bold 
        )
        
        # カスタムホバーを適用
        fig.update_traces(hovertemplate="%{customdata[0]}")
        
        # ゼロライン（基準線）をさらに黒く太く強調
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2.5)

        fig.update_layout(
            xaxis_title="",
            yaxis_title="加重平均スコア",
            legend_title="構成銘柄",
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='rgba(245, 245, 245, 1)' # ドラッグ(マイナス側)を際立たせる背景
        )
        return fig

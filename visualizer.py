import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st

class Visualizer:
    """
    【修正版 V18.2】可視化・グラフ生成モジュール (長期・月次データ最適化版)
    多変量回帰分析結果（マクロ視点）のレーダーチャートと、
    銘柄固有スコアの加重平均寄与度（ミクロ視点）の棒グラフを統合したプロ仕様。
    グラフ内に分析期間を明示し、月次データの大きなスケール変動にも対応。
    【V19.3追加】plot_radar_chart_vs_benchmark: ポートフォリオ vs ベンチマーク比較レーダー
    """

    @staticmethod
    def plot_radar_chart(portfolio_z_scores, title_suffix="", period_text=""):
        """
        ポートフォリオの5ファクター・エクスポージャー（ZスコアまたはBeta）をレーダーチャートで描画。
        
        Parameters:
        portfolio_z_scores (dict or pd.Series): 各ファクターのZスコア
            想定キー: 'Beta', 'Value', 'Size', 'Quality', 'Investment'
        title_suffix (str): タイトルに追加する補足テキスト (Adj R^2, N数など)
        period_text (str): グラフ内に明示する分析期間 (例: "1990/01 - 2025/12")
        """
        display_categories = ['Market Beta', 'Value (HML)', 'Size (SMB)', 'Quality (RMW)', 'Investment (CMA)']
        
        values = [
            portfolio_z_scores.get('Beta', 0.0),
            portfolio_z_scores.get('Value', 0.0),
            portfolio_z_scores.get('Size', 0.0),
            portfolio_z_scores.get('Quality', 0.0),
            portfolio_z_scores.get('Investment', 0.0)
        ]
        
        values_closed = values + [values[0]]
        categories_closed = display_categories + [display_categories[0]]

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
            line_color='#FF4B4B',
            fillcolor='rgba(255, 75, 75, 0.4)',
            marker=dict(size=8),
            text=hover_texts_closed,
            hovertemplate="%{text}<br>スコア: %{r:.3f}<extra></extra>"
        ))

        safe_values = np.nan_to_num(values, nan=0.0)
        max_val = max([abs(v) for v in safe_values])
        dynamic_range = max(1.5, float(np.ceil(max_val * 1.3)))

        period_html = f"<br><span style='font-size:12px; color:gray;'>分析期間: {period_text}</span>" if period_text else ""

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-dynamic_range, dynamic_range],
                    tickmode='linear',
                    tick0=0,
                    dtick=round(dynamic_range/3, 1),
                    gridcolor='lightgrey'
                ),
                angularaxis=dict(
                    gridcolor='lightgrey'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False,
            title=dict(
                text=f"多変量回帰 5ファクター・エクスポージャー {title_suffix}{period_html}",
                font=dict(size=16, color='black')
            ),
            margin=dict(l=60, r=60, t=80, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig

    @staticmethod
    def plot_radar_chart_vs_benchmark(
        portfolio_z_scores, bench_z_scores, relative_z_scores,
        bench_label="ベンチマーク", title_suffix="", period_text=""
    ):
        """
        【V19.3新規】ポートフォリオ・ベンチマーク・超過エクスポージャーの3トレースを重ねたレーダーチャート。

        Parameters:
        portfolio_z_scores (dict): ポートフォリオの5ファクター回帰係数
        bench_z_scores (dict):     ベンチマークの5ファクター回帰係数
        relative_z_scores (dict):  差分 (ポートフォリオ - ベンチマーク) = 超過エクスポージャー
        bench_label (str):         ベンチマーク名 (例: "TOPIX Core 30", "Nikkei 225")
        title_suffix (str):        タイトルの補足 (Adj R², N数など)
        period_text (str):         分析期間テキスト
        """
        display_categories = ['Market Beta', 'Value (HML)', 'Size (SMB)', 'Quality (RMW)', 'Investment (CMA)']

        def to_values(d):
            vals = [
                d.get('Beta', 0), d.get('Value', 0), d.get('Size', 0),
                d.get('Quality', 0), d.get('Investment', 0)
            ]
            return [float(v) if pd.notna(v) else 0.0 for v in vals]

        port_vals  = to_values(portfolio_z_scores)
        bench_vals = to_values(bench_z_scores)
        rel_vals   = to_values(relative_z_scores)

        cats_closed  = display_categories + [display_categories[0]]
        port_closed  = port_vals  + [port_vals[0]]
        bench_closed = bench_vals + [bench_vals[0]]
        rel_closed   = rel_vals   + [rel_vals[0]]

        all_vals  = port_vals + bench_vals + rel_vals
        max_abs   = max((abs(v) for v in all_vals), default=1.5)
        dyn_range = max(1.5, float(np.ceil(max_abs * 1.3)))

        fig = go.Figure()

        # ① ベンチマーク（グレー・背景）
        fig.add_trace(go.Scatterpolar(
            r=bench_closed, theta=cats_closed,
            fill='toself', name=bench_label,
            line=dict(color='#888780', width=1.5, dash='dot'),
            fillcolor='rgba(136,135,128,0.15)',
            marker=dict(size=5),
            hovertemplate=f"<b>{bench_label}</b><br>%{{theta}}<br>係数: %{{r:.3f}}<extra></extra>"
        ))

        # ② ポートフォリオ（赤・前面）
        fig.add_trace(go.Scatterpolar(
            r=port_closed, theta=cats_closed,
            fill='toself', name='ポートフォリオ',
            line=dict(color='#FF4B4B', width=2),
            fillcolor='rgba(255,75,75,0.35)',
            marker=dict(size=7),
            hovertemplate="<b>ポートフォリオ</b><br>%{theta}<br>係数: %{r:.3f}<extra></extra>"
        ))

        # ③ 超過エクスポージャー（青の破線・差分）
        fig.add_trace(go.Scatterpolar(
            r=rel_closed, theta=cats_closed,
            fill='none', name=f'超過 (PF − {bench_label})',
            line=dict(color='#185FA5', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate="<b>超過エクスポージャー</b><br>%{theta}<br>差分: %{r:.3f}<extra></extra>"
        ))

        period_html = f"<br><span style='font-size:12px; color:gray;'>分析期間: {period_text}</span>" if period_text else ""

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-dyn_range, dyn_range],
                    tickmode='linear', tick0=0,
                    dtick=round(dyn_range / 3, 1),
                    gridcolor='lightgrey'
                ),
                angularaxis=dict(gridcolor='lightgrey'),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='center', x=0.5),
            title=dict(
                text=f"5ファクター・エクスポージャー vs {bench_label} {title_suffix}{period_html}",
                font=dict(size=15, color='black')
            ),
            margin=dict(l=60, r=60, t=90, b=90),
            paper_bgcolor='white', plot_bgcolor='white'
        )
        return fig

    @staticmethod
    def plot_contribution_bar_chart(df_portfolio):
        """
        回帰前（固有）データの「加重平均寄与度」を積み上げ棒グラフで可視化。
        ポートフォリオ全体がなぜそのファクター特性を持っているのかを、構成銘柄の積み上げで説明する。
        
        Parameters:
        df_portfolio (pd.DataFrame): 
            必須カラム: 'Ticker', 'Beta_Z_Contrib', 'Value_Z_Contrib', 'Size_Z_Contrib', 'Quality_Z_Contrib', 'Investment_Z_Contrib'
        """
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
        
        df_melt = df_contrib.melt(
            id_vars=['Ticker'], 
            value_vars=contrib_cols,
            var_name='Factor', 
            value_name='Contribution'
        )
        
        rename_map = {
            'Beta_Z_Contrib': '固有 Beta',
            'Value_Z_Contrib': '固有 Value',
            'Size_Z_Contrib': '固有 Size',
            'Quality_Z_Contrib': '固有 Quality',
            'Investment_Z_Contrib': '固有 Investment'
        }
        df_melt['Factor'] = df_melt['Factor'].map(rename_map)

        def create_hover_text(row):
            val = row['Contribution']
            if pd.isna(val):
                return f"<b>銘柄: {row['Ticker']}</b><br>ファクター: {row['Factor']}<br>データ不足"
                
            if val > 0:
                status = "🟢 全体スコアを押し上げ"
            elif val < 0:
                status = "🔴 全体スコアを引き下げ"
            else:
                status = "⚪ ニュートラル"
            return f"<b>銘柄: {row['Ticker']}</b><br>ファクター: {row['Factor']}<br>加重寄与度: {val:.3f}<br>{status}"

        df_melt['Hover_Text'] = df_melt.apply(create_hover_text, axis=1)

        fig = px.bar(
            df_melt, 
            x='Factor', 
            y='Contribution', 
            color='Ticker',
            custom_data=['Hover_Text'],
            title="構成銘柄ごとの加重平均寄与度 (Intrinsic Z-Scores)",
            labels={'Contribution': 'ポートフォリオへの寄与 (固有Zスコア × ウェイト)', 'Factor': 'ファクター'},
            barmode='relative',
            color_discrete_sequence=px.colors.qualitative.Bold 
        )
        
        fig.update_traces(hovertemplate="%{customdata[0]}")
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2.5)

        fig.update_layout(
            xaxis_title="",
            yaxis_title="加重平均スコア",
            legend_title="構成銘柄",
            margin=dict(l=40, r=40, t=80, b=40),
            paper_bgcolor='white',
            plot_bgcolor='rgba(245, 245, 245, 1)'
        )
        return fig

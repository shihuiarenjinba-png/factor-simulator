import io
import os
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

def create_pdf_report(payload, figs_dict):
    """
    app.py から受け取ったデータ(payload)とグラフ(figs_dict)を元にPDFを作成する
    """
    buffer = io.BytesIO()
    
    # 1. ドキュメント設定
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40,
        title="Portfolio Report"
    )
    
    # 2. 日本語フォント登録 (超堅牢版)
    font_filename = "ipaexg.ttf"
    font_name = 'IPAexGothic'
    font_registered = False
    
    # 3つの場所を順番に探す (クラウド環境対策)
    possible_paths = [
        os.path.join(os.getcwd(), font_filename),  # 1. カレントディレクトリ (最優先)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), font_filename), # 2. このファイルの隣
        font_filename # 3. ファイル名のみ
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, path))
                font_registered = True
                print(f"✅ Font loaded from: {path}") 
                break
            except Exception as e:
                print(f"⚠️ Font load failed at {path}: {e}")
                continue
    
    # フォールバック処理: 日本語フォントがない場合
    if not font_registered:
        # 万が一見つからない場合はエラー停止せず、英語フォント(Helvetica)に切り替えて続行
        print("⚠️ Japanese font not found. Fallback to Helvetica.")
        font_name = 'Helvetica' 

    # 3. スタイル定義
    styles = getSampleStyleSheet()
    
    # フォント名を動的に適用
    title_style = ParagraphStyle('JpTitle', parent=styles['Title'], fontName=font_name, fontSize=24, leading=30, spaceAfter=20)
    heading_style = ParagraphStyle('JpHeading', parent=styles['Heading2'], fontName=font_name, fontSize=14, leading=18, spaceBefore=15, spaceAfter=10, textColor=colors.darkblue)
    normal_style = ParagraphStyle('JpNormal', parent=styles['Normal'], fontName=font_name, fontSize=10.5, leading=16, spaceAfter=10)
    alert_style = ParagraphStyle('JpAlert', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14, textColor=colors.firebrick, spaceAfter=10)
    small_style = ParagraphStyle('JpSmall', parent=styles['Normal'], fontName=font_name, fontSize=9, leading=12, textColor=colors.gray, spaceAfter=5)

    # 4. コンテンツ構築
    story = []

    # --- ヘッダー ---
    # フォントがない場合に備えて英語タイトルも併記のイメージ（またはそのまま）
    title_text = "Portfolio Analysis Report" if font_name == 'Helvetica' else "ポートフォリオ詳細分析レポート"
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph(f"Date: {payload.get('date', '-')}", normal_style))
    story.append(Spacer(1, 20))

    # --- 第1章: サマリー ---
    section_title = "1. Summary" if font_name == 'Helvetica' else "1. 分析サマリー"
    story.append(Paragraph(section_title, heading_style))
    
    # 基本メトリクス
    if font_name == 'Helvetica':
        summary_text = f"""
        CAGR: <b>{payload['metrics']['CAGR']}</b><br/>
        Risk (Vol): <b>{payload['metrics']['Vol']}</b><br/>
        Sharpe Ratio: <b>{payload['metrics']['Sharpe']}</b><br/>
        Max Drawdown: <b>{payload['metrics']['MaxDD']}</b>
        """
    else:
        summary_text = f"""
        本ポートフォリオの年平均成長率(CAGR)は <b>{payload['metrics']['CAGR']}</b>、
        リスク(Volatility)は <b>{payload['metrics']['Vol']}</b> です。
        シャープレシオは <b>{payload['metrics']['Sharpe']}</b> を記録しており、
        最大ドローダウンは <b>{payload['metrics']['MaxDD']}</b> と予測されます。
        """
    story.append(Paragraph(summary_text, normal_style))
    
    # モンテカルロ統計
    if 'mc_stats' in payload:
        label = "Simulation (20Y):" if font_name == 'Helvetica' else "将来シミュレーション(20年後):"
        story.append(Paragraph(f"<b>{label}</b> {payload['mc_stats']}", small_style))

    # AI詳細レビュー
    if 'detailed_review' in payload:
        story.append(Spacer(1, 5))
        for line in payload['detailed_review'].split('\n'):
            # フォールバック時は日本語が文字化けする可能性がありますが、処理は止めません
            story.append(Paragraph(line, normal_style))

    story.append(Spacer(1, 10))

    # --- 第2章: AI診断 ---
    section_2 = "2. AI Diagnosis" if font_name == 'Helvetica' else "2. AI ポートフォリオ診断"
    story.append(Paragraph(section_2, heading_style))
    
    diag = payload.get('diagnosis', {})
    if diag:
        type_label = "Type:" if font_name == 'Helvetica' else "タイプ判定:"
        div_label = "Diversification:" if font_name == 'Helvetica' else "分散状況:"
        risk_label = "Risk:" if font_name == 'Helvetica' else "リスク評価:"
        action_label = "Action:" if font_name == 'Helvetica' else "アクションプラン:"

        story.append(Paragraph(f"<b>{type_label} {diag.get('type', '-')}</b>", normal_style))
        story.append(Paragraph(f"{div_label} {diag.get('diversification_comment', '-')}", normal_style))
        story.append(Paragraph(f"{risk_label} {diag.get('risk_comment', '-')}", alert_style))
        story.append(Paragraph(f"{action_label} {diag.get('action_plan', '-')}", normal_style))

    if 'factor_comment' in payload:
        story.append(Spacer(1, 10))
        factor_label = "Factor Analysis" if font_name == 'Helvetica' else "▼ ファクター特性分析"
        story.append(Paragraph(f"<b>{factor_label}</b>", normal_style))
        story.append(Paragraph(payload['factor_comment'], normal_style))

    story.append(PageBreak())

    # --- 第3章: チャート ---
    section_3 = "3. Chart Analysis" if font_name == 'Helvetica' else "3. 詳細チャート分析"
    story.append(Paragraph(section_3, heading_style))
    story.append(Spacer(1, 10))

    # グラフの表示順序
    plot_order = ['allocation', 'correlation', 'monte_carlo', 'cumulative', 'drawdown', 'factors', 'attribution']
    
    title_map = {
        'allocation': '■ 資産配分 (Allocation)',
        'correlation': '■ 相関マトリックス (Correlation)',
        'monte_carlo': '■ 将来シミュレーション (Monte Carlo)',
        'cumulative': '■ 累積リターン推移 (Cumulative Return)',
        'drawdown': '■ ドローダウン (Drawdown)',
        'factors': '■ ファクター感応度 (Factor Exposure)',
        'attribution': '■ 寄与度分析 (Attribution)'
    }

    for key in plot_order:
        if key in figs_dict:
            # タイトル追加
            story.append(Paragraph(title_map.get(key, f"■ {key}"), heading_style))
            
            try:
                # Plotly -> 画像変換
                fig = figs_dict[key]
                # 画像サイズ調整 (メモリ対策のため scale=1.5 程度に抑える)
                img_bytes = fig.to_image(format="png", width=850, height=480, scale=1.5)
                img_io = io.BytesIO(img_bytes)
                
                # PDF上のサイズ
                im = RLImage(img_io, width=460, height=255) 
                story.append(im)
                story.append(Spacer(1, 15))
                
                # 改ページ調整
                if key in ['monte_carlo', 'drawdown', 'correlation']: 
                    story.append(PageBreak())
                    
            except Exception as e:
                # 画像生成失敗時のエラーメッセージ
                err_msg = f"Graph Error: {e}" if font_name == 'Helvetica' else f"※グラフ生成エラー: {e}"
                story.append(Paragraph(err_msg, alert_style))

    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"PDF Build Error: {e}") # サーバーログに記録
        st.error(f"PDFビルドエラー: {e}")
        return None

from __future__ import annotations

import datetime as dt
import io
from email.parser import BytesParser
from email.policy import default
from html import escape
from pathlib import Path
import sys
from urllib.parse import parse_qs

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_provider import DataProvider
from quant_engine import QuantEngine


def _get_arg(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    return values[0] if values else default


def _get_int(params: dict[str, list[str]], key: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(float(_get_arg(params, key, str(default))))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _get_float(
    params: dict[str, list[str]],
    key: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(_get_arg(params, key, str(default)))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _parse_request(environ) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    method = environ.get("REQUEST_METHOD", "GET").upper()
    if method == "POST":
        try:
            params: dict[str, str] = {}
            files: dict[str, dict[str, object]] = {}
            content_type = environ.get("CONTENT_TYPE", "")
            content_length = int(environ.get("CONTENT_LENGTH", "0") or "0")
            body = environ["wsgi.input"].read(content_length) if content_length else environ["wsgi.input"].read()

            if content_type.startswith("application/x-www-form-urlencoded"):
                parsed = parse_qs(body.decode("utf-8", errors="ignore"), keep_blank_values=True)
                return ({key: values[0] for key, values in parsed.items()}, {})

            if content_type.startswith("multipart/form-data"):
                header = f"Content-Type: {content_type}\nMIME-Version: 1.0\n\n".encode("utf-8")
                message = BytesParser(policy=default).parsebytes(header + body)
                for part in message.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if not name:
                        continue
                    filename = part.get_filename()
                    payload = part.get_payload(decode=True) or b""
                    if filename:
                        files[name] = {
                            "filename": filename,
                            "value": payload,
                        }
                    else:
                        charset = part.get_content_charset() or "utf-8"
                        params[name] = payload.decode(charset, errors="ignore")
            return params, files
        except Exception:
            pass

    parsed = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
    return ({key: values[0] for key, values in parsed.items()}, {})


def _render_metrics(items: list[tuple[str, str]]) -> str:
    cards = []
    for label, value in items:
        cards.append(
            f"<div class='metric-card'><div class='metric-label'>{escape(label)}</div><div class='metric-value'>{escape(value)}</div></div>"
        )
    return "<div class='metric-grid'>" + "".join(cards) + "</div>"


def _frame_to_html(frame: pd.DataFrame, max_rows: int = 12) -> str:
    preview = frame.head(max_rows).copy()
    for col in preview.columns:
        if pd.api.types.is_numeric_dtype(preview[col]):
            preview[col] = preview[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
        elif pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].dt.strftime("%Y-%m-%d")
    return preview.to_html(index=False, classes="data-table", border=0, escape=True)


def _bar_chart(exposures: dict[str, float]) -> str:
    labels = [
        ("Beta", "Market Beta", "#d84a4a"),
        ("Value", "Value", "#3c6ed9"),
        ("Size", "Size", "#1e9d74"),
        ("Quality", "Quality", "#a45bd7"),
        ("Investment", "Investment", "#d98c23"),
    ]
    max_abs = max((abs(float(exposures.get(key, 0.0))) for key, _, _ in labels), default=1.0)
    max_abs = max(max_abs, 0.5)
    rows = []
    for key, label, color in labels:
        value = float(exposures.get(key, 0.0))
        width = abs(value) / max_abs * 100.0
        rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{escape(label)}</div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{width:.1f}%; background:{color}; opacity:{0.95 if value >= 0 else 0.55}'></div>"
            "</div>"
            f"<div class='bar-value'>{value:.3f}</div>"
            "</div>"
        )
    return "<div class='chart-card'><h3>回帰エクスポージャー</h3>" + "".join(rows) + "</div>"


def _normalize_codes(raw_codes: str) -> list[str]:
    codes = []
    for part in raw_codes.replace("\n", ",").split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        if cleaned.endswith(".T"):
            codes.append(cleaned)
        elif cleaned.isdigit() and len(cleaned) == 4:
            codes.append(f"{cleaned}.T")
        else:
            codes.append(cleaned)
    return list(dict.fromkeys(codes))


def _normalize_weights(raw_weights: str, count: int) -> list[float]:
    parsed = []
    for part in raw_weights.replace("\n", ",").split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        try:
            parsed.append(float(cleaned))
        except Exception:
            continue
    if len(parsed) != count or sum(parsed) <= 0:
        return [1.0 / max(count, 1)] * count
    total = sum(parsed)
    return [value / total for value in parsed]


def _extract_portfolio_from_upload(filename: str, file_bytes: bytes) -> tuple[list[str], list[float] | None, pd.DataFrame]:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "csv":
        df_up = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    elif suffix in {"xlsx", "xls"}:
        df_up = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("CSV または Excel ファイルを指定してください。")

    ticker_col = next((c for c in df_up.columns if any(k in str(c) for k in ["コード", "Ticker", "ticker", "銘柄", "ティッカー", "Symbol", "symbol", "Code", "code"])), None)
    if not ticker_col:
        best_col = None
        best_count = 0
        for col in df_up.columns:
            count = df_up[col].astype(str).str.contains(r"\b\d{4}\b", regex=True).sum()
            if count > best_count:
                best_count = int(count)
                best_col = col
        ticker_col = best_col if best_count > 0 else None

    if ticker_col is None:
        raise ValueError("銘柄コード列が見つかりませんでした。")

    raw_codes = df_up[ticker_col].astype(str).str.extract(r"(\d{4})")[0].dropna().tolist()
    codes = [f"{code}.T" for code in list(dict.fromkeys(raw_codes))]
    if not codes:
        raise ValueError("4桁コードを読み取れませんでした。")

    weight_col = next((c for c in df_up.columns if any(k in str(c) for k in ["Weight", "weight", "ウェイト", "比率", "割合", "保有", "Ratio", "ratio", "%"])), None)
    weights: list[float] | None = None
    if weight_col:
        raw_weights = pd.to_numeric(df_up[weight_col], errors="coerce").fillna(0.0).tolist()[: len(codes)]
        if sum(raw_weights) > 0:
            total = float(sum(raw_weights))
            weights = [float(value) / total for value in raw_weights]

    preview = df_up.head(12).copy()
    return codes, weights, preview


def _build_demo_case(lookback_years: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    months = max(lookback_years * 12, 60)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-01-31", periods=months, freq="ME")
    ff5 = pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.006, 0.035, months),
            "smb": rng.normal(0.001, 0.022, months),
            "hml": rng.normal(0.001, 0.019, months),
            "rmw": rng.normal(0.001, 0.015, months),
            "cma": rng.normal(0.0005, 0.013, months),
            "rf": np.full(months, 0.0004),
        },
        index=dates,
    )
    exposure = {
        "Alpha": 0.0008,
        "Beta": 1.08,
        "Size": -0.22,
        "Value": 0.34,
        "Quality": 0.41,
        "Investment": -0.17,
    }
    excess = (
        exposure["Alpha"]
        + exposure["Beta"] * ff5["mkt_rf"]
        + exposure["Size"] * ff5["smb"]
        + exposure["Value"] * ff5["hml"]
        + exposure["Quality"] * ff5["rmw"]
        + exposure["Investment"] * ff5["cma"]
        + rng.normal(0.0, 0.012, months)
    )
    hist_ret = pd.DataFrame({"DEMO.T": excess + ff5["rf"]}, index=dates)
    weights = pd.DataFrame({"Ticker": ["DEMO.T"], "Weight": [1.0]})
    return hist_ret, weights, ff5, "デモポートフォリオ"


def _build_live_case(raw_codes: str, raw_weights: str, lookback_years: int, upload_info: dict[str, object] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, pd.DataFrame | None]:
    preview = None
    if upload_info:
        tickers = upload_info["tickers"]
        weights = upload_info["weights"] or [1.0 / len(tickers)] * len(tickers)
        preview = upload_info.get("preview")
    else:
        tickers = _normalize_codes(raw_codes)
        weights = _normalize_weights(raw_weights, len(tickers))
    if not tickers:
        raise ValueError("有効な銘柄コードがありません。例: 7203, 8306, 9984")
    start_date = (dt.date.today() - dt.timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    end_date = dt.date.today().strftime("%Y-%m-%d")
    hist_ret = DataProvider.fetch_historical_prices_monthly(tickers, days=365 * lookback_years)
    ff5 = DataProvider.fetch_ken_french_5factors(start_date=start_date, end_date=end_date)
    if hist_ret.empty:
        raise ValueError("株価データの取得に失敗しました。時間を置いて再試行してください。")
    if ff5.empty:
        raise ValueError("5ファクターデータの取得に失敗しました。ネットワークかローカルCSVを確認してください。")
    weight_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    label = ", ".join(tickers[:4]) + (" ..." if len(tickers) > 4 else "")
    return hist_ret, weight_df, ff5, label, preview


def _render_page(params: dict[str, str], files: dict[str, dict[str, object]]) -> str:
    mode = params.get("mode", "demo")
    raw_codes = params.get("codes", "7203, 8306, 9984")
    raw_weights = params.get("weights", "40, 30, 30")
    lookback_years = _get_int({k: [v] for k, v in params.items()}, "lookback_years", 5, minimum=2, maximum=20)
    upload_info = None
    upload_note = ""
    if "portfolio_file" in files and files["portfolio_file"].get("value"):
        try:
            tickers, weights, preview = _extract_portfolio_from_upload(
                str(files["portfolio_file"]["filename"]),
                files["portfolio_file"]["value"],
            )
            raw_codes = ", ".join(ticker.replace(".T", "") for ticker in tickers)
            if weights:
                raw_weights = ", ".join(f"{weight * 100:.1f}" for weight in weights)
            upload_info = {
                "tickers": tickers,
                "weights": weights,
                "preview": preview,
            }
            mode = "live"
            upload_note = f"アップロードファイル {files['portfolio_file']['filename']} を優先して解析しました。"
        except Exception as exc:
            upload_note = f"アップロードの解析に失敗したため手入力へ戻しました: {exc}"

    if mode == "live":
        try:
            hist_ret, weight_df, ff5, label, preview = _build_live_case(raw_codes, raw_weights, lookback_years, upload_info=upload_info)
            note = "ライブデータで月次5ファクター回帰を実行しています。"
        except Exception as exc:
            hist_ret, weight_df, ff5, label = _build_demo_case(lookback_years)
            note = f"ライブ取得に失敗したためデモへ切り替えました: {exc}"
            mode = "demo"
            preview = upload_info.get("preview") if upload_info else None
    else:
        hist_ret, weight_df, ff5, label = _build_demo_case(lookback_years)
        note = "デモデータで回帰ロジックを確認できます。"
        preview = upload_info.get("preview") if upload_info else None

    regression = QuantEngine.run_5factor_regression(hist_ret, weight_df, ff5)
    if regression is None:
        raise ValueError("回帰分析の実行に失敗しました。入力条件を見直してください。")

    exposures = {
        "Alpha": float(regression.get("Alpha", 0.0)),
        "Beta": float(regression.get("Beta", 0.0)),
        "Value": float(regression.get("Value", 0.0)),
        "Size": float(regression.get("Size", 0.0)),
        "Quality": float(regression.get("Quality", 0.0)),
        "Investment": float(regression.get("Investment", 0.0)),
    }
    insights = QuantEngine.generate_insights(exposures)

    form = f"""
    <form class="control-grid" method="post" enctype="multipart/form-data">
      <label>モード
        <select name="mode">
          <option value="demo" {'selected' if mode == 'demo' else ''}>Demo</option>
          <option value="live" {'selected' if mode == 'live' else ''}>Live Japan stocks</option>
        </select>
      </label>
      <label>分析年数
        <input name="lookback_years" type="number" min="2" max="20" step="1" value="{lookback_years}">
      </label>
      <label class="full">証券コード
        <input name="codes" value="{escape(raw_codes)}" placeholder="7203, 8306, 9984">
      </label>
      <label class="full">ウェイト
        <input name="weights" value="{escape(raw_weights)}" placeholder="40, 30, 30">
      </label>
      <label class="full">ポートフォリオCSV / Excel
        <input name="portfolio_file" type="file" accept=".csv,.xlsx,.xls">
      </label>
      <div class="actions"><button type="submit">回帰分析を実行</button></div>
    </form>
    """

    metrics = _render_metrics(
        [
            ("分析対象", label),
            ("回帰方式", str(regression.get("Method", "-"))),
            ("サンプル数", str(regression.get("N_Observations", "-"))),
            ("決定係数 R²", f"{float(regression.get('R_squared', 0.0)):.3f}"),
            ("調整済み R²", f"{float(regression.get('Adjusted_R_squared', 0.0)):.3f}"),
            ("Alpha", f"{float(regression.get('Alpha', 0.0)):.4f}"),
        ]
    )

    table = pd.DataFrame(
        [
            {"factor": "Market Beta", "value": exposures["Beta"]},
            {"factor": "Value (HML)", "value": exposures["Value"]},
            {"factor": "Size (SMB)", "value": exposures["Size"]},
            {"factor": "Quality (RMW)", "value": exposures["Quality"]},
            {"factor": "Investment (CMA)", "value": exposures["Investment"]},
        ]
    )
    insights_html = "".join(f"<li>{escape(item)}</li>" for item in insights)

    preview_html = ""
    if preview is not None and not preview.empty:
        preview_html = (
            "<div class='table-card'><h3>アップロード内容のプレビュー</h3>"
            + _frame_to_html(preview)
            + "</div>"
        )

    upload_note_html = f"<p class='note'>{escape(upload_note)}</p>" if upload_note else ""

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Factor Simulator</title>
  <style>
    :root {{
      --ink:#1e293b; --muted:#5b677a; --line:rgba(30,41,59,0.12); --card:rgba(255,255,255,0.92);
      --accent:#d84a4a; --bg-a:#eff5ff; --bg-b:#fff6ea;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; font-family:"Avenir Next","Hiragino Sans","Yu Gothic","Meiryo",sans-serif; color:var(--ink);
      background: radial-gradient(circle at top left, rgba(255,255,255,0.92), transparent 35%), linear-gradient(135deg,var(--bg-a),var(--bg-b));
    }}
    .wrap {{ max-width:1120px; margin:0 auto; padding:28px 18px 56px; }}
    .hero,.panel,.metric-card,.table-card,.chart-card {{ background:var(--card); border:1px solid var(--line); border-radius:24px; box-shadow:0 18px 44px rgba(30,41,59,0.08); }}
    .hero {{ padding:26px 28px; margin-bottom:16px; }}
    .hero h1 {{ margin:0 0 10px 0; font-size:42px; line-height:1.05; }}
    .hero p,.note,.copy {{ color:var(--muted); line-height:1.7; }}
    .panel {{ padding:20px; }}
    .kicker {{ color:var(--accent); font-size:12px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:10px; }}
    .control-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:18px 0; }}
    .control-grid label {{ display:flex; flex-direction:column; gap:6px; font-size:14px; color:var(--muted); }}
    .control-grid label.full {{ grid-column:1 / -1; }}
    input,select,button {{ font:inherit; }}
    input,select {{
      width:100%; border:1px solid rgba(30,41,59,0.15); border-radius:14px; padding:11px 12px; background:white; color:var(--ink);
    }}
    .actions {{ display:flex; align-items:end; }}
    button {{ border:0; border-radius:14px; padding:12px 16px; background:var(--accent); color:white; font-weight:700; cursor:pointer; box-shadow:0 10px 22px rgba(216,74,74,0.28); }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:18px 0; }}
    .metric-card {{ padding:16px; }}
    .metric-label {{ font-size:13px; color:var(--muted); margin-bottom:8px; }}
    .metric-value {{ font-size:24px; font-weight:700; line-height:1.1; }}
    .two-col {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; margin-top:14px; }}
    .table-card,.chart-card {{ padding:16px; }}
    .table-card h3,.chart-card h3 {{ margin:0 0 12px 0; font-size:20px; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    .data-table th,.data-table td {{ text-align:left; padding:8px 10px; border-bottom:1px solid rgba(30,41,59,0.08); }}
    .data-table th {{ background:rgba(216,74,74,0.06); }}
    .bar-row {{ display:grid; grid-template-columns:150px 1fr 82px; gap:10px; align-items:center; margin:10px 0; }}
    .bar-label,.bar-value {{ font-size:14px; }}
    .bar-track {{ width:100%; height:14px; background:rgba(30,41,59,0.08); border-radius:999px; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px; }}
    ul {{ margin:0; padding-left:18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="kicker">Factor Simulator</div>
      <h1>月次5ファクター回帰を Vercel で実行</h1>
      <p>Vercel 向けの軽量版です。既存の数値エンジンを使って、ポートフォリオのファクター・エクスポージャーをすぐ確認できます。</p>
    </section>
    <section class="panel">
      <div class="kicker">Regression</div>
      <h2>ポートフォリオの5ファクター分析</h2>
      <p class="copy">日本株コードとウェイトを入れて、月次の Fama-French 5 Factor 回帰を実行します。まずは Demo でロジック確認、次に Live で実データ確認がおすすめです。</p>
      {form}
      {upload_note_html}
      <p class="note">{escape(note)}</p>
      {metrics}
      {preview_html}
      <div class="two-col">
        {_bar_chart(exposures)}
        <div class="table-card">
          <h3>回帰結果テーブル</h3>
          {_frame_to_html(table)}
        </div>
      </div>
      <div class="two-col">
        <div class="table-card">
          <h3>インサイト</h3>
          <ul>{insights_html}</ul>
        </div>
        <div class="table-card">
          <h3>使用ウェイト</h3>
          {_frame_to_html(weight_df)}
        </div>
      </div>
    </section>
  </div>
</body>
</html>"""


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    if path == "/health":
        body = b"ok"
        start_response("200 OK", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
        return [body]

    params, files = _parse_request(environ)
    try:
        html = _render_page(params, files)
        status = "200 OK"
    except Exception as exc:
        html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>Factor Simulator Error</title></head>
<body style="font-family:sans-serif;padding:32px;"><h1>Factor Simulator</h1>
<p>画面の生成中にエラーが発生しました。</p><pre>{escape(str(exc))}</pre></body></html>"""
        status = "500 Internal Server Error"

    body = html.encode("utf-8")
    start_response(status, [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
    return [body]

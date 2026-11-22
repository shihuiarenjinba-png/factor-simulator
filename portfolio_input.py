from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd

TICKER_ALIASES = ("コード","ticker","銘柄","ティッカー","symbol","code")
WEIGHT_ALIASES = ("weight","ウェイト","比率","割合","保有","ratio","%")


def _norm_header(value) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().lower().replace(" ", "").replace("　", "")


def _read_csv(uploaded):
    if hasattr(uploaded, "seek"):
        uploaded.seek(0)
    payload = uploaded.read()
    if isinstance(payload, str):
        text = payload.lstrip("\ufeff")
    else:
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = payload.decode("cp932")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=object, keep_default_na=False)


def load_portfolio_upload(uploaded) -> pd.DataFrame:
    name = getattr(uploaded, "name", "").lower()
    if name.endswith(".csv"):
        df = _read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
        df = pd.read_excel(uploaded, dtype=object)
    else:
        raise ValueError("CSV / Excel のみ対応しています。")
    return parse_portfolio_frame(df)


def parse_portfolio_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("ポートフォリオ入力が空です。")

    work = df.copy()
    work.columns = [str(c).strip().lstrip("\ufeff") for c in work.columns]
    work = work.dropna(how="all").reset_index(drop=True)
    if work.empty:
        raise ValueError("ポートフォリオ入力が空です。")

    ticker_col = next(
        (c for c in work.columns if any(k in _norm_header(c) for k in TICKER_ALIASES)),
        None,
    )
    if ticker_col is None:
        best_col, best_count = None, 0
        for c in work.columns:
            normalized = work[c].astype(str).map(lambda x: unicodedata.normalize("NFKC", x))
            count = normalized.str.contains(r"(?<!\d)\d{4}(?!\d)", regex=True).sum()
            if count > best_count:
                best_col, best_count = c, int(count)
        ticker_col = best_col if best_count > 0 else None
    if ticker_col is None:
        raise ValueError("銘柄コード列が見つかりません。")

    weight_col = next(
        (c for c in work.columns if any(k in _norm_header(c) for k in WEIGHT_ALIASES)),
        None,
    )

    codes: list[str] = []
    weights: list[float] = []
    seen: set[str] = set()

    for i, row in work.iterrows():
        raw_ticker = unicodedata.normalize("NFKC", str(row.get(ticker_col, ""))).strip()
        raw_weight = "" if weight_col is None else str(row.get(weight_col, "")).strip()

        if raw_ticker.lower() in {"", "nan", "none", "null", "na", "n/a"}:
            if weight_col is None or raw_weight.lower() in {"", "nan", "none", "null", "na", "n/a"}:
                continue
            raise ValueError(f"{i + 2}行目: 銘柄コードが空です。")

        match = re.search(r"(?<!\d)(\d{4})(?!\d)", raw_ticker)
        if not match:
            raise ValueError(f"{i + 2}行目: 4桁の証券コードを解釈できません。")
        code = match.group(1)
        if code in seen:
            raise ValueError(f"証券コードが重複しています: {code}")
        seen.add(code)
        codes.append(code)

        if weight_col is not None:
            clean = unicodedata.normalize("NFKC", raw_weight).replace("%", "").replace(",", "").strip()
            if not clean:
                raise ValueError(f"{i + 2}行目: ウェイトが空です。")
            try:
                value = float(clean)
            except ValueError as exc:
                raise ValueError(f"{i + 2}行目: ウェイトが数値ではありません。") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{i + 2}行目: ウェイトは0以上の有限数にしてください。")
            weights.append(value)

    if not codes:
        raise ValueError("有効な証券コードがありません。")

    if weight_col is None:
        weights = [100.0 / len(codes)] * len(codes)
    elif sum(weights) <= 0:
        raise ValueError("ウェイト合計は0より大きくしてください。")

    return pd.DataFrame({"コード": codes, "ウェイト": [float(w) for w in weights]})

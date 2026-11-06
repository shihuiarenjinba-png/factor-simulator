# EDINET SQLite データパイプライン

## 本番構成

本番Webは外部APIを呼ばない。

    取得/整形工程
        ↓
    normalized CSV
        ↓
    tools/build_edinet_sqlite.py
        ↓
    edinet_financials.sqlite3
        ↓
    Webはread-only query

## SQLiteに保存するもの

### filings
出典・会社・決算期:
- docID
- EDINETコード
- 証券コード
- 会社名
- 決算期
- 会計基準
- 業種
- 原典URL
- 原典SHA-256
- 取込日時

### financials
診断に必要な正規化財務数値:
- 売上高
- 営業利益
- 経常利益
- 当期純利益
- 総資産 / 前期総資産
- 自己資本
- 流動資産 / 流動負債
- 棚卸資産 / 前期棚卸資産
- 売上債権 / 前期売上債権

### industry_benchmarks
同一業種・同一年度・J-GAAPだけから事前計算:
- sample_count
- Q10
- Q25
- median
- Q75
- Q90

## データ投入

```bash
python tools/build_edinet_sqlite.py data/normalized_edinet.csv \
  --db data/edinet_financials.sqlite3 \
  --dataset-version 2026-11-07
```

入力CSVの雛形は `data/edinet_import_template.csv`。

投入時に:
- J-GAAP以外を拒否
- source SHA-256の形式を確認
- docID単位でupsert
- 業種別分布を全再計算
- dataset_version / updated_atを記録

## 配布

`data/edinet_financials.sqlite3` は生成物なのでソース管理には入れない。
Windows/Webの配布パッケージ作成時に、検証済みDBをアプリ資産として同梱する。

環境変数 `EDINET_DB_PATH` があれば別パスのDBを読み込める。
未指定時は `data/edinet_financials.sqlite3`。

## APIの扱い

`edinet_api.py` はデータ投入側の補助コードとしてのみ残す。
本番Streamlitページからはimportも呼出もしない。
APIキーを本番Webへ配置しない。

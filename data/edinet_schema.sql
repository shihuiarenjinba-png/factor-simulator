PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS filings (
    doc_id TEXT PRIMARY KEY,
    edinet_code TEXT NOT NULL,
    sec_code TEXT,
    company_name TEXT NOT NULL,
    fiscal_year_end TEXT NOT NULL,
    accounting_standard TEXT NOT NULL,
    industry_code TEXT NOT NULL,
    industry_label TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS financials (
    doc_id TEXT PRIMARY KEY REFERENCES filings(doc_id) ON DELETE CASCADE,
    revenue REAL NOT NULL,
    operating_profit REAL NOT NULL,
    ordinary_profit REAL NOT NULL,
    net_profit REAL,
    total_assets REAL,
    prior_total_assets REAL,
    equity REAL,
    current_assets REAL,
    current_liabilities REAL,
    inventory REAL,
    prior_inventory REAL,
    receivables REAL,
    prior_receivables REAL
);

CREATE TABLE IF NOT EXISTS industry_benchmarks (
    industry_code TEXT NOT NULL,
    industry_label TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    accounting_standard TEXT NOT NULL,
    metric TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    q10 REAL,
    q25 REAL,
    median REAL,
    q75 REAL,
    q90 REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (industry_code, fiscal_year, accounting_standard, metric)
);

CREATE TABLE IF NOT EXISTS dataset_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_filings_company_name ON filings(company_name);
CREATE INDEX IF NOT EXISTS idx_filings_edinet_code ON filings(edinet_code);
CREATE INDEX IF NOT EXISTS idx_filings_sec_code ON filings(sec_code);
CREATE INDEX IF NOT EXISTS idx_filings_industry_year
    ON filings(industry_code, fiscal_year_end, accounting_standard);

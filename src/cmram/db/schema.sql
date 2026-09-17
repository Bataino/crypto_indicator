-- CMRAM DuckDB schema (DE Spec V0.1 §6)
-- Empty tables for Phase 1 research scaffold. No fake data.

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    symbol TEXT,
    name TEXT,
    category TEXT,
    listing_date DATE,
    source TEXT,
    is_active BOOLEAN,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    universe_eligible BOOLEAN,
    universe_exclude_reason TEXT
);

CREATE TABLE IF NOT EXISTS market_daily (
    timestamp DATE,
    asset_id TEXT,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume_usd DOUBLE,
    market_cap_usd DOUBLE,
    source TEXT,
    ingested_at TIMESTAMP,
    PRIMARY KEY (timestamp, asset_id)
);

CREATE TABLE IF NOT EXISTS universe_membership (
    timestamp DATE,
    asset_id TEXT,
    band TEXT,
    market_cap_usd DOUBLE,
    history_days INTEGER,
    liquidity_pass BOOLEAN,
    amihud DOUBLE,
    reason_excluded TEXT,
    PRIMARY KEY (timestamp, asset_id, band)
);

CREATE TABLE IF NOT EXISTS social_daily (
    timestamp DATE,
    asset_id TEXT,
    source TEXT,
    mentions DOUBLE,
    unique_users DOUBLE,
    engagement DOUBLE,
    community_growth DOUBLE,
    raw_payload_ref TEXT
);

CREATE TABLE IF NOT EXISTS features_daily (
    timestamp DATE,
    asset_id TEXT,
    band TEXT,
    model TEXT,
    score_C DOUBLE,
    score_V DOUBLE,
    score_N DOUBLE,
    score_P DOUBLE,
    MREI DOUBLE,
    nsi_social DOUBLE,
    nsi_price_ext DOUBLE,
    nsi_vol_exh DOUBLE,
    nsi_mom_dec DOUBLE,
    NSI DOUBLE,
    Rotation_Gap DOUBLE,
    n_available BOOLEAN,
    small_sample BOOLEAN,
    n_in_band INTEGER,
    model_version TEXT,
    PRIMARY KEY (timestamp, asset_id, band, model)
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    timestamp DATE,
    asset_id TEXT,
    band TEXT,
    entry_price DOUBLE,
    market_cap_usd DOUBLE,
    volume_usd DOUBLE,
    MREI DOUBLE,
    NSI DOUBLE,
    Rotation_Gap DOUBLE,
    tau_m DOUBLE,
    tau_g DOUBLE,
    model TEXT,
    liquidity_pass BOOLEAN
);

CREATE TABLE IF NOT EXISTS backtest_results (
    signal_id TEXT,
    horizon_d INTEGER,
    ret DOUBLE,
    mfe DOUBLE,
    mae DOUBLE,
    benchmark_id TEXT,
    excess_ret DOUBLE
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMP,
    config_hash TEXT,
    git_or_spec_version TEXT,
    notes TEXT
);

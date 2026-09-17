"""Backtest Report: metrics by band × model × τ × horizon vs benchmarks.

Always scream SMALL SAMPLE when n assets < 50 (n=11 still tiny). Never invent performance.
Never claim alpha or statistical significance from this Phase 1 grid.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

LAGOS = ZoneInfo("Africa/Lagos")


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def _f(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def signal_counts_table(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or len(signals) == 0:
        return pd.DataFrame(
            columns=["band", "model", "tau_m", "tau_g", "n_signals", "n_assets"]
        )
    return (
        signals.groupby(["band", "model", "tau_m", "tau_g"], sort=True)
        .agg(n_signals=("signal_id", "size"), n_assets=("asset_id", "nunique"))
        .reset_index()
    )


def write_backtest_report(
    output_path: Path | str,
    *,
    results: Any = None,
    signals: pd.DataFrame | None = None,
    aggregate: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write markdown Backtest Report. Never invent performance numbers.

    If ``results``/``signals``/``aggregate`` are provided, uses them.
    Always includes explicit non-conclusion / SMALL SAMPLE language.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})

    n_assets = int(meta.get("n_assets") or 0)
    if signals is not None and len(signals) and n_assets == 0:
        n_assets = int(signals["asset_id"].nunique())
    small = bool(meta.get("small_sample", n_assets < 50))
    chart_only = bool(meta.get("chart_close_only", False))
    entry_mode = str(meta.get("entry_mode") or meta.get("entry_convention") or "next_day_open")

    now = datetime.now(tz=LAGOS).strftime("%Y-%m-%d %H:%M %Z")

    lines: list[str] = []
    lines.append("# CMRAM Phase 1 — Backtest Report")
    lines.append("")
    lines.append(f"**Generated:** {now} (Africa/Lagos)")
    lines.append("**Status:** research exploration — **NOT a trading system**")
    lines.append("")
    lines.append("## ⚠️ SMALL SAMPLE / NOT CONCLUSIVE")
    lines.append("")
    if small:
        lines.append(
            f"> **HARD SMALL SAMPLE WARNING: universe n={n_assets} assets "
            "(widened from prior n=3, still tiny).** All metrics below are "
            "**exploratory only**. Do **not** claim alpha, edge, or statistical "
            "significance. Nested τ grids inflate multiple-testing risk. "
            "No holdout confirmation. **Not conclusive.**"
        )
    else:
        lines.append(
            "> Sample may still be insufficient for significance claims. "
            "Treat grid search as exploration until a pre-registered holdout confirms."
        )
    lines.append("")
    lines.append("## Intent")
    lines.append("")
    lines.append(
        "Test whether **our** indicators (MREI / Rotation Gap) show any "
        "association with forward returns under a transparent rule. "
        "**No alpha claim. No ML. No social. No trading.**"
    )
    lines.append("")
    lines.append("## Entry convention")
    lines.append("")
    lines.append(f"- Configured: `next_day_open`")
    lines.append(f"- Effective mode: `{entry_mode}`")
    if chart_only:
        lines.append(
            "- **Limitation:** market OHLC is chart-close-only (`O=H=L=C`). "
            "Entry uses **next available bar's close as open proxy**. "
            "MFE/MAE also collapse to close path (no true intraday high/low)."
        )
    lines.append("")

    # Signal counts
    lines.append("## Signal counts by band × model × τ")
    lines.append("")
    counts = signal_counts_table(signals) if signals is not None else pd.DataFrame()
    if len(counts) == 0:
        lines.append("_No signals emitted (empty grid or no PASS rows)._")
    else:
        lines.append(
            f"Total signal rows (all τ nested): **{int(counts['n_signals'].sum())}**"
        )
        lines.append("")
        lines.append("| band | model | τ_m | τ_g | n_signals | n_assets |")
        lines.append("|------|-------|-----|-----|-----------|----------|")
        for r in counts.itertuples(index=False):
            lines.append(
                f"| {r.band} | {r.model} | {r.tau_m:g} | {r.tau_g:g} | "
                f"{int(r.n_signals)} | {int(r.n_assets)} |"
            )
    lines.append("")

    # Aggregate metrics
    lines.append("## Aggregate forward metrics")
    lines.append("")
    lines.append(
        "Grouped by band × model × τ_m × τ_g × horizon. "
        f"Benchmark for excess: `{meta.get('aggregate_benchmark', 'random')}` "
        "(same band/day when possible)."
    )
    lines.append("")
    if aggregate is None or len(aggregate) == 0:
        lines.append("_No aggregate rows (no completable forward paths)._")
    else:
        lines.append(
            "| band | model | τ_m | τ_g | h | n | mean | median | win% | best | worst | "
            "mean MFE | mean MAE | mean excess |"
        )
        lines.append(
            "|------|-------|-----|-----|---|---|------|--------|------|------|-------|"
            "----------|----------|-------------|"
        )
        show = aggregate.sort_values(
            ["band", "model", "tau_m", "tau_g", "horizon_d"], kind="mergesort"
        )
        # Cap markdown size: still show all for research honesty if not huge
        for r in show.itertuples(index=False):
            lines.append(
                f"| {r.band} | {r.model} | {r.tau_m:g} | {r.tau_g:g} | {int(r.horizon_d)} | "
                f"{int(r.n)} | {_pct(r.mean_ret)} | {_pct(r.median_ret)} | {_pct(r.win_rate)} | "
                f"{_pct(r.best)} | {_pct(r.worst)} | {_pct(r.mean_mfe)} | {_pct(r.mean_mae)} | "
                f"{_pct(r.mean_excess)} |"
            )
    lines.append("")

    # Result row counts
    n_result_rows = int(len(results)) if results is not None else 0
    lines.append("## Raw backtest_results")
    lines.append("")
    lines.append(f"- Rows written: **{n_result_rows}** (signal × horizon × benchmark)")
    lines.append(
        "- Benchmarks: random same-band member, simple momentum (top ret_7), "
        "volume breakout (top RVOL else top volume), market (BTC if present else "
        "equal-weight band index)."
    )
    lines.append("")

    lines.append("## Explicit non-conclusion")
    lines.append("")
    lines.append(
        "This report does **not** conclude that the strategy works or fails. "
        "With n still tiny (even at 11 names), coarse cross-sectional ranks, chart-close-only "
        "OHLC, nested τ combinations, and no holdout split, any apparent win rate "
        "or excess is **indistinguishable from noise** for research decisions. "
        "**Do not trade on this.** Expand the universe, fix true OHLC, and "
        "pre-register a holdout before any claim."
    )
    lines.append("")
    lines.append("---")
    lines.append("*CMRAM V0.1 Phase 1 research scaffold*")

    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path

"""Universe eligibility: exclude tokenized stocks, stables/FX, etc.

Config-driven allow/deny filter for Band A/B membership. Prefers CoinGecko
``assets.category`` when present; otherwise symbol/name/id heuristics plus
optional denylist YAML (``config/eligibility.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

REASON_DENY_ID = "deny_id"
REASON_DENY_CATEGORY = "deny_category"
REASON_DENY_KEYWORD = "deny_keyword"
REASON_DENY_ID_SUBSTRING = "deny_id_substring"


@dataclass(frozen=True)
class EligibilityDecision:
    """Result of evaluating one asset for universe membership."""

    asset_id: str
    eligible: bool
    reason: str | None = None
    detail: str | None = None

    @property
    def reason_excluded(self) -> str | None:
        """Compact reason string for membership / assets flagging."""
        if self.eligible:
            return None
        if self.reason and self.detail:
            return f"ineligible:{self.reason}:{self.detail}"
        if self.reason:
            return f"ineligible:{self.reason}"
        return "ineligible"


def _norm(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower()


def _as_str_list(cfg: Mapping[str, Any], key: str) -> list[str]:
    raw = cfg.get(key) or []
    if isinstance(raw, str):
        raw = [raw]
    return [_norm(x) for x in raw if _norm(x)]


def load_eligibility_rules(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize eligibility config into lowercase rule sets."""
    cfg = dict(config or {})
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "deny_ids": set(_as_str_list(cfg, "deny_ids")),
        "allow_ids": set(_as_str_list(cfg, "allow_ids")),
        "deny_categories": _as_str_list(cfg, "deny_categories"),
        "deny_keywords": _as_str_list(cfg, "deny_keywords"),
        "deny_id_substrings": _as_str_list(cfg, "deny_id_substrings"),
    }


def _category_denied(category: str, deny_categories: Iterable[str]) -> str | None:
    if not category:
        return None
    # category may be a single label or comma/pipe-separated list
    parts = [p.strip() for p in category.replace("|", ",").split(",") if p.strip()]
    haystacks = [category] + parts
    for cat in deny_categories:
        for h in haystacks:
            if cat == h or cat in h:
                return cat
    return None


def _keyword_hit(text: str, keywords: Iterable[str]) -> str | None:
    if not text:
        return None
    for kw in keywords:
        if kw and kw in text:
            return kw
    return None


def evaluate_asset(
    asset_id: str,
    *,
    symbol: str | None = None,
    name: str | None = None,
    category: str | None = None,
    rules: Mapping[str, Any] | None = None,
    eligibility_config: Mapping[str, Any] | None = None,
) -> EligibilityDecision:
    """Decide whether ``asset_id`` may enter Band A/B membership.

    Precedence:
      1. If eligibility disabled → eligible
      2. allow_ids → eligible (override)
      3. deny_ids → ineligible
      4. deny_categories vs assets.category
      5. deny_id_substrings vs asset_id
      6. deny_keywords vs id / symbol / name
    """
    aid = _norm(asset_id) or str(asset_id)
    rules = rules or load_eligibility_rules(eligibility_config)
    if not rules.get("enabled", True):
        return EligibilityDecision(asset_id=str(asset_id), eligible=True)

    allow_ids = rules.get("allow_ids") or set()
    if aid in allow_ids:
        return EligibilityDecision(asset_id=str(asset_id), eligible=True)

    deny_ids = rules.get("deny_ids") or set()
    if aid in deny_ids:
        return EligibilityDecision(
            asset_id=str(asset_id),
            eligible=False,
            reason=REASON_DENY_ID,
            detail=aid,
        )

    cat = _norm(category)
    hit = _category_denied(cat, rules.get("deny_categories") or [])
    if hit:
        return EligibilityDecision(
            asset_id=str(asset_id),
            eligible=False,
            reason=REASON_DENY_CATEGORY,
            detail=hit,
        )

    for sub in rules.get("deny_id_substrings") or []:
        if sub and sub in aid:
            return EligibilityDecision(
                asset_id=str(asset_id),
                eligible=False,
                reason=REASON_DENY_ID_SUBSTRING,
                detail=sub,
            )

    blob = " ".join(filter(None, [aid, _norm(symbol), _norm(name)]))
    kw = _keyword_hit(blob, rules.get("deny_keywords") or [])
    if kw:
        return EligibilityDecision(
            asset_id=str(asset_id),
            eligible=False,
            reason=REASON_DENY_KEYWORD,
            detail=kw,
        )

    return EligibilityDecision(asset_id=str(asset_id), eligible=True)


def evaluate_assets_frame(
    assets: pd.DataFrame,
    eligibility_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate each row; returns asset_id, eligible, reason_excluded."""
    rules = load_eligibility_rules(eligibility_config)
    if assets is None or len(assets) == 0:
        return pd.DataFrame(
            columns=["asset_id", "eligible", "reason_excluded", "reason", "detail"]
        )

    df = assets.copy()
    if "asset_id" not in df.columns:
        raise ValueError("assets frame missing asset_id")
    for col in ("symbol", "name", "category"):
        if col not in df.columns:
            df[col] = None

    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        d = evaluate_asset(
            str(rec["asset_id"]),
            symbol=rec.get("symbol"),
            name=rec.get("name"),
            category=rec.get("category"),
            rules=rules,
        )
        rows.append(
            {
                "asset_id": d.asset_id,
                "eligible": d.eligible,
                "reason_excluded": d.reason_excluded,
                "reason": d.reason,
                "detail": d.detail,
            }
        )
    return pd.DataFrame(rows)


def ineligible_asset_ids(
    assets: pd.DataFrame,
    eligibility_config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Map asset_id → reason_excluded for assets that fail eligibility."""
    ev = evaluate_assets_frame(assets, eligibility_config)
    out: dict[str, str] = {}
    for rec in ev.itertuples(index=False):
        if not bool(rec.eligible):
            out[str(rec.asset_id)] = str(rec.reason_excluded or "ineligible")
    return out


def filter_market_by_eligibility(
    market_daily: pd.DataFrame,
    assets: pd.DataFrame | None,
    eligibility_config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Drop ineligible assets from market_daily; return filtered + exclusion map.

    When ``assets`` is None/empty, builds a minimal frame from market ``asset_id``
    only (heuristics on id; no category/name).
    """
    if market_daily is None or len(market_daily) == 0:
        return market_daily.copy() if market_daily is not None else pd.DataFrame(), {}

    if assets is None or len(assets) == 0:
        ids = sorted({str(x) for x in market_daily["asset_id"].astype(str).unique()})
        assets = pd.DataFrame(
            {"asset_id": ids, "symbol": None, "name": None, "category": None}
        )

    excluded = ineligible_asset_ids(assets, eligibility_config)
    if not excluded:
        return market_daily.copy(), {}

    mask = ~market_daily["asset_id"].astype(str).isin(excluded)
    return market_daily.loc[mask].copy(), excluded

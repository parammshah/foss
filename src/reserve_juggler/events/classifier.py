"""
Deterministic First-Pass Financial Event Classifier.
Extracts financial concepts, classifies market categories, scores textual sentiment,
and determines affected treasury asset classes without hallucinating external model calls.
"""

from __future__ import annotations
import re
from typing import Any, List, Tuple
from reserve_juggler.contracts import EventCategory, RiskSeverity


class ClassificationResult(tuple):
    category: str
    affected_markets: List[str]
    sentiment_score: float
    confidence_status: str
    severity: str

    def __new__(cls, category: str, affected_markets: List[str], sentiment_score: float, confidence_status: str, severity: str = "MEDIUM"):
        inst = super().__new__(cls, (category, affected_markets, sentiment_score, confidence_status))
        inst.category = category
        inst.affected_markets = affected_markets
        inst.sentiment_score = sentiment_score
        inst.confidence_status = confidence_status
        inst.severity = severity
        return inst

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return getattr(self, item)
        return super().__getitem__(item)


class FinancialEventClassifier:
    """
    Deterministic rule-based concept classifier for financial headlines and wire summaries.
    """

    KEYWORD_TAXONOMY = {
        EventCategory.BANKING_STRESS.value: [
            "bank failure", "discount window", "fdic", "insolvency", "supervisory monitoring",
            "capital shortfall", "credit suisse", "svb", "emergency borrowing", "counterparty risk",
            "tier-1", "bailout", "bank run", "liquidity assistance", "deposit run", "troubled bank",
            "regional bank", "bank",
        ],
        EventCategory.WITHDRAWAL_ACCELERATION.value: [
            "deposit flight", "corporate withdrawal", "cash sweep", "depositor panic",
            "uninsured deposits", "accelerated withdrawal", "outflow surge", "redemption request",
        ],
        EventCategory.MONETARY_POLICY.value: [
            "rate hike", "rate cut", "basis points", "fomc", "ecb", "central bank", "policy rate",
            "monetary policy", "quantitative tightening", "qt", "interest rate decision",
        ],
        EventCategory.MARKET_SHOCK.value: [
            "yield spike", "curve inversion", "repo spread", "haircut widen", "margin call",
            "bond plunge", "treasury selloff", "flash crash", "liquidity freeze", "repo squeeze",
        ],
        EventCategory.MARKET_DISRUPTION.value: [
            "auction tail", "circuit breaker", "clearing fail", "settlement failure",
            "primary dealer", "market disruption", "sovereign debt ceiling",
        ],
    }

    ASSET_CLASS_KEYWORDS = {
        "CASH": ["cash", "deposit", "depositor", "withdrawal", "withdrew", "fedwire", "overdraft", "overnight", "liquidity"],
        "REPO": ["repo", "tri-party", "collateral", "haircut", "interbank", "gcf"],
        "RATES": ["rate", "rates", "fed funds", "sofr", "policy rate", "fomc", "benchmark", "bps"],
        "SOVEREIGN_BONDS": ["treasury", "ust", "bund", "sovereign", "yield", "duration", "note", "bond", "federal reserve"],
        "EQUITIES": ["stock", "equity", "s&p", "dividend", "shares", "nasdaq", "dow"],
    }

    SENTIMENT_NEGATIVE = {
        "emergency", "fail", "failure", "crisis", "plunge", "spike", "drop", "default", "freeze",
        "haircut", "panic", "run", "widening", "stress", "shortfall", "loss", "warning", "tightness",
        "hike", "raises", "inflation", "pressure", "pressures", "shock", "aggressive", "aggressively",
    }
    SENTIMENT_POSITIVE = {
        "stable", "balanced", "calm", "easing", "expansion", "growth", "surplus", "rebound", "recovery",
    }

    def classify(self, text: str = "", title: str = "", summary: str = "") -> ClassificationResult:
        """
        Classifies raw financial text into:
        (category, affected_markets, sentiment_score, confidence_status)
        """
        clean_text = f"{title} {summary} {text}".strip().lower()
        matched_category = EventCategory.UNKNOWN.value
        highest_keyword_matches = 0

        for cat, keywords in self.KEYWORD_TAXONOMY.items():
            matches = sum(1 for kw in keywords if kw in clean_text)
            if matches > highest_keyword_matches:
                highest_keyword_matches = matches
                matched_category = cat

        if highest_keyword_matches == 0:
            matched_category = EventCategory.GENERAL_NEWS.value
            confidence_status = "LOW_CONFIDENCE"
        else:
            confidence_status = "CONFIDENT"

        # Affected markets
        affected: List[str] = []
        for asset_class, kws in self.ASSET_CLASS_KEYWORDS.items():
            if any(kw in clean_text for kw in kws):
                affected.append(asset_class)

        if not affected:
            affected = ["MARKETS"]

        # Textual sentiment scoring (-1.0 to +1.0)
        tokens = set(re.findall(r"\b\w+\b", clean_text))
        neg_count = len(tokens.intersection(self.SENTIMENT_NEGATIVE))
        pos_count = len(tokens.intersection(self.SENTIMENT_POSITIVE))

        tot = neg_count + pos_count
        if tot > 0:
            sentiment = (pos_count - neg_count) / tot
        else:
            sentiment = 0.0

        severity = "CRITICAL" if sentiment < -0.5 else ("HIGH" if sentiment < -0.2 else "MEDIUM")

        return ClassificationResult(matched_category, affected, round(sentiment, 2), confidence_status, severity)

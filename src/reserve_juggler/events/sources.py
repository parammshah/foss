"""
Modular event sources for Macro-Panic Sentinel.
Supports external RSS/API feeds with RFC 2822 date parsing, deterministic classification,
rate limiting, and explicit status labeling (LIVE, PARTIAL, SYNTHETIC_DEMO, FALLBACK, UNAVAILABLE).
"""

from __future__ import annotations
import email.utils
import logging
import os
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import requests

from reserve_juggler.contracts import EventCategory, ServiceStatus
from reserve_juggler.events.classifier import FinancialEventClassifier
from reserve_juggler.events.models import RawEventRecord

logger = logging.getLogger(__name__)


def parse_rfc_or_iso_datetime(date_str: str) -> Optional[str]:
    """
    Parses RFC 2822 dates (e.g. 'Wed, 02 Oct 2026 13:00:00 GMT') or ISO 8601 strings,
    returning a normalized UTC ISO 8601 timestamp string.
    Returns None if date_str is empty, malformed, or unparseable.
    """
    if not date_str:
        return None

    clean_str = date_str.strip()
    # Try RFC 2822 first
    try:
        parsed_dt = email.utils.parsedate_to_datetime(clean_str)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
        return parsed_dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass

    # Try ISO format
    try:
        iso_str = clean_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass

    logger.warning("Unparseable date string '%s'; returning None.", date_str)
    return None


class EventSource(ABC):
    """Abstract base for event sources."""

    @abstractmethod
    def fetch_events(self, since_iso: Optional[str] = None) -> List[RawEventRecord]:
        """Fetch raw event records since a given ISO timestamp."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the event source."""
        pass

    @property
    @abstractmethod
    def is_synthetic(self) -> bool:
        """Whether the source is synthetic."""
        pass


class SyntheticEventSource(EventSource):
    """
    Explicitly labeled synthetic financial event generator.
    Provides realistic scenarios for testing and offline environments.
    """

    def __init__(self, scenario_name: str = "NORMAL"):
        self.scenario_name = scenario_name
        self._custom_events: List[RawEventRecord] = []
        self._scenario_library = self._build_scenario_library()

    @property
    def source_name(self) -> str:
        return f"SYNTHETIC_LAB_{self.scenario_name}"

    @property
    def is_synthetic(self) -> bool:
        return True

    def set_scenario(self, scenario_name: str) -> None:
        self.scenario_name = scenario_name

    def inject_event(self, event: RawEventRecord) -> None:
        """Inject a custom event into the synthetic stream."""
        self._custom_events.append(event)

    def clear_injected_events(self) -> None:
        self._custom_events.clear()

    def fetch_events(self, since_iso: Optional[str] = None) -> List[RawEventRecord]:
        base_events = list(self._scenario_library.get(self.scenario_name, []))
        all_events = base_events + list(self._custom_events)
        if since_iso:
            all_events = [e for e in all_events if e.publication_time >= since_iso]
        return all_events

    def _build_scenario_library(self) -> Dict[str, List[RawEventRecord]]:
        now = datetime.now(timezone.utc).isoformat()
        
        return {
            "NORMAL": [
                RawEventRecord(
                    event_id="SYN-NORM-01",
                    source="OFFICIAL_CENTRAL_BANK_FEED",
                    title="Federal Open Market Committee Releases Routine Meeting Minutes",
                    summary="Economic activity continues to expand at a moderate pace with stable labor markets.",
                    publication_time=now,
                    ingestion_time=now,
                    category=EventCategory.MONETARY_POLICY.value,
                    affected_markets=["RATES", "SOVEREIGN_BONDS"],
                    evidence_url="https://federalreserve.example.gov/minutes",
                    source_credibility=0.98,
                    sentiment_score=0.10,
                ),
                RawEventRecord(
                    event_id="SYN-NORM-02",
                    source="REUTERS_FINANCIAL_WIRE",
                    title="Treasury Repo Market Liquidity Remains Stable Across Tenors",
                    summary="General collateral financing rates trade inside policy corridor amidst balanced cash supply.",
                    publication_time=now,
                    ingestion_time=now,
                    category=EventCategory.GENERAL_NEWS.value,
                    affected_markets=["REPO", "CASH"],
                    evidence_url="https://reuters.example.com/repo-stability",
                    source_credibility=0.92,
                    sentiment_score=0.25,
                )
            ],
            "BANKING_STRESS": [
                RawEventRecord(
                    event_id="SYN-BANK-01",
                    source="CENTRAL_BANK_PRUDENTIAL_WATCH",
                    title="Regional Bank Placed Under Heightened Supervisory Monitoring Due to Commercial Real Estate Run",
                    summary="Regulators evaluate emergency discount window liquidity borrowing amid accelerated corporate deposit flight.",
                    publication_time=now,
                    ingestion_time=now,
                    category=EventCategory.BANKING_STRESS.value,
                    affected_markets=["BANKING_SYSTEM", "CASH", "REPO"],
                    evidence_url="https://fdic.example.gov/monitoring-notice",
                    source_credibility=0.96,
                    sentiment_score=-0.82,
                ),
                RawEventRecord(
                    event_id="SYN-BANK-02",
                    source="BLOOMBERG_CREDIT_DESK",
                    title="Interbank Counterparty Credit Spreads Widen Sharply by 35 bps",
                    summary="Tier-1 financial institutions widen haircuts on non-sovereign repo collateral following counterparty downgrade.",
                    publication_time=now,
                    ingestion_time=now,
                    category=EventCategory.MARKET_SHOCK.value,
                    affected_markets=["REPO", "CREDIT_LINES"],
                    evidence_url="https://bloomberg.example.com/repo-haircuts",
                    source_credibility=0.90,
                    sentiment_score=-0.75,
                ),
                RawEventRecord(
                    event_id="SYN-BANK-03",
                    source="FINANCIAL_TIMES_WIRE",
                    title="Corporate Treasurers Accelerate Cash Sweep Into Overnight Sovereign Bills",
                    summary="Institutional clients withdraw deposits exceeding insurance limits, seeking direct safety.",
                    publication_time=now,
                    ingestion_time=now,
                    category=EventCategory.WITHDRAWAL_ACCELERATION.value,
                    affected_markets=["CASH", "DEPOSITS"],
                    evidence_url="https://ft.example.com/cash-sweep",
                    source_credibility=0.88,
                    sentiment_score=-0.70,
                )
            ],
            "RATE_HIKE_SHOCK": [
                RawEventRecord(
                    event_id="SYN-RATE-01",
                    source="OFFICIAL_CENTRAL_BANK_FEED",
                    title="Emergency Central Bank Statement: Policy Rate Increased by 75 bps Effective Immediately",
                    summary="Committee takes aggressive front-loaded action to curb inflation expectations, warns of further firming.",
                    publication_time=now,
                    ingestion_time=now,
                    category=EventCategory.MONETARY_POLICY.value,
                    affected_markets=["RATES", "SOVEREIGN_BONDS", "EQUITIES"],
                    evidence_url="https://federalreserve.example.gov/emergency-rate-hike",
                    source_credibility=0.99,
                    sentiment_score=-0.65,
                ),
                RawEventRecord(
                    event_id="SYN-RATE-02",
                    source="WALL_STREET_JOURNAL_MARKETS",
                    title="2-Year Treasury Yield Spikes 40 bps In Biggest One-Day Jump Since 2008",
                    summary="Short-duration sovereign yield curve re-prices violently, triggering margin calls on bond portfolios.",
                    publication_time=now,
                    ingestion_time=now,
                    category=EventCategory.MARKET_SHOCK.value,
                    affected_markets=["SOVEREIGN_BONDS", "RATES"],
                    evidence_url="https://wsj.example.com/yield-jump",
                    source_credibility=0.92,
                    sentiment_score=-0.80,
                )
            ],
            "SOVEREIGN_DEBT_DISRUPTION": [
                RawEventRecord(
                    event_id="SYN-SOV-01",
                    source="DEBT_MANAGEMENT_AGENCY",
                    title="Treasury Auction Sees Tail as Primary Dealer Absorption Softens",
                    summary="Foreign official bid-to-cover falls below historical averages, prompting secondary market yield expansion.",
                    publication_time=now,
                    ingestion_time=now,
                    category=EventCategory.MARKET_DISRUPTION.value,
                    affected_markets=["SOVEREIGN_BONDS"],
                    evidence_url="https://treasury.example.gov/auction-tail",
                    source_credibility=0.95,
                    sentiment_score=-0.60,
                )
            ]
        }


class RSSFeedSource(EventSource):
    """
    Authorized RSS feed source with RFC 2822 parsing, concept classification,
    rate-limiting, and error tracking.
    """

    def __init__(
        self,
        name: str,
        feed_url: str,
        category: str = EventCategory.MONETARY_POLICY.value,
        default_credibility: float = 0.90,
        rate_limit_seconds: float = 60.0,
        timeout_seconds: float = 4.0,
    ):
        self._name = name
        self.feed_url = feed_url
        self.category = category
        self.default_credibility = default_credibility
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self._last_fetch_time: float = 0.0
        self._cached_events: List[RawEventRecord] = []
        self.classifier = FinancialEventClassifier()
        self.last_error: Optional[str] = None
        self.is_failing: bool = False

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def is_synthetic(self) -> bool:
        return False

    @property
    def status(self) -> str:
        if self.is_failing:
            return "UNAVAILABLE"
        if self._last_fetch_time > 0:
            return "LIVE"
        return "READY"

    def fetch_events(self, since_iso: Optional[str] = None) -> List[RawEventRecord]:
        now_ts = time.time()
        # Rate limit check
        if now_ts - self._last_fetch_time < self.rate_limit_seconds and self._cached_events:
            logger.info("Serving RSS events from cache to respect rate limit for %s", self._name)
            events = self._cached_events
        else:
            headers = {
                "User-Agent": "ReserveJuggler-MacroSentinel/1.0 (+https://internal.treasury/bot)",
                "Accept": "application/rss+xml, application/xml, text/xml",
            }
            
            token = os.environ.get(f"{self._name.upper()}_API_KEY")
            if token:
                headers["Authorization"] = f"Bearer {token}"

            try:
                resp = requests.get(self.feed_url, headers=headers, timeout=self.timeout_seconds)
                resp.raise_for_status()
                events = self._parse_rss_xml(resp.text)
                self._cached_events = events
                self._last_fetch_time = now_ts
                self.last_error = None
                self.is_failing = False
            except Exception as e:
                self.last_error = str(e)
                self.is_failing = True
                logger.warning("Failed to fetch RSS from %s: %s. Returning cached/empty.", self.feed_url, str(e))
                events = self._cached_events

        if since_iso:
            events = [
                e for e in events
                if e.publication_time and e.publication_time != "INVALID_TIMESTAMP" and e.publication_time >= since_iso
            ]
        return events

    def _parse_rss_xml(self, xml_text: str) -> List[RawEventRecord]:
        events: List[RawEventRecord] = []
        try:
            root = ET.fromstring(xml_text)
            channel = root.find("channel") or root
            now_iso = datetime.now(timezone.utc).isoformat()
            
            for item in channel.findall("item"):
                title_elem = item.find("title")
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Untitled"
                
                desc_elem = item.find("description")
                summary = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
                
                link_elem = item.find("link")
                link = link_elem.text.strip() if link_elem is not None and link_elem.text else None
                
                pubdate_elem = item.find("pubDate")
                raw_pub_date = pubdate_elem.text.strip() if pubdate_elem is not None and pubdate_elem.text else ""
                parsed_pub = parse_rfc_or_iso_datetime(raw_pub_date)
                pub_iso = parsed_pub if parsed_pub is not None else "INVALID_TIMESTAMP"
                
                guid_elem = item.find("guid")
                event_id = guid_elem.text.strip() if guid_elem is not None and guid_elem.text else f"RSS-{hash(title)}"
                
                # First-pass concept classification and sentiment
                classified_cat, affected, sentiment, conf_status = self.classifier.classify(title + " " + summary)
                if classified_cat == EventCategory.GENERAL_NEWS.value and self.category != EventCategory.MONETARY_POLICY.value:
                    classified_cat = self.category

                events.append(
                    RawEventRecord(
                        event_id=str(event_id),
                        source=self._name,
                        title=title,
                        summary=summary,
                        publication_time=pub_iso,
                        ingestion_time=now_iso,
                        category=classified_cat,
                        affected_markets=affected,
                        evidence_url=link,
                        source_credibility=self.default_credibility,
                        sentiment_score=sentiment,
                    )
                )
        except Exception as e:
            logger.error("XML parse error for %s: %s", self._name, str(e))
        return events


class EventSourceManager:
    """Manages active event sources, prioritizing reliability and truthful status reporting."""

    def __init__(self, use_synthetic: bool = True):
        self.use_synthetic = use_synthetic
        self.synthetic_source = SyntheticEventSource(scenario_name="NORMAL")
        self.live_sources: List[RSSFeedSource] = []
        self.status = ServiceStatus.SYNTHETIC_DEMO if use_synthetic else ServiceStatus.UNAVAILABLE
        self.last_sync_time: Optional[str] = None
        self.source_errors: Dict[str, str] = {}

    def set_mode(self, use_synthetic: bool) -> None:
        self.use_synthetic = use_synthetic
        self.status = ServiceStatus.SYNTHETIC_DEMO if use_synthetic else ServiceStatus.LIVE

    def add_live_source(self, source: RSSFeedSource) -> None:
        self.live_sources.append(source)

    def fetch_all_events(self, since_iso: Optional[str] = None) -> List[RawEventRecord]:
        self.last_sync_time = datetime.now(timezone.utc).isoformat()
        self.source_errors.clear()

        if self.use_synthetic:
            self.status = ServiceStatus.SYNTHETIC_DEMO
            return self.synthetic_source.fetch_events(since_iso=since_iso)

        if not self.live_sources:
            self.status = ServiceStatus.UNAVAILABLE
            return []

        collected: List[RawEventRecord] = []
        failing_count = 0

        for src in self.live_sources:
            evs = src.fetch_events(since_iso=since_iso)
            if src.is_failing and src.last_error:
                failing_count += 1
                self.source_errors[src.source_name] = src.last_error
            collected.extend(evs)

        if failing_count == 0:
            self.status = ServiceStatus.LIVE
        elif failing_count < len(self.live_sources):
            self.status = ServiceStatus.PARTIAL
        else:
            self.status = ServiceStatus.UNAVAILABLE

        return collected

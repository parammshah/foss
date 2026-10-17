"""
Event deduplication, clustering, and reporting acceleration tracking.
Maintains bounded stateful history across evaluation cycles to calculate genuine
reporting velocity and acceleration, and distinguishes syndicated echo coverage
from independent corroboration.
"""

from __future__ import annotations
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple
from reserve_juggler.events.models import EventCluster, RawEventRecord


class EventDeduplicator:
    """
    Deduplicates raw event streams and tracks corroboration clusters over a bounded state window.
    Retains history across evaluations to calculate reporting velocity and acceleration.
    """

    def __init__(self, similarity_threshold: float = 0.50, ttl_hours: float = 72.0):
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self._clusters: Dict[str, EventCluster] = {}
        self._seen_event_ids: Set[str] = set()
        self._previous_velocity: float = 0.0

    def _tokenize(self, text: str) -> Set[str]:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        tokens = set(cleaned.split())
        stopwords = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "is", "are", "was", "were", "by", "from", "as", "be", "this", "that", "it"
        }
        return tokens - stopwords

    def _jaccard_similarity(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union if union > 0 else 0.0

    def _evict_stale_clusters(self, now: datetime) -> None:
        """Evicts clusters older than TTL retention limit."""
        expired_keys = []
        for cid, cluster in self._clusters.items():
            try:
                last_dt = datetime.fromisoformat(cluster.last_seen.replace("Z", "+00:00"))
                if (now - last_dt).total_seconds() > self.ttl_hours * 3600.0:
                    expired_keys.append(cid)
            except Exception:
                pass
        for k in expired_keys:
            del self._clusters[k]

    def cluster_events(self, events: List[RawEventRecord]) -> List[EventCluster]:
        """
        Groups incoming raw events into clusters based on title/summary token similarity.
        Maintains bounded state across calls without resetting.
        """
        now = datetime.now(timezone.utc)
        self._evict_stale_clusters(now)
        now_iso = now.isoformat()

        new_events_count = 0

        for event in events:
            if event.event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event.event_id)
            new_events_count += 1

            ev_title_tokens = self._tokenize(event.title)
            ev_full_tokens = self._tokenize(event.title + " " + event.summary)
            matched_cluster_id: Optional[str] = None
            highest_sim = 0.0

            for cid, cluster in self._clusters.items():
                canonical = cluster.canonical_event
                can_title_tokens = self._tokenize(canonical.title)
                can_full_tokens = self._tokenize(canonical.title + " " + canonical.summary)
                
                sim_title = self._jaccard_similarity(ev_title_tokens, can_title_tokens)
                sim_full = self._jaccard_similarity(ev_full_tokens, can_full_tokens)
                sim = max(sim_title, sim_full)

                if sim >= self.similarity_threshold and sim > highest_sim:
                    highest_sim = sim
                    matched_cluster_id = cid

            if matched_cluster_id:
                # Add to existing cluster
                cluster = self._clusters[matched_cluster_id]
                cluster.duplicates.append(event)
                cluster.reporting_count += 1
                cluster.last_seen = event.publication_time or now_iso
                
                # Check if this new report has higher credibility
                if event.source_credibility > cluster.canonical_event.source_credibility:
                    old_can = cluster.canonical_event
                    cluster.canonical_event = event
                    cluster.duplicates.append(old_can)

                # Calculate velocity and acceleration
                try:
                    t_first = datetime.fromisoformat(cluster.first_seen.replace("Z", "+00:00"))
                    t_last = datetime.fromisoformat((event.publication_time or now_iso).replace("Z", "+00:00"))
                    span_hours = max((t_last - t_first).total_seconds() / 3600.0, 0.05)
                    cluster.reporting_velocity_per_hour = round(cluster.reporting_count / span_hours, 2)
                except Exception:
                    cluster.reporting_velocity_per_hour = float(cluster.reporting_count)
                cluster.reporting_acceleration = cluster.reporting_velocity_per_hour
            else:
                # Create new cluster
                new_cid = f"CLUSTER-{len(self._clusters) + 1}-{event.event_id}"
                self._clusters[new_cid] = EventCluster(
                    cluster_id=new_cid,
                    canonical_event=event,
                    duplicates=[],
                    first_seen=event.publication_time or now_iso,
                    last_seen=event.publication_time or now_iso,
                    reporting_count=1,
                    reporting_acceleration=1.0,
                    reporting_velocity_per_hour=1.0,
                )

        return list(self._clusters.values())

    def get_clusters(self) -> List[EventCluster]:
        return list(self._clusters.values())

    def reset(self) -> None:
        """Explicitly reset state (only used in test isolation)."""
        self._clusters.clear()
        self._seen_event_ids.clear()
        self._previous_velocity = 0.0

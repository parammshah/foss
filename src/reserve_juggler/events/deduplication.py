"""
Event deduplication, clustering, and reporting acceleration tracking.
Distinguishes original announcements from echo/syndicated coverage and calculates
reporting frequency and acceleration.
"""

from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple
from reserve_juggler.events.models import RawEventRecord, EventCluster


class EventDeduplicator:
    """
    Deduplicates raw event streams and tracks corroboration clusters.
    """

    def __init__(self, similarity_threshold: float = 0.55):
        self.similarity_threshold = similarity_threshold
        self._clusters: Dict[str, EventCluster] = {}
        self._seen_event_ids: Set[str] = set()

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

    def cluster_events(self, events: List[RawEventRecord]) -> List[EventCluster]:
        """
        Groups incoming raw events into clusters based on token similarity and topic.
        Identifies the canonical (original/primary) event and marks echo coverage.
        """
        for event in events:
            if event.event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event.event_id)

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

            now_iso = datetime.now(timezone.utc).isoformat()

            if matched_cluster_id:
                # Add to existing cluster
                cluster = self._clusters[matched_cluster_id]
                cluster.duplicates.append(event)
                cluster.reporting_count += 1
                cluster.last_seen = event.publication_time or now_iso
                
                # Check if this new report has higher credibility or earlier timestamp
                if event.source_credibility > cluster.canonical_event.source_credibility:
                    # Upgrade canonical
                    old_can = cluster.canonical_event
                    cluster.canonical_event = event
                    cluster.duplicates.append(old_can)

                # Update reporting acceleration (simple frequency per cluster)
                cluster.reporting_acceleration = float(cluster.reporting_count)
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
                )

        return list(self._clusters.values())

    def get_clusters(self) -> List[EventCluster]:
        return list(self._clusters.values())

    def reset(self) -> None:
        self._clusters.clear()
        self._seen_event_ids.clear()

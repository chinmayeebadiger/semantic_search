"""Metrics and threshold evaluation helpers for the semantic cache."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheMetrics:
    """Track basic cache health during semantic cache lookups."""

    hit_count: int = 0
    miss_count: int = 0

    def record_hit(self) -> None:
        """Record one cache hit."""

        self.hit_count += 1

    def record_miss(self) -> None:
        """Record one cache miss."""

        self.miss_count += 1

    def reset(self) -> None:
        """Reset cache counters after clearing the in-memory cache."""

        self.hit_count = 0
        self.miss_count = 0

    @property
    def total_requests(self) -> int:
        """Total number of cache lookups."""

        return self.hit_count + self.miss_count

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache."""

        if self.total_requests == 0:
            return 0.0
        return self.hit_count / self.total_requests


@dataclass(frozen=True)
class ThresholdEvaluation:
    """Confusion counts for one similarity threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def cache_effectiveness(self) -> float:
        """Accuracy of cache hit/miss decisions for labeled query pairs."""

        total = (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def precision(self) -> float:
        """How often cache hits were actually acceptable matches."""

        predicted_hits = self.true_positives + self.false_positives
        if predicted_hits == 0:
            return 0.0
        return self.true_positives / predicted_hits

    @property
    def recall(self) -> float:
        """How often acceptable matches were found by the cache."""

        actual_matches = self.true_positives + self.false_negatives
        if actual_matches == 0:
            return 0.0
        return self.true_positives / actual_matches


def print_threshold_evaluations(evaluations: list[ThresholdEvaluation]) -> None:
    """Print false positives, false negatives, and effectiveness by threshold."""

    print("\nSemantic cache threshold experiments")
    for evaluation in evaluations:
        print(
            f"threshold={evaluation.threshold:.2f} "
            f"false_positives={evaluation.false_positives} "
            f"false_negatives={evaluation.false_negatives} "
            f"cache_effectiveness={evaluation.cache_effectiveness:.3f} "
            f"precision={evaluation.precision:.3f} "
            f"recall={evaluation.recall:.3f}"
        )

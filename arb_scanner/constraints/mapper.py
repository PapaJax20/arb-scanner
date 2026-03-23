"""Market Relationship Mapper — identify logical relationships between Polymarket markets."""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from rapidfuzz import fuzz

from arb_scanner.models import Market

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    MUTEX = "mutex"          # Markets that cannot both resolve YES
    NESTED = "nested"        # Market A can only resolve YES if market B resolves YES
    COMPLEMENTARY = "complementary"  # Markets whose YES probabilities must sum to ~1


class MarketRelationship(BaseModel):
    """A logical relationship between two or more markets."""

    type: RelationshipType
    markets: List[Market]
    confidence: float = 0.0
    description: str = ""


# Keywords that signal mutually exclusive outcomes
_MUTEX_KEYWORDS = [
    "will", "win", "become", "be elected", "nominated", "selected",
    "champion", "winner", "mvp",
]

# Keywords that signal nested/conditional relationships
_NESTED_KEYWORDS = [
    "if", "given that", "conditional on", "assuming", "after",
]


class MarketRelationshipMapper:
    """Identify logical relationships between markets on the same platform."""

    def __init__(
        self,
        mutex_similarity_threshold: float = 75.0,
        nested_similarity_threshold: float = 65.0,
    ):
        self.mutex_similarity_threshold = mutex_similarity_threshold
        self.nested_similarity_threshold = nested_similarity_threshold

    def find_relationships(self, markets: List[Market]) -> List[MarketRelationship]:
        """Identify all relationships between the given markets."""
        relationships: List[MarketRelationship] = []

        relationships.extend(self._find_mutex_groups(markets))
        relationships.extend(self._find_nested_pairs(markets))
        relationships.extend(self._find_complementary_pairs(markets))

        logger.info(
            "Found %d relationships: %d mutex, %d nested, %d complementary",
            len(relationships),
            sum(1 for r in relationships if r.type == RelationshipType.MUTEX),
            sum(1 for r in relationships if r.type == RelationshipType.NESTED),
            sum(1 for r in relationships if r.type == RelationshipType.COMPLEMENTARY),
        )
        return relationships

    def _find_mutex_groups(self, markets: List[Market]) -> List[MarketRelationship]:
        """Find groups of markets that are mutually exclusive.

        Example: "Will Trump win 2024?" and "Will Biden win 2024?" under the
        same election event — at most one can resolve YES.
        """
        relationships: List[MarketRelationship] = []
        n = len(markets)
        visited: set[tuple[str, str]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                a, b = markets[i], markets[j]
                key = tuple(sorted([a.market_id, b.market_id]))
                if key in visited:
                    continue

                score = self._mutex_score(a, b)
                if score >= self.mutex_similarity_threshold:
                    visited.add(key)
                    relationships.append(MarketRelationship(
                        type=RelationshipType.MUTEX,
                        markets=[a, b],
                        confidence=score,
                        description=f"Mutually exclusive: '{a.title}' vs '{b.title}'",
                    ))

        return relationships

    def _find_nested_pairs(self, markets: List[Market]) -> List[MarketRelationship]:
        """Find nested/conditional relationships.

        Market A is nested inside B if A can only resolve YES when B resolves YES.
        Example: "Will X win the final?" nested inside "Will X make the semifinals?"
        """
        relationships: List[MarketRelationship] = []
        n = len(markets)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = markets[i], markets[j]
                score = self._nested_score(a, b)
                if score >= self.nested_similarity_threshold:
                    relationships.append(MarketRelationship(
                        type=RelationshipType.NESTED,
                        markets=[a, b],  # a is nested inside b (a implies b)
                        confidence=score,
                        description=f"Nested: '{a.title}' implies '{b.title}'",
                    ))

        return relationships

    def _find_complementary_pairs(self, markets: List[Market]) -> List[MarketRelationship]:
        """Find complementary markets whose probabilities should sum to ~1.

        Example: "Will it rain tomorrow?" and "Will it be dry tomorrow?"
        """
        relationships: List[MarketRelationship] = []
        n = len(markets)
        visited: set[tuple[str, str]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                a, b = markets[i], markets[j]
                key = tuple(sorted([a.market_id, b.market_id]))
                if key in visited:
                    continue

                score = self._complementary_score(a, b)
                if score >= self.mutex_similarity_threshold:
                    visited.add(key)
                    relationships.append(MarketRelationship(
                        type=RelationshipType.COMPLEMENTARY,
                        markets=[a, b],
                        confidence=score,
                        description=f"Complementary: '{a.title}' + '{b.title}' ≈ 1",
                    ))

        return relationships

    def _mutex_score(self, a: Market, b: Market) -> float:
        """Score how likely two markets are mutually exclusive.

        High score when titles share a common "event frame" but differ in the
        specific outcome (e.g., same election, different candidates).
        """
        title_sim = fuzz.token_sort_ratio(a.title, b.title)

        # Markets on the same platform with very similar titles but different
        # IDs are likely mutex alternatives under the same event.
        if a.platform == b.platform and title_sim > 60:
            # Check if titles share a common prefix/frame
            partial = fuzz.partial_ratio(a.title, b.title)
            # Boost if titles are similar but not identical
            if 60 < title_sim < 95 and partial > 70:
                return min(100.0, (title_sim + partial) / 2 + 10)

        return 0.0

    def _nested_score(self, child: Market, parent: Market) -> float:
        """Score how likely child is nested inside parent.

        A nested relationship means child=YES implies parent=YES.
        """
        # The child title should be "more specific" — often longer or containing
        # additional qualifying terms.
        child_lower = child.title.lower()
        parent_lower = parent.title.lower()

        if child.platform != parent.platform:
            return 0.0

        partial = fuzz.partial_ratio(child_lower, parent_lower)
        if partial < 50:
            return 0.0

        # The child's price should be <= parent's price (logically)
        price_consistent = child.yes_price <= parent.yes_price + 0.05

        # Child title often contains parent title as substring
        containment = fuzz.partial_ratio(parent_lower, child_lower)

        if containment > 80 and price_consistent and len(child_lower) > len(parent_lower):
            return min(100.0, containment)

        return 0.0

    def _complementary_score(self, a: Market, b: Market) -> float:
        """Score how likely two markets are complements (P(A) + P(B) ≈ 1)."""
        if a.platform != b.platform:
            return 0.0

        # Check if titles look like negations of each other
        title_sim = fuzz.token_sort_ratio(a.title, b.title)
        prob_sum = a.yes_price + b.yes_price

        # Complements should have similar titles and probabilities summing near 1
        if title_sim > 60 and 0.85 <= prob_sum <= 1.15:
            return min(100.0, title_sim + (1.0 - abs(prob_sum - 1.0)) * 20)

        return 0.0

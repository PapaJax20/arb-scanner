"""Cross-platform event matching using rapidfuzz."""

from __future__ import annotations

import logging
import re
from typing import List

from rapidfuzz import fuzz

from arb_scanner.models import Market, MatchedPair

logger = logging.getLogger(__name__)

# Words/patterns to strip for normalization
_STRIP_PATTERNS = [
    r"\bwill\b",
    r"\bthe\b",
    r"\ba\b",
    r"\ban\b",
    r"\bby\b",
    r"\bin\b\s+\d{4}",
    r"\bon\b",
    r"\bof\b",
    r"\?",
    r"[^\w\s]",
]

_STRIP_RE = re.compile("|".join(_STRIP_PATTERNS), re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Normalize a market title for comparison."""
    text = title.lower().strip()
    text = _STRIP_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


class MatchingEngine:
    """Matches markets across platforms using fuzzy string similarity."""

    def __init__(self, threshold: float = 70.0):
        self.threshold = threshold

    def find_matches(
        self,
        markets_a: List[Market],
        markets_b: List[Market],
    ) -> List[MatchedPair]:
        """Find matching markets between two lists from different platforms."""
        matches: List[MatchedPair] = []

        # Pre-normalize all titles
        norms_a = [(m, normalize_title(m.title)) for m in markets_a]
        norms_b = [(m, normalize_title(m.title)) for m in markets_b]

        for market_a, norm_a in norms_a:
            best_score = 0.0
            best_match: Market | None = None
            best_norm_b = ""

            for market_b, norm_b in norms_b:
                if market_a.platform == market_b.platform:
                    continue

                score = fuzz.token_sort_ratio(norm_a, norm_b)
                if score > best_score:
                    best_score = score
                    best_match = market_b
                    best_norm_b = norm_b

            if best_match and best_score >= self.threshold:
                matches.append(
                    MatchedPair(
                        market_a=market_a,
                        market_b=best_match,
                        similarity_score=best_score,
                        normalized_title=norm_a,
                    )
                )

        logger.info(
            "Matching %s↔%s: %d pairs found (threshold=%.0f%%)",
            markets_a[0].platform.value if markets_a else "?",
            markets_b[0].platform.value if markets_b else "?",
            len(matches),
            self.threshold,
        )
        return matches

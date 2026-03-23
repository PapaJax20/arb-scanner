"""LP Constraint Solver — detect pricing violations using linear programming."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel
from scipy.optimize import linprog

from arb_scanner.constraints.mapper import MarketRelationship, RelationshipType
from arb_scanner.models import Market

logger = logging.getLogger(__name__)


class ConstraintViolation(BaseModel):
    """A detected pricing violation from the LP solver."""

    markets: List[Market]
    relationship: MarketRelationship
    violation_magnitude: float  # How far prices deviate from feasibility
    current_prices: List[float]
    fair_prices: List[float]
    description: str = ""


class ConstraintSolver:
    """Build and solve LP constraints from market relationships.

    For a set of markets with known logical relationships, we check whether
    current prices are consistent with those constraints. If not, an
    arbitrage opportunity (constraint violation) exists.

    Constraint types:
    - MUTEX: sum of YES prices for mutually exclusive markets <= 1
    - NESTED: child price <= parent price (child implies parent)
    - COMPLEMENTARY: sum of YES prices ≈ 1
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def check_violations(
        self,
        relationships: List[MarketRelationship],
    ) -> List[ConstraintViolation]:
        """Check all relationships for pricing violations."""
        violations: List[ConstraintViolation] = []

        for rel in relationships:
            violation = self._check_single(rel)
            if violation is not None:
                violations.append(violation)

        if violations:
            logger.info("Found %d constraint violations", len(violations))
        return violations

    def _check_single(self, rel: MarketRelationship) -> Optional[ConstraintViolation]:
        """Check a single relationship for a pricing violation."""
        if rel.type == RelationshipType.MUTEX:
            return self._check_mutex(rel)
        elif rel.type == RelationshipType.NESTED:
            return self._check_nested(rel)
        elif rel.type == RelationshipType.COMPLEMENTARY:
            return self._check_complementary(rel)
        return None

    def _check_mutex(self, rel: MarketRelationship) -> Optional[ConstraintViolation]:
        """MUTEX constraint: sum of YES prices must be <= 1.

        If two markets are mutually exclusive (at most one resolves YES),
        then P(A) + P(B) <= 1. Violation means arb exists.

        LP formulation:
        - Minimize: sum of adjusted prices (find closest feasible point)
        - Subject to: sum(p_i) <= 1, 0 <= p_i <= 1
        """
        prices = np.array([m.yes_price for m in rel.markets])
        price_sum = prices.sum()

        if price_sum <= 1.0 + self.tolerance:
            return None

        # Violation exists — prices sum to more than 1
        magnitude = price_sum - 1.0

        # Find fair prices via LP: minimize distance from current prices
        # subject to sum(p) <= 1, 0 <= p <= 1
        fair = self._project_to_mutex(prices)

        return ConstraintViolation(
            markets=rel.markets,
            relationship=rel,
            violation_magnitude=magnitude,
            current_prices=prices.tolist(),
            fair_prices=fair.tolist(),
            description=(
                f"MUTEX violation: prices sum to {price_sum:.4f} > 1.0 "
                f"(excess={magnitude:.4f})"
            ),
        )

    def _check_nested(self, rel: MarketRelationship) -> Optional[ConstraintViolation]:
        """NESTED constraint: child price <= parent price.

        If market[0] (child) can only resolve YES when market[1] (parent)
        resolves YES, then P(child) <= P(parent).
        """
        if len(rel.markets) < 2:
            return None

        child = rel.markets[0]
        parent = rel.markets[1]

        if child.yes_price <= parent.yes_price + self.tolerance:
            return None

        magnitude = child.yes_price - parent.yes_price
        # Fair prices: project to feasible set {(c,p) : c <= p}
        fair = self._project_to_nested(child.yes_price, parent.yes_price)

        return ConstraintViolation(
            markets=rel.markets,
            relationship=rel,
            violation_magnitude=magnitude,
            current_prices=[child.yes_price, parent.yes_price],
            fair_prices=fair,
            description=(
                f"NESTED violation: child={child.yes_price:.4f} > "
                f"parent={parent.yes_price:.4f} (excess={magnitude:.4f})"
            ),
        )

    def _check_complementary(
        self, rel: MarketRelationship
    ) -> Optional[ConstraintViolation]:
        """COMPLEMENTARY constraint: sum of YES prices ≈ 1.

        For complementary events, P(A) + P(B) = 1. Deviation in either
        direction is a violation.
        """
        prices = np.array([m.yes_price for m in rel.markets])
        price_sum = prices.sum()
        deviation = abs(price_sum - 1.0)

        if deviation <= self.tolerance:
            return None

        # Fair prices: project to simplex {p : sum(p) = 1, p >= 0}
        fair = self._project_to_simplex(prices)

        return ConstraintViolation(
            markets=rel.markets,
            relationship=rel,
            violation_magnitude=deviation,
            current_prices=prices.tolist(),
            fair_prices=fair.tolist(),
            description=(
                f"COMPLEMENTARY violation: prices sum to {price_sum:.4f} "
                f"(deviation={deviation:.4f} from 1.0)"
            ),
        )

    def _project_to_mutex(self, prices: np.ndarray) -> np.ndarray:
        """Project prices onto the feasible set {p : sum(p) <= 1, 0 <= p <= 1}.

        Uses scipy.optimize.linprog to find the closest feasible point
        that minimizes total deviation from current prices.

        We solve: minimize sum(d_i) subject to:
          p_i + d_i >= current_i  (for overpriced direction)
          sum(p_i) <= 1
          0 <= p_i <= 1
          d_i >= 0

        Simplified: proportionally scale down prices to sum to 1.
        For the LP, we minimize |p - current|_1 subject to sum(p) <= 1.
        """
        n = len(prices)
        if prices.sum() <= 1.0:
            return prices.copy()

        # LP: minimize sum(s_i) where s_i = current_i - p_i (slack vars)
        # Variables: [p_0, ..., p_{n-1}, s_0, ..., s_{n-1}]
        # Objective: minimize sum(s_i) = [0,...,0, 1,...,1]
        c = np.zeros(2 * n)
        c[n:] = 1.0  # minimize sum of slacks

        # Inequality constraints (Ax <= b):
        # sum(p_i) <= 1
        A_ub = np.zeros((1, 2 * n))
        A_ub[0, :n] = 1.0
        b_ub = np.array([1.0])

        # Equality constraints:
        # p_i + s_i = current_i  =>  s_i = current_i - p_i
        A_eq = np.zeros((n, 2 * n))
        for i in range(n):
            A_eq[i, i] = 1.0      # p_i
            A_eq[i, n + i] = 1.0  # s_i
        b_eq = prices.copy()

        bounds = [(0.0, 1.0)] * n + [(0.0, None)] * n  # p bounds + s bounds

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if result.success:
            return np.clip(result.x[:n], 0.0, 1.0)

        # Fallback: proportional scaling
        return prices * (1.0 / prices.sum())

    def _project_to_nested(self, child_p: float, parent_p: float) -> List[float]:
        """Project (child, parent) to {(c, p) : c <= p, 0 <= c,p <= 1}.

        Closest feasible point: meet in the middle.
        """
        if child_p <= parent_p:
            return [child_p, parent_p]

        # Project to the line c = p at the midpoint
        mid = (child_p + parent_p) / 2.0
        mid = max(0.0, min(1.0, mid))
        return [mid, mid]

    def _project_to_simplex(self, prices: np.ndarray) -> np.ndarray:
        """Project prices onto the probability simplex {p : sum(p) = 1, p >= 0}.

        Uses the efficient algorithm from Duchi et al. (2008).
        """
        n = len(prices)
        sorted_prices = np.sort(prices)[::-1]  # descending
        cumsum = np.cumsum(sorted_prices)
        rho = np.max(np.where((sorted_prices - (cumsum - 1) / np.arange(1, n + 1)) > 0))
        theta = (cumsum[rho] - 1.0) / (rho + 1)
        return np.maximum(prices - theta, 0.0)

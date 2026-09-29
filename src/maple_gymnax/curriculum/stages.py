"""Movement curriculum stages and promotion gates.

All numeric thresholds are DRAFT values. Tune them after the first experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

BASELINES = ("wall_fixed", "noop", "random", "rule_based")


@dataclass(frozen=True)
class Stage:
    name: str
    num_actions: int
    obs_dim: int
    min_goal_rate: float
    min_margin: float


STAGES = (
    Stage("flat_walk", num_actions=3, obs_dim=3, min_goal_rate=0.90, min_margin=0.30),
    Stage("jump", num_actions=4, obs_dim=3, min_goal_rate=0.85, min_margin=0.25),
    Stage("gravity_slope", num_actions=4, obs_dim=5, min_goal_rate=0.80, min_margin=0.20),
    Stage("multi_platform", num_actions=6, obs_dim=7, min_goal_rate=0.75, min_margin=0.20),
    Stage("debris_dodge", num_actions=6, obs_dim=23, min_goal_rate=0.70, min_margin=0.15),
)


def promote(
    stage: Stage,
    policy_goal_rate: float,
    baseline_goal_rates: Mapping[str, float],
) -> bool:
    """Return True if the policy may advance past ``stage``.

    Requires every baseline in BASELINES to be evaluated, the policy goal rate
    to reach ``min_goal_rate``, and to beat the best baseline by ``min_margin``.
    """
    missing = [b for b in BASELINES if b not in baseline_goal_rates]
    if missing:
        raise ValueError(f"missing baseline results: {missing}")
    best = max(baseline_goal_rates[b] for b in BASELINES)
    return policy_goal_rate >= stage.min_goal_rate and (
        policy_goal_rate - best >= stage.min_margin
    )

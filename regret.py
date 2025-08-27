"""
Regret computation utilities.

This module provides functions to compute standard online-learning regret
metrics given observed rewards (or losses) and, optionally, the full
per-round payoff matrix. It supports:

- Static regret (against the single best fixed action in hindsight)
- Dynamic regret (against the best action each round)
- Rewards or losses (toggle via losses=True)

Example:
    >>> import numpy as np
    >>> from regret import regret_from_payoff_matrix
    >>> rng = np.random.default_rng(0)
    >>> T, A = 100, 5
    >>> payoffs = rng.normal(loc=0.0, scale=1.0, size=(T, A))
    >>> actions = rng.integers(low=0, high=A, size=T)
    >>> result = regret_from_payoff_matrix(payoffs, actions, mode="static")
    >>> float(result["final_regret"])  # doctest: +ELLIPSIS
    ...

Notes
-----
"Regret" here follows the common definition in online decision-making:

- For rewards R_t(a), dynamic regret at time T is:
    sum_{t=1..T} [max_a R_t(a) - R_t(a_t)]

- Static regret compares to the best fixed action a* in hindsight:
    sum_{t=1..T} [R_t(a*) - R_t(a_t)],
  where a* = argmax_a sum_{t=1..T} R_t(a).

Set losses=True to compute loss-based regret instead (min replaces max,
and signs flip accordingly).
"""

from __future__ import annotations

from typing import Dict, Literal, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _as_numpy_1d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {arr.shape}")
    return arr


def _as_numpy_2d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D; got shape {arr.shape}")
    return arr


def cumulative_regret_from_rewards(
    chosen_rewards: ArrayLike,
    optimal_rewards: ArrayLike,
    *,
    average: bool = False,
) -> Dict[str, np.ndarray | float]:
    """Compute regret given the realized reward and an optimal baseline.

    Parameters
    ----------
    chosen_rewards: array-like, shape (T,)
        Rewards obtained by the algorithm each round.
    optimal_rewards: array-like, shape (T,)
        Baseline rewards (e.g., per-round max reward or best-fixed action's reward).
    average: bool, default False
        Also compute average regret per round when True.

    Returns
    -------
    dict with keys:
        - instant_regret: (T,) array of per-round regret
        - cumulative_regret: (T,) array of cumulative regret up to each round
        - final_regret: float, cumulative regret at the final round
        - average_regret: (T,) array (only if average=True)
    """
    chosen = _as_numpy_1d(chosen_rewards, "chosen_rewards")
    optimal = _as_numpy_1d(optimal_rewards, "optimal_rewards")
    if chosen.shape[0] != optimal.shape[0]:
        raise ValueError(
            f"Length mismatch: chosen_rewards={chosen.shape[0]}, optimal_rewards={optimal.shape[0]}"
        )

    instant_regret = optimal - chosen
    cumulative_regret = np.cumsum(instant_regret)
    result: Dict[str, np.ndarray | float] = {
        "instant_regret": instant_regret,
        "cumulative_regret": cumulative_regret,
        "final_regret": float(cumulative_regret[-1]),
    }
    if average:
        rounds = np.arange(1, chosen.shape[0] + 1, dtype=float)
        result["average_regret"] = cumulative_regret / rounds
    return result


def regret_from_payoff_matrix(
    payoffs: ArrayLike,
    chosen_actions: Sequence[int] | np.ndarray,
    *,
    mode: Literal["static", "dynamic"] = "static",
    losses: bool = False,
) -> Dict[str, np.ndarray | float]:
    """Compute regret against static/dynamic baselines using a payoff matrix.

    Parameters
    ----------
    payoffs: array-like, shape (T, A)
        Per-round payoff (reward or loss) for each action.
    chosen_actions: array-like of ints, shape (T,)
        Indices of actions selected by the algorithm for each round.
    mode: {"static", "dynamic"}, default "static"
        - "static": compares to the best single action in hindsight
        - "dynamic": compares to the per-round best action
    losses: bool, default False
        If True, treats input as losses (lower is better). If False, treats as rewards.

    Returns
    -------
    dict with keys from cumulative_regret_from_rewards().
    """
    P = _as_numpy_2d(payoffs, "payoffs")
    T, A = P.shape

    actions = np.asarray(chosen_actions)
    if actions.ndim != 1 or actions.shape[0] != T:
        raise ValueError(
            f"chosen_actions must be shape (T,) with T={T}; got {actions.shape}"
        )
    if np.any((actions < 0) | (actions >= A)):
        raise ValueError("chosen_actions has indices outside [0, A)")

    # Extract realized series for the taken actions
    chosen_series = P[np.arange(T), actions]

    if not losses:
        # Rewards: larger is better
        if mode == "dynamic":
            optimal_series = P.max(axis=1)
        elif mode == "static":
            totals = P.sum(axis=0)
            a_star = int(np.argmax(totals))
            optimal_series = P[:, a_star]
        else:
            raise ValueError("mode must be 'static' or 'dynamic'")
        return cumulative_regret_from_rewards(chosen_series, optimal_series, average=True)
    else:
        # Losses: smaller is better -> reverse comparisons/signs
        if mode == "dynamic":
            optimal_series = P.min(axis=1)  # best attainable loss each round
        elif mode == "static":
            totals = P.sum(axis=0)
            a_star = int(np.argmin(totals))  # smallest cumulative loss
            optimal_series = P[:, a_star]
        else:
            raise ValueError("mode must be 'static' or 'dynamic'")

        # For losses, regret is chosen_loss - optimal_loss
        instant_regret = chosen_series - optimal_series
        cumulative_regret = np.cumsum(instant_regret)
        rounds = np.arange(1, T + 1, dtype=float)
        return {
            "instant_regret": instant_regret,
            "cumulative_regret": cumulative_regret,
            "final_regret": float(cumulative_regret[-1]),
            "average_regret": cumulative_regret / rounds,
        }


__all__ = [
    "cumulative_regret_from_rewards",
    "regret_from_payoff_matrix",
]


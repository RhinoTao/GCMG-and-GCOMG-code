#!/usr/bin/env python3
"""GCOMG simulator with strategy-specific counterfactual matching scores.

GCOMG = Great-Canonical Order Matching decided Game.  The original GCMG
strategy choice, grand-canonical participation rule, demand aggregation,
price formation, and public-information update are retained.  Actual orders
use random priority matching: all minority-side orders and m randomly selected
majority-side orders receive the Majority-Game sign; residual majority orders
receive the Minority-Game sign.

Every held strategy is evaluated through its own counterfactual order book.
For an active agent, the candidate strategy replaces that agent's actual order.
For an inactive agent, the candidate strategy is added to the actual order
book, so the hypothetical participation count is M+m+1.  A strategy's matching
role therefore depends on its own action, not on the agent's chosen strategy.
The actually submitted action keeps the actual chi and its realized matching
result; every unadopted candidate uses chi' from its own counterfactual demand.
Only two action cases (-1 and +1) are evaluated per agent, then mapped onto
all held strategies.  Virtual calculations cannot change actual prices,
public information, other agents' scores, or the market random stream.

After ordinary scoring, residual-majority losses trigger a random score
redistribution.  In a nonempty market with nonzero actual chi, (M-m)*s unit
points are assigned independently and with replacement to all N*s strategy
slots.  In an empty actual market, ordinary counterfactual score changes are
waived; their absolute values become the redistribution budget (N*s points
when L=0), so the empty market can restart without retaining its former g=-1
absorbing penalty.  Redistribution conserves only its own unit budget: score
decay and the empty-market waiver mean that total scores are not conserved.

Scores are stored per (agent, held strategy).  FSS strategy maps can still be
shared between agents.  For m>4 the full strategy pool is implicit and only
the N*s held strategies are sampled/stored.

Linear price updates remain unchanged; only a step whose linear update would
make price non-positive uses the logarithmic boundary fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


MAX_FSS_M = 10
DENSE_FSS_MAX_M = 4


class StrategySpace(Protocol):
    m: int
    actions: Optional[np.ndarray]


@dataclass(frozen=True)
class FSSStrategySpace:
    """Descriptor of the complete FSS; ``actions`` is materialized only m<=4.

    The complete strategy count is 2**(2**m).  For m>4, ``actions`` is None:
    the complete pool is represented implicitly and strategies are sampled
    directly as length-P independent fair bit strings.
    """

    m: int
    actions: Optional[np.ndarray] = None

    @property
    def P(self) -> int:
        return 1 << self.m

    @property
    def n_strategies(self) -> int:
        # Python integers can represent 2**1024 and larger exactly.
        return 1 << self.P

    @property
    def is_materialized(self) -> bool:
        return self.actions is not None


def build_fss(m: int, verify: bool = True) -> FSSStrategySpace:
    """Describe the FSS; materialize its matrix only for 1<=m<=4.

    For m>=5 the full strategy pool is far too large to store.  A single
    strategy is simply a uniformly random P-bit vector, so the simulator can
    sample each agent's holdings without enumerating the full pool.
    """
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
        raise ValueError("m must be an integer >= 1.")
    m = int(m)
    if m > MAX_FSS_M:
        raise ValueError(
            f"This implementation supports 1 <= m <= {MAX_FSS_M}; got m={m}."
        )

    P = 1 << m
    if m > DENSE_FSS_MAX_M:
        return FSSStrategySpace(m=m, actions=None)

    n_strategies = 1 << P
    strategy_ids = np.arange(n_strategies, dtype=np.uint32)
    bit_positions = np.arange(P, dtype=np.uint32)
    bits = (strategy_ids[:, None] >> bit_positions[None, :]) & 1
    actions = (2 * bits.astype(np.int8) - 1).astype(np.int8, copy=False)

    if verify:
        if actions.shape != (n_strategies, P):
            raise RuntimeError("FSS construction failed: wrong action-matrix shape.")
        if not np.all(np.isin(actions, (-1, 1))):
            raise RuntimeError("FSS construction failed: actions must be +/-1.")
        if not np.all(actions[0] == -1) or not np.all(actions[-1] == 1):
            raise RuntimeError("FSS construction failed: binary encoding is inconsistent.")
    return FSSStrategySpace(m=m, actions=actions)


def _bits_to_python_ids(bits: np.ndarray) -> list[int]:
    """Encode rows of 0/1 bits as exact Python integer FSS strategy IDs."""
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), byteorder="little", signed=False) for row in packed]


def _sample_fss_holdings(
    N: int,
    s: int,
    P: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly sample s distinct FSS strategies per agent without enumerating FSS.

    Returns (unique_actions, holdings_indices, unique_strategy_ids,
    holdings_strategy_ids).  Different agents sample independently, so they
    may share strategies.  Each individual agent's s strategies are distinct.

    Rejection sampling of independent fair P-bit strings is exactly uniform
    sampling without replacement from the FSS for each agent.
    """
    held_bits = np.empty((N, s, P), dtype=np.uint8)
    held_ids: list[list[int]] = []
    for i in range(N):
        seen: set[bytes] = set()
        ids_i: list[int] = []
        j = 0
        while j < s:
            bit_row = rng.integers(0, 2, size=P, dtype=np.uint8)
            key = bit_row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            held_bits[i, j] = bit_row
            ids_i.append(int.from_bytes(np.packbits(bit_row, bitorder="little").tobytes(), "little"))
            j += 1
        held_ids.append(ids_i)

    unique_bits, inverse = np.unique(
        held_bits.reshape(N * s, P), axis=0, return_inverse=True
    )
    unique_actions = (2 * unique_bits.astype(np.int8) - 1).astype(np.int8, copy=False)
    unique_id_list = _bits_to_python_ids(unique_bits)

    # Use compact numeric dtypes whenever possible; Python ints are required
    # once a strategy ID is wider than 64 bits (m>=7).
    if P <= 64:
        unique_ids = np.asarray(unique_id_list, dtype=np.uint64)
        holdings_ids = np.asarray(held_ids, dtype=np.uint64)
    else:
        unique_ids = np.asarray(unique_id_list, dtype=object)
        holdings_ids = np.asarray(held_ids, dtype=object)

    holdings_indices = inverse.reshape(N, s).astype(np.int32, copy=False)
    return unique_actions, holdings_indices, unique_ids, holdings_ids


@dataclass
class GCOMGConfig:
    # Parameters explicitly listed on lecture p.25
    N: int = 501
    m: int = 3
    s: int = 2
    r: float = 4.0
    T: float = 100.0
    L: float = 0.0

    # Price formation also contains lambda on p.25; it is included explicitly.
    lam: Optional[float] = None  # None -> 10*N, a numerical depth convention
    x0: float = 100.0
    price_mode: str = "linear"  # "linear" follows p.25; "log" follows p.14 alternative

    steps: int = 100_000
    seed: int = 123
    mu0: Optional[int] = None
    initial_score_offset: float = 10.0  # default: initialize scores at r + offset
    initial_score: Optional[float] = None  # optional absolute override (e.g. 0)
    zero_score_tie: str = "zero"  # direct-config default; CLI defaults to "random"
    tie_tolerance: float = 1e-12

    def resolved_lambda(self) -> float:
        value = 10.0 * self.N if self.lam is None else float(self.lam)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("lambda (lam) must be a finite positive number.")
        return value

    def validate(self, n_strategies: Optional[int] = None) -> None:
        if self.N < 1:
            raise ValueError("N must be >= 1.")
        if self.m < 1:
            raise ValueError("m must be >= 1.")
        if self.s < 1:
            raise ValueError("s must be >= 1.")
        if self.T < 1:
            raise ValueError("T must be >= 1 so alpha=1-1/T is in [0,1).")
        if self.steps < 1:
            raise ValueError("steps must be >= 1.")
        if self.x0 <= 0 or not np.isfinite(self.x0):
            raise ValueError("x0 must be finite and positive.")
        if self.price_mode not in {"linear", "log"}:
            raise ValueError("price_mode must be 'linear' or 'log'.")
        if not np.isfinite(self.initial_score_offset) or self.initial_score_offset < 0:
            raise ValueError("initial_score_offset must be finite and non-negative.")
        if self.initial_score is not None and not np.isfinite(self.initial_score):
            raise ValueError("initial_score must be finite when specified.")
        if self.zero_score_tie not in {"random", "zero"}:
            raise ValueError("zero_score_tie must be 'random' or 'zero'.")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative.")
        self.resolved_lambda()
        if n_strategies is not None and self.s > n_strategies:
            raise ValueError(
                f"s={self.s} exceeds the available strategy count {n_strategies}."
            )
        if self.mu0 is not None and not (0 <= self.mu0 < (1 << self.m)):
            raise ValueError(f"mu0 must lie in [0, {(1 << self.m) - 1}].")




@dataclass
class MarketMaker:
    """Market maker with linear updates and a one-step boundary fallback.

    With mode='linear', the original x_new=x_old+D/lambda update is used
    whenever it gives a positive price.  If and only if that update would be
    non-positive, this single step uses x_new=x_old*exp(D/lambda).  The next
    step returns to the linear rule automatically.
    """

    lam: float
    price: float
    mode: str = "linear"
    last_fallback: bool = field(default=False, init=False)

    def clear(self, excess_demand: int) -> tuple[float, float, float, float]:
        if self.lam <= 0 or self.price <= 0 or not np.isfinite(self.price):
            raise ValueError("MarketMaker requires finite positive lambda and price.")
        impact = float(excess_demand) / self.lam
        old_price = self.price
        self.last_fallback = False

        if self.mode == "linear":
            new_price = old_price + impact
            if new_price > 0.0:
                # Keep these operations exactly as in the previous linear code.
                simple_ret = impact / old_price
                log_ret = float(np.log(new_price / old_price))
                price_change = impact
            else:
                # Boundary-only fallback to the lecture's logarithmic rule:
                # log(x_new)-log(x_old)=D/lambda.
                self.last_fallback = True
                with np.errstate(under="ignore"):
                    new_price = old_price * float(np.exp(impact))
                if new_price <= 0.0:
                    # Preserve strict positivity in extreme underflow cases.
                    new_price = float(np.nextafter(0.0, 1.0))
                    log_ret = float(np.log(new_price) - np.log(old_price))
                    simple_ret = float(np.expm1(log_ret))
                else:
                    log_ret = impact
                    simple_ret = float(np.expm1(impact))
                # This array/CSV field is the actual price increment on the
                # fallback step; ordinary linear steps remain exactly D/lambda.
                price_change = new_price - old_price
        elif self.mode == "log":
            # Explicit log mode remains unchanged from the previous version.
            log_ret = impact
            simple_ret = float(np.expm1(impact))
            new_price = old_price * float(np.exp(impact))
            price_change = new_price - old_price
        else:
            raise ValueError("MarketMaker mode must be 'linear' or 'log'.")

        self.price = new_price
        return new_price, price_change, simple_ret, log_ret


@dataclass
class GCOMGResult:
    config: GCOMGConfig
    strategy_space: FSSStrategySpace
    demand: np.ndarray
    buy_orders: np.ndarray
    sell_orders: np.ndarray
    active_agents: np.ndarray
    majority_count: np.ndarray
    minority_count: np.ndarray
    priority_count: np.ndarray
    majority_game_agents: np.ndarray
    minority_game_agents: np.ndarray
    majority_direction: np.ndarray
    price: np.ndarray
    price_change: np.ndarray
    simple_returns: np.ndarray
    log_returns: np.ndarray
    global_info: np.ndarray
    appended_bit: np.ndarray
    score_signal: np.ndarray
    redistribution_units: np.ndarray
    max_redistribution_to_one_strategy: np.ndarray
    empty_market: np.ndarray
    final_scores: np.ndarray
    total_redistribution_received: np.ndarray
    final_holdings: np.ndarray
    sampled_strategy_actions: Optional[np.ndarray] = None
    price_fallback: Optional[np.ndarray] = None


def _pick_best_strategies(
    scores: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick each agent's highest-score *personal* strategy; randomize ties.

    ``scores`` has shape (N, s).  In GCOMG, identical strategy maps held by
    different agents can receive different scores because their owners can
    experience different random matching roles.
    """
    best_scores = scores.max(axis=1)
    tie_mask = scores == best_scores[:, None]
    tie_random = rng.random(scores.shape)
    tie_random[~tie_mask] = -1.0
    chosen_columns = np.argmax(tie_random, axis=1)
    return chosen_columns, best_scores


def random_priority_matching(
    agent_orders: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int, int]:
    """Select the GCOMG majority-game cohort by random internal crossing.

    The active side with M=max(n_buy,n_sell) is the majority side and the
    opposite side has m=min(n_buy,n_sell) agents.  Exactly m agents are drawn
    uniformly without replacement from the majority side.  These selected
    majority-side agents are the majority-side half of the internal-crossing
    cohort.  The opposite (minority) side is also internally crossed with them.
    The caller assigns the + payoff sign to both sides of those m pairs.

    Returns:
        priority_mask: bool array of length N
        M: majority-side count
        m: minority-side count
        majority_direction: +1 for buyers, -1 for sellers, 0 for a tie

    In a tie there is no strict majority, so no priority cohort is drawn.
    """
    orders = np.asarray(agent_orders, dtype=np.int8)
    if orders.ndim != 1 or not np.all(np.isin(orders, (-1, 0, 1))):
        raise ValueError("agent_orders must be a one-dimensional array of -1, 0, +1.")

    buy_idx = np.flatnonzero(orders == 1)
    sell_idx = np.flatnonzero(orders == -1)
    n_buy, n_sell = int(buy_idx.size), int(sell_idx.size)
    priority_mask = np.zeros(orders.size, dtype=bool)

    if n_buy > n_sell:
        majority_idx = buy_idx
        M, m = n_buy, n_sell
        direction = 1
    elif n_sell > n_buy:
        majority_idx = sell_idx
        M, m = n_sell, n_buy
        direction = -1
    else:
        M = m = n_buy
        direction = 0
        majority_idx = np.empty(0, dtype=np.int64)

    if m > 0 and direction != 0:
        selected = rng.choice(majority_idx, size=m, replace=False)
        priority_mask[selected] = True

    return priority_mask, M, m, direction


def randomly_redistribute_units(
    N: int,
    s: int,
    units: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign unit score points independently and with replacement to N*s slots.

    Every point has the same probability 1/(N*s) of reaching every held
    strategy of every agent.  Multiple points may therefore land on the same
    strategy.  The returned floating-point matrix is suitable for direct
    addition to scores and always sums to ``units``.
    """
    if N < 1 or s < 1 or units < 0:
        raise ValueError("N and s must be positive and units must be non-negative.")
    received = np.zeros((N, s), dtype=np.float64)
    if units == 0:
        return received
    targets = rng.integers(0, N * s, size=units)
    received = np.bincount(targets, minlength=N * s).reshape(N, s)
    return received.astype(np.float64, copy=False)


def counterfactual_action_score_changes(
    agent_orders: np.ndarray,
    priority_mask: np.ndarray,
    lam: float,
    L: float,
    tie_tolerance: float,
    zero_score_tie: str,
    actual_score_signal: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return score increments g_i(-1) and g_i(+1), one pair per agent.

    Only two candidate actions exist, so counterfactual matching and chi' are
    computed once for each action rather than once per held strategy.  The
    caller subsequently maps these two arrays onto all N*s strategy slots.

    For an active agent, each candidate replaces that agent's actual order;
    for an inactive agent, it is added to the actual order book, giving M+m+1
    hypothetical participants.  An unadopted candidate uses its own
    chi'=sign(D'/lambda-L).  A candidate on the minority side receives role
    +1.  A majority-side candidate receives role +1 with probability m'/M'
    (its exact marginal random-priority probability), otherwise role -1.
    A balanced counterfactual book is fully crossed and receives role +1.

    Candidates equal to the agent's actual submitted action instead reuse the
    actual score signal and realized matching result.  No virtual demand,
    price, or public information is fed back into the actual market.
    """
    orders = np.asarray(agent_orders, dtype=np.int8)
    realized_priority = np.asarray(priority_mask, dtype=bool)
    if orders.ndim != 1 or not np.all(np.isin(orders, (-1, 0, 1))):
        raise ValueError("agent_orders must be a one-dimensional array of -1, 0, +1.")
    if realized_priority.shape != orders.shape:
        raise ValueError("priority_mask must have the same shape as agent_orders.")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive.")
    if not np.isfinite(L) or tie_tolerance < 0:
        raise ValueError("L must be finite and tie_tolerance non-negative.")
    if zero_score_tie not in {"random", "zero"}:
        raise ValueError("zero_score_tie must be 'random' or 'zero'.")
    if actual_score_signal not in {-1.0, 0.0, 1.0}:
        raise ValueError("actual_score_signal must be -1, 0, or +1.")

    n_buy = int(np.count_nonzero(orders == 1))
    n_sell = int(np.count_nonzero(orders == -1))

    # Realized score change for an active agent's actual action.
    actual_direction = int(np.sign(n_buy - n_sell))
    actual_roles = -np.ones(orders.size, dtype=np.float64)
    if actual_direction == 0:
        actual_roles[orders != 0] = 1.0
    else:
        actual_roles[(orders == -actual_direction) | realized_priority] = 1.0
    actual_change = actual_roles * orders * actual_score_signal

    # The order book excluding each agent's own actual order.  For inactive
    # agents no order is removed, so each candidate is simply added to it.
    base_buy = n_buy - (orders == 1).astype(np.int64)
    base_sell = n_sell - (orders == -1).astype(np.int64)
    changes: list[np.ndarray] = []

    for action in (-1, 1):
        hypothetical_buy = base_buy + int(action == 1)
        hypothetical_sell = base_sell + int(action == -1)
        D_virtual = hypothetical_buy - hypothetical_sell
        M_virtual = np.maximum(hypothetical_buy, hypothetical_sell)
        m_virtual = np.minimum(hypothetical_buy, hypothetical_sell)
        direction_virtual = np.sign(D_virtual).astype(np.int8)

        # Each unadopted candidate has its own counterfactual price signal.
        excess_virtual = D_virtual.astype(np.float64) / lam - L
        chi_virtual = np.zeros(orders.size, dtype=np.float64)
        chi_virtual[excess_virtual > tie_tolerance] = 1.0
        chi_virtual[excess_virtual < -tie_tolerance] = -1.0
        tied = np.abs(excess_virtual) <= tie_tolerance
        if zero_score_tie == "random" and np.any(tied):
            chi_virtual[tied] = 2.0 * rng.integers(
                0, 2, size=int(tied.sum())
            ) - 1.0

        # Minority-side and balanced-book candidates are internally crossed.
        roles = np.ones(orders.size, dtype=np.float64)
        majority_candidate = (
            (direction_virtual != 0) & (direction_virtual == action)
        )
        roles[majority_candidate] = -1.0
        selection_probability = np.divide(
            m_virtual,
            M_virtual,
            out=np.zeros(orders.size, dtype=np.float64),
            where=M_virtual > 0,
        )
        virtual_priority = rng.random(orders.size) < selection_probability
        roles[majority_candidate & virtual_priority] = 1.0

        change = roles * float(action) * chi_virtual
        # The actual action is not a virtual candidate: preserve actual chi
        # and the realized priority result exactly.
        actual_action = orders == action
        change[actual_action] = actual_change[actual_action]
        changes.append(change)

    return changes[0], changes[1]


def simulate_gcomg(
    config: GCOMGConfig,
    strategy_space: Optional[FSSStrategySpace] = None,
) -> GCOMGResult:
    """Run GCOMG with strategy-specific counterfactual matching scores.

    Actual orders retain the original GCMG decision, participation, demand,
    price, and public-state rules.  Actual actions use actual chi and realized
    matching.  Every unadopted candidate instead uses chi' and a matching role
    from its own counterfactual order book.  Active agents replace their actual
    order; inactive agents hypothetically add one order.  Only the two possible
    actions are evaluated per agent before mapping score changes to strategies.
    Residual-majority score units are then redistributed with replacement;
    empty-market ordinary changes are waived before its larger redistribution.

    Scores are agent-specific (shape N x s), because identical strategy maps
    held by different agents can experience different matching outcomes.
    """
    if strategy_space is None:
        strategy_space = build_fss(config.m)
    if getattr(strategy_space, "m", None) != config.m:
        raise ValueError("strategy_space.m must equal config.m.")

    P = 1 << config.m
    n_total_strategies = 1 << P
    config.validate(n_strategies=n_total_strategies)
    if P > 1024:
        raise ValueError("This project supports at most m=10 (P=1024 public states).")

    # Separate random streams prevent counterfactual score draws from changing
    # actual strategy initialization, tie breaking, or realized order matching.
    seed_sequence = np.random.SeedSequence(config.seed)
    market_seed, virtual_score_seed, redistribution_seed = seed_sequence.spawn(3)
    rng = np.random.default_rng(market_seed)
    virtual_score_rng = np.random.default_rng(virtual_score_seed)
    redistribution_rng = np.random.default_rng(redistribution_seed)
    lam = config.resolved_lambda()
    alpha = 1.0 - 1.0 / config.T

    full_actions = strategy_space.actions
    if full_actions is not None:
        actions = np.asarray(full_actions, dtype=np.int8)
        if actions.shape != (n_total_strategies, P):
            raise ValueError(
                f"Materialized FSS actions must have shape ({n_total_strategies}, {P})."
            )
        if not np.all(np.isin(actions, (-1, 1))):
            raise ValueError("All FSS actions must be -1 or +1.")

        holdings = np.empty((config.N, config.s), dtype=np.int32)
        for i in range(config.N):
            holdings[i] = rng.choice(n_total_strategies, size=config.s, replace=False)
        used_strategy_ids = np.unique(holdings)
        candidate_actions = actions[used_strategy_ids]
        holdings_positions = np.searchsorted(used_strategy_ids, holdings)
    else:
        # Only strategies actually held are materialized: O(N*s*P) storage.
        candidate_actions, holdings_positions, _unique_ids, holdings = (
            _sample_fss_holdings(config.N, config.s, P, rng)
        )

    initial_score_value = (
        float(config.initial_score)
        if config.initial_score is not None
        else config.r + config.initial_score_offset
    )
    # Critical GCOMG change: personal score for each (agent, held strategy).
    scores = np.full((config.N, config.s), initial_score_value, dtype=np.float64)

    mu = int(rng.integers(P)) if config.mu0 is None else int(config.mu0)
    market_maker = MarketMaker(lam=lam, price=float(config.x0), mode=config.price_mode)

    steps = config.steps
    demand = np.zeros(steps, dtype=np.int32)
    buy_orders = np.zeros(steps, dtype=np.int32)
    sell_orders = np.zeros(steps, dtype=np.int32)
    active_agents = np.zeros(steps, dtype=np.int32)
    majority_count = np.zeros(steps, dtype=np.int32)
    minority_count = np.zeros(steps, dtype=np.int32)
    priority_count = np.zeros(steps, dtype=np.int32)
    majority_game_agents = np.zeros(steps, dtype=np.int32)
    minority_game_agents = np.zeros(steps, dtype=np.int32)
    majority_direction = np.zeros(steps, dtype=np.int8)
    price = np.zeros(steps + 1, dtype=np.float64)
    price_change = np.zeros(steps, dtype=np.float64)
    simple_returns = np.zeros(steps, dtype=np.float64)
    log_returns = np.zeros(steps, dtype=np.float64)
    global_info = np.zeros(steps, dtype=np.int32)
    appended_bit = np.zeros(steps, dtype=np.int8)
    score_signal_series = np.zeros(steps, dtype=np.float64)
    redistribution_units = np.zeros(steps, dtype=np.int32)
    max_redistribution_to_one_strategy = np.zeros(steps, dtype=np.int32)
    empty_market = np.zeros(steps, dtype=bool)
    total_redistribution_received = np.zeros((config.N, config.s), dtype=np.int64)
    price_fallback = np.zeros(steps, dtype=bool)
    price[0] = market_maker.price

    agent_rows = np.arange(config.N)

    for t in range(steps):
        mu_t = mu
        chosen_columns, best_score = _pick_best_strategies(scores, rng)

        # Grand-canonical entry rule is unchanged: best personal score > r.
        active = best_score > config.r
        held_actions_now = candidate_actions[holdings_positions, mu_t]  # (N, s)
        chosen_actions = held_actions_now[agent_rows, chosen_columns]
        agent_orders = chosen_actions * active.astype(np.int8)

        D = int(agent_orders.sum())
        n_buy = int(np.count_nonzero(agent_orders == 1))
        n_sell = int(np.count_nonzero(agent_orders == -1))
        q = D / lam
        excess_vs_reference = q - config.L

        if excess_vs_reference > config.tie_tolerance:
            score_signal = 1.0
            new_bit = 1
        elif excess_vs_reference < -config.tie_tolerance:
            score_signal = -1.0
            new_bit = 0
        else:
            new_bit = int(rng.integers(0, 2))
            if config.zero_score_tie == "random":
                score_signal = float(1 if rng.integers(0, 2) else -1)
            else:
                score_signal = 0.0

        # Random internal crossing: pair every minority-side agent with one
        # uniformly selected majority-side agent. Both sides of each internal
        # pair receive the Majority-Game score sign. Only residual majority
        # orders receive the Minority-Game score sign.
        priority_mask, M, m_small, direction = random_priority_matching(agent_orders, rng)
        majority_game_mask = priority_mask.copy()
        if direction == 0:
            # Balanced buy/sell flow: all active orders can be internally crossed.
            majority_game_mask |= active
        else:
            minority_side_mask = agent_orders == -direction
            majority_game_mask |= minority_side_mask
        # There are only two candidate actions.  Compute g_i(-1) and g_i(+1)
        # once per agent using the corresponding counterfactual D' and chi';
        # actual actions are overridden by actual chi and realized matching.
        # Then classify all held strategies by their current action.  This
        # removes O(N*s) counterfactual-book and random-priority calculations,
        # while the unavoidable score-matrix update remains O(N*s).
        change_if_sell, change_if_buy = counterfactual_action_score_changes(
            agent_orders=agent_orders,
            priority_mask=priority_mask,
            lam=lam,
            L=config.L,
            tie_tolerance=config.tie_tolerance,
            zero_score_tie=config.zero_score_tie,
            actual_score_signal=score_signal,
            rng=virtual_score_rng,
        )
        score_change = np.where(
            held_actions_now == 1,
            change_if_buy[:, None],
            change_if_sell[:, None],
        )

        is_empty = (n_buy + n_sell == 0)
        if is_empty:
            # First compute the ordinary empty-book changes above, then waive
            # all of them.  Their absolute values become the random bonus
            # budget.  With L=0 every g=-1, hence the budget is exactly N*s.
            budget = int(round(float(np.abs(score_change).sum())))
            score_change = np.zeros_like(score_change)
        else:
            # Reduced-form reward for historical limit-order providers:
            # (M-m)*s independent unit points when residual orders lose.
            budget = (M - m_small) * config.s if score_signal != 0.0 else 0

        scores = alpha * scores + score_change
        redistribution = randomly_redistribute_units(
            config.N, config.s, budget, redistribution_rng
        )
        scores += redistribution
        total_redistribution_received += redistribution.astype(np.int64)
        redistribution_units[t] = budget
        max_redistribution_to_one_strategy[t] = int(redistribution.max())
        empty_market[t] = is_empty

        # Price and information dynamics are intentionally unchanged.
        price_now, impact_or_price_change, simple_ret, log_ret = market_maker.clear(D)
        price_fallback[t] = market_maker.last_fallback
        mu = (2 * mu_t + new_bit) % P

        demand[t] = D
        buy_orders[t] = n_buy
        sell_orders[t] = n_sell
        active_agents[t] = int(active.sum())
        majority_count[t] = M
        minority_count[t] = m_small
        priority_count[t] = int(priority_mask.sum())
        majority_game_agents[t] = int(majority_game_mask.sum())
        minority_game_agents[t] = int(active.sum()) - int(majority_game_mask.sum())
        majority_direction[t] = direction
        price[t + 1] = price_now
        price_change[t] = impact_or_price_change
        simple_returns[t] = simple_ret
        log_returns[t] = log_ret
        global_info[t] = mu_t
        appended_bit[t] = new_bit
        score_signal_series[t] = score_signal

    return GCOMGResult(
        config=config,
        strategy_space=strategy_space,
        demand=demand,
        buy_orders=buy_orders,
        sell_orders=sell_orders,
        active_agents=active_agents,
        majority_count=majority_count,
        minority_count=minority_count,
        priority_count=priority_count,
        majority_game_agents=majority_game_agents,
        minority_game_agents=minority_game_agents,
        majority_direction=majority_direction,
        price=price,
        price_change=price_change,
        simple_returns=simple_returns,
        log_returns=log_returns,
        global_info=global_info,
        appended_bit=appended_bit,
        score_signal=score_signal_series,
        redistribution_units=redistribution_units,
        max_redistribution_to_one_strategy=max_redistribution_to_one_strategy,
        empty_market=empty_market,
        final_scores=scores.copy(),
        total_redistribution_received=total_redistribution_received.copy(),
        final_holdings=holdings.copy(),
        sampled_strategy_actions=candidate_actions.copy(),
        price_fallback=price_fallback,
    )


def autocorrelation_function(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Fast normalized ACF using an FFT (lags 0..max_lag)."""
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("series must be a one-dimensional array with at least 2 points")
    if max_lag < 0 or max_lag >= x.size:
        raise ValueError("max_lag must be in [0, len(series)-1]")
    x = x - x.mean()
    if np.allclose(x, 0.0):
        out = np.full(max_lag + 1, np.nan)
        out[0] = 1.0
        return out
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:max_lag + 1]
    acov /= n  # biased estimator; consistent across all requested lags
    if acov[0] <= 0:
        return np.full(max_lag + 1, np.nan)
    return acov / acov[0]


def lag_autocorrelation(series: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at a positive integer lag."""
    x = np.asarray(series, dtype=np.float64)
    if lag <= 0 or lag >= x.size:
        return float("nan")
    return float(autocorrelation_function(x, lag)[lag])


def summarize_result(result: GCOMGResult, burn_in: int = 10_000) -> dict[str, float]:
    """Compute basic stylized-fact diagnostics after discarding burn-in."""
    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must be >=0 and leave at least 4 observations.")
    r = np.asarray(result.log_returns[burn_in:], dtype=np.float64)
    d = np.asarray(result.demand[burn_in:], dtype=np.float64)
    v = np.asarray(result.active_agents[burn_in:], dtype=np.float64)
    maj = np.asarray(result.majority_count[burn_in:], dtype=np.float64)
    mino = np.asarray(result.minority_count[burn_in:], dtype=np.float64)
    priority = np.asarray(result.priority_count[burn_in:], dtype=np.float64)
    redistribution = np.asarray(result.redistribution_units[burn_in:], dtype=np.float64)
    empty = np.asarray(result.empty_market[burn_in:], dtype=bool)
    restart_count = int(np.count_nonzero(empty[:-1] & ~empty[1:]))
    mg_share = np.divide(
        result.majority_game_agents[burn_in:],
        v,
        out=np.zeros_like(v),
        where=v > 0,
    )

    r_centered = r - r.mean()
    r_std = r_centered.std()
    if r_std > 0:
        z = r_centered / r_std
        excess_kurtosis = float(np.mean(z**4) - 3.0)
    else:
        excess_kurtosis = float("nan")

    demand_std = float(d.std())
    demand_kurtosis = (
        float(np.mean(((d - d.mean()) / demand_std) ** 4) - 3.0)
        if demand_std > 0 else float("nan")
    )
    summary: dict[str, float] = {
        "observations_after_burn_in": float(r.size),
        "mean_demand": float(d.mean()),
        "std_demand": demand_std,
        "excess_kurtosis_demand": demand_kurtosis,
        "mean_active_agents": float(v.mean()),
        "std_active_agents": float(v.std()),
        "mean_majority_side_agents": float(maj.mean()),
        "mean_minority_side_agents": float(mino.mean()),
        "mean_priority_matched_agents": float(priority.mean()),
        "mean_majority_game_share_of_active": float(mg_share.mean()),
        "mean_redistribution_units": float(redistribution.mean()),
        "empty_market_share": float(empty.mean()),
        "market_restart_count": float(restart_count),
        "max_redistribution_to_one_strategy": float(
            result.max_redistribution_to_one_strategy[burn_in:].max()
        ),
        "mean_final_strategy_score": float(result.final_scores.mean()),
        "max_final_strategy_score": float(result.final_scores.max()),
        "mean_log_return": float(r.mean()),
        "std_log_return": float(r.std()),
        "excess_kurtosis_log_return": excess_kurtosis,
        "acf_return_lag1": lag_autocorrelation(r, 1),
        "acf_abs_return_lag1": lag_autocorrelation(np.abs(r_centered), 1),
        "acf_abs_return_lag2": lag_autocorrelation(np.abs(r_centered), 2),
        "acf_abs_return_lag5": lag_autocorrelation(np.abs(r_centered), 5),
        "acf_abs_return_lag10": lag_autocorrelation(np.abs(r_centered), 10),
        "acf_volume_lag1": lag_autocorrelation(v, 1),
        "acf_volume_lag2": lag_autocorrelation(v, 2),
        "acf_volume_lag5": lag_autocorrelation(v, 5),
        "acf_volume_lag10": lag_autocorrelation(v, 10),
    }
    return summary


def save_csv(result: GCOMGResult, path: Path) -> None:
    """Write one row per round, including the random matching composition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t", "mu_t", "new_bit", "demand", "buy_orders", "sell_orders",
        "active_agents", "majority_count", "minority_count", "priority_count",
        "majority_game_agents", "minority_game_agents", "majority_direction",
        "redistribution_units", "max_redistribution_to_one_strategy", "empty_market",
        "price", "price_change", "simple_return", "log_return", "score_signal",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(result.demand.size):
            writer.writerow([
                t,
                int(result.global_info[t]),
                int(result.appended_bit[t]),
                int(result.demand[t]),
                int(result.buy_orders[t]),
                int(result.sell_orders[t]),
                int(result.active_agents[t]),
                int(result.majority_count[t]),
                int(result.minority_count[t]),
                int(result.priority_count[t]),
                int(result.majority_game_agents[t]),
                int(result.minority_game_agents[t]),
                int(result.majority_direction[t]),
                int(result.redistribution_units[t]),
                int(result.max_redistribution_to_one_strategy[t]),
                int(result.empty_market[t]),
                f"{result.price[t + 1]:.12g}",
                f"{result.price_change[t]:.12g}",
                f"{result.simple_returns[t]:.12g}",
                f"{result.log_returns[t]:.12g}",
                f"{result.score_signal[t]:.1f}",
            ])

def save_diagnostics_plot(
    result: GCOMGResult,
    path: Path,
    burn_in: int = 10_000,
    show_steps: int = 3_000,
) -> None:
    """Save price/activity, return-tail and autocorrelation diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import sqrt, pi

    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must leave enough observations for diagnostics.")

    r = result.log_returns[burn_in:]
    rc = r - r.mean()
    rstd = rc.std()
    z = rc / rstd if rstd > 0 else rc
    lags = np.arange(1, 101)
    acf_r = autocorrelation_function(rc, 100)[1:]
    acf_abs = autocorrelation_function(np.abs(rc), 100)[1:]
    volume = result.active_agents[burn_in:].astype(float)
    acf_vol = autocorrelation_function(volume, 100)[1:]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    t_price = np.arange(min(show_steps, result.price.size))
    ax.plot(t_price, result.price[:t_price.size], linewidth=1.0)
    ax.set_title("Market price")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")

    ax = axes[0, 1]
    t_vol = np.arange(min(show_steps, result.active_agents.size))
    ax.plot(t_vol, result.active_agents[:t_vol.size], linewidth=0.8, label="active agents")
    ax.plot(t_vol, result.priority_count[:t_vol.size], linewidth=0.8,
            label="priority-matched majority agents")
    ax.plot(t_vol, result.redistribution_units[:t_vol.size], linewidth=0.8,
            label="redistributed score units")
    ax.set_title("Activity, priority matching, and score redistribution")
    ax.set_xlabel("t")
    ax.set_ylabel("count / score units")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    bins = np.linspace(-6, 6, 81)
    ax.hist(z, bins=bins, density=True, alpha=0.65, label="GCOMG log returns")
    grid = np.linspace(-6, 6, 500)
    normal_pdf = np.exp(-0.5 * grid**2) / sqrt(2 * pi)
    ax.plot(grid, normal_pdf, linewidth=1.6, label="Standard normal")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)
    ax.set_title("Standardized return distribution (log scale)")
    ax.set_xlabel("z = (r - mean) / std")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(lags, acf_r, label="return")
    ax.plot(lags, acf_abs, label="|return|")
    ax.plot(lags, acf_vol, label="activity V(t)")
    ax.axhline(0.0, linewidth=0.7)
    ax.set_title("Autocorrelation functions")
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.legend(frameon=False)

    fig.suptitle(
        f"GCOMG-FSS: N={result.config.N}, m={result.config.m}, s={result.config.s}, "
        f"r={result.config.r:g}, T={result.config.T:g}, L={result.config.L:g}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_price_change_distribution_plot(
    result: GCOMGResult,
    path: Path,
    burn_in: int = 10_000,
    bins: int = 80,
    max_qq_points: int = 5_000,
) -> None:
    """Save raw price-increment PDF and normal Q-Q plot (lecture p.27 style).

    The horizontal variable is the actual x[t+1]-x[t], not a standardized
    return.  The Gaussian reference uses the sample mean and standard deviation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import sqrt, pi
    from statistics import NormalDist

    if not (0 <= burn_in < result.demand.size - 3):
        raise ValueError("burn_in must be >=0 and leave at least 4 observations.")
    if bins < 5:
        raise ValueError("bins must be >= 5.")
    if max_qq_points < 100:
        raise ValueError("max_qq_points must be >= 100.")

    # In linear mode this equals D/lambda. Computing it from price also makes
    # the label correct when price_mode='log'.
    delta_x = np.diff(result.price)[burn_in:]
    delta_x = delta_x[np.isfinite(delta_x)]
    if delta_x.size < 4:
        raise ValueError("Not enough finite price increments to plot.")

    mean = float(delta_x.mean())
    std = float(delta_x.std())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) PDF of raw price increments.
    ax = axes[0]
    if std > 0:
        _, edges, _ = ax.hist(
            delta_x, bins=bins, density=True, alpha=0.7,
            label="GCOMG price increments",
        )
        grid = np.linspace(float(edges[0]), float(edges[-1]), 600)
        gaussian_pdf = np.exp(-0.5 * ((grid - mean) / std) ** 2) / (std * sqrt(2 * pi))
        ax.plot(grid, gaussian_pdf, linewidth=1.7, label="Gaussian with same mean/std")
    else:
        ax.axvline(mean, linewidth=2, label="GCOMG price increments")
    ax.set_title("PDF of price increments")
    ax.set_xlabel(r"Price increment $\Delta x=x_{t+1}-x_t$")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)

    # (b) Normal Q-Q plot, using standard-library inverse normal CDF.
    ax = axes[1]
    ordered = np.sort(delta_x)
    n = ordered.size
    n_plot = min(n, max_qq_points)
    indices = np.unique(np.linspace(0, n - 1, n_plot).round().astype(int))
    sample_quantiles = ordered[indices]
    probabilities = (indices + 0.5) / n
    normal = NormalDist()
    theoretical_quantiles = np.fromiter(
        (normal.inv_cdf(float(prob)) for prob in probabilities),
        dtype=np.float64, count=probabilities.size,
    )
    ax.scatter(theoretical_quantiles, sample_quantiles, s=7, alpha=0.55,
               label="Empirical quantiles")
    if std > 0:
        qline = np.linspace(float(theoretical_quantiles.min()),
                            float(theoretical_quantiles.max()), 200)
        ax.plot(qline, mean + std * qline, "--", linewidth=1.7,
                label="Gaussian reference")
    ax.set_title("Normal Q-Q plot")
    ax.set_xlabel("Quantiles of standard normal distribution")
    ax.set_ylabel(r"Quantiles of $\Delta x$")
    ax.legend(frameon=False)

    fig.suptitle(
        f"GCOMG-FSS price increments: N={result.config.N}, m={result.config.m}, "
        f"s={result.config.s}, lambda={result.config.resolved_lambda():g}, "
        f"burn-in={burn_in}", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the GCOMG with the Full Strategy Space (FSS).")
    parser.add_argument("--N", type=int, default=101, help="number of agents")
    parser.add_argument("--m", type=int, default=5, help="global-information memory length")
    parser.add_argument("--s", type=int, default=2, help="strategies held by each agent")
    parser.add_argument("--r", type=float, default=-100, help="strategy-score participation threshold")
    parser.add_argument("--T", type=float, default=10000, help="score memory/effective period")
    parser.add_argument("--L", type=float, default=0.0, help="market reference threshold")
    parser.add_argument("--lambda", dest="lam", type=float, default=None, help="market depth; default 10*N")
    parser.add_argument("--x0", type=float, default=100.0, help="initial price")
    parser.add_argument("--price-mode", choices=("linear", "log"), default="linear")
    parser.add_argument("--steps", type=int, default=10000, help="number of market rounds")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--mu0", type=int, default=None, help="initial m-bit public-information state")
    parser.add_argument(
        "--initial-score-offset", type=float, default=10.0,
        help="initialize every strategy score at r + offset (default: 10)",
    )
    parser.add_argument(
        "--initial-score", type=float, default=10,
        help="absolute initial score for all strategies; overrides --initial-score-offset",
    )
    parser.add_argument(
        "--zero-score-tie", choices=("random", "zero"), default="random",
        help="score signal at D/lambda=L: literal sign(0)=0 (default) or random regularization",
    )
    parser.add_argument("--burn-in", type=int, default=10_000, help="discarded transient for diagnostics")
    parser.add_argument("--outdir", type=Path, default=Path("gcomg_output"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = GCOMGConfig(
        N=args.N,
        m=args.m,
        s=args.s,
        r=args.r,
        T=args.T,
        L=args.L,
        lam=args.lam,
        x0=args.x0,
        price_mode=args.price_mode,
        steps=args.steps,
        seed=args.seed,
        mu0=args.mu0,
        initial_score_offset=args.initial_score_offset,
        initial_score=args.initial_score,
        zero_score_tie=args.zero_score_tie,
    )
    strategy_space = build_fss(config.m)
    result = simulate_gcomg(config, strategy_space)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "gcomg_timeseries.csv"
    plot_path = args.outdir / "gcomg_diagnostics.png"
    distribution_plot_path = args.outdir / "gcomg_price_change_distribution.png"
    save_csv(result, csv_path)
    save_diagnostics_plot(result, plot_path, burn_in=args.burn_in)
    save_price_change_distribution_plot(
        result, distribution_plot_path, burn_in=args.burn_in
    )
    summary = summarize_result(result, burn_in=args.burn_in)
    summary_path = args.outdir / "gcomg_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"parameters": vars(config), "lambda_used": config.resolved_lambda(), "diagnostics": summary},
            f, ensure_ascii=False, indent=2,
        )

    if strategy_space.actions is not None:
        print(f"FSS: P={strategy_space.P}, number of strategies={strategy_space.n_strategies}")
    else:
        print(f"FSS: P={strategy_space.P}, total strategies=2^{strategy_space.P}; "
              f"sampled unique strategies={result.sampled_strategy_actions.shape[0]}")
    initial_score_value = (
        config.initial_score if config.initial_score is not None
        else config.r + config.initial_score_offset
    )
    print(
        f"lambda={config.resolved_lambda():g}; price_mode={config.price_mode}; "
        f"initial_score={initial_score_value:g}; zero_score_tie={config.zero_score_tie}; "
        f"score_storage=agent_strategy; redistribution=enabled"
    )
    tail_window = max(1, result.active_agents.size // 10)
    if np.all(result.active_agents[-tail_window:] == 0):
        print(
            "\nWARNING: no agents were active in the final 10% of the run, "
            "despite empty-market score waivers and redistribution. Inspect "
            "redistribution_units, final scores, and the participation threshold."
        )
    fallback_count = int(np.count_nonzero(result.price_fallback))
    if fallback_count:
        print(f"\nPrice-boundary log fallback used in {fallback_count} step(s).")
    print("\nDiagnostics (after burn-in):")
    for key, value in summary.items():
        print(f"  {key:32s}: {value:.6g}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {distribution_plot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

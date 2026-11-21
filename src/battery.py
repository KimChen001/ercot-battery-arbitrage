"""Battery scheduling: the MILP from the proposal, a simple heuristic, and profit.

MILP (one operating day, T hours, prices p_t in $/MWh):
    max  sum_t p_t (d_t - c_t) - kappa * sum_t (c_t + d_t)
    s.t. s_t = s_{t-1} + eta * c_t - d_t / eta        (state of charge, MWh)
         0 <= s_t <= E_max
         0 <= c_t <= P_max * u_t,  0 <= d_t <= P_max * (1 - u_t),  u_t in {0, 1}
         s_T = s_0
c_t / d_t are grid-side charge / discharge power (MW, = MWh per hour).
Solved with scipy's HiGHS MILP solver, so no commercial license is needed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Battery:
    p_max: float = 100.0      # MW
    e_max: float = 200.0      # MWh (two-hour battery)
    rte: float = 0.90         # round-trip efficiency
    kappa: float = 5.0        # wear cost, $ per MWh charged or discharged
    s0: float = 0.0           # start (and end) state of charge, MWh

    @property
    def eta(self):
        return self.rte ** 0.5  # one-way efficiency


def solve_milp(prices, bat: Battery):
    """Optimal schedule for one day given (forecast or actual) prices."""
    p = np.asarray(prices, dtype=float)
    T = len(p)
    # variable order: c[0:T], d[T:2T], s[2T:3T], u[3T:4T]
    ic, id_, is_, iu = 0, T, 2 * T, 3 * T
    cost = np.zeros(4 * T)
    cost[ic:ic + T] = p + bat.kappa          # minimize the negative profit
    cost[id_:id_ + T] = bat.kappa - p

    A = lil_matrix((3 * T + 1, 4 * T))
    lo, hi = [], []
    row = 0
    for t in range(T):  # SOC balance
        A[row, is_ + t] = 1.0
        if t > 0:
            A[row, is_ + t - 1] = -1.0
        A[row, ic + t] = -bat.eta
        A[row, id_ + t] = 1.0 / bat.eta
        rhs = bat.s0 if t == 0 else 0.0
        lo.append(rhs); hi.append(rhs); row += 1
    for t in range(T):  # c_t <= P u_t
        A[row, ic + t] = 1.0
        A[row, iu + t] = -bat.p_max
        lo.append(-np.inf); hi.append(0.0); row += 1
    for t in range(T):  # d_t <= P (1 - u_t)
        A[row, id_ + t] = 1.0
        A[row, iu + t] = bat.p_max
        lo.append(-np.inf); hi.append(bat.p_max); row += 1
    A[row, is_ + T - 1] = 1.0  # end where we started
    lo.append(bat.s0); hi.append(bat.s0)

    ub = np.concatenate([np.full(T, bat.p_max), np.full(T, bat.p_max),
                         np.full(T, bat.e_max), np.ones(T)])
    integrality = np.concatenate([np.zeros(3 * T), np.ones(T)])
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi),
               bounds=Bounds(np.zeros(4 * T), ub), integrality=integrality)
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    return x[ic:ic + T], x[id_:id_ + T], x[is_:is_ + T]


def heuristic(prices, bat: Battery, k=2):
    """Charge at full power in the k cheapest hours, discharge in the k dearest.

    Simulated hour by hour within the battery limits; a discharge hour that comes
    before any charging does nothing, and energy still stored at the end of the day
    is sold in the last hour so every strategy ends at s_0.
    """
    p = np.asarray(prices, dtype=float)
    T = len(p)
    order = np.argsort(p)
    charge_hours, discharge_hours = set(order[:k]), set(order[-k:])
    c, d, s = np.zeros(T), np.zeros(T), np.zeros(T)
    soc = bat.s0
    for t in range(T):
        if t in charge_hours:
            c[t] = min(bat.p_max, (bat.e_max - soc) / bat.eta)
        elif t in discharge_hours:
            d[t] = min(bat.p_max, (soc - bat.s0) * bat.eta)
        if t == T - 1 and soc > bat.s0:  # settle back to s_0
            d[t] = min(bat.p_max, (soc - bat.s0) * bat.eta)
        soc += bat.eta * c[t] - d[t] / bat.eta
        s[t] = soc
    return c, d, s


def profit(actual_prices, c, d, bat: Battery):
    """Realized profit of a schedule settled at actual prices ($)."""
    p = np.asarray(actual_prices, dtype=float)
    return float(np.sum(p * (d - c)) - bat.kappa * np.sum(c + d))

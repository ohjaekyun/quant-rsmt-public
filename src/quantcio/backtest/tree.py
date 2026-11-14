"""RSMT-style model-based tree on monthly factor-IC vectors (docs/BACKTEST_MACRO.md section 3.1, ``rsmt_tree``).

One row = one month s: the vector of the factors' monthly Spearman ICs and date-level regime variables known at s
(market 60-session return / realised volatility, macro regime score). The node model is the IC mean vector; its
scores are the deviations ``S_i = IC_i - mean``, standardised by ``B^-1/2`` with ``B = S'S / n``. A node splits when
the parameter-instability (sup-LM) statistic of the scores ordered by a split variable is significant:

- statistic: ``max_k n / (k (n - k)) * || sum_{i <= k} U_i ||^2`` over tie boundaries with both children holding at
  least ``min_leaf`` months (the ``quant.rsmt.split_scan`` statistic, implemented locally so that this module does
  not depend on the RSMT-Q code);
- null: block multiplier bootstrap - every block of ``block`` consecutive calendar months gets one N(0, 1)
  multiplier, the multiplied scores are re-centred and the statistic recomputed with the same ``B``. This keeps the
  scores' serial correlation (regime variables are persistent, so months adjacent in a split variable are often
  adjacent in time) and destroys any relation to the split variable;
- multiple split variables: Simes at level ``alpha``; on rejection the variable with the smallest p is split at its
  sup-LM boundary. Depth <= ``max_depth``; a month whose split variable is missing stays in the parent only, and a
  date whose split variable is missing is routed to the parent.

Leaf means are shrunk towards the (shrunk) parent mean with ``shrink_months`` pseudo-months.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_LEAF = 24  # months per child
MAX_DEPTH = 2
ALPHA = 0.05  # Simes level across split variables
N_BOOT = 499
BLOCK = 6  # months per bootstrap block
SHRINK_MONTHS = 12.0  # leaf mean = (n * m + k * parent) / (n + k)
SEED = 20260929


@dataclass
class ICNode:
    n: int  # months in the node
    depth: int
    raw_mean: np.ndarray  # plain IC mean vector of the node
    mean: np.ndarray  # shrunk towards the parent (root: the plain mean)
    var: str | None = None  # split variable (None = leaf)
    threshold: float | None = None  # left: x <= threshold
    left: ICNode | None = None
    right: ICNode | None = None
    tests: dict[str, dict[str, float]] = field(default_factory=dict)  # variable -> {stat, p, threshold}
    p_simes: float | None = None

    def route(self, x: dict[str, float]) -> ICNode:
        """Leaf (or the deepest node whose split variable is known) of a date with split variables ``x``."""
        node = self
        while node.var is not None and node.left is not None and node.right is not None:
            v = x.get(node.var)
            if v is None or not np.isfinite(v):
                break
            node = node.left if v <= float(node.threshold) else node.right
        return node

    @property
    def n_leaves(self) -> int:
        if self.left is None or self.right is None:
            return 1
        return self.left.n_leaves + self.right.n_leaves

    @property
    def n_splits(self) -> int:
        return self.n_leaves - 1

    def split_vars(self) -> list[str]:
        if self.var is None or self.left is None or self.right is None:
            return []
        return [self.var, *self.left.split_vars(), *self.right.split_vars()]

    def describe(self, factors: list[str] | None = None) -> dict[str, Any]:
        """JSON-safe description (split variable, threshold, months, children, leaf IC means, test p-values)."""
        out: dict[str, Any] = {"n": int(self.n), "depth": int(self.depth)}
        if factors is not None:
            out["ic_mean"] = {f: round(float(v), 4) for f, v in zip(factors, self.mean, strict=True)}
        if self.tests:
            out["tests"] = {v: {k: round(float(x), 4) for k, x in t.items()} for v, t in self.tests.items()}
            out["p_simes"] = None if self.p_simes is None else round(float(self.p_simes), 4)
        if self.var is not None and self.left is not None and self.right is not None:
            out["split"] = {"var": self.var, "threshold": float(self.threshold)}
            out["left"] = self.left.describe(factors)
            out["right"] = self.right.describe(factors)
        return out


def standardiser(s: np.ndarray, rel_floor: float = 1e-8) -> np.ndarray | None:
    """``B^-1/2`` of ``B = S'S / n`` (eigenvalues floored at ``rel_floor`` x the largest; None when B = 0)."""
    n = s.shape[0]
    b = s.T @ s / max(n, 1)
    vals, vecs = np.linalg.eigh(b)
    if not np.isfinite(vals).all() or vals.max() <= 0:
        return None
    vals = np.maximum(vals, rel_floor * vals.max())
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def boundaries(x_sorted: np.ndarray, min_leaf: int) -> np.ndarray:
    """Admissible left sizes k (tie-group ends) with ``min_leaf <= k <= n - min_leaf``."""
    n = x_sorted.size
    k = np.nonzero(x_sorted[1:] > x_sorted[:-1])[0] + 1
    return k[(k >= min_leaf) & (k <= n - min_leaf)]


def sup_lm(x: np.ndarray, u: np.ndarray, min_leaf: int) -> tuple[float, float | None]:
    """(sup-LM statistic, split threshold) of standardised scores ``u`` (n x k) ordered by ``x``.

    No admissible boundary -> (0.0, None). The threshold is the midpoint between the two values at the boundary.
    """
    n = x.size
    order = np.argsort(x, kind="stable")
    xs = x[order]
    k = boundaries(xs, min_leaf)
    if k.size == 0:
        return 0.0, None
    cs = np.cumsum(u[order], axis=0)
    q = n / (k * (n - k)) * (cs[k - 1] ** 2).sum(axis=1)
    j = int(np.argmax(q))
    lo, hi = float(xs[k[j] - 1]), float(xs[k[j]])
    thr = (lo + hi) / 2.0
    if not thr < hi:  # float midpoint collapse
        thr = lo
    return float(q[j]), thr


def boot_sup_lm(x: np.ndarray, s: np.ndarray, w: np.ndarray, mult: np.ndarray, min_leaf: int) -> np.ndarray:
    """Bootstrap statistics: rows of ``mult`` (draws x n) multiply the scores ``s``; re-centred, standardised by
    the fixed ``w`` and scanned over the same boundaries as the observed statistic."""
    n = x.size
    order = np.argsort(x, kind="stable")
    k = boundaries(x[order], min_leaf)
    if k.size == 0:
        return np.zeros(mult.shape[0])
    star = mult[:, :, None] * s[None, :, :]
    star -= star.mean(axis=1, keepdims=True)
    u = star[:, order, :] @ w
    cs = np.cumsum(u, axis=1)[:, k - 1, :]
    q = n / (k * (n - k))[None, :] * (cs**2).sum(axis=2)
    return q.max(axis=1)


def simes(p: list[float]) -> float:
    """Simes combined p-value min_j m p_(j) / j (1.0 when no test)."""
    if not p:
        return 1.0
    ps = np.sort(np.asarray(p, dtype=float))
    m = ps.size
    return float(min(1.0, np.min(m * ps / np.arange(1, m + 1))))


def fit_ic_tree(
    ic: np.ndarray,
    split: dict[str, np.ndarray],
    *,
    month_pos: np.ndarray | None = None,
    min_leaf: int = MIN_LEAF,
    max_depth: int = MAX_DEPTH,
    alpha: float = ALPHA,
    n_boot: int = N_BOOT,
    block: int = BLOCK,
    shrink_months: float = SHRINK_MONTHS,
    seed: int = SEED,
) -> ICNode:
    """Fit the tree on ``ic`` (n months x k factors, no NaN) with split variables ``split`` (name -> n values,
    NaN = unknown). ``month_pos`` (n,) is each row's calendar month index (default 0..n-1), used for the
    bootstrap blocks. Deterministic for given inputs (seeded by ``seed``, n and the node path)."""
    ic = np.asarray(ic, dtype=float)
    n_all = ic.shape[0]
    pos = np.arange(n_all) if month_pos is None else np.asarray(month_pos, dtype=int)
    blocks = (pos - pos.min()) // max(int(block), 1) if n_all else pos
    n_blocks = int(blocks.max()) + 1 if n_all else 0
    xs = {v: np.asarray(x, dtype=float) for v, x in split.items()}

    def grow(idx: np.ndarray, parent: np.ndarray | None, depth: int, code: int) -> ICNode:
        raw = ic[idx].mean(axis=0)
        mean = raw if parent is None else (idx.size * raw + shrink_months * parent) / (idx.size + shrink_months)
        node = ICNode(n=int(idx.size), depth=depth, raw_mean=raw, mean=mean)
        if depth >= max_depth or idx.size < 2 * min_leaf:
            return node
        rng = np.random.default_rng([int(seed), int(n_all), int(code)])
        xi_blocks = rng.standard_normal((int(n_boot), n_blocks))  # common draws for every variable of the node
        found: dict[str, tuple[float, float, float, np.ndarray]] = {}
        for v, x_all in xs.items():
            x = x_all[idx]
            ok = np.isfinite(x)
            if ok.sum() < 2 * min_leaf:
                continue
            rows = idx[ok]
            s = ic[rows] - ic[rows].mean(axis=0)
            w = standardiser(s)
            if w is None:
                continue
            stat, thr = sup_lm(x[ok], s @ w, min_leaf)
            if thr is None:
                continue
            null = boot_sup_lm(x[ok], s, w, xi_blocks[:, blocks[rows]], min_leaf)
            p = (1.0 + float((null >= stat - 1e-12).sum())) / (null.size + 1.0)
            found[v] = (stat, p, thr, rows)
            node.tests[v] = {"stat": stat, "p": p, "threshold": thr}
        if not found:
            return node
        node.p_simes = simes([f[1] for f in found.values()])
        if node.p_simes > alpha:
            return node
        best = min(found, key=lambda v: (found[v][1], -found[v][0]))
        _, _, thr, rows = found[best]
        x = xs[best][rows]
        node.var, node.threshold = best, float(thr)
        node.left = grow(rows[x <= thr], mean, depth + 1, 2 * code)
        node.right = grow(rows[x > thr], mean, depth + 1, 2 * code + 1)
        return node

    return grow(np.arange(n_all), None, 0, 1)


__all__ = ["ICNode", "boot_sup_lm", "fit_ic_tree", "simes", "standardiser", "sup_lm"]

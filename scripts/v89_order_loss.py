"""Ordering-loss indicator (docs/MODELING.md 89, pre-declared 2026-11-05): a linear index s = theta . x whose 20
within-date quantiles rise in order, learnt by minimising an ordering loss of the 20-quantile curve itself. Research
only: src/quantcio and the served models are not changed.

    .venv/bin/python scripts/v89_order_loss.py smoke --market KR   # every new fitter once on the last outer window
    .venv/bin/python scripts/v89_order_loss.py study --market KR

Per date the index ranks the names into 20 quantiles; m_tk = mean per-date standardised y_e (not winsorised) of
quantile k; the 19 steps D_tk = m_t,k+1 - m_tk; t_k = mean_t D_tk / (sd_t D_tk / sqrt(T)). Losses (89.1):
``up`` = -mean_k tanh(t_k / 2) (a smooth count of rising steps), ``min`` = logsumexp(-t_k) (the weakest step),
``slope`` = -t of the per-date curve slope (the reference). The forward pass uses the exact quantiles; the gradient
flows through soft quantile memberships of Phi(z) of the within-date standardised index (straight-through).
Start = ridge on the standardised return; the number of Adam steps is chosen on the annual blocks of 87.3 by the
same loss on the pooled out-of-fold curve (1-SE toward fewer steps, block-bootstrap SE).
Writes reports/modeling/<M>_v89.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v86_full_es as v86  # noqa: E402
import v87_validation as v87  # noqa: E402
import v88_ventile as v88  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402

OUT = v86.OUT
V89_VERSION = 1
K = v88.K
CHECKS = (0, 10, 25, 50, 100, 200, 400)
LR = 0.01
TAU = 1.0 / (4 * K)  # soft quantile edge width (rank units) of the gradient path
RIDGE0 = 1.0  # the start: ridge on the standardised return (inputs ~ unit variance, date weights sum to 1)
VAL_BOOT = 200
LOSSES = {"vo_up": "up", "vo_min": "min", "vo_slope": "slope"}
NEW = (*LOSSES, "vo_start")
NEW_KO = {
    "vo_up": "순서 손실(매끈한 오른 칸 수)",
    "vo_min": "순서 손실(가장 약한 칸)",
    "vo_slope": "기울기 손실",
    "vo_start": "시작점(표준화 수익 릿지)",
}
CONTRASTS = {  # 89.4
    "D1 순서 학습 − 시작점": ("vo_up", "vo_start"),
    "D2 순서 손실 − 기울기 손실": ("vo_up", "vo_slope"),
    "D3 가장 약한 칸 − 오른 칸 수": ("vo_min", "vo_up"),
    "D4 `vo_up` − `lgbm_reg_w_cv`": ("vo_up", "lgbm_reg_w_cv"),
    "D5 `vo_up` − `mom_12_1`": ("vo_up", "mom_12_1"),
}
LOG: list[Path] = []
_WIN: dict = {}
_CHOICE: dict = {}

say = v86.say


def _log(rec: dict) -> None:
    if LOG:
        with LOG[0].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


def _torch():
    import torch

    torch.set_num_threads(1)  # >1 hangs after LightGBM ran in the same process (two OpenMP runtimes, macOS, 89.5)
    return torch


# ------------------------------------------------------------------------------------------------ losses


def step_t(m, xp):
    """Per step k: t of the per-date difference m_t,k+1 - m_tk (``xp`` = numpy or torch)."""
    d = m[:, 1:] - m[:, :-1]
    n = d.shape[0]
    sd = d.std(0, ddof=1) if xp is np else d.std(0, correction=1)
    return d.mean(0) / (sd / np.sqrt(n) + 1e-12)


def loss_value(m, kind: str, xp):
    """89.1: the ordering losses of a (dates x 20) quantile-mean matrix (smaller = more orderly)."""
    if kind == "slope":
        c = np.arange(K) - (K - 1) / 2.0
        c = xp.asarray(c) if xp is np else xp.as_tensor(c, dtype=m.dtype)
        sl = (m * c).sum(1) / (c * c).sum()
        n = sl.shape[0]
        sd = sl.std(ddof=1) if xp is np else sl.std(correction=1)
        return -(sl.mean() / (sd / np.sqrt(n) + 1e-12))
    t = step_t(m, xp)
    if kind == "up":
        return -xp.tanh(t / 2.0).mean()
    if xp is np:
        return float(np.log(np.exp(-t).sum()))
    return xp.logsumexp(-t, dim=0)


def all_losses(m: np.ndarray) -> dict:
    return {k: float(loss_value(m, k, np)) for k in ("up", "min", "slope")}


# ------------------------------------------------------------------------------------------------ data


def zret(y: np.ndarray, codes: np.ndarray, n_dates: int) -> np.ndarray:
    """Per-date standardised return (date mean 0, SD 1; not winsorised)."""
    cnt = np.bincount(codes, minlength=n_dates)
    mu = np.bincount(codes, y, minlength=n_dates) / np.maximum(cnt, 1)
    dev = y - mu[codes]
    sd = np.sqrt(np.bincount(codes, dev * dev, minlength=n_dates) / np.maximum(cnt, 1))
    return np.where(sd[codes] > 0, dev / np.where(sd[codes] > 0, sd[codes], 1.0), 0.0)


@dataclass
class MWin:
    """One training window of month-end rows: inputs (NaN = 0), standardised return, date codes, annual folds."""

    key: tuple
    xs: list[str]
    X: np.ndarray
    r: np.ndarray
    codes: np.ndarray
    folds: list
    spans: list


def mwindow(train: pd.DataFrame, ctx) -> MWin:
    key = v88._wkey(train, ctx)
    if _WIN.get("key") == key:
        return _WIN["win"]
    _WIN.clear()
    xs = [c for c in ctx.xs_cols if c in train and train[c].notna().any()]
    d = train[train["is_me"].to_numpy(bool) & np.isfinite(train["y_e"].to_numpy(dtype=float))]
    d = d[d.groupby("date")["y_e"].transform("size").to_numpy() >= v88.mt.MIN_NAMES].reset_index(drop=True)
    codes, uniq = pd.factorize(d["date"], sort=True)
    dates = d["date"].to_numpy(dtype="datetime64[ns]")
    folds, spans = v88.year_masks(dates, np.ones(len(d), bool), ctx)
    win = MWin(
        key,
        xs,
        np.nan_to_num(d[xs].to_numpy(dtype=np.float64), nan=0.0),
        zret(d["y_e"].to_numpy(dtype=float), codes, len(uniq)),
        codes,
        folds,
        spans,
    )
    _WIN.update(key=key, win=win)
    return win


class Batch:
    """Rows of some dates as padded (dates x names) tensors."""

    def __init__(self, X: np.ndarray, r: np.ndarray, codes: np.ndarray):
        torch = _torch()
        codes, uniq = pd.factorize(codes, sort=True)
        d = len(uniq)
        order = np.argsort(codes, kind="stable")
        cs = codes[order]
        cnt = np.bincount(cs, minlength=d)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        pos = np.arange(len(cs)) - start[cs]
        n = int(cnt.max())
        xp = np.zeros((d, n, X.shape[1]))
        rp = np.zeros((d, n))
        mk = np.zeros((d, n), bool)
        xp[cs, pos] = X[order]
        rp[cs, pos] = r[order]
        mk[cs, pos] = True
        self.X, self.r = torch.from_numpy(xp), torch.from_numpy(rp)
        self.mask = torch.from_numpy(mk)
        self.nt = self.mask.sum(1, keepdim=True).to(self.X.dtype)
        self.order, self.cs, self.pos = order, cs, pos  # row i of the input = padded cell (cs, pos)[argsort(order)]


def curve(theta, b: Batch, tau: float = TAU):
    """(dates x K) quantile means: exact quantiles forward, soft Phi(z) memberships for the gradient (89.1)."""
    return curve_scores(b.X @ theta, b, tau)


def curve_scores(s, b: Batch, tau: float = TAU):
    """``curve`` of given padded (dates x names) scores (docs/MODELING.md 90: the per-row gradient of a booster)."""
    torch = _torch()
    mk = b.mask.to(s.dtype)
    mu = (s * mk).sum(1, keepdim=True) / b.nt
    sd = torch.sqrt((((s - mu) ** 2) * mk).sum(1, keepdim=True) / b.nt + 1e-18)
    z = (s - mu) / sd
    with torch.no_grad():
        filled = s.masked_fill(~b.mask, float("inf"))
        rank = torch.argsort(torch.argsort(filled, dim=1, stable=True), dim=1, stable=True).to(s.dtype)
        bins = torch.clamp(torch.floor((rank + 0.5) / b.nt * K), 0, K - 1).long()
        hard = torch.nn.functional.one_hot(bins, K).to(s.dtype) * mk[..., None]
        m_hard = (hard * b.r[..., None]).sum(1) / hard.sum(1).clamp_min(1e-9)
    u = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    edges = torch.arange(1, K, dtype=s.dtype) / K
    c = torch.sigmoid((u[..., None] - edges) / tau)
    w = torch.cat([1.0 - c[..., :1], c[..., :-1] - c[..., 1:], c[..., -1:]], dim=-1) * mk[..., None]
    m_soft = (w * b.r[..., None]).sum(1) / w.sum(1).clamp_min(1e-9)
    return m_hard + m_soft - m_soft.detach()


def ridge_start(X: np.ndarray, r: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """89.1: theta_0 = ridge of the standardised return (dates weighted equally), unit length."""
    cnt = np.bincount(codes)
    w = 1.0 / cnt[codes]
    w = w / w.sum()
    xw = X * w[:, None]
    th = np.linalg.solve(xw.T @ X + RIDGE0 * np.eye(X.shape[1]), xw.T @ r)
    nrm = np.linalg.norm(th)
    return th / nrm if nrm > 0 else th


def train_path(
    X: np.ndarray, r: np.ndarray, codes: np.ndarray, kind: str, steps: tuple, tau: float = TAU
) -> tuple[dict, dict]:
    """Adam from the ridge start on one loss; theta and the (exact) training loss at every requested step."""
    torch = _torch()
    th0 = ridge_start(X, r, codes)
    b = Batch(X, r, codes)
    theta = torch.tensor(th0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    snaps, train = {0: th0.copy()}, {}
    with torch.no_grad():
        train[0] = float(loss_value(curve(theta, b, tau), kind, torch))
    for it in range(1, max(steps) + 1):
        opt.zero_grad()
        loss = loss_value(curve(theta, b, tau), kind, torch)
        loss.backward()
        opt.step()
        if it in steps:
            snaps[it] = theta.detach().numpy().copy()
            with torch.no_grad():
                train[it] = float(loss_value(curve(theta, b, tau), kind, torch))
    return snaps, train


def _vm(X: np.ndarray, r: np.ndarray, codes: np.ndarray, theta: np.ndarray) -> np.ndarray:
    c, u = pd.factorize(codes, sort=True)
    m = v88.ventile_matrix(c, len(u), X @ theta, r)
    return m[np.isfinite(m).all(axis=1)]


def _boot_loss(m: np.ndarray, kind: str, seed: int, n_boot: int) -> np.ndarray:
    return np.array([float(loss_value(m[i], kind, np)) for i in v88._boot_idx(len(m), seed, n_boot=n_boot)])


def choose(mode: str, train: pd.DataFrame, ctx) -> dict:
    """89.2: per annual fold the path from that fold's start; the out-of-fold curve at every checkpoint; the loss on
    it (score = -loss), its block-bootstrap SE, 1-SE toward fewer steps."""
    win = mwindow(train, ctx)
    key = (mode, win.key)
    if key in _CHOICE:
        return _CHOICE[key]
    kind = LOSSES[mode]
    oof: dict[int, list] = {n: [] for n in CHECKS}
    train_loss: dict[int, list] = {n: [] for n in CHECKS}
    for fit, val in win.folds:
        snaps, tr = train_path(win.X[fit], win.r[fit], win.codes[fit], kind, CHECKS)
        for n in CHECKS:
            oof[n].append(_vm(win.X[val], win.r[val], win.codes[val], snaps[n]))
            train_loss[n].append(tr[n])
    pts = []
    for n in CHECKS:
        m = np.concatenate(oof[n]) if oof[n] else np.zeros((0, K))
        if len(m) < 3:
            continue
        lv = float(loss_value(m, kind, np))
        se = float(_boot_loss(m, kind, seed=n, n_boot=VAL_BOOT).std(ddof=1))
        pts.append({"n": n, "score": -lv, "se": se, "train": float(np.mean(train_loss[n]))})
    pick = v88.pick_1se(pts, lambda p: p["n"]) if pts else None
    out = {"n": int(pick["n"]), **{k: pick[k] for k in ("score", "se", "best")}} if pick else {"n": 0, "fallback": True}
    out["curve"] = [{k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()} for p in pts]
    out["blocks"] = len(win.folds)
    _CHOICE[key] = out
    return out


def make_fit(mode: str):
    def fit(c, train, params, scheme, ctx):
        win = mwindow(train, ctx)
        if mode == "vo_start":
            theta, ch, tr = ridge_start(win.X, win.r, win.codes), {"n": 0, "blocks": len(win.folds)}, {}
        else:
            ch = choose(mode, train, ctx)
            snaps, tr = train_path(win.X, win.r, win.codes, LOSSES[mode], (ch["n"],) if ch["n"] else (1,))
            theta = snaps[ch["n"]]
        info = {**ch, "train_loss_full": tr.get(ch["n"])}
        _log({"cand": c.name, "cutoff": v88._cut(ctx), "n_train": len(train), **info})
        weights = {col: float(w) for col, w in zip(win.xs, theta, strict=True)}
        return cmod.Fitted(c.name, params, scheme, None, list(win.xs), "weights", {"weights": weights, "v89": info})

    return fit


def candidates() -> list[cmod.Candidate]:
    by = {c.name: c for c in cmod.default_candidates()}
    v87c = {c.name: c for c in v87.candidates()}  # registers v87's fitters (lgbm_reg_w_cv)
    out = [by["mom_12_1"], by["lgbm_rank"], v87c["lgbm_reg_w_cv"]]
    for n in NEW:
        cmod._FITTERS[n] = make_fit(n)
        out.append(cmod.Candidate(n, "linear", NEW_KO[n], f"v89 {n}", rows="M", tuned=False, fitter=n))
    return out


# ------------------------------------------------------------------------------------------------ analysis


def zmat(frame: pd.DataFrame | None, name: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The quantile matrix on the per-date standardised y_e (the training loss's unit)."""
    if frame is None or name not in frame:
        return pd.DatetimeIndex([]), np.zeros((0, K))
    d = frame[["date", name, "y_e"]].dropna()
    d = d[d.groupby("date")["y_e"].transform("size").to_numpy() >= v88.mt.MIN_NAMES]
    codes, uniq = pd.factorize(d["date"], sort=True)
    m = v88.ventile_matrix(
        codes, len(uniq), d[name].to_numpy(dtype=float), zret(d["y_e"].to_numpy(float), codes, len(uniq))
    )
    ok = np.isfinite(m).all(axis=1)
    return pd.DatetimeIndex(uniq)[ok], m[ok]


def loss_stats(z: tuple, seed: int) -> dict:
    d, m = z
    if len(m) < 3:
        return {}
    draws = _boot_loss(m, "up", seed, v88.BOOT_SIMS)
    return {
        **all_losses(m),
        "up_lo": float(np.quantile(draws, 0.025)),
        "up_hi": float(np.quantile(draws, 0.975)),
        "t": [round(float(x), 2) for x in step_t(m, np)],
    }


def paired_loss(a: tuple, b: tuple, seed: int) -> dict:
    (da, ma), (db, mb) = a, b
    common = da.intersection(db)
    if len(common) < 3:
        return {}
    ma, mb = ma[da.get_indexer(common)], mb[db.get_indexer(common)]
    idx = v88._boot_idx(len(common), seed)
    draws = np.array([float(loss_value(ma[i], "up", np)) - float(loss_value(mb[i], "up", np)) for i in idx])
    return {
        "d_up": float(loss_value(ma, "up", np)) - float(loss_value(mb, "up", np)),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
    }


def insample(res, ds, names: list[str]) -> dict:
    """Final-context models on their own training month-ends: losses and rho20 (the in-sample gap)."""
    data = sel._Data(ds)
    _, _, final, _ = sel.make_contexts(data)
    rows = data.train(final.train_end, "M")
    out = {}
    for n in names:
        f = res.final.fitted.get(n) if res.final is not None else None
        if f is None:
            continue
        fr = rows[["date", "entity", "y_e", "r"]].assign(**{n: f.predict(rows)})
        dts, m = v88.vmat(fr, n)
        out[n] = {"rho": v88.rho((m - m.mean(axis=1, keepdims=True)).mean(axis=0)) if len(m) else None}
        out[n].update(all_losses(zmat(fr, n)[1]))
    return out


def choice_stats(path: Path, outer_starts: set[str]) -> dict:
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    seen, by = set(), {}
    for r in recs:
        k = (r["cand"], r["cutoff"], r["n_train"])
        if r["cutoff"] not in outer_starts or k in seen:
            continue
        seen.add(k)
        by.setdefault(r["cand"], []).append(r)
    return {
        n: {
            "fits": len(rr),
            "steps": [r.get("n") for r in rr],
            "blocks": [r.get("blocks") for r in rr],
            "val_curves": [r.get("curve") for r in rr],
        }
        for n, rr in by.items()
    }


def _f(x, spec: str = "+.4f") -> str:
    return v86._f(x, spec)


def _pct(x) -> str:
    return v88._pct(x)


def write_md(m: str, r: dict) -> Path:
    gate, v = r["gate"], r["ventile"]
    lines = [
        r.get("title") or f"# {m} 순서 손실 지표 (docs/MODELING.md 89)",
        "",
        f"- 데이터셋 `{r['snapshot']}` (84.6 고정 자료, 갈래 A 입력), 20거래일 y_e, 곡선은 비용 전·날짜 평균을 뺀 값",
        f"- 돈 지표 선정(v4.3): `{r['selected']}` (게이트 {'통과' if gate.get('passed') else '실패'}: "
        f"값 {_f(gate.get('value'))}, 하한 {_f(gate.get('lo'))}, 모멘텀 {_f(gate.get('baseline'))})",
        f"- 87·88절 재현(기준 후보 외부 1차 지표): {r['repro']}",
        "",
        "## 단조 판정과 표본 밖 순서 손실 (작을수록 순서가 뚜렷함; L_up −1 = 19칸 모두 확실히 오름, 0 = 관계 없음)",
        "",
        "| 모델 | 외부 ρ20 [95%] | D20−D1(월 %) t | 오른 칸 | 5분위 MR p · 최소 t | 연도 ρ20 양수/최소 "
        "| 외부 L_up [95%] | L_min | L_slope | 홀드아웃 ρ20 | 홀드아웃 L_up | 1차 지표 외부/홀드아웃 | 통과 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, e in v.items():
        o, h, lo, lh = (
            e.get("outer") or {},
            e.get("holdout") or {},
            e.get("loss_outer") or {},
            e.get("loss_holdout") or {},
        )
        if not o:
            continue
        sp, t = o["spread"], r["table"].get(n, {})
        verdict = "**엄격 통과**" if e["strict"] else ("**통과**" if e["pass"] else "–")
        lines.append(
            f"| `{n}` | {o['rho']:+.3f} [{o['rho_lo']:+.3f}, {o['rho_hi']:+.3f}] | "
            f"{_pct(sp['mean'])} {_f(sp['t'], '.2f')} | "
            f"{o['ups']}/19 | {o['mr5']['p']:.2f} · {o['mr5']['t_min']:+.2f} | "
            f"{o['years_pos']}/{len(o['years'])}, {o['years_min']:+.2f} | "
            f"{_f(lo.get('up'), '+.3f')} [{_f(lo.get('up_lo'), '+.3f')}, {_f(lo.get('up_hi'), '+.3f')}] | "
            f"{_f(lo.get('min'), '+.2f')} | {_f(lo.get('slope'), '+.2f')} | {_f(h.get('rho'), '+.3f')} | "
            f"{_f(lh.get('up'), '+.3f')} | {_f(t.get('outer'))} / {_f(t.get('holdout'))} | {verdict} |"
        )
    lines += ["", "## 대비 (외부 Δρ20 [95%] / 홀드아웃 Δρ20; 외부 ΔL_up [95%], 음수 = 순서가 더 뚜렷함)", ""]
    lines += [
        "| 비교 | 외부 Δρ20 [95%] | 홀드아웃 Δρ20 | 외부 ΔL_up [95%] | 홀드아웃 ΔL_up | 판정 |",
        "|---|---|---|---|---|---|",
    ]
    for k, e in r["judgments"].items():
        o, h, lo, lh = (
            e.get("outer") or {},
            e.get("holdout") or {},
            e.get("loss_outer") or {},
            e.get("loss_holdout") or {},
        )
        if not o:
            continue
        lines.append(
            f"| {k} | {o['d_rho']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}] | {_f(h.get('d_rho'), '+.3f')} | "
            f"{_f(lo.get('d_up'), '+.3f')} [{_f(lo.get('lo'), '+.3f')}, {_f(lo.get('hi'), '+.3f')}] | "
            f"{_f(lh.get('d_up'), '+.3f')} | {'**도움**' if e['helps'] else '도움 없음'} |"
        )
    head = "| 모델 | 구간 | " + " | ".join(f"D{i}" for i in range(1, K + 1)) + " |"
    lines += ["", "## 20분위 곡선 (y_e, 날짜 평균 대비 월 %)", "", head, "|---|---|" + "---|" * K]
    for n, e in v.items():
        for part, ko in (("outer", "외부"), ("holdout", "홀드아웃")):
            c = (e.get(part) or {}).get("curve")
            if c:
                lines.append(f"| `{n}` | {ko} | " + " | ".join(f"{100 * x:+.2f}" for x in c) + " |")
    lines += ["", "## 단계별 t (외부, 19칸: D1→D2 … D19→D20)", ""]
    for n, e in v.items():
        t = (e.get("loss_outer") or {}).get("t")
        if t:
            lines.append(f"- `{n}`: " + " ".join(f"{x:+.1f}" for x in t))
    lines += ["", "## 학습 구간 대 표본 밖 (마지막 학습 구간 모델, 과적합 차이)", ""]
    lines += [
        "| 모델 | 학습 ρ20 | 외부 ρ20 | 학습 L_up | 외부 L_up | 고른 단계 수(바깥·홀드아웃·최종) |",
        "|---|---|---|---|---|---|",
    ]
    for n, e in v.items():
        i, ch = r["insample"].get(n, {}), r["choices"].get(n, {})
        lines.append(
            f"| `{n}` | {_f(i.get('rho'), '+.3f')} | {_f((e.get('outer') or {}).get('rho'), '+.3f')} | "
            f"{_f(i.get('up'), '+.3f')} | {_f((e.get('loss_outer') or {}).get('up'), '+.3f')} | {ch.get('steps', '')} |"
        )
    lines += ["", "## 검증 곡선 (바깥 구간별: 단계 수 → 검증 점수 −L, 학습 손실)", ""]
    for n, ch in r["choices"].items():
        for cv in ch.get("val_curves") or []:
            if cv:
                lines.append(
                    f"- `{n}`: "
                    + ", ".join(f"{p['n']}: {p['score']:+.3f} (학습 {_f(p.get('train'), '+.3f')})" for p in cv)
                )
    comp = r["composition"]
    lines += ["", "## 지표 구성 (마지막 학습 구간 = 서빙 구간, 단위 길이 가중치 상위 20)", ""]
    for n, w in comp.items():
        lines.append(f"- `{n}`: " + ", ".join(f"`{k}` {x:+.3f}" for k, x in list(w.items())[:20]))
    p = OUT / f"{m}_{r.get('tag', 'v89')}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ run


def research_dir(s, m: str) -> Path:
    p = s.cache_path / "research" / "v89" / m / v86.SNAP[m][:12]
    p.mkdir(parents=True, exist_ok=True)
    return p


def smoke(m: str) -> None:
    """Every new fitter once on the last outer window: runs, chosen steps, seconds (no out-of-sample result)."""
    import time

    s = load_settings()
    ds = v86.load_frozen(m, s, DiskCache(s.cache_path))
    cands = candidates()
    LOG[:] = [research_dir(s, m) / "choice_smoke.jsonl"]
    data = sel._Data(ds)
    outer, _, _, _ = sel.make_contexts(data)
    ctx0 = outer[-1]
    test = data.eval_rows(ctx0.test_dates)
    for c in cands:
        if c.name not in NEW:
            continue
        t0 = time.monotonic()
        tr = data.train(ctx0.train_end, c.rows)
        fitted = c.fit(tr, {}, "none", sel._ctx(data, ctx0.test_dates[0], s, 8))
        p = fitted.predict(test)
        info = fitted.extra.get("v89") or {}
        brief = {k: info.get(k) for k in ("n", "blocks", "train_loss_full")}
        say(f"{c.name}: {len(tr)} rows, finite {np.isfinite(p).mean():.3f}, {brief} ({time.monotonic() - t0:.1f}s)")
        for pt in info.get("curve") or []:
            say(f"   {pt}")


def study(m: str) -> None:
    s = load_settings()
    cache = DiskCache(s.cache_path)
    ds = v86.load_frozen(m, s, cache)
    v88.patch_selection()
    cands = candidates()
    rd = research_dir(s, m)
    LOG[:] = [rd / "choice_log.jsonl"]
    snap0 = ds.snapshot
    data = sel._Data(ds)
    outer, holdout, final, _ = sel.make_contexts(data)
    outer_starts = {str(pd.Timestamp(c.test_dates[0]).date()) for c in [*outer, holdout, final]}
    del data
    ds.snapshot = f"{snap0}|v89|{V89_VERSION}"
    say(f"{m}: {len(ds.xs_cols)} features, {len(cands)} candidates")
    res = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    d = res.to_dict()
    names = [c.name for c in cands]
    table = v86.table(res)
    v88_t = json.loads((OUT / f"{m}_v88.json").read_text())["table"]
    repro = {
        n: (round(table[n]["outer"], 6), round(v88_t[n]["outer"], 6))
        for n in ("mom_12_1", "lgbm_rank", "lgbm_reg_w_cv")
    }
    mats, ventile = {}, {}
    for i, n in enumerate(names):
        mo, mh = v88.vmat(res.oos, n), v88.vmat(res.holdout_preds, n)
        zo, zh = zmat(res.oos, n), zmat(res.holdout_preds, n)
        mats[n] = {"outer": mo, "holdout": mh, "z_outer": zo, "z_holdout": zh}
        o, h = v88.vstats(*mo, seed=10 * i), v88.vstats(*mh, seed=10 * i + 5)
        ok = bool(o and h and o["rho_lo"] > 0 and o["spread"]["lo"] > 0 and h["rho"] > 0 and h["spread"]["mean"] > 0)
        ventile[n] = {
            "outer": o,
            "holdout": h,
            "loss_outer": loss_stats(zo, 10 * i + 7),
            "loss_holdout": loss_stats(zh, 10 * i + 8),
            "pass": ok,
            "strict": bool(ok and o["mr5"]["p"] < 0.05 and min(np.diff(o["q5"])) > 0),
        }
    judg = {}
    for j, (k, (a, b)) in enumerate(CONTRASTS.items()):
        o = v88.paired(mats[a]["outer"], mats[b]["outer"], seed=1000 + 10 * j)
        h = v88.paired(mats[a]["holdout"], mats[b]["holdout"], seed=1005 + 10 * j)
        judg[k] = {
            "a": a,
            "b": b,
            "outer": o,
            "holdout": h,
            "loss_outer": paired_loss(mats[a]["z_outer"], mats[b]["z_outer"], seed=2000 + 10 * j),
            "loss_holdout": paired_loss(mats[a]["z_holdout"], mats[b]["z_holdout"], seed=2005 + 10 * j),
            "helps": bool(o and h and o["lo"] > 0 and h["d_rho"] >= 0),
        }
    f = res.final.fitted if res.final is not None else {}
    comp = {n: dict(sorted(f[n].extra["weights"].items(), key=lambda kv: -abs(kv[1]))) for n in NEW if n in f}
    r = {
        "market": m,
        "snapshot": snap0,
        "cost_bps": float(ds.cost_bps),
        "settings": {"k": K, "checks": CHECKS, "lr": LR, "tau": TAU, "ridge0": RIDGE0, "val_boot": VAL_BOOT},
        "selected": d["selected"],
        "gate": d["gate"],
        "table": table,
        "ventile": ventile,
        "judgments": judg,
        "insample": insample(res, ds, names),
        "choices": choice_stats(LOG[0], outer_starts),
        "composition": comp,
        "repro": "같음" if all(abs(a - b) < 1e-9 for a, b in repro.values()) else f"다름 {repro}",
    }
    js = OUT / f"{m}_v89.json"
    js.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say(f"wrote {write_md(m, r)} and {js}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["study", "smoke"])
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke(a.market)
    else:
        study(a.market)


if __name__ == "__main__":
    main()

"""New model families for the stock-selection protocol (docs/MODELING.md 62, pre-declared 2026-10-16). Research only;
the torch models need the optional ``research`` extra.

Every model exposes ``predict(df) -> scores`` on the protocol's test frame (date + feature columns) through
``ZooFitted`` (a ``candidates.Fitted`` whose predict / importance are delegated), so the selection code runs them like
any other candidate. Kept in a module (not the study's __main__) so pickled final-context fits reload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcio.modeling import candidates as cmod

VAL_FRAC = 0.2


class ZooFitted(cmod.Fitted):
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.array([])
        return np.asarray(self.model.predict_frame(df, self.features), dtype=float)

    def importance(self) -> dict[str, float]:
        return dict(getattr(self.model, "importance_", {}) or {})


def _xy(train: pd.DataFrame, cols: list[str], scheme: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.nan_to_num(train[cols].to_numpy(dtype=np.float32), nan=0.0)
    y = cmod._rg_target(train, scheme).astype(np.float32)
    ok = np.isfinite(y)
    return x[ok], y[ok], train["date"].to_numpy()[ok]


def _time_split(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.sort(np.unique(dates))
    cut = u[max(1, int(len(u) * (1 - VAL_FRAC)))]
    gap = u[max(0, int(len(u) * (1 - VAL_FRAC)) - 1)]  # one month-end gap: overlapping 20-session labels
    return dates < gap, dates >= cut


# ================================================================================================ torch models


def _torch():
    import torch

    torch.set_num_threads(1)  # no torch OpenMP pool: a second libomp next to LightGBM segfaulted (67.3)
    return torch


_NETS = ("_FTNet", "_XSNet", "_SoftTreeNet")


def _define_nets() -> None:
    """Create the torch net classes on first use only: importing torch at module load put a third OpenMP runtime next
    to LightGBM / XGBoost / CatBoost and segfaulted tree-only runs (67 amendment 2). The classes are published as
    module attributes (and through __getattr__ for unpickling in a fresh process), so pickled fits reload."""
    if "_FTNet" in globals():
        return
    import torch
    from torch import nn

    class _FTNet(nn.Module):
        def __init__(self, n_in: int, d: int):
            super().__init__()
            self.d = d
            self.w = nn.Parameter(torch.randn(n_in, d) * 0.1)
            self.b = nn.Parameter(torch.zeros(n_in, d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=32, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def forward(self, x):
            tok = x[:, :, None] * self.w[None] + self.b[None]
            tok = torch.cat([self.cls.expand(len(x), 1, self.d), tok], dim=1)
            return self.head(self.enc(tok)[:, 0])

    class _XSNet(nn.Module):
        def __init__(self, n_in: int, d: int):
            super().__init__()
            self.emb = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=64, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.head = nn.Linear(d, 1)

        def forward(self, x):  # x: (1, n_stocks, n_in)
            return self.head(self.enc(self.emb(x))).squeeze(-1)

    class _SoftTreeNet(nn.Module):
        """Soft decision tree with scalar leaves (docs/MODELING.md 65): sigmoid gates, heap-indexed paths."""

        def __init__(self, n_in: int, depth: int):
            super().__init__()
            k = 2**depth
            self.depth = depth
            self.gate = nn.Linear(n_in, k - 1)
            nn.init.normal_(self.gate.weight, std=0.1)
            nn.init.zeros_(self.gate.bias)
            self.leaf = nn.Parameter(torch.zeros(k))
            idx = np.zeros((depth, k), dtype=np.int64)
            left = np.zeros((depth, k), dtype=bool)
            for leaf in range(k):  # node n -> children 2n+1 (left), 2n+2 (right)
                node = 0
                for d in range(depth):
                    bit = (leaf >> (depth - 1 - d)) & 1
                    idx[d, leaf], left[d, leaf] = node, bit == 0
                    node = 2 * node + 1 + bit
            self.register_buffer("idx", torch.tensor(idx))
            self.register_buffer("left", torch.tensor(left))

        def forward(self, x):
            g = torch.sigmoid(self.gate(x))
            p = torch.ones(x.shape[0], self.idx.shape[1], dtype=x.dtype, device=x.device)
            for d in range(self.depth):
                gd = g[:, self.idx[d]]
                p = p * torch.where(self.left[d], gd, 1.0 - gd)
            return p @ self.leaf

    for name, cls in (("_FTNet", _FTNet), ("_XSNet", _XSNet), ("_SoftTreeNet", _SoftTreeNet)):
        cls.__module__, cls.__qualname__ = __name__, name
        globals()[name] = cls


def warm_up() -> None:
    """Import torch and define the nets in the main thread before any parallel selection: the first torch import inside
    a worker thread of the selection pool deadlocked (67.3 zoo part, 2026-10-20)."""
    _torch()
    _define_nets()


def _net_class(name: str):
    _define_nets()
    return globals()[name]


def __getattr__(name: str):
    if name in _NETS:
        _define_nets()
        return globals()[name]
    raise AttributeError(name)


class _TorchRegressor:
    """Shared fit loop: MSE on the rank-gauss target, Adam, early stopping on the last 20% of training dates."""

    epochs, patience, batch, lr = 40, 4, 1024, 1e-3
    max_rows_per_epoch: int | None = None  # a random subset of the fit rows each epoch (cost cap)
    max_val_rows: int | None = None  # a fixed random subset of the validation rows (cost cap, 67 amendment 2)

    def __init__(self, seeds: int, **params):
        self.seeds, self.params, self.nets = seeds, params, []

    def build(self, n_in: int):  # pragma: no cover - subclasses
        raise NotImplementedError

    def penalty(self, net):
        return 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, dates: np.ndarray):
        torch = _torch()
        fit_m, val_m = _time_split(dates)
        xf, yf = torch.tensor(x[fit_m]), torch.tensor(y[fit_m])
        vi = np.where(val_m)[0]
        if self.max_val_rows and len(vi) > self.max_val_rows:
            vi = np.sort(np.random.default_rng(0).choice(vi, self.max_val_rows, replace=False))
        xv, yv = torch.tensor(x[vi]), torch.tensor(y[vi])
        for s in range(self.seeds):
            torch.manual_seed(s)
            net = self.build(x.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            best, state, bad = np.inf, None, 0
            gen = torch.Generator().manual_seed(s)
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(len(yf), generator=gen)
                if self.max_rows_per_epoch:
                    perm = perm[: self.max_rows_per_epoch]
                for i in range(0, len(perm), self.batch):
                    b = perm[i : i + self.batch]
                    if len(b) < 2:
                        continue
                    opt.zero_grad()
                    loss = ((net(xf[b]).squeeze(-1) - yf[b]) ** 2).mean() + self.penalty(net)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(((net(xv).squeeze(-1) - yv) ** 2).mean()) if len(yv) else 0.0
                if v < best - 1e-6:
                    best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if state is not None:
                net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
        return self

    def predict_frame(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        torch = _torch()
        x = torch.tensor(np.nan_to_num(df[cols].to_numpy(dtype=np.float32), nan=0.0))
        with torch.no_grad():
            preds = [
                np.concatenate([net(x[i : i + 4096]).squeeze(-1).numpy() for i in range(0, len(x), 4096)])
                for net in self.nets
            ]
        return np.mean(preds, axis=0)


class GKXNet(_TorchRegressor):
    """Gu-Kelly-Xiu NN3: 32-16-8 ReLU with batch norm, L1 on the weights."""

    def build(self, n_in: int):
        torch = _torch()
        nn = torch.nn
        layers, prev = [], n_in
        for h in (32, 16, 8):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 1))

    def penalty(self, net):
        l1 = float(self.params.get("l1", 1e-5))
        return l1 * sum(p.abs().sum() for n, p in net.named_parameters() if n.endswith("weight") and p.dim() == 2)


class FTTransformer(_TorchRegressor):
    """Feature-token transformer (Gorishniy et al. 2021): a per-feature linear embedding, a CLS token, 2 layers."""

    epochs, patience, batch = 12, 3, 512
    max_rows_per_epoch = 20000
    max_val_rows = 20000

    def build(self, n_in: int):
        _torch()
        return _net_class("_FTNet")(n_in, 16)


class XSAttention:
    """Cross-sectional transformer: the stocks of one date are the tokens (permutation-equivariant, no position),
    trained with the ListNet loss per date on random subsets of 128 stocks; early stopping on the last 20% dates."""

    epochs, patience, subset, lr = 30, 4, 128, 1e-3

    def __init__(self, seeds: int = 3):
        self.seeds, self.nets = seeds, []

    def _net(self, n_in: int):
        _torch()
        return _net_class("_XSNet")(n_in, 32)

    @staticmethod
    def _listnet(s, y):
        torch = _torch()
        return -(torch.softmax(y, -1) * torch.log_softmax(s, -1)).sum(-1).mean()

    def fit(self, x: np.ndarray, y: np.ndarray, dates: np.ndarray):
        torch = _torch()
        fit_m, val_m = _time_split(dates)
        groups = {d: np.where(dates == d)[0] for d in np.unique(dates)}
        fit_d = [d for d in groups if fit_m[groups[d][0]]]
        val_d = [d for d in groups if val_m[groups[d][0]]]
        for s in range(self.seeds):
            torch.manual_seed(s)
            rng = np.random.default_rng(s)
            net = self._net(x.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            best, state, bad = np.inf, None, 0
            for _ in range(self.epochs):
                net.train()
                for d in rng.permutation(fit_d):
                    idx = groups[d]
                    if len(idx) > self.subset:
                        idx = rng.choice(idx, self.subset, replace=False)
                    xb, yb = torch.tensor(x[idx])[None], torch.tensor(y[idx])[None]
                    opt.zero_grad()
                    loss = self._listnet(net(xb), yb)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(
                        np.mean(
                            [
                                float(
                                    self._listnet(
                                        net(torch.tensor(x[groups[d]])[None]), torch.tensor(y[groups[d]])[None]
                                    )
                                )
                                for d in val_d
                            ]
                        )
                    )
                if v < best - 1e-6:
                    best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if state is not None:
                net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
        return self

    def predict_frame(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        torch = _torch()
        x = np.nan_to_num(df[cols].to_numpy(dtype=np.float32), nan=0.0)
        out = np.zeros(len(df))
        dates = df["date"].to_numpy()
        for d in np.unique(dates):
            idx = np.where(dates == d)[0]
            with torch.no_grad():
                out[idx] = np.mean([net(torch.tensor(x[idx])[None]).squeeze(0).numpy() for net in self.nets], axis=0)
        return out


class SoftForest:
    """Soft decision forest as a score model (docs/MODELING.md 65): 10 depth-4 soft trees, each on a bootstrap of the
    fit rows and a 70% feature subspace; MSE on the rank-gauss target, AdamW, early stopping on the last 20% dates."""

    trees, depth, feat_frac, batch, lr, wd, epochs, patience = 10, 4, 0.7, 4096, 0.01, 1e-4, 40, 4

    def __init__(self):
        self.members: list[tuple[np.ndarray, object]] = []

    def fit(self, x: np.ndarray, y: np.ndarray, dates: np.ndarray):
        torch = _torch()
        fit_m, val_m = _time_split(dates)
        xf_all, yf_all = x[fit_m], y[fit_m]
        xv, yv = torch.tensor(x[val_m]), torch.tensor(y[val_m])
        p = x.shape[1]
        for s in range(self.trees):
            rng = np.random.default_rng(s)
            torch.manual_seed(s)
            feats = np.sort(rng.choice(p, max(1, int(np.ceil(self.feat_frac * p))), replace=False))
            boot = rng.integers(0, len(yf_all), len(yf_all))
            xf, yf = torch.tensor(xf_all[boot][:, feats]), torch.tensor(yf_all[boot])
            net = _net_class("_SoftTreeNet")(len(feats), self.depth)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            gen = torch.Generator().manual_seed(s)
            best, state, bad = np.inf, None, 0
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(len(yf), generator=gen)
                for i in range(0, len(perm), self.batch):
                    b = perm[i : i + self.batch]
                    opt.zero_grad()
                    ((net(xf[b]) - yf[b]) ** 2).mean().backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(((net(xv[:, feats]) - yv) ** 2).mean()) if len(yv) else 0.0
                if v < best - 1e-6:
                    best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if state is not None:
                net.load_state_dict(state)
            net.eval()
            self.members.append((feats, net))
        return self

    def predict_frame(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        torch = _torch()
        x = torch.tensor(np.nan_to_num(df[cols].to_numpy(dtype=np.float32), nan=0.0))
        with torch.no_grad():
            return np.mean([net(x[:, feats]).numpy() for feats, net in self.members], axis=0)


# ================================================================================================ classic models


class IPCA:
    """Instrumented PCA (Kelly-Pruitt-Su 2019): r_{i,t} = z_{i,t}' Gamma f_t + e, alternating least squares;
    expected return = z' Gamma mean(f)."""

    def __init__(self, k: int, iters: int = 15, ridge: float = 1e-4):
        self.k, self.iters, self.ridge = int(k), int(iters), float(ridge)

    def fit(self, x: np.ndarray, y: np.ndarray, dates: np.ndarray):
        z = np.hstack([x.astype(np.float64), np.ones((len(x), 1))])
        y = np.clip(y.astype(np.float64), -3, 3)
        ts = np.unique(dates)
        zz = [None] * len(ts)
        zr = np.zeros((len(ts), z.shape[1]))
        for j, t in enumerate(ts):
            m = dates == t
            zt = z[m]
            zz[j] = zt.T @ zt / m.sum()
            zr[j] = zt.T @ y[m] / m.sum()
        _, _, vt = np.linalg.svd(zr, full_matrices=False)
        g = vt[: self.k].T  # L x K
        lk = z.shape[1] * self.k
        for _ in range(self.iters):
            f = np.array(
                [np.linalg.solve(g.T @ zz[j] @ g + self.ridge * np.eye(self.k), g.T @ zr[j]) for j in range(len(ts))]
            )
            a = np.zeros((lk, lk))
            b = np.zeros(lk)
            for j in range(len(ts)):
                ff = np.outer(f[j], f[j])
                a += np.kron(zz[j], ff)
                b += np.kron(zr[j], f[j])
            g = np.linalg.solve(a + self.ridge * np.eye(lk), b).reshape(z.shape[1], self.k)
        self.gamma, self.lam = g, f.mean(axis=0)
        return self

    def predict_frame(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        x = np.nan_to_num(df[cols].to_numpy(dtype=np.float64), nan=0.0)
        z = np.hstack([x, np.ones((len(x), 1))])
        return z @ self.gamma @ self.lam


class SklearnFrame:
    """Wraps a fitted sklearn pipeline for predict_frame."""

    def __init__(self, model):
        self.model = model

    def predict_frame(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        return self.model.predict(np.nan_to_num(df[cols].to_numpy(dtype=np.float64), nan=0.0))


# ================================================================================================ fitters


def fit_zoo(c, train, params, scheme, ctx):
    cols = list(ctx.xs_cols)
    x, y, d = _xy(train, cols, scheme)
    name = c.fitter or c.name
    if name == "nn_gkx":
        model = GKXNet(5, l1=params["l1"]).fit(x, y, d)
    elif name == "ft_transformer":
        model = FTTransformer(3).fit(x, y, d)
    elif name == "xs_attention":
        model = XSAttention(3).fit(x, y, d)
    elif name == "soft_forest":
        model = SoftForest().fit(x, y, d)
    elif name == "ipca":
        model = IPCA(params["k"]).fit(x, y, d)
    elif name == "gam_spline":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer

        alpha = float(params["a"]) * len(y) / 100.0
        model = SklearnFrame(make_pipeline(SplineTransformer(n_knots=5, degree=3), Ridge(alpha=alpha)).fit(x, y))
    elif name == "knn":
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline

        model = SklearnFrame(
            make_pipeline(PCA(20, random_state=0), KNeighborsRegressor(int(params["k"]), n_jobs=8)).fit(x, y)
        )
    else:
        raise ValueError(name)
    return ZooFitted(c.name, params, scheme, model, cols, "zoo")


def candidates() -> list[cmod.Candidate]:
    for n in ("nn_gkx", "ft_transformer", "xs_attention", "ipca", "gam_spline", "knn"):
        cmod._FITTERS[n] = fit_zoo
    C = cmod.Candidate  # noqa: N806
    return [
        C("nn_gkx", "nn", "신경망 GKX(32·16·8, 배치 정규화, L1, 시드 5)", "GKX NN3", grid=[{"l1": 1e-5}, {"l1": 1e-4}]),
        C("ft_transformer", "nn", "특징 토큰 트랜스포머(FT-Transformer)", "FT-Transformer", tuned=False),
        C(
            "xs_attention",
            "nn",
            "횡단면 어텐션(같은 날 종목 간 트랜스포머, ListNet)",
            "cross-sectional attention",
            tuned=False,
        ),
        C("ipca", "factor", "도구 PCA(IPCA, 조건부 요인 모형)", "instrumented PCA", grid=[{"k": 3}, {"k": 5}]),
        C(
            "gam_spline",
            "linear",
            "스플라인 가법 모형(GAM) + 릿지",
            "spline GAM + ridge",
            grid=[{"a": 1.0}, {"a": 10.0}],
        ),
        C("knn", "neighbors", "k-최근접 이웃(PCA 20)", "k nearest neighbours", grid=[{"k": 100}, {"k": 300}]),
    ]


def soft_forest_candidate() -> cmod.Candidate:
    """docs/MODELING.md 65 (kept out of candidates() so the 62 study stays as declared)."""
    cmod._FITTERS["soft_forest"] = fit_zoo
    return cmod.Candidate(
        "soft_forest", "nn", "소프트 결정 숲(깊이 4 × 10그루, 부분공간·부트스트랩)", "soft decision forest", tuned=False
    )


__all__ = [
    "FTTransformer",
    "GKXNet",
    "IPCA",
    "SoftForest",
    "XSAttention",
    "ZooFitted",
    "candidates",
    "fit_zoo",
    "soft_forest_candidate",
]

"""docs/MODELING.md 86.6 (after the KR result, post hoc): do the blocks fetched only for names that were ever in the old
top-200 universe leak "will enter the top 200" into the 201-400 rows?

For each block, on labelled month-end rows split by the date's market-cap rank (top 200 / 201-400): coverage, and in
the 201-400 rows the within-date AUC of "block present" for the above-median label. The 201-400 rows are further split
by whether the name was in the top 200 on an EARLIER month-end (past) or only on a LATER one (future only): data
present only because of a future top-200 spell is look-ahead.

usage: v86_leak_diag.py [KR]
"""

import json
import sys

import pandas as pd

from quantcio.config import load_settings
from quantcio.data.cache import DiskCache

sys.path.insert(0, "scripts")
import v86_full_es as v86  # noqa: E402


def auc_within(g: pd.DataFrame, flag: str) -> float | None:
    from sklearn.metrics import roc_auc_score

    acc, w = 0.0, 0
    for _, d in g.groupby("date"):
        y = (d["y_e"] > d["y_e"].median()).astype(int)
        if y.nunique() < 2 or d[flag].nunique() < 2:
            continue
        acc += roc_auc_score(y, d[flag]) * len(d)
        w += len(d)
    return round(acc / w, 4) if w else None


def main() -> None:
    m = sys.argv[1] if len(sys.argv) > 1 else "KR"
    s = load_settings()
    ds = v86.load_frozen(m, s, DiskCache(s.cache_path))
    rd = v86.research_dir(s, m)
    pool = json.loads((rd / "pool.json").read_text())
    final = set(pool["final_xs"])
    f = ds.frame
    me = f[f["labelled"].to_numpy(bool) & f["is_me"].to_numpy(bool)].copy()
    me["top"] = me.groupby("date")["mcap"].rank(ascending=False, method="first") <= 200
    me = me.sort_values(["entity", "date"])
    grp = me.groupby("entity")["top"]
    me["past_top"] = grp.transform(lambda x: x.shift(1).fillna(False).astype(bool).cummax())
    me["future_top"] = grp.transform(lambda x: x[::-1].shift(1).fillna(False).astype(bool).cummax()[::-1])
    out = {}
    for tag in v86.STEPS[m]:
        js = rd / f"{tag}.json"
        if not js.exists():
            continue
        cols = [c for c in json.loads(js.read_text()).get("xs", []) if c in final]
        if not cols:
            continue
        blk = pd.read_parquet(rd / f"{tag}.parquet", columns=cols)
        blk.index = f.index
        present = blk.loc[me.index].notna().mean(axis=1)
        me["has"] = (present > 0).astype(int)
        low = me[~me["top"]]
        fut_only = low[~low["past_top"] & low["future_top"]]
        never = low[~low["past_top"] & ~low["future_top"]]
        out[tag] = {
            "n": len(cols),
            "cover_top200": round(float(me.loc[me["top"], "has"].mean()), 3),
            "cover_201_400": round(float(low["has"].mean()), 3),
            "cover_201_400_past_top": round(float(low.loc[low["past_top"], "has"].mean()), 3),
            "cover_201_400_future_only": round(float(fut_only["has"].mean()), 3) if len(fut_only) else None,
            "cover_201_400_never": round(float(never["has"].mean()), 3) if len(never) else None,
            "auc_has_201_400": auc_within(low, "has"),
            "auc_has_all": auc_within(me, "has"),
        }
        print(tag, out[tag], flush=True)
    low = me[~me["top"]]
    print(
        "201-400 rows",
        len(low),
        "past top200",
        int(low["past_top"].sum()),
        "future only",
        int((~low["past_top"] & low["future_top"]).sum()),
        "never",
        int((~low["past_top"] & ~low["future_top"]).sum()),
    )
    fut = (~low["past_top"] & low["future_top"]).astype(int).to_frame("fut").join(low[["date", "y_e"]])
    print("AUC of 'future-only top-200 name' within 201-400:", auc_within(fut, "fut"))
    (rd / "leak_diag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

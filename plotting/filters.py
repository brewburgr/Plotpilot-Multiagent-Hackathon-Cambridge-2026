from __future__ import annotations

import pandas as pd

from .specs import FilterSpec


def apply_filters(df: pd.DataFrame, filters: list[FilterSpec]) -> pd.DataFrame:
    out = df
    for f in filters:
        if f.column not in out.columns:
            # ignore unknown columns; graph should usually prevent this
            continue

        s = out[f.column]
        op = f.op
        v = f.value

        if op == "==":
            out = out[s == v]
        elif op == "!=":
            out = out[s != v]
        elif op == ">":
            out = out[s > v]
        elif op == ">=":
            out = out[s >= v]
        elif op == "<":
            out = out[s < v]
        elif op == "<=":
            out = out[s <= v]
        elif op == "in":
            if not isinstance(v, list):
                v = [v]
            out = out[s.isin(v)]
        elif op == "contains":
            out = out[s.astype(str).str.contains(str(v), na=False)]
        else:
            continue

    return out

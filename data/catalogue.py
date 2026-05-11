from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class Catalogue:
    path: Path
    df: pd.DataFrame


def load_catalogue(path: str | Path) -> Catalogue:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif p.suffix.lower() in (".csv",):
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported catalogue type: {p.suffix}")

    # Ensure an id column for point mapping
    if "row_id" not in df.columns:
        df = df.copy()
        df.insert(0, "row_id", range(len(df)))

    return Catalogue(path=p, df=df)


def describe_columns(df: pd.DataFrame, max_cols: int = 80) -> str:
    cols = list(df.columns)[:max_cols]
    lines = []
    for c in cols:
        lines.append(f"- {c} ({df[c].dtype})")
    if len(df.columns) > max_cols:
        lines.append(f"- ... ({len(df.columns)-max_cols} more)")
    return "\n".join(lines)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class RawSourceInfo:
    kind: str
    path: Path
    details: dict


def _resolve_raw_path(
    row: dict, base_dir: Optional[str] = None, path_column: Optional[str] = None
) -> Optional[Path]:
    if path_column and row.get(path_column):
        raw_path = row.get(path_column)
    else:
        # Common catalogue conventions; keep this list generous.
        raw_path = (
            row.get("trace_file_path")
            or row.get("trace_filepath")
            or row.get("trace_path")
            or row.get("trace_file")
            or row.get("raw_file_path")
            or row.get("raw_filepath")
            or row.get("raw_path")
            or row.get("raw_file")
            or row.get("file_path")
            or row.get("filepath")
            or row.get("path")
        )

    if not raw_path:
        return None

    p = Path(str(raw_path))
    if base_dir and not p.is_absolute():
        p = Path(base_dir) / p
    return p


def inspect_raw_source(
    row: dict, base_dir: Optional[str] = None, path_column: Optional[str] = None
) -> Optional[RawSourceInfo]:
    p = _resolve_raw_path(row, base_dir=base_dir, path_column=path_column)
    if not p or not p.exists():
        return None

    suf = p.suffix.lower()

    if suf in (".csv", ".txt", ".dat"):
        return RawSourceInfo(kind="table", path=p, details={"format": suf})
    if suf in (".parquet", ".pq"):
        return RawSourceInfo(kind="table", path=p, details={"format": suf})
    if suf == ".tdms":
        # We'll enumerate groups/channels lazily (fast enough for one file)
        try:
            from nptdms import TdmsFile

            tdms = TdmsFile.read(p)
            groups = tdms.groups()
            group_names = [g.name for g in groups]
            channels = {}
            for g in groups[:8]:
                channels[g.name] = [ch.name for ch in g.channels()][:40]
            return RawSourceInfo(
                kind="tdms",
                path=p,
                details={"groups": group_names, "channels": channels},
            )
        except Exception as e:
            return RawSourceInfo(kind="tdms", path=p, details={"error": str(e)})

    if suf in (".h5", ".hdf5"):
        try:
            import h5py

            keys: list[str] = []
            with h5py.File(p, "r") as f:

                def _visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        keys.append(name)

                f.visititems(_visit)
            return RawSourceInfo(kind="hdf5", path=p, details={"datasets": keys[:200]})
        except Exception as e:
            return RawSourceInfo(kind="hdf5", path=p, details={"error": str(e)})

    return RawSourceInfo(kind="unknown", path=p, details={"suffix": suf})


def load_raw_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".csv",):
        return pd.read_csv(path)
    if suf in (".txt", ".dat"):
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(f"Unsupported table format: {suf}")


def load_tdms_channels(path: Path, group: Optional[str] = None) -> dict[str, list[str]]:
    from nptdms import TdmsFile

    tdms = TdmsFile.read(path)
    out: dict[str, list[str]] = {}
    for g in tdms.groups():
        if group and g.name != group:
            continue
        out[g.name] = [ch.name for ch in g.channels()]
    return out


def load_tdms_series(
    path: Path, group: Optional[str] = None, channel: Optional[str] = None
) -> pd.DataFrame:
    """Load one TDMS channel as a 2-col dataframe: x(index), y(value)."""
    from nptdms import TdmsFile

    tdms = TdmsFile.read(path)

    # pick defaults if not provided
    gobj = None
    if group:
        gobj = tdms[group]
    else:
        groups = tdms.groups()
        gobj = groups[0] if groups else None

    if gobj is None:
        raise ValueError("No TDMS groups found")

    chs = gobj.channels()
    if not chs:
        raise ValueError(f"No channels found in TDMS group '{gobj.name}'")

    chobj = None
    if channel:
        for ch in chs:
            if ch.name == channel:
                chobj = ch
                break
    else:
        chobj = chs[0]

    if chobj is None:
        raise ValueError(f"Channel '{channel}' not found in group '{gobj.name}'")

    y = chobj[:]  # numpy array
    return pd.DataFrame({"x": range(len(y)), "y": y})


def load_hdf5_dataset(path: Path, key: str) -> pd.DataFrame:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        ds = f[key]
        arr = ds[()]

    if hasattr(arr, "dtype") and getattr(arr.dtype, "names", None):
        # structured array -> dataframe
        return pd.DataFrame(arr)

    if getattr(arr, "ndim", 0) == 1:
        return pd.DataFrame({"y": arr, "x": range(len(arr))})
    if getattr(arr, "ndim", 0) == 2:
        # treat as table
        return pd.DataFrame(arr)

    raise ValueError(f"Unsupported HDF5 dataset shape: {getattr(arr, 'shape', None)}")


def load_raw_trace(
    row: dict,
    base_dir: Optional[str] = None,
    path_column: Optional[str] = None,
    *,
    tdms_group: Optional[str] = None,
    tdms_channel: Optional[str] = None,
    hdf5_key: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load an associated raw trace, if present.

    Returns a dataframe.
    For TDMS returns columns: x, y.
    For HDF5/CSV/TXT returns table-like frames.
    """

    p = _resolve_raw_path(row, base_dir=base_dir, path_column=path_column)
    if not p or not p.exists():
        return None

    suf = p.suffix.lower()

    if suf in (".csv", ".txt", ".dat", ".parquet", ".pq"):
        return load_raw_table(p)

    if suf == ".tdms":
        return load_tdms_series(p, group=tdms_group, channel=tdms_channel)

    if suf in (".h5", ".hdf5"):
        if not hdf5_key:
            raise ValueError("hdf5_key is required for HDF5 sources")
        return load_hdf5_dataset(p, key=hdf5_key)

    return None

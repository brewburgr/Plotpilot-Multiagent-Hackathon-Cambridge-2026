from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

AggFn = Literal[
    "count",
    "mean",
    "median",
    "min",
    "max",
    "sum",
]


class FilterSpec(BaseModel):
    """A simple, safe filter language (no eval), applied to catalogue columns."""

    column: str = Field(..., description="Catalogue column name")
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: object


class RawTraceSpec(BaseModel):
    """How to load/plot raw traces referenced by catalogue rows."""

    path_column: Optional[str] = Field(
        None,
        description="Catalogue column that contains a filepath to the raw data.",
    )
    kind: Literal["overlay", "single"] = "single"

    # For table-like raw sources (csv/txt/hdf5 table)
    x: Optional[str] = Field(None, description="X column for raw table sources")
    y: Optional[str] = Field(None, description="Y column (or channel) for raw sources")

    # For TDMS
    tdms_group: Optional[str] = None
    tdms_channel: Optional[str] = None

    # For HDF5
    hdf5_key: Optional[str] = Field(
        None,
        description="Dataset path or table key within the HDF5 file.",
    )


class PlotSpec(BaseModel):
    """Validated plot specification produced by the LLM."""

    kind: Literal["scatter", "hist", "box", "bar", "raw_trace"] = "scatter"

    x: str = Field(..., description="X axis column")
    y: Optional[str] = Field(
        None, description="Y axis column (required for scatter/box)"
    )
    color: Optional[str] = Field(None, description="Column used for color grouping")

    title: Optional[str] = None

    filters: list[FilterSpec] = Field(default_factory=list)

    agg: Optional[AggFn] = Field(
        None,
        description="Optional aggregation function (used for bar/hist depending on intent)",
    )
    groupby: Optional[str] = Field(None, description="Group-by column when aggregating")

    raw: Optional[RawTraceSpec] = None

    def minimal_kind_requirements(self) -> None:
        if self.kind == "scatter" and (self.x is None or self.y is None):
            raise ValueError("scatter requires both x and y")
        if self.kind in ("box",) and (self.y is None):
            raise ValueError("box requires y")
        if self.kind == "raw_trace" and not self.raw:
            raise ValueError("raw_trace requires raw spec")


class RawTraceOperation(BaseModel):
    """Specification for raw trace plotting operations."""

    # Selection
    selection: Literal["selected", "all", "filtered", "row_ids", "grouped"] = "all"
    row_ids: Optional[List[int]] = None
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None

    # Grouping for complex selections
    group_by: Optional[str] = None  # e.g., "setting_value"
    select_per_group: Optional[
        Literal["lowest_row_id", "highest_row_id", "first", "random"]
    ] = None

    # Operations
    overlay: bool = True
    normalize: bool = False
    offset: float = 0.0
    alpha: float = 0.8
    color_by: Optional[str] = None

    # Legend customization
    legend_template: str = "row_id={row_id}"

    # Limit
    max_traces: int = 50  # Default higher, but reasonable limit

from __future__ import annotations

import plotly.express as px
import pandas as pd

from .filters import apply_filters
from .specs import PlotSpec


def build_plot(df: pd.DataFrame, spec: PlotSpec):
    spec.minimal_kind_requirements()
    dff = apply_filters(df, spec.filters)

    title = spec.title or f"{spec.kind}: {spec.x}" + (f" vs {spec.y}" if spec.y else "")

    custom = ["row_id"] if "row_id" in dff.columns else None

    if spec.kind == "scatter":
        fig = px.scatter(
            dff,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            title=title,
            custom_data=custom,
        )
    elif spec.kind == "hist":
        fig = px.histogram(
            dff, x=spec.x, color=spec.color, title=title, custom_data=custom
        )
    elif spec.kind == "box":
        fig = px.box(
            dff,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            title=title,
            custom_data=custom,
        )
    elif spec.kind == "bar":
        if spec.groupby and spec.agg:
            g = dff.groupby(spec.groupby, dropna=False)[spec.x]
            if spec.agg == "count":
                agg_df = g.count().reset_index(name="value")
            else:
                func = spec.agg
                agg_df = getattr(g, func)().reset_index(name="value")
            fig = px.bar(agg_df, x=spec.groupby, y="value", title=title)
        else:
            fig = px.bar(
                dff,
                x=spec.x,
                y=spec.y,
                color=spec.color,
                title=title,
                custom_data=custom,
            )
    else:
        fig = px.scatter(
            dff, x=spec.x, y=spec.y, color=spec.color, title=title, custom_data=custom
        )

    # Make selections usable
    fig.update_layout(dragmode="lasso")
    return fig, dff

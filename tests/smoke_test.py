from __future__ import annotations

"""Minimal internal smoke test.

This does NOT require network access / OpenAI.
It validates that we can:
- load the real catalogue
- normalize/resolve columns
- build a PlotSpec
- create a Plotly figure

Run (from repo root, with env0 activated):
  python -m agentic_plotter.tests.smoke_test
"""

from pathlib import Path

from agentic_plotter.data.catalogue import load_catalogue
from agentic_plotter.plotting.plotly_builder import build_plot
from agentic_plotter.plotting.specs import FilterSpec, PlotSpec


def main():
    catalogue_path = Path(r"C:\hackathon\Data\data_1.csv")
    cat = load_catalogue(catalogue_path)

    # This dataset uses friendly names with spaces/units; test a common user intent
    spec = PlotSpec(
        kind="scatter",
        x="Event duration (us)",
        y="Mean event current (nA)",
        color="category",
        title="Smoke test: duration vs mean current",
        filters=[FilterSpec(column="category", op="==", value=2)],
    )

    fig, dff = build_plot(cat.df, spec)

    assert len(dff) > 0, "Filtered dataframe is empty; check filter/columns"

    # Basic fig sanity
    data_len = len(fig.data) if fig and getattr(fig, "data", None) is not None else 0
    assert data_len > 0, "Figure has no traces"

    print("OK")
    print(f"Loaded rows: {len(cat.df)}")
    print(f"Filtered rows: {len(dff)}")
    print(f"Traces: {data_len}")


if __name__ == "__main__":
    main()

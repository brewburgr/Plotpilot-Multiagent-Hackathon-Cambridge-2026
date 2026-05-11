from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import dash
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
from dotenv import load_dotenv

from .data.catalogue import load_catalogue, describe_columns
from .graph.build_graph import build_analysis_graph
from .graph.state import AnalysisState
from .graph.nodes import llm_to_raw_operation, Clarification
from .plotting.plotly_builder import build_plot
from .storage.sessions import load_session, save_session
from .storage.recipes import recipe_to_dict

load_dotenv()

APP_TITLE = "Agentic Plotter"


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _ensure_session(session_id: str | None) -> str:
    return session_id or _new_session_id()


def _default_catalogue_path() -> str:
    # start with bundled example
    return str(Path(__file__).resolve().parent / "examples" / "example_catalogue.csv")


graph = build_analysis_graph()

# Dash 2.18 serves component assets locally by default in dev.
# Avoid using private/internal attributes like `app._assets_config` (removed in newer releases).
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = APP_TITLE

app.layout = html.Div(
    style={"fontFamily": "system-ui", "maxWidth": "1200px", "margin": "20px auto"},
    children=[
        html.H2(APP_TITLE),
        dcc.Store(id="session-id", storage_type="local"),
        dcc.Store(id="catalogue-store"),
        dcc.Store(id="state-store"),
        dcc.Store(id="selected-row-id"),
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "flex-end"},
            children=[
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Catalogue path (CSV/Parquet)"),
                        dcc.Input(
                            id="catalogue-path",
                            type="text",
                            value=_default_catalogue_path(),
                            style={"width": "100%"},
                        ),
                    ],
                ),
                html.Button("Load", id="load-btn"),
            ],
        ),
        html.Details(
            style={"marginTop": "8px"},
            children=[
                html.Summary("Optional: HDF5 load settings"),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "12px",
                        "marginTop": "8px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    "HDF5 key / dataset path (e.g. /table or /group/ds)"
                                ),
                                dcc.Input(
                                    id="hdf5-key",
                                    type="text",
                                    placeholder="/path/in/file",
                                    style={"width": "100%"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Raw path column override (if not auto-detected)"
                                ),
                                dcc.Input(
                                    id="raw-path-col",
                                    type="text",
                                    placeholder="file_path",
                                    style={"width": "100%"},
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "6px", "color": "#57606a", "fontSize": "12px"},
            children=[
                "Tip: LLM calls can take a few seconds. A spinner will show while plotting.",
            ],
        ),
        html.Hr(),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                html.Div(
                    children=[
                        html.H4("Chat"),
                        dcc.Textarea(
                            id="chat-input",
                            placeholder="Ask for a plot...",
                            style={"width": "100%", "height": "90px"},
                        ),
                        html.Button("Send", id="send-btn"),
                        dcc.Loading(
                            type="circle",
                            children=html.Pre(
                                id="chat-log",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "background": "#f6f8fa",
                                    "padding": "12px",
                                    "borderRadius": "8px",
                                    "height": "340px",
                                    "overflow": "auto",
                                },
                            ),
                        ),
                        html.H4("Clicked point metadata"),
                        dcc.Loading(
                            type="dot",
                            children=html.Pre(
                                id="clicked-metadata",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "background": "#f6f8fa",
                                    "padding": "12px",
                                    "borderRadius": "8px",
                                    "height": "240px",
                                    "overflow": "auto",
                                },
                            ),
                        ),
                        html.H4("Selected row_ids"),
                        dcc.Loading(
                            type="dot",
                            children=html.Pre(
                                id="selection-metadata",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "background": "#f6f8fa",
                                    "padding": "12px",
                                    "borderRadius": "8px",
                                    "height": "120px",
                                    "overflow": "auto",
                                },
                            ),
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.H4("Metadata plot"),
                        dcc.Loading(
                            type="circle",
                            children=dcc.Graph(id="plot", figure={}),
                        ),
                        dcc.Loading(type="dot", children=html.Div(id="catalogue-info")),
                        html.Hr(),
                        html.H4("Raw trace plot"),
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "8px",
                                "alignItems": "center",
                            },
                            children=[
                                html.Button(
                                    "Plot selected data point",
                                    id="plot-selected-btn",
                                    title="Uses the last clicked row on the metadata plot",
                                ),
                                html.Div(
                                    id="raw-status",
                                    style={"fontSize": "12px", "color": "#57606a"},
                                ),
                            ],
                        ),
                        dcc.Loading(
                            type="circle",
                            children=dcc.Graph(id="raw-trace-plot", figure={}),
                        ),
                        html.H4("Raw trace chat"),
                        dcc.Textarea(
                            id="raw-chat-input",
                            placeholder="Ask for raw trace operations, for example overlay, normalize, offset, color-code, alpha, but anything the user asks for can be fine...",
                            style={"width": "100%", "height": "90px"},
                        ),
                        html.Button("Send raw", id="raw-send-btn"),
                        dcc.Loading(
                            type="circle",
                            children=html.Pre(
                                id="raw-chat-log",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "background": "#f6f8fa",
                                    "padding": "12px",
                                    "borderRadius": "8px",
                                    "height": "200px",
                                    "overflow": "auto",
                                },
                            ),
                        ),
                    ]
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("session-id", "data"),
    Input("session-id", "data"),
    prevent_initial_call=False,
)
def init_session(existing):
    return _ensure_session(existing)


@app.callback(
    Output("catalogue-store", "data"),
    Output("catalogue-info", "children"),
    Input("load-btn", "n_clicks"),
    State("catalogue-path", "value"),
)
def load_catalogue_cb(n_clicks, path):
    if not n_clicks:
        # initial
        path = path or _default_catalogue_path()
    cat = load_catalogue(path)
    preview = cat.df.head(25).to_json(orient="records")
    info = html.Pre(
        "Loaded: " + str(cat.path) + "\n\nColumns:\n" + describe_columns(cat.df),
        style={"whiteSpace": "pre-wrap", "fontSize": "12px"},
    )
    return (
        {
            "path": str(cat.path),
            "columns": list(cat.df.columns),
            "preview_json": preview,
        },
        info,
    )


@app.callback(
    Output("state-store", "data"),
    Input("catalogue-store", "data"),
    State("session-id", "data"),
)
def init_state(catalogue_data, session_id):
    session_id = _ensure_session(session_id)
    rec = load_session(session_id)
    if rec:
        return rec.state

    state = AnalysisState(
        session_id=session_id,
        catalogue_path=catalogue_data.get("path") if catalogue_data else None,
        catalogue_columns=catalogue_data.get("columns", []) if catalogue_data else [],
        messages=[],
    )
    return state.model_dump()


@app.callback(
    Output("chat-log", "children"),
    Output("plot", "figure"),
    Output("state-store", "data", allow_duplicate=True),
    Input("send-btn", "n_clicks"),
    State("chat-input", "value"),
    State("catalogue-store", "data"),
    State("state-store", "data"),
    prevent_initial_call=True,
)
def chat_send(n_clicks, prompt, catalogue_data, state_data):
    if not prompt:
        raise dash.exceptions.PreventUpdate

    state = AnalysisState.model_validate(state_data)
    state.catalogue_path = (
        catalogue_data.get("path") if catalogue_data else state.catalogue_path
    )
    state.catalogue_columns = (
        catalogue_data.get("columns", []) if catalogue_data else state.catalogue_columns
    )

    # Put the prompt into state for the graph to consume
    state.last_prompt = prompt

    merged = AnalysisState.model_validate(graph.invoke(state))

    meta_fig = {}
    raw_fig = dash.no_update

    # If the LLM asked for a raw trace plot, handle it here (second plot)
    if merged.last_plot_spec and merged.last_plot_spec.kind == "raw_trace":
        if not merged.catalogue_path:
            merged.messages.append(
                {
                    "role": "assistant",
                    "content": "No catalogue loaded, so I can't select traces yet.",
                }
            )
        else:
            from .data.raw_loader import inspect_raw_source, load_raw_trace
            from .plotting.filters import apply_filters

            df = load_catalogue(merged.catalogue_path).df
            dff = apply_filters(df, merged.last_plot_spec.filters)

            if len(dff) == 0:
                merged.messages.append(
                    {
                        "role": "assistant",
                        "content": "Your filters selected 0 catalogue rows. Please widen the selection (e.g. remove filters).",
                    }
                )
            else:
                raw_spec = merged.last_plot_spec.raw
                path_col = raw_spec.path_column if raw_spec else None

                base_dir = None
                try:
                    base_dir = str(Path(merged.catalogue_path).resolve().parent)
                except Exception:
                    base_dir = None

                # Preflight: inspect first few rows to see if we can resolve raw paths
                infos = []
                for _, r in dff.head(5).iterrows():
                    info = inspect_raw_source(
                        r.to_dict(), base_dir=base_dir, path_column=path_col
                    )
                    if info:
                        infos.append(info)

                if not infos:
                    merged.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "I couldn't resolve any raw trace files for the selected rows.\n\n"
                                "Which catalogue column contains the trace file path? "
                                "(Common names: trace_file_path, file_path, filepath, raw_path.)\n"
                                "You can reply like: `use trace_file_path` or `use filepath`."
                            ),
                        }
                    )
                else:
                    # Load one or more traces and build a simple overlay figure
                    max_traces = 10 if (raw_spec and raw_spec.kind == "overlay") else 1
                    traces_loaded = 0
                    fig = {"data": [], "layout": {"title": "Raw traces"}}

                    for _, r in dff.head(max_traces).iterrows():
                        raw = load_raw_trace(
                            r.to_dict(),
                            base_dir=base_dir,
                            path_column=path_col,
                            tdms_group=(raw_spec.tdms_group if raw_spec else None),
                            tdms_channel=(raw_spec.tdms_channel if raw_spec else None),
                            hdf5_key=(raw_spec.hdf5_key if raw_spec else None),
                        )
                        if raw is None or len(raw) == 0:
                            continue

                        if set(["x", "y"]).issubset(raw.columns):
                            xcol, ycol = "x", "y"
                        else:
                            num_cols = [
                                c
                                for c in raw.columns
                                if pd.api.types.is_numeric_dtype(raw[c])
                            ]
                            if len(num_cols) >= 2:
                                xcol, ycol = num_cols[0], num_cols[1]
                            elif len(num_cols) == 1:
                                raw = raw.copy()
                                raw["x"] = range(len(raw))
                                xcol, ycol = "x", num_cols[0]
                            else:
                                continue

                        # Build a helpful legend label
                        label = f"row_id={r.get('row_id')}"
                        try:
                            if "category" in r.index:
                                label += f", category={r.get('category')}"
                        except Exception:
                            pass

                        fig["data"].append(
                            {
                                "type": "scatter",
                                "mode": "lines",
                                "x": raw[xcol],
                                "y": raw[ycol],
                                "name": label,
                            }
                        )
                        traces_loaded += 1

                    if traces_loaded == 0:
                        merged.messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "I found trace files, but couldn't load any traces to plot. "
                                    "If these are TDMS/HDF5, tell me which channel/dataset key to use."
                                ),
                            }
                        )
                    else:
                        fig.setdefault("layout", {})
                        fig["layout"].update(
                            {
                                "title": f"Raw traces (n={traces_loaded})",
                                "xaxis": {"title": "x"},
                                "yaxis": {"title": "y"},
                            }
                        )
                        raw_fig = fig

    # Otherwise render the normal metadata plot
    elif merged.last_plot_spec and merged.catalogue_path:
        df = load_catalogue(merged.catalogue_path).df
        meta_fig, _ = build_plot(df, merged.last_plot_spec)

        merged.plot_history.append(
            recipe_to_dict(
                type(
                    "R",
                    (),
                    {"prompt": merged.last_prompt, "spec": merged.last_plot_spec},
                )()
            )
        )

    save_session(merged.session_id, merged.model_dump())

    chat_text = "\n\n".join(
        [f"{m['role']}: {m['content']}" for m in merged.messages][-12:]
    )
    return chat_text, meta_fig, merged.model_dump()


@app.callback(
    Output("clicked-metadata", "children"),
    Output("selected-row-id", "data"),
    Output("state-store", "data", allow_duplicate=True),
    Input("plot", "clickData"),
    State("catalogue-store", "data"),
    State("state-store", "data"),
    prevent_initial_call=True,
)
def on_click(click_data, catalogue_data, state_data):
    if not click_data:
        raise dash.exceptions.PreventUpdate

    state = AnalysisState.model_validate(state_data)
    path = (catalogue_data or {}).get("path") or state.catalogue_path
    if not path:
        raise dash.exceptions.PreventUpdate

    df = load_catalogue(path).df

    # Plotly clickData includes pointNumber; we also ensure row_id exists and is included in dataframe
    point = click_data["points"][0]
    custom = point.get("customdata")
    row_id = None
    if isinstance(custom, (list, tuple)) and len(custom) >= 1:
        row_id = custom[0]

    row = {}
    if row_id is not None and "row_id" in df.columns:
        match = df[df["row_id"] == row_id]
        if len(match) >= 1:
            row = match.iloc[0].to_dict()

    state.last_clicked_row_id = int(row_id) if row_id is not None else None

    save_session(state.session_id, state.model_dump())
    return (
        json.dumps(row, indent=2, default=str),
        state.last_clicked_row_id,
        state.model_dump(),
    )


@app.callback(
    Output("selection-metadata", "children"),
    Output("state-store", "data", allow_duplicate=True),
    Input("plot", "selectedData"),
    State("catalogue-store", "data"),
    State("state-store", "data"),
    prevent_initial_call=True,
)
def on_select(selected_data, catalogue_data, state_data):
    if not selected_data:
        raise dash.exceptions.PreventUpdate

    state = AnalysisState.model_validate(state_data)

    row_ids: list[int] = []
    for pt in selected_data.get("points", []):
        custom = pt.get("customdata")
        if isinstance(custom, (list, tuple)) and len(custom) >= 1:
            try:
                row_ids.append(int(custom[0]))
            except Exception:
                pass

    # de-dup but keep stable order
    seen = set()
    dedup = []
    for rid in row_ids:
        if rid not in seen:
            seen.add(rid)
            dedup.append(rid)

    state.last_selected_row_ids = dedup
    save_session(state.session_id, state.model_dump())

    return json.dumps({"row_ids": dedup, "n": len(dedup)}, indent=2), state.model_dump()


@app.callback(
    Output("raw-trace-plot", "figure"),
    Output("raw-status", "children"),
    Output("chat-log", "children", allow_duplicate=True),
    Output("state-store", "data", allow_duplicate=True),
    Input("plot-selected-btn", "n_clicks"),
    State("selected-row-id", "data"),
    State("catalogue-store", "data"),
    State("state-store", "data"),
    State("hdf5-key", "value"),
    State("raw-path-col", "value"),
    prevent_initial_call=True,
)
def plot_selected_point(
    n_clicks,
    selected_row_id,
    catalogue_data,
    state_data,
    hdf5_key,
    raw_path_col,
):
    from .data.raw_loader import inspect_raw_source, load_raw_trace

    state = AnalysisState.model_validate(state_data)

    def _append_assistant(msg: str):
        state.messages.append({"role": "assistant", "content": msg})
        chat_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in state.messages][-12:]
        )
        return chat_text

    # Determine a base_dir (so relative raw paths can be resolved)
    catalogue_path = (catalogue_data or {}).get("path") or state.catalogue_path
    base_dir = None
    if catalogue_path:
        try:
            base_dir = str(Path(catalogue_path).resolve().parent)
        except Exception:
            base_dir = None

    if selected_row_id is None:
        msg = "Click a point first, then press 'Plot selected data point'."
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    if not catalogue_path:
        msg = "No catalogue loaded."
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    df = load_catalogue(catalogue_path).df
    if "row_id" not in df.columns:
        msg = "Catalogue has no row_id column."
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    match = df[df["row_id"] == selected_row_id]
    if len(match) < 1:
        msg = f"Could not find row_id={selected_row_id} in catalogue."
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    row = match.iloc[0].to_dict()

    # If the raw path column isn't obvious, ask.
    if (raw_path_col and raw_path_col not in df.columns) or (
        not raw_path_col
        and not any(
            c in df.columns
            for c in (
                "trace_file_path",
                "trace_filepath",
                "trace_path",
                "trace_file",
                "raw_file_path",
                "raw_filepath",
                "raw_path",
                "raw_file",
                "file_path",
                "filepath",
                "path",
            )
        )
    ):
        msg = (
            "I couldn't find a raw file path column automatically.\n"
            "Please tell me which catalogue column contains the raw file path "
            "(e.g. `trace_file_path`, `file_path` or `filepath`). You can also type it into 'Raw path column override'."
        )
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    info = inspect_raw_source(row, base_dir=base_dir, path_column=raw_path_col)
    if not info:
        # Provide actionable diagnostics in chat
        known_cols = [
            c
            for c in (
                raw_path_col,
                "trace_file_path",
                "trace_filepath",
                "trace_path",
                "trace_file",
                "raw_file_path",
                "raw_filepath",
                "raw_path",
                "raw_file",
                "file_path",
                "filepath",
                "path",
            )
            if c
        ]
        subset = {k: row.get(k) for k in known_cols if k in row}
        msg = (
            "Could not resolve or open the raw file for the clicked row.\n\n"
            f"Tried base_dir: {base_dir}\n"
            f"Tried columns: {known_cols}\n"
            f"Row values (paths): {json.dumps(subset, indent=2, default=str)}\n\n"
            "If the path is relative, make sure it is relative to the catalogue folder, or set 'Raw path column override'."
        )
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, "Raw file not found.", chat_text, state.model_dump()

    if info.kind == "unknown":
        msg = (
            f"Unsupported raw file type '{info.details.get('suffix')}' for {info.path.name}.\n\n"
            "Reply with the file format and what you want plotted (x vs y)."
        )
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    if info.kind == "hdf5" and not hdf5_key:
        datasets = info.details.get("datasets")
        msg = (
            f"The raw file is HDF5: {info.path.name}.\n"
            "Please provide the dataset key/path to load (use the HDF5 key input).\n"
        )
        if datasets:
            msg += "\nSome available datasets:\n" + "\n".join(
                [f"- {k}" for k in datasets[:30]]
            )
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    if info.kind == "tdms" and "error" in info.details:
        msg = f"TDMS file found but failed to inspect: {info.details['error']}"
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    if info.kind == "tdms":
        cols = info.details.get("channels", {})
        msg = (
            f"Detected TDMS file: {info.path.name}.\n"
            "I can plot a TDMS channel. If you want a specific channel, reply with group + channel name."
        )
        if cols:
            msg += "\n\nAvailable (sample) groups/channels:\n" + json.dumps(
                cols, indent=2
            )
        _append_assistant(msg)

    try:
        raw = load_raw_trace(
            row,
            base_dir=base_dir,
            path_column=raw_path_col,
            hdf5_key=hdf5_key,
        )
    except Exception as e:
        msg = f"Failed to load raw trace: {e}"
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    if raw is None or len(raw) == 0:
        msg = "Raw trace loaded but is empty (or loader returned None)."
        chat_text = _append_assistant(msg)
        save_session(state.session_id, state.model_dump())
        return {}, msg, chat_text, state.model_dump()

    # Plot
    if set(["x", "y"]).issubset(raw.columns):
        xcol, ycol = "x", "y"
    else:
        num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if len(num_cols) >= 2:
            xcol, ycol = num_cols[0], num_cols[1]
        elif len(num_cols) == 1:
            xcol, ycol = "x", num_cols[0]
            raw = raw.copy()
            raw["x"] = range(len(raw))
        else:
            msg = f"Loaded raw data from {info.path.name} but couldn't find numeric columns to plot."
            chat_text = _append_assistant(msg)
            save_session(state.session_id, state.model_dump())
            return {}, msg, chat_text, state.model_dump()

    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": raw[xcol],
                "y": raw[ycol],
                "name": f"{info.path.name}",
            }
        ],
        "layout": {
            "title": f"Raw trace: {info.path.name}",
            "xaxis": {"title": xcol},
            "yaxis": {"title": ycol},
        },
    }

    msg = f"Loaded raw trace from {info.path.name}."
    chat_text = _append_assistant(msg)
    save_session(state.session_id, state.model_dump())
    return fig, msg, chat_text, state.model_dump()


@app.callback(
    Output("raw-chat-log", "children"),
    Output("raw-trace-plot", "figure", allow_duplicate=True),
    Output("state-store", "data", allow_duplicate=True),
    Input("raw-send-btn", "n_clicks"),
    State("raw-chat-input", "value"),
    State("catalogue-store", "data"),
    State("state-store", "data"),
    State("hdf5-key", "value"),
    State("raw-path-col", "value"),
    prevent_initial_call=True,
)
def raw_chat_send(n_clicks, prompt, catalogue_data, state_data, hdf5_key, raw_path_col):
    if not prompt:
        raise dash.exceptions.PreventUpdate

    state = AnalysisState.model_validate(state_data)
    state.raw_messages.append({"role": "user", "content": prompt})

    catalogue_path = (catalogue_data or {}).get("path") or state.catalogue_path
    if not catalogue_path:
        msg = "No catalogue loaded."
        state.raw_messages.append({"role": "assistant", "content": msg})
        save_session(state.session_id, state.model_dump())
        raw_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
        )
        return raw_text, dash.no_update, state.model_dump()

    df = load_catalogue(catalogue_path).df

    # Use LLM to parse the operation
    try:
        operation = llm_to_raw_operation(state, prompt, list(df.columns))
    except Exception as e:
        msg = f"Failed to parse request: {e}"
        state.raw_messages.append({"role": "assistant", "content": msg})
        save_session(state.session_id, state.model_dump())
        raw_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
        )
        return raw_text, dash.no_update, state.model_dump()

    if isinstance(operation, Clarification):
        state.raw_messages.append({"role": "assistant", "content": operation.question})
        save_session(state.session_id, state.model_dump())
        raw_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
        )
        return raw_text, dash.no_update, state.model_dump()

    # Apply selection
    sel = df
    if operation.selection == "selected":
        if state.last_selected_row_ids and "row_id" in df.columns:
            sel = df[df["row_id"].isin(state.last_selected_row_ids)]
        else:
            msg = "No selected points found. Use box/lasso select on the metadata plot first."
            state.raw_messages.append({"role": "assistant", "content": msg})
            save_session(state.session_id, state.model_dump())
            raw_text = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
            )
            return raw_text, dash.no_update, state.model_dump()
    elif operation.selection == "row_ids" and operation.row_ids:
        if "row_id" not in df.columns:
            msg = "No row_id column in catalogue."
            state.raw_messages.append({"role": "assistant", "content": msg})
            save_session(state.session_id, state.model_dump())
            raw_text = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
            )
            return raw_text, dash.no_update, state.model_dump()
        sel = df[df["row_id"].isin(operation.row_ids)]
    elif (
        operation.selection == "filtered"
        and operation.filter_column
        and operation.filter_value
    ):
        if operation.filter_column not in df.columns:
            msg = f"Column '{operation.filter_column}' not found."
            state.raw_messages.append({"role": "assistant", "content": msg})
            save_session(state.session_id, state.model_dump())
            raw_text = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
            )
            return raw_text, dash.no_update, state.model_dump()
        series = df[operation.filter_column]
        if pd.api.types.is_numeric_dtype(series):
            try:
                val_num = float(operation.filter_value)
                sel = df[series.astype(float) == val_num]
            except:
                sel = df.iloc[0:0]
        else:
            sel = df[series.astype(str) == str(operation.filter_value)]
    elif (
        operation.selection == "grouped"
        and operation.group_by
        and operation.select_per_group
    ):
        if operation.group_by not in df.columns:
            msg = f"Group column '{operation.group_by}' not found."
            state.raw_messages.append({"role": "assistant", "content": msg})
            save_session(state.session_id, state.model_dump())
            raw_text = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
            )
            return raw_text, dash.no_update, state.model_dump()
        if operation.select_per_group == "lowest_row_id":
            sel = df.loc[df.groupby(operation.group_by)["row_id"].idxmin()]
        elif operation.select_per_group == "highest_row_id":
            sel = df.loc[df.groupby(operation.group_by)["row_id"].idxmax()]
        elif operation.select_per_group == "first":
            sel = df.groupby(operation.group_by).first().reset_index()
        # Add more options as needed

    if len(sel) == 0:
        msg = "No rows match the selection criteria."
        state.raw_messages.append({"role": "assistant", "content": msg})
        save_session(state.session_id, state.model_dump())
        raw_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
        )
        return raw_text, dash.no_update, state.model_dump()

    # Handle complex selections like "for each setting_value, plot the lowest row_id"
    # Now handled by LLM in selection="grouped"

    # Limit to max_traces
    n = min(len(sel), operation.max_traces)

    from .data.raw_loader import load_raw_trace, inspect_raw_source

    base_dir = str(Path(catalogue_path).resolve().parent)

    # Preflight path resolution
    any_ok = False
    for _, r in sel.head(5).iterrows():
        if inspect_raw_source(r.to_dict(), base_dir=base_dir, path_column=raw_path_col):
            any_ok = True
            break
    if not any_ok:
        msg = "Cannot resolve raw trace paths. Set 'Raw path column override'."
        state.raw_messages.append({"role": "assistant", "content": msg})
        save_session(state.session_id, state.model_dump())
        raw_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
        )
        return raw_text, dash.no_update, state.model_dump()

    fig = {"data": [], "layout": {"title": f"Raw traces (n={n})"}}

    loaded = 0
    for i, (_, r) in enumerate(sel.head(n).iterrows()):
        raw = load_raw_trace(
            r.to_dict(),
            base_dir=base_dir,
            path_column=raw_path_col,
            hdf5_key=hdf5_key,
        )
        if raw is None or len(raw) == 0:
            continue

        # Choose x/y
        if set(["x", "y"]).issubset(raw.columns):
            x = raw["x"].to_numpy()
            y = raw["y"].to_numpy()
        else:
            num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
            if len(num_cols) == 0:
                continue
            y = raw[num_cols[0]].to_numpy()
            x = list(range(len(y)))

        if operation.normalize:
            ymax = float(pd.Series(y).abs().max()) if len(y) else 0.0
            if ymax > 0:
                y = y / ymax

        if operation.offset != 0.0:
            y = y + (i * operation.offset)

        # Custom legend
        name = operation.legend_template.format(**r.to_dict())

        trace = {
            "type": "scatter",
            "mode": "lines",
            "x": x,
            "y": y,
            "name": name,
            "opacity": operation.alpha,
        }

        if operation.color_by and operation.color_by in r.index:
            # Simple color mapping - could be enhanced
            colors = [
                "blue",
                "red",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ]
            color_idx = hash(str(r[operation.color_by])) % len(colors)
            trace["line"] = {"color": colors[color_idx]}

        fig["data"].append(trace)
        loaded += 1

    if loaded == 0:
        msg = "Could not load any raw traces."
        state.raw_messages.append({"role": "assistant", "content": msg})
        save_session(state.session_id, state.model_dump())
        raw_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
        )
        return raw_text, dash.no_update, state.model_dump()

    msg = f"Plotted {loaded} raw traces."
    state.raw_messages.append({"role": "assistant", "content": msg})

    save_session(state.session_id, state.model_dump())
    raw_text = "\n\n".join(
        [f"{m['role']}: {m['content']}" for m in state.raw_messages][-12:]
    )
    return raw_text, fig, state.model_dump()


def main():
    port = int(os.getenv("PORT", "8050"))
    app.run_server(debug=True, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

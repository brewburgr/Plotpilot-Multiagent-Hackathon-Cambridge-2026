from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ..plotting.specs import PlotSpec


class AnalysisState(BaseModel):
    """LangGraph state for one analysis session."""

    session_id: str

    catalogue_path: Optional[str] = None
    catalogue_columns: list[str] = Field(default_factory=list)

    # Conversation
    messages: list[dict[str, Any]] = Field(default_factory=list)

    # Raw trace chat (separate transcript)
    raw_messages: list[dict[str, Any]] = Field(default_factory=list)

    # Latest plot
    last_prompt: Optional[str] = None
    last_plot_spec: Optional[PlotSpec] = None

    # Raw trace plot request (chat-driven). When set, the raw-trace callback will consume it.
    raw_trace_request: Optional[dict[str, Any]] = None

    # UI interactions
    last_clicked_row_id: Optional[int] = None
    last_selected_row_ids: list[int] = Field(default_factory=list)

    # History (recipes)
    plot_history: list[dict[str, Any]] = Field(default_factory=list)

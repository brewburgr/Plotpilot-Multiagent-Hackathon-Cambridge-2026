from __future__ import annotations

from langgraph.graph import END, StateGraph

from .state import AnalysisState
from .nodes import llm_to_plotspec


def build_analysis_graph():
    # Use a single state object; prompts are passed by setting state.last_prompt upstream.
    g = StateGraph(AnalysisState)

    g.add_node("to_plotspec", llm_to_plotspec)

    g.set_entry_point("to_plotspec")
    g.add_edge("to_plotspec", END)

    return g.compile()

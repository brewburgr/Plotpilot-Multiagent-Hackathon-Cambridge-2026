from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from .prompts import PLOT_SPEC_SYSTEM, RAW_TRACE_SYSTEM
from .state import AnalysisState
from ..plotting.specs import PlotSpec, RawTraceOperation

# Always load env from the project folder (agentic_plotter/.env)
_PROJECT_DIR = Path(__file__).resolve().parents[1]
_ENV_PATH = _PROJECT_DIR / ".env"
if not _ENV_PATH.exists():
    raise RuntimeError(f"Missing .env file at: {_ENV_PATH}")

# Override=True ensures the value in .env is the one actually used.
load_dotenv(dotenv_path=_ENV_PATH, override=True)


class Clarification(BaseModel):
    needs_clarification: bool = True
    question: str


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        api_key = api_key.strip()

    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set/empty after loading {_ENV_PATH}. "
            "Check agentic_plotter/.env (see .env.example)."
        )

    # Ensure we don't keep a stripped-only value out of os.environ
    os.environ["OPENAI_API_KEY"] = api_key

    return OpenAI(api_key=api_key)


def llm_to_plotspec(state: AnalysisState) -> Dict[str, Any]:
    """Node: convert the latest user prompt into PlotSpec (or clarification).

    This graph uses `AnalysisState` as the only input. The chat callback sets
    `state.last_prompt` before invoking the graph.
    """

    user_prompt = state.last_prompt
    if not user_prompt:
        return {
            "messages": state.messages
            + [{"role": "assistant", "content": "No prompt provided."}],
            "last_plot_spec": None,
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    cols = state.catalogue_columns

    schema_hint = PlotSpec.model_json_schema()

    user_msg = (
        "Catalogue columns:\n"
        + "\n".join([f"- {c}" for c in cols])
        + "\n\nUser request:\n"
        + user_prompt
        + "\n\nReturn ONLY JSON."
    )

    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLOT_SPEC_SYSTEM},
            {
                "role": "system",
                "content": f"PlotSpec JSON Schema (for reference): {json.dumps(schema_hint)}",
            },
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )

    text = (resp.choices[0].message.content or "").strip()

    def _parse(s: str):
        return json.loads(s)

    try:
        data = _parse(text)
    except Exception:
        stripped = text
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            stripped = stripped.lstrip("json").strip()
        data = _parse(stripped)

    if isinstance(data, dict) and data.get("needs_clarification") is True:
        clar = Clarification.model_validate(data)
        return {
            "messages": state.messages
            + [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": clar.question},
            ],
            "last_plot_spec": None,
        }

    spec = PlotSpec.model_validate(data)
    spec.minimal_kind_requirements()

    return {
        "messages": state.messages
        + [
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": f"PlotSpec: {spec.model_dump_json(indent=2)}",
            },
        ],
        "last_plot_spec": spec,
    }


def record_click(state: AnalysisState, row_id: int | None) -> Dict[str, Any]:
    return {"last_clicked_row_id": row_id}


def record_selection(state: AnalysisState, row_ids: list[int]) -> Dict[str, Any]:
    return {"last_selected_row_ids": row_ids}


def llm_to_raw_operation(
    state: AnalysisState, prompt: str, df_columns: list[str]
) -> RawTraceOperation | Clarification:
    """Convert raw trace operation prompt to RawTraceOperation using LLM."""

    schema_hint = RawTraceOperation.model_json_schema()

    user_msg = (
        "Catalogue columns:\n"
        + "\n".join([f"- {c}" for c in df_columns])
        + "\n\nUser request:\n"
        + prompt
        + "\n\nReturn ONLY JSON."
    )

    client = _client()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
        messages=[
            {"role": "system", "content": RAW_TRACE_SYSTEM},
            {
                "role": "system",
                "content": f"RawTraceOperation JSON Schema (for reference): {json.dumps(schema_hint)}",
            },
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )

    text = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(text)
    except Exception:
        stripped = text
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            stripped = stripped.lstrip("json").strip()
        data = json.loads(stripped)

    if isinstance(data, dict) and data.get("needs_clarification") is True:
        return Clarification.model_validate(data)

    return RawTraceOperation.model_validate(data)

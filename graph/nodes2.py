import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from graph.state import AnalysisState
from plotting.specs import PlotSpec
from graph.prompts import PLOT_SPEC_SYSTEM


def _find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking upwards until .env or .env.example is found."""
    start = start or Path(__file__).resolve()

    for parent in [start.parent, *start.parents]:
        if (parent / ".env").exists() or (parent / ".env.example").exists():
            return parent

    # Fallback: current working directory
    return Path.cwd()


_PROJECT_DIR = _find_project_root()
_ENV_PATH = _PROJECT_DIR / ".env"

# Load env deterministically from the discovered project folder.
# override=False means real shell environment variables still win.
load_dotenv(dotenv_path=_ENV_PATH, override=False)


class Clarification(BaseModel):
    needs_clarification: bool = True
    question: str


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Create a cached OpenAI client after loading the project .env file."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            f"Looked for .env at: {_ENV_PATH}\n\n"
            "Create a .env file containing:\n"
            "OPENAI_API_KEY=sk-...\n\n"
            "Also check that the script is being run from inside the project folder."
        )

    return OpenAI(api_key=api_key)


def _strip_json_fence(text: str) -> str:
    """Remove ```json ... ``` fences if the model returns fenced JSON."""
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


def _get_state_value(state: AnalysisState, key: str, default: Any = None) -> Any:
    """Support both TypedDict-style and object/dataclass-style AnalysisState."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def llm_to_plotspec(state: AnalysisState) -> Dict[str, Any]:
    """Node: convert the latest user prompt into PlotSpec or clarification.

    The chat callback should set `last_prompt` before invoking the graph.
    """

    user_prompt = _get_state_value(state, "last_prompt")
    messages = _get_state_value(state, "messages", [])
    cols = _get_state_value(state, "catalogue_columns", [])

    if not user_prompt:
        return {
            "messages": messages
            + [{"role": "assistant", "content": "No prompt provided."}],
            "last_plot_spec": None,
        }

    if not cols:
        return {
            "messages": messages
            + [
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": "No catalogue columns are available. Please load a catalogue first.",
                },
            ],
            "last_plot_spec": None,
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    schema_hint = PlotSpec.model_json_schema()

    user_msg = (
        "Catalogue columns:\n"
        + "\n".join([f"- {c}" for c in cols])
        + "\n\nUser request:\n"
        + user_prompt
        + "\n\nReturn ONLY valid JSON. Do not use Markdown fences."
    )

    client = _client()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLOT_SPEC_SYSTEM},
            {
                "role": "system",
                "content": (
                    "PlotSpec JSON Schema for reference:\n"
                    + json.dumps(schema_hint, indent=2)
                ),
            },
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )

    text = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_strip_json_fence(text))

    if isinstance(data, dict) and data.get("needs_clarification") is True:
        clar = Clarification.model_validate(data)

        return {
            "messages": messages
            + [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": clar.question},
            ],
            "last_plot_spec": None,
        }

    spec = PlotSpec.model_validate(data)
    spec.minimal_kind_requirements()

    return {
        "messages": messages
        + [
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": f"PlotSpec: {spec.model_dump_json(indent=2)}",
            },
        ],
        "last_plot_spec": spec.model_dump(),
    }


def record_click(state: AnalysisState, row_id: int | None) -> Dict[str, Any]:
    return {"last_clicked_row_id": row_id}


def record_selection(state: AnalysisState, row_ids: list[int]) -> Dict[str, Any]:
    return {"last_selected_row_ids": row_ids}

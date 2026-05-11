from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..plotting.specs import PlotSpec


@dataclass
class PlotRecipe:
    prompt: str
    spec: PlotSpec


def recipe_to_dict(r: PlotRecipe) -> dict[str, Any]:
    return {"prompt": r.prompt, "spec": r.spec.model_dump()}


def recipe_from_dict(d: dict[str, Any]) -> PlotRecipe:
    return PlotRecipe(prompt=d["prompt"], spec=PlotSpec.model_validate(d["spec"]))

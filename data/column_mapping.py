from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColumnMapping:
    """Optional: map user-friendly names to actual columns.

    Keep minimal for now; expand later for domain-specific synonyms.
    """

    aliases: dict[str, str]

    def resolve(self, name: str) -> str:
        return self.aliases.get(name, name)

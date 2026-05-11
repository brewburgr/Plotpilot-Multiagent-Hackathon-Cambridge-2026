from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_SESSIONS_DIR = Path(__file__).resolve().parent.parent / ".sessions"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str
    state: dict[str, Any]


def sessions_dir() -> Path:
    DEFAULT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_SESSIONS_DIR


def session_path(session_id: str) -> Path:
    return sessions_dir() / f"{session_id}.json"


def load_session(session_id: str) -> Optional[SessionRecord]:
    p = session_path(session_id)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return SessionRecord(
        session_id=data["session_id"],
        created_at=data["created_at"],
        updated_at=data.get("updated_at", data["created_at"]),
        state=data.get("state", {}),
    )


def save_session(
    session_id: str, state: dict[str, Any], created_at: Optional[str] = None
) -> SessionRecord:
    existing = load_session(session_id)
    rec = SessionRecord(
        session_id=session_id,
        created_at=created_at or (existing.created_at if existing else _utc_now_iso()),
        updated_at=_utc_now_iso(),
        state=state,
    )
    session_path(session_id).write_text(
        json.dumps(
            {
                "session_id": rec.session_id,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "state": rec.state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rec

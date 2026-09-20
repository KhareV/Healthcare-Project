"""Structured JSONL logging without patient-level payloads."""

import json
from pathlib import Path
from typing import Mapping, Union


PathLike = Union[str, Path]


class JsonlRunLogger:
    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, event: Mapping[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True) + "\n")


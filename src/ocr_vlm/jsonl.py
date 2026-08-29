"""Small JSONL readers and writers shared by the pipeline stages."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read non-empty JSONL records and report malformed line numbers."""
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            records.append(record)

    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write records to a JSONL file, replacing any existing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

"""Shared CSV reading and table naming for the Profiler and Bronze agents."""

import csv
from pathlib import Path

import pandas as pd

from core.paths import safe_stem

ENCODINGS = ("utf-8-sig", "latin-1")


def table_names(paths: list[str]) -> dict[str, str]:
    """Map each CSV path to a unique table name (its safe file stem)."""
    tables: dict[str, str] = {}
    for path in paths:
        table = safe_stem(Path(path).name)
        if table in tables.values():
            raise ValueError(f"Two files map to the same table name {table!r}; rename one of them")
        tables[str(path)] = table
    return tables


def read_csv(path: str | Path, as_text: bool = False) -> tuple[pd.DataFrame, str]:
    """Read a CSV and return (frame, encoding).

    Empty, blank or duplicate (case-insensitive) headers are rejected. With
    `as_text` every column is read as a nullable string so no value is
    reinterpreted before the STTM's explicit casts run; pandas' default
    null tokens ("", "NA", "NULL", ...) still become missing values.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    for encoding in ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                header = next(csv.reader(handle), None)
            if not header:
                raise ValueError(f"CSV file has no header: {csv_path.name}")
            names = [name.strip() for name in header]
            if any(not name for name in names):
                raise ValueError(f"CSV file contains an empty column name: {csv_path.name}")
            if len({name.casefold() for name in names}) != len(names):
                raise ValueError(f"CSV file contains duplicate column names: {csv_path.name}")
            frame = pd.read_csv(csv_path, encoding=encoding, dtype="string" if as_text else None)
            frame.columns = names
            return frame, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode CSV file: {csv_path.name}")

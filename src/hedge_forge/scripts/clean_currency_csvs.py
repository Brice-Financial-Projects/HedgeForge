"""Clean CSV files containing malformed currency fields.

Repairs currency values that were incorrectly split across CSV fields because
thousands separators were not quoted, for example:

    $99,000.00

being parsed as:

    ["$99", "000.00"]

Usage:
    python scripts/clean_currency_csvs.py data/raw/portfolio_acc001_taxable.csv
"""

import csv
import re
from pathlib import Path


CURRENCY_START = re.compile(r"^-?\$\d{1,3}(?:,\d{3})*(?:\.\d+)?$")
CURRENCY_CONTINUATION = re.compile(r"^\d{3}(?:\.\d+)?$")
NUMBER_START = re.compile(r"^-?\d+(?:\.\d+)?$")


def _should_merge(value: str, next_value: str) -> bool:
    """Return True when two adjacent fields should be joined into one value."""
    value = value.strip()
    next_value = next_value.strip()

    if not value or not next_value:
        return False

    if CURRENCY_START.fullmatch(value) and CURRENCY_CONTINUATION.fullmatch(next_value):
        return True

    if NUMBER_START.fullmatch(value) and CURRENCY_CONTINUATION.fullmatch(next_value):
        return True

    return False


def merge_currency_fields(parts: list[str]) -> list[str]:
    """Merge CSV fields that represent one comma-separated currency value."""
    merged: list[str] = []
    i = 0

    while i < len(parts):
        value = parts[i].strip()

        if i + 1 < len(parts) and _should_merge(value, parts[i + 1]):
            value = f"{value},{parts[i + 1].strip()}"
            i += 2

            while i < len(parts) and CURRENCY_CONTINUATION.fullmatch(parts[i].strip()):
                value += "," + parts[i].strip()
                i += 1

            merged.append(value)
        else:
            merged.append(value)
            i += 1

    return merged


def clean_csv(file_path: Path) -> Path:
    """Clean malformed currency fields and write a validated CSV."""
    print(f"Cleaning {file_path.name}...")

    clean_path = file_path.parent.parent / "processed" / f"{file_path.stem}_clean.csv"
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        file_path.open("r", encoding="utf-8", newline="") as infile,
        clean_path.open("w", encoding="utf-8", newline="") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        expected_fields = len(header)

        writer.writerow(header)

        for line_number, row in enumerate(reader, start=2):
            fixed = merge_currency_fields(row)

            if len(fixed) != expected_fields:
                raise ValueError(
                    f"Malformed row {line_number}: "
                    f"expected {expected_fields} fields, got {len(fixed)}. "
                    f"Row: {row}"
                )

            writer.writerow(fixed)

    print(f"Cleaned file saved to {clean_path}")
    return clean_path

"""
clean_currency_csvs.py
----------------------

Cleans CSVs with malformed currency fields (e.g., "$99, 000.00") by
merging split numeric values. Designed for HedgeForge data preparation.

Usage:
    python scripts/clean_currency_csvs.py ../data/raw/portfolio_acc001_taxable.csv
    # or to clean multiple:
    python scripts/clean_currency_csvs.py ../data/raw/*.csv
"""

# scripts/clean_currency_csvs.py
import re
from pathlib import Path
import csv

def merge_currency_fields(parts):
    """Merge pieces of split currency values inside one CSV line."""
    merged = []
    i = 0
    while i < len(parts):
        p = parts[i].strip()
        # start of a currency token like $99 or -$8
        if p.startswith("$") or p.startswith("-$"):
            val = p
            # merge following tokens until we hit something ending with a digit group or decimal
            j = i + 1
            while j < len(parts) and re.match(r"^\d{3}(\.?\d*)?$", parts[j].strip()):
                val += parts[j]
                j += 1
            merged.append(val)
            i = j
        else:
            merged.append(p)
            i += 1
    return merged


def clean_csv(file_path: Path) -> Path:
    """Fully reconstruct CSV rows, merging currency fields that were split by commas."""
    print(f"🔍 Cleaning {file_path.name} ...")

    clean_path = file_path.with_name(file_path.stem + "_clean.csv")

    with open(file_path, "r", encoding="utf-8") as infile, open(clean_path, "w", newline="", encoding="utf-8") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            fixed = merge_currency_fields(row)
            writer.writerow(fixed)

    print(f"✅ Cleaned file saved to {clean_path}")
    return clean_path

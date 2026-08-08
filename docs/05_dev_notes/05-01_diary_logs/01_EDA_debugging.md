# 🧾 HedgeForge Data Cleaning & EDA Debug Log
**Module:** `scripts/clean_currency_csvs.py`
**Focus:** Fixing malformed CSV fields (currency splits in `portfolio_acc001_taxable.csv`)
**Author:** Brice Nelson
**Last Updated:** 2025-11-06

---

## 📅 2025-11-06 — Iterative Currency Field Fixing (In Progress)
**Context:**
While loading HedgeForge’s portfolio data, `pd.read_csv()` repeatedly failed with `ParserError: Expected 17 fields, saw 19+`.
Root cause identified as **commas inside currency values** (e.g., `$99, 000.00` → interpreted as two fields).

**Actions Taken:**
1. Implemented multiple regex-based versions of `fix_currency_commas()` to merge split dollar values.
2. Confirmed problem lines via diagnostic loop that printed field counts per row.
3. Moved cleaning logic into `scripts/clean_currency_csvs.py` for reusability.
4. Transitioned from regex-only to **line-level reconstruction** using `csv.reader` + custom `merge_currency_fields()` logic.
5. Verified progress by counting number of malformed rows before/after cleaning.
6. Confirmed that issue persists for certain lines with embedded quotes and irregular spacing (requires next-phase refactor).

**Next Steps:**
- Enhance `merge_currency_fields()` to handle quoted strings and multi-comma patterns (e.g., `$1, 050.00`, `-$8, 960.00`).
- Add logging for number of merged fields and affected lines.
- Validate by confirming all non-header rows have 17 fields.
- Once validated, re-run EDA notebook to confirm successful DataFrame load.

**Status:**
🚧 *Partial fix implemented; additional parsing refinement required.*

---

## 📅 2025-11-05 — Initial Data Load Failures
**Issue:**
`ParserError: Expected 17 fields in line 3, saw 19` when reading raw CSVs (`portfolio_acc001_taxable.csv` and `portfolio_acc002_ira.csv`).

**Diagnostics:**
- Verified raw file contained fields like `$99, 000.00` and `$1, 050.00`.
- Confirmed with line-by-line CSV reader: rows contained 19–21 fields due to embedded commas.

**Actions:**
- Added `fix_currency_commas()` (basic regex version).
- Tested via CLI and notebook; confirmed partial success but inconsistent results.
- Shifted cleaner into `scripts/` folder for pipeline compatibility.

**Outcome:**
Established root cause and built foundation for reusable cleaning workflow.

---

## 📅 2025-11-04 — Notebook EDA Setup
**Goal:**
Prepare EDA for HedgeForge Phase 2 — data ingestion and validation.

**Progress:**
- Drafted `01_eda.ipynb` summary and objectives cell.
- Began loading raw CSVs for account-level testing.
- Encountered field mismatch and missing quotes → triggered data-cleaning effort above.

**Outcome:**
EDA postponed pending resolution of malformed CSVs.

---

### 🧠 Notes
- The current bug resides in **comma-split numeric fields** rather than pandas parsing.
- Once CSVs are normalized, EDA can proceed with:
  - Statistical overview of holdings
  - Return and volatility computation
  - Correlation heatmaps
- Recommend adding a `data_validation.py` utility later for automated schema checks.

---

> 🪶 *Next planned log entry:* “Successful currency merge and full DataFrame load validation.”

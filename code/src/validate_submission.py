"""Validate submission result.csv format.

Checks:
  1. File exists and is readable as UTF-8
  2. Correct columns: stock_id, weight
  3. stock_id is 6-digit string
  4. No duplicate stocks
  5. All weights > 0
  6. Weight sum <= 1
  7. Stock count <= 5
  8. All stocks are in CSI 300 pool (from training data)
  9. No empty/NaN values

Usage: python src/validate_submission.py [result_csv_path] [data_csv_path]
"""

import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def validate(result_path: str, data_path: str) -> bool:
    """Validate a submission file. Returns True if all checks pass."""
    print("=" * 60)
    print("Submission Validator")
    print("=" * 60)
    print(f"  Result file: {result_path}")
    print(f"  Data file:   {data_path}")

    all_pass = True
    checks = []

    # ── Check 0: File exists ──
    if not os.path.exists(result_path):
        print(f"\n  [FAIL] File not found: {result_path}")
        return False

    # ── Check 1: Readable as UTF-8 ──
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
        checks.append(("UTF-8 encoding", True, ""))
    except Exception as e:
        checks.append(("UTF-8 encoding", False, str(e)))
        all_pass = False
        # Can't continue without reading
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        return False

    # ── Check 2: Parse with pandas ──
    try:
        df = pd.read_csv(result_path, dtype={"stock_id": str})
    except Exception as e:
        checks.append(("CSV parse", False, str(e)))
        all_pass = False
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        return False

    # Handle empty output (all cash)
    if len(df) == 0:
        checks.append(("CSV parse", True, "empty (all cash — valid)"))
        checks.append(("Columns", True, "empty file"))
        checks.append(("Stock count <= 5", True, "0 stocks"))
        checks.append(("No duplicates", True, "empty"))
        checks.append(("Weights > 0", True, "empty"))
        checks.append(("Weight sum <= 1", True, "0.0"))
        checks.append(("CSI 300 pool", True, "empty"))

        print(f"\n  Validation Results ({len(checks)} checks):")
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        print(f"\n  => ALL CHECKS PASSED (empty portfolio = 100% cash)")
        return True

    # ── Check 3: Correct columns ──
    expected_cols = {"stock_id", "weight"}
    actual_cols = set(df.columns)
    if actual_cols == expected_cols:
        checks.append(("Columns", True, f"stock_id, weight ({len(df)} rows)"))
    else:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        detail = ""
        if missing:
            detail += f"missing: {missing} "
        if extra:
            detail += f"extra: {extra}"
        checks.append(("Columns", False, detail))
        all_pass = False

    if "stock_id" not in df.columns:
        # Can't run remaining checks
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        print(f"\n  => VALIDATION FAILED")
        return False

    # ── Check 4: Stock count ──
    n_stocks = len(df)
    if n_stocks <= 5:
        checks.append(("Stock count <= 5", True, f"{n_stocks} stocks"))
    else:
        checks.append(("Stock count <= 5", False, f"{n_stocks} stocks (max 5)"))

    # ── Check 5: No empty/NaN ──
    if df["stock_id"].isna().any() or df["weight"].isna().any():
        checks.append(("No NaN/empty", False, ""))
        all_pass = False
    else:
        checks.append(("No NaN/empty", True, ""))

    # ── Check 6: Stock ID format (6-digit string) ──
    df["stock_id"] = df["stock_id"].astype(str)
    bad_format = []
    for sid in df["stock_id"]:
        if len(sid) != 6 or not sid.isdigit():
            bad_format.append(sid)
    if not bad_format:
        checks.append(("Stock ID format (6-digit)", True, ""))
    else:
        checks.append(("Stock ID format (6-digit)", False, str(bad_format)))
        all_pass = False

    # ── Check 7: No duplicates ──
    dupes = df["stock_id"].duplicated()
    if dupes.any():
        dup_list = df["stock_id"][dupes].tolist()
        checks.append(("No duplicates", False, str(dup_list)))
        all_pass = False
    else:
        checks.append(("No duplicates", True, ""))

    # ── Check 8: All weights > 0 ──
    if (df["weight"] > 0).all():
        checks.append(("Weights > 0", True, ""))
    else:
        bad = df[df["weight"] <= 0]
        checks.append(("Weights > 0", False, f"{len(bad)} non-positive weights"))
        all_pass = False

    # ── Check 9: Weight sum <= 1 ──
    weight_sum = df["weight"].sum()
    if weight_sum <= 1.0 + 1e-6:  # small tolerance for float rounding
        checks.append(("Weight sum <= 1", True, f"{weight_sum:.6f}"))
    else:
        checks.append(("Weight sum <= 1", False, f"{weight_sum:.6f} > 1.0"))
        all_pass = False

    # ── Check 10: All stocks in CSI 300 pool ──
    if os.path.exists(data_path):
        try:
            pool = pd.read_csv(data_path, dtype={"股票代码": str, "stock_id": str})
            pool_stocks = set(pool["股票代码"].str.zfill(6).unique())
            invalid = [s for s in df["stock_id"] if s not in pool_stocks]
            if not invalid:
                checks.append(("CSI 300 pool membership", True, f"{len(pool_stocks)} stocks in pool"))
            else:
                checks.append(("CSI 300 pool membership", False, str(invalid)))
                all_pass = False
        except Exception as e:
            checks.append(("CSI 300 pool membership", False, f"could not read data: {e}"))
            all_pass = False
    else:
        checks.append(("CSI 300 pool membership", True, "data file not found, skipped"))
        print(f"  [WARN] Data file not found at {data_path}, skipping pool check")

    # ── Print results ──
    print(f"\n  Validation Results ({len(checks)} checks):")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        line = f"  [{status}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    if all_pass:
        print(f"\n  => ALL CHECKS PASSED — ready for submission")
    else:
        print(f"\n  => VALIDATION FAILED — fix issues above")

    return all_pass


if __name__ == "__main__":
    result_path = sys.argv[1] if len(sys.argv) > 1 else "output/result.csv"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data/train.csv"
    ok = validate(result_path, data_path)
    sys.exit(0 if ok else 1)

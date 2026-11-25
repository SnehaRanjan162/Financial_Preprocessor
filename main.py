from pathlib import Path
import numpy as np
import pandas as pd

INPUT_PATH = Path("dataset/financial_data.csv")
OUTPUT_PATH = Path("output/cleaned_financial_data.csv")

REQUIRED_COLUMNS = ["Company", "Quarter", "Revenue", "Profit"]
KEY_COLUMNS = ["Company", "Quarter"]
NUMERIC_COLUMNS = ["Revenue", "Profit"]

NEGATIVE_REVENUE_AS_MISSING = True
DUPLICATE_KEY_STRATEGY = "last"  # Resolves conflicting (Company, Quarter) rows


def load_data(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    if p.stat().st_size == 0:
        raise ValueError(f"File is completely empty: {p.resolve()}")
    try:
        return pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="warn")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="latin1", on_bad_lines="warn")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Matches columns case-insensitively using pure string methods (no re)."""
    rename_map = {}
    for col in df.columns:
        norm = str(col).replace(" ", "").strip().lower()
        for req in REQUIRED_COLUMNS:
            if norm == req.lower():
                rename_map[col] = req
                break
    df = df.rename(columns=rename_map)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes Company to Title Case and Quarter to uppercase (no re)."""
    blank_values = ["", "nan", "none", "null", "n/a", "<na>", "-"]

    # Standardize Company: trim and Title Case
    comp = df["Company"].astype("string").str.strip().str.title()
    df["Company"] = comp.mask(comp.str.lower().isin(blank_values), pd.NA)

    # Standardize Quarter: trim and uppercase
    q = df["Quarter"].astype("string").str.strip().str.upper()
    df["Quarter"] = q.mask(q.str.lower().isin(blank_values), pd.NA)

    return df


def parse_numeric(df: pd.DataFrame):
    """Converts strings to numeric without importing re."""
    unparseable = 0
    symbols_to_remove = ["$", "€", "£", "₹", "¥", ",", " "]
    invalid_words = ["inf", "-inf", "+inf", "nan", "none", "null", "n/a", "-"]

    for col in NUMERIC_COLUMNS:
        raw = df[col]
        had_value = raw.notna() & (~raw.astype("string").str.strip().str.lower().isin(["", "nan", "none", "<na>"]))

        if not pd.api.types.is_numeric_dtype(raw):
            s = raw.astype("string").str.strip()

            # Handle accounting parentheses: '(2000)' -> '-2000'
            has_paren = s.str.startswith("(") & s.str.endswith(")")
            s = s.mask(has_paren, "-" + s.str.slice(1, -1))

            # Remove currency symbols and commas without regex
            for sym in symbols_to_remove:
                s = s.str.replace(sym, "", regex=False)

            # Replace invalid keywords
            s = s.mask(s.str.lower().isin(invalid_words), np.nan)

            parsed = pd.to_numeric(s, errors="coerce")
        else:
            parsed = pd.to_numeric(raw, errors="coerce")

        df[col] = parsed.replace([np.inf, -np.inf], np.nan).astype("float64")
        unparseable += int((had_value & df[col].isna()).sum())

    return df, unparseable


def clean(df: pd.DataFrame):
    stats = {"original_records": len(df)}

    # 1. Standardize columns & key text
    df = standardize_columns(df)
    df = standardize_keys(df)
    df, stats["unparseable_values"] = parse_numeric(df)

    # 2. Drop rows missing Company or Quarter
    before_keys = len(df)
    df = df.dropna(subset=KEY_COLUMNS)
    stats["rows_missing_key"] = before_keys - len(df)

    # 3. [FIX MISTAKE 2] Drop ghost rows (both Revenue & Profit are missing, e.g. Google)
    before_empty = len(df)
    df = df.dropna(subset=NUMERIC_COLUMNS, how="all")
    stats["ghost_records_dropped"] = before_empty - len(df)

    # 4. Remove exact duplicate rows
    before_exact = len(df)
    df = df.drop_duplicates()
    stats["exact_duplicates_removed"] = before_exact - len(df)

    # 5. [FIX MISTAKE 1] Resolve conflicting duplicate keys (e.g. Tesla Q2 2025)
    before_conflicts = len(df)
    df = df.drop_duplicates(subset=KEY_COLUMNS, keep=DUPLICATE_KEY_STRATEGY)
    stats["conflicting_keys_resolved"] = before_conflicts - len(df)

    # 6. Flag negative revenue and treat as missing for imputation
    invalid_rev = df["Revenue"] < 0
    stats["invalid_revenue_flagged"] = int(invalid_rev.sum())
    if NEGATIVE_REVENUE_AS_MISSING:
        df["Revenue"] = np.where(invalid_rev, np.nan, df["Revenue"])

    # 7. Robust Two-Tier Imputation (Company median -> Overall median -> 0.0)
    filled_count = 0
    for col in NUMERIC_COLUMNS:
        before_nulls = int(df[col].isna().sum())
        company_median = df.groupby("Company")[col].transform("median")
        overall_median = df[col].median()
        fallback = 0.0 if pd.isna(overall_median) else overall_median

        df[col] = df[col].fillna(company_median).fillna(overall_median).fillna(fallback)
        filled_count += before_nulls - int(df[col].isna().sum())

    stats["missing_values_filled"] = filled_count

    # 8. [FIX MISTAKE 4] Clean integer formatting (no trailing .0)
    for col in NUMERIC_COLUMNS:
        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype("int64")
        else:
            df[col] = df[col].round(2)

    df = df.reset_index(drop=True)
    stats["final_records"] = len(df)
    return df, stats


def print_summary(s: dict):
    print("\n========== DATA CLEANING SUMMARY ==========")
    print(f"Original records                 : {s['original_records']}")
    print(f"Unparseable numeric values       : {s['unparseable_values']}")
    print(f"Rows missing Company/Quarter     : {s['rows_missing_key']}")
    print(f"Ghost records dropped (no data)  : {s['ghost_records_dropped']}")
    print(f"Exact duplicates removed         : {s['exact_duplicates_removed']}")
    print(f"Conflicting key rows resolved    : {s['conflicting_keys_resolved']}")
    print(f"Invalid revenue records flagged  : {s['invalid_revenue_flagged']}")
    print(f"Missing values filled            : {s['missing_values_filled']}")
    print(f"Final clean records              : {s['final_records']}")
    print("===========================================")


def main():
    try:
        df = load_data(INPUT_PATH)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    cleaned, stats = clean(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned dataset saved successfully to: {OUTPUT_PATH}")
    print_summary(stats)


if __name__ == "__main__":
    main()
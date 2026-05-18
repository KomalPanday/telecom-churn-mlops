import pandas as pd

# -----------------------------
# STANDARD COLUMN MAPPING
# -----------------------------
COLUMN_MAP = {
    "monthlycharges": "monthly_charges",
    "totalcharges": "total_charges",
    "contract": "contract",
}

EXPECTED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract"
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = df.rename(columns=COLUMN_MAP)
    return df[EXPECTED_COLUMNS]
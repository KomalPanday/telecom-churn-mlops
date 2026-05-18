import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Evidently 0.7.x keeps the old Report API under evidently.legacy.
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.schema import apply_schema


DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "telecom_mlops"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres123"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

PREDICTION_TABLES = [
    "predictions_log2",
    "predictions_log",
]

FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
]


def load_reference_data() -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "data" / "raw" / "Telcom-churn.csv"
    reference = apply_schema(pd.read_csv(csv_path))
    return clean_features(reference)


def load_current_data() -> pd.DataFrame:
    engine = create_db_engine()

    with engine.connect() as conn:
        for table_name in PREDICTION_TABLES:
            if not table_exists(conn, table_name):
                continue

            query = text(f"""
                SELECT tenure, monthly_charges, total_charges, contract
                FROM {table_name}
            """)
            current = pd.read_sql_query(query, conn)

            if not current.empty:
                print(f"Using current data from table: {table_name}")
                return clean_features(current)

    tables = ", ".join(PREDICTION_TABLES)
    raise ValueError(
        f"No prediction rows found in {tables}. "
        "Start the FastAPI app and call /predict/{customerid} at least once first."
    )


def create_db_engine():
    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
    )
    return create_engine(db_url)


def table_exists(conn, table_name: str) -> bool:
    query = text("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = :table_name
        )
    """)
    return bool(conn.execute(query, {"table_name": table_name}).scalar())


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[FEATURE_COLUMNS]

    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["contract"] = df["contract"].astype(str)

    return df.dropna(subset=FEATURE_COLUMNS)


def generate_drift_report() -> Path:
    reference = load_reference_data()
    current = load_current_data()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    output_path = PROJECT_ROOT / "reports" / "drift_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))

    return output_path


if __name__ == "__main__":
    report_path = generate_drift_report()
    print(f"Drift report generated successfully: {report_path}")

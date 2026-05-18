import psycopg2
import pandas as pd
from pathlib import Path


# -----------------------------
# Database Connection
# -----------------------------
def get_connection():

    return psycopg2.connect(
        host="localhost",
        database="telecom_mlops",
        user="postgres",
        password="postgres123",
        port=5432
    )


# -----------------------------
# Fetch Customer Features
# -----------------------------
def fetch_customer_data(customerid):

    conn = None

    try:
        conn = get_connection()

        query = """
            SELECT tenure, monthlycharges, totalcharges, contract
            FROM customer_churn
            WHERE customerid = %s;
        """

        cursor = conn.cursor()
        cursor.execute(query, (customerid,))
        row = cursor.fetchone()

        # No data found
        if row is None:
            return fetch_customer_data_from_csv(customerid)

        return pd.DataFrame(
            [row],
            columns=["tenure", "monthlycharges", "totalcharges", "contract"],
        )

    except Exception as e:
        print("DB ERROR:", str(e))
        return fetch_customer_data_from_csv(customerid)

    finally:
        if conn:
            conn.close()


def fetch_customer_data_from_csv(customerid):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telcom-churn.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    customer = df[df["customerid"] == customerid]
    if customer.empty:
        return None

    return customer[["tenure", "monthlycharges", "totalcharges", "contract"]]

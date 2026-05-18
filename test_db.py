import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="localhost",
    database="telecom_mlops",
    user="postgres",
    password="postgres123",
    port=5432
)

df = pd.read_sql("SELECT * FROM predictions_log2", conn)
print(df)

conn.close()
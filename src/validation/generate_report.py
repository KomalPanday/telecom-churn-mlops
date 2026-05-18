"""Generate validation reports for the telecom churn pipeline."""
import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\Komal_360digitmg\Project\telecom-churn-mlops\data\raw\Telcom-churn.csv")

# Generate HTML report manually
html_content = f"""
<html>
<head>
    <title>Data Validation Report</title>
</head>
<body>

<h1>Telecom Churn Data Validation Report</h1>

<h2>Dataset Shape</h2>
<p>{df.shape}</p>

<h2>Missing Values</h2>
{df.isnull().sum().to_frame().to_html()}

<h2>Data Types</h2>
{df.dtypes.to_frame().to_html()}

<h2>Duplicate Rows</h2>
<p>{df.duplicated().sum()}</p>

<h2>Sample Data</h2>
{df.head().to_html()}

</body>
</html>
"""

# Save report
report_path = "reports/validation/data_validation_report.html"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Validation report generated at: {report_path}")

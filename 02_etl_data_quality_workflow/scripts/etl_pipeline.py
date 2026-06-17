import pandas as pd, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / "data/raw/raw_transactions.csv"
out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw)
report = {
    "raw_rows": int(len(df)),
    "duplicate_transaction_ids": int(df.duplicated(subset=["transaction_id"]).sum()),
    "missing_email": int((df["customer_email"].isna() | (df["customer_email"].astype(str).str.strip()=="")).sum()),
    "missing_amount": int((df["amount"].isna() | (df["amount"].astype(str).str.strip()=="")).sum())
}

for c in ["customer_email","country","payment_method","currency","status"]:
    df[c] = df[c].astype(str).str.strip()
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df = df.drop_duplicates(subset=["transaction_id"])
df = df.dropna(subset=["transaction_date","amount"])
df = df[(df["customer_email"]!="") & (df["amount"]>=0)]
df["month"] = df["transaction_date"].dt.to_period("M").astype(str)
df["amount_eur_estimate"] = df.apply(lambda r: r["amount"] if r["currency"]=="EUR" else r["amount"]*1.17 if r["currency"]=="GBP" else r["amount"]*.92, axis=1)

report["clean_rows"] = int(len(df))
report["rows_removed"] = report["raw_rows"] - report["clean_rows"]
report["successful_transactions"] = int((df["status"]=="success").sum())
report["failed_transactions"] = int((df["status"]=="failed").sum())
report["refunded_transactions"] = int((df["status"]=="refunded").sum())

df.to_csv(out/"clean_transactions.csv", index=False)
with open(out/"quality_report.json","w") as f:
    json.dump(report, f, indent=2)
df.groupby(["month","status"], as_index=False).agg(transactions=("transaction_id","nunique"), amount_eur=("amount_eur_estimate","sum")).to_csv(out/"monthly_transaction_summary.csv", index=False)
print("ETL complete.")
print(json.dumps(report, indent=2))

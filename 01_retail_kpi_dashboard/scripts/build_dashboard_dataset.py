import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / "data/raw/retail_orders_2024.csv"
out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw, parse_dates=["order_date"])
df = df.drop_duplicates(subset=["order_id"])
for c in ["revenue","profit","quantity","cost"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["order_date","revenue","profit","quantity"])
completed = df[df["status"]=="Completed"].copy()
completed["month"] = completed["order_date"].dt.to_period("M").astype(str)
completed["profit_margin"] = completed["profit"] / completed["revenue"]

monthly = completed.groupby("month", as_index=False).agg(
    revenue=("revenue","sum"), profit=("profit","sum"), orders=("order_id","nunique"),
    customers=("customer_id","nunique"), units_sold=("quantity","sum"))
monthly["avg_order_value"] = monthly["revenue"] / monthly["orders"]
monthly["profit_margin"] = monthly["profit"] / monthly["revenue"]

category = completed.groupby("category", as_index=False).agg(revenue=("revenue","sum"), profit=("profit","sum"), orders=("order_id","nunique"))
region = completed.groupby("region", as_index=False).agg(revenue=("revenue","sum"), profit=("profit","sum"), orders=("order_id","nunique"))
channel = completed.groupby("channel", as_index=False).agg(revenue=("revenue","sum"), profit=("profit","sum"), orders=("order_id","nunique"))

completed.to_csv(out/"clean_retail_orders.csv", index=False)
monthly.to_csv(out/"monthly_kpis.csv", index=False)
category.to_csv(out/"category_kpis.csv", index=False)
region.to_csv(out/"region_kpis.csv", index=False)
channel.to_csv(out/"channel_kpis.csv", index=False)

summary = {
    "total_revenue": round(float(completed["revenue"].sum()), 2),
    "total_profit": round(float(completed["profit"].sum()), 2),
    "orders": int(completed["order_id"].nunique()),
    "customers": int(completed["customer_id"].nunique()),
    "units_sold": int(completed["quantity"].sum()),
    "avg_order_value": round(float(completed["revenue"].sum()/completed["order_id"].nunique()), 2),
    "profit_margin": round(float(completed["profit"].sum()/completed["revenue"].sum()), 4),
    "return_rate": round(float((df["status"]=="Returned").mean()), 4)
}
pd.Series(summary).to_json(out/"dashboard_summary.json", indent=2)
print("Retail KPI dashboard datasets created.")
print(summary)

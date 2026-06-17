import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data/customer_churn_sample.csv")
out = ROOT/"outputs"
out.mkdir(exist_ok=True)

eda = {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_values": df.isna().sum().to_dict(),
    "churn_rate": round(float((df["churn"]=="yes").mean()), 4),
    "avg_tenure": round(float(df["tenure_months"].mean()), 2),
    "avg_monthly_charge": round(float(df["monthly_charge"].mean()), 2)
}
pd.Series(eda).to_json(out/"eda_summary.json", indent=2)

X = df.drop(columns=["customer_id","churn"])
y = df["churn"].map({"no":0,"yes":1})
numeric = ["tenure_months","monthly_charge","support_tickets"]
categorical = ["contract_type","auto_pay"]
pre = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
model = Pipeline([("preprocessor", pre), ("classifier", LogisticRegression(max_iter=1000))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
pd.Series({"accuracy": round(float(accuracy_score(y_test, pred)), 4)}).to_json(out/"model_metrics.json", indent=2)
pd.DataFrame(classification_report(y_test, pred, output_dict=True)).transpose().to_csv(out/"classification_report.csv")
print("EDA and ML prep complete.")
print(eda)

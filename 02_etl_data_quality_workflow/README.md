# ETL Data Quality Workflow

## Goal
Clean messy transaction data and prepare it for reliable reporting.

## What this proves
- ETL
- Data validation
- Duplicate removal
- Missing value handling
- Reporting automation

## How to run
```bash
pip install pandas
python scripts/etl_pipeline.py
```

## Interview explanation
I built an ETL workflow that checks raw transaction data for duplicates, missing emails, missing amounts, invalid dates, and inconsistent formats. It outputs a clean reporting dataset and a JSON quality report.

# HCMC Real-time Traffic Forecasting MLOps

AWS-oriented MLOps project for HCMC traffic forecasting.

Current pipeline:

1. Store raw history and graph topology in S3.
2. Run offline ETL to build processed/train.parquet.
3. Check train-data quality with DuckDB.
4. Train an STGTN model and upload artifacts to S3.

Useful commands:

`ash
uv run prepare-graph
uv run run-etl
uv run check-train-data
uv run train-model
`

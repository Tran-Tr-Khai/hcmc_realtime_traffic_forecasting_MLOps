from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

common_args = {
    "owner": "tntkhai",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2026, 2, 23),
}

common_env = {
    "MINIO_ENDPOINT_URL": "{{ var.value.MINIO_ENDPOINT_URL }}",
    "MINIO_ACCESS_KEY": "{{ var.value.MINIO_ACCESS_KEY }}",
    "MINIO_SECRET_KEY": "{{ var.value.MINIO_SECRET_KEY }}",
    "MINIO_BUCKET_NAME": "{{ var.value.MINIO_BUCKET_NAME }}",
}

PROJECT_ROOT = "/opt/airflow/dags/repo"


with DAG(
    dag_id="hcmc_traffic_continuous_training",
    default_args=common_args,
    schedule_interval="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=4),
    tags=["mlops", "traffic", "continuous-training"],
) as dag:

    archive_raw = BashOperator(
        task_id="archive_daily_raw_data",
        bash_command=f"cd {PROJECT_ROOT} && python scripts/run_ingest_daily_kafka.py",
        env=common_env,
        append_env=True,
        execution_timeout=timedelta(minutes=30),
    )

    process_clean = BashOperator(
        task_id="process_and_clean_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"python scripts/run_pipeline.py "
            f"--kafka-dumps-prefix raw/kafka-dumps "
            f"--window-days 30"  # 30-day rolling window for imputation context
        ),
        env=common_env,
        append_env=True,
        execution_timeout=timedelta(hours=1),
    )

    retrain_model = BashOperator(
        task_id="retrain_stgtn_model",
        bash_command=f"cd {PROJECT_ROOT} && python scripts/train.py --epochs 50",
        env={**common_env, "CUDA_VISIBLE_DEVICES": "0"},
        append_env=True,
        execution_timeout=timedelta(hours=3),
    )

    archive_raw >> process_clean >> retrain_model
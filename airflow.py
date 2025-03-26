from airflow DAG
from datetime import datetime, timedelta
from airflow.operators import PythonOperator

default_args = {
    "owner":"airflow",
    "start_date":datetime(year=2023,month=11,day=3),
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    "ml_pipe",
    default_args=default_args,
    schedule_interval ='@weekly'
) as dag:
    train_task = PythonOperator
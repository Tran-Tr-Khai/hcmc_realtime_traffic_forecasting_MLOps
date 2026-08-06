with train_data as (
    select *
    from read_parquet('{parquet_path}')
),
timeline as (
    select
        timestamp,
        lag(timestamp) over (order by timestamp) as previous_timestamp
    from train_data
)
select
    previous_timestamp,
    timestamp,
    date_diff('minute', previous_timestamp, timestamp) as gap_minutes
from timeline
where previous_timestamp is not null
  and date(previous_timestamp) = date(timestamp)
  and date_diff('minute', previous_timestamp, timestamp) != {expected_interval_minutes}
order by previous_timestamp;
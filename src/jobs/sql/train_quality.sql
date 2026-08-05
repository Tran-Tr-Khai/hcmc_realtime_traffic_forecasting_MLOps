with train_data as (
    select *
    from read_parquet('{parquet_path}')
),
timeline as (
    select
        timestamp,
        lag(timestamp) over (order by timestamp) as previous_timestamp
    from train_data
),
duplicate_timestamps as (
    select timestamp
    from train_data
    group by 1
    having count(*) > 1
)
select
    (select count(*) from train_data) as row_count,
    (select min(timestamp) from train_data) as min_timestamp,
    (select max(timestamp) from train_data) as max_timestamp,
    (select count(*) from duplicate_timestamps) as duplicate_timestamps,
    (select count(*) - count(timestamp) from train_data) as null_timestamps,
    (
        select count(*)
        from timeline
        where previous_timestamp is not null
          and date(previous_timestamp) = date(timestamp)
          and date_diff('minute', previous_timestamp, timestamp) != {expected_interval_minutes}
    ) as unexpected_intervals;
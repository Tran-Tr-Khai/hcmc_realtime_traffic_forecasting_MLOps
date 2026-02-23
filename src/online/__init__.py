"""
Streaming module for real-time traffic data processing.

This module contains Kafka producers, consumers, and dumpers for handling
real-time traffic data streams.

Available Components:
    - Producer: Send traffic data to Kafka topic
    - Consumer: Consume and process traffic data from Kafka
    - KafkaDailyDumper: Archive Kafka messages to daily JSON files
"""

__version__ = '1.0.0'

from src.online.dumper import KafkaDailyDumper

__all__ = [
    'KafkaDailyDumper',
]


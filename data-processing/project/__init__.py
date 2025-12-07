"""
ETL Pipeline Package
Scalable architecture for processing multiple data sources into a unified fact table.
"""

from .pkd_mapper import PKDMapper
from .processors import (
    DataProcessor,
    UpadlosciProcessor,
    WskaznikFinansowyProcessor,
    QuarterlyInfoProcessorTabl4,
    QuarterlyInfoProcessorTabl5,
    QuarterlyInfoProcessorTabl7,
    MonthlyInfoProcessorTabl5
)
from .etl_pipeline_new import ETLPipeline

__all__ = [
    'PKDMapper',
    'DataProcessor',
    'UpadlosciProcessor',
    'WskaznikFinansowyProcessor',
    'QuarterlyInfoProcessorTabl4',
    'QuarterlyInfoProcessorTabl5',
    'QuarterlyInfoProcessorTabl7',
    'MonthlyInfoProcessorTabl5',
    'ETLPipeline',
]

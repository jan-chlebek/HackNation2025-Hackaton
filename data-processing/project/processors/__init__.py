"""
Data Processors Package
Contains all data processor classes for different data sources.
"""

from processors.data_processor import DataProcessor
from processors.upadlosci_processor import UpadlosciProcessor
from processors.wskaznik_finansowy_processor import WskaznikFinansowyProcessor
from processors.quarterly_info_processor_tabl4 import QuarterlyInfoProcessorTabl4
from processors.quarterly_info_processor_tabl5 import QuarterlyInfoProcessorTabl5
from processors.quarterly_info_processor_tabl7 import QuarterlyInfoProcessorTabl7

__all__ = [
    'DataProcessor',
    'UpadlosciProcessor',
    'WskaznikFinansowyProcessor',
    'QuarterlyInfoProcessorTabl4',
    'QuarterlyInfoProcessorTabl5',
    'QuarterlyInfoProcessorTabl7',
]

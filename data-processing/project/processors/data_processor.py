"""
Data Processor Module
Base class for processing different data sources.
"""

import pandas as pd
from decimal import Decimal, InvalidOperation
from pkd_mapper import PKDMapper


class DataProcessor:
    """Base class for processing different data sources."""
    
    def __init__(self, pkd_mapper: PKDMapper):
        """
        Initialize data processor.
        
        Args:
            pkd_mapper: PKD mapper instance
        """
        self.pkd_mapper = pkd_mapper
    
    def process(self, **kwargs) -> pd.DataFrame:
        """
        Process data and return standardized DataFrame.
        
        Returns:
            DataFrame with columns: [rok, pkd_2025, WSKAZNIK, wartosc]
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    @staticmethod
    def clean_value(val):
        """
        Clean and convert values to Decimal or None.
        
        Args:
            val: Value to clean
            
        Returns:
            Decimal or None
        """
        if isinstance(val, str):
            val_clean = val.replace('\xa0', '').replace(' ', '').replace(',', '.')
            if val_clean in ['', '-', '–', 'bd']:
                return None
            try:
                return Decimal(val_clean)
            except InvalidOperation:
                print(f"Warning: Invalid value encountered: '{val}'")
                return None
        if pd.isna(val):
            return None
        return val

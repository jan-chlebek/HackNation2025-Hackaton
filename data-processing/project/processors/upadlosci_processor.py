"""
Bankruptcy Data Processor Module
Processes bankruptcy (Upadłości) data from CSV files.
"""

import pandas as pd
from processors.data_processor import DataProcessor


class UpadlosciProcessor(DataProcessor):
    """Processor for bankruptcy (Upadłości) data."""
    
    def process(self, file_path: str) -> pd.DataFrame:
        """
        Process bankruptcy data from CSV.
        
        Args:
            file_path: Path to krz_pkd.csv file
            
        Returns:
            Standardized DataFrame
        """
        df = pd.read_csv(file_path, sep=";", encoding="utf-8")
        
        # Format PKD codes
        df["pkd_formatted"] = (
            df["pkd"]
            .str.strip()
            .str.upper()
            .str.replace(r"^(\d{2})(\d{2})([A-Z])$", r"\1.\2.\3", regex=True)
        )
        
        df["PKD_2007"] = df["pkd_formatted"]
        
        # Map to PKD 2025
        df_mapped = self.pkd_mapper.map_pkd_2007_to_2025(
            df, 
            pkd_column="PKD_2007",
            include_og=False
        )
        
        # Create standardized output
        result = df_mapped[["rok", "pkd_2025", "liczba_upadlosci"]].copy()
        result["WSKAZNIK"] = "Upadłość"
        result = result.rename(columns={"liczba_upadlosci": "wartosc"})
        
        return result[["rok", "pkd_2025", "WSKAZNIK", "wartosc"]]

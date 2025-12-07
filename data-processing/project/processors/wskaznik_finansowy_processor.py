"""
Financial Indicators Processor Module
Processes financial indicators (Wskaźnik Finansowy) data from CSV files.
"""

import pandas as pd
from processors.data_processor import DataProcessor


class WskaznikFinansowyProcessor(DataProcessor):
    """Processor for financial indicators (Wskaźnik Finansowy) data."""
    
    def process(self, file_path: str) -> pd.DataFrame:
        """
        Process financial indicators from CSV.
        
        Args:
            file_path: Path to wsk_fin.csv file
            
        Returns:
            Standardized DataFrame
        """
        df = pd.read_csv(file_path, sep=";", encoding="utf-8")
        
        # Format PKD codes
        df["PKD_formatted"] = df["PKD"].str.replace(r"^SEK_", "", regex=True).str.rstrip('.')
        
        # Drop unnecessary columns
        df = df.drop(columns=["NAZWA_PKD", "NUMER_NAZWA_PKD", "PKD"], errors='ignore')
        
        # Map to PKD 2025
        df_mapped = df.join(
            self.pkd_mapper.mapping_2007_2025_with_og[["symbol_2007", "symbol_2025"]].set_index("symbol_2007"),
            on="PKD_formatted"
        )
        
        # Check for unmapped codes
        unmapped_mask = df_mapped["symbol_2025"].isna()
        if unmapped_mask.any():
            missing_codes = (
                df_mapped.loc[unmapped_mask, "PKD_formatted"]
                .dropna()
                .unique()
                .tolist()
            )
            raise ValueError(f"Unmapped PKD 2007 codes: {missing_codes}")
        
        df_mapped = df_mapped.drop(columns=["PKD_formatted"])
        
        # Transpose year columns to rows
        df_transposed = df_mapped.melt(
            id_vars=['symbol_2025', 'WSKAZNIK'],
            var_name='rok',
            value_name='wartosc'
        )
        df_transposed['rok'] = df_transposed['rok'].astype(int)
        
        # Clean values
        df_transposed['wartosc'] = df_transposed['wartosc'].replace('bd', pd.NA)
        
        # Strip whitespace from WSKAZNIK names
        df_transposed['WSKAZNIK'] = df_transposed['WSKAZNIK'].str.strip()
        
        # Rename to standard schema
        result = df_transposed.rename(columns={'symbol_2025': 'pkd_2025'})
        
        return result[["rok", "pkd_2025", "WSKAZNIK", "wartosc"]]

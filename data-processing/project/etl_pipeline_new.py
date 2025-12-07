"""
ETL Pipeline Module
Main orchestrator for processing multiple data sources into a unified fact table.
"""

import os
import pandas as pd
from typing import List, Tuple
from pkd_mapper import PKDMapper
from processors.data_processor import DataProcessor


class ETLPipeline:
    """Main ETL pipeline orchestrator."""
    
    def __init__(self, mapping_file_path: str):
        """
        Initialize ETL pipeline.
        
        Args:
            mapping_file_path: Path to PKD mapping Excel file
        """
        self.pkd_mapper = PKDMapper(mapping_file_path)
        self.data_sources: List[pd.DataFrame] = []
        
    def add_data_source(
        self, 
        processor: DataProcessor, 
        name: str,
        **kwargs
    ) -> 'ETLPipeline':
        """
        Add a data source to the pipeline.
        
        Args:
            processor: DataProcessor instance
            name: Name of data source (for logging)
            **kwargs: Arguments to pass to processor.process()
            
        Returns:
            Self for chaining
        """
        print(f"Processing {name}...")
        try:
            df = processor.process(**kwargs)
            self.data_sources.append(df)
            print(f"  ✓ Loaded {len(df):,} rows")
        except Exception as e:
            print(f"  ✗ Error processing {name}: {e}")
            raise
        
        return self
    
    def combine_data(self) -> pd.DataFrame:
        """
        Combine all data sources into one DataFrame.
        
        Returns:
            Combined DataFrame with all KPI data
        """
        if not self.data_sources:
            raise ValueError("No data sources added to pipeline")
        
        combined = pd.concat(self.data_sources, ignore_index=True)
        
        # Clean values
        combined['wartosc'] = combined['wartosc'].apply(DataProcessor.clean_value)
        
        return combined
    
    def build_dictionaries(
        self, 
        combined_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build dimension tables and map indices.
        
        Args:
            combined_data: Combined fact table
            
        Returns:
            Tuple of (fact_table, wskaznik_dict, pkd_dict, pkd_typ_dict)
        """
        combined_data = combined_data.copy()
        
        # Build WSKAZNIK dictionary
        unique_wskaznik = combined_data["WSKAZNIK"].dropna().sort_values().unique()
        wskaznik_dict = {value: idx for idx, value in enumerate(unique_wskaznik)}
        wskaznik_dictionary_table = pd.DataFrame([
            {"WSKAZNIK_INDEX": idx, "WSKAZNIK": value}
            for value, idx in wskaznik_dict.items()
        ]).sort_values("WSKAZNIK_INDEX").reset_index(drop=True)
        
        combined_data["WSKAZNIK_INDEX"] = combined_data["WSKAZNIK"].map(wskaznik_dict)
        combined_data.drop(columns=["WSKAZNIK"], inplace=True)
        
        # Build PKD typ dictionary
        pkd_2025_with_og = self.pkd_mapper.pkd_2025_with_og.copy()
        unique_typ_values = pkd_2025_with_og["typ"].dropna().unique()
        typ_dict = {value: idx for idx, value in enumerate(unique_typ_values)}
        pkd_typ_dictionary_table = pd.DataFrame([
            {"TYP_INDEX": idx, "typ": value}
            for value, idx in typ_dict.items()
        ]).sort_values("TYP_INDEX").reset_index(drop=True)
        
        pkd_2025_with_og["TYP_INDEX"] = pkd_2025_with_og["typ"].map(typ_dict)
        
        # Normalize PKD codes
        pkd_2025_with_og["symbol_normalized"] = pkd_2025_with_og["symbol"].map(
            PKDMapper.normalize_pkd_code
        )
        combined_data["pkd_2025_normalized"] = combined_data["pkd_2025"].map(
            PKDMapper.normalize_pkd_code
        )
        
        # Build PKD dictionary
        unique_pkd_values = pkd_2025_with_og["symbol_normalized"].dropna().sort_values().unique()
        pkd_dict = {value: idx for idx, value in enumerate(unique_pkd_values)}
        pkd_2025_with_og["PKD_INDEX"] = pkd_2025_with_og["symbol_normalized"].map(pkd_dict)
        
        pkd_dictionary_table = (
            pkd_2025_with_og[["PKD_INDEX", "symbol", "nazwa", "TYP_INDEX"]]
            .drop_duplicates(subset=["PKD_INDEX"])
            .sort_values("PKD_INDEX")
            .reset_index(drop=True)
        )
        
        # Map PKD indices to fact table
        combined_data["PKD_INDEX"] = combined_data["pkd_2025_normalized"].map(pkd_dict)
        
        # Check for unmapped PKD codes
        missing_pkd_mask = combined_data["PKD_INDEX"].isna()
        if missing_pkd_mask.any():
            missing_pkd_codes = (
                combined_data.loc[missing_pkd_mask, "pkd_2025_normalized"]
                .dropna()
                .unique()
                .tolist()
            )
            print(f"Warning: Unmapped PKD codes in fact table: {missing_pkd_codes}")
            combined_data = combined_data[~missing_pkd_mask]
        
        # Drop temporary columns
        combined_data.drop(
            columns=["pkd_2025", "pkd_2025_normalized"], 
            inplace=True, 
            errors='ignore'
        )
        
        return (
            combined_data,
            wskaznik_dictionary_table,
            pkd_dictionary_table,
            pkd_typ_dictionary_table
        )
    
    def save_results(
        self,
        fact_table: pd.DataFrame,
        wskaznik_dict: pd.DataFrame,
        pkd_dict: pd.DataFrame,
        pkd_typ_dict: pd.DataFrame,
        output_dir: str = "../results"
    ):
        """
        Save all tables to CSV files.
        
        Args:
            fact_table: Fact table with KPI values
            wskaznik_dict: WSKAZNIK dimension table
            pkd_dict: PKD dimension table
            pkd_typ_dict: PKD type dimension table
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        wskaznik_dict.to_csv(
            os.path.join(output_dir, "wskaznik_dictionary.csv"),
            sep=";", encoding="utf-8", index=False
        )
        pkd_dict.to_csv(
            os.path.join(output_dir, "pkd_dictionary.csv"),
            sep=";", encoding="utf-8", index=False
        )
        pkd_typ_dict.to_csv(
            os.path.join(output_dir, "pkd_typ_dictionary.csv"),
            sep=";", encoding="utf-8", index=False
        )
        fact_table.to_csv(
            os.path.join(output_dir, "kpi-value-table.csv"),
            sep=";", encoding="utf-8", index=False
        )
        
        print(f"\n✓ All tables saved to {output_dir}/")
        print(f"  - Fact table: {len(fact_table):,} rows")
        print(f"  - WSKAZNIK dictionary: {len(wskaznik_dict):,} indicators")
        print(f"  - PKD dictionary: {len(pkd_dict):,} codes")
        print(f"  - PKD type dictionary: {len(pkd_typ_dict):,} types")

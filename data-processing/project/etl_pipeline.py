"""
ETL Pipeline for KPI Data Processing
Scalable architecture for processing multiple data sources into a unified fact table.
"""

import os
import pandas as pd
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path


class PKDMapper:
    """Handles PKD code mapping and normalization."""
    
    def __init__(self, mapping_file_path: str):
        """
        Initialize PKD mapper with mapping file.
        
        Args:
            mapping_file_path: Path to Excel file with PKD mappings
        """
        self.mapping_2007_2025 = pd.read_excel(
            mapping_file_path, 
            sheet_name="MAP_PKD_2007_2025"
        )
        self.pkd_2025 = pd.read_excel(
            mapping_file_path, 
            sheet_name="PKD_2025"
        )
        
        # Add OG (OGÓŁEM) to mappings
        self.mapping_2007_2025_with_og = pd.concat([
            self.mapping_2007_2025,
            pd.DataFrame({'symbol_2007': ['OG'], 'symbol_2025': ['OG']})
        ], ignore_index=True)
        
        self.pkd_2025_with_og = pd.concat([
            self.pkd_2025,
            pd.DataFrame({'typ': ['OGÓŁEM'], 'symbol': ['OG'], 'nazwa': ['OGÓŁEM']})
        ], ignore_index=True)
    
    @staticmethod
    def normalize_pkd_code(value) -> str:
        """
        Normalize PKD code for consistent matching.
        
        Args:
            value: PKD code to normalize
            
        Returns:
            Normalized PKD code
        """
        if pd.isna(value):
            return pd.NA
        code = str(value).strip().upper()
        if code.endswith('.0'):
            code = code[:-2]
        return code
    
    def map_pkd_2007_to_2025(
        self, 
        df: pd.DataFrame, 
        pkd_column: str = "PKD_2007",
        include_og: bool = True
    ) -> pd.DataFrame:
        """
        Map PKD 2007 codes to PKD 2025 codes.
        
        Args:
            df: DataFrame with PKD 2007 codes
            pkd_column: Name of column containing PKD 2007 codes
            include_og: Whether to use mapping with OG included
            
        Returns:
            DataFrame with mapped PKD 2025 codes
        """
        df = df.copy()
        
        # Extract letter suffix if present
        df["pkd_formatted_no_letter"] = df[pkd_column].str.replace(r"\.[A-Z]$", "", regex=True)
        df["pkd_letter"] = df[pkd_column].str.extract(r"\.([A-Z])$")
        
        # Choose appropriate mapping
        mapping = self.mapping_2007_2025_with_og if include_og else self.mapping_2007_2025
        
        # Perform mapping
        df_mapped = df.merge(
            mapping[["symbol_2007", "symbol_2025"]],
            how="left",
            left_on="pkd_formatted_no_letter",
            right_on="symbol_2007"
        )
        
        # Combine with letter suffix
        df_mapped["pkd_2025"] = df_mapped.apply(
            lambda row: (
                row["symbol_2025"] + '.' + row["pkd_letter"] 
                if pd.notna(row["pkd_letter"]) and pd.notna(row["symbol_2025"]) 
                else row["symbol_2025"]
            ),
            axis=1
        )
        
        # Check for unmapped codes
        unmapped_mask = df_mapped["symbol_2025"].isna()
        if unmapped_mask.any():
            missing_codes = (
                df_mapped.loc[unmapped_mask, "pkd_formatted_no_letter"]
                .dropna()
                .unique()
                .tolist()
            )
            print(f"Warning: Unmapped PKD 2007 codes: {missing_codes}")
            df_mapped = df_mapped[~unmapped_mask]
        
        return df_mapped


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
        
        # Rename to standard schema
        result = df_transposed.rename(columns={'symbol_2025': 'pkd_2025'})
        
        return result[["rok", "pkd_2025", "WSKAZNIK", "wartosc"]]


class QuarterlyInfoProcessor(DataProcessor):
    """Processor for quarterly employment forecast data."""
    
    def process(
        self, 
        folder_path: str, 
        sheet_name: str = 'Tabl 4',
        wskaznik_prefix: str = "Przewidywana liczba pracujących"
    ) -> pd.DataFrame:
        """
        Process quarterly info from multiple Excel files.
        
        Args:
            folder_path: Path to folder with Excel files
            sheet_name: Name of sheet to extract
            wskaznik_prefix: Prefix to add to indicator names
            
        Returns:
            Standardized DataFrame
        """
        all_data = []
        files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and not f.startswith('~$')]
        
        if not files:
            print(f"No Excel files found in {folder_path}")
            return pd.DataFrame(columns=["rok", "pkd_2025", "WSKAZNIK", "wartosc"])

        for file in files:
            file_path = os.path.join(folder_path, file)
            
            # Load Excel and find correct sheet
            xl = pd.ExcelFile(file_path)
            candidates = [sheet_name, 'Tabl. 4'] if sheet_name == 'Tabl 4' else [sheet_name]
            
            target_sheet = None
            for candidate in candidates:
                if candidate in xl.sheet_names:
                    target_sheet = candidate
                    break
            
            if target_sheet is None:
                raise ValueError(
                    f"Sheet '{sheet_name}' not found in '{file}'. "
                    f"Available: {xl.sheet_names}"
                )
            
            # Find header row
            df_temp = pd.read_excel(file_path, sheet_name=target_sheet, header=None)
            header_idx = None
            for idx, row in df_temp.iterrows():
                if isinstance(row.iloc[0], str) and "Sekcja" in row.iloc[0]:
                    header_idx = idx
                    break
            
            if header_idx is not None:
                df = pd.read_excel(file_path, sheet_name=target_sheet, header=header_idx)
            else:
                df = pd.read_excel(file_path, sheet_name=target_sheet)
            
            df['Source_File'] = file
            all_data.append(df)

        df_all = pd.concat(all_data, ignore_index=True)
        
        # Filter and clean
        df_all = df_all[df_all['Sekcja'] != 'Brak PKD']
        df_all.loc[df_all['Sekcja'].str.contains('POLSKA', na=False), 'Sekcja'] = 'OG'
        
        # Format PKD codes
        if 'Dział' in df_all.columns:
            df_all['Dział'] = (
                df_all['Dział']
                .astype(str)
                .str.replace(r'\.0$', '', regex=True)
                .replace('nan', None)
            )
            df_all['Dział'] = df_all['Dział'].apply(
                lambda x: x.zfill(2) if x is not None else x
            )

        if 'Podklasa' in df_all.columns:
            df_all['Podklasa'] = (
                df_all['Podklasa']
                .astype(str)
                .str.replace(r'^(\d{2})(\d{2})([A-Z])$', r'\1.\2.\3', regex=True)
                .replace('nan', None)
            )

        # Consolidate PKD columns
        df_all['PKD_2007'] = (
            df_all['Podklasa']
            .combine_first(df_all['Dział'])
            .combine_first(df_all['Sekcja'])
        )
        
        df_all.drop(columns=['Sekcja', 'Dział', 'Podklasa'], inplace=True, errors='ignore')
        
        # Extract year from filename
        df_all['rok'] = (
            df_all['Source_File']
            .astype(str)
            .str.replace('.xlsx', '', regex=False)
            .astype(int)
        )
        df_all.drop(columns=['Source_File'], inplace=True)
        df_all.drop(columns=['Unnamed: 3'], inplace=True, errors='ignore')
        
        # Add prefix to indicator columns
        new_columns = {}
        for col in df_all.columns:
            if col not in ['PKD_2007', 'rok']:
                new_columns[col] = f"{wskaznik_prefix} {col}"
        df_all.rename(columns=new_columns, inplace=True)
        
        # Transpose to long format
        df_transposed = df_all.melt(
            id_vars=['PKD_2007', 'rok'], 
            var_name='WSKAZNIK', 
            value_name='wartosc'
        )
        
        # Map to PKD 2025
        df_mapped = self.pkd_mapper.map_pkd_2007_to_2025(
            df_transposed, 
            pkd_column="PKD_2007",
            include_og=True
        )
        
        result = df_mapped[["rok", "pkd_2025", "WSKAZNIK", "wartosc"]].copy()
        
        return result


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

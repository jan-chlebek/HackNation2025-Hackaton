"""
PKD Mapper Module
Handles PKD code mapping and normalization between 2007 and 2025 standards.
"""

import pandas as pd


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

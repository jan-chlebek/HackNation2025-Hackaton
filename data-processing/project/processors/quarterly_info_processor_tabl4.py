import os
import pandas as pd
from processors.data_processor import DataProcessor


class QuarterlyInfoProcessorTabl4(DataProcessor):

    def process(
        self, 
        folder_path: str, 
        sheet_name: str = 'Tabl 4',
        wskaznik_prefix: str = "Przewidywana liczba pracujących"
    ) -> pd.DataFrame:

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

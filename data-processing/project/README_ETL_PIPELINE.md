# ETL Pipeline - Scalable Architecture

## 📋 Overview

This is a refactored, scalable ETL pipeline for processing multiple KPI data sources into a unified fact table with dimension tables.

## 🏗️ Architecture

```
data-processing/
├── etl-pipeline-scalable.ipynb    # Main notebook - run this!
├── data_sources_config.py          # Configure data sources here
└── ../src/etl_pipeline.py          # Core pipeline logic
```

### Key Components

1. **ETL Pipeline Module** (`src/etl_pipeline.py`)
   - `PKDMapper`: Handles PKD code mapping and normalization
   - `DataProcessor`: Base class for all data processors
   - `UpadlosciProcessor`: Processes bankruptcy data
   - `WskaznikFinansowyProcessor`: Processes financial indicators
   - `QuarterlyInfoProcessor`: Processes quarterly employment data
   - `ETLPipeline`: Orchestrates the entire pipeline

2. **Configuration** (`data_sources_config.py`)
   - Lists all data sources to process
   - Easy to add new sources without changing core code

3. **Execution Notebook** (`etl-pipeline-scalable.ipynb`)
   - Clean, simple notebook that runs the pipeline
   - Shows data quality checks and results

## ✨ Benefits vs Original Code

| Feature | Original | New Pipeline |
|---------|----------|--------------|
| Adding new data source | Copy-paste 50+ lines of code | Add 5-line config entry |
| Code reusability | Each source has duplicate code | Shared functions, DRY |
| Maintainability | Hard to update common logic | Change once, apply everywhere |
| Testing | Must run entire notebook | Can unit test individual processors |
| Documentation | Scattered in notebook | Centralized in module docstrings |
| Error handling | Inconsistent | Standardized warnings/errors |

## 🚀 Adding a New KPI Data Source

### Option A: Using Existing Processor

If your new data source has a similar format to existing ones:

1. Open `data_sources_config.py`
2. Add a new dictionary to the `data_sources` list:

```python
{
    'processor_class': WskaznikFinansowyProcessor,  # Choose appropriate processor
    'name': 'My New KPI Source',
    'kwargs': {
        'file_path': os.path.join(base_path, 'new_kpi_data.csv')
    }
}
```

3. Run `etl-pipeline-scalable.ipynb`

**Done!** ✓

### Option B: Creating Custom Processor

If your data source has a unique format:

1. Open `src/etl_pipeline.py`
2. Create a new processor class:

```python
class MyCustomProcessor(DataProcessor):
    """Processor for my custom data format."""
    
    def process(self, file_path: str) -> pd.DataFrame:
        """
        Process custom data from file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame with columns: [rok, pkd_2025, WSKAZNIK, wartosc]
        """
        # 1. Load your data
        df = pd.read_csv(file_path, ...)
        
        # 2. Transform to standard format
        # Your custom logic here...
        
        # 3. Map PKD codes if needed
        df_mapped = self.pkd_mapper.map_pkd_2007_to_2025(df, ...)
        
        # 4. Return standardized format
        return df[["rok", "pkd_2025", "WSKAZNIK", "wartosc"]]
```

3. Import it in `data_sources_config.py`:
```python
from etl_pipeline import MyCustomProcessor
```

4. Add to config:
```python
{
    'processor_class': MyCustomProcessor,
    'name': 'My Custom Data',
    'kwargs': {
        'file_path': os.path.join(base_path, 'custom_data.csv')
    }
}
```

5. Run `etl-pipeline-scalable.ipynb`

**Done!** ✓

## 📊 Output Schema

### Fact Table (`kpi-value-table.csv`)
```
rok              : int        # Year
PKD_INDEX        : int        # FK to pkd_dictionary
WSKAZNIK_INDEX   : int        # FK to wskaznik_dictionary
wartosc          : Decimal    # KPI value (nullable)
```

### WSKAZNIK Dictionary (`wskaznik_dictionary.csv`)
```
WSKAZNIK_INDEX   : int        # PK
WSKAZNIK         : str        # Indicator name
```

### PKD Dictionary (`pkd_dictionary.csv`)
```
PKD_INDEX        : int        # PK
symbol           : str        # PKD code (e.g., "01", "A", "01.11.Z")
nazwa            : str        # PKD name/description
TYP_INDEX        : int        # FK to pkd_typ_dictionary
```

### PKD Type Dictionary (`pkd_typ_dictionary.csv`)
```
TYP_INDEX        : int        # PK
typ              : str        # Type name (Sekcja, Dział, etc.)
```

## 🔧 Common Tasks

### Task: Add a new CSV file with financial indicators

```python
# In data_sources_config.py, add:
{
    'processor_class': WskaznikFinansowyProcessor,
    'name': 'New Financial Indicators',
    'kwargs': {
        'file_path': os.path.join(base_path, 'new_indicators.csv')
    }
}
```

### Task: Add quarterly data from a new folder

```python
# In data_sources_config.py, add:
{
    'processor_class': QuarterlyInfoProcessor,
    'name': 'New Quarterly Data',
    'kwargs': {
        'folder_path': os.path.join(base_path, 'external', 'New_Folder'),
        'sheet_name': 'Tabl 4',
        'wskaznik_prefix': 'Custom Prefix'
    }
}
```

### Task: Process data that's already in PKD 2025 format

Create a simple processor:

```python
class SimplePKD2025Processor(DataProcessor):
    def process(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, sep=";")
        # Already in correct format: [rok, pkd_2025, WSKAZNIK, wartosc]
        return df
```

## 📝 Example: Complete Workflow

```python
# 1. Initialize pipeline
from etl_pipeline import ETLPipeline
pipeline = ETLPipeline(mapping_file_path='../data/mapowanie_pkd.xlsx')

# 2. Add data sources from config
from data_sources_config import get_data_sources_config
for source in get_data_sources_config():
    processor = source['processor_class'](pipeline.pkd_mapper)
    pipeline.add_data_source(processor, source['name'], **source['kwargs'])

# 3. Combine and build dictionaries
combined = pipeline.combine_data()
fact, wskaznik, pkd, typ = pipeline.build_dictionaries(combined)

# 4. Save results
pipeline.save_results(fact, wskaznik, pkd, typ, output_dir='../results-pipeline')
```

## 🐛 Troubleshooting

### Error: "Unmapped PKD codes"
- Your data contains PKD codes not in the mapping file
- Check the warning message for which codes are missing
- Either add them to `mapowanie_pkd.xlsx` or fix the source data

### Error: "Sheet not found"
- For QuarterlyInfoProcessor, verify the sheet name exists
- Common variants: 'Tabl 4' vs 'Tabl. 4' (automatically handled)

### Error: "Invalid value encountered"
- The data contains non-numeric values in the value column
- These are automatically converted to None
- Check the warning message to see which values failed

## 📚 Additional Resources

- Original notebook: `data-processing.ipynb`
- New scalable notebook: `etl-pipeline-scalable.ipynb`
- Core module: `../src/etl_pipeline.py`
- Configuration: `data_sources_config.py`

## 🎯 Next Steps

1. Run `etl-pipeline-scalable.ipynb` to verify it works with existing data
2. Compare output with original results in `../results/`
3. Add your new data sources to `data_sources_config.py`
4. Enjoy the simplified workflow! 🎉

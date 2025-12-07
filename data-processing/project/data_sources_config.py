"""
Configuration for ETL data sources.
Add new data sources here to include them in the pipeline.
"""

import os

from processors import (
    UpadlosciProcessor,
    WskaznikFinansowyProcessor,
    QuarterlyInfoProcessorTabl4,
    QuarterlyInfoProcessorTabl5,
    QuarterlyInfoProcessorTabl7
)


def get_data_sources_config():
    """
    Define all data sources to be processed.
    
    Returns:
        List of dictionaries with data source configurations.
        Each dict should have:
        - processor_class: The processor class to use
        - name: Human-readable name for logging
        - kwargs: Arguments to pass to processor.process()
    """
    
    base_path = os.path.join('..', '..', 'data')
    
    data_sources = [
        {
            'processor_class': UpadlosciProcessor,
            'name': 'Upadłości (KRZ_PKD)',
            'kwargs': {
                'file_path': os.path.join(base_path, 'krz_pkd.csv')
            }
        },
        {
            'processor_class': WskaznikFinansowyProcessor,
            'name': 'Wskaźniki Finansowe',
            'kwargs': {
                'file_path': os.path.join(base_path, 'wsk_fin.csv')
            }
        },
        {
            'processor_class': QuarterlyInfoProcessorTabl4,
            'name': 'Dane Kwartalne - Pracujący',
            'kwargs': {
                'folder_path': os.path.join(
                    base_path, 
                    'external', 
                    '_Full Kwartalna informacja o podmiotach gospodarki narodowej w rejestrze REGON'
                ),
                'sheet_name': 'Tabl 4',
                'wskaznik_prefix': 'Przewidywana liczba pracujących'
            }
        },
        {
            'processor_class': QuarterlyInfoProcessorTabl5,
            'name': 'Dane Kwartalne - liczba firm vs działalności gospodarczych',
            'kwargs': {
                'folder_path': os.path.join(
                    base_path, 
                    'external', 
                    '_Full Kwartalna informacja o podmiotach gospodarki narodowej w rejestrze REGON'
                ),
                'sheet_name': 'Tabl 5'
            }
        },
        {
            'processor_class': QuarterlyInfoProcessorTabl7,
            'name': 'Dane Kwartalne - liczba firm vs działalności gospodarczych',
            'kwargs': {
                'folder_path': os.path.join(
                    base_path, 
                    'external', 
                    '_Full Kwartalna informacja o podmiotach gospodarki narodowej w rejestrze REGON'
                ),
                'sheet_name': 'Tabl 7',
                'wskaznik_prefix': 'Liczba nowych firm'
            }
        }
        # =============================================================
        # ADD NEW DATA SOURCES BELOW THIS LINE
        # =============================================================
        # 
        # Example: Add another CSV file with custom processor
        # {
        #     'processor_class': CustomProcessor,  # Create new processor class
        #     'name': 'New KPI Source',
        #     'kwargs': {
        #         'file_path': os.path.join(base_path, 'new_kpi_file.csv')
        #     }
        # },
        #
        # Example: Add another quarterly-style data source
        # {
        #     'processor_class': QuarterlyInfoProcessor,
        #     'name': 'Inne Dane Kwartalne',
        #     'kwargs': {
        #         'folder_path': os.path.join(base_path, 'external', 'Other_Folder'),
        #         'sheet_name': 'Data Sheet',
        #         'wskaznik_prefix': 'Custom Indicator'
        #     }
        # },
        
    ]
    
    return data_sources


# =================================================================
# TO ADD A NEW CUSTOM DATA SOURCE:
# =================================================================
# 1. Create a new processor class in processors/ folder that inherits from DataProcessor
# 2. Implement the process() method that returns DataFrame with columns:
#    [rok, pkd_2025, WSKAZNIK, wartosc]
# 3. Import the processor class at the top of this file
# 4. Add a new dictionary to the data_sources list above
# =================================================================

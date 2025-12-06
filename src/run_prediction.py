"""
Time Series Forecasting for Financial Indicators.

This script loads data from kpi-value-table.csv, trains on specified historical years,
and generates predictions in the same format.

Run from project root: python src/run_prediction.py
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from prediction import (
    ForecastConfig, 
    CategoryPredictor, 
    LinearTrendForecaster,
    ExponentialSmoothingForecaster,
    SimpleARIMAForecaster,
    MonteCarloForecaster,
    EnsembleForecaster
)


def load_kpi_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Load KPI data from results-pipeline folder.
    
    Args:
        start_year: Starting year for historical data
        end_year: Ending year for historical data
        
    Returns:
        DataFrame filtered by year range
    """
    input_dir = 'results-pipeline'
    
    print(f"Loading data from {input_dir}/kpi-value-table.csv...")
    kpi_data = pd.read_csv(os.path.join(input_dir, 'kpi-value-table.csv'), sep=';')
    
    print(f"  Total records: {len(kpi_data)}")
    print(f"  Year range in data: {kpi_data['rok'].min()} - {kpi_data['rok'].max()}")
    
    # Filter by year range
    filtered_data = kpi_data[(kpi_data['rok'] >= start_year) & (kpi_data['rok'] <= end_year)].copy()
    
    print(f"  Filtered to years {start_year}-{end_year}: {len(filtered_data)} records")
    print(f"  Unique PKD_INDEX: {filtered_data['PKD_INDEX'].nunique()}")
    print(f"  Unique WSKAZNIK_INDEX: {filtered_data['WSKAZNIK_INDEX'].nunique()}")
    
    return filtered_data


def prepare_prediction_input(kpi_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert kpi-value-table format to prediction input format.
    
    Args:
        kpi_data: DataFrame with columns [rok, wartosc, WSKAZNIK_INDEX, PKD_INDEX]
        
    Returns:
        DataFrame with columns [kpi_id, cat_id, value, direction, year]
    """
    print("\nPreparing data for prediction...")
    
    # Create prediction-ready format
    # PKD_INDEX as kpi_id (alternatives), WSKAZNIK_INDEX as cat_id (criteria)
    prediction_df = pd.DataFrame({
        'kpi_id': kpi_data['WSKAZNIK_INDEX'].astype(str) + '_' + kpi_data['PKD_INDEX'].astype(str),
        'cat_id': kpi_data['WSKAZNIK_INDEX'].astype(str),
        'value': kpi_data['wartosc'],
        'direction': 'max',  # Default direction
        'year': kpi_data['rok'],
        'PKD_INDEX': kpi_data['PKD_INDEX'],
        'WSKAZNIK_INDEX': kpi_data['WSKAZNIK_INDEX']
    })
    
    # Remove missing values
    initial_rows = len(prediction_df)
    prediction_df = prediction_df.dropna(subset=['value'])
    removed = initial_rows - len(prediction_df)
    
    if removed > 0:
        print(f"  Removed {removed} rows with missing values")
    
    # Remove infinite values
    initial_rows = len(prediction_df)
    prediction_df = prediction_df[~np.isinf(prediction_df['value'])]
    removed = initial_rows - len(prediction_df)
    
    if removed > 0:
        print(f"  Removed {removed} rows with infinite values")
    
    print(f"  Final dataset: {len(prediction_df)} rows")
    print(f"  Unique combinations: {prediction_df['kpi_id'].nunique()}")
    
    return prediction_df


def convert_predictions_to_kpi_format(predictions: pd.DataFrame, 
                                     original_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prediction output back to kpi-value-table format.
    
    Args:
        predictions: Prediction results
        original_data: Original kpi_data with PKD_INDEX and WSKAZNIK_INDEX
        
    Returns:
        DataFrame in kpi-value-table format [rok, wartosc, WSKAZNIK_INDEX, PKD_INDEX]
    """
    print("\nConverting predictions to kpi-value-table format...")
    
    # Extract PKD_INDEX and WSKAZNIK_INDEX from kpi_id and cat_id
    predictions_copy = predictions.copy()
    
    # Parse kpi_id to get WSKAZNIK_INDEX and PKD_INDEX
    predictions_copy[['WSKAZNIK_INDEX', 'PKD_INDEX']] = predictions_copy['kpi_id'].str.split('_', expand=True)
    predictions_copy['WSKAZNIK_INDEX'] = predictions_copy['WSKAZNIK_INDEX'].astype(int)
    predictions_copy['PKD_INDEX'] = predictions_copy['PKD_INDEX'].astype(float)
    
    # Create output in same format as input
    output_df = pd.DataFrame({
        'rok': predictions_copy['year'].astype(int),
        'wartosc': predictions_copy['ensemble_forecast'],
        'WSKAZNIK_INDEX': predictions_copy['WSKAZNIK_INDEX'],
        'PKD_INDEX': predictions_copy['PKD_INDEX']
    })
    
    # Sort to match original format
    output_df = output_df.sort_values(['rok', 'PKD_INDEX', 'WSKAZNIK_INDEX']).reset_index(drop=True)
    
    print(f"  Generated {len(output_df)} prediction records")
    print(f"  Years: {sorted(output_df['rok'].unique())}")
    
    return output_df


def run_predictions(start_year: int = 2018,
                   end_year: int = 2024,
                   forecast_years: int = 3,
                   mc_simulations: int = 1000,
                   output_dir: str = 'results-future') -> pd.DataFrame:
    """
    Run complete prediction pipeline.
    
    Args:
        start_year: Start of historical data range
        end_year: End of historical data range
        forecast_years: Number of years to forecast into the future
        mc_simulations: Number of Monte Carlo simulations
        output_dir: Directory to save results
        
    Returns:
        DataFrame with predictions in kpi-value-table format
    """
    print("="*90)
    print("FINANCIAL INDICATOR FORECASTING")
    print("="*90)
    print(f"\nConfiguration:")
    print(f"  Historical data: {start_year} - {end_year} ({end_year - start_year + 1} years)")
    print(f"  Forecast horizon: {forecast_years} years ({end_year + 1} - {end_year + forecast_years})")
    print(f"  Monte Carlo simulations: {mc_simulations}")
    print(f"  Output directory: {output_dir}/")
    
    # Step 1: Load historical data
    kpi_data = load_kpi_data(start_year, end_year)
    
    # Step 2: Prepare for prediction
    prediction_input = prepare_prediction_input(kpi_data)
    
    # Step 3: Configure forecasting
    config = ForecastConfig(
        forecast_years=forecast_years,
        min_historical_years=3,
        mc_simulations=mc_simulations,
        mc_noise_std=0.10
    )
    
    print("\n" + "="*90)
    print("RUNNING FORECASTING MODELS")
    print("="*90)
    
    # Step 4: Run predictions
    predictor = CategoryPredictor(prediction_input, config)
    forecasts = predictor.predict_all_categories()
    
    # Step 5: Convert back to kpi-value-table format
    predictions_kpi_format = convert_predictions_to_kpi_format(forecasts, kpi_data)
    
    # Step 6: Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 7: Save results
    output_file = output_path / 'kpi-value-table-predicted.csv'
    predictions_kpi_format.to_csv(output_file, sep=';', index=False)
    
    print("\n" + "="*90)
    print("SAVING RESULTS")
    print("="*90)
    print(f"\n✓ Predictions saved to: {output_file}")
    
    # Also save detailed forecasts with all methods
    detailed_file = output_path / 'detailed-forecasts.csv'
    forecasts.to_csv(detailed_file, index=False)
    print(f"✓ Detailed forecasts saved to: {detailed_file}")
    
    # Save summary statistics
    summary = predictor.get_forecast_summary(forecasts)
    summary_file = output_path / 'forecast-summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"✓ Summary statistics saved to: {summary_file}")
    
    # Print statistics
    print("\n" + "="*90)
    print("FORECAST STATISTICS")
    print("="*90)
    
    future_years = sorted(predictions_kpi_format['rok'].unique())
    print(f"\nFuture years predicted: {future_years}")
    
    for year in future_years:
        year_data = predictions_kpi_format[predictions_kpi_format['rok'] == year]
        print(f"\nYear {year}:")
        print(f"  Records: {len(year_data)}")
        print(f"  PKD sectors: {year_data['PKD_INDEX'].nunique()}")
        print(f"  Indicators: {year_data['WSKAZNIK_INDEX'].nunique()}")
        print(f"  Value range: {year_data['wartosc'].min():.2f} to {year_data['wartosc'].max():.2f}")
        print(f"  Mean value: {year_data['wartosc'].mean():.2f}")
    
    # Show sample of predictions
    print("\n" + "="*90)
    print("SAMPLE PREDICTIONS (first 20 rows)")
    print("="*90)
    print(predictions_kpi_format.head(20).to_string(index=False))
    
    return predictions_kpi_format


if __name__ == '__main__':
    # Configuration - EDIT THESE PARAMETERS
    START_YEAR = 2015      # First year of historical data to use
    END_YEAR = 2022        # Last year of historical data to use
    FORECAST_YEARS = 3     # Number of years to predict into the future
    MC_SIMULATIONS = 1000  # Number of Monte Carlo simulations
    OUTPUT_DIR = 'results-future'  # Output directory
    
    try:
        # Run predictions
        predictions = run_predictions(
            start_year=START_YEAR,
            end_year=END_YEAR,
            forecast_years=FORECAST_YEARS,
            mc_simulations=MC_SIMULATIONS,
            output_dir=OUTPUT_DIR
        )
        
        print("\n" + "="*90)
        print("FORECASTING COMPLETE!")
        print("="*90)
        print(f"\nOutput files in {OUTPUT_DIR}/:")
        print("  • kpi-value-table-predicted.csv - Predictions in original format")
        print("  • detailed-forecasts.csv - Detailed forecasts with all methods")
        print("  • forecast-summary.csv - Summary statistics")
        print(f"\nTo use predictions:")
        print(f"  1. Original data is in: results-pipeline/kpi-value-table.csv")
        print(f"  2. Predictions are in: {OUTPUT_DIR}/kpi-value-table-predicted.csv")
        print(f"  3. You can combine them for complete time series analysis")
        
    except Exception as e:
        print(f"\n✗ Error during forecasting: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
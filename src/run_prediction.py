"""
Advanced Financial Indicator Forecasting with Multi-Model Ensemble.

Features:
- Multiple forecasting models with ensemble approach
- Cross-indicator correlations for improved accuracy
- Multiprocessing for faster execution
- Complete coverage of all PKD×WSKAZNIK combinations

Run from project root: python src/run_prediction.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Configuration
START_YEAR = 2012
END_YEAR = 2024
FORECAST_YEARS = 4
MAX_WORKERS = mp.cpu_count() - 1  # Leave one CPU free

print(f"Using {MAX_WORKERS} CPU cores for parallel processing")


def parse_polish_number(value):
    """Parse Polish number format (comma decimal, non-breaking space thousand separator)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    
    value_str = str(value).replace('\xa0', '').replace(' ', '').replace(',', '.')
    try:
        return float(value_str)
    except:
        return np.nan


def load_and_prepare_data():
    """Load and prepare data for forecasting."""
    print("Loading data from results-pipeline/kpi-value-table.csv...")
    
    df = pd.read_csv('results-pipeline/kpi-value-table.csv', sep=';')
    
    # Parse Polish numbers
    df['wartosc'] = df['wartosc'].apply(parse_polish_number)
    
    # Filter by year range
    df = df[(df['rok'] >= START_YEAR) & (df['rok'] <= END_YEAR)].copy()
    
    print(f"  Loaded {len(df)} records from {START_YEAR} to {END_YEAR}")
    print(f"  Unique indicators: {df['WSKAZNIK_INDEX'].nunique()}")
    print(f"  Unique PKD categories: {df['PKD_INDEX'].nunique()}")
    
    return df


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix between indicators."""
    print("\nCalculating cross-indicator correlations...")
    
    # Pivot to get indicators as columns
    pivot = df.pivot_table(
        index=['rok', 'PKD_INDEX'],
        columns='WSKAZNIK_INDEX',
        values='wartosc',
        aggfunc='mean'
    )
    
    # Calculate correlation matrix
    corr_matrix = pivot.corr()
    
    print(f"  Calculated correlations for {len(corr_matrix)} indicators")
    
    return corr_matrix


def get_correlated_features(wskaznik_idx: int, corr_matrix: pd.DataFrame, 
                            threshold: float = 0.7, max_features: int = 5) -> List[int]:
    """Get top correlated indicators for a given indicator."""
    if wskaznik_idx not in corr_matrix.columns:
        return []
    
    correlations = corr_matrix[wskaznik_idx].abs().sort_values(ascending=False)
    
    # Exclude self-correlation and low correlations
    correlations = correlations[correlations.index != wskaznik_idx]
    correlations = correlations[correlations >= threshold]
    
    return list(correlations.head(max_features).index)


def simple_exponential_smoothing(series: pd.Series, alpha: float = 0.3) -> float:
    """Simple exponential smoothing for next value."""
    if len(series) == 0:
        return 0.0
    if len(series) == 1:
        return series.iloc[0]
    
    # Initialize with first value
    smoothed = series.iloc[0]
    
    # Apply exponential smoothing
    for value in series.iloc[1:]:
        if pd.notna(value):
            smoothed = alpha * value + (1 - alpha) * smoothed
    
    # Forecast next value
    return smoothed


def linear_trend_forecast(series: pd.Series) -> float:
    """Forecast next value using linear regression."""
    if len(series) < 2:
        return series.iloc[-1] if len(series) > 0 else 0.0
    
    # Remove NaN values
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return series.iloc[-1] if len(series) > 0 else 0.0
    
    # Create time index
    x = np.arange(len(clean_series))
    y = clean_series.values
    
    # Calculate linear regression coefficients
    try:
        slope, intercept = np.polyfit(x, y, 1)
        # Forecast for next time step
        next_value = slope * len(clean_series) + intercept
        return next_value
    except:
        return clean_series.iloc[-1]


def weighted_moving_average(series: pd.Series, window: int = 3) -> float:
    """Weighted moving average forecast."""
    if len(series) == 0:
        return 0.0
    
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0.0
    
    # Use last N values
    recent_values = clean_series.iloc[-min(window, len(clean_series)):]
    
    if len(recent_values) == 0:
        return 0.0
    
    # Create linearly decreasing weights (more recent = higher weight)
    n = len(recent_values)
    weights = np.arange(1, n + 1)
    weights = weights / weights.sum()
    
    return np.sum(recent_values.values * weights)


def seasonal_naive_forecast(series: pd.Series, season_length: int = 1) -> float:
    """Seasonal naive forecast (use value from same period last year)."""
    if len(series) == 0:
        return 0.0
    
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0.0
    
    # If we have data from same season last year, use it
    if len(clean_series) >= season_length:
        return clean_series.iloc[-season_length]
    else:
        return clean_series.iloc[-1]


def correlation_based_forecast(pkd_idx: float, wskaznik_idx: int, 
                               correlated_wskazniki: List[int],
                               df: pd.DataFrame) -> float:
    """Forecast based on correlated indicators."""
    if not correlated_wskazniki:
        return None
    
    # Get most recent values of correlated indicators for same PKD
    recent_values = []
    
    for corr_wskaznik in correlated_wskazniki:
        mask = (df['PKD_INDEX'] == pkd_idx) & (df['WSKAZNIK_INDEX'] == corr_wskaznik)
        corr_series = df[mask].sort_values('rok')['wartosc']
        
        if len(corr_series) > 0:
            # Get growth rate of correlated indicator
            if len(corr_series) >= 2:
                growth_rate = (corr_series.iloc[-1] - corr_series.iloc[-2]) / (abs(corr_series.iloc[-2]) + 1e-10)
                recent_values.append(growth_rate)
    
    if not recent_values:
        return None
    
    # Average growth rate from correlated indicators
    avg_growth = np.mean(recent_values)
    
    # Apply to current indicator
    mask = (df['PKD_INDEX'] == pkd_idx) & (df['WSKAZNIK_INDEX'] == wskaznik_idx)
    current_series = df[mask].sort_values('rok')['wartosc']
    
    if len(current_series) > 0:
        last_value = current_series.iloc[-1]
        return last_value * (1 + avg_growth)
    
    return None


def ensemble_forecast(series: pd.Series, pkd_idx: float, wskaznik_idx: int,
                     corr_matrix: pd.DataFrame, df_full: pd.DataFrame) -> float:
    """
    Ensemble forecast combining multiple methods.
    
    Weights:
    - Exponential Smoothing: 25%
    - Linear Trend: 25%
    - Weighted MA: 30%
    - Seasonal Naive: 10%
    - Correlation-based: 10%
    """
    forecasts = []
    weights = []
    
    # 1. Exponential Smoothing
    es_forecast = simple_exponential_smoothing(series)
    if pd.notna(es_forecast):
        forecasts.append(es_forecast)
        weights.append(0.25)
    
    # 2. Linear Trend
    lt_forecast = linear_trend_forecast(series)
    if pd.notna(lt_forecast):
        forecasts.append(lt_forecast)
        weights.append(0.25)
    
    # 3. Weighted Moving Average
    wma_forecast = weighted_moving_average(series)
    if pd.notna(wma_forecast):
        forecasts.append(wma_forecast)
        weights.append(0.30)
    
    # 4. Seasonal Naive
    sn_forecast = seasonal_naive_forecast(series)
    if pd.notna(sn_forecast):
        forecasts.append(sn_forecast)
        weights.append(0.10)
    
    # 5. Correlation-based
    correlated = get_correlated_features(wskaznik_idx, corr_matrix)
    corr_forecast = correlation_based_forecast(pkd_idx, wskaznik_idx, correlated, df_full)
    if corr_forecast is not None and pd.notna(corr_forecast):
        forecasts.append(corr_forecast)
        weights.append(0.10)
    
    # If no forecasts available, use last value
    if len(forecasts) == 0:
        return series.iloc[-1] if len(series) > 0 else 0.0
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted average
    ensemble_value = np.average(forecasts, weights=weights)
    
    # Bounds check: clip to reasonable range (mean ± 3 std devs)
    if len(series) >= 3:
        series_mean = series.mean()
        series_std = series.std()
        lower_bound = series_mean - 3 * series_std
        upper_bound = series_mean + 3 * series_std
        ensemble_value = np.clip(ensemble_value, lower_bound, upper_bound)
    
    return ensemble_value


def forecast_single_category(args: Tuple) -> List[Dict]:
    """
    Forecast a single PKD×WSKAZNIK combination.
    
    Args:
        args: Tuple of (pkd_idx, wskaznik_idx, series, corr_matrix, df_full)
    
    Returns:
        List of forecast dictionaries
    """
    pkd_idx, wskaznik_idx, series, corr_matrix, df_full = args
    
    results = []
    
    try:
        # Create extended series for iterative forecasting
        forecast_series = series.copy()
        
        for year_ahead in range(1, FORECAST_YEARS + 1):
            forecast_year = END_YEAR + year_ahead
            
            # Generate forecast using ensemble
            forecast_value = ensemble_forecast(
                forecast_series, pkd_idx, wskaznik_idx, corr_matrix, df_full
            )
            
            # Add to results
            results.append({
                'rok': forecast_year,
                'wartosc': forecast_value,
                'WSKAZNIK_INDEX': wskaznik_idx,
                'PKD_INDEX': pkd_idx
            })
            
            # Add to series for next iteration
            new_point = pd.Series([forecast_value], index=[forecast_year])
            forecast_series = pd.concat([forecast_series, new_point])
    
    except Exception as e:
        # If forecasting fails, use last known value
        last_value = series.iloc[-1] if len(series) > 0 else 0.0
        
        for year_ahead in range(1, FORECAST_YEARS + 1):
            forecast_year = END_YEAR + year_ahead
            
            results.append({
                'rok': forecast_year,
                'wartosc': last_value,
                'WSKAZNIK_INDEX': wskaznik_idx,
                'PKD_INDEX': pkd_idx
            })
    
    return results


def forecast_all_categories(df: pd.DataFrame, corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast all PKD categories using multiprocessing.
    
    Args:
        df: Historical data
        corr_matrix: Correlation matrix between indicators
    
    Returns:
        DataFrame with forecasts for ALL PKD×WSKAZNIK combinations
    """
    print(f"\nForecasting {FORECAST_YEARS} years into the future...")
    print(f"  Years to forecast: {END_YEAR + 1} to {END_YEAR + FORECAST_YEARS}")
    
    # Get all unique combinations from historical data
    all_pkd = df['PKD_INDEX'].unique()
    all_wskaznik = df['WSKAZNIK_INDEX'].unique()
    
    print(f"  Total PKD categories: {len(all_pkd)}")
    print(f"  Total indicators: {len(all_wskaznik)}")
    print(f"  Expected combinations: {len(all_pkd) * len(all_wskaznik)}")
    
    # Prepare tasks - one for each existing combination
    tasks = []
    existing_combinations = set()
    
    for (pkd_idx, wskaznik_idx), group in df.groupby(['PKD_INDEX', 'WSKAZNIK_INDEX']):
        series = group.sort_values('rok')['wartosc']
        
        if len(series) >= 2:  # Need at least 2 points for forecasting
            tasks.append((pkd_idx, wskaznik_idx, series, corr_matrix, df))
            existing_combinations.add((pkd_idx, wskaznik_idx))
    
    print(f"  Forecasting tasks (with historical data): {len(tasks)}")
    
    # Run forecasting in parallel
    all_forecasts = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(forecast_single_category, task): task for task in tasks}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            try:
                forecast_results = future.result()
                all_forecasts.extend(forecast_results)
                
                completed += 1
                if completed % 100 == 0:
                    print(f"  Progress: {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%)")
            
            except Exception as e:
                print(f"  Error in task: {e}")
    
    print(f"  Completed: {completed}/{len(tasks)} tasks")
    
    # Create DataFrame from forecasts
    forecast_df = pd.DataFrame(all_forecasts)
    
    # Fill missing combinations with median values per indicator
    print("\nFilling missing combinations...")
    
    all_forecast_data = []
    
    for year in range(END_YEAR + 1, END_YEAR + FORECAST_YEARS + 1):
        for pkd_idx in all_pkd:
            for wskaznik_idx in all_wskaznik:
                # Check if this combination exists in forecasts
                existing = forecast_df[
                    (forecast_df['rok'] == year) &
                    (forecast_df['PKD_INDEX'] == pkd_idx) &
                    (forecast_df['WSKAZNIK_INDEX'] == wskaznik_idx)
                ]
                
                if len(existing) > 0:
                    # Use existing forecast
                    all_forecast_data.append({
                        'rok': year,
                        'wartosc': existing.iloc[0]['wartosc'],
                        'WSKAZNIK_INDEX': wskaznik_idx,
                        'PKD_INDEX': pkd_idx
                    })
                else:
                    # Fill with median value for this indicator across all PKDs
                    median_value = forecast_df[
                        (forecast_df['rok'] == year) &
                        (forecast_df['WSKAZNIK_INDEX'] == wskaznik_idx)
                    ]['wartosc'].median()
                    
                    if pd.isna(median_value):
                        # If still no data, use global median for this indicator from historical data
                        median_value = df[df['WSKAZNIK_INDEX'] == wskaznik_idx]['wartosc'].median()
                    
                    if pd.isna(median_value):
                        median_value = 0.0
                    
                    all_forecast_data.append({
                        'rok': year,
                        'wartosc': median_value,
                        'WSKAZNIK_INDEX': wskaznik_idx,
                        'PKD_INDEX': pkd_idx
                    })
    
    complete_forecast_df = pd.DataFrame(all_forecast_data)
    
    print(f"  Complete forecasts: {len(complete_forecast_df)} records")
    print(f"  Coverage: {len(complete_forecast_df) / (len(all_pkd) * len(all_wskaznik) * FORECAST_YEARS) * 100:.1f}%")
    
    return complete_forecast_df


def save_predictions(forecast_df: pd.DataFrame):
    """Save predictions in the EXACT same format as input data."""
    output_dir = Path('results-future')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'kpi-value-table-predicted.csv'
    
    # CRITICAL: Match input column order exactly
    # Input has: rok;wartosc;WSKAZNIK_INDEX;PKD_INDEX
    forecast_df = forecast_df[['rok', 'wartosc', 'WSKAZNIK_INDEX', 'PKD_INDEX']]
    
    # Sort by year, then WSKAZNIK_INDEX, then PKD_INDEX (to match input pattern)
    forecast_df = forecast_df.sort_values(['rok', 'WSKAZNIK_INDEX', 'PKD_INDEX'])
    
    # Save with semicolon separator (same as input)
    forecast_df.to_csv(output_file, sep=';', index=False)
    
    print(f"\n✓ Predictions saved to: {output_file}")
    print(f"  Total predictions: {len(forecast_df)}")
    print(f"  Years: {forecast_df['rok'].min()} to {forecast_df['rok'].max()}")
    print(f"  Format: rok;wartosc;WSKAZNIK_INDEX;PKD_INDEX (matches input)")
    
    # Save summary statistics
    summary_file = output_dir / 'forecast-summary.txt'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("FORECAST SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Forecast period: {END_YEAR + 1} to {END_YEAR + FORECAST_YEARS}\n")
        f.write(f"Total predictions: {len(forecast_df)}\n\n")
        
        f.write("Predictions per year:\n")
        for year in sorted(forecast_df['rok'].unique()):
            count = len(forecast_df[forecast_df['rok'] == year])
            f.write(f"  {year}: {count} predictions\n")
        
        f.write(f"\nUnique indicators: {forecast_df['WSKAZNIK_INDEX'].nunique()}\n")
        f.write(f"Unique PKD categories: {forecast_df['PKD_INDEX'].nunique()}\n")
        
        f.write("\nValue statistics:\n")
        f.write(f"  Mean: {forecast_df['wartosc'].mean():.2f}\n")
        f.write(f"  Median: {forecast_df['wartosc'].median():.2f}\n")
        f.write(f"  Std Dev: {forecast_df['wartosc'].std():.2f}\n")
        f.write(f"  Min: {forecast_df['wartosc'].min():.2f}\n")
        f.write(f"  Max: {forecast_df['wartosc'].max():.2f}\n")
        
        # Check for missing values
        missing_count = forecast_df['wartosc'].isna().sum()
        f.write(f"\nMissing values: {missing_count}\n")
    
    print(f"✓ Summary saved to: {summary_file}")


if __name__ == '__main__':
    print("="*80)
    print("ADVANCED FINANCIAL FORECASTING SYSTEM")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Historical data: {START_YEAR}-{END_YEAR}")
    print(f"  Forecast horizon: {FORECAST_YEARS} years")
    print(f"  CPU cores: {MAX_WORKERS}")
    print(f"  Methods: Exponential Smoothing, Linear Trend, WMA, Seasonal Naive, Correlation-based")
    
    # Load data
    df = load_and_prepare_data()
    
    # Calculate correlations
    corr_matrix = calculate_correlations(df)
    
    # Forecast ALL combinations
    forecasts = forecast_all_categories(df, corr_matrix)
    
    # Save results
    save_predictions(forecasts)
    
    print("\n" + "="*80)
    print("FORECASTING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review predictions in: results-future/kpi-value-table-predicted.csv")
    print("  2. Run analysis: python src/outcome.py results-future")
    print("  3. Compare with historical: python src/checking_predictions.py")
"""
Evaluate prediction accuracy by comparing predicted values with actual values.

This script compares predictions in results-future/ with actual values in results-pipeline/
for overlapping years, using normalized metrics for better comparison.

Run from project root: python src/evaluate_predictions.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_log_error


def load_data():
    """Load actual and predicted data."""
    print("Loading data...")
    
    # Load actual data
    actual = pd.read_csv('results-pipeline/kpi-value-table.csv', sep=';')
    print(f"  Actual data: {len(actual)} records, years {actual['rok'].min()}-{actual['rok'].max()}")
    
    # Load predictions
    predicted = pd.read_csv('results-future/kpi-value-table-predicted.csv', sep=';')
    print(f"  Predicted data: {len(predicted)} records, years {predicted['rok'].min()}-{predicted['rok'].max()}")
    
    return actual, predicted


def clean_comparison_data(comparison: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid values from comparison data."""
    initial_len = len(comparison)
    
    # Remove rows with NaN or infinite values in either actual or predicted
    comparison = comparison[
        np.isfinite(comparison['wartosc_actual']) & 
        np.isfinite(comparison['wartosc_predicted'])
    ]
    
    removed = initial_len - len(comparison)
    if removed > 0:
        print(f"  Removed {removed} records with NaN or infinite values")
    
    return comparison


def calculate_metrics(actual_values, predicted_values):
    """Calculate various accuracy metrics with safety checks."""
    # Ensure no infinite values
    mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[mask]
    predicted_values = predicted_values[mask]
    
    if len(actual_values) == 0:
        return {key: np.nan for key in [
            'MAE', 'RMSE', 'NRMSE (range)', 'NRMSE (mean)', 
            'MAPE (%)', 'sMAPE (%)', 'R²', 'MASE', 'MedAE', 'MSLE', 'RMSLE'
        ]}
    
    metrics = {}
    
    # Basic error metrics
    error = predicted_values - actual_values
    abs_error = np.abs(error)
    
    # Mean Absolute Error
    metrics['MAE'] = np.mean(abs_error)
    
    # Root Mean Squared Error
    metrics['RMSE'] = np.sqrt(np.mean(error ** 2))
    
    # Normalized RMSE (by range of actual values)
    value_range = np.max(actual_values) - np.min(actual_values)
    if value_range > 0 and np.isfinite(value_range):
        metrics['NRMSE (range)'] = metrics['RMSE'] / value_range
    else:
        metrics['NRMSE (range)'] = np.nan
    
    # Normalized RMSE (by mean of actual values)
    mean_val = np.mean(np.abs(actual_values))
    if mean_val > 0 and np.isfinite(mean_val):
        metrics['NRMSE (mean)'] = metrics['RMSE'] / mean_val
    else:
        metrics['NRMSE (mean)'] = np.nan
    
    # Mean Absolute Percentage Error (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_error = (error / actual_values) * 100
        pct_error = pct_error[np.isfinite(pct_error)]
        if len(pct_error) > 0:
            metrics['MAPE (%)'] = np.mean(np.abs(pct_error))
        else:
            metrics['MAPE (%)'] = np.nan
    
    # Symmetric MAPE (handles zero values better)
    denominator = np.abs(actual_values) + np.abs(predicted_values)
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = 200 * abs_error / denominator
        smape = smape[np.isfinite(smape)]
        if len(smape) > 0:
            metrics['sMAPE (%)'] = np.mean(smape)
        else:
            metrics['sMAPE (%)'] = np.nan
    
    # R-squared (coefficient of determination)
    try:
        metrics['R²'] = r2_score(actual_values, predicted_values)
        if not np.isfinite(metrics['R²']):
            metrics['R²'] = np.nan
    except:
        metrics['R²'] = np.nan
    
    # Mean Absolute Scaled Error (MASE) - scaled by naive forecast
    try:
        naive_error = np.mean(np.abs(np.diff(actual_values)))
        if naive_error > 0 and np.isfinite(naive_error):
            metrics['MASE'] = metrics['MAE'] / naive_error
        else:
            metrics['MASE'] = np.nan
    except:
        metrics['MASE'] = np.nan
    
    # Median Absolute Error (more robust to outliers)
    metrics['MedAE'] = np.median(abs_error)
    
    # Mean Squared Logarithmic Error (for positive values only)
    if (actual_values > 0).all() and (predicted_values > 0).all():
        try:
            metrics['MSLE'] = mean_squared_log_error(actual_values, predicted_values)
            metrics['RMSLE'] = np.sqrt(metrics['MSLE'])
            if not np.isfinite(metrics['MSLE']):
                metrics['MSLE'] = np.nan
                metrics['RMSLE'] = np.nan
        except:
            metrics['MSLE'] = np.nan
            metrics['RMSLE'] = np.nan
    else:
        metrics['MSLE'] = np.nan
        metrics['RMSLE'] = np.nan
    
    return metrics


def evaluate_accuracy(actual: pd.DataFrame, predicted: pd.DataFrame):
    """Compare predictions with actual values."""
    
    # Find overlapping years
    actual_years = set(actual['rok'].unique())
    predicted_years = set(predicted['rok'].unique())
    overlap_years = sorted(actual_years & predicted_years)
    
    if not overlap_years:
        print("\n⚠ No overlapping years found between actual and predicted data!")
        print(f"  Actual years: {sorted(actual_years)}")
        print(f"  Predicted years: {sorted(predicted_years)}")
        return None
    
    print(f"\nOverlapping years for comparison: {overlap_years}")
    
    # Merge on year, PKD_INDEX, and WSKAZNIK_INDEX
    comparison = actual[actual['rok'].isin(overlap_years)].merge(
        predicted[predicted['rok'].isin(overlap_years)],
        on=['rok', 'PKD_INDEX', 'WSKAZNIK_INDEX'],
        suffixes=('_actual', '_predicted')
    )
    
    print(f"\nMatched {len(comparison)} records for comparison")
    
    if len(comparison) == 0:
        print("⚠ No matching records found!")
        return None
    
    # Clean data - remove infinite and NaN values
    comparison = clean_comparison_data(comparison)
    
    if len(comparison) == 0:
        print("⚠ No valid records after cleaning!")
        return None
    
    print(f"Valid records for analysis: {len(comparison)}")
    
    # Calculate error columns
    comparison['error'] = comparison['wartosc_predicted'] - comparison['wartosc_actual']
    comparison['abs_error'] = np.abs(comparison['error'])
    
    # Percentage error (with safety)
    with np.errstate(divide='ignore', invalid='ignore'):
        comparison['pct_error'] = (comparison['error'] / comparison['wartosc_actual']) * 100
        comparison['abs_pct_error'] = np.abs(comparison['pct_error'])
        
        # Symmetric absolute percentage error
        denominator = np.abs(comparison['wartosc_actual']) + np.abs(comparison['wartosc_predicted'])
        comparison['smape'] = 200 * comparison['abs_error'] / denominator
    
    # Replace infinite values with NaN
    comparison['pct_error'] = comparison['pct_error'].replace([np.inf, -np.inf], np.nan)
    comparison['abs_pct_error'] = comparison['abs_pct_error'].replace([np.inf, -np.inf], np.nan)
    comparison['smape'] = comparison['smape'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(
        comparison['wartosc_actual'].values,
        comparison['wartosc_predicted'].values
    )
    
    # Print overall metrics
    print("\n" + "="*80)
    print("OVERALL ACCURACY METRICS")
    print("="*80)
    print("\nAbsolute Error Metrics:")
    print(f"  Mean Absolute Error (MAE):              {overall_metrics['MAE']:.4f}")
    print(f"  Median Absolute Error (MedAE):          {overall_metrics['MedAE']:.4f}")
    print(f"  Root Mean Squared Error (RMSE):         {overall_metrics['RMSE']:.4f}")
    
    print("\nNormalized Error Metrics:")
    if np.isfinite(overall_metrics['NRMSE (range)']):
        print(f"  Normalized RMSE (by range):             {overall_metrics['NRMSE (range)']:.4f}")
    else:
        print(f"  Normalized RMSE (by range):             N/A")
    
    if np.isfinite(overall_metrics['NRMSE (mean)']):
        print(f"  Normalized RMSE (by mean):              {overall_metrics['NRMSE (mean)']:.4f}")
    else:
        print(f"  Normalized RMSE (by mean):              N/A")
    
    print("\nPercentage Error Metrics:")
    if np.isfinite(overall_metrics['MAPE (%)']):
        print(f"  Mean Absolute % Error (MAPE):           {overall_metrics['MAPE (%)']:.2f}%")
    else:
        print(f"  Mean Absolute % Error (MAPE):           N/A")
    
    if np.isfinite(overall_metrics['sMAPE (%)']):
        print(f"  Symmetric MAPE (sMAPE):                 {overall_metrics['sMAPE (%)']:.2f}%")
    else:
        print(f"  Symmetric MAPE (sMAPE):                 N/A")
    
    print("\nGoodness-of-Fit Metrics:")
    if np.isfinite(overall_metrics['R²']):
        print(f"  R-squared (R²):                         {overall_metrics['R²']:.4f}")
    else:
        print(f"  R-squared (R²):                         N/A")
    
    if np.isfinite(overall_metrics['MASE']):
        print(f"  Mean Absolute Scaled Error (MASE):      {overall_metrics['MASE']:.4f}")
    else:
        print(f"  Mean Absolute Scaled Error (MASE):      N/A")
    
    if not np.isnan(overall_metrics['RMSLE']) and np.isfinite(overall_metrics['RMSLE']):
        print("\nLogarithmic Metrics (for positive values):")
        print(f"  Root Mean Squared Log Error (RMSLE):    {overall_metrics['RMSLE']:.4f}")
    
    # Interpretation guide (only if we have valid metrics)
    if np.isfinite(overall_metrics['R²']) or np.isfinite(overall_metrics['sMAPE (%)']):
        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        
        if np.isfinite(overall_metrics['R²']):
            print("R² (closer to 1.0 is better):")
            if overall_metrics['R²'] >= 0.9:
                print("  ✓ Excellent fit (≥0.9)")
            elif overall_metrics['R²'] >= 0.7:
                print("  ✓ Good fit (0.7-0.9)")
            elif overall_metrics['R²'] >= 0.5:
                print("  ~ Moderate fit (0.5-0.7)")
            else:
                print("  ✗ Poor fit (<0.5)")
        
        if np.isfinite(overall_metrics['sMAPE (%)']):
            print("\nsMAPE (closer to 0% is better):")
            if overall_metrics['sMAPE (%)'] <= 10:
                print("  ✓ Excellent accuracy (≤10%)")
            elif overall_metrics['sMAPE (%)'] <= 20:
                print("  ✓ Good accuracy (10-20%)")
            elif overall_metrics['sMAPE (%)'] <= 30:
                print("  ~ Moderate accuracy (20-30%)")
            else:
                print("  ✗ Poor accuracy (>30%)")
    
    # Metrics by year
    print("\n" + "="*80)
    print("ACCURACY BY YEAR")
    print("="*80)
    
    year_metrics = []
    for year in overlap_years:
        year_data = comparison[comparison['rok'] == year]
        if len(year_data) > 0:
            year_m = calculate_metrics(
                year_data['wartosc_actual'].values,
                year_data['wartosc_predicted'].values
            )
            year_metrics.append({
                'Year': year,
                'Records': len(year_data),
                'MAE': year_m['MAE'] if np.isfinite(year_m['MAE']) else np.nan,
                'RMSE': year_m['RMSE'] if np.isfinite(year_m['RMSE']) else np.nan,
                'sMAPE (%)': year_m['sMAPE (%)'] if np.isfinite(year_m['sMAPE (%)']) else np.nan,
                'R²': year_m['R²'] if np.isfinite(year_m['R²']) else np.nan
            })
    
    if year_metrics:
        year_df = pd.DataFrame(year_metrics)
        print(year_df.to_string(index=False))
    
    # Top 10 worst and best predictions (using sMAPE where available)
    valid_smape = comparison.dropna(subset=['smape'])
    
    if len(valid_smape) > 0:
        print("\n" + "="*80)
        print("TOP 10 WORST PREDICTIONS (by sMAPE)")
        print("="*80)
        worst = valid_smape.nlargest(10, 'smape')[
            ['rok', 'PKD_INDEX', 'WSKAZNIK_INDEX', 'wartosc_actual', 'wartosc_predicted', 'smape']
        ]
        print(worst.to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP 10 BEST PREDICTIONS (by sMAPE)")
        print("="*80)
        best = valid_smape.nsmallest(10, 'smape')[
            ['rok', 'PKD_INDEX', 'WSKAZNIK_INDEX', 'wartosc_actual', 'wartosc_predicted', 'smape']
        ]
        print(best.to_string(index=False))
    
    # Save detailed comparison
    output_dir = Path('results-future')
    output_file = output_dir / 'prediction-accuracy.csv'
    comparison.to_csv(output_file, index=False)
    print(f"\n✓ Detailed comparison saved to: {output_file}")
    
    # Save metrics summary
    metrics_file = output_dir / 'accuracy-metrics.csv'
    metrics_df = pd.DataFrame([overall_metrics])
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Metrics summary saved to: {metrics_file}")
    
    return comparison, overall_metrics


if __name__ == '__main__':
    try:
        actual, predicted = load_data()
        results, metrics = evaluate_accuracy(actual, predicted)
        
        if results is not None:
            print("\n" + "="*80)
            print("EVALUATION COMPLETE")
            print("="*80)
            valid_r2 = np.isfinite(metrics['R²'])
            valid_smape = np.isfinite(metrics['sMAPE (%)'])
            
            if valid_r2 or valid_smape:
                print("\nKey Takeaway:")
                if valid_r2:
                    print(f"  Overall prediction accuracy (R²): {metrics['R²']:.4f}")
                if valid_smape:
                    print(f"  Average error (sMAPE): {metrics['sMAPE (%)']:.2f}%")
            else:
                print("\n⚠ Unable to calculate reliable accuracy metrics")
                print("  This may be due to data quality issues or extreme values")
        else:
            print("\n⚠ Evaluation could not be completed")
            
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have:")
        print("  1. results-pipeline/kpi-value-table.csv (actual data)")
        print("  2. results-future/kpi-value-table-predicted.csv (predictions)")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
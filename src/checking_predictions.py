"""
Evaluate prediction accuracy by comparing predicted values with actual values.

This script compares predictions in results-future/ with actual values in results-pipeline/
for overlapping years, using normalized metrics for better comparison.

Run from project root: python src/checking_predictions.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score


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


def clean_comparison_data(actual_values, predicted_values):
    """Remove invalid data points (NaN, Inf) from comparison."""
    # Convert to numpy arrays if they aren't already
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    # Create mask for valid data
    valid_mask = (
        np.isfinite(actual_values) & 
        np.isfinite(predicted_values) &
        (actual_values != 0) &  # Avoid division by zero
        (predicted_values != 0)
    )
    
    actual_clean = actual_values[valid_mask]
    predicted_clean = predicted_values[valid_mask]
    
    removed = len(actual_values) - len(actual_clean)
    if removed > 0:
        print(f"  ⚠ Removed {removed} invalid data points for metric calculation")
    
    return actual_clean, predicted_clean


def calculate_metrics(actual_values, predicted_values):
    """Calculate various accuracy metrics with robust error handling."""
    # Clean data first
    actual_clean, predicted_clean = clean_comparison_data(actual_values, predicted_values)
    
    if len(actual_clean) == 0:
        print("  ⚠ No valid data points for metric calculation!")
        return {
            'MAE': np.nan, 'RMSE': np.nan, 'NRMSE (range)': np.nan,
            'NRMSE (mean)': np.nan, 'MAPE (%)': np.nan, 'sMAPE (%)': np.nan,
            'R²': np.nan, 'MASE': np.nan, 'MedAE': np.nan
        }
    
    metrics = {}
    
    # Basic error metrics
    error = predicted_clean - actual_clean
    abs_error = np.abs(error)
    
    # Mean Absolute Error
    metrics['MAE'] = np.mean(abs_error)
    
    # Root Mean Squared Error
    metrics['RMSE'] = np.sqrt(np.mean(error ** 2))
    
    # Normalized RMSE (by range of actual values)
    value_range = actual_clean.max() - actual_clean.min()
    if value_range > 0:
        metrics['NRMSE (range)'] = metrics['RMSE'] / value_range
    else:
        metrics['NRMSE (range)'] = np.nan
    
    # Normalized RMSE (by mean of actual values)
    mean_actual = np.abs(np.mean(actual_clean))
    if mean_actual > 0:
        metrics['NRMSE (mean)'] = metrics['RMSE'] / mean_actual
    else:
        metrics['NRMSE (mean)'] = np.nan
    
    # Mean Absolute Percentage Error
    pct_error = (error / actual_clean) * 100
    metrics['MAPE (%)'] = np.mean(np.abs(pct_error))
    
    # Symmetric MAPE (handles zero values better)
    smape = 200 * abs_error / (np.abs(actual_clean) + np.abs(predicted_clean))
    metrics['sMAPE (%)'] = np.mean(smape)
    
    # R-squared (coefficient of determination)
    try:
        if len(actual_clean) > 1 and np.var(actual_clean) > 0:
            metrics['R²'] = r2_score(actual_clean, predicted_clean)
        else:
            metrics['R²'] = np.nan
    except:
        metrics['R²'] = np.nan
    
    # Mean Absolute Scaled Error (MASE) - scaled by naive forecast
    if len(actual_clean) > 1:
        naive_error = np.mean(np.abs(np.diff(actual_clean)))
        if naive_error > 0:
            metrics['MASE'] = metrics['MAE'] / naive_error
        else:
            metrics['MASE'] = np.nan
    else:
        metrics['MASE'] = np.nan
    
    # Median Absolute Error (more robust to outliers)
    metrics['MedAE'] = np.median(abs_error)
    
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
        return None, None
    
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
        return None, None
    
    # Check for data quality issues
    print(f"\nData quality check:")
    print(f"  Records with NaN actual: {comparison['wartosc_actual'].isna().sum()}")
    print(f"  Records with NaN predicted: {comparison['wartosc_predicted'].isna().sum()}")
    print(f"  Records with Inf actual: {np.isinf(comparison['wartosc_actual']).sum()}")
    print(f"  Records with Inf predicted: {np.isinf(comparison['wartosc_predicted']).sum()}")
    print(f"  Records with zero actual: {(comparison['wartosc_actual'] == 0).sum()}")
    print(f"  Records with zero predicted: {(comparison['wartosc_predicted'] == 0).sum()}")
    
    # Calculate error columns
    comparison['error'] = comparison['wartosc_predicted'] - comparison['wartosc_actual']
    comparison['abs_error'] = np.abs(comparison['error'])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        comparison['pct_error'] = (comparison['error'] / comparison['wartosc_actual']) * 100
        comparison['abs_pct_error'] = np.abs(comparison['pct_error'])
        
        # Symmetric absolute percentage error
        comparison['smape'] = 200 * comparison['abs_error'] / (
            np.abs(comparison['wartosc_actual']) + np.abs(comparison['wartosc_predicted'])
        )
    
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
    if not np.isnan(overall_metrics['NRMSE (range)']):
        print(f"  Normalized RMSE (by range):             {overall_metrics['NRMSE (range)']:.4f}")
    else:
        print(f"  Normalized RMSE (by range):             N/A (zero range)")
    
    if not np.isnan(overall_metrics['NRMSE (mean)']):
        print(f"  Normalized RMSE (by mean):              {overall_metrics['NRMSE (mean)']:.4f}")
    else:
        print(f"  Normalized RMSE (by mean):              N/A (zero mean)")
    
    print("\nPercentage Error Metrics:")
    print(f"  Mean Absolute % Error (MAPE):           {overall_metrics['MAPE (%)']:.2f}%")
    print(f"  Symmetric MAPE (sMAPE):                 {overall_metrics['sMAPE (%)']:.2f}%")
    
    print("\nGoodness-of-Fit Metrics:")
    if not np.isnan(overall_metrics['R²']):
        print(f"  R-squared (R²):                         {overall_metrics['R²']:.4f}")
    else:
        print(f"  R-squared (R²):                         N/A (insufficient variance)")
    
    if not np.isnan(overall_metrics['MASE']):
        print(f"  Mean Absolute Scaled Error (MASE):      {overall_metrics['MASE']:.4f}")
    else:
        print(f"  Mean Absolute Scaled Error (MASE):      N/A")
    
    # Interpretation guide
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    
    if not np.isnan(overall_metrics['R²']):
        print("R² (closer to 1.0 is better):")
        if overall_metrics['R²'] >= 0.9:
            print("  ✓ Excellent fit (≥0.9)")
        elif overall_metrics['R²'] >= 0.7:
            print("  ✓ Good fit (0.7-0.9)")
        elif overall_metrics['R²'] >= 0.5:
            print("  ~ Moderate fit (0.5-0.7)")
        else:
            print("  ✗ Poor fit (<0.5)")
    
    print("\nsMAPE (closer to 0% is better):")
    if overall_metrics['sMAPE (%)'] <= 10:
        print("  ✓ Excellent accuracy (≤10%)")
    elif overall_metrics['sMAPE (%)'] <= 20:
        print("  ✓ Good accuracy (10-20%)")
    elif overall_metrics['sMAPE (%)'] <= 30:
        print("  ~ Moderate accuracy (20-30%)")
    else:
        print("  ✗ Poor accuracy (>30%)")
    
    # Diagnostic info
    print("\n" + "="*80)
    print("DIAGNOSTIC INFORMATION")
    print("="*80)
    print(f"Valid comparison points: {len(comparison)}")
    actual_clean, predicted_clean = clean_comparison_data(
        comparison['wartosc_actual'].values,
        comparison['wartosc_predicted'].values
    )
    print(f"Valid points for metrics: {len(actual_clean)}")
    print(f"Data range (actual): [{comparison['wartosc_actual'].min():.2f}, {comparison['wartosc_actual'].max():.2f}]")
    print(f"Data range (predicted): [{comparison['wartosc_predicted'].min():.2f}, {comparison['wartosc_predicted'].max():.2f}]")
    
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
            print("\nKey Takeaway:")
            if not np.isnan(metrics['R²']):
                print(f"  Overall prediction accuracy (R²): {metrics['R²']:.4f}")
            else:
                print(f"  Overall prediction accuracy (R²): N/A")
            print(f"  Average error (sMAPE): {metrics['sMAPE (%)']:.2f}%")
            print(f"  Mean Absolute Error: {metrics['MAE']:.4f}")
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
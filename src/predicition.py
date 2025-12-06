"""
Time Series Forecasting for Multi-Criteria Categories.

This module provides tools for predicting future values of categories using:
1. Linear Regression with trend analysis
2. ARIMA (AutoRegressive Integrated Moving Average)
3. Exponential Smoothing (Holt's method)
4. Ensemble forecasting combining multiple methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ForecastConfig:
    """Configuration for forecasting."""
    forecast_years: int = 3
    min_historical_years: int = 3
    ensemble_weights: Dict[str, float] = None
    confidence_level: float = 0.95
    
    def __post_init__(self):
        """Set default ensemble weights if not provided."""
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'linear': 0.33,
                'exponential': 0.33,
                'arima': 0.34
            }
        
        # Validate weights sum to 1
        total_weight = sum(self.ensemble_weights.values())
        if not abs(total_weight - 1.0) < 0.01:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total_weight}")


class LinearTrendForecaster:
    """
    Linear regression-based trend forecasting.
    
    Fits a linear model to historical data and extrapolates future values.
    """
    
    def __init__(self, data: pd.Series, years: pd.Series):
        """
        Initialize linear trend forecaster.
        
        Args:
            data: Historical values
            years: Corresponding years
        """
        self.data = data.values
        self.years = years.values
        self._coefficients = None
        
    def fit(self):
        """Fit linear regression model."""
        # Convert to arrays
        X = self.years.reshape(-1, 1)
        y = self.data
        
        # Calculate coefficients using normal equation
        # β = (X^T X)^(-1) X^T y
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        try:
            self._coefficients = np.linalg.lstsq(
                X_with_intercept.T @ X_with_intercept,
                X_with_intercept.T @ y,
                rcond=None
            )[0]
        except np.linalg.LinAlgError:
            # Fallback to simple mean if regression fails
            self._coefficients = np.array([np.mean(y), 0])
    
    def predict(self, future_years: np.ndarray) -> np.ndarray:
        """
        Predict values for future years.
        
        Args:
            future_years: Array of years to predict
            
        Returns:
            Array of predicted values
        """
        if self._coefficients is None:
            self.fit()
        
        X_future = np.column_stack([np.ones(len(future_years)), future_years])
        predictions = X_future @ self._coefficients
        
        return predictions
    
    def get_trend_info(self) -> Dict[str, float]:
        """Get information about the fitted trend."""
        if self._coefficients is None:
            self.fit()
        
        intercept, slope = self._coefficients
        
        # Calculate R-squared
        X_with_intercept = np.column_stack([np.ones(len(self.years)), self.years])
        predictions = X_with_intercept @ self._coefficients
        
        ss_total = np.sum((self.data - np.mean(self.data)) ** 2)
        ss_residual = np.sum((self.data - predictions) ** 2)
        
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        return {
            'intercept': intercept,
            'slope': slope,
            'r_squared': r_squared,
            'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        }


class ExponentialSmoothingForecaster:
    """
    Holt's exponential smoothing for trend forecasting.
    
    Uses level and trend smoothing to capture both current value and rate of change.
    """
    
    def __init__(self, data: pd.Series, alpha: float = 0.3, beta: float = 0.1):
        """
        Initialize exponential smoothing forecaster.
        
        Args:
            data: Historical values
            alpha: Level smoothing parameter (0-1)
            beta: Trend smoothing parameter (0-1)
        """
        self.data = data.values
        self.alpha = alpha
        self.beta = beta
        self._level = None
        self._trend = None
        
    def fit(self):
        """Fit Holt's exponential smoothing model."""
        # Initialize level and trend
        self._level = self.data[0]
        
        if len(self.data) > 1:
            self._trend = self.data[1] - self.data[0]
        else:
            self._trend = 0
        
        # Apply Holt's method
        for value in self.data[1:]:
            prev_level = self._level
            
            # Update level
            self._level = self.alpha * value + (1 - self.alpha) * (prev_level + self._trend)
            
            # Update trend
            self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Predict values for future steps.
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Array of predicted values
        """
        if self._level is None or self._trend is None:
            self.fit()
        
        predictions = np.zeros(steps)
        
        for h in range(steps):
            predictions[h] = self._level + (h + 1) * self._trend
        
        return predictions


class SimpleARIMAForecaster:
    """
    Simplified ARIMA-like forecasting.
    
    Implements a basic autoregressive model with differencing for stationarity.
    """
    
    def __init__(self, data: pd.Series, ar_order: int = 2):
        """
        Initialize ARIMA forecaster.
        
        Args:
            data: Historical values
            ar_order: Order of autoregressive model
        """
        self.data = data.values
        self.ar_order = min(ar_order, len(data) - 1)
        self._coefficients = None
        self._last_values = None
        
    def _difference(self, series: np.ndarray, order: int = 1) -> np.ndarray:
        """Apply differencing to make series stationary."""
        diff_series = series.copy()
        
        for _ in range(order):
            diff_series = np.diff(diff_series)
        
        return diff_series
    
    def fit(self):
        """Fit AR model to differenced data."""
        # Apply first-order differencing
        diff_data = self._difference(self.data, order=1)
        
        # Store last values for reconstruction
        self._last_values = self.data[-self.ar_order:].copy()
        
        # Fit AR model to differenced data
        if len(diff_data) <= self.ar_order:
            # Not enough data, use simple average
            self._coefficients = np.ones(self.ar_order) / self.ar_order
            return
        
        # Create lagged features
        X = []
        y = []
        
        for i in range(self.ar_order, len(diff_data)):
            X.append(diff_data[i-self.ar_order:i][::-1])
            y.append(diff_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit using least squares
        try:
            self._coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self._coefficients = np.ones(self.ar_order) / self.ar_order
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Predict values for future steps.
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Array of predicted values
        """
        if self._coefficients is None:
            self.fit()
        
        predictions = []
        
        # Start with last known value
        current_level = self.data[-1]
        
        # Keep track of recent differences
        recent_diffs = np.diff(self._last_values).tolist()
        
        for _ in range(steps):
            # Predict next difference
            if len(recent_diffs) >= self.ar_order:
                lagged_diffs = np.array(recent_diffs[-self.ar_order:][::-1])
            else:
                # Pad with zeros if not enough history
                lagged_diffs = np.zeros(self.ar_order)
                lagged_diffs[-len(recent_diffs):] = recent_diffs[::-1]
            
            next_diff = np.dot(self._coefficients, lagged_diffs)
            
            # Add difference to current level
            next_value = current_level + next_diff
            
            predictions.append(next_value)
            
            # Update for next iteration
            recent_diffs.append(next_diff)
            current_level = next_value
        
        return np.array(predictions)


class EnsembleForecaster:
    """
    Ensemble forecasting combining multiple methods.
    
    Combines linear trend, exponential smoothing, and ARIMA forecasts.
    """
    
    def __init__(self, data: pd.Series, years: pd.Series, config: ForecastConfig):
        """
        Initialize ensemble forecaster.
        
        Args:
            data: Historical values
            years: Corresponding years
            config: Forecast configuration
        """
        self.data = data
        self.years = years
        self.config = config
        
        # Initialize individual forecasters
        self.linear_forecaster = LinearTrendForecaster(data, years)
        self.exp_forecaster = ExponentialSmoothingForecaster(data)
        self.arima_forecaster = SimpleARIMAForecaster(data)
    
    def forecast(self, future_years: np.ndarray) -> pd.DataFrame:
        """
        Generate ensemble forecast.
        
        Args:
            future_years: Array of years to forecast
            
        Returns:
            DataFrame with forecasts from each method and ensemble prediction
        """
        steps = len(future_years)
        
        # Get predictions from each method
        linear_pred = self.linear_forecaster.predict(future_years)
        exp_pred = self.exp_forecaster.predict(steps)
        arima_pred = self.arima_forecaster.predict(steps)
        
        # Calculate ensemble prediction
        ensemble_pred = (
            self.config.ensemble_weights['linear'] * linear_pred +
            self.config.ensemble_weights['exponential'] * exp_pred +
            self.config.ensemble_weights['arima'] * arima_pred
        )
        
        # Calculate prediction intervals (simplified)
        # Use standard deviation of individual forecasts as uncertainty measure
        individual_forecasts = np.column_stack([linear_pred, exp_pred, arima_pred])
        forecast_std = np.std(individual_forecasts, axis=1)
        
        # Confidence intervals (assuming normal distribution)
        from scipy import stats
        z_score = stats.norm.ppf((1 + self.config.confidence_level) / 2)
        
        lower_bound = ensemble_pred - z_score * forecast_std
        upper_bound = ensemble_pred + z_score * forecast_std
        
        # Create results DataFrame
        results = pd.DataFrame({
            'year': future_years,
            'linear_forecast': linear_pred,
            'exponential_forecast': exp_pred,
            'arima_forecast': arima_pred,
            'ensemble_forecast': ensemble_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'forecast_std': forecast_std
        })
        
        return results


class CategoryPredictor:
    """
    Predict future values for all categories in the dataset.
    
    Handles multiple categories and KPIs with historical time series data.
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[ForecastConfig] = None):
        """
        Initialize category predictor.
        
        Args:
            df: DataFrame with columns [kpi_id, cat_id, value, direction, year]
            config: Forecast configuration (default: ForecastConfig())
        """
        self.df = df
        self.config = config or ForecastConfig()
        
        # Validate required columns
        required_columns = {'kpi_id', 'cat_id', 'value', 'year', 'direction'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    def _validate_historical_data(self, data: pd.Series) -> bool:
        """Check if there's enough historical data for forecasting."""
        return len(data) >= self.config.min_historical_years
    
    def predict_category(self, cat_id: str, kpi_id: str) -> pd.DataFrame:
        """
        Predict future values for a specific category and KPI.
        
        Args:
            cat_id: Category identifier
            kpi_id: KPI identifier
            
        Returns:
            DataFrame with forecast results
        """
        # Filter data for this category and KPI
        mask = (self.df['cat_id'] == cat_id) & (self.df['kpi_id'] == kpi_id)
        historical_data = self.df[mask].sort_values('year')
        
        if not self._validate_historical_data(historical_data):
            raise ValueError(
                f"Insufficient historical data for cat_id={cat_id}, kpi_id={kpi_id}. "
                f"Need at least {self.config.min_historical_years} years."
            )
        
        # Extract time series
        years = historical_data['year']
        values = historical_data['value']
        direction = historical_data['direction'].iloc[0]
        
        # Generate future years
        last_year = years.max()
        future_years = np.arange(
            last_year + 1, 
            last_year + 1 + self.config.forecast_years
        )
        
        # Create and run ensemble forecaster
        forecaster = EnsembleForecaster(values, years, self.config)
        forecast_df = forecaster.forecast(future_years)
        
        # Add metadata
        forecast_df['cat_id'] = cat_id
        forecast_df['kpi_id'] = kpi_id
        forecast_df['direction'] = direction
        
        # Add trend information
        trend_info = forecaster.linear_forecaster.get_trend_info()
        forecast_df['trend'] = trend_info['trend']
        forecast_df['trend_strength'] = abs(trend_info['slope'])
        
        return forecast_df
    
    def predict_all_categories(self) -> pd.DataFrame:
        """
        Predict future values for all categories and KPIs.
        
        Returns:
            DataFrame with forecasts for all category-KPI combinations
        """
        all_forecasts = []
        
        # Get unique combinations
        combinations = self.df[['cat_id', 'kpi_id']].drop_duplicates()
        
        total_combinations = len(combinations)
        print(f"Forecasting {total_combinations} category-KPI combinations...")
        
        for idx, (_, row) in enumerate(combinations.iterrows(), 1):
            cat_id = row['cat_id']
            kpi_id = row['kpi_id']
            
            try:
                forecast = self.predict_category(cat_id, kpi_id)
                all_forecasts.append(forecast)
                
                if idx % 10 == 0 or idx == total_combinations:
                    print(f"  Progress: {idx}/{total_combinations} ({idx/total_combinations*100:.1f}%)")
                    
            except ValueError as e:
                print(f"  ⚠ Skipping {cat_id}-{kpi_id}: {str(e)}")
                continue
        
        if not all_forecasts:
            raise ValueError("No valid forecasts generated. Check your data.")
        
        return pd.concat(all_forecasts, ignore_index=True)
    
    def get_forecast_summary(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for forecasts.
        
        Args:
            forecasts: DataFrame with forecast results
            
        Returns:
            Summary DataFrame with key statistics
        """
        summary = forecasts.groupby(['cat_id', 'kpi_id']).agg({
            'ensemble_forecast': ['mean', 'min', 'max'],
            'trend': 'first',
            'trend_strength': 'mean',
            'forecast_std': 'mean'
        }).reset_index()
        
        summary.columns = [
            'cat_id', 'kpi_id',
            'avg_forecast', 'min_forecast', 'max_forecast',
            'trend', 'avg_trend_strength', 'avg_uncertainty'
        ]
        
        return summary


class ForecastExporter:
    """Export forecast results to various formats."""
    
    @staticmethod
    def save_forecasts(forecasts: pd.DataFrame, 
                      summary: pd.DataFrame,
                      base_filename: str = 'forecast'):
        """
        Save forecast results to CSV files.
        
        Args:
            forecasts: Detailed forecast DataFrame
            summary: Summary statistics DataFrame
            base_filename: Base name for output files
        """
        # Save detailed forecasts
        forecasts.to_csv(f'{base_filename}_detailed.csv', index=False)
        
        # Save summary
        summary.to_csv(f'{base_filename}_summary.csv', index=False)
        
        # Save ensemble forecasts in format compatible with analysis.py
        analysis_format = forecasts[['kpi_id', 'cat_id', 'ensemble_forecast', 'direction', 'year']].copy()
        analysis_format.rename(columns={'ensemble_forecast': 'value'}, inplace=True)
        analysis_format.to_csv(f'{base_filename}_for_analysis.csv', index=False)
        
        print(f"\n✓ Forecasts saved:")
        print(f"  • {base_filename}_detailed.csv (all methods + confidence intervals)")
        print(f"  • {base_filename}_summary.csv (summary statistics)")
        print(f"  • {base_filename}_for_analysis.csv (ready for analysis.py)")
    
    @staticmethod
    def print_forecast_summary(summary: pd.DataFrame, top_n: int = 10):
        """Print summary of forecasted categories."""
        print(f"\n{'='*100}")
        print(f"FORECAST SUMMARY - TOP {top_n} BY AVERAGE FORECAST VALUE")
        print(f"{'='*100}\n")
        
        # Sort by average forecast (descending)
        top_categories = summary.sort_values('avg_forecast', ascending=False).head(top_n)
        
        print(top_categories[['cat_id', 'kpi_id', 'avg_forecast', 'trend', 
                             'avg_trend_strength', 'avg_uncertainty']].to_string(index=False))
        
        print(f"\n{'='*100}")
        
        # Print some interesting statistics
        print("\nOverall Statistics:")
        print(f"  • Categories with increasing trend: {(summary['trend'] == 'increasing').sum()}")
        print(f"  • Categories with decreasing trend: {(summary['trend'] == 'decreasing').sum()}")
        print(f"  • Categories with stable trend: {(summary['trend'] == 'stable').sum()}")
        print(f"  • Average uncertainty (std): {summary['avg_uncertainty'].mean():.2f}")


def predict_future_values(filepath: str,
                         forecast_years: int = 3,
                         min_historical_years: int = 3,
                         output_base: str = 'forecast') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to predict future values for all categories.
    
    Args:
        filepath: Path to input CSV file (must include 'year' column)
        forecast_years: Number of years to forecast ahead
        min_historical_years: Minimum years of historical data required
        output_base: Base filename for output files
        
    Returns:
        Tuple of (detailed_forecasts, summary_statistics)
    """
    # Load data
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Validate year column exists
    if 'year' not in df.columns:
        raise ValueError("Input file must contain 'year' column for time series forecasting")
    
    print(f"Data loaded: {len(df)} records")
    print(f"Years range: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique categories: {df['cat_id'].nunique()}")
    print(f"Unique KPIs: {df['kpi_id'].nunique()}")
    
    # Configure and run prediction
    config = ForecastConfig(
        forecast_years=forecast_years,
        min_historical_years=min_historical_years
    )
    
    predictor = CategoryPredictor(df, config)
    
    # Generate forecasts
    print(f"\nGenerating {forecast_years}-year forecasts...")
    forecasts = predictor.predict_all_categories()
    
    # Generate summary
    summary = predictor.get_forecast_summary(forecasts)
    
    # Export results
    ForecastExporter.print_forecast_summary(summary)
    ForecastExporter.save_forecasts(forecasts, summary, output_base)
    
    return forecasts, summary


if __name__ == '__main__':
    # Example usage
    try:
        forecasts, summary = predict_future_values(
            filepath='data.csv',
            forecast_years=3,
            min_historical_years=3,
            output_base='forecast'
        )
        
        print("\n✓ Forecasting completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        raise
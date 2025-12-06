"""
Multi-Criteria Decision Analysis using TOPSIS and Monte Carlo Simulation.

This module provides tools for analyzing alternatives using:
1. TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
2. Monte Carlo simulation for robust ranking
3. Ensemble method combining both approaches
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for ensemble analysis."""
    n_simulations: int = 1000
    topsis_weight: float = 0.5
    mc_noise_std: float = 0.05
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.topsis_weight <= 1:
            raise ValueError("topsis_weight must be between 0 and 1")
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be positive")


class TOPSISAnalyzer:
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) analyzer.
    
    TOPSIS ranks alternatives based on their distance from ideal and anti-ideal solutions.
    """
    
    def __init__(self, matrix: pd.DataFrame, directions: pd.Series, weights: Optional[np.ndarray] = None):
        """
        Initialize TOPSIS analyzer.
        
        Args:
            matrix: Decision matrix (alternatives × criteria)
            directions: Series indicating 'max' or 'min' for each criterion
            weights: Optional weights for criteria (default: equal weights)
        """
        self.matrix = matrix
        self.directions = directions
        self.weights = weights if weights is not None else self._default_weights()
        
    def _default_weights(self) -> np.ndarray:
        """Generate equal weights for all criteria."""
        n_criteria = len(self.matrix.columns)
        return np.ones(n_criteria) / n_criteria
    
    def _normalize_matrix(self) -> pd.DataFrame:
        """Normalize matrix using vector normalization."""
        return self.matrix / np.sqrt((self.matrix ** 2).sum(axis=0))
    
    def _get_ideal_solutions(self, weighted_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine ideal and anti-ideal solutions.
        
        Returns:
            Tuple of (ideal_solution, anti_ideal_solution)
        """
        ideal = np.zeros(len(self.matrix.columns))
        anti_ideal = np.zeros(len(self.matrix.columns))
        
        for i, col in enumerate(self.matrix.columns):
            col_values = weighted_matrix.iloc[:, i]
            
            if self.directions[col] == 'max':
                ideal[i] = col_values.max()
                anti_ideal[i] = col_values.min()
            else:  # min
                ideal[i] = col_values.min()
                anti_ideal[i] = col_values.max()
        
        return ideal, anti_ideal
    
    def _calculate_distances(self, weighted_matrix: pd.DataFrame, 
                            ideal: np.ndarray, anti_ideal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Euclidean distances to ideal and anti-ideal solutions.
        
        Returns:
            Tuple of (distance_to_ideal, distance_to_anti_ideal)
        """
        dist_ideal = np.sqrt(((weighted_matrix - ideal) ** 2).sum(axis=1))
        dist_anti_ideal = np.sqrt(((weighted_matrix - anti_ideal) ** 2).sum(axis=1))
        
        return dist_ideal.values, dist_anti_ideal.values
    
    def analyze(self) -> pd.Series:
        """
        Perform TOPSIS analysis.
        
        Returns:
            Series of TOPSIS scores (higher is better)
        """
        # Step 1: Normalize matrix
        normalized = self._normalize_matrix()
        
        # Step 2: Apply weights
        weighted = normalized * self.weights
        
        # Step 3: Determine ideal solutions
        ideal, anti_ideal = self._get_ideal_solutions(weighted)
        
        # Step 4: Calculate distances
        dist_ideal, dist_anti_ideal = self._calculate_distances(weighted, ideal, anti_ideal)
        
        # Step 5: Calculate relative closeness to ideal solution
        # Avoid division by zero
        denominator = dist_ideal + dist_anti_ideal
        scores = np.where(denominator == 0, 0.5, dist_anti_ideal / denominator)
        
        return pd.Series(scores, index=self.matrix.index, name='topsis_score')


class MonteCarloAnalyzer:
    """
    Monte Carlo simulation for robust multi-criteria decision analysis.
    
    Uses stochastic sampling to account for uncertainty in weights and data.
    """
    
    def __init__(self, matrix: pd.DataFrame, directions: pd.Series, config: AnalysisConfig):
        """
        Initialize Monte Carlo analyzer.
        
        Args:
            matrix: Decision matrix (alternatives × criteria)
            directions: Series indicating 'max' or 'min' for each criterion
            config: Configuration object with simulation parameters
        """
        self.matrix = matrix
        self.directions = directions
        self.config = config
    
    def _generate_random_weights(self) -> np.ndarray:
        """Generate random weights using Dirichlet distribution (sum to 1)."""
        n_criteria = len(self.matrix.columns)
        return np.random.dirichlet(np.ones(n_criteria))
    
    def _add_noise(self) -> pd.DataFrame:
        """Add random noise to matrix to simulate uncertainty."""
        noise = np.random.normal(1, self.config.mc_noise_std, self.matrix.shape)
        return self.matrix * noise
    
    def simulate(self) -> pd.Series:
        """
        Perform Monte Carlo simulation.
        
        Returns:
            Series of averaged scores across all simulations
        """
        n_alternatives = len(self.matrix)
        accumulated_scores = np.zeros(n_alternatives)
        
        for iteration in range(self.config.n_simulations):
            # Generate random weights
            random_weights = self._generate_random_weights()
            
            # Add stochastic noise
            noisy_matrix = self._add_noise()
            
            # Run TOPSIS with this iteration's parameters
            topsis = TOPSISAnalyzer(noisy_matrix, self.directions, random_weights)
            iteration_scores = topsis.analyze()
            
            accumulated_scores += iteration_scores.values
        
        # Average across all simulations
        avg_scores = accumulated_scores / self.config.n_simulations
        
        return pd.Series(avg_scores, index=self.matrix.index, name='monte_carlo_score')


class EnsembleAnalyzer:
    """
    Ensemble method combining TOPSIS and Monte Carlo for robust ranking.
    
    Combines deterministic TOPSIS with stochastic Monte Carlo to leverage
    strengths of both approaches.
    """
    
    def __init__(self, matrix: pd.DataFrame, directions: pd.Series, 
                 config: Optional[AnalysisConfig] = None):
        """
        Initialize ensemble analyzer.
        
        Args:
            matrix: Decision matrix (alternatives × criteria)
            directions: Series indicating 'max' or 'min' for each criterion
            config: Configuration object (default: AnalysisConfig())
        """
        self.matrix = matrix
        self.directions = directions
        self.config = config or AnalysisConfig()
    
    def analyze(self) -> pd.DataFrame:
        """
        Perform ensemble analysis.
        
        Returns:
            DataFrame with TOPSIS, Monte Carlo, ensemble scores and rankings
        """
        # Run TOPSIS analysis
        print("Running TOPSIS analysis...")
        topsis_analyzer = TOPSISAnalyzer(self.matrix, self.directions)
        topsis_scores = topsis_analyzer.analyze()
        
        # Run Monte Carlo simulation
        print(f"Running Monte Carlo simulation ({self.config.n_simulations} iterations)...")
        mc_analyzer = MonteCarloAnalyzer(self.matrix, self.directions, self.config)
        mc_scores = mc_analyzer.simulate()
        
        # Combine scores using weighted average
        ensemble_scores = (
            self.config.topsis_weight * topsis_scores +
            (1 - self.config.topsis_weight) * mc_scores
        )
        
        # Create results DataFrame
        results = pd.DataFrame({
            'alternative_id': self.matrix.index,
            'topsis_score': topsis_scores.values,
            'monte_carlo_score': mc_scores.values,
            'ensemble_score': ensemble_scores.values,
            'topsis_rank': topsis_scores.rank(ascending=False),
            'monte_carlo_rank': mc_scores.rank(ascending=False),
            'ensemble_rank': ensemble_scores.rank(ascending=False)
        })
        
        return results.sort_values('ensemble_rank')


class DataLoader:
    """Load and prepare data for analysis."""
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Expected format:
        - kpi_id: identifier for alternatives
        - cat_id: identifier for criteria
        - value: numeric value
        - direction: 'max' or 'min'
        """
        df = pd.read_csv(filepath)
        required_columns = {'kpi_id', 'cat_id', 'value', 'direction'}
        
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        return df
    
    @staticmethod
    def prepare_decision_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert long-format data to decision matrix.
        
        Args:
            df: DataFrame with columns [kpi_id, cat_id, value, direction]
            
        Returns:
            Tuple of (decision_matrix, directions)
        """
        # Pivot to create decision matrix
        matrix = df.pivot(index='kpi_id', columns='cat_id', values='value')
        
        # Extract directions for each criterion
        directions = df.groupby('cat_id')['direction'].first()
        
        # Validate
        if matrix.isnull().any().any():
            raise ValueError("Decision matrix contains missing values")
        
        return matrix, directions


class ResultsExporter:
    """Export analysis results to various formats."""
    
    @staticmethod
    def save_to_csv(results: pd.DataFrame, base_filename: str = 'results'):
        """
        Save analysis results to CSV files.
        
        Creates three files:
        - {base_filename}_topsis.csv
        - {base_filename}_monte_carlo.csv
        - {base_filename}_ensemble.csv
        """
        # TOPSIS results
        topsis_df = results[['alternative_id', 'topsis_score', 'topsis_rank']].copy()
        topsis_df = topsis_df.sort_values('topsis_rank')
        topsis_df.to_csv(f'{base_filename}_topsis.csv', index=False)
        
        # Monte Carlo results
        mc_df = results[['alternative_id', 'monte_carlo_score', 'monte_carlo_rank']].copy()
        mc_df = mc_df.sort_values('monte_carlo_rank')
        mc_df.to_csv(f'{base_filename}_monte_carlo.csv', index=False)
        
        # Ensemble results
        ensemble_df = results[['alternative_id', 'ensemble_score', 'ensemble_rank']].copy()
        ensemble_df.to_csv(f'{base_filename}_ensemble.csv', index=False)
        
        print(f"\n✓ Results saved:")
        print(f"  • {base_filename}_topsis.csv")
        print(f"  • {base_filename}_monte_carlo.csv")
        print(f"  • {base_filename}_ensemble.csv")
    
    @staticmethod
    def print_summary(results: pd.DataFrame, top_n: int = 10):
        """Print summary of top alternatives."""
        print(f"\n{'='*80}")
        print(f"TOP {top_n} ALTERNATIVES - ENSEMBLE RANKING")
        print(f"{'='*80}\n")
        
        top_results = results.head(top_n)
        
        print(top_results[['ensemble_rank', 'alternative_id', 'ensemble_score', 
                          'topsis_score', 'monte_carlo_score']].to_string(index=False))
        
        print(f"\n{'='*80}")


def run_analysis(filepath: str, 
                n_simulations: int = 1000,
                topsis_weight: float = 0.5,
                output_base: str = 'results') -> pd.DataFrame:
    """
    Main function to run complete ensemble analysis.
    
    Args:
        filepath: Path to input CSV file
        n_simulations: Number of Monte Carlo iterations
        topsis_weight: Weight for TOPSIS in ensemble (0-1)
        output_base: Base filename for output files
        
    Returns:
        DataFrame with complete results
    """
    # Load and prepare data
    print(f"Loading data from {filepath}...")
    df = DataLoader.load_csv(filepath)
    matrix, directions = DataLoader.prepare_decision_matrix(df)
    
    print(f"Decision matrix: {matrix.shape[0]} alternatives × {matrix.shape[1]} criteria")
    
    # Configure and run analysis
    config = AnalysisConfig(
        n_simulations=n_simulations,
        topsis_weight=topsis_weight
    )
    
    analyzer = EnsembleAnalyzer(matrix, directions, config)
    results = analyzer.analyze()
    
    # Export results
    ResultsExporter.print_summary(results)
    ResultsExporter.save_to_csv(results, output_base)
    
    return results


if __name__ == '__main__':
    # Example usage
    results = run_analysis(
        filepath='data.csv',
        n_simulations=1000,
        topsis_weight=0.5,
        output_base='results'
    )
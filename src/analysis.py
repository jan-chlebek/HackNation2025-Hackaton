"""
Multi-Criteria Decision Analysis using TOPSIS, VIKOR and Monte Carlo Simulation.

This module provides tools for analyzing alternatives using:
1. TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
2. Fuzzy VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje)
3. Monte Carlo simulation for robust ranking
4. Ensemble method combining multiple approaches
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for ensemble analysis."""
    n_simulations: int = 1000
    topsis_weight: float = 0.33
    vikor_weight: float = 0.33
    mc_weight: float = 0.34
    mc_noise_std: float = 0.05
    use_fuzzy_vikor: bool = True
    fuzzy_spread: float = 0.1
    vikor_v: float = 0.5  # Weight for strategy of maximum group utility (0-1)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        total_weight = self.topsis_weight + self.vikor_weight + self.mc_weight
        if not abs(total_weight - 1.0) < 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be positive")
        if not 0 <= self.fuzzy_spread <= 1:
            raise ValueError("fuzzy_spread must be between 0 and 1")
        if not 0 <= self.vikor_v <= 1:
            raise ValueError("vikor_v must be between 0 and 1")


@dataclass
class FuzzyNumber:
    """Triangular fuzzy number representation (l, m, u)."""
    lower: float  # Lower bound
    middle: float  # Most likely value
    upper: float  # Upper bound
    
    def __post_init__(self):
        """Validate fuzzy number."""
        if not (self.lower <= self.middle <= self.upper):
            raise ValueError(f"Invalid fuzzy number: {self.lower}, {self.middle}, {self.upper}")
    
    def defuzzify(self, method: str = 'centroid') -> float:
        """
        Convert fuzzy number to crisp value.
        
        Args:
            method: 'centroid' (center of gravity) or 'mean'
        
        Returns:
            Crisp value
        """
        if method == 'centroid':
            return (self.lower + 4 * self.middle + self.upper) / 6
        else:  # mean
            return (self.lower + self.middle + self.upper) / 3
    
    @staticmethod
    def distance(fn1: 'FuzzyNumber', fn2: 'FuzzyNumber') -> float:
        """Calculate distance between two fuzzy numbers using vertex method."""
        return np.sqrt(
            ((fn1.lower - fn2.lower) ** 2 +
             (fn1.middle - fn2.middle) ** 2 +
             (fn1.upper - fn2.upper) ** 2) / 3
        )
    
    def __add__(self, other: 'FuzzyNumber') -> 'FuzzyNumber':
        """Add two fuzzy numbers."""
        return FuzzyNumber(
            self.lower + other.lower,
            self.middle + other.middle,
            self.upper + other.upper
        )
    
    def __sub__(self, other: 'FuzzyNumber') -> 'FuzzyNumber':
        """Subtract two fuzzy numbers."""
        return FuzzyNumber(
            self.lower - other.upper,
            self.middle - other.middle,
            self.upper - other.lower
        )
    
    def __mul__(self, scalar: float) -> 'FuzzyNumber':
        """Multiply fuzzy number by scalar."""
        if scalar >= 0:
            return FuzzyNumber(
                self.lower * scalar,
                self.middle * scalar,
                self.upper * scalar
            )
        else:
            return FuzzyNumber(
                self.upper * scalar,
                self.middle * scalar,
                self.lower * scalar
            )
    
    def __truediv__(self, scalar: float) -> 'FuzzyNumber':
        """Divide fuzzy number by scalar."""
        if scalar == 0:
            raise ValueError("Division by zero")
        return self * (1 / scalar)
    
    def __repr__(self):
        return f"FN({self.lower:.2f}, {self.middle:.2f}, {self.upper:.2f})"


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
            weights: Optional weights for criteria (default: coefficient of variation based)
        """
        self.matrix = matrix
        self.directions = directions
        self.weights = weights if weights is not None else self._calculate_cv_weights()
        
    def _calculate_cv_weights(self) -> np.ndarray:
        """
        Calculate weights based on coefficient of variation (CV).
        
        Formula: w_j = (1/cv_j) / sum(1/cv_i) for all i
        
        Criteria with lower variation (more stable) get higher weights.
        This is an objective weighting method based on data characteristics.
        
        Returns:
            Array of normalized weights
        """
        n_criteria = len(self.matrix.columns)
        weights = np.zeros(n_criteria)
        
        for i, col in enumerate(self.matrix.columns):
            mean_val = self.matrix[col].mean()
            std_val = self.matrix[col].std()
            
            if mean_val == 0:
                cv = std_val if std_val > 0 else 1e-10
            else:
                cv = std_val / abs(mean_val)
            
            if cv == 0:
                cv = 1e-10
            
            weights[i] = 1 / cv
        
        weights_normalized = weights / weights.sum()
        
        return weights_normalized
    
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
    
    def get_weights_info(self) -> pd.DataFrame:
        """
        Get information about calculated weights.
        
        Returns:
            DataFrame with criteria names, CV values, and weights
        """
        cv_values = []
        
        for col in self.matrix.columns:
            mean_val = self.matrix[col].mean()
            std_val = self.matrix[col].std()
            
            if mean_val == 0:
                cv = std_val if std_val > 0 else 1e-10
            else:
                cv = std_val / abs(mean_val)
            
            cv_values.append(cv)
        
        return pd.DataFrame({
            'criterion': self.matrix.columns,
            'cv': cv_values,
            'weight': self.weights,
            'weight_pct': self.weights * 100
        })
    
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
        denominator = dist_ideal + dist_anti_ideal
        scores = np.where(denominator == 0, 0.5, dist_anti_ideal / denominator)
        
        return pd.Series(scores, index=self.matrix.index, name='topsis_score')


class FuzzyVIKORAnalyzer:
    """
    Fuzzy VIKOR analyzer for multi-criteria decision making.
    
    VIKOR determines compromise ranking by measuring closeness to ideal solution.
    Uses fuzzy numbers to handle uncertainty in decision making.
    """
    
    def __init__(self, matrix: pd.DataFrame, directions: pd.Series, 
                 weights: Optional[np.ndarray] = None,
                 v: float = 0.5,
                 fuzzy_spread: float = 0.1):
        """
        Initialize Fuzzy VIKOR analyzer.
        
        Args:
            matrix: Decision matrix (alternatives × criteria)
            directions: Series indicating 'max' or 'min' for each criterion
            weights: Optional weights for criteria
            v: Weight for strategy of maximum group utility (0-1)
            fuzzy_spread: Spread for creating fuzzy numbers
        """
        self.matrix = matrix
        self.directions = directions
        self.v = v
        self.fuzzy_spread = fuzzy_spread
        self.weights = weights if weights is not None else self._calculate_cv_weights()
        
    def _calculate_cv_weights(self) -> np.ndarray:
        """Calculate weights based on coefficient of variation."""
        n_criteria = len(self.matrix.columns)
        weights = np.zeros(n_criteria)
        
        for i, col in enumerate(self.matrix.columns):
            mean_val = self.matrix[col].mean()
            std_val = self.matrix[col].std()
            
            if mean_val == 0:
                cv = std_val if std_val > 0 else 1e-10
            else:
                cv = std_val / abs(mean_val)
            
            if cv == 0:
                cv = 1e-10
            
            weights[i] = 1 / cv
        
        return weights / weights.sum()
    
    def _create_fuzzy_matrix(self) -> np.ndarray:
        """Convert crisp matrix to fuzzy matrix."""
        fuzzy_matrix = np.empty(self.matrix.shape, dtype=object)
        
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                value = self.matrix.iloc[i, j]
                spread = abs(value * self.fuzzy_spread)
                
                fuzzy_matrix[i, j] = FuzzyNumber(
                    lower=max(0, value - spread),
                    middle=value,
                    upper=value + spread
                )
        
        return fuzzy_matrix
    
    def _normalize_fuzzy_matrix(self, fuzzy_matrix: np.ndarray) -> np.ndarray:
        """Normalize fuzzy matrix."""
        normalized = np.empty(fuzzy_matrix.shape, dtype=object)
        
        for j in range(fuzzy_matrix.shape[1]):
            column = fuzzy_matrix[:, j]
            
            # Find max upper bound for beneficial criteria, min lower for non-beneficial
            if self.directions.iloc[j] == 'max':
                max_upper = max(fn.upper for fn in column)
                
                for i in range(fuzzy_matrix.shape[0]):
                    fn = column[i]
                    if max_upper > 0:
                        normalized[i, j] = FuzzyNumber(
                            fn.lower / max_upper,
                            fn.middle / max_upper,
                            fn.upper / max_upper
                        )
                    else:
                        normalized[i, j] = FuzzyNumber(0, 0, 0)
            else:  # min
                min_lower = min(fn.lower for fn in column if fn.lower > 0)
                
                for i in range(fuzzy_matrix.shape[0]):
                    fn = column[i]
                    if fn.upper > 0:
                        normalized[i, j] = FuzzyNumber(
                            min_lower / fn.upper,
                            min_lower / fn.middle if fn.middle > 0 else 0,
                            min_lower / fn.lower if fn.lower > 0 else min_lower / 1e-10
                        )
                    else:
                        normalized[i, j] = FuzzyNumber(0, 0, 0)
        
        return normalized
    
    def _get_fuzzy_ideal_solutions(self, normalized_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get fuzzy positive and negative ideal solutions."""
        n_criteria = normalized_matrix.shape[1]
        fpis = np.empty(n_criteria, dtype=object)  # Fuzzy Positive Ideal Solution
        fnis = np.empty(n_criteria, dtype=object)  # Fuzzy Negative Ideal Solution
        
        for j in range(n_criteria):
            column = normalized_matrix[:, j]
            
            # FPIS: maximum for all criteria (after normalization)
            fpis[j] = FuzzyNumber(
                max(fn.lower for fn in column),
                max(fn.middle for fn in column),
                max(fn.upper for fn in column)
            )
            
            # FNIS: minimum for all criteria (after normalization)
            fnis[j] = FuzzyNumber(
                min(fn.lower for fn in column),
                min(fn.middle for fn in column),
                min(fn.upper for fn in column)
            )
        
        return fpis, fnis
    
    def analyze(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Perform Fuzzy VIKOR analysis.
        
        Returns:
            Tuple of (Q scores, detailed results DataFrame)
        """
        # Step 1: Create fuzzy matrix
        fuzzy_matrix = self._create_fuzzy_matrix()
        
        # Step 2: Normalize fuzzy matrix
        normalized = self._normalize_fuzzy_matrix(fuzzy_matrix)
        
        # Step 3: Get fuzzy ideal solutions
        fpis, fnis = self._get_fuzzy_ideal_solutions(normalized)
        
        # Step 4: Calculate S and R values
        n_alternatives = self.matrix.shape[0]
        S = np.zeros(n_alternatives)  # Group utility
        R = np.zeros(n_alternatives)  # Individual regret
        
        for i in range(n_alternatives):
            s_components = []
            r_components = []
            
            for j in range(self.matrix.shape[1]):
                # Calculate distance from ideal
                dist = FuzzyNumber.distance(fpis[j], normalized[i, j])
                ideal_range = FuzzyNumber.distance(fpis[j], fnis[j])
                
                if ideal_range > 0:
                    normalized_dist = dist / ideal_range
                else:
                    normalized_dist = 0
                
                weighted_dist = self.weights[j] * normalized_dist
                
                s_components.append(weighted_dist)
                r_components.append(weighted_dist)
            
            S[i] = sum(s_components)
            R[i] = max(r_components) if r_components else 0
        
        # Step 5: Calculate Q values
        S_star = S.min()
        S_minus = S.max()
        R_star = R.min()
        R_minus = R.max()
        
        Q = np.zeros(n_alternatives)
        
        for i in range(n_alternatives):
            if (S_minus - S_star) > 0:
                s_term = self.v * (S[i] - S_star) / (S_minus - S_star)
            else:
                s_term = 0
            
            if (R_minus - R_star) > 0:
                r_term = (1 - self.v) * (R[i] - R_star) / (R_minus - R_star)
            else:
                r_term = 0
            
            Q[i] = s_term + r_term
        
        # Create detailed results
        results_df = pd.DataFrame({
            'alternative_id': self.matrix.index,
            'S': S,
            'R': R,
            'Q': Q,
            'S_rank': pd.Series(S).rank(),
            'R_rank': pd.Series(R).rank(),
            'Q_rank': pd.Series(Q).rank()
        })
        
        # VIKOR score (inverse of Q - higher is better for consistency with TOPSIS)
        vikor_scores = 1 - Q
        vikor_scores = pd.Series(vikor_scores, index=self.matrix.index, name='vikor_score')
        
        return vikor_scores, results_df


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
    Ensemble method combining TOPSIS, VIKOR and Monte Carlo for robust ranking.
    
    Combines multiple methods to leverage strengths of each approach.
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
            DataFrame with TOPSIS, VIKOR, Monte Carlo, ensemble scores and rankings
        """
        # Run TOPSIS analysis
        print("Running TOPSIS analysis...")
        topsis_analyzer = TOPSISAnalyzer(self.matrix, self.directions)
        topsis_scores = topsis_analyzer.analyze()
        
        # Print weight information
        print("\nCriterion weights (based on coefficient of variation):")
        weights_info = topsis_analyzer.get_weights_info()
        print(weights_info.to_string(index=False))
        
        # Run Fuzzy VIKOR analysis
        if self.config.use_fuzzy_vikor:
            print(f"\nRunning Fuzzy VIKOR analysis (v={self.config.vikor_v}, spread=±{self.config.fuzzy_spread*100}%)...")
            vikor_analyzer = FuzzyVIKORAnalyzer(
                self.matrix, 
                self.directions,
                weights=topsis_analyzer.weights,
                v=self.config.vikor_v,
                fuzzy_spread=self.config.fuzzy_spread
            )
            vikor_scores, vikor_details = vikor_analyzer.analyze()
        else:
            print("\nSkipping VIKOR analysis (disabled in config)...")
            vikor_scores = pd.Series(0, index=self.matrix.index, name='vikor_score')
        
        # Run Monte Carlo simulation
        print(f"\nRunning Monte Carlo simulation ({self.config.n_simulations} iterations)...")
        mc_analyzer = MonteCarloAnalyzer(self.matrix, self.directions, self.config)
        mc_scores = mc_analyzer.simulate()
        
        # Combine scores using weighted average
        ensemble_scores = (
            self.config.topsis_weight * topsis_scores +
            self.config.vikor_weight * vikor_scores +
            self.config.mc_weight * mc_scores
        )
        
        # Create results DataFrame
        results = pd.DataFrame({
            'alternative_id': self.matrix.index,
            'topsis_score': topsis_scores.values,
            'vikor_score': vikor_scores.values,
            'monte_carlo_score': mc_scores.values,
            'ensemble_score': ensemble_scores.values,
            'topsis_rank': topsis_scores.rank(ascending=False),
            'vikor_rank': vikor_scores.rank(ascending=False),
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
        
        Creates four files:
        - {base_filename}_topsis.csv
        - {base_filename}_vikor.csv
        - {base_filename}_monte_carlo.csv
        - {base_filename}_ensemble.csv
        """
        # TOPSIS results
        topsis_df = results[['alternative_id', 'topsis_score', 'topsis_rank']].copy()
        topsis_df = topsis_df.sort_values('topsis_rank')
        topsis_df.to_csv(f'{base_filename}_topsis.csv', index=False)
        
        # VIKOR results
        vikor_df = results[['alternative_id', 'vikor_score', 'vikor_rank']].copy()
        vikor_df = vikor_df.sort_values('vikor_rank')
        vikor_df.to_csv(f'{base_filename}_vikor.csv', index=False)
        
        # Monte Carlo results
        mc_df = results[['alternative_id', 'monte_carlo_score', 'monte_carlo_rank']].copy()
        mc_df = mc_df.sort_values('monte_carlo_rank')
        mc_df.to_csv(f'{base_filename}_monte_carlo.csv', index=False)
        
        # Ensemble results
        ensemble_df = results[['alternative_id', 'ensemble_score', 'ensemble_rank']].copy()
        ensemble_df.to_csv(f'{base_filename}_ensemble.csv', index=False)
        
        print(f"\n✓ Results saved:")
        print(f"  • {base_filename}_topsis.csv")
        print(f"  • {base_filename}_vikor.csv")
        print(f"  • {base_filename}_monte_carlo.csv")
        print(f"  • {base_filename}_ensemble.csv")
    
    @staticmethod
    def print_summary(results: pd.DataFrame, top_n: int = 10):
        """Print summary of top alternatives."""
        print(f"\n{'='*90}")
        print(f"TOP {top_n} ALTERNATIVES - ENSEMBLE RANKING")
        print(f"{'='*90}\n")
        
        top_results = results.head(top_n)
        
        print(top_results[['ensemble_rank', 'alternative_id', 'ensemble_score', 
                          'topsis_score', 'vikor_score', 'monte_carlo_score']].to_string(index=False))
        
        print(f"\n{'='*90}")


def run_analysis(filepath: str, 
                n_simulations: int = 1000,
                topsis_weight: float = 0.33,
                vikor_weight: float = 0.33,
                mc_weight: float = 0.34,
                use_fuzzy_vikor: bool = True,
                fuzzy_spread: float = 0.1,
                vikor_v: float = 0.5,
                output_base: str = 'results') -> pd.DataFrame:
    """
    Main function to run complete ensemble analysis.
    
    Args:
        filepath: Path to input CSV file
        n_simulations: Number of Monte Carlo iterations
        topsis_weight: Weight for TOPSIS in ensemble (0-1)
        vikor_weight: Weight for VIKOR in ensemble (0-1)
        mc_weight: Weight for Monte Carlo in ensemble (0-1)
        use_fuzzy_vikor: Whether to use Fuzzy VIKOR
        fuzzy_spread: Spread for fuzzy numbers (0-1)
        vikor_v: VIKOR strategy weight (0-1)
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
        topsis_weight=topsis_weight,
        vikor_weight=vikor_weight,
        mc_weight=mc_weight,
        use_fuzzy_vikor=use_fuzzy_vikor,
        fuzzy_spread=fuzzy_spread,
        vikor_v=vikor_v
    )
    
    analyzer = EnsembleAnalyzer(matrix, directions, config)
    results = analyzer.analyze()
    
    # Export results
    ResultsExporter.print_summary(results)
    ResultsExporter.save_to_csv(results, output_base)
    
    return results


if __name__ == '__main__':
    # Example usage with TOPSIS, Fuzzy VIKOR, and Monte Carlo
    results = run_analysis(
        filepath='data.csv',
        n_simulations=1000,
        topsis_weight=0.33,
        vikor_weight=0.33,
        mc_weight=0.34,
        use_fuzzy_vikor=True,
        fuzzy_spread=0.1,
        vikor_v=0.5,
        output_base='results'
    )
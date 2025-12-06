"""
Ensemble Multi-Criteria Decision Analysis using Monte Carlo Forecast Scenarios.

This module runs TOPSIS/VIKOR/Monte Carlo analysis across multiple forecast scenarios
and combines results for robust decision making under uncertainty.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import glob

from analysis import (
    AnalysisConfig, 
    EnsembleAnalyzer, 
    DataLoader,
    ResultsExporter
)


@dataclass
class ScenarioAnalysisConfig:
    """Configuration for multi-scenario analysis."""
    analysis_config: AnalysisConfig
    aggregation_method: str = 'mean'  # 'mean', 'median', 'weighted_mean'
    scenario_weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.aggregation_method not in ['mean', 'median', 'weighted_mean']:
            raise ValueError("aggregation_method must be 'mean', 'median', or 'weighted_mean'")
        
        if self.aggregation_method == 'weighted_mean' and self.scenario_weights is None:
            raise ValueError("scenario_weights required for weighted_mean aggregation")


class ScenarioAnalyzer:
    """
    Analyze multiple forecast scenarios and aggregate results.
    
    Runs complete TOPSIS/VIKOR/MC analysis on each scenario and combines rankings.
    """
    
    def __init__(self, scenario_files: List[str], config: Optional[ScenarioAnalysisConfig] = None):
        """
        Initialize scenario analyzer.
        
        Args:
            scenario_files: List of paths to scenario CSV files
            config: Configuration for scenario analysis
        """
        self.scenario_files = scenario_files
        self.config = config or ScenarioAnalysisConfig(
            analysis_config=AnalysisConfig()
        )
        self.scenario_results = []
        
    def analyze_scenario(self, filepath: str, scenario_idx: int) -> pd.DataFrame:
        """
        Analyze a single forecast scenario.
        
        Args:
            filepath: Path to scenario CSV file
            scenario_idx: Index of scenario for logging
            
        Returns:
            DataFrame with analysis results for this scenario
        """
        print(f"\n{'='*80}")
        print(f"Analyzing Scenario {scenario_idx + 1}/{len(self.scenario_files)}")
        print(f"File: {filepath}")
        print(f"{'='*80}")
        
        # Load and prepare data
        df = DataLoader.load_csv(filepath)
        matrix, directions = DataLoader.prepare_decision_matrix(df)
        
        # Run ensemble analysis
        analyzer = EnsembleAnalyzer(matrix, directions, self.config.analysis_config)
        results = analyzer.analyze()
        
        # Add scenario identifier
        results['scenario'] = scenario_idx
        
        return results
    
    def analyze_all_scenarios(self) -> List[pd.DataFrame]:
        """
        Analyze all forecast scenarios.
        
        Returns:
            List of result DataFrames, one per scenario
        """
        print(f"\nRunning ensemble analysis on {len(self.scenario_files)} scenarios...")
        
        for idx, filepath in enumerate(self.scenario_files):
            try:
                results = self.analyze_scenario(filepath, idx)
                self.scenario_results.append(results)
            except Exception as e:
                print(f"✗ Error analyzing scenario {idx + 1}: {str(e)}")
                continue
        
        if not self.scenario_results:
            raise ValueError("No scenarios analyzed successfully")
        
        print(f"\n✓ Successfully analyzed {len(self.scenario_results)} scenarios")
        
        return self.scenario_results
    
    def aggregate_results(self) -> pd.DataFrame:
        """
        Aggregate results across all scenarios.
        
        Returns:
            DataFrame with aggregated rankings and scores
        """
        if not self.scenario_results:
            self.analyze_all_scenarios()
        
        # Combine all scenario results
        all_results = pd.concat(self.scenario_results, ignore_index=True)
        
        # Aggregate by alternative
        agg_funcs = {
            'topsis_score': [self.config.aggregation_method, 'std', 'min', 'max'],
            'vikor_score': [self.config.aggregation_method, 'std', 'min', 'max'],
            'monte_carlo_score': [self.config.aggregation_method, 'std', 'min', 'max'],
            'ensemble_score': [self.config.aggregation_method, 'std', 'min', 'max'],
            'topsis_rank': [self.config.aggregation_method, 'std'],
            'vikor_rank': [self.config.aggregation_method, 'std'],
            'monte_carlo_rank': [self.config.aggregation_method, 'std'],
            'ensemble_rank': [self.config.aggregation_method, 'std']
        }
        
        # Apply weighted mean if specified
        if self.config.aggregation_method == 'weighted_mean' and self.config.scenario_weights is not None:
            # Custom weighted aggregation
            aggregated = self._weighted_aggregation(all_results)
        else:
            # Standard aggregation
            aggregated = all_results.groupby('alternative_id').agg(agg_funcs).reset_index()
            
            # Flatten column names
            aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
        
        # Calculate final ensemble ranks based on aggregated scores
        aggregated['final_topsis_rank'] = aggregated['topsis_score_' + self.config.aggregation_method].rank(ascending=False)
        aggregated['final_vikor_rank'] = aggregated['vikor_score_' + self.config.aggregation_method].rank(ascending=False)
        aggregated['final_mc_rank'] = aggregated['monte_carlo_score_' + self.config.aggregation_method].rank(ascending=False)
        aggregated['final_ensemble_rank'] = aggregated['ensemble_score_' + self.config.aggregation_method].rank(ascending=False)
        
        # Sort by final ensemble rank
        aggregated = aggregated.sort_values('final_ensemble_rank')
        
        return aggregated
    
    def _weighted_aggregation(self, all_results: pd.DataFrame) -> pd.DataFrame:
        """Apply weighted aggregation using scenario weights."""
        # Group by alternative
        grouped = all_results.groupby('alternative_id')
        
        aggregated_data = []
        
        for alt_id, group in grouped:
            # Sort by scenario to match weights
            group = group.sort_values('scenario')
            
            # Calculate weighted means
            weighted_record = {
                'alternative_id': alt_id,
                'topsis_score_weighted_mean': np.average(group['topsis_score'], weights=self.config.scenario_weights),
                'vikor_score_weighted_mean': np.average(group['vikor_score'], weights=self.config.scenario_weights),
                'monte_carlo_score_weighted_mean': np.average(group['monte_carlo_score'], weights=self.config.scenario_weights),
                'ensemble_score_weighted_mean': np.average(group['ensemble_score'], weights=self.config.scenario_weights),
                'topsis_score_std': group['topsis_score'].std(),
                'vikor_score_std': group['vikor_score'].std(),
                'monte_carlo_score_std': group['monte_carlo_score'].std(),
                'ensemble_score_std': group['ensemble_score'].std(),
            }
            
            aggregated_data.append(weighted_record)
        
        return pd.DataFrame(aggregated_data)
    
    def get_ranking_stability(self, aggregated: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ranking stability metrics.
        
        Args:
            aggregated: Aggregated results DataFrame
            
        Returns:
            DataFrame with stability metrics
        """
        stability = aggregated[['alternative_id']].copy()
        
        # Rank volatility (standard deviation of ranks across scenarios)
        stability['topsis_rank_volatility'] = aggregated['topsis_rank_std']
        stability['vikor_rank_volatility'] = aggregated['vikor_rank_std']
        stability['mc_rank_volatility'] = aggregated['monte_carlo_rank_std']
        stability['ensemble_rank_volatility'] = aggregated['ensemble_rank_std']
        
        # Score volatility
        stability['topsis_score_volatility'] = aggregated['topsis_score_std']
        stability['vikor_score_volatility'] = aggregated['vikor_score_std']
        stability['mc_score_volatility'] = aggregated['monte_carlo_score_std']
        stability['ensemble_score_volatility'] = aggregated['ensemble_score_std']
        
        # Overall stability score (lower is more stable)
        stability['overall_stability'] = (
            stability['ensemble_rank_volatility'] + 
            stability['ensemble_score_volatility']
        ) / 2
        
        return stability.sort_values('overall_stability')


class ScenarioResultsExporter:
    """Export multi-scenario analysis results."""
    
    @staticmethod
    def save_results(aggregated: pd.DataFrame,
                    stability: pd.DataFrame,
                    scenario_results: List[pd.DataFrame],
                    base_filename: str = 'scenario_analysis'):
        """
        Save scenario analysis results.
        
        Args:
            aggregated: Aggregated results across scenarios
            stability: Ranking stability metrics
            scenario_results: List of individual scenario results
            base_filename: Base name for output files
        """
        # Save aggregated results
        aggregated.to_csv(f'{base_filename}_aggregated.csv', index=False)
        
        # Save stability metrics
        stability.to_csv(f'{base_filename}_stability.csv', index=False)
        
        # Save combined scenario results
        all_scenarios = pd.concat(scenario_results, ignore_index=True)
        all_scenarios.to_csv(f'{base_filename}_all_scenarios.csv', index=False)
        
        print(f"\n✓ Scenario analysis results saved:")
        print(f"  • {base_filename}_aggregated.csv (aggregated rankings across scenarios)")
        print(f"  • {base_filename}_stability.csv (ranking stability metrics)")
        print(f"  • {base_filename}_all_scenarios.csv (detailed results for all scenarios)")
    
    @staticmethod
    def print_summary(aggregated: pd.DataFrame, 
                     stability: pd.DataFrame,
                     top_n: int = 10):
        """Print summary of scenario analysis."""
        agg_method = 'mean' if 'ensemble_score_mean' in aggregated.columns else 'weighted_mean'
        score_col = f'ensemble_score_{agg_method}'
        
        print(f"\n{'='*120}")
        print(f"SCENARIO ANALYSIS SUMMARY - TOP {top_n} ALTERNATIVES (Aggregated across all scenarios)")
        print(f"{'='*120}\n")
        
        top_results = aggregated.head(top_n)
        
        display_cols = ['alternative_id', 'final_ensemble_rank', score_col, 
                       'ensemble_score_std', 'ensemble_rank_std']
        
        # Rename for display
        display_df = top_results[display_cols].copy()
        display_df.columns = ['Alternative', 'Rank', 'Avg Score', 'Score Std', 'Rank Std']
        
        print(display_df.to_string(index=False))
        
        print(f"\n{'='*120}")
        print(f"\nTOP {top_n} MOST STABLE ALTERNATIVES (Low volatility across scenarios)")
        print(f"{'='*120}\n")
        
        top_stable = stability.head(top_n)
        
        stable_display = top_stable[['alternative_id', 'ensemble_rank_volatility', 
                                     'ensemble_score_volatility', 'overall_stability']].copy()
        stable_display.columns = ['Alternative', 'Rank Volatility', 'Score Volatility', 'Overall Stability']
        
        print(stable_display.to_string(index=False))
        
        print(f"\n{'='*120}")


def run_scenario_analysis(scenario_pattern: str,
                         n_simulations: int = 1000,
                         topsis_weight: float = 0.33,
                         vikor_weight: float = 0.33,
                         mc_weight: float = 0.34,
                         aggregation_method: str = 'mean',
                         output_base: str = 'scenario_analysis') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ensemble analysis across multiple Monte Carlo forecast scenarios.
    
    Args:
        scenario_pattern: Glob pattern for scenario files (e.g., 'forecast_mc_scenario_*.csv')
        n_simulations: Number of Monte Carlo iterations in each analysis
        topsis_weight: Weight for TOPSIS in ensemble
        vikor_weight: Weight for VIKOR in ensemble
        mc_weight: Weight for Monte Carlo in ensemble
        aggregation_method: Method to aggregate across scenarios ('mean', 'median')
        output_base: Base filename for output files
        
    Returns:
        Tuple of (aggregated_results, stability_metrics)
    """
    # Find scenario files
    scenario_files = sorted(glob.glob(scenario_pattern))
    
    if not scenario_files:
        raise ValueError(f"No files found matching pattern: {scenario_pattern}")
    
    print(f"Found {len(scenario_files)} scenario files")
    
    # Configure analysis
    analysis_config = AnalysisConfig(
        n_simulations=n_simulations,
        topsis_weight=topsis_weight,
        vikor_weight=vikor_weight,
        mc_weight=mc_weight
    )
    
    scenario_config = ScenarioAnalysisConfig(
        analysis_config=analysis_config,
        aggregation_method=aggregation_method
    )
    
    # Run scenario analysis
    analyzer = ScenarioAnalyzer(scenario_files, scenario_config)
    analyzer.analyze_all_scenarios()
    
    # Aggregate results
    print("\nAggregating results across scenarios...")
    aggregated = analyzer.aggregate_results()
    
    # Calculate stability
    stability = analyzer.get_ranking_stability(aggregated)
    
    # Export and print results
    ScenarioResultsExporter.print_summary(aggregated, stability)
    ScenarioResultsExporter.save_results(
        aggregated, 
        stability, 
        analyzer.scenario_results,
        output_base
    )
    
    return aggregated, stability


if __name__ == '__main__':
    # Example usage
    try:
        aggregated, stability = run_scenario_analysis(
            scenario_pattern='forecast_mc_scenario_*.csv',
            n_simulations=1000,
            aggregation_method='mean',
            output_base='scenario_analysis'
        )
        
        print("\n✓ Multi-scenario analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        raise
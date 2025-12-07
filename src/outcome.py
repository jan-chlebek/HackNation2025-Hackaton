"""
Sector Analysis using TOPSIS, VIKOR and Monte Carlo for SEKCJA level data.

This script loads data from kpi-value-table.csv, filters by year and sector type,
and performs multi-criteria decision analysis using only calculated indicators (ID >= 1000).

Run from project root: python src/outcome.py
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from analysis import DataLoader, AnalysisConfig, EnsembleAnalyzer, ResultsExporter, TOPSISAnalyzer


def load_indicator_directions(input_dir: str = 'results-pipeline') -> dict:
    """
    Load indicator optimization directions from wskaznik_dictionary_minmax.csv.
    
    Args:
        input_dir: Directory containing the dictionary file
        
    Returns:
        Dictionary mapping WSKAZNIK names to 'max' or 'min'
    """
    wskaznik_minmax = pd.read_csv(
        os.path.join(input_dir, 'wskaznik_dictionary_minmax.csv'), 
        sep=';'
    )
    
    # Strip whitespace from column names
    wskaznik_minmax.columns = wskaznik_minmax.columns.str.strip()
    
    # Create mapping from WSKAZNIK name to direction
    direction_map = {}
    for _, row in wskaznik_minmax.iterrows():
        wskaznik_name = row['WSKAZNIK'].strip()
        minmax_value = str(row['MinMax']).strip().lower()
        
        # Normalize to 'max' or 'min'
        if minmax_value == 'max':
            direction_map[wskaznik_name] = 'max'
        elif minmax_value == 'min':
            direction_map[wskaznik_name] = 'min'
        else:
            # Default to max if unclear
            direction_map[wskaznik_name] = 'max'
    
    # Show distribution
    max_count = sum(1 for d in direction_map.values() if d == 'max')
    min_count = sum(1 for d in direction_map.values() if d == 'min')
    print(f"  Loaded directions: {max_count} maximize, {min_count} minimize")
    
    return direction_map


def calculate_percentage_changes(df_merged: pd.DataFrame, year: int, typ: str, min_wskaznik_index: int) -> pd.DataFrame:
    """
    Calculate percentage changes for the previous year and two years earlier.
    
    Args:
        df_merged: Merged dataframe with all data
        year: Current year to analyze
        typ: PKD type to filter
        min_wskaznik_index: Minimum indicator index
        
    Returns:
        DataFrame with percentage change data
    """
    print(f"\nCalculating percentage changes for years {year-2}, {year-1}, {year}...")
    
    # Get data for current year and two previous years
    df_years = df_merged[
        (df_merged['rok'].isin([year-2, year-1, year])) & 
        (df_merged['typ'] == typ) &
        (df_merged['WSKAZNIK_INDEX'] >= min_wskaznik_index)
    ].copy()
    
    # Pivot to get values by year
    pivot_data = df_years.pivot_table(
        index=['symbol', 'WSKAZNIK'],
        columns='rok',
        values='wartosc',
        aggfunc='first'
    ).reset_index()
    
    pct_changes = []
    
    # Calculate percentage change from year-1 to year (1-year change)
    if year in pivot_data.columns and (year-1) in pivot_data.columns:
        df_1yr = pivot_data[['symbol', 'WSKAZNIK', year, year-1]].copy()
        df_1yr = df_1yr.dropna(subset=[year, year-1])
        
        df_1yr['pct_change'] = ((df_1yr[year] - df_1yr[year-1]) / df_1yr[year-1].abs()) * 100
        df_1yr['cat_id'] = df_1yr['WSKAZNIK'] + ' (Δ% 1yr)'
        
        pct_changes.append(df_1yr[['symbol', 'cat_id', 'pct_change']].rename(columns={'pct_change': 'value'}))
        print(f"  ✓ Calculated 1-year changes: {len(df_1yr)} records")
    
    # Calculate percentage change from year-2 to year (2-year change)
    if year in pivot_data.columns and (year-2) in pivot_data.columns:
        df_2yr = pivot_data[['symbol', 'WSKAZNIK', year, year-2]].copy()
        df_2yr = df_2yr.dropna(subset=[year, year-2])
        
        df_2yr['pct_change'] = ((df_2yr[year] - df_2yr[year-2]) / df_2yr[year-2].abs()) * 100
        df_2yr['cat_id'] = df_2yr['WSKAZNIK'] + ' (Δ% 2yr)'
        
        pct_changes.append(df_2yr[['symbol', 'cat_id', 'pct_change']].rename(columns={'pct_change': 'value'}))
        print(f"  ✓ Calculated 2-year changes: {len(df_2yr)} records")
    
    if pct_changes:
        result = pd.concat(pct_changes, ignore_index=True)
        result.columns = ['kpi_id', 'cat_id', 'value']
        return result
    else:
        print("  ⚠ No historical data available for percentage changes")
        return pd.DataFrame(columns=['kpi_id', 'cat_id', 'value'])


def load_and_prepare_sector_data(year: int = 2024, typ: str = 'SEKCJA', min_wskaznik_index: int = 1000) -> pd.DataFrame:
    """
    Load and prepare data for sector analysis.
    
    Args:
        year: Year to filter (default: 2024)
        typ: PKD type to filter (default: 'SEKCJA')
        min_wskaznik_index: Minimum WSKAZNIK_INDEX to include (default: 1000)
        
    Returns:
        DataFrame in format ready for analysis (kpi_id, cat_id, value, direction)
    """
    # Define paths (relative to project root) - INPUT from results-pipeline
    input_dir = 'results-pipeline'
    
    print(f"Loading data for year {year}, type {typ}, indicators >= {min_wskaznik_index}...")
    
    # Read CSV files from results-pipeline
    kpi_value_table = pd.read_csv(os.path.join(input_dir, 'kpi-value-table.csv'), sep=';')
    pkd_dictionary = pd.read_csv(os.path.join(input_dir, 'pkd_dictionary.csv'), sep=';')
    pkd_typ_dictionary = pd.read_csv(os.path.join(input_dir, 'pkd_typ_dictionary.csv'), sep=';')
    wskaznik_dictionary = pd.read_csv(os.path.join(input_dir, 'wskaznik_dictionary.csv'), sep=';')
    
    # Load indicator directions from wskaznik_dictionary_minmax.csv
    indicator_directions = load_indicator_directions(input_dir)
    
    print(f"  Original data: {len(kpi_value_table)} rows")
    
    # Merge to get PKD type information
    pkd_enhanced = pkd_dictionary.merge(pkd_typ_dictionary, on='TYP_INDEX', how='left')
    
    # Merge kpi_value_table with PKD info
    df_merged = kpi_value_table.merge(
        pkd_enhanced[['PKD_INDEX', 'symbol', 'nazwa', 'typ']], 
        on='PKD_INDEX', 
        how='left'
    )
    
    # Merge with wskaznik dictionary to get indicator names
    df_merged = df_merged.merge(wskaznik_dictionary, on='WSKAZNIK_INDEX', how='left')
    
    # Filter by year, type, and indicator index for CURRENT YEAR data
    df_filtered = df_merged[
        (df_merged['rok'] == year) & 
        (df_merged['typ'] == typ) &
        (df_merged['WSKAZNIK_INDEX'] >= min_wskaznik_index)
    ].copy()
    
    print(f"  After filtering by year={year}, typ={typ}, WSKAZNIK_INDEX>={min_wskaznik_index}: {len(df_filtered)} rows")
    print(f"  Unique sectors: {df_filtered['symbol'].nunique()}")
    print(f"  Unique indicators: {df_filtered['WSKAZNIK'].nunique()}")
    
    if len(df_filtered) == 0:
        raise ValueError(f"No data found for year={year}, typ={typ}, WSKAZNIK_INDEX>={min_wskaznik_index}")
    
    # Create analysis-ready format for CURRENT YEAR values
    analysis_df = pd.DataFrame({
        'kpi_id': df_filtered['symbol'],  # Sectors (alternatives)
        'cat_id': df_filtered['WSKAZNIK'],  # Indicators (criteria)
        'value': df_filtered['wartosc'],
        'direction': 'max'  # Will be loaded from dictionary
    })
    
    # Calculate percentage changes and add as additional criteria
    pct_change_df = calculate_percentage_changes(df_merged, year, typ, min_wskaznik_index)
    
    if len(pct_change_df) > 0:
        # Add direction column to percentage changes (same as base indicator)
        pct_change_df['direction'] = 'max'  # Will be set based on base indicator
        
        # Combine current year values with percentage changes
        analysis_df = pd.concat([analysis_df, pct_change_df], ignore_index=True)
        print(f"\n  ✓ Added {len(pct_change_df)} percentage change records")
    
    # Apply directions from wskaznik_dictionary_minmax.csv
    # For percentage changes, use the same direction as the base indicator
    def get_direction(cat_id):
        # Extract base indicator name (remove " (Δ% 1yr)" or " (Δ% 2yr)")
        base_indicator = cat_id.replace(' (Δ% 1yr)', '').replace(' (Δ% 2yr)', '')
        direction = indicator_directions.get(base_indicator, 'max')
        
        # For percentage changes: 
        # - If base is "max", positive change is good → max
        # - If base is "min", positive change is bad → min (we want negative changes)
        if '(Δ%' in cat_id and direction == 'min':
            return 'min'  # Negative changes are good for minimization indicators
        else:
            return 'max'  # Positive changes are good for maximization indicators
    
    analysis_df['direction'] = analysis_df['cat_id'].apply(get_direction)
    
    # Fill any unmapped directions with 'max' as default
    analysis_df['direction'] = analysis_df['direction'].fillna('max')
    
    # Remove rows with missing values
    initial_rows = len(analysis_df)
    analysis_df = analysis_df.dropna(subset=['value'])
    removed_rows = initial_rows - len(analysis_df)
    
    if removed_rows > 0:
        print(f"  Removed {removed_rows} rows with missing values")
    
    # Remove infinite values
    initial_rows = len(analysis_df)
    analysis_df = analysis_df[~np.isinf(analysis_df['value'])]
    removed_inf = initial_rows - len(analysis_df)
    
    if removed_inf > 0:
        print(f"  Removed {removed_inf} rows with infinite values")
    
    print(f"\nFinal analysis dataset: {len(analysis_df)} rows")
    print(f"Alternatives (sectors): {analysis_df['kpi_id'].nunique()}")
    print(f"Criteria (indicators): {analysis_df['cat_id'].nunique()}")
    
    # Show sectors included
    print("\nSectors included in analysis:")
    sectors = df_filtered[['symbol', 'nazwa']].drop_duplicates().sort_values('symbol')
    for _, row in sectors.iterrows():
        print(f"  {row['symbol']}: {row['nazwa']}")
    
    # Show indicators included
    print("\nIndicators included in analysis:")
    indicators = analysis_df.groupby('cat_id').agg({
        'direction': 'first',
        'value': 'count'
    }).reset_index()
    indicators.columns = ['Indicator', 'Direction', 'Count']
    print(indicators.to_string(index=False))
    
    # Show summary statistics for each indicator
    print("\nIndicator statistics:")
    stats = analysis_df.groupby('cat_id')['value'].agg(['mean', 'std', 'min', 'max']).round(4)
    print(stats.to_string())
    
    return analysis_df


def save_results_by_year_type(results: pd.DataFrame, year: int, typ: str):
    """
    Save results organized by year and type.
    
    Args:
        results: Results DataFrame
        year: Year of analysis
        typ: PKD type
    """
    # Create directory structure: results/year/type/
    output_dir = Path('results') / str(year) / typ.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TOPSIS results
    topsis_df = results[['alternative_id', 'nazwa', 'topsis_score', 'topsis_rank']].copy()
    topsis_df = topsis_df.sort_values('topsis_rank')
    topsis_df.to_csv(output_dir / 'topsis.csv', index=False, sep=';', encoding='utf-8')
    
    # VIKOR results
    vikor_df = results[['alternative_id', 'nazwa', 'vikor_score', 'vikor_rank']].copy()
    vikor_df = vikor_df.sort_values('vikor_rank')
    vikor_df.to_csv(output_dir / 'vikor.csv', index=False, sep=';', encoding='utf-8')
    
    # Monte Carlo results
    mc_df = results[['alternative_id', 'nazwa', 'monte_carlo_score', 'monte_carlo_rank']].copy()
    mc_df = mc_df.sort_values('monte_carlo_rank')
    mc_df.to_csv(output_dir / 'monte_carlo.csv', index=False, sep=';', encoding='utf-8')
    
    # Ensemble results
    ensemble_df = results[['alternative_id', 'nazwa', 'ensemble_score', 'ensemble_rank']].copy()
    ensemble_df.to_csv(output_dir / 'ensemble.csv', index=False, sep=';', encoding='utf-8')
    
    # Complete results
    results.to_csv(output_dir / 'complete.csv', index=False, sep=';', encoding='utf-8')
    
    print(f"\n✓ Results saved to: {output_dir}/")
    print(f"  • topsis.csv")
    print(f"  • vikor.csv")
    print(f"  • monte_carlo.csv")
    print(f"  • ensemble.csv")
    print(f"  • complete.csv")
    
    return output_dir


def run_sector_analysis(year: int = 2024, 
                       typ: str = 'SEKCJA',
                       min_wskaznik_index: int = 1000,
                       n_simulations: int = 1000,
                       mc_weight_variance: float = 0.15,
                       temporal_weight_1yr: float = 0.66,
                       temporal_weight_2yr: float = 0.34) -> pd.DataFrame:
    """
    Run complete sector analysis.
    
    Args:
        year: Year to analyze
        typ: PKD type level
        min_wskaznik_index: Minimum indicator index to include (default: 1000 for calculated ratios)
        n_simulations: Number of Monte Carlo simulations
        mc_weight_variance: Variance for Monte Carlo weight perturbation (default: 0.15 = 15%)
        temporal_weight_1yr: Weight multiplier for 1-year changes (default: 0.66)
        temporal_weight_2yr: Weight multiplier for 2-year changes (default: 0.34)
        
    Returns:
        DataFrame with analysis results
    """
    # Load and prepare data
    analysis_df = load_and_prepare_sector_data(year, typ, min_wskaznik_index)
    
    # Create output directory
    output_dir = Path('results') / str(year) / typ.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save prepared input data
    analysis_df.to_csv(output_dir / 'input_data.csv', index=False, sep=';', encoding='utf-8')
    print(f"\n✓ Saved input data to: {output_dir / 'input_data.csv'}")
    
    # Prepare decision matrix
    try:
        matrix, directions = DataLoader.prepare_decision_matrix(analysis_df)
    except ValueError as e:
        print(f"\n⚠ Error preparing decision matrix: {e}")
        print("\nChecking for missing data by sector...")
        
        # Debug: show which sectors have which indicators
        pivot_check = analysis_df.pivot_table(
            index='kpi_id', 
            columns='cat_id', 
            values='value', 
            aggfunc='count'
        )
        print("\nData availability by sector (count of values):")
        print(pivot_check)
        
        # Show sectors with incomplete data
        incomplete = pivot_check[pivot_check.isnull().any(axis=1)]
        if len(incomplete) > 0:
            print("\n⚠ Sectors with missing indicators:")
            print(incomplete)
        
        raise
    
    # Calculate temporal weights (current year gets CV weight, changes get proportional weights)
    print(f"\nCalculating temporal weights:")
    print(f"  • Current year indicators: CV-based weight (w)")
    print(f"  • 1-year change indicators: {temporal_weight_1yr} × w")
    print(f"  • 2-year change indicators: {temporal_weight_2yr} × w")
    
    temporal_weights = DataLoader.calculate_temporal_weights(
        matrix, 
        weight_1yr=temporal_weight_1yr,
        weight_2yr=temporal_weight_2yr
    )
    
    # Configure analysis with temporal weights and MC variance
    config = AnalysisConfig(
        n_simulations=n_simulations,
        topsis_weight=0.33,
        vikor_weight=0.33,
        mc_weight=0.34,
        use_fuzzy_vikor=True,
        fuzzy_spread=0.1,
        vikor_v=0.5,
        mc_weight_variance=mc_weight_variance
    )
    
    # Run ensemble analysis with custom weights
    print("\n" + "="*90)
    print(f"RUNNING ENSEMBLE ANALYSIS - Year {year}, Type {typ}, Indicators >= {min_wskaznik_index}")
    print(f"Temporal weighting: 1yr={temporal_weight_1yr}, 2yr={temporal_weight_2yr}")
    print(f"Monte Carlo weight variance: ±{mc_weight_variance*100}%")
    print("="*90)
    
    # Create TOPSIS analyzer with temporal weights
    print("Running TOPSIS analysis with temporal weights...")
    topsis_analyzer = TOPSISAnalyzer(matrix, directions, weights=temporal_weights)
    topsis_scores = topsis_analyzer.analyze()
    
    # Print weight information
    print("\nCriterion weights (temporal weighting applied):")
    weights_info = topsis_analyzer.get_weights_info()
    print(weights_info.to_string(index=False))
    
    # Run VIKOR with same temporal weights
    if config.use_fuzzy_vikor:
        print(f"\nRunning Fuzzy VIKOR analysis (v={config.vikor_v}, spread=±{config.fuzzy_spread*100}%)...")
        from analysis import FuzzyVIKORAnalyzer
        vikor_analyzer = FuzzyVIKORAnalyzer(
            matrix, 
            directions,
            weights=temporal_weights,
            v=config.vikor_v,
            fuzzy_spread=config.fuzzy_spread
        )
        vikor_scores, vikor_details = vikor_analyzer.analyze()
    else:
        vikor_scores = pd.Series(0, index=matrix.index, name='vikor_score')
    
    # Run Monte Carlo with temporal weights as base
    print(f"\nRunning Monte Carlo simulation ({config.n_simulations} iterations, ±{config.mc_weight_variance*100}% variance)...")
    from analysis import MonteCarloAnalyzer
    mc_analyzer = MonteCarloAnalyzer(matrix, directions, config, base_weights=temporal_weights)
    mc_scores = mc_analyzer.simulate()
    
    # Combine scores using weighted average
    ensemble_scores = (
        config.topsis_weight * topsis_scores +
        config.vikor_weight * vikor_scores +
        config.mc_weight * mc_scores
    )
    
    # Create results DataFrame
    results = pd.DataFrame({
        'alternative_id': matrix.index,
        'topsis_score': topsis_scores.values,
        'vikor_score': vikor_scores.values,
        'monte_carlo_score': mc_scores.values,
        'ensemble_score': ensemble_scores.values,
        'topsis_rank': topsis_scores.rank(ascending=False),
        'vikor_rank': vikor_scores.rank(ascending=False),
        'monte_carlo_rank': mc_scores.rank(ascending=False),
        'ensemble_rank': ensemble_scores.rank(ascending=False)
    })
    
    results = results.sort_values('ensemble_rank')
    
    # Add sector names to results (from input directory)
    input_dir = 'results-pipeline'
    pkd_dictionary = pd.read_csv(os.path.join(input_dir, 'pkd_dictionary.csv'), sep=';')
    pkd_typ_dictionary = pd.read_csv(os.path.join(input_dir, 'pkd_typ_dictionary.csv'), sep=';')
    
    pkd_enhanced = pkd_dictionary.merge(pkd_typ_dictionary, on='TYP_INDEX', how='left')
    sector_names = pkd_enhanced[pkd_enhanced['typ'] == typ][['symbol', 'nazwa']]
    
    results = results.merge(sector_names, left_on='alternative_id', right_on='symbol', how='left')
    results = results.drop('symbol', axis=1)
    
    # Reorder columns for better readability
    cols = ['ensemble_rank', 'alternative_id', 'nazwa', 'ensemble_score', 
            'topsis_score', 'vikor_score', 'monte_carlo_score',
            'topsis_rank', 'vikor_rank', 'monte_carlo_rank']
    results = results[cols]
    
    # Export results using custom saver
    print("\n" + "="*90)
    print("TOP 10 SECTORS - ENSEMBLE RANKING")
    print("="*90)
    top_10 = results.head(10)
    print(top_10[['ensemble_rank', 'alternative_id', 'nazwa', 'ensemble_score']].to_string(index=False))
    print("="*90)
    
    # Save results
    save_results_by_year_type(results, year, typ)
    
    return results


if __name__ == '__main__':
    # Run sector analysis for year 2024, SEKCJA level, only calculated indicators (>= 1000)
    results = run_sector_analysis(
        year=2024,
        typ='DZIAŁ',
        min_wskaznik_index=1000,
        n_simulations=1000,
        mc_weight_variance=0.15,  # 15% variance around temporal weights
        temporal_weight_1yr=0.66,  # 1-year changes get 66% of base weight
        temporal_weight_2yr=0.34   # 2-year changes get 34% of base weight
    )
    
    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print("="*90)
    print("\nInput: results-pipeline/")
    print("Output: results/2024/sekcja/")
    print("\nFiles generated:")
    print("  • input_data.csv - Input data used for analysis")
    print("  • topsis.csv - TOPSIS rankings")
    print("  • vikor.csv - VIKOR rankings")
    print("  • monte_carlo.csv - Monte Carlo rankings")
    print("  • ensemble.csv - Combined ensemble rankings")
    print("  • complete.csv - Complete results with all scores and metadata")
    print("\nAnalysis includes:")
    print("  • Current year indicator values (1000-1007) - CV-based weights")
    print("  • 1-year percentage changes (Δ% 1yr) - 66% of base weight")
    print("  • 2-year percentage changes (Δ% 2yr) - 34% of base weight")
    print("  • All weights normalized to sum to 1.0")
    print("  • Directions loaded from wskaznik_dictionary_minmax.csv")
    print("\nWeighting methodology:")
    print("  • Base indicators: w = 1/CV (normalized)")
    print("  • 1-year changes: 0.66 × w")
    print("  • 2-year changes: 0.34 × w")
    print("  • Final normalization: all weights sum to 1.0")
    print("\nMonte Carlo method:")
    print("  • Uses temporal weights as base")
    print("  • Adds controlled randomization (±15% variance)")
    print("  • Tests robustness across weight perturbations")
    print("\nDirection logic for percentage changes:")
    print("  • Maximize indicators: positive change is good → max")
    print("  • Minimize indicators: negative change is good → min")
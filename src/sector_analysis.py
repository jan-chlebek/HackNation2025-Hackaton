"""
Sector Analysis using TOPSIS, VIKOR and Monte Carlo for SEKCJA level data.

This script loads data from kpi-value-table.csv, filters by year and sector type,
and performs multi-criteria decision analysis using only calculated indicators (ID >= 1000).

Run from project root: python src/sector_analysis.py
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from analysis import DataLoader, AnalysisConfig, EnsembleAnalyzer, ResultsExporter


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
    
    # Filter by year, type, and indicator index
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
    
    # Create analysis-ready format
    # Use PKD symbol as alternative (kpi_id) and WSKAZNIK as criteria (cat_id)
    analysis_df = pd.DataFrame({
        'kpi_id': df_filtered['symbol'],  # Sectors (alternatives)
        'cat_id': df_filtered['WSKAZNIK'],  # Indicators (criteria)
        'value': df_filtered['wartosc'],
        'direction': 'max'  # Default to max, will be customized per indicator
    })
    
    # Define direction for each calculated indicator (1000+)
    # All new calculated indicators are ratios/margins - context determines direction
    indicator_directions = {
        # Financial ratios - maximize
        'Marża netto (NP/PNPM)': 'max',
        'Marża operacyjna (OP/PNPM)': 'max',
        'Cash flow margin (CF/PNPM)': 'max',
        'Pokrycie odsetek (OP/IP)': 'max',
        'Rotacja należności (PNPM/REC)': 'max',
        
        # Liquidity ratios - maximize
        'Wskaźnik bieżącej płynności ((C+REC+INV)/STL)': 'max',
        'Wskaźnik szybki ((C+REC)/STL)': 'max',
        
        # Debt ratio - minimize (lower is better)
        'Wskaźnik zadłużenia ((STL+LTL)/PNPM)': 'min',
    }
    
    # Apply directions
    analysis_df['direction'] = analysis_df['cat_id'].map(indicator_directions)
    
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
                       n_simulations: int = 1000) -> pd.DataFrame:
    """
    Run complete sector analysis.
    
    Args:
        year: Year to analyze
        typ: PKD type level
        min_wskaznik_index: Minimum indicator index to include (default: 1000 for calculated ratios)
        n_simulations: Number of Monte Carlo simulations
        
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
    
    # Configure analysis
    config = AnalysisConfig(
        n_simulations=n_simulations,
        topsis_weight=0.33,
        vikor_weight=0.33,
        mc_weight=0.34,
        use_fuzzy_vikor=True,
        fuzzy_spread=0.1,
        vikor_v=0.5
    )
    
    # Run ensemble analysis
    print("\n" + "="*90)
    print(f"RUNNING ENSEMBLE ANALYSIS - Year {year}, Type {typ}, Indicators >= {min_wskaznik_index}")
    print("="*90)
    
    analyzer = EnsembleAnalyzer(matrix, directions, config)
    results = analyzer.analyze()
    
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
        typ='SEKCJA',
        min_wskaznik_index=1000,
        n_simulations=1000
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
    print("\nAnalysis uses only calculated financial ratios (indicators 1000-1007):")
    print("  1000: Marża netto (Net Margin)")
    print("  1001: Marża operacyjna (Operating Margin)")
    print("  1002: Wskaźnik bieżącej płynności (Current Ratio)")
    print("  1003: Wskaźnik szybki (Quick Ratio)")
    print("  1004: Wskaźnik zadłużenia (Debt Ratio)")
    print("  1005: Pokrycie odsetek (Interest Coverage)")
    print("  1006: Rotacja należności (Receivables Turnover)")
    print("  1007: Cash Flow Margin")
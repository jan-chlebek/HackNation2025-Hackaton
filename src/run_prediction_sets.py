"""
Run sector analysis for 3 indicator sets based on creating_complex_indicators.ipynb divisions:
- Zestaw 1: Credit Assessment (Zdolność kredytowa) - indicators 1000-1013
- Zestaw 2: Operational Efficiency (Efektywność operacyjna) - indicators 1020-1029
- Zestaw 3: Industry Development (Rozwój branży) - indicators 1040-1043

Each set saves results to a separate folder:
- results-credit/
- results-effectivity/
- results-development/

Run from project root: python src/run_prediction_sets.py
"""

from outcome import run_sector_analysis
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Tuple


def run_credit_analysis(year: int, typ: str):
    """
    Run analysis for Credit Assessment indicators (1000-1013).
    
    Indicators:
    - 1000: Net Profit Margin (Marża netto)
    - 1001: Operating Margin (Marża operacyjna)
    - 1002: Current Ratio (Płynność bieżąca)
    - 1003: Quick Ratio (Płynność szybka)
    - 1004: Cash Ratio (Płynność gotówkowa)
    - 1005: Short-term Debt Share
    - 1006: Long-term Debt Share
    - 1007: Interest Coverage Ratio
    - 1008: Financial Risk Ratio
    - 1009: Cash Flow Margin
    - 1010: Operating Cash Coverage
    - 1011: Bankruptcy Rate
    - 1012: Closure Rate
    - 1013: Profitable Firms Share
    """
    print("\n" + "="*90)
    print("ZESTAW 1: CREDIT ASSESSMENT (ZDOLNOŚĆ KREDYTOWA)")
    print("="*90)
    print("Indicators: 1000-1013 (14 indicators)")
    print("Output folder: results-credit/")
    print("="*90)
    
    return run_sector_analysis(
        year=year,
        typ=typ,
        min_wskaznik_index=1000,
        max_wskaznik_index=1013,
        n_simulations=1000,
        mc_weight_variance=0.15,
        temporal_weight_1yr=0.3,
        temporal_weight_2yr=0.1,
        output_folder='results-credit'
    )


def run_effectivity_analysis(year: int, typ: str):
    """
    Run analysis for Operational Efficiency indicators (1020-1029).
    
    Indicators:
    - 1020: Sales Profitability (Rentowność sprzedaży)
    - 1021: Core Revenue Share (Udział przychodów podstawowych)
    - 1022: Cost Share Ratio (Wskaźnik udziału kosztów)
    - 1023: Receivables Turnover (Rotacja należności)
    - 1024: Inventory Turnover (Rotacja zapasów)
    - 1025: Current Asset Turnover (Rotacja aktywów obrotowych)
    - 1026: Investment Ratio (Wskaźnik inwestycji)
    - 1027: Financial Revenue Share (Udział przychodów finansowych)
    - 1028: Net Firm Growth Rate
    - 1029: Average Firm Size
    """
    print("\n" + "="*90)
    print("ZESTAW 2: OPERATIONAL EFFICIENCY (EFEKTYWNOŚĆ OPERACYJNA)")
    print("="*90)
    print("Indicators: 1020-1029 (10 indicators)")
    print("Output folder: results-effectivity/")
    print("="*90)
    
    return run_sector_analysis(
        year=year,
        typ=typ,
        min_wskaznik_index=1020,
        max_wskaznik_index=1029,
        n_simulations=1000,
        mc_weight_variance=0.15,
        temporal_weight_1yr=0.3,
        temporal_weight_2yr=0.1,
        output_folder='results-effectivity'
    )


def run_development_analysis(year: int, typ: str):
    """
    Run analysis for Industry Development indicators (1040-1043).
    
    Indicators:
    - 1040: Amortization Ratio (Wskaźnik amortyzacji)
    - 1041: New Firms Rate (Wskaźnik nowych firm)
    - 1042: Suspension Rate (Wskaźnik zawieszeń)
    - 1043: Bank Loans Ratio (Wskaźnik kredytów bankowych)
    """
    print("\n" + "="*90)
    print("ZESTAW 3: INDUSTRY DEVELOPMENT (ROZWÓJ BRANŻY)")
    print("="*90)
    print("Indicators: 1040-1043 (4 indicators)")
    print("Output folder: results-development/")
    print("="*90)
    
    return run_sector_analysis(
        year=year,
        typ=typ,
        min_wskaznik_index=1040,
        max_wskaznik_index=1043,
        n_simulations=1000,
        mc_weight_variance=0.15,
        temporal_weight_1yr=0.3,
        temporal_weight_2yr=0.1,
        output_folder='results-development'
    )


def run_single_analysis(args: Tuple[str, int, str]) -> Tuple[str, int, str, bool]:
    """
    Worker function to run a single analysis task.
    
    Args:
        args: Tuple of (analysis_type, year, typ)
        
    Returns:
        Tuple of (analysis_type, year, typ, success)
    """
    analysis_type, year, typ = args
    try:
        if analysis_type == 'credit':
            run_credit_analysis(year, typ)
        elif analysis_type == 'effectivity':
            run_effectivity_analysis(year, typ)
        elif analysis_type == 'development':
            run_development_analysis(year, typ)
        return (analysis_type, year, typ, True)
    except Exception as e:
        print(f"\n⚠ Error in {analysis_type} analysis for {year}/{typ}: {e}")
        return (analysis_type, year, typ, False)


def run_all_sets(start_year: int = 2013, end_year: int = 2024, typ: str = 'SEKCJA', max_workers: int = None):
    """
    Run all 3 indicator sets for a range of years in parallel.
    
    Args:
        start_year: First year to analyze (default: 2013)
        end_year: Last year to analyze (default: 2024)
        typ: PKD type level - 'SEKCJA' or 'DZIAŁ' (default: 'SEKCJA')
        max_workers: Maximum number of parallel workers (default: CPU count - 1)
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    print("\n" + "="*90)
    print("RUNNING 3-SET ANALYSIS PIPELINE (PARALLEL)")
    print("="*90)
    print(f"Years: {start_year}-{end_year}")
    print(f"Type: {typ}")
    print(f"Workers: {max_workers}")
    print("\nSets:")
    print("  1. Credit Assessment (1000-1013) → results-credit/")
    print("  2. Operational Efficiency (1020-1029) → results-effectivity/")
    print("  3. Industry Development (1040-1043) → results-development/")
    print("="*90)
    
    # Create list of all tasks
    tasks = []
    for year in range(start_year, end_year + 1):
        tasks.append(('credit', year, typ))
        tasks.append(('effectivity', year, typ))
        tasks.append(('development', year, typ))
    
    total_tasks = len(tasks)
    completed = 0
    failed = 0
    
    print(f"\nTotal tasks: {total_tasks}")
    print("Starting parallel execution...\n")
    
    # Run tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_single_analysis, task): task for task in tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            analysis_type, year, typ_result, success = future.result()
            completed += 1
            
            if success:
                status = "✓"
            else:
                status = "✗"
                failed += 1
            
            progress = (completed / total_tasks) * 100
            print(f"[{progress:5.1f}%] {status} {year}/{typ_result}/{analysis_type:12s} ({completed}/{total_tasks})")
    
    print("\n" + "="*90)
    print("ALL SETS COMPLETED")
    print("="*90)
    print(f"Total: {total_tasks} | Successful: {completed - failed} | Failed: {failed}")
    print("\nOutput folders created:")
    print("  • results-credit/")
    print("  • results-effectivity/")
    print("  • results-development/")
    print("\nEach folder contains subdirectories by year and type:")
    print(f"  results-credit/YYYY/{typ.lower()}/")
    print(f"  results-effectivity/YYYY/{typ.lower()}/")
    print(f"  results-development/YYYY/{typ.lower()}/")


def run_all_sets_sequential(start_year: int = 2013, end_year: int = 2024, typ: str = 'SEKCJA'):
    """
    Run all 3 indicator sets for a range of years sequentially (original implementation).
    
    Args:
        start_year: First year to analyze (default: 2013)
        end_year: Last year to analyze (default: 2024)
        typ: PKD type level - 'SEKCJA' or 'DZIAŁ' (default: 'SEKCJA')
    """
    print("\n" + "="*90)
    print("RUNNING 3-SET ANALYSIS PIPELINE")
    print("="*90)
    print(f"Years: {start_year}-{end_year}")
    print(f"Type: {typ}")
    print("\nSets:")
    print("  1. Credit Assessment (1000-1013) → results-credit/")
    print("  2. Operational Efficiency (1020-1029) → results-effectivity/")
    print("  3. Industry Development (1040-1043) → results-development/")
    print("="*90)
    
    for year in range(start_year, end_year + 1):
        print(f"\n{'#'*90}")
        print(f"# YEAR: {year}")
        print(f"{'#'*90}")
        
        try:
            # Run all 3 sets for this year
            run_credit_analysis(year, typ)
            run_effectivity_analysis(year, typ)
            run_development_analysis(year, typ)
            
            print(f"\n✓ Completed all 3 sets for year {year}")
            
        except Exception as e:
            print(f"\n⚠ Error processing year {year}: {e}")
            continue
    
    print("\n" + "="*90)
    print("ALL SETS COMPLETED")
    print("="*90)
    print("\nOutput folders created:")
    print("  • results-credit/")
    print("  • results-effectivity/")
    print("  • results-development/")
    print("\nEach folder contains subdirectories by year and type:")
    print("  results-credit/YYYY/sekcja/")
    print("  results-effectivity/YYYY/sekcja/")
    print("  results-development/YYYY/sekcja/")


if __name__ == '__main__':
    # Run for all years and both levels in parallel
    
    # SEKCJA level (higher aggregation)
    print("\n" + "="*90)
    print("PROCESSING SEKCJA LEVEL")
    print("="*90)
    run_all_sets(start_year=2013, end_year=2024, typ='SEKCJA')
    
    # DZIAŁ level (more detailed)
    print("\n" + "="*90)
    print("PROCESSING DZIAŁ LEVEL")
    print("="*90)
    run_all_sets(start_year=2013, end_year=2024, typ='DZIAŁ')
    
    print("\n" + "="*90)
    print("PIPELINE COMPLETE!")
    print("="*90)

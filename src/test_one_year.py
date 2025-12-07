"""
Test script to run all 3 indicator sets for 2024 only.
"""

from run_prediction_sets import run_credit_analysis, run_effectivity_analysis, run_development_analysis

if __name__ == '__main__':
    year = 2024
    typ = 'SEKCJA'
    
    print(f"\n{'='*90}")
    print(f"TESTING 3-SET ANALYSIS FOR YEAR {year}, TYPE {typ}")
    print(f"{'='*90}\n")
    
    # Run all 3 sets
    print("\n[1/3] Running Credit Assessment...")
    run_credit_analysis(year=year, typ=typ)
    
    print("\n[2/3] Running Operational Efficiency...")
    run_effectivity_analysis(year=year, typ=typ)
    
    print("\n[3/3] Running Industry Development...")
    run_development_analysis(year=year, typ=typ)
    
    print(f"\n{'='*90}")
    print("TEST COMPLETED SUCCESSFULLY!")
    print(f"{'='*90}")
    print("\nCheck the following folders for results:")
    print(f"  • results-credit/{year}/{typ.lower()}/")
    print(f"  • results-effectivity/{year}/{typ.lower()}/")
    print(f"  • results-development/{year}/{typ.lower()}/")


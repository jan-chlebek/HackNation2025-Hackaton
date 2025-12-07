"""
Test script to verify parallel execution works with 2 years.
"""

from run_prediction_sets import run_all_sets

if __name__ == '__main__':
    print("Testing parallel execution with 2023-2024, SEKCJA level...")
    run_all_sets(start_year=2023, end_year=2024, typ='SEKCJA', max_workers=4)
    print("\nParallel test completed!")

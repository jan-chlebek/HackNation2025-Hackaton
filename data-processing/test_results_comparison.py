import pandas as pd
import os
from pandas.testing import assert_frame_equal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
PIPELINE_DIR = os.path.join(BASE_DIR, "..", "results-pipeline")

FILES_TO_COMPARE = [
    "kpi-value-table.csv",
    "pkd_dictionary.csv",
    "pkd_typ_dictionary.csv",
    "wskaznik_dictionary.csv"
]

def compare_files():
    print(f"Comparing files between:\n{RESULTS_DIR}\n{PIPELINE_DIR}\n")
    
    all_passed = True
    
    for filename in FILES_TO_COMPARE:
        file_path_1 = os.path.join(RESULTS_DIR, filename)
        file_path_2 = os.path.join(PIPELINE_DIR, filename)
        
        if not os.path.exists(file_path_1):
            print(f"❌ {filename}: Missing in results")
            all_passed = False
            continue
            
        if not os.path.exists(file_path_2):
            print(f"❌ {filename}: Missing in results-pipeline")
            all_passed = False
            continue
            
        try:
            # Read CSVs
            # Using sep=";" as seen in the notebook code
            df1 = pd.read_csv(file_path_1, sep=";")
            df2 = pd.read_csv(file_path_2, sep=";")
            
            # Sort by columns to ensure order doesn't matter for columns
            df1 = df1.reindex(sorted(df1.columns), axis=1)
            df2 = df2.reindex(sorted(df2.columns), axis=1)
            
            # Sort values to ensure row order doesn't matter
            # We try to sort by all columns
            df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
            df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

            # Compare
            assert_frame_equal(df1, df2, check_dtype=False, check_like=True)
            print(f"✅ {filename}: Match")
            
        except AssertionError as e:
            print(f"❌ {filename}: Mismatch")
            # Print a short summary of the difference
            print(e)
            all_passed = False
        except Exception as e:
            print(f"❌ {filename}: Error comparing - {e}")
            all_passed = False

    if all_passed:
        print("\nAll files match successfully!")
    else:
        print("\nSome files do not match.")

if __name__ == "__main__":
    compare_files()

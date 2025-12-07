import os
import pandas as pd

# Define years range
years = range(2013, 2029)
results_dir = "results"
output_file_sekcja = "dashboard/public/data/trends_sekcja.csv"
output_file_dzial = "dashboard/public/data/trends_dzial.csv"

def aggregate_data(level_name, output_path):
    all_data = []
    for year in years:
        # Handle 'dział' vs 'sekcja' folder naming if needed
        # Based on file system check: 'dział' and 'sekcja'
        folder_name = "dział" if level_name == "dzial" else level_name
        
        file_path = os.path.join(results_dir, str(year), folder_name, "complete.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep=';')
                df_subset = df[['alternative_id', 'nazwa', 'ensemble_score']].copy()
                
                if level_name == "dzial":
                    # Pad with leading zero if it's a single digit number
                    df_subset['alternative_id'] = df_subset['alternative_id'].apply(
                        lambda x: f"{int(x):02d}" if pd.notnull(x) and str(x).replace('.','',1).isdigit() else x
                    )
                
                df_subset['year'] = year
                all_data.append(df_subset)
                print(f"Processed {year} for {level_name}")
            except Exception as e:
                print(f"Error processing {year} for {level_name}: {e}")
        else:
            print(f"File not found for {year} {level_name}: {file_path}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False, sep=';')
        print(f"Successfully saved aggregated trends to {output_path}")
    else:
        print(f"No data found to aggregate for {level_name}.")

aggregate_data("sekcja", output_file_sekcja)
aggregate_data("dzial", output_file_dzial)

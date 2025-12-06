"""
Generate mock data for testing the TOPSIS and Monte Carlo analysis.

This creates a CSV file with sample alternatives and criteria for testing.
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define mock scenarios
scenarios = {
    'Financial Sectors': {
        'alternatives': ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 
                        'Retail', 'Energy', 'Real Estate', 'Telecom', 
                        'Consumer Goods', 'Utilities'],
        'criteria': {
            'profit_margin': {'direction': 'max', 'range': (5, 25)},
            'revenue_growth': {'direction': 'max', 'range': (-5, 30)},
            'debt_ratio': {'direction': 'min', 'range': (0.2, 1.5)},
            'bankruptcy_risk': {'direction': 'min', 'range': (0, 100)},
            'cash_flow': {'direction': 'max', 'range': (10, 50)},
            'market_share': {'direction': 'max', 'range': (5, 40)},
        }
    },
    'Investment Projects': {
        'alternatives': [f'Project_{i}' for i in range(1, 16)],
        'criteria': {
            'roi': {'direction': 'max', 'range': (5, 35)},
            'risk_score': {'direction': 'min', 'range': (1, 10)},
            'implementation_time': {'direction': 'min', 'range': (3, 24)},
            'npv': {'direction': 'max', 'range': (100, 5000)},
            'strategic_fit': {'direction': 'max', 'range': (1, 10)},
        }
    },
    'Supplier Selection': {
        'alternatives': [f'Supplier_{chr(65+i)}' for i in range(12)],
        'criteria': {
            'price': {'direction': 'min', 'range': (500, 2000)},
            'quality_score': {'direction': 'max', 'range': (60, 100)},
            'delivery_time': {'direction': 'min', 'range': (1, 14)},
            'reliability': {'direction': 'max', 'range': (70, 100)},
            'flexibility': {'direction': 'max', 'range': (50, 95)},
            'sustainability': {'direction': 'max', 'range': (40, 90)},
        }
    }
}


def generate_mock_data(scenario_name='Financial Sectors', noise_level=0.1):
    """
    Generate mock data for testing.
    
    Args:
        scenario_name: Name of the scenario to generate
        noise_level: Amount of random variation (0-1)
    
    Returns:
        DataFrame in long format ready for analysis
    """
    scenario = scenarios[scenario_name]
    alternatives = scenario['alternatives']
    criteria = scenario['criteria']
    
    data_records = []
    
    for alt in alternatives:
        for crit_name, crit_info in criteria.items():
            # Generate base value within range
            min_val, max_val = crit_info['range']
            base_value = np.random.uniform(min_val, max_val)
            
            # Add some noise
            noise = np.random.normal(0, (max_val - min_val) * noise_level)
            value = np.clip(base_value + noise, min_val, max_val)
            
            data_records.append({
                'kpi_id': alt,
                'cat_id': crit_name,
                'value': round(value, 2),
                'direction': crit_info['direction']
            })
    
    return pd.DataFrame(data_records)


def add_realistic_correlations(df):
    """
    Add realistic correlations between criteria.
    For example: high profit margin often correlates with lower bankruptcy risk.
    """
    # This is a simplified version - just adds some patterns
    alternatives = df['kpi_id'].unique()
    
    for alt in alternatives:
        alt_data = df[df['kpi_id'] == alt]
        
        # If profit margin is high, reduce bankruptcy risk
        profit_row = alt_data[alt_data['cat_id'] == 'profit_margin']
        if not profit_row.empty:
            profit_val = profit_row['value'].values[0]
            
            # Update bankruptcy risk inversely
            bankruptcy_idx = df[(df['kpi_id'] == alt) & (df['cat_id'] == 'bankruptcy_risk')].index
            if len(bankruptcy_idx) > 0:
                # Higher profit = lower bankruptcy risk
                adjustment = (profit_val - 15) * -2  # -2 factor for inverse correlation
                current_val = df.loc[bankruptcy_idx[0], 'value']
                df.loc[bankruptcy_idx[0], 'value'] = np.clip(current_val + adjustment, 0, 100)
    
    return df


def create_test_files():
    """Create multiple test files for different scenarios."""
    
    # Scenario 1: Financial Sectors (default test file)
    print("Generating mockdata/data.csv (Financial Sectors)...")
    df1 = generate_mock_data('Financial Sectors', noise_level=0.15)
    df1 = add_realistic_correlations(df1)
    df1.to_csv('mockdata/data.csv', index=False)
    print(f"✓ Created mockdata/data.csv: {len(df1['kpi_id'].unique())} alternatives, {len(df1['cat_id'].unique())} criteria")
    
    # Scenario 2: Investment Projects
    print("\nGenerating data_projects.csv (Investment Projects)...")
    df2 = generate_mock_data('Investment Projects', noise_level=0.12)
    df2.to_csv('mockdata/dataprojects.csv', index=False)
    print(f"✓ Created mockdata/dataprojects.csv: {len(df2['kpi_id'].unique())} alternatives, {len(df2['cat_id'].unique())} criteria")
    
    # Scenario 3: Supplier Selection
    print("\nGenerating mockdata/datasuppliers.csv (Supplier Selection)...")
    df3 = generate_mock_data('Supplier Selection', noise_level=0.08)
    df3.to_csv('mockdata/datasuppliers.csv', index=False)
    print(f"✓ Created mockdata/datasuppliers.csv: {len(df3['kpi_id'].unique())} alternatives, {len(df3['cat_id'].unique())} criteria")
    
    # Create a small test file for quick testing
    print("\nGenerating mockdata/datasmall.csv (Small test set)...")
    df_small = df1[df1['kpi_id'].isin(['Technology', 'Healthcare', 'Finance', 'Manufacturing'])].copy()
    df_small.to_csv('mockdata/datasmall.csv', index=False)
    print(f"✓ Created mockdata/datasmall.csv: {len(df_small['kpi_id'].unique())} alternatives (for quick testing)")
    
    # Print sample data
    print("\n" + "="*80)
    print("SAMPLE DATA (mockdata/data.csv - first 20 rows):")
    print("="*80)
    print(df1.head(20).to_string(index=False))
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    summary = df1.groupby('cat_id').agg({
        'value': ['mean', 'std', 'min', 'max'],
        'direction': 'first'
    }).round(2)
    print(summary)
    
    return df1


if __name__ == '__main__':
    create_test_files()
    
    print("\n" + "="*80)
    print("TEST FILES READY!")
    print("="*80)
    print("\nYou can now test analysis.py with:")
    print("  python src/analysis.py")
    print("\nOr use different test files:")
    print("  - data.csv (Financial Sectors - default)")
    print("  - mockdata/data_projects.csv (Investment Projects)")
    print("  - mockdata/data_suppliers.csv (Supplier Selection)")
    print("  - mockdata/data_small.csv (Quick test - 4 alternatives)")
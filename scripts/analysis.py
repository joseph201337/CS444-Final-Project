import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = 'data/weatherAUS.csv'

def analyze_missing_values(df):
    """Comprehensive missing value analysis"""
    print("="*80)
    print("MISSING VALUES ANALYSIS")
    print("="*80)
    
    # Calculate missing statistics
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percent': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data_Type': df.dtypes.values
    })
    
    # Sort by missing percentage
    missing_stats = missing_stats.sort_values('Missing_Percent', ascending=False)
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0]
    
    print(f"\nTotal Rows: {len(df)}")
    print(f"Columns with Missing Values: {len(missing_stats)}/{len(df.columns)}")
    print("\n" + missing_stats.to_string(index=False))
    
    return missing_stats

def analyze_numerical_columns(df):
    """Statistical summary of numerical columns"""
    print("\n" + "="*80)
    print("NUMERICAL COLUMNS STATISTICS")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = pd.DataFrame({
        'Column': numeric_cols,
        'Mean': df[numeric_cols].mean().values,
        'Median': df[numeric_cols].median().values,
        'Std': df[numeric_cols].std().values,
        'Min': df[numeric_cols].min().values,
        'Max': df[numeric_cols].max().values,
        'Q25': df[numeric_cols].quantile(0.25).values,
        'Q75': df[numeric_cols].quantile(0.75).values
    })
    
    # Format for readability
    for col in ['Mean', 'Median', 'Std', 'Min', 'Max', 'Q25', 'Q75']:
        stats[col] = stats[col].round(2)
    
    print("\n" + stats.to_string(index=False))
    
    return stats

def analyze_categorical_columns(df):
    """Analysis of categorical columns"""
    print("\n" + "="*80)
    print("CATEGORICAL COLUMNS ANALYSIS")
    print("="*80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        print(f"\n--- {col} ---")
        value_counts = df[col].value_counts()
        null_count = df[col].isnull().sum()
        
        print(f"Unique Values: {df[col].nunique()}")
        print(f"Missing Values: {null_count} ({null_count/len(df)*100:.2f}%)")
        
        if df[col].nunique() <= 20:  # Show distribution for columns with few categories
            print(f"\nValue Distribution:")
            for val, count in value_counts.head(10).items():
                print(f"  {val}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print(f"\nTop 10 Values:")
            for val, count in value_counts.head(10).items():
                print(f"  {val}: {count} ({count/len(df)*100:.2f}%)")

def analyze_target_variable(df, target='RainTomorrow'):
    """Analyze target variable distribution"""
    print("\n" + "="*80)
    print(f"TARGET VARIABLE ANALYSIS: {target}")
    print("="*80)
    
    if target in df.columns:
        print(f"\nClass Distribution:")
        value_counts = df[target].value_counts()
        for val, count in value_counts.items():
            print(f"  {val}: {count} ({count/len(df)*100:.2f}%)")
        
        # Class imbalance ratio
        if len(value_counts) == 2:
            imbalance_ratio = value_counts.max() / value_counts.min()
            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
    else:
        print(f"\nTarget column '{target}' not found in dataset")

def analyze_location_distribution(df):
    """Analyze data distribution across locations"""
    print("\n" + "="*80)
    print("LOCATION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    if 'Location' in df.columns:
        location_counts = df['Location'].value_counts()
        print(f"\nTotal Locations: {df['Location'].nunique()}")
        print(f"\nSamples per Location:")
        print(location_counts.to_string())
        
        print(f"\nLocation Statistics:")
        print(f"  Min samples: {location_counts.min()} ({location_counts.idxmin()})")
        print(f"  Max samples: {location_counts.max()} ({location_counts.idxmax()})")
        print(f"  Mean samples: {location_counts.mean():.2f}")
        print(f"  Median samples: {location_counts.median():.2f}")

def analyze_missing_patterns(df):
    """Analyze patterns in missing data"""
    print("\n" + "="*80)
    print("MISSING DATA PATTERNS")
    print("="*80)
    
    # Find rows with most missing values
    rows_missing = df.isnull().sum(axis=1)
    print(f"\nRows with no missing values: {(rows_missing == 0).sum()} ({(rows_missing == 0).sum()/len(df)*100:.2f}%)")
    print(f"Rows with 1-5 missing values: {((rows_missing >= 1) & (rows_missing <= 5)).sum()}")
    print(f"Rows with 6-10 missing values: {((rows_missing >= 6) & (rows_missing <= 10)).sum()}")
    print(f"Rows with >10 missing values: {(rows_missing > 10).sum()}")
    
    # Check for columns that are often missing together
    print("\n\nCorrelation between missing values (top pairs):")
    missing_df = df.isnull().astype(int)
    corr_matrix = missing_df.corr()
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs with high correlation
    high_corr = []
    for column in upper.columns:
        for index in upper.index:
            if abs(upper.loc[index, column]) > 0.3:  # Threshold for correlation
                high_corr.append((index, column, upper.loc[index, column]))
    
    if high_corr:
        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, corr_val in high_corr[:10]:
            print(f"  {col1} <-> {col2}: {corr_val:.3f}")
    else:
        print("  No strong correlations found")

def generate_summary_report(df):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())
    
    print(f"\nDate Range:")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"  Start: {df['Date'].min()}")
        print(f"  End: {df['Date'].max()}")
        print(f"  Duration: {(df['Date'].max() - df['Date'].min()).days} days")

def save_analysis_to_file(df, output_file='data_analysis_report.txt'):
    """Save all analysis to a text file"""
    import sys
    from io import StringIO
    
    # Redirect stdout to capture all prints
    old_stdout = sys.stdout
    sys.stdout = result = StringIO()
    
    # Run all analyses
    generate_summary_report(df)
    missing_stats = analyze_missing_values(df)
    num_stats = analyze_numerical_columns(df)
    analyze_categorical_columns(df)
    analyze_target_variable(df)
    analyze_location_distribution(df)
    analyze_missing_patterns(df)
    
    # Get the output
    output = result.getvalue()
    sys.stdout = old_stdout
    
    # Print to console
    print(output)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(output)
    
    print(f"\n{'='*80}")
    print(f"Analysis saved to: {output_file}")
    print(f"{'='*80}")
    
    return missing_stats, num_stats

def create_visualization_summary(df, output_dir='analysis_plots'):
    """Create visualization plots for key insights"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Missing values heatmap
    plt.figure(figsize=(12, 8))
    missing_data = df.isnull()
    plt.imshow(missing_data.iloc[:1000].T, cmap='RdYlGn_r', aspect='auto', interpolation='none')
    plt.colorbar(label='Missing (1) vs Present (0)')
    plt.xlabel('Sample Index (first 1000 rows)')
    plt.ylabel('Features')
    plt.title('Missing Data Pattern (First 1000 Samples)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/missing_data_pattern.png', dpi=150)
    print(f"Saved: {output_dir}/missing_data_pattern.png")
    plt.close()
    
    # 2. Missing values bar chart
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if len(missing_counts) > 0:
        plt.figure(figsize=(12, 6))
        missing_counts.plot(kind='bar', color='coral')
        plt.title('Missing Values Count by Column')
        plt.xlabel('Column')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/missing_values_bar.png', dpi=150)
        print(f"Saved: {output_dir}/missing_values_bar.png")
        plt.close()
    
    # 3. Target distribution
    if 'RainTomorrow' in df.columns:
        plt.figure(figsize=(8, 6))
        df['RainTomorrow'].value_counts().plot(kind='bar', color=['skyblue', 'coral'])
        plt.title('Target Variable Distribution (RainTomorrow)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_distribution.png', dpi=150)
        print(f"Saved: {output_dir}/target_distribution.png")
        plt.close()

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Run comprehensive analysis
    missing_stats, num_stats = save_analysis_to_file(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualization_summary(df)
    
    print("\nAnalysis complete!")
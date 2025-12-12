import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler  
import torch

# Configuration
SEQ_LENGTH = 14
TARGET = 'RainTomorrow'
FEATURES = [
    "Month_Sin", "Month_Cos", "MinTemp", "MaxTemp", "Rainfall",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm", "RainToday_Enc",
    "WindGustDir_Sin", "WindGustDir_Cos", "WindDir9am_Sin", "WindDir9am_Cos", 
    "WindDir3pm_Sin", "WindDir3pm_Cos"
]

def load_and_clean_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Drop crucial missing values (Target)
    df = df.dropna(subset=[TARGET])
    df = df.dropna(subset=["Pressure9am", "Pressure3pm"])
    
    # Date conversion & Sorting
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Location', 'Date'])

    # Sqrt-transform Rainfall to squash outliers 
    df['Rainfall'] = np.sqrt(df['Rainfall'])

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feats_to_impute = [c for c in numeric_cols if c not in ['RainTomorrow', 'RainTomorrow_Numeric']]
    
    for col in feats_to_impute:
        df[col] = df[col].fillna(df.groupby('Location')[col].transform('median'))
        df[col] = df[col].fillna(df[col].median())

    # Encode season
    df['Month'] = df['Date'].dt.month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df.drop('Month', axis=1, inplace=True)
    
    # Handle RainToday/Tomorrow
    df['RainToday'] = df['RainToday'].fillna('No')
    df['RainToday_Enc'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    
    # Keep target as numeric
    target_map = {'Yes': 1, 'No': 0}
    df['RainTomorrow_Numeric'] = df['RainTomorrow'].map(target_map)
    
    # Encode Location
    location_dummies = pd.get_dummies(df['Location'], prefix='Loc')
    df = pd.concat([df, location_dummies], axis=1)    

    # Encode Wind Directions (cyclical)
    wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    direction_mapping = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }

    for col in wind_cols:
        most_common = df[col].mode()[0] if not df[col].mode().empty else 'N'
        df[col] = df[col].fillna(most_common)
        
        df[f'{col}_Degrees'] = df[col].map(direction_mapping)
        radians = np.radians(df[f'{col}_Degrees'])
        df[f'{col}_Sin'] = np.sin(radians)
        df[f'{col}_Cos'] = np.cos(radians)
        df.drop(f'{col}_Degrees', axis=1, inplace=True)
        
    # Fill any remaining NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['RainTomorrow_Numeric']]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

def create_sequences_fixed(df, seq_length=14, train_ratio=0.8, features_to_use=None):
    """
    Fixed sequence creator that properly splits BEFORE creating sequences
    """
    print("Creating sequences with proper train/test split...")
    
    if features_to_use is None:
        features_to_use = FEATURES

    # Identify feature columns
    feature_cols = [f for f in features_to_use if f in df.columns]
    
    train_dfs = []
    test_dfs = []
    
    for location in df['Location'].unique():
        loc_df = df[df['Location'] == location].copy()
        n = len(loc_df)
        split_idx = int(n * train_ratio)
        
        train_dfs.append(loc_df.iloc[:split_idx])
        test_dfs.append(loc_df.iloc[split_idx:])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Scaling 
    scaler = RobustScaler()  
    scaler.fit(train_df[feature_cols])
    
    def build_arrays(data_df):
        groups = data_df.groupby('Location')
        valid_indices = []
        
        # Transform to float32
        data_matrix = scaler.transform(data_df[feature_cols]).astype(np.float32)
        target_array = data_df['RainTomorrow_Numeric'].values.astype(np.float32)
        dates_array = data_df['Date'].values
        
        for _, indices in groups.indices.items():
            indices = np.sort(indices)
            group_dates = dates_array[indices]
            
            for i in range(len(indices) - seq_length):
                start_date = group_dates[i]
                end_date = group_dates[i + seq_length]
                days_gap = (end_date - start_date).astype('timedelta64[D]').astype(int)
                
                if days_gap == seq_length:
                    valid_indices.append(indices[i])

        num_samples = len(valid_indices)
        print(f"  - Found {num_samples} valid sequences")
        
        num_features = len(feature_cols)
        X = np.empty((num_samples, seq_length, num_features), dtype=np.float32)
        y = np.empty((num_samples,), dtype=np.float32)
        
        for i, start_idx in enumerate(valid_indices):
            X[i] = data_matrix[start_idx : start_idx + seq_length]
            y[i] = target_array[start_idx + seq_length - 1]            
        return X, y

    print("Building Train set...")
    X_train, y_train = build_arrays(train_df)
    
    print("Building Test set...")
    X_test, y_test = build_arrays(test_df)
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    return X_train, y_train, X_test, y_test, scaler, feature_cols

def get_data_loaders(filepath, batch_size=64, region_filter=None):
    """Returns data loaders with SMOTE applied """
    df = load_and_clean_data(filepath)
    
    # Add location features to FEATURES list
    location_features = [col for col in df.columns if col.startswith('Loc_')]
    feature_list = FEATURES.copy()
    feature_list.extend(location_features)
    
    print(f"Total features: {len(feature_list)}")
    
    if region_filter:
        df = df[df['Location'].isin(region_filter)]
        print(f"Filtered for regions: {region_filter}")

    # Create sequences with proper splitting
    X_train, y_train, X_test, y_test, scaler, features = create_sequences_fixed(
        df, SEQ_LENGTH, train_ratio=0.8, features_to_use=feature_list
    )
    
    
    # Create datasets
    train_data = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), 
        torch.from_numpy(y_train)
    )
    test_data = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), 
        torch.from_numpy(y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=batch_size, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=batch_size, num_workers=0
    )
    
    return train_loader, test_loader, scaler, features
import pandas as pd
import numpy as np
from typing import List, Tuple, Any, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Feature Extraction & Creation
def extract_date_features(df: pd.DataFrame, column: str, features: List[str]) -> pd.DataFrame:
    df_feat = df.copy()
    if column not in df_feat.columns:
        return df_feat
        
    if not pd.api.types.is_datetime64_any_dtype(df_feat[column]):
        try:
            df_feat[column] = pd.to_datetime(df_feat[column], errors='coerce')
        except Exception:
            return df_feat 

    for feat in features:
        new_col_name = f"{column}_{feat}"
        try:
            if feat == 'year': df_feat[new_col_name] = df_feat[column].dt.year
            elif feat == 'month': df_feat[new_col_name] = df_feat[column].dt.month
            elif feat == 'day': df_feat[new_col_name] = df_feat[column].dt.day
            elif feat == 'weekday': df_feat[new_col_name] = df_feat[column].dt.day_name()
            elif feat == 'quarter': df_feat[new_col_name] = df_feat[column].dt.quarter
        except Exception:
            continue
    return df_feat

data = {
        'Age': [20, 30, 40, 100], 
        'Salary': [50000, 60000, 80000, 150000],
        'City': ['NY', 'LA', 'NY', 'Chicago'],
        'Join_Date': pd.to_datetime(['2020-01-01', '2019-05-20', '2021-08-10', '2022-01-01']),
        'Target': [0, 1, 0, 1]
    }
df = pd.DataFrame(data)

def create_interaction_features(df: pd.DataFrame, col1: str, col2: str, operation: str) -> pd.DataFrame:
    df_feat = df.copy()
    if col1 not in df.columns or col2 not in df.columns:
        return df_feat
        
    new_col_name = f"{col1}_{operation}_{col2}"
    
    try:
        s1 = pd.to_numeric(df_feat[col1], errors='coerce')
        s2 = pd.to_numeric(df_feat[col2], errors='coerce')

        if operation == 'add': df_feat[new_col_name] = s1 + s2
        elif operation == 'subtract': df_feat[new_col_name] = s1 - s2
        elif operation == 'multiply': df_feat[new_col_name] = s1 * s2
        elif operation == 'divide': df_feat[new_col_name] = s1 / s2.replace(0, np.nan)
    except Exception as e:
        print(f"Interaction error: {e}")
        
    return df_feat

# 2. Feature Encoding
def one_hot_encode(df: pd.DataFrame, columns: List[str], drop_first: bool = False, fitted_encoder: Any = None) -> Tuple[pd.DataFrame, Any]:
    df_feat = df.copy()
    valid_cols = [c for c in columns if c in df_feat.columns]
    
    if not valid_cols:
        return df_feat, fitted_encoder

    try:
        if fitted_encoder is None:
            if drop_first:
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore', dtype=int)
            else:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
            
            encoded_array = encoder.fit_transform(df_feat[valid_cols].astype(str))
        else:
            encoder = fitted_encoder
            encoded_array = encoder.transform(df_feat[valid_cols].astype(str))
    
        try:
            feature_names = encoder.get_feature_names_out(valid_cols)
        except AttributeError:
            feature_names = encoder.get_feature_names(valid_cols)
            
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_feat.index).astype(int)
        
        df_feat = pd.concat([df_feat.drop(columns=valid_cols), encoded_df], axis=1)
        
    except Exception as e:
        print(f"One-Hot Encoding error: {e}")
        return df_feat, fitted_encoder
        
    return df_feat, encoder


def label_encode(df: pd.DataFrame, columns: List[str], fitted_encoder: Any = None) -> Tuple[pd.DataFrame, Any]:
    df_feat = df.copy()
    valid_cols = [c for c in columns if c in df_feat.columns]
    
    if not valid_cols:
        return df_feat, fitted_encoder

    try:
        if fitted_encoder is None:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoder.fit(df_feat[valid_cols])
        else:
            encoder = fitted_encoder

        encoded_data = encoder.transform(df_feat[valid_cols])
        
        df_encoded = pd.DataFrame(encoded_data, columns=valid_cols, index=df_feat.index)
        df_feat[valid_cols] = df_encoded.astype("int64")

    except Exception as e:
        print(f"Label encoding error: {e}")
        return df_feat, fitted_encoder

    return df_feat, encoder

# 3. Data Splitting
def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42, stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    stratify_param = y if stratify else None
    
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)
    except ValueError as e:
        print(f"Warning: Stratified split failed. Falling back to random split. Error: {e}")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
        
# 4. Feature Scaling
def scale_numerical_columns(df: pd.DataFrame, columns: List[str], method: str = 'standard', fitted_scaler: Any = None) -> Tuple[pd.DataFrame, Any]:
    df_feat = df.copy()
    valid_cols = [c for c in columns if c in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[c])]
    
    if not valid_cols:
        return df_feat, fitted_scaler
    
    # 1. Select Scaler
    if fitted_scaler is None:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
            
        try:
            scaled_data = scaler.fit_transform(df_feat[valid_cols])
        except Exception as e:
            print(f"Scaling fit error: {e}")
            return df_feat, None
    else:
        scaler = fitted_scaler
        try:
            scaled_data = scaler.transform(df_feat[valid_cols])
        except Exception as e:
            print(f"Scaling transform error: {e}")
            return df_feat, scaler

    df_feat[valid_cols] = pd.DataFrame(scaled_data, columns=valid_cols, index=df_feat.index)
    return df_feat, scaler


# Test
if __name__ == '__main__':
    print("--- QA Test for dataforge.preprocessing ---")
    

    data = {
        'Age': [20, 30, 40, 100], 
        'Salary': [50000, 60000, 80000, 150000],
        'City': ['NY', 'LA', 'NY', 'Chicago'],
        'Join_Date': pd.to_datetime(['2020-01-01', '2019-05-20', '2021-08-10', '2022-01-01']),
        'Target': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    print("\n1. Date Extraction (Creation)...")
    df_dates = extract_date_features(df, 'Join_Date', ['year', 'month'])
    print("   New Columns:", [c for c in df_dates.columns if 'Join_Date' in c])

    print("\n2. Encoding...")
    df_encoded, _ = one_hot_encode(df_dates, ['City'])
    print("   Encoded Columns:", df_encoded.columns.tolist())
    
    print("\n3. Splitting Data...")
    X_train, X_test, y_train, y_test = split_data(df_encoded, 'Target', test_size=0.5)
    print(f"   Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    print("\n4. Scaling (Training Step)...")
    X_train_scaled, scaler = scale_numerical_columns(X_train, ['Age', 'Salary'], 'robust')
    print("   Train Age (Scaled):\n", X_train_scaled['Age'].values)
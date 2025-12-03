import pandas as pd
import numpy as np
from typing import List, Any, Optional, Dict, Union

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    cols_to_drop = [c for c in columns if c in df.columns]
    if not cols_to_drop:
        return df
    return df.drop(columns=cols_to_drop)

def rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=rename_map)

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)

def change_column_dtype(df: pd.DataFrame, column: str, new_type: str) -> pd.DataFrame:
    df_clean = df.copy()
    try:
        df_clean[column] = df_clean[column].astype(new_type)
    except Exception as e:
        print(f"Error converting '{column}' to {new_type}: {e}")
        
    return df_clean

def convert_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df_clean = df.copy()
    try:
        df_clean[column] = pd.to_datetime(df_clean[column])
    except Exception as e:
        print(f"Error converting '{column}' to 'datetime': {e}")
    return df_clean

def handle_missing_values(df: pd.DataFrame, strategy: str, column: str, fill_value: Any = None) -> pd.DataFrame:
    df_clean = df.copy()

    if df_clean[column].isnull().sum() == 0:
        return df_clean
    
    if strategy == 'drop_rows':
        return df_clean.dropna(subset=[column])
            
    dtype = df_clean[column].dtype
    
    try:
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(dtype):
            val = df_clean[column].mean()
            df_clean[column] = df_clean[column].fillna(val)
            
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(dtype):
            val = df_clean[column].median()
            df_clean[column] = df_clean[column].fillna(val)
            
        elif strategy == 'mode':
            if not df_clean[column].mode().empty:
                val = df_clean[column].mode()[0]
                df_clean[column] = df_clean[column].fillna(val)
                
        elif strategy == 'constant' and fill_value is not None:
            df_clean[column] = df_clean[column].fillna(fill_value)
            
    except Exception as e:
        print(f"Imputation warning for {column}: {e}")
            
    return df_clean

def find_and_replace(df: pd.DataFrame, find_val: Any, replace_val: Any, column: Optional[str] = None) -> pd.DataFrame:
    df_clean = df.copy()
    
    if isinstance(replace_val, str) and replace_val.lower().strip() == 'nan':
        actual_replace_val = np.nan
    else:
        try:
            actual_replace_val = float(replace_val)
            if actual_replace_val.is_integer():
                actual_replace_val = int(actual_replace_val)
        except ValueError:
            actual_replace_val = replace_val

    vals_to_find = [find_val]
    
    try:
        num_val = float(find_val)
        vals_to_find.append(num_val)
        
        if num_val.is_integer():
            vals_to_find.append(int(num_val))
    except (ValueError, TypeError):
        pass

    try:
        if column:
            df_clean[column] = df_clean[column].replace(vals_to_find, actual_replace_val)
        else:
            df_clean = df_clean.replace(vals_to_find, actual_replace_val)
    except Exception as e:
        print(f"Replacement warning: {e}")
        
    return df_clean

def process_text_column(df: pd.DataFrame, column: str, operation: str, **kwargs) -> pd.DataFrame:
    df_clean = df.copy()
    s = df_clean[column]
    
    try:
        if operation == 'lower':
            df_clean[column] = s.str.lower()
        elif operation == 'upper':
            df_clean[column] = s.str.upper()
        elif operation == 'title':
            df_clean[column] = s.str.title()
        elif operation == 'strip':
            df_clean[column] = s.str.strip()
        elif operation == 'replace':
            pat = kwargs.get('old_val', '')
            repl = kwargs.get('new_val', '')
            df_clean[column] = s.str.replace(pat, repl, regex=False)
    except Exception as e:
        print(f"Text processing error on {column}: {e}")
        
    return df_clean

# Test
if __name__ == '__main__':
    pass
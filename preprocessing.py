"""
preprocessing.py - Data preprocessing pipeline for football transfer market value prediction.

This module provides a complete data pipeline for:
- Loading football transfer data from ZIP archives
- Handling missing values
- Validating numeric types
- Removing outliers using IQR method
- Feature engineering (if needed)
- One-hot encoding categorical features
- Feature scaling with StandardScaler
"""

import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple


def load_football_data(zip_path: str = 'footballTransfer.zip') -> pd.DataFrame:
    """
    Load football transfer dataset from ZIP archive.
    
    Args:
        zip_path: Path to footballTransfer.zip file
        
    Returns:
        DataFrame containing player market value data
    """
    print(f"Loading data from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Read the main player values CSV
        df = pd.read_csv(
            z.open('Football_Player_Market_Value_Trajectories/transfermarkt_player_values.csv')
        )
    print(f"Loaded {len(df)} players with {len(df.columns)} features")
    return df


def validate_numeric_types(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Validate and convert numeric columns to proper types.
    Remove rows with non-convertible values.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of column names that should be numeric
        
    Returns:
        DataFrame with validated numeric columns
    """
    print("\nValidating numeric types...")
    rows_before = len(df)
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        # Attempt conversion
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where target variable (current_value_eur) is null
    df = df.dropna(subset=['current_value_eur'])
    
    rows_after = len(df)
    removed = rows_before - rows_after
    print(f"Removed {removed} rows with invalid numeric values. Remaining: {rows_after}")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Strategy:
    - Drop rows missing >50% of features
    - Fill numeric nulls with median (grouped by position if sensible)
    - Fill categorical nulls with mode or 'Unknown'
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with handled missing values
    """
    print("\nHandling missing values...")
    rows_before = len(df)
    
    # Drop rows with >50% missing values
    threshold = 0.5 * len(df.columns)
    df = df.dropna(thresh=threshold)
    print(f"Dropped rows with >50% missing. Removed: {rows_before - len(df)}")
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode or 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(mode_val)
    
    print(f"Filled missing values. Nulls remaining: {df.isnull().sum().sum()}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: list, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using Interquartile Range (IQR) method.
    
    For each numeric column:
    - Calculate Q1 (25th percentile) and Q3 (75th percentile)
    - IQR = Q3 - Q1
    - Lower bound = Q1 - iqr_multiplier * IQR
    - Upper bound = Q3 + iqr_multiplier * IQR
    - Remove rows outside [lower, upper] bounds
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names to check for outliers
        iqr_multiplier: Multiplier for IQR (default 1.5 is standard)
        
    Returns:
        DataFrame with outliers removed
    """
    print(f"\nRemoving outliers using IQR method (multiplier={iqr_multiplier})...")
    rows_before = len(df)
    
    df_clean = df.copy()
    
    for col in numeric_cols:
        if col not in df_clean.columns:
            continue
        
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        # Count outliers before removing
        outliers = len(df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)])
        
        # Remove outliers
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]
        
        if outliers > 0:
            print(f"  {col}: removed {outliers} outliers (Q1={Q1:.2e}, Q3={Q3:.2e}, IQR={IQR:.2e})")
    
    rows_after = len(df_clean)
    print(f"Total rows removed: {rows_before - rows_after}. Remaining: {rows_after}")
    
    return df_clean


def select_features_and_prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    """
    Select relevant features for model training.
    Separate into numeric and categorical features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (feature_df, numeric_features, categorical_features)
    """
    print("\nSelecting features for model...")
    
    # Exclude these columns from features
    exclude_cols = {
        'player_id', 'name', 'peak_date', 'first_date', 'last_date',
        'peak_club', 'current_club', 'current_value_eur', 'current_value_tier',
        'peak_value_tier', 'data_source', 'dataset_built_at', 'nationality'
    }
    
    # Select feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify numeric and categorical
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    X = df[feature_cols].copy()
    
    print(f"Selected {len(feature_cols)} features:")
    print(f"  Numeric: {len(numeric_features)} - {numeric_features}")
    print(f"  Categorical: {len(categorical_features)} - {categorical_features}")
    
    return X, numeric_features, categorical_features


def create_preprocessing_pipeline(numeric_features: list, categorical_features: list):
    """
    Create a scikit-learn pipeline for preprocessing.
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        ColumnTransformer pipeline
    """
    print("\nCreating preprocessing pipeline...")
    
    # Numeric pipeline: StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    print(f"Pipeline created: scales {len(numeric_features)} numeric + one-hot encodes {len(categorical_features)} categorical features")
    return preprocessor


def preprocess_data(
    zip_path: str = 'footballTransfer.zip',
    test_size: float = 0.2,
    random_state: int = 42,
    remove_outliers: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, list, list]:
    """
    Complete preprocessing pipeline orchestration.
    
    Args:
        zip_path: Path to ZIP file containing data
        test_size: Proportion of data for testing (0.2 = 80-20 split)
        random_state: Random seed for reproducibility
        remove_outliers: Whether to apply IQR outlier removal
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor, numeric_cols, categorical_cols)
    """
    from sklearn.model_selection import train_test_split
    
    print("=" * 80)
    print("PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load data
    df = load_football_data(zip_path)
    print(f"\nInitial dataset shape: {df.shape}")
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Validate numeric types
    numeric_cols_to_validate = df.select_dtypes(include=[np.number]).columns.tolist()
    df = validate_numeric_types(df, numeric_cols_to_validate)
    
    # Step 4: Remove outliers
    if remove_outliers:
        outlier_cols = ['current_value_eur', 'age', 'career_span_years', 
                       'value_multiplier_x', 'value_volatility']
        outlier_cols = [c for c in outlier_cols if c in df.columns]
        df = remove_outliers_iqr(df, outlier_cols)
    
    # Step 5: Select features
    X, numeric_features, categorical_features = select_features_and_prepare(df)
    y = df['current_value_eur'].values
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Target range: EUR {y.min():.2e} to {y.max():.2e}")
    
    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain-test split ({(1-test_size)*100:.0f}-{test_size*100:.0f}):")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")
    
    # Step 7: Fit preprocessor on training data
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"\nProcessed feature shapes:")
    print(f"  X_train: {X_train_processed.shape}")
    print(f"  X_test: {X_test_processed.shape}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, numeric_features, categorical_features


if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = preprocess_data()
    print("\nPreprocessing successful!")
    print(f"Preprocessor type: {type(preprocessor)}")

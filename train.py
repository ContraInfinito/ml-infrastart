"""
train.py — Train and serialize a RandomForest regressor on football transfer data.

This script:
- Loads football player market value data from footballTransfer.zip
- Preprocesses features (scaling, encoding, outlier removal)
- Trains RandomForestRegressor with different n_estimators values
- Performs cross-validation for robust evaluation
- Logs metrics (R2, RMSE, MAE, training time) to training_logs.csv
- Saves best model to model.joblib
- Records feature importances

Target: Predict player current market value (EUR) using career trajectory,
position, league, age, and engineered features (CAGR, value multiplier, etc.)
"""

import joblib
import numpy as np
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from preprocessing import preprocess_data


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """
    Evaluate model on train and test sets.
    Calculate R2, RMSE, MAE, and prediction time.
    
    Args:
        model: Trained RandomForestRegressor
        X_train, X_test: Feature sets (already preprocessed)
        y_train, y_test: Target values
        model_name: Name for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Training predictions and metrics
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Testing predictions and metrics
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Prediction timing (milliseconds per sample)
    start = time.time()
    for _ in range(100):
        model.predict(X_test[:10])
    pred_time_ms = (time.time() - start) / 10.0
    
    metrics = {
        'model_name': model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'pred_time_ms': pred_time_ms,
        'y_test_pred': y_test_pred,
        'y_test': y_test
    }
    
    return metrics


def train_with_cross_validation(
    X_train, y_train,
    n_estimators: int,
    cv_folds: int = 5
) -> dict:
    """
    Train model and perform k-fold cross-validation.
    
    Args:
        X_train: Training features (preprocessed)
        y_train: Training targets
        n_estimators: Number of trees in RandomForest
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation results
    """
    print(f"\n  Performing {cv_folds}-fold cross-validation...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Cross-validation scoring
    cv_results = cross_validate(
        model,
        X_train, y_train,
        cv=cv_folds,
        scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
        n_jobs=-1,
        return_train_score=True
    )
    
    # Extract meaningful metrics
    cv_metrics = {
        'cv_train_r2_mean': cv_results['train_r2'].mean(),
        'cv_train_r2_std': cv_results['train_r2'].std(),
        'cv_test_r2_mean': cv_results['test_r2'].mean(),
        'cv_test_r2_std': cv_results['test_r2'].std(),
        'cv_train_rmse_mean': np.sqrt(-cv_results['train_neg_mean_squared_error'].mean()),
        'cv_test_rmse_mean': np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()),
        'cv_train_mae_mean': -cv_results['train_neg_mean_absolute_error'].mean(),
        'cv_test_mae_mean': -cv_results['test_neg_mean_absolute_error'].mean(),
    }
    
    print(f"    CV Train R2: {cv_metrics['cv_train_r2_mean']:.4f} (+/- {cv_metrics['cv_train_r2_std']:.4f})")
    print(f"    CV Test R2: {cv_metrics['cv_test_r2_mean']:.4f} (+/- {cv_metrics['cv_test_r2_std']:.4f})")
    print(f"    CV Test RMSE: {cv_metrics['cv_test_rmse_mean']:.2e}")
    print(f"    CV Test MAE: {cv_metrics['cv_test_mae_mean']:.2e}")
    
    return cv_metrics


def main():
    """Train RandomForest models with different n_estimators and log results."""
    
    print("=" * 80)
    print("RANDOM FOREST REGRESSOR - FOOTBALL TRANSFER MARKET VALUE PREDICTION")
    print("=" * 80)
    
    # Step 1: Preprocess data
    print("\nStep 1: Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = preprocess_data(
        remove_outliers=True
    )
    print(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Target range - Train: {y_train.min():.2e} to {y_train.max():.2e}")
    
    # Step 2: Define n_estimators to test
    n_estimators_list = [50, 100, 200, 500, 1000]
    print(f"\nStep 2: Testing n_estimators: {n_estimators_list}")
    
    # Step 3: Train models and collect results
    print("\nStep 3: Training models...")
    results_list = []
    best_test_r2 = -np.inf
    best_model = None
    best_n_estimators = None
    
    for n_est in n_estimators_list:
        print(f"\n  Training with n_estimators={n_est}...")
        start_time = time.time()
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_est,
            random_state=42,
            n_jobs=-1,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            verbose=0
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"    Training time: {training_time:.2f} seconds")
        
        # Evaluate
        eval_metrics = evaluate_model(model, X_train, X_test, y_train, y_test, f"RF_n={n_est}")
        
        # Cross-validation
        cv_metrics = train_with_cross_validation(X_train, y_train, n_est, cv_folds=5)
        
        # Combine results
        result_row = {
            'timestamp': datetime.now().isoformat(),
            'n_estimators': n_est,
            'training_time_sec': training_time,
            'train_r2': eval_metrics['train_r2'],
            'test_r2': eval_metrics['test_r2'],
            'train_rmse': eval_metrics['train_rmse'],
            'test_rmse': eval_metrics['test_rmse'],
            'train_mae': eval_metrics['train_mae'],
            'test_mae': eval_metrics['test_mae'],
            'pred_time_ms': eval_metrics['pred_time_ms'],
            'cv_train_r2_mean': cv_metrics['cv_train_r2_mean'],
            'cv_train_r2_std': cv_metrics['cv_train_r2_std'],
            'cv_test_r2_mean': cv_metrics['cv_test_r2_mean'],
            'cv_test_r2_std': cv_metrics['cv_test_r2_std'],
            'cv_train_rmse_mean': cv_metrics['cv_train_rmse_mean'],
            'cv_test_rmse_mean': cv_metrics['cv_test_rmse_mean'],
            'cv_train_mae_mean': cv_metrics['cv_train_mae_mean'],
            'cv_test_mae_mean': cv_metrics['cv_test_mae_mean'],
        }
        results_list.append(result_row)
        
        # Track best model
        if eval_metrics['test_r2'] > best_test_r2:
            best_test_r2 = eval_metrics['test_r2']
            best_model = model
            best_n_estimators = n_est
            print(f"    NEW BEST: Test R2 = {best_test_r2:.4f}")
        else:
            print(f"    Test R2 = {eval_metrics['test_r2']:.4f} (best so far: {best_test_r2:.4f})")
    
    # Step 4: Log results to CSV
    print("\n" + "=" * 80)
    print("Step 4: Logging results...")
    results_df = pd.DataFrame(results_list)
    
    # Append to existing log or create new
    log_file = Path('training_logs.csv')
    if log_file.exists():
        existing_df = pd.read_csv(log_file)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        print(f"Appended to existing log. Total records: {len(results_df)}")
    else:
        print(f"Created new log file")
    
    results_df.to_csv('training_logs.csv', index=False)
    print(f"Logged to training_logs.csv")
    
    # Display results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(results_df[['n_estimators', 'train_r2', 'test_r2', 'train_rmse', 'test_rmse',
                      'training_time_sec', 'cv_test_r2_mean', 'cv_test_rmse_mean']].to_string(index=False))
    
    # Step 5: Save best model
    print("\n" + "=" * 80)
    print(f"Step 5: Saving best model (n_estimators={best_n_estimators}, Test R2={best_test_r2:.4f})...")
    
    # Bundle model with preprocessing info
    artifact = {
        'model': best_model,
        'preprocessor': preprocessor,
        'numeric_features': num_cols,
        'categorical_features': cat_cols,
        'n_estimators': best_n_estimators,
        'test_r2': best_test_r2,
        'feature_importance': dict(enumerate(best_model.feature_importances_))
    }
    
    joblib.dump(artifact, 'model.joblib')
    print(f"Saved to model.joblib")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest configuration: n_estimators={best_n_estimators}")
    print(f"Best test R2: {best_test_r2:.4f}")
    print(f"Best test RMSE: {results_df[results_df['n_estimators']==best_n_estimators]['test_rmse'].values[0]:.2e}")
    print(f"Best test MAE: {results_df[results_df['n_estimators']==best_n_estimators]['test_mae'].values[0]:.2e}")


if __name__ == "__main__":
    main()
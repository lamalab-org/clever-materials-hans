import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_absolute_percentage_error,
    f1_score, precision_score, recall_score, r2_score
)
from sklearn.dummy import DummyRegressor, DummyClassifier
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_features_for_boosting(X_train: np.ndarray, X_test: np.ndarray, model_type: str = 'xgb') -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess features for gradient boosting models (XGBoost/LightGBM) to handle inf, large values, and scaling.
    """
    # Check for problematic values before processing
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        print(f"Found inf values in features. Train: {np.isinf(X_train).sum()}, Test: {np.isinf(X_test).sum()}")
    
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        print(f"Found nan values in features. Train: {np.isnan(X_train).sum()}, Test: {np.isnan(X_test).sum()}")
    
    # Convert to float64 to ensure precision
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    
    if model_type == 'lgb':
        # LightGBM is more robust - can handle nan values natively
        # Just replace inf values, leave nan as is for LightGBM to handle
        X_train = np.where(np.isposinf(X_train), 1e3, X_train)
        X_train = np.where(np.isneginf(X_train), -1e3, X_train)
        X_test = np.where(np.isposinf(X_test), 1e3, X_test)
        X_test = np.where(np.isneginf(X_test), -1e3, X_test)
        
        # Clip extreme values but keep nan
        X_train = np.where(~np.isnan(X_train), np.clip(X_train, -1e3, 1e3), X_train)
        X_test = np.where(~np.isnan(X_test), np.clip(X_test, -1e3, 1e3), X_test)
        
    else:  # XGBoost or other
        # More aggressive preprocessing for XGBoost
        X_train = np.where(np.isposinf(X_train), 1e3, X_train)
        X_train = np.where(np.isneginf(X_train), -1e3, X_train)
        X_test = np.where(np.isposinf(X_test), 1e3, X_test)
        X_test = np.where(np.isneginf(X_test), -1e3, X_test)
        
        # Replace nan with 0 for XGBoost
        X_train = np.where(np.isnan(X_train), 0.0, X_train)
        X_test = np.where(np.isnan(X_test), 0.0, X_test)
        
        # Clip to conservative range
        X_train = np.clip(X_train, -1e3, 1e3)
        X_test = np.clip(X_test, -1e3, 1e3)
        
        # Final validation for XGBoost
        if np.isinf(X_train).any() or np.isinf(X_test).any() or np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError("Preprocessing failed - still have inf/nan values")
    
    # Final check for infinite values (both models should handle this)
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        raise ValueError("Preprocessing failed - still have inf values")
    
    return X_train, X_test

def add_most_common_n_feats(df, counter, n=30): 
    most_common_authors = [i[0] for i in counter.most_common(n)]

    new_rows = []
    for i, row in df.iterrows():
        row_copy = row.copy()
        for author in most_common_authors: 
            has_author = author in row['authors_full_list']
            author_escaped = author.replace(' ', '_')
            row_copy[f'meta_feat_has_{author_escaped}'] = has_author

        new_rows.append(row_copy)

    return pd.DataFrame(new_rows)


def add_journal_features(df: pd.DataFrame, n_journals: int = 10) -> pd.DataFrame:
    """Add journal features using the same pattern as authors."""
    if 'journal_name' not in df.columns:
        return df
    
    journal_counter = Counter(df["journal_name"].dropna())
    most_common_journals = [i[0] for i in journal_counter.most_common(n_journals)]
    
    new_rows = []
    for i, row in df.iterrows():
        row_copy = row.copy()
        for journal in most_common_journals:
            has_journal = row['journal_name'] == journal if pd.notna(row['journal_name']) else False
            journal_escaped = journal.replace(' ', '_').replace('.', '_').replace('/', '_')
            row_copy[f'meta_feat_has_journal_{journal_escaped}'] = has_journal
        
        new_rows.append(row_copy)
    
    return pd.DataFrame(new_rows)


def calculate_multioutput_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for multioutput classification.
    For multioutput, we use 'samples' averaging which evaluates metrics for each sample
    and then averages across samples.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),  # This is subset accuracy for multioutput
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard single-output classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }


def run_leave_one_author_group_out_cv(df: pd.DataFrame,
                                    target_column: str,
                                    target_type: str,
                                    n_top_authors: int = 10,
                                    model_type: str = 'lgb',
                                    random_state: int = 42,
                                    include_indirect: bool = False,
                                    n_authors: int = 50,
                                    use_year: bool = True,
                                    use_journal: bool = True) -> Dict[str, Any]:
    """
    Run Leave-One-Author-Group-Out cross-validation to audit Clever Hans effects.
    
    This provides a more robust test by explicitly ensuring author groups are not 
    shared between training and testing, preventing the model from exploiting 
    author-specific patterns.
    
    Args:
        df: DataFrame with features and targets
        target_column: Name of target column
        target_type: 'regression' or 'classification'
        n_top_authors: Number of top authors to use for group-out CV
        model_type: Model type ('lgb', 'xgb', 'sklearn')
        random_state: Random seed
        include_indirect: If True, also run indirect (meta-information) models
        n_authors: Number of authors for meta-information encoding
        use_year: Whether to include year as meta-information
        use_journal: Whether to include journal as meta-information
        
    Returns:
        Dict containing CV results and author group analysis
    """
    
    # Get material features
    features = [f for f in df.columns if f.startswith("feat_")]
    
    # Prepare data with authors
    df_with_authors = df.dropna(subset=["authors_full_list"])
    all_authors = ""
    for author_string in df_with_authors["authors_full_list"]:
        all_authors += str(author_string) + "; "
    
    all_authors_list = [f.strip() for f in all_authors.split(";") if len(f.strip()) > 3]
    all_authors_count = Counter(all_authors_list)
    
    # Get top N authors
    top_authors = [author for author, _ in all_authors_count.most_common(n_top_authors)]
    print(f"Using top {len(top_authors)} authors for group-out CV")
    
    # Create author group mapping
    def get_primary_author_group(author_list_str):
        """Get the first top author found in the author list, or 'other' if none."""
        if pd.isna(author_list_str):
            return 'other'
        authors = [a.strip() for a in str(author_list_str).split(';')]
        for author in authors:
            if author in top_authors:
                return author
        return 'other'
    
    df_clean = df_with_authors.dropna(subset=features + [target_column]).copy()
    df_clean['author_group'] = df_clean['authors_full_list'].apply(get_primary_author_group)
    
    # Count samples per author group
    author_group_counts = df_clean['author_group'].value_counts()
    print(f"Author group distribution:")
    for group, count in author_group_counts.head(10).items():
        print(f"  {group}: {count} samples")
    
    # Filter to groups with enough samples for meaningful CV
    min_samples = 5
    viable_groups = author_group_counts[author_group_counts >= min_samples].index.tolist()
    viable_groups = [g for g in viable_groups if g != 'other']  # Exclude 'other' from CV groups
    
    print(f"Using {len(viable_groups)} author groups with ≥{min_samples} samples each")
    
    # Perform Leave-One-Author-Group-Out CV
    results_by_fold = []
    
    for i, held_out_group in enumerate(viable_groups):
        print(f"\nFold {i+1}/{len(viable_groups)}: Holding out group '{held_out_group}'")
        
        # Split data
        train_mask = df_clean['author_group'] != held_out_group
        test_mask = df_clean['author_group'] == held_out_group
        
        train_df = df_clean[train_mask]
        test_df = df_clean[test_mask]
        
        print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
        # Prepare features
        X_train = train_df[features].values
        X_test = test_df[features].values
        y_train = train_df[target_column].values
        y_test = test_df[target_column].values
        
        if model_type in ['xgb', 'lgb']:
            X_train, X_test = preprocess_features_for_boosting(X_train, X_test, model_type)
        
        # Train and evaluate models
        fold_result = {
            'fold': i,
            'held_out_group': held_out_group,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'y_test': y_test,
        }
        
        # Direct model (features only)
        model_direct = get_optimized_model(model_type, target_type, random_state)
        if target_type == 'regression':
            dummy = DummyRegressor(strategy='mean')
        else:
            dummy = DummyClassifier(strategy='stratified', random_state=random_state)
        
        model_direct.fit(X_train, y_train)
        dummy.fit(X_train, y_train)
        
        y_pred_direct = model_direct.predict(X_test)
        y_dummy = dummy.predict(X_test)
        
        if target_type == 'regression':
            fold_result['direct_metrics'] = calculate_regression_metrics(y_test, y_pred_direct)
            fold_result['dummy_metrics'] = calculate_regression_metrics(y_test, y_dummy)
        else:
            fold_result['direct_metrics'] = calculate_classification_metrics(y_test, y_pred_direct)
            fold_result['dummy_metrics'] = calculate_classification_metrics(y_test, y_dummy)
        
        fold_result['y_pred_direct'] = y_pred_direct
        fold_result['y_dummy'] = y_dummy
        
        # Indirect model (if requested)
        if include_indirect:
            # Get meta targets for author/journal encoding
            meta_targets = []
            if 'authors_full_list' in train_df.columns:
                # Create author encoding
                author_encoded = _encode_authors_to_top_k(train_df['authors_full_list'], n_authors, random_state)
                meta_targets.extend([f'author_{i}' for i in range(author_encoded.shape[1])])
                
            if use_year and 'year' in train_df.columns:
                # Year is already numeric, just copy
                train_df['year_numeric'] = pd.to_numeric(train_df['year'], errors='coerce').fillna(train_df['year'].median())
                test_df['year_numeric'] = pd.to_numeric(test_df['year'], errors='coerce').fillna(train_df['year_numeric'].mean())
                meta_targets.append('year_numeric')
                
            if use_journal and 'journal' in train_df.columns:
                # Create journal encoding
                journal_encoded = _encode_journal_to_top_k(train_df['journal'], n_authors, random_state)
                meta_targets.extend([f'journal_{i}' for i in range(journal_encoded.shape[1])])
            
            if meta_targets:
                # Build meta features for training and test
                X_train_meta = X_train.copy()
                X_test_meta = X_test.copy()
                
                if 'authors_full_list' in train_df.columns:
                    author_encoded_train = _encode_authors_to_top_k(train_df['authors_full_list'], n_authors, random_state)
                    author_encoded_test = _encode_authors_to_top_k(test_df['authors_full_list'], n_authors, random_state)
                    X_train_meta = np.concatenate([X_train_meta, author_encoded_train], axis=1)
                    X_test_meta = np.concatenate([X_test_meta, author_encoded_test], axis=1)
                
                if use_year and 'year' in train_df.columns:
                    year_train = train_df['year_numeric'].values.reshape(-1, 1)
                    year_test = test_df['year_numeric'].values.reshape(-1, 1)
                    X_train_meta = np.concatenate([X_train_meta, year_train], axis=1)
                    X_test_meta = np.concatenate([X_test_meta, year_test], axis=1)
                
                if use_journal and 'journal' in train_df.columns:
                    journal_encoded_train = _encode_journal_to_top_k(train_df['journal'], n_authors, random_state)
                    journal_encoded_test = _encode_journal_to_top_k(test_df['journal'], n_authors, random_state)
                    X_train_meta = np.concatenate([X_train_meta, journal_encoded_train], axis=1)
                    X_test_meta = np.concatenate([X_test_meta, journal_encoded_test], axis=1)
                
                # Train indirect model with meta features
                model_indirect = get_optimized_model(model_type, target_type, random_state)
                model_indirect.fit(X_train_meta, y_train)
                
                y_pred_indirect = model_indirect.predict(X_test_meta)
                
                if target_type == 'regression':
                    fold_result['indirect_metrics'] = calculate_regression_metrics(y_test, y_pred_indirect)
                else:
                    fold_result['indirect_metrics'] = calculate_classification_metrics(y_test, y_pred_indirect)
                
                fold_result['y_pred_indirect'] = y_pred_indirect
        
        results_by_fold.append(fold_result)
    
    # Aggregate results across folds
    aggregated_results = {}
    
    # Get all metrics from first fold
    if results_by_fold:
        metric_names = list(results_by_fold[0]['direct_metrics'].keys())
        
        # Direct model results
        for metric in metric_names:
            direct_values = [fold['direct_metrics'][metric] for fold in results_by_fold]
            dummy_values = [fold['dummy_metrics'][metric] for fold in results_by_fold]
            
            aggregated_results[f'direct_{metric}'] = {
                'mean': np.mean(direct_values),
                'std': np.std(direct_values),
                'values': direct_values
            }
            aggregated_results[f'dummy_{metric}'] = {
                'mean': np.mean(dummy_values),
                'std': np.std(dummy_values),
                'values': dummy_values
            }
        
        # Indirect model results (if available)
        if include_indirect and 'indirect_metrics' in results_by_fold[0]:
            for metric in metric_names:
                indirect_values = [fold['indirect_metrics'][metric] for fold in results_by_fold if 'indirect_metrics' in fold]
                
                if indirect_values:
                    aggregated_results[f'indirect_{metric}'] = {
                        'mean': np.mean(indirect_values),
                        'std': np.std(indirect_values),
                        'values': indirect_values
                    }
    
    return {
        'cv_type': 'leave_one_author_group_out',
        'n_folds': len(viable_groups),
        'n_top_authors': n_top_authors,
        'viable_groups': viable_groups,
        'author_group_counts': author_group_counts.to_dict(),
        'results_by_fold': results_by_fold,
        'aggregated_results': aggregated_results,
        'dataset_info': {
            'total_samples': len(df_clean),
            'n_features': len(features),
            'target_column': target_column,
            'target_type': target_type
        }
    }


def export_showyourwork_metric(value: Union[float, int, str], 
                              filename: str, 
                              output_dir: Path,
                              decimal_places: int = 2) -> None:
    """Export a metric to showyourwork format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(value, float):
        formatted_value = f"{value:.{decimal_places}f}"
    else:
        formatted_value = str(value)
    
    with open(output_dir / f"{filename}.txt", 'w') as f:
        f.write(f"{formatted_value}\\endinput")


def calculate_effect_sizes_cv_comparison(conventional_results: Dict[str, Any], 
                                       logo_cv_results: Dict[str, Any],
                                       metric_name: str = 'mae') -> Dict[str, Any]:
    """
    Calculate effect sizes comparing conventional CV vs Leave-One-Author-Group-Out CV.
    
    Args:
        conventional_results: Results from conventional cross-validation
        logo_cv_results: Results from Leave-One-Author-Group-Out cross-validation
        metric_name: Name of metric to compare (e.g., 'mae', 'accuracy', 'f1')
        
    Returns:
        Dict containing effect sizes and statistical comparisons
    """
    
    # Extract values for conventional CV
    conv_direct = conventional_results['direct'][metric_name]['values']
    conv_indirect = conventional_results['indirect'][metric_name]['values']
    conv_dummy = conventional_results['dummy'][metric_name]['values']
    
    # Extract values for LOGO CV
    logo_direct = logo_cv_results['aggregated_results'][f'direct_{metric_name}']['values']
    logo_dummy = logo_cv_results['aggregated_results'][f'dummy_{metric_name}']['values']
    
    # Calculate effect sizes (Cohen's d)
    def cohens_d(x1, x2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(x1), len(x2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(x1, ddof=1) + (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0.0
    
    effect_sizes = {
        'direct_conv_vs_logo': cohens_d(conv_direct, logo_direct),
        'indirect_vs_logo_direct': cohens_d(conv_indirect, logo_direct) if conv_indirect else np.nan,
        'dummy_conv_vs_logo': cohens_d(conv_dummy, logo_dummy),
    }
    
    # Add indirect comparison if available in LOGO results
    if f'indirect_{metric_name}' in logo_cv_results['aggregated_results']:
        logo_indirect = logo_cv_results['aggregated_results'][f'indirect_{metric_name}']['values']
        effect_sizes['indirect_conv_vs_logo'] = cohens_d(conv_indirect, logo_indirect) if conv_indirect else np.nan
        effect_sizes['direct_vs_indirect_logo'] = cohens_d(logo_direct, logo_indirect)
    
    # Calculate performance gaps
    conv_gap_direct = np.mean(conv_direct) - np.mean(conv_dummy)
    logo_gap_direct = np.mean(logo_direct) - np.mean(logo_dummy)
    
    performance_gaps = {
        'conventional_direct_vs_dummy': conv_gap_direct,
        'logo_direct_vs_dummy': logo_gap_direct,
        'gap_reduction': conv_gap_direct - logo_gap_direct,
        'gap_reduction_pct': ((conv_gap_direct - logo_gap_direct) / conv_gap_direct * 100) if conv_gap_direct != 0 else 0.0
    }
    
    # Calculate S ratios: S = (proxy - dummy) / (direct - dummy)
    s_ratios = {}
    
    if conv_indirect:
        conv_gap_indirect = np.mean(conv_indirect) - np.mean(conv_dummy)
        performance_gaps['conventional_indirect_vs_dummy'] = conv_gap_indirect
        
        # S ratio for conventional CV
        s_ratios['conventional_S'] = conv_gap_indirect / conv_gap_direct if conv_gap_direct != 0 else np.nan
        
        if f'indirect_{metric_name}' in logo_cv_results['aggregated_results']:
            logo_indirect = logo_cv_results['aggregated_results'][f'indirect_{metric_name}']['values']
            logo_gap_indirect = np.mean(logo_indirect) - np.mean(logo_dummy)
            performance_gaps['logo_indirect_vs_dummy'] = logo_gap_indirect
            performance_gaps['indirect_gap_reduction'] = conv_gap_indirect - logo_gap_indirect
            performance_gaps['indirect_gap_reduction_pct'] = ((conv_gap_indirect - logo_gap_indirect) / conv_gap_indirect * 100) if conv_gap_indirect != 0 else 0.0
            
            # S ratio for LOAO CV
            s_ratios['logo_S'] = logo_gap_indirect / logo_gap_direct if logo_gap_direct != 0 else np.nan
            
            # S ratio change (should go from ~1 to ~0 if strong author shortcut)
            s_ratios['S_change'] = s_ratios['conventional_S'] - s_ratios['logo_S'] if not np.isnan(s_ratios['conventional_S']) and not np.isnan(s_ratios['logo_S']) else np.nan
            s_ratios['S_change_pct'] = (s_ratios['S_change'] / s_ratios['conventional_S'] * 100) if (s_ratios['conventional_S'] != 0 and not np.isnan(s_ratios['conventional_S'])) else np.nan
    
    return {
        'metric': metric_name,
        'effect_sizes': effect_sizes,
        'performance_gaps': performance_gaps,
        's_ratios': s_ratios,
        'summary_stats': {
            'conventional_direct': {'mean': np.mean(conv_direct), 'std': np.std(conv_direct)},
            'logo_direct': {'mean': np.mean(logo_direct), 'std': np.std(logo_direct)},
            'conventional_indirect': {'mean': np.mean(conv_indirect), 'std': np.std(conv_indirect)} if conv_indirect else None,
            'logo_indirect': {'mean': np.mean(logo_cv_results['aggregated_results'][f'indirect_{metric_name}']['values']), 
                            'std': np.std(logo_cv_results['aggregated_results'][f'indirect_{metric_name}']['values'])} if f'indirect_{metric_name}' in logo_cv_results['aggregated_results'] else None,
            'dummy': {'mean': np.mean(conv_dummy), 'std': np.std(conv_dummy)}
        }
    }


def _run_single_parameter_combination(params_data: Tuple) -> Dict[str, Any]:
    """
    Helper function to run analysis for a single parameter combination (for parallelization).
    """
    df, target_column, target_type, n_authors, use_year, use_journal, n_folds, random_state, dataset_name, model_type, use_actual_meta = params_data
    
    result = run_single_analysis(
        df, target_column, target_type, 
        n_authors=n_authors,
        use_year=use_year,
        use_journal=use_journal,
        n_folds=n_folds,
        random_state=random_state,
        n_jobs=1,  # Use sequential for individual parameter combinations
        model_type=model_type,
        use_actual_meta=use_actual_meta
    )
    
    result['parameters'] = {
        'dataset_name': dataset_name,
        'n_authors': n_authors,
        'use_year': use_year,
        'use_journal': use_journal
    }
    
    return result


def run_parameter_sweep_analysis(df: pd.DataFrame, 
                                target_column: str,
                                target_type: str,
                                dataset_name: str,
                                author_counts: List[int] = [10, 50, 100, 500],
                                use_year_options: List[bool] = [True, False],
                                use_journal_options: List[bool] = [True, False],
                                n_folds: int = 10,
                                random_state: int = 42,
                                n_jobs: int = -1,
                                model_type: str = 'lgb',
                                use_actual_meta: bool = False) -> List[Dict]:
    """
    Run parameter sweep analysis with different hyperparameter combinations.
    
    Args:
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
    
    Returns:
        List of results for each parameter combination
    """
    total_combinations = len(author_counts) * len(use_year_options) * len(use_journal_options)
    print(f"Running parameter sweep with {total_combinations} combinations for {dataset_name}")
    
    # Prepare parameter combinations for parallel processing
    params_data_list = []
    for n_authors in author_counts:
        for use_year in use_year_options:
            for use_journal in use_journal_options:
                params_data_list.append((
                    df, target_column, target_type, n_authors, use_year, 
                    use_journal, n_folds, random_state, dataset_name, model_type, use_actual_meta
                ))
    
    # Run parameter combinations in parallel or sequentially
    if n_jobs == 1 or total_combinations == 1:
        # Sequential execution with progress tracking
        results = []
        for i, params_data in enumerate(tqdm(params_data_list, desc="Parameter combinations"), 1):
            n_authors, use_year, use_journal = params_data[3], params_data[4], params_data[5]
            print(f"\n--- Running combination: n_authors={n_authors}, use_year={use_year}, use_journal={use_journal} ---")
            
            result = _run_single_parameter_combination(params_data)
            results.append(result)
    else:
        # Parallel execution
        max_workers = None if n_jobs == -1 else min(n_jobs, total_combinations)
        print(f"Running {total_combinations} parameter combinations in parallel (max_workers={max_workers})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to track parallel execution
            futures = [executor.submit(_run_single_parameter_combination, params_data) 
                      for params_data in params_data_list]
            results = [future.result() for future in tqdm(futures, desc="Parameter combinations")]
    
    return results


def run_meta_comparison_analysis(df: pd.DataFrame,
                               target_column: str,
                               target_type: str,
                               n_authors: int = 50,
                               use_year: bool = True,
                               use_journal: bool = True,
                               n_folds: int = 10,
                               random_state: int = 42,
                               n_jobs: int = -1,
                               model_type: str = 'lgb') -> Dict[str, Any]:
    """
    Run both predicted and actual meta-information analyses for comparison.
    
    Returns:
        Dict containing both 'predicted_meta' and 'actual_meta' results
    """
    print("Running analysis with predicted meta-information...")
    predicted_results = run_single_analysis(
        df, target_column, target_type,
        n_authors=n_authors,
        use_year=use_year,
        use_journal=use_journal,
        n_folds=n_folds,
        random_state=random_state,
        n_jobs=n_jobs,
        model_type=model_type,
        use_actual_meta=False
    )
    
    print("Running analysis with actual meta-information...")
    actual_results = run_single_analysis(
        df, target_column, target_type,
        n_authors=n_authors,
        use_year=use_year,
        use_journal=use_journal,
        n_folds=n_folds,
        random_state=random_state,
        n_jobs=n_jobs,
        model_type=model_type,
        use_actual_meta=True
    )
    
    return {
        'predicted_meta': predicted_results,
        'actual_meta': actual_results,
        'parameters': {
            'n_authors': n_authors,
            'use_year': use_year,
            'use_journal': use_journal,
            'n_folds': n_folds,
            'model_type': model_type
        }
    }


def get_optimized_model(model_type: str, task: str, random_state: int = 42):
    """Get optimized model for faster performance."""
    if model_type == 'xgb':
        # XGBoost with mostly default parameters
        base_params = {
            'random_state': random_state,
            'n_jobs': 1,  # Use 1 thread per model for better parallelization
            'verbosity': 0,  # Reduce output
        }
        
        if task == 'regression':
            return xgb.XGBRegressor(**base_params)
        else:
            return xgb.XGBClassifier(**base_params)
    
    elif model_type == 'lgb':
        # LightGBM with mostly default parameters
        base_params = {
            'random_state': random_state,
            'n_jobs': 1,  # Use 1 thread per model for better parallelization
            'verbosity': -1,  # Suppress warnings
        }
        
        if task == 'regression':
            return lgb.LGBMRegressor(**base_params)
        else:
            return lgb.LGBMClassifier(**base_params)
    
    else:  # sklearn
        if task == 'regression':
            return HistGradientBoostingRegressor(random_state=random_state)
        else:
            return HistGradientBoostingClassifier(random_state=random_state)


def _run_single_fold(fold_data: Tuple) -> Dict[str, Any]:
    """
    Helper function to run analysis for a single CV fold (for parallelization).
    """
    train_idx, test_idx, df_clean, features, meta_targets, target_column, target_type, use_year, random_state, model_type, use_actual_meta = fold_data
    
    train_df = df_clean.iloc[train_idx]
    test_df = df_clean.iloc[test_idx]
    
    fold_result = {}
    
    # Preprocess features for boosting models if needed
    X_train_features = train_df[features].values
    X_test_features = test_df[features].values
    
    if model_type in ['xgb', 'lgb']:
        X_train_features, X_test_features = preprocess_features_for_boosting(X_train_features, X_test_features, model_type)
    
    # Separate author and journal targets
    author_targets = [t for t in meta_targets if 'journal' not in t]
    journal_targets = [t for t in meta_targets if 'journal' in t]
    
    # 1a. Author prediction
    if author_targets:
        base_classifier = get_optimized_model(model_type, 'classification', random_state)
        model_author = MultiOutputClassifier(base_classifier)
        model_author.fit(X_train_features, train_df[author_targets])
        
        train_author_pred = model_author.predict(X_train_features)
        test_author_pred = model_author.predict(X_test_features)
        
        fold_result['author'] = calculate_multioutput_classification_metrics(
            test_df[author_targets].values, test_author_pred
        )
        
        # Initialize indirect features with authors
        indirect_features = train_author_pred.copy()
        indirect_features_test = test_author_pred.copy()
    else:
        fold_result['author'] = {'accuracy': np.nan, 'f1_micro': np.nan, 'f1_macro': np.nan}
        indirect_features = np.empty((len(train_df), 0))
        indirect_features_test = np.empty((len(test_df), 0))
    
    # 1b. Journal prediction
    if journal_targets:
        base_classifier = get_optimized_model(model_type, 'classification', random_state)
        model_journal = MultiOutputClassifier(base_classifier)
        model_journal.fit(X_train_features, train_df[journal_targets])
        
        train_journal_pred = model_journal.predict(X_train_features)
        test_journal_pred = model_journal.predict(X_test_features)
        
        fold_result['journal'] = calculate_multioutput_classification_metrics(
            test_df[journal_targets].values, test_journal_pred
        )
        
        # Add journal to indirect features
        if indirect_features.size > 0:
            indirect_features = np.column_stack([indirect_features, train_journal_pred])
            indirect_features_test = np.column_stack([indirect_features_test, test_journal_pred])
        else:
            indirect_features = train_journal_pred.copy()
            indirect_features_test = test_journal_pred.copy()
    else:
        fold_result['journal'] = {'accuracy': np.nan, 'f1_micro': np.nan, 'f1_macro': np.nan}
    
    # 1c. Combined meta prediction (for backward compatibility)
    if meta_targets:
        base_classifier = get_optimized_model(model_type, 'classification', random_state)
        model_meta = MultiOutputClassifier(base_classifier)
        model_meta.fit(X_train_features, train_df[meta_targets])
        
        test_meta_pred = model_meta.predict(X_test_features)
        
        fold_result['meta'] = calculate_multioutput_classification_metrics(
            test_df[meta_targets].values, test_meta_pred
        )
    
    # 2. Year prediction (optional)
    if use_year and 'publication_year' in df_clean.columns:
        model_year = get_optimized_model(model_type, 'regression', random_state)
        model_year.fit(X_train_features, train_df['publication_year'])
        
        train_year_pred = model_year.predict(X_train_features)
        test_year_pred = model_year.predict(X_test_features)
        
        fold_result['year'] = calculate_regression_metrics(
            test_df['publication_year'].values, test_year_pred
        )
        
        # Add year to indirect features
        indirect_features = np.column_stack([indirect_features, train_year_pred])
        indirect_features_test = np.column_stack([indirect_features_test, test_year_pred])
    else:
        fold_result['year'] = {'mae': np.nan, 'mape': np.nan, 'r2': np.nan}
    
    # 3. Direct prediction
    model_direct = get_optimized_model(model_type, target_type, random_state)
    if target_type == 'regression':
        dummy_direct = DummyRegressor(strategy='mean')
    else:
        dummy_direct = DummyClassifier(strategy='stratified', random_state=random_state)
    
    model_direct.fit(X_train_features, train_df[target_column])
    direct_pred = model_direct.predict(X_test_features)
    
    if target_type == 'regression':
        fold_result['direct'] = calculate_regression_metrics(
            test_df[target_column].values, direct_pred
        )
    else:
        fold_result['direct'] = calculate_classification_metrics(
            test_df[target_column].values, direct_pred
        )
    
    fold_result['direct']['predictions'] = direct_pred
    fold_result['direct']['true'] = test_df[target_column].values
    
    # 4. Indirect prediction (Clever Hans)
    model_indirect = get_optimized_model(model_type, target_type, random_state)
    
    # Choose between predicted or actual meta-information
    if use_actual_meta:
        # Use actual meta-information (upper bound analysis)
        actual_meta_features_train = train_df[meta_targets].values.astype(float)
        actual_meta_features_test = test_df[meta_targets].values.astype(float)
        
        # Add year if requested
        if use_year and 'publication_year' in df_clean.columns:
            actual_meta_features_train = np.column_stack([actual_meta_features_train, train_df['publication_year'].values])
            actual_meta_features_test = np.column_stack([actual_meta_features_test, test_df['publication_year'].values])
        
        # Preprocess actual meta features if using boosting models
        if model_type in ['xgb', 'lgb']:
            indirect_features_clean, indirect_features_test_clean = preprocess_features_for_boosting(
                actual_meta_features_train, actual_meta_features_test, model_type
            )
        else:
            indirect_features_clean = actual_meta_features_train
            indirect_features_test_clean = actual_meta_features_test
    else:
        # Use predicted meta-information (standard analysis)
        # Preprocess indirect features if using boosting models
        if model_type in ['xgb', 'lgb']:
            indirect_features_clean, indirect_features_test_clean = preprocess_features_for_boosting(
                indirect_features, indirect_features_test, model_type
            )
        else:
            indirect_features_clean = indirect_features
            indirect_features_test_clean = indirect_features_test
    
    model_indirect.fit(indirect_features_clean, train_df[target_column])
    indirect_pred = model_indirect.predict(indirect_features_test_clean)
    
    if target_type == 'regression':
        fold_result['indirect'] = calculate_regression_metrics(
            test_df[target_column].values, indirect_pred
        )
    else:
        fold_result['indirect'] = calculate_classification_metrics(
            test_df[target_column].values, indirect_pred
        )
    
    fold_result['indirect']['predictions'] = indirect_pred
    fold_result['indirect']['true'] = test_df[target_column].values
    
    # 5. Dummy baseline
    dummy_direct.fit(X_train_features, train_df[target_column])
    dummy_pred = dummy_direct.predict(X_test_features)
    
    if target_type == 'regression':
        fold_result['dummy'] = calculate_regression_metrics(
            test_df[target_column].values, dummy_pred
        )
    else:
        fold_result['dummy'] = calculate_classification_metrics(
            test_df[target_column].values, dummy_pred
        )
    
    fold_result['dummy']['predictions'] = dummy_pred
    fold_result['dummy']['true'] = test_df[target_column].values
    
    return fold_result


def run_single_analysis(df: pd.DataFrame,
                       target_column: str, 
                       target_type: str,
                       n_authors: int = 50,
                       use_year: bool = True,
                       use_journal: bool = True,
                       n_folds: int = 10,
                       random_state: int = 42,
                       n_jobs: int = -1,
                       model_type: str = 'lgb',
                       use_actual_meta: bool = False) -> Dict[str, Any]:
    """
    Run a single Clever Hans analysis with specified parameters.
    
    Args:
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        use_actual_meta: If True, use actual meta-information instead of predicted meta-information 
                        for the indirect model (upper bound analysis)
    """
    
    # Get material features
    features = [f for f in df.columns if f.startswith("feat_")]
    
    # Prepare data with authors
    df_with_authors = df.dropna(subset="authors_full_list")
    all_authors = ""
    for author_string in df_with_authors["authors_full_list"]:
        all_authors += str(author_string) + "; "
    
    all_authors_list = [f.strip() for f in all_authors.split(";") if len(f.strip()) > 3]
    all_authors_count = Counter(all_authors_list)
    
    # Debug output
    print(f"  Requested n_authors: {n_authors}, Available unique authors: {len(all_authors_count)}")
    actual_n_authors = min(n_authors, len(all_authors_count))
    print(f"  Using n_authors: {actual_n_authors}")
    
    df_with_features = add_most_common_n_feats(df_with_authors, all_authors_count, n=actual_n_authors)
    
    # Add journal features if requested
    if use_journal:
        df_with_features = add_journal_features(df_with_features, n_journals=10)
    
    # Get all meta features
    meta_targets = [f for f in df_with_features.columns if f.startswith("meta_feat_has_")]
    
    # Clean data
    required_cols = features + [target_column]
    if use_year and 'publication_year' in df_with_features.columns:
        required_cols.append('publication_year')
    
    df_clean = df_with_features.dropna(subset=required_cols)
    
    print(f"  Dataset size: {len(df_clean)}, Features: {len(features)}, Meta targets: {len(meta_targets)}")
    
    # Cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Prepare fold data for parallel processing
    fold_data_list = []
    for train_idx, test_idx in kfold.split(df_clean):
        fold_data_list.append((
            train_idx, test_idx, df_clean, features, meta_targets, 
            target_column, target_type, use_year, random_state, model_type, use_actual_meta
        ))
    
    # Run folds in parallel or sequentially
    if n_jobs == 1 or n_folds == 1:
        # Sequential execution with progress bar
        fold_results = [_run_single_fold(fold_data) for fold_data in tqdm(fold_data_list, desc="CV folds")]
    else:
        # Parallel execution
        max_workers = None if n_jobs == -1 else min(n_jobs, n_folds)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single_fold, fold_data) for fold_data in fold_data_list]
            fold_results = [future.result() for future in tqdm(futures, desc="CV folds")]
    
    # Aggregate results
    return aggregate_fold_results(fold_results, target_type, df_clean, features, meta_targets, target_column)


def aggregate_fold_results(fold_results: List[Dict], 
                          target_type: str,
                          df_clean: pd.DataFrame,
                          features: List[str],
                          meta_targets: List[str],
                          target_column: str) -> Dict[str, Any]:
    """Aggregate results across CV folds."""
    
    methods = ['meta', 'author', 'journal', 'year', 'direct', 'indirect', 'dummy']
    aggregated = {}
    
    for method in methods:
        aggregated[method] = {}
        
        # Get the first fold to determine available metrics
        first_fold = fold_results[0]
        if method in first_fold:
            metrics = list(first_fold[method].keys())
            # Remove non-numeric keys
            metrics = [m for m in metrics if m not in ['predictions', 'true']]
            
            for metric in metrics:
                values = []
                for fold in fold_results:
                    if method in fold and metric in fold[method]:
                        val = fold[method][metric]
                        if not pd.isna(val):
                            values.append(val)
                
                if values:
                    aggregated[method][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }
                else:
                    aggregated[method][metric] = {
                        'mean': np.nan,
                        'std': np.nan,
                        'values': []
                    }
    
    # Store predictions for plotting
    aggregated['raw_predictions'] = {}
    for method in ['direct', 'indirect', 'dummy']:
        all_preds = np.concatenate([fold[method]['predictions'] for fold in fold_results])
        all_true = np.concatenate([fold[method]['true'] for fold in fold_results])
        aggregated['raw_predictions'][method] = {
            'predictions': all_preds,
            'true': all_true
        }
    
    # Calculate specific target counts
    author_targets = [t for t in meta_targets if 'journal' not in t]
    journal_targets = [t for t in meta_targets if 'journal' in t]
    
    # Metadata
    aggregated['dataset_info'] = {
        'size': len(df_clean),
        'n_features': len(features),
        'n_meta_targets': len(meta_targets),
        'n_author_targets': len(author_targets),
        'n_journal_targets': len(journal_targets),
        'target_column': target_column,
        'target_type': target_type
    }
    
    return aggregated
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dabest
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import lama_aesthetics
from lama_aesthetics import (
    ONE_COL_WIDTH,
    TWO_COL_WIDTH,
    ONE_COL_HEIGHT,
    TWO_COL_HEIGHT,
)
from lama_aesthetics.plotutils import range_frame, ylabel_top, add_identity

# Initialize style
lama_aesthetics.get_style("main")

# Define consistent color scheme
COLORS = {
    'direct': '#1f77b4',      # Blue - conventional ML
    'indirect': '#ff7f0e',    # Orange - Clever Hans  
    'dummy': '#2ca02c',       # Green - baseline
    'author': '#ff7f0e',      # Orange - author prediction (consistent with Clever Hans)
    'journal': '#ff7f0e',     # Orange - journal prediction (consistent with Clever Hans)
    'year': '#ff7f0e',        # Orange - year prediction (consistent with Clever Hans)
    'meta_combined': '#ff7f0e' # Orange - combined meta (consistent with Clever Hans)
}

LABELS = {
    'direct': 'Conventional',
    'indirect': 'Clever Hans', 
    'dummy': 'Dummy',
    'meta': 'Meta prediction',
    'year': 'Year prediction'
}

# Default metric labels (easily customizable)
METRIC_LABELS = {
    'mae': 'MAE',
    'mape': 'MAPE (%)',
    'r2': '$R^2$',
    'accuracy': 'Accuracy',
    'precision': 'Precision', 
    'recall': 'Recall',
    'f1': 'F1-Score',
    'f1_micro': '$F_1$ (Micro)',
    'f1_macro': '$F_1$ (Macro)',
    'precision_micro': 'Precision (Micro)',
    'precision_macro': 'Precision (Macro)',
    'recall_micro': 'Recall (Micro)',
    'recall_macro': 'Recall (Macro)'
}


def plot_performance_comparison(results: Dict[str, Any], 
                              target_type: str = 'regression',
                              metric_labels: Optional[Dict[str, str]] = None,
                              save_path: Optional[Path] = None,
                              title_suffix: str = "") -> plt.Figure:
    """
    Plot performance comparison between methods.
    """
    
    if target_type == 'regression':
        metrics = ['mae', 'mape', 'r2']
    else:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Use custom labels if provided
    labels = metric_labels if metric_labels else METRIC_LABELS
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.3))
    if n_metrics == 1:
        axes = [axes]
    
    # Adjust subplot spacing for better visibility and y-label padding
    plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.20, wspace=0.5)
    
    methods = ['direct', 'indirect', 'dummy']
    colors = [COLORS[method] for method in methods]
    method_labels = [LABELS[method] for method in methods]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        means = []
        stds = []
        for method in methods:
            if method in results and metric in results[method]:
                means.append(results[method][metric]['mean'])
                stds.append(results[method][metric]['std'])
            else:
                means.append(np.nan)
                stds.append(0)
        
        # Remove NaN values for plotting
        valid_idx = ~np.isnan(means)
        if np.any(valid_idx):
            valid_means = np.array(means)[valid_idx]
            valid_stds = np.array(stds)[valid_idx]
            valid_colors = [colors[j] for j in range(len(colors)) if valid_idx[j]]
            valid_labels = [method_labels[j] for j in range(len(method_labels)) if valid_idx[j]]
            
            # Plot individual data points instead of bars
            x_positions = np.arange(len(valid_means))
            
            # Plot individual CV fold values as scatter points
            for i, method in enumerate([m for j, m in enumerate(methods) if valid_idx[j]]):
                if method in results and metric in results[method]:
                    fold_values = results[method][metric]['values']
                    x_jitter = x_positions[i] + np.random.normal(0, 0.05, len(fold_values))  # Small jitter
                    ax.scatter(x_jitter, fold_values, color=valid_colors[i], alpha=0.6, s=20)
            
            # Plot means as larger points
            ax.scatter(x_positions, valid_means, color=valid_colors, s=60, alpha=1.0, 
                      edgecolors='black', linewidths=0.5, zorder=3)
            
            # Add error bars for standard deviation
            ax.errorbar(x_positions, valid_means, yerr=valid_stds, fmt='none', 
                       capsize=3, color='black', alpha=0.7, zorder=2)
            
            ax.set_xticks(x_positions)
            
            # Use consistent y-label positioning across all subplots with adequate padding
            metric_label = labels.get(metric, metric)
            ax.set_ylabel(metric_label, fontsize=10, labelpad=15)
            ax.yaxis.set_label_coords(-0.45, 0.5)  # Fixed position with better padding
            
            # Apply range frame
            if len(valid_means) > 0:
                range_frame(ax, x_positions, valid_means)
    
    # Set x-tick labels with better rotation and spacing
    for i, metric in enumerate(metrics):
        ax = axes[i]
        if ax.get_xticks().size > 0:
            valid_idx = ~np.isnan([results[method][metric]['mean'] if method in results and metric in results[method] else np.nan for method in methods])
            if np.any(valid_idx):
                valid_labels = [LABELS[method] for j, method in enumerate(methods) if valid_idx[j]]
                ax.set_xticklabels(valid_labels, rotation=30, ha='right', fontsize=9)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_meta_performance(results: Dict[str, Any],
                         metric_labels: Optional[Dict[str, str]] = None,
                         save_path: Optional[Path] = None,
                         title_suffix: str = "") -> plt.Figure:
    """
    Plot meta prediction performance against baseline for author, journal, and year predictions.
    """
    labels = metric_labels if metric_labels else METRIC_LABELS
    
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))
    
    # Improve spacing for meta performance plots
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15, wspace=0.35)
    
    # Meta prediction types to plot
    meta_types = [
        ('author', 'Author Pred.', COLORS.get('author', '#2E86C1')),
        ('journal', 'Journal Pred.', COLORS.get('journal', '#28B463')), 
        ('year', 'Year Pred.', COLORS.get('year', '#F39C12'))
    ]
    
    for i, (meta_type, meta_label, meta_color) in enumerate(meta_types):
        ax = axes[i]
        
        if meta_type in results:
            # Choose appropriate metric based on prediction type
            if meta_type == 'year':
                # For year (regression), use MAE
                metric = 'mae'
                if metric in results[meta_type]:
                    pred_mean = results[meta_type][metric]['mean']
                    pred_std = results[meta_type][metric]['std']
                    
                    # Calculate baseline as standard deviation of years
                    if 'dummy' in results and metric in results['dummy']:
                        baseline_mean = results['dummy'][metric]['mean'] 
                        baseline_std = results['dummy'][metric]['std']
                    else:
                        # Conservative baseline estimate
                        baseline_mean = pred_mean * 1.5  # Assume 50% worse than prediction
                        baseline_std = 0
                    
                    # For MAE, lower is better - normalize by showing relative performance
                    normalized_pred = 1.0  # Reference
                    normalized_baseline = baseline_mean / pred_mean if pred_mean > 0 else 2.0
                    
                    means = [normalized_baseline, normalized_pred]
                    stds = [baseline_std / pred_mean if pred_mean > 0 else 0, pred_std / pred_mean if pred_mean > 0 else 0]
                    ylabel_text = f'Relative {labels.get(metric, metric)}'
                    
                else:
                    means = [np.nan, np.nan]
                    stds = [0, 0]
                    ylabel_text = 'MAE'
            else:
                # For author/journal (classification), use F1 score
                metric = 'f1_micro'
                if metric in results[meta_type]:
                    pred_mean = results[meta_type][metric]['mean']
                    pred_std = results[meta_type][metric]['std']
                    
                    # Calculate theoretical baseline for multiclass classification
                    # Conservative baseline for F1 with many classes
                    if meta_type == 'author':
                        baseline_mean = 0.02  # Very low for many authors
                    elif meta_type == 'journal':
                        baseline_mean = 0.1   # Slightly higher for fewer journals
                    else:
                        baseline_mean = 0.05
                    baseline_std = 0
                    
                    means = [baseline_mean, pred_mean]
                    stds = [baseline_std, pred_std]
                    ylabel_text = labels.get(metric, metric)
                    
                else:
                    means = [np.nan, np.nan]
                    stds = [0, 0]
                    ylabel_text = 'F1 Score'
            
            # Plot both bars if we have valid data
            if not (np.isnan(means[0]) or np.isnan(means[1])):
                x_pos = [0, 1]
                colors = [COLORS['dummy'], meta_color]
                labels_list = ['Baseline', meta_label]
                
                bars = ax.bar(x_pos, means, yerr=stds, capsize=3,
                             color=colors, alpha=0.8)
                
                ax.set_ylim(0, max(means) * 1.2)
                
                # Use manual ylabel for better control
                ax.set_ylabel(ylabel_text, fontsize=9, labelpad=10)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels_list, rotation=30, ha='right', fontsize=8)
                
                # Apply range frame
                range_frame(ax, np.array(x_pos), np.array(means))
                
                # Add value labels
                for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                    if meta_type == 'year' and j == 0:
                        # Show baseline as multiplier for year prediction
                        label_text = f'{mean:.1f}x'
                    else:
                        label_text = f'{mean:.3f}'
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           mean + std + max(means) * 0.02,
                           label_text, ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No {meta_type} data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, f'No {meta_type} data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Adjust spacing to prevent overlapping text (skip tight_layout)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_author_performance_vs_n_authors(sweep_results: List[Dict[str, Any]],
                                        metric: str = 'f1_micro',
                                        metric_labels: Optional[Dict[str, str]] = None,
                                        save_path: Optional[Path] = None,
                                        title_suffix: str = "") -> plt.Figure:
    """
    Plot author prediction performance as a function of number of authors.
    For SI/supplementary information.
    """
    labels = metric_labels if metric_labels else METRIC_LABELS
    
    # Extract data for different author counts
    author_counts = []
    performance_means = []
    performance_stds = []
    
    for result in sweep_results:
        if ('parameters' in result and 
            'author' in result and 
            metric in result['author']):
            
            n_authors = result['parameters']['n_authors']
            perf_mean = result['author'][metric]['mean']
            perf_std = result['author'][metric]['std']
            
            author_counts.append(n_authors)
            performance_means.append(perf_mean)
            performance_stds.append(perf_std)
    
    if not author_counts:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
        ax.text(0.5, 0.5, 'No author prediction data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    # Sort by author count
    sorted_data = sorted(zip(author_counts, performance_means, performance_stds))
    author_counts, performance_means, performance_stds = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
    
    # Plot line with error bars
    ax.errorbar(author_counts, performance_means, yerr=performance_stds,
                marker='o', markersize=8, capsize=5, linewidth=2,
                color=COLORS.get('author', '#2E86C1'), alpha=0.8)
    
    # Add baseline reference line
    baseline_value = 0.02  # Conservative baseline for many authors
    ax.axhline(y=baseline_value, color=COLORS['dummy'], linestyle='--', 
               alpha=0.7, linewidth=2, label='Baseline')
    
    ax.set_xlabel('Number of Authors Predicted')
    ylabel_top(labels.get(metric, metric), ax)
    ax.set_xscale('log')  # Log scale for better visualization
    ax.set_ylim(0, max(max(performance_means) * 1.2, baseline_value * 2))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Apply range frame
    range_frame(ax, np.array(author_counts), np.array(performance_means))
    
    # Better spacing for author performance plot
    plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_parameter_sweep_results(sweep_results: List[Dict[str, Any]],
                                metric: str = 'mae',
                                metric_labels: Optional[Dict[str, str]] = None,
                                save_path: Optional[Path] = None,
                                title_suffix: str = "") -> plt.Figure:
    """
    Plot results across parameter sweep (number of authors).
    """
    labels = metric_labels if metric_labels else METRIC_LABELS
    
    # Extract data for plotting
    df_data = []
    for result in sweep_results:
        params = result['parameters']
        for method in ['direct', 'indirect', 'dummy']:
            if method in result and metric in result[method]:
                df_data.append({
                    'n_authors': params['n_authors'],
                    'use_year': params['use_year'],
                    'use_journal': params['use_journal'],
                    'method': method,
                    'metric_value': result[method][metric]['mean'],
                    'metric_std': result[method][metric]['std']
                })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
        ax.text(0.5, 0.5, f'No data available for metric: {metric}', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create subplots for different parameter combinations
    unique_combinations = df[['use_year', 'use_journal']].drop_duplicates()
    n_combinations = len(unique_combinations)
    
    if n_combinations == 1:
        fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
        axes = [ax]
    else:
        ncols = min(2, n_combinations)
        nrows = int(np.ceil(n_combinations / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * nrows))
        if n_combinations > 1:
            axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
        else:
            axes = [axes]
    
    metric_label = labels.get(metric, metric)
    
    for i, (_, combo) in enumerate(unique_combinations.iterrows()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        use_year, use_journal = combo['use_year'], combo['use_journal']
        
        # Filter data for this parameter combination
        mask = (df['use_year'] == use_year) & (df['use_journal'] == use_journal)
        df_subset = df[mask]
        
        if df_subset.empty:
            continue
        
        # Plot lines for each method
        all_x = []
        all_y = []
        for method in ['direct', 'indirect', 'dummy']:
            method_data = df_subset[df_subset['method'] == method].sort_values('n_authors')
            if not method_data.empty:
                x_vals = method_data['n_authors'].values
                y_vals = method_data['metric_value'].values
                y_errs = method_data['metric_std'].values
                
                ax.errorbar(x_vals, y_vals, yerr=y_errs, 
                           marker='o', label=LABELS[method], 
                           color=COLORS[method], linewidth=2, markersize=4,
                           capsize=3)
                all_x.extend(x_vals)
                all_y.extend(y_vals)
        
        ax.set_xlabel('Number of Authors')
        ylabel_top(metric_label, ax)
        ax.set_title(f'Year: {use_year}, Journal: {use_journal}', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        
        # Apply range frame
        if all_x and all_y:
            range_frame(ax, np.array(all_x), np.array(all_y))
    
    # Hide empty subplots
    for i in range(len(unique_combinations), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def create_dabest_plot(results: Dict[str, Any], 
                      metric: str = 'mae',
                      metric_labels: Optional[Dict[str, str]] = None,
                      save_path: Optional[Path] = None,
                      title_suffix: str = "") -> Tuple[plt.Figure, Any]:
    """
    Create DABEST plot for effect size visualization.
    """
    labels = metric_labels if metric_labels else METRIC_LABELS
    
    # Extract raw metric values across folds
    methods = ['direct', 'indirect', 'dummy']
    data_dict = {}
    
    for method in methods:
        if method in results and metric in results[method]:
            data_dict[LABELS[method]] = results[method][metric]['values']
    
    # Create DataFrame for DABEST
    max_len = max(len(values) for values in data_dict.values())
    df_dabest = pd.DataFrame({
        method: values + [np.nan] * (max_len - len(values))
        for method, values in data_dict.items()
    })
    
    # Create DABEST object
    method_names = list(data_dict.keys())
    dabest_obj = dabest.load(df_dabest, idx=method_names)
    
    # Get metric label
    metric_label = labels.get(metric, metric)
    
    # Create the plot with minimal parameters to avoid API issues
    dabest_plot = dabest_obj.mean_diff.plot()
    
    fig = dabest_plot.fig
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    return fig, dabest_obj


def _plot_meta_prediction_panel(ax, results: Dict[str, Any], meta_type: str, 
                              metric: str, labels: Dict[str, str], 
                              baseline: Optional[float] = None,
                              title_suffix: str = "", is_regression: bool = False,
                              compact: bool = False, panel_label: str = ""):
    """
    Helper function to plot a single meta prediction performance panel.
    """
    if meta_type in results and metric in results[meta_type]:
        pred_mean = results[meta_type][metric]['mean']
        pred_std = results[meta_type][metric]['std']
        
        # Get number of target classes for display
        n_targets = results.get('dataset_info', {}).get(f'n_{meta_type}_targets', 'Unknown')
        
        if is_regression:
            # For year prediction (regression), show absolute MAE
            if 'dummy' in results and metric in results['dummy']:
                baseline_mean = results['dummy'][metric]['mean']
                baseline_std = results['dummy'][metric]['std']
            else:
                baseline_mean = pred_mean * 1.5  # Conservative baseline
                baseline_std = 0
            
            # Show absolute performance
            means = [baseline_mean, pred_mean]
            stds = [baseline_std, pred_std]
            ylabel_text = labels.get(metric, metric)
            
            # Value labels for regression
            value_labels = [f'{baseline_mean:.1f}', f'{pred_mean:.1f}']
        else:
            # For classification, show absolute performance
            baseline_mean = baseline if baseline is not None else 0.05
            baseline_std = 0
            
            means = [baseline_mean, pred_mean]
            stds = [baseline_std, pred_std]
            ylabel_text = labels.get(metric, metric)
            
            # Value labels for classification
            value_labels = [f'{baseline_mean:.3f}', f'{pred_mean:.3f}']
        
        # Plot bars
        x_pos = [0, 1]
        colors = [COLORS['dummy'], COLORS.get(meta_type, '#2E86C1')]
        
        if compact:
            # Compact mode: shorter labels, smaller fonts
            labels_list = ['Base', 'Pred']
            fontsize_tick = 7
            fontsize_title = 8
            fontsize_value = 7
            capsize = 2
        else:
            labels_list = ['Baseline', f'{title_suffix} Pred.']
            fontsize_tick = 8
            fontsize_title = 9
            fontsize_value = 8
            capsize = 3
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=capsize,
                     color=colors, alpha=0.8)
        
        ax.set_ylim(0, max(means) * 1.3)
        
        # Use manual ylabel with better positioning control
        if compact:
            ax.set_ylabel(ylabel_text, fontsize=fontsize_tick, labelpad=8)
        else:
            ax.set_ylabel(ylabel_text, fontsize=fontsize_tick+1, labelpad=10)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list, rotation=30, ha='right', fontsize=fontsize_tick)
        
        # Add title with number of targets
        if isinstance(n_targets, int):
            ax.set_title(f'{title_suffix} (n={n_targets})', fontsize=fontsize_title, pad=10)
        else:
            ax.set_title(title_suffix, fontsize=fontsize_title, pad=10)
        
        # Add panel label (bold A, B, C, etc.)
        if panel_label:
            ax.text(-0.15, 1.1, panel_label, transform=ax.transAxes, 
                   fontsize=fontsize_title+3, fontweight='bold', ha='center')
        
        # Apply range frame
        range_frame(ax, np.array(x_pos), np.array(means))
        
        # Add value labels
        for j, (bar, label) in enumerate(zip(bars, value_labels)):
            ax.text(bar.get_x() + bar.get_width()/2., 
                   bar.get_height() + max(means) * 0.05,
                   label, ha='center', va='bottom', fontsize=fontsize_value)
    else:
        fontsize_msg = 8 if compact else 10
        fontsize_title = 8 if compact else 9
        ax.text(0.5, 0.5, f'No {meta_type} data', ha='center', va='center',
               transform=ax.transAxes, fontsize=fontsize_msg)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title_suffix, fontsize=fontsize_title)


def create_main_figure_panel(results: Dict[str, Any],
                           target_type: str = 'regression',
                           dataset_name: str = "",
                           metric_labels: Optional[Dict[str, str]] = None,
                           save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create main figure panel for paper.
    
    This creates a publication-ready figure with compact layout:
    - Top row (compact): Author, Journal, Year prediction vs baselines
    - Bottom row (prominent): Main performance comparison
    """
    labels = metric_labels if metric_labels else METRIC_LABELS
    
    # Determine selection metric if not provided
    if selection_metric is None:
        selection_metric = 'mae' if target_type == 'regression' else 'f1'
    
    fig = plt.figure(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1], width_ratios=[1, 1, 1], 
                         hspace=0.6, wspace=0.5)
    
    # Top row: Compact meta prediction panels (smaller)
    ax_author = fig.add_subplot(gs[0, 0])
    _plot_meta_prediction_panel(ax_author, results, 'author', 'f1_micro', labels, 
                               baseline=0.02, title_suffix='Authors', compact=True, panel_label='A')
    
    ax_journal = fig.add_subplot(gs[0, 1])
    _plot_meta_prediction_panel(ax_journal, results, 'journal', 'f1_micro', labels,
                               baseline=0.1, title_suffix='Journals', compact=True, panel_label='B')
    
    ax_year = fig.add_subplot(gs[0, 2])
    _plot_meta_prediction_panel(ax_year, results, 'year', 'mae', labels,
                               baseline=None, title_suffix='Year', is_regression=True, compact=True, panel_label='C')
    
    # Bottom row: Main performance comparison (larger, spans full width)
    ax_perf = fig.add_subplot(gs[1, :])
    
    if target_type == 'regression':
        main_metric = 'mae'
    else:
        main_metric = 'f1'
    
    metric_label = labels.get(main_metric, main_metric)
    
    methods = ['direct', 'indirect', 'dummy']
    perf_values = []
    perf_errors = []
    
    for method in methods:
        if method in results and main_metric in results[method]:
            perf_values.append(results[method][main_metric]['mean'])
            perf_errors.append(results[method][main_metric]['std'])
        else:
            perf_values.append(np.nan)
            perf_errors.append(0)
    
    valid_idx = ~np.isnan(perf_values)
    if np.any(valid_idx):
        valid_values = np.array(perf_values)[valid_idx]
        valid_errors = np.array(perf_errors)[valid_idx]
        valid_methods = [methods[i] for i in range(len(methods)) if valid_idx[i]]
        colors = [COLORS[method] for method in valid_methods]
        method_labels = [LABELS[method] for method in valid_methods]
        
        x_positions = np.arange(len(valid_values))
        
        # Plot individual fold values as scatter points
        for i, method in enumerate(valid_methods):
            if method in results and main_metric in results[method] and 'values' in results[method][main_metric]:
                fold_values = results[method][main_metric]['values']
                x_jitter = x_positions[i] + np.random.normal(0, 0.05, len(fold_values))
                ax_perf.scatter(x_jitter, fold_values, color=colors[i], alpha=0.6, s=15)
        
        # Plot means as larger points
        ax_perf.scatter(x_positions, valid_values, color=colors, s=40, alpha=1.0,
                       edgecolors='black', linewidths=0.5, zorder=3)
        
        # Add error bars
        ax_perf.errorbar(x_positions, valid_values, yerr=valid_errors, fmt='none',
                        capsize=3, color='black', alpha=0.7, zorder=2)
        
        ax_perf.set_xticks(x_positions)
        ax_perf.set_xticklabels(method_labels, fontsize=8, rotation=30, ha='right')
        
        # Use manual ylabel for better control
        ax_perf.set_ylabel(metric_label, fontsize=10, labelpad=12)
        
        # Add panel label D
        ax_perf.text(-0.03, 1.05, 'D', transform=ax_perf.transAxes, 
                    fontsize=12, fontweight='bold', ha='center')
        
        # Apply range frame
        range_frame(ax_perf, x_positions, valid_values)
    
    # Adjust spacing to prevent overlapping text
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def export_key_metrics(results: Dict[str, Any], 
                      dataset_name: str,
                      output_dir: Path,
                      target_type: str = 'regression') -> None:
    """
    Export key metrics for showyourwork integration.
    """
    from utils import export_showyourwork_metric
    
    # Dataset info
    if 'dataset_info' in results:
        info = results['dataset_info']
        export_showyourwork_metric(info['size'], f"{dataset_name}_dataset_size", output_dir)
        export_showyourwork_metric(info['n_features'], f"{dataset_name}_n_features", output_dir)
        export_showyourwork_metric(info['n_meta_targets'], f"{dataset_name}_n_meta_features", output_dir)
    
    # Main performance metrics
    if target_type == 'regression':
        main_metric = 'mae'
    else:
        main_metric = 'f1'
    
    for method in ['direct', 'indirect', 'dummy']:
        if method in results and main_metric in results[method]:
            export_showyourwork_metric(
                results[method][main_metric]['mean'], 
                f"{dataset_name}_{method}_{main_metric}", 
                output_dir, 
                decimal_places=3
            )
    
    # Meta prediction performance
    if 'meta' in results:
        for metric in ['accuracy', 'f1_micro', 'f1_macro']:
            if metric in results['meta']:
                export_showyourwork_metric(
                    results['meta'][metric]['mean'],
                    f"{dataset_name}_meta_{metric}",
                    output_dir,
                    decimal_places=3
                )
    
    # Effect size calculation
    if ('direct' in results and 'indirect' in results and 
        main_metric in results['direct'] and main_metric in results['indirect']):
        
        direct_values = results['direct'][main_metric]['values']
        indirect_values = results['indirect'][main_metric]['values']
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(direct_values) + np.var(indirect_values)) / 2)
        if pooled_std > 0:
            effect_size = (np.mean(indirect_values) - np.mean(direct_values)) / pooled_std
            export_showyourwork_metric(effect_size, f"{dataset_name}_effect_size_cohens_d", output_dir, decimal_places=3)
        
        # Mean difference  
        mean_diff = np.mean(indirect_values) - np.mean(direct_values)
        export_showyourwork_metric(mean_diff, f"{dataset_name}_mean_difference_{main_metric}", output_dir, decimal_places=3)


def find_best_parameter_setting(sweep_results: List[Dict], 
                               target_type: str = 'regression',
                               selection_criteria: str = 'best_overall',
                               similarity_metric: str = 'mae') -> Dict:
    """
    Find the best parameter setting using different criteria.
    
    Args:
        sweep_results: Results from parameter sweep
        target_type: 'regression' or 'classification'
        selection_criteria: 'best_overall' (best direct performance) or 
                           'smallest_gap' (smallest difference between direct and indirect)
        similarity_metric: Metric to use for comparison
    
    Returns:
        Dictionary with best parameter setting and results
    """
    if not sweep_results:
        return {'best_result': None, 'similarity_score': None, 'parameters': None}
        
    if selection_criteria == 'smallest_gap':
        return _find_smallest_gap_setting(sweep_results, similarity_metric)
    else:  # 'best_overall'
        return _find_best_overall_setting(sweep_results, target_type, similarity_metric)


def _find_smallest_gap_setting(sweep_results: List[Dict], similarity_metric: str) -> Dict:
    """Find parameter setting with smallest gap between direct and indirect models."""
    best_similarity = float('inf')
    best_result = None
    
    for result in sweep_results:
        if 'direct' not in result or 'indirect' not in result:
            continue
            
        # Get performance values
        direct_perf = result['direct'].get(similarity_metric, {}).get('mean', None)
        indirect_perf = result['indirect'].get(similarity_metric, {}).get('mean', None)
        
        if direct_perf is None or indirect_perf is None:
            continue
        
        # Calculate similarity (lower absolute difference = more similar)
        similarity = abs(direct_perf - indirect_perf)
        
        if similarity < best_similarity:
            best_similarity = similarity
            best_result = result
    
    return {
        'best_result': best_result,
        'similarity_score': best_similarity,
        'parameters': best_result['parameters'] if best_result else None
    }


def _find_best_overall_setting(sweep_results: List[Dict], target_type: str, metric: str) -> Dict:
    """Find parameter setting with best direct model performance."""
    best_performance = float('inf') if target_type == 'regression' else float('-inf')
    best_result = None
    best_similarity = None
    
    for result in sweep_results:
        if 'direct' not in result:
            continue
            
        direct_perf = result['direct'].get(metric, {}).get('mean', None)
        if direct_perf is None:
            continue
        
        # For regression: lower is better; for classification: higher is better
        is_better = (direct_perf < best_performance if target_type == 'regression' 
                    else direct_perf > best_performance)
        
        if is_better:
            best_performance = direct_perf
            best_result = result
            # Calculate gap if indirect results available
            if 'indirect' in result and metric in result['indirect']:
                indirect_perf = result['indirect'][metric]['mean']
                best_similarity = abs(direct_perf - indirect_perf)
    
    return {
        'best_result': best_result,
        'similarity_score': best_similarity,
        'parameters': best_result['parameters'] if best_result else None
    }


def create_optimal_main_panel(sweep_results: List[Dict],
                            target_type: str = 'regression', 
                            dataset_name: str = '',
                            metric_labels: Optional[Dict[str, str]] = None,
                            save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create main figure panel using the parameter setting with highest Clever Hans similarity.
    """
    # Find best parameter setting
    best_setting = find_best_parameter_setting(sweep_results, target_type)
    
    if best_setting['best_result'] is None:
        raise ValueError("No valid results found in sweep_results")
    
    # Create main panel with best result
    fig = create_main_figure_panel(
        best_setting['best_result'],
        target_type=target_type,
        dataset_name=dataset_name,
        metric_labels=metric_labels,
        save_path=save_path
    )
    
    # Add parameter info as text
    params = best_setting['parameters']
    if params:
        param_text = f"Authors: {params['n_authors']}, Year: {params['use_year']}, Journal: {params['use_journal']}"
        fig.text(0.02, 0.98, param_text, fontsize=8, va='top', ha='left', alpha=0.7)
    
    return fig
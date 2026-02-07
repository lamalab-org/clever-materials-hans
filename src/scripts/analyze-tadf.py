import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import paths
    import lama_aesthetics
    from utils import (
        run_parameter_sweep_analysis,
        run_single_analysis,
        run_meta_comparison_analysis,
        export_showyourwork_metric,
    )
    from plotting_utils import (
        plot_performance_comparison,
        plot_meta_performance,
        plot_parameter_sweep_results,
        create_dabest_plot,
        create_main_figure_panel,
        create_main_figure_panel_with_meta_comparison,
        export_key_metrics,
        METRIC_LABELS,
    )
    import numpy as np

    lama_aesthetics.get_style("main")
    return (
        METRIC_LABELS,
        create_main_figure_panel,
        create_main_figure_panel_with_meta_comparison,
        export_key_metrics,
        np,
        paths,
        pd,
        plot_meta_performance,
        plot_parameter_sweep_results,
        plot_performance_comparison,
        run_meta_comparison_analysis,
        run_parameter_sweep_analysis,
        run_single_analysis,
    )


@app.cell
def _(paths, pd):
    df = pd.read_parquet(paths.output / "tadf_preprocess.parquet")
    return (df,)


@app.cell
def _(df, np):
    features = [f for f in df.columns if f.startswith("feat_")]

    df_with_authors = df.dropna(subset="authors_full_list")

    raw_value_float = []
    for i, row in df_with_authors.iterrows():
        try:
            number = float(row["raw_value"])
        except Exception:
            number = np.nan
        raw_value_float.append(number)

    df_with_authors["raw_value_float"] = raw_value_float

    df_final = df_with_authors.dropna(subset=["raw_value_float"])

    print(f"Dataset shape: {df_final.shape}")
    print(f"Number of features: {len(features)}")

    target_column = "raw_value_float"
    return df_final, target_column


@app.cell
def _(df_final, run_single_analysis, target_column):
    # Run single analysis with default parameters
    results_default = run_single_analysis(
        df_final,
        target_column,
        target_type="regression",
        n_authors=500,
        use_year=True,
        use_journal=True,
        n_folds=10,
    )
    return (results_default,)


@app.cell
def _(METRIC_LABELS, paths, plot_performance_comparison, results_default):
    # Custom labels for TADF dataset
    tadf_labels = METRIC_LABELS.copy()
    tadf_labels["mae"] = "MAE (eV)"

    fig_performance = plot_performance_comparison(
        results_default,
        target_type="regression",
        metric_labels=tadf_labels,
        save_path=paths.figures / "tadf_performance_comparison.pdf",
        title_suffix="TADF Dataset",
    )
    fig_performance.show()
    return fig_performance, tadf_labels


@app.cell
def _(fig_performance):
    fig_performance
    return


@app.cell
def _(df_final, run_parameter_sweep_analysis, target_column):
    # Run parameter sweep across different author counts
    sweep_results = run_parameter_sweep_analysis(
        df_final,
        target_column,
        target_type="regression",
        dataset_name="TADF",
        author_counts=[10, 50, 100, 1000],
        use_year_options=[True, False],
        use_journal_options=[True, False],  # No journal data for TADF
        n_folds=5,  # Reduced for faster computation
    )
    return (sweep_results,)


@app.cell
def _(paths, plot_meta_performance, results_default):
    fig_meta = plot_meta_performance(
        results_default,
        save_path=paths.figures / "tadf_meta_performance.pdf",
        title_suffix="TADF Dataset",
    )
    fig_meta
    return


@app.cell
def _(paths, plot_parameter_sweep_results, sweep_results, tadf_labels):
    fig_sweep = plot_parameter_sweep_results(
        sweep_results,
        metric="mae",
        metric_labels=tadf_labels,
        save_path=paths.figures / "tadf_parameter_sweep.pdf",
        title_suffix="TADF Dataset",
    )
    fig_sweep
    return


@app.cell
def _(create_main_figure_panel, paths, sweep_results, tadf_labels):
    from plotting_utils import find_best_parameter_setting

    best_setting = find_best_parameter_setting(
        sweep_results,
        target_type="regression",
        selection_criteria="smallest_gap",
        similarity_metric="mae",
    )

    fig_main = create_main_figure_panel(
        best_setting["best_result"],
        target_type="regression",
        dataset_name="TADF",
        metric_labels=tadf_labels,
        save_path=paths.figures / "tadf_main_panel.pdf",
    )
    fig_main
    return (best_setting,)


@app.cell
def _(df_final, run_meta_comparison_analysis, target_column):
    # Run meta comparison analysis for four-column main panel
    meta_comparison = run_meta_comparison_analysis(
        df_final,
        target_column,
        target_type="regression",
        n_authors=1000,
        use_year=True,
        use_journal=True,
        n_folds=10,
    )
    
    # Quick comparison summary
    pred_mae = meta_comparison['predicted_meta']['indirect']['mae']['mean']
    actual_mae = meta_comparison['actual_meta']['indirect']['mae']['mean']
    direct_mae = meta_comparison['predicted_meta']['direct']['mae']['mean']
    
    print(f"TADF Meta-information comparison:")
    print(f"Direct (Conventional): {direct_mae:.2f} MAE")
    print(f"Predicted meta: {pred_mae:.2f} MAE") 
    print(f"Actual meta: {actual_mae:.2f} MAE")
    print(f"Performance gap: {abs(pred_mae - actual_mae):.2f} MAE")
    
    return meta_comparison,


@app.cell
def _(create_main_figure_panel_with_meta_comparison, meta_comparison, paths, tadf_labels):
    # Create four-column main panel with actual meta results
    fig_main_four_col = create_main_figure_panel_with_meta_comparison(
        meta_comparison,
        target_type="regression",
        dataset_name="TADF",
        metric_labels=tadf_labels,
        save_path=paths.figures / "tadf_main_panel_four_columns.pdf",
        selection_metric="mae"
    )
    fig_main_four_col
    return (fig_main_four_col,)


@app.cell
def _(export_key_metrics, paths, results_default):
    export_key_metrics(
        results_default,
        dataset_name="tadf",
        output_dir=paths.output,
        target_type="regression",
    )
    return


@app.cell
def _(best_setting, export_key_metrics, paths):
    export_key_metrics(
        best_setting["best_result"],
        dataset_name="tadf_best",
        output_dir=paths.output,
        target_type="regression",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

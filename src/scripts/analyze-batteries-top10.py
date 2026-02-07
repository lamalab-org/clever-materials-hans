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
    df = pd.read_parquet(paths.output / "battery_preprocessed.parquet")
    return (df,)


@app.cell
def _(df, np):
    df_with_capacity = df.dropna(subset=["authors_full_list", "Value"])

    # Calculate the 90th percentile (top 10%) threshold for battery capacity
    capacity_threshold = np.percentile(df_with_capacity["Value"], 90)
    print(f"Top 10% battery capacity threshold: {capacity_threshold:.1f}")

    # Create binary classification target
    df_with_capacity["is_top10_capacity"] = (
        df_with_capacity["Value"] >= capacity_threshold
    ).astype(int)

    # Show class distribution
    class_counts = df_with_capacity["is_top10_capacity"].value_counts()
    print(f"Class distribution:")
    print(
        f"  Not top 10% (0): {class_counts[0]} samples ({class_counts[0] / len(df_with_capacity) * 100:.1f}%)"
    )
    print(
        f"  Top 10% (1): {class_counts[1]} samples ({class_counts[1] / len(df_with_capacity) * 100:.1f}%)"
    )

    df_final = df_with_capacity
    print(f"Dataset shape: {df_final.shape}")
    print(
        f"Number of features: {len([f for f in df_final.columns if f.startswith('feat_')])}"
    )

    target_column = "is_top10_capacity"
    return capacity_threshold, df_final, target_column


@app.cell
def _(df_final, run_single_analysis, target_column):
    results_default = run_single_analysis(
        df_final,
        target_column,
        target_type="classification",
        n_authors=500,
        use_year=True,
        use_journal=True,
        n_folds=10,
    )
    return (results_default,)


@app.cell
def _(METRIC_LABELS, paths, plot_performance_comparison, results_default):
    battery_top10_labels = METRIC_LABELS.copy()

    fig_performance = plot_performance_comparison(
        results_default,
        target_type="classification",
        metric_labels=battery_top10_labels,
        save_path=paths.figures / "battery_top10_performance_comparison.pdf",
        title_suffix="Battery Top 10% Capacity Dataset",
    )
    fig_performance.show()
    return battery_top10_labels, fig_performance


@app.cell
def _(fig_performance):
    fig_performance
    return


@app.cell
def _(df_final, run_parameter_sweep_analysis, target_column):
    sweep_results = run_parameter_sweep_analysis(
        df_final,
        target_column,
        target_type="classification",
        dataset_name="Battery_Top10",
        author_counts=[10, 50, 100, 1000],
        use_year_options=[True, False],
        use_journal_options=[True, False],
        n_folds=5,
    )
    return (sweep_results,)


@app.cell
def _(paths, plot_meta_performance, results_default):
    fig_meta = plot_meta_performance(
        results_default,
        save_path=paths.figures / "battery_top10_meta_performance.pdf",
        title_suffix="Battery Top 10% Capacity Dataset",
    )
    fig_meta
    return


@app.cell
def _(
    battery_top10_labels,
    paths,
    plot_parameter_sweep_results,
    sweep_results,
):
    fig_sweep = plot_parameter_sweep_results(
        sweep_results,
        metric="f1",
        metric_labels=battery_top10_labels,
        save_path=paths.figures / "battery_top10_parameter_sweep.pdf",
        title_suffix="Battery Top 10% Capacity Dataset",
    )
    fig_sweep
    return


@app.cell
def _(battery_top10_labels, create_main_figure_panel, paths, sweep_results):
    from plotting_utils import find_best_parameter_setting

    best_setting = find_best_parameter_setting(
        sweep_results,
        target_type="classification",
        selection_criteria="smallest_gap",
        similarity_metric="f1",
    )

    fig_main = create_main_figure_panel(
        best_setting["best_result"],
        target_type="classification",
        dataset_name="Battery Top 10% Capacity",
        metric_labels=battery_top10_labels,
        save_path=paths.figures / "battery_top10_main_panel.pdf",
    )
    fig_main
    return


@app.cell
def _(df_final, run_meta_comparison_analysis, target_column):
    # Run meta comparison analysis for four-column main panel
    meta_comparison = run_meta_comparison_analysis(
        df_final,
        target_column,
        target_type="classification",
        n_authors=500,
        use_year=True,
        use_journal=True,
        n_folds=10,
    )

    # Quick comparison summary
    pred_f1 = meta_comparison["predicted_meta"]["indirect"]["f1"]["mean"]
    actual_f1 = meta_comparison["actual_meta"]["indirect"]["f1"]["mean"]
    direct_f1 = meta_comparison["predicted_meta"]["direct"]["f1"]["mean"]

    print(f"Battery Meta-information comparison:")
    print(f"Direct (Conventional): {direct_f1:.3f} F1")
    print(f"Predicted meta: {pred_f1:.3f} F1")
    print(f"Actual meta: {actual_f1:.3f} F1")
    print(f"Performance gap: {abs(pred_f1 - actual_f1):.3f} F1")
    return (meta_comparison,)


@app.cell
def _(
    battery_top10_labels,
    create_main_figure_panel_with_meta_comparison,
    meta_comparison,
    paths,
):
    # Create four-column main panel with actual meta results
    fig_main_four_col = create_main_figure_panel_with_meta_comparison(
        meta_comparison,
        target_type="classification",
        dataset_name="Battery Top 10% Capacity",
        metric_labels=battery_top10_labels,
        save_path=paths.figures / "battery_top10_main_panel_four_columns.pdf",
        selection_metric="f1",
    )
    fig_main_four_col
    return


@app.cell
def _(export_key_metrics, paths, results_default):
    export_key_metrics(
        results_default,
        dataset_name="battery_top10",
        output_dir=paths.output,
        target_type="classification",
    )
    return


@app.cell
def _(capacity_threshold, df_final):
    # Show some statistics about the top 10% threshold
    print(f"Battery capacity threshold for top 10%: {capacity_threshold:.1f}")
    print(
        f"Range of top 10% capacity values: {df_final[df_final['is_top10_capacity'] == 1]['Value'].min():.1f} - {df_final[df_final['is_top10_capacity'] == 1]['Value'].max():.1f}"
    )
    print(
        f"Range of bottom 90% capacity values: {df_final[df_final['is_top10_capacity'] == 0]['Value'].min():.1f} - {df_final[df_final['is_top10_capacity'] == 0]['Value'].max():.1f}"
    )

    # Show some interesting statistics
    print(f"\nMean battery capacity:")
    print(
        f"  Top 10%: {df_final[df_final['is_top10_capacity'] == 1]['Value'].mean():.1f}"
    )
    print(
        f"  Bottom 90%: {df_final[df_final['is_top10_capacity'] == 0]['Value'].mean():.1f}"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

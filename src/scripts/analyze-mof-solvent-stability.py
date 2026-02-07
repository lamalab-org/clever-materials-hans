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
    df = pd.read_parquet(paths.output / "mof_solvent_stability.parquet")
    return (df,)


@app.cell
def _(df):
    features = [f for f in df.columns if f.startswith("feat_")]
    df_final = df.dropna(
        subset=["authors_full_list", "assigned_solvent_removal_stability"]
        + features
    )

    print(f"Dataset shape: {df_final.shape}")
    print(f"Number of features: {len(features)}")

    target_column = "assigned_solvent_removal_stability"
    return df_final, target_column


@app.cell
def _(df_final, run_single_analysis, target_column):
    results_default = run_single_analysis(
        df_final,
        target_column,
        target_type="classification",
        n_authors=50,
        use_year=True,
        use_journal=True,
        n_folds=10,
    )
    return (results_default,)


@app.cell
def _(METRIC_LABELS, paths, plot_performance_comparison, results_default):
    mof_solvent_labels = METRIC_LABELS.copy()

    fig_performance = plot_performance_comparison(
        results_default,
        target_type="classification",
        metric_labels=mof_solvent_labels,
        save_path=paths.figures / "mof_solvent_performance_comparison.pdf",
        title_suffix="MOF Solvent Stability Dataset",
    )
    fig_performance.show()
    return fig_performance, mof_solvent_labels


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
        dataset_name="MOF_Solvent",
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
        save_path=paths.figures / "mof_solvent_meta_performance.pdf",
        title_suffix="MOF Solvent Stability Dataset",
    )
    fig_meta
    return


@app.cell
def _(mof_solvent_labels, paths, plot_parameter_sweep_results, sweep_results):
    fig_sweep = plot_parameter_sweep_results(
        sweep_results,
        metric="f1",
        metric_labels=mof_solvent_labels,
        save_path=paths.figures / "mof_solvent_parameter_sweep.pdf",
        title_suffix="MOF Solvent Stability Dataset",
    )
    fig_sweep
    return


@app.cell
def _(create_main_figure_panel, mof_solvent_labels, paths, sweep_results):
    from plotting_utils import find_best_parameter_setting

    best_setting = find_best_parameter_setting(
        sweep_results,
        target_type="classification",
        selection_criteria="smallest_gap",
        similarity_metric="accuracy",
    )

    fig_main = create_main_figure_panel(
        best_setting["best_result"],
        target_type="classification",
        dataset_name="MOF Solvent Stability",
        metric_labels=mof_solvent_labels,
        save_path=paths.figures / "mof_solvent_main_panel.pdf",
        selection_metric="accuracy",
    )
    fig_main
    return (best_setting,)


@app.cell
def _(df_final, run_meta_comparison_analysis, target_column):
    # Run meta comparison analysis for four-column main panel
    meta_comparison = run_meta_comparison_analysis(
        df_final,
        target_column,
        target_type="classification",
        n_authors=50,
        use_year=True,
        use_journal=True,
        n_folds=10,
    )

    # Quick comparison summary
    pred_acc = meta_comparison["predicted_meta"]["indirect"]["accuracy"]["mean"]
    actual_acc = meta_comparison["actual_meta"]["indirect"]["accuracy"]["mean"]
    direct_acc = meta_comparison["predicted_meta"]["direct"]["accuracy"]["mean"]

    print(f"MOF Solvent Meta-information comparison:")
    print(f"Direct (Conventional): {direct_acc:.3f} Accuracy")
    print(f"Predicted meta: {pred_acc:.3f} Accuracy")
    print(f"Actual meta: {actual_acc:.3f} Accuracy")
    print(f"Performance gap: {abs(pred_acc - actual_acc):.3f} Accuracy")
    return (meta_comparison,)


@app.cell
def _(
    create_main_figure_panel_with_meta_comparison,
    meta_comparison,
    mof_solvent_labels,
    paths,
):
    # Create four-column main panel with actual meta results
    fig_main_four_col = create_main_figure_panel_with_meta_comparison(
        meta_comparison,
        target_type="classification",
        dataset_name="MOF Solvent Stability",
        metric_labels=mof_solvent_labels,
        save_path=paths.figures / "mof_solvent_main_panel_four_columns.pdf",
        selection_metric="accuracy",
    )
    fig_main_four_col
    return


@app.cell
def _(best_setting, export_key_metrics, paths):
    export_key_metrics(
        best_setting["best_result"],
        dataset_name="mof_solvent",
        output_dir=paths.output,
        target_type="classification",
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

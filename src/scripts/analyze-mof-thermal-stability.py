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
        export_showyourwork_metric,
    )
    from plotting_utils import (
        plot_performance_comparison,
        plot_meta_performance,
        plot_parameter_sweep_results,
        create_main_figure_panel,
        export_key_metrics,
        METRIC_LABELS,
    )
    import numpy as np

    lama_aesthetics.get_style("main")
    return (
        METRIC_LABELS,
        create_main_figure_panel,
        export_key_metrics,
        paths,
        pd,
        plot_meta_performance,
        plot_parameter_sweep_results,
        plot_performance_comparison,
        run_parameter_sweep_analysis,
        run_single_analysis,
    )


@app.cell
def _(paths, pd):
    df = pd.read_parquet(paths.output / "mof_thermal_stability.parquet")
    return (df,)


@app.cell
def _(df):
    df_final = df.dropna(subset=["authors_full_list", "assigned_T_decomp"])

    print(f"Dataset shape: {df_final.shape}")
    print(
        f"Number of features: {len([f for f in df_final.columns if f.startswith('feat_')])}"
    )

    target_column = "assigned_T_decomp"
    return df_final, target_column


@app.cell
def _(df_final, run_single_analysis, target_column):
    results_default = run_single_analysis(
        df_final,
        target_column,
        target_type="regression",
        n_authors=50,
        use_year=True,
        use_journal=True,
        n_folds=10,
    )
    return (results_default,)


@app.cell
def _(METRIC_LABELS, paths, plot_performance_comparison, results_default):
    mof_thermal_labels = METRIC_LABELS.copy()
    mof_thermal_labels["mae"] = "MAE (°C)"

    fig_performance = plot_performance_comparison(
        results_default,
        target_type="regression",
        metric_labels=mof_thermal_labels,
        save_path=paths.figures / "mof_thermal_performance_comparison.pdf",
        title_suffix="MOF Thermal Stability Dataset",
    )
    fig_performance.show()
    return fig_performance, mof_thermal_labels


@app.cell
def _(fig_performance):
    fig_performance
    return


@app.cell
def _(df_final, run_parameter_sweep_analysis, target_column):
    sweep_results = run_parameter_sweep_analysis(
        df_final,
        target_column,
        target_type="regression",
        dataset_name="MOF_Thermal",
        author_counts=[10, 50, 100, 1000],
        use_year_options=[True, False],
        use_journal_options=[False, True],
        n_folds=5,
    )
    return (sweep_results,)


@app.cell
def _(paths, plot_meta_performance, results_default):
    fig_meta = plot_meta_performance(
        results_default,
        save_path=paths.figures / "mof_thermal_meta_performance.pdf",
        title_suffix="MOF Thermal Stability Dataset",
    )
    fig_meta
    return


@app.cell
def _(mof_thermal_labels, paths, plot_parameter_sweep_results, sweep_results):
    fig_sweep = plot_parameter_sweep_results(
        sweep_results,
        metric="mae",
        metric_labels=mof_thermal_labels,
        save_path=paths.figures / "mof_thermal_parameter_sweep.pdf",
        title_suffix="MOF Thermal Stability Dataset",
    )
    fig_sweep
    return


@app.cell
def _(create_main_figure_panel, mof_thermal_labels, paths, sweep_results):
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
        dataset_name="MOF Thermal Stability",
        metric_labels=mof_thermal_labels,
        save_path=paths.figures / "mof_thermal_main_panel.pdf",
        selection_metric="mae",
    )
    fig_main
    return (best_setting,)


@app.cell
def _(best_setting, export_key_metrics, paths):
    export_key_metrics(
        best_setting["best_result"],
        dataset_name="mof_thermal",
        output_dir=paths.output,
        target_type="regression",
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

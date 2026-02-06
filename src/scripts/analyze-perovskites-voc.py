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
    df = pd.read_parquet(paths.output / "perovskite_processed.parquet")
    return (df,)


@app.cell
def _(df):
    features = [f for f in df.columns if f.startswith("feat_")]
    df_final = df.dropna(
        subset=["authors_full_list", "data.jv.default_Voc"] + features
    )

    print(f"Dataset shape: {df_final.shape}")
    print(f"Number of features: {len(features)}")

    target_column = "data.jv.default_Voc"
    return df_final, target_column


@app.cell
def _(df_final, run_single_analysis, target_column):
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
    perovskite_voc_labels = METRIC_LABELS.copy()
    perovskite_voc_labels["mae"] = "MAE (V)"

    fig_performance = plot_performance_comparison(
        results_default,
        target_type="regression",
        metric_labels=perovskite_voc_labels,
        save_path=paths.figures / "perovskite_voc_performance_comparison.pdf",
        title_suffix="Perovskite Voc Dataset",
    )
    fig_performance.show()
    return fig_performance, perovskite_voc_labels


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
        dataset_name="Perovskite_Voc",
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
        save_path=paths.figures / "perovskite_voc_meta_performance.pdf",
        title_suffix="Perovskite Voc Dataset",
    )
    fig_meta
    return


@app.cell
def _(paths, perovskite_voc_labels, plot_parameter_sweep_results, sweep_results):
    fig_sweep = plot_parameter_sweep_results(
        sweep_results,
        metric="mae",
        metric_labels=perovskite_voc_labels,
        save_path=paths.figures / "perovskite_voc_parameter_sweep.pdf",
        title_suffix="Perovskite Voc Dataset",
    )
    fig_sweep
    return


@app.cell
def _(create_main_figure_panel, paths, perovskite_voc_labels, sweep_results):
    from plotting_utils import find_best_parameter_setting
    
    best_setting = find_best_parameter_setting(
        sweep_results,
        target_type="regression",
        selection_criteria="best_overall",
        similarity_metric="mae"
    )
    
    fig_main = create_main_figure_panel(
        best_setting["best_result"],
        target_type="regression",
        dataset_name="Perovskite Voc",
        metric_labels=perovskite_voc_labels,
        save_path=paths.figures / "perovskite_voc_main_panel.pdf",
        selection_metric="mae",
    )
    fig_main
    return


@app.cell
def _(export_key_metrics, paths, results_default):
    export_key_metrics(
        results_default,
        dataset_name="perovskite_voc",
        output_dir=paths.output,
        target_type="regression",
    )
    return


if __name__ == "__main__":
    app.run()
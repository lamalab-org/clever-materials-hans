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
        np,
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
def _(df, np):
    features = [f for f in df.columns if f.startswith("feat_")]
    df_with_pce = df.dropna(
        subset=["authors_full_list", "data.jv.default_PCE"] + features
    )

    # Calculate the 90th percentile (top 10%) threshold
    pce_threshold = np.percentile(df_with_pce["data.jv.default_PCE"], 90)
    print(f"Top 10% PCE threshold: {pce_threshold:.2f}%")

    # Create binary classification target
    df_with_pce["is_top10_pce"] = (df_with_pce["data.jv.default_PCE"] >= pce_threshold).astype(int)

    # Show class distribution
    class_counts = df_with_pce["is_top10_pce"].value_counts()
    print(f"Class distribution:")
    print(f"  Not top 10% (0): {class_counts[0]} samples ({class_counts[0]/len(df_with_pce)*100:.1f}%)")
    print(f"  Top 10% (1): {class_counts[1]} samples ({class_counts[1]/len(df_with_pce)*100:.1f}%)")

    df_final = df_with_pce
    print(f"Dataset shape: {df_final.shape}")
    print(f"Number of features: {len(features)}")

    target_column = "is_top10_pce"
    return df_final, pce_threshold, target_column


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
    perovskite_top10_labels = METRIC_LABELS.copy()

    fig_performance = plot_performance_comparison(
        results_default,
        target_type="classification",
        metric_labels=perovskite_top10_labels,
        save_path=paths.figures / "perovskite_top10_performance_comparison.pdf",
        title_suffix="Perovskite Top 10% PCE Dataset",
    )
    fig_performance.show()
    return fig_performance, perovskite_top10_labels


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
        dataset_name="Perovskite_Top10",
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
        save_path=paths.figures / "perovskite_top10_meta_performance.pdf",
        title_suffix="Perovskite Top 10% PCE Dataset",
    )
    fig_meta
    return


@app.cell
def _(
    paths,
    perovskite_top10_labels,
    plot_parameter_sweep_results,
    sweep_results,
):
    fig_sweep = plot_parameter_sweep_results(
        sweep_results,
        metric="accuracy",
        metric_labels=perovskite_top10_labels,
        save_path=paths.figures / "perovskite_top10_parameter_sweep.pdf",
        title_suffix="Perovskite Top 10% PCE Dataset",
    )
    fig_sweep
    return


@app.cell
def _(
    create_main_figure_panel,
    paths,
    perovskite_top10_labels,
    results_default,
):
    fig_main = create_main_figure_panel(
        results_default,
        target_type="classification",
        dataset_name="Perovskite Top 10% PCE",
        metric_labels=perovskite_top10_labels,
        save_path=paths.figures / "perovskite_top10_main_panel.pdf",
    )
    fig_main
    return


@app.cell
def _(export_key_metrics, paths, results_default):
    export_key_metrics(
        results_default,
        dataset_name="perovskite_top10",
        output_dir=paths.output,
        target_type="classification",
    )
    return


@app.cell
def _(df_final, pce_threshold):
    # Show some statistics about the top 10% threshold
    print(f"PCE threshold for top 10%: {pce_threshold:.2f}%")
    print(f"Range of top 10% PCE values: {df_final[df_final['is_top10_pce']==1]['data.jv.default_PCE'].min():.2f}% - {df_final[df_final['is_top10_pce']==1]['data.jv.default_PCE'].max():.2f}%")
    print(f"Range of bottom 90% PCE values: {df_final[df_final['is_top10_pce']==0]['data.jv.default_PCE'].min():.2f}% - {df_final[df_final['is_top10_pce']==0]['data.jv.default_PCE'].max():.2f}%")
    return


if __name__ == "__main__":
    app.run()

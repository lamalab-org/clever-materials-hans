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
    from utils import run_meta_comparison_analysis
    from plotting_utils import (
        plot_meta_comparison,
        create_main_figure_panel_with_meta_comparison,
        METRIC_LABELS,
    )
    import numpy as np

    lama_aesthetics.get_style("main")
    return (
        METRIC_LABELS,
        create_main_figure_panel_with_meta_comparison,
        np,
        paths,
        pd,
        plot_meta_comparison,
        run_meta_comparison_analysis,
    )


@app.cell
def _(paths, pd):
    df = pd.read_parquet(paths.output / "tadf_preprocess.parquet")
    return (df,)


@app.cell
def _(df, np):
    features = [f for f in df.columns if f.startswith("feat_")]
    df_final = df.dropna(subset=["authors_full_list", "raw_value"] + features)


    raw_value_float = []
    for i, row in df_final.iterrows():
        try:
            number = float(row["raw_value"])
        except Exception:
            number = np.nan
        raw_value_float.append(number)

    df_final["raw_value_float"] = raw_value_float

    print(f"Dataset shape: {df_final.shape}")
    print(f"Number of features: {len(features)}")

    target_column = "raw_value_float"
    return df_final, target_column


@app.cell
def _(df_final, run_meta_comparison_analysis, target_column):
    # Run comparison between predicted and actual meta-information
    comparison_results = run_meta_comparison_analysis(
        df_final,
        target_column,
        target_type="regression",
        n_authors=1000,
        use_year=True,
        use_journal=True,
        n_folds=5,  # Reduced for demo
    )
    return (comparison_results,)


@app.cell
def _(METRIC_LABELS, comparison_results, paths, plot_meta_comparison):
    # Create comparison plot
    fig_comparison = plot_meta_comparison(
        comparison_results,
        target_type="regression",
        metric_labels=METRIC_LABELS,
        save_path=paths.figures / "tadf_meta_comparison.pdf",
        dataset_name="TADF Emission Wavelength",
    )
    fig_comparison
    return


@app.cell
def _(
    METRIC_LABELS,
    comparison_results,
    create_main_figure_panel_with_meta_comparison,
    paths,
):
    # Create main figure panel with all four methods
    fig_main_comparison = create_main_figure_panel_with_meta_comparison(
        comparison_results,
        target_type="regression",
        dataset_name="TADF",
        metric_labels=METRIC_LABELS,
        save_path=paths.figures / "tadf_main_panel_meta_comparison.pdf",
        selection_metric="mae",
    )
    fig_main_comparison
    return


@app.cell
def _(comparison_results):
    # Compare performance between predicted and actual meta
    pred_indirect = comparison_results["predicted_meta"]["indirect"]["mae"]["mean"]
    actual_indirect = comparison_results["actual_meta"]["indirect"]["mae"]["mean"]
    direct_perf = comparison_results["predicted_meta"]["direct"]["mae"]["mean"]

    print(f"Direct (Conventional) MAE: {direct_perf:.2f}")
    print(f"Indirect (Predicted Meta) MAE: {pred_indirect:.2f}")
    print(f"Indirect (Actual Meta) MAE: {actual_indirect:.2f}")
    print(
        f"Performance gap (Predicted vs Actual Meta): {abs(pred_indirect - actual_indirect):.2f}"
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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

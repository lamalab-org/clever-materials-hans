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
        run_leave_one_author_group_out_cv,
        calculate_effect_sizes_cv_comparison,
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
        calculate_effect_sizes_cv_comparison,
        create_main_figure_panel,
        create_main_figure_panel_with_meta_comparison,
        export_key_metrics,
        np,
        paths,
        pd,
        plot_meta_performance,
        plot_parameter_sweep_results,
        plot_performance_comparison,
        run_leave_one_author_group_out_cv,
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
    pred_mae = meta_comparison["predicted_meta"]["indirect"]["mae"]["mean"]
    actual_mae = meta_comparison["actual_meta"]["indirect"]["mae"]["mean"]
    direct_mae = meta_comparison["predicted_meta"]["direct"]["mae"]["mean"]

    print(f"TADF Meta-information comparison:")
    print(f"Direct (Conventional): {direct_mae:.2f} MAE")
    print(f"Predicted meta: {pred_mae:.2f} MAE")
    print(f"Actual meta: {actual_mae:.2f} MAE")
    print(f"Performance gap: {abs(pred_mae - actual_mae):.2f} MAE")
    return (meta_comparison,)


@app.cell
def _(
    create_main_figure_panel_with_meta_comparison,
    meta_comparison,
    paths,
    tadf_labels,
):
    # Create four-column main panel with actual meta results
    fig_main_four_col = create_main_figure_panel_with_meta_comparison(
        meta_comparison,
        target_type="regression",
        dataset_name="TADF",
        metric_labels=tadf_labels,
        save_path=paths.figures / "tadf_main_panel_four_columns.pdf",
        selection_metric="mae",
    )
    fig_main_four_col
    return


@app.cell
def _(df_final, run_leave_one_author_group_out_cv, target_column):
    # Run Leave-One-Author-Group-Out CV to audit for Clever Hans effects
    # This ensures author groups are not shared between train/test
    logo_cv_results = run_leave_one_author_group_out_cv(
        df_final,
        target_column,
        target_type="regression",
        n_top_authors=10,  # Use top 10 authors for TADF dataset
        model_type="lgb",
        random_state=42,
        include_indirect=True,  # Include indirect (proxy) model
        n_authors=500,
        use_year=True,
        use_journal=True,
    )

    # Print summary of Leave-One-Author-Group-Out CV
    direct_mae_ = logo_cv_results["aggregated_results"]["direct_mae"]["mean"]
    dummy_mae_ = logo_cv_results["aggregated_results"]["dummy_mae"]["mean"]

    print(f"\n=== Leave-One-Author-Group-Out Cross-Validation Results ===")
    print(
        f"Direct Model MAE: {direct_mae_:.3f} ± {logo_cv_results['aggregated_results']['direct_mae']['std']:.3f}"
    )
    print(
        f"Dummy MAE: {dummy_mae_:.3f} ± {logo_cv_results['aggregated_results']['dummy_mae']['std']:.3f}"
    )
    print(f"Direct vs Dummy Gap: {dummy_mae_ - direct_mae_:.3f} MAE")

    if "indirect_mae" in logo_cv_results["aggregated_results"]:
        indirect_mae = logo_cv_results["aggregated_results"]["indirect_mae"][
            "mean"
        ]
        print(
            f"Indirect Model MAE: {indirect_mae:.3f} ± {logo_cv_results['aggregated_results']['indirect_mae']['std']:.3f}"
        )
        print(f"Indirect vs Dummy Gap: {dummy_mae_ - indirect_mae:.3f} MAE")

    print(f"Number of Author Groups: {logo_cv_results['n_folds']}")
    return (logo_cv_results,)


@app.cell
def _(calculate_effect_sizes_cv_comparison, logo_cv_results, results_default):
    # Calculate effect sizes comparing conventional CV vs Leave-One-Author-Group-Out CV
    effect_comparison = calculate_effect_sizes_cv_comparison(
        results_default, logo_cv_results, metric_name="mae"
    )

    print(
        f"\n=== Effect Size Analysis: Conventional CV vs Leave-One-Author-Group-Out CV ==="
    )

    # Print S ratios (key metric for author shortcuts)
    if "s_ratios" in effect_comparison and effect_comparison["s_ratios"]:
        s_ratios = effect_comparison["s_ratios"]
        print(f"\nS Ratios (proxy - dummy) / (direct - dummy):")
        if "conventional_S" in s_ratios:
            print(f"  Conventional CV S: {s_ratios['conventional_S']:.3f}")
        if "logo_S" in s_ratios:
            print(f"  Leave-One-Author-Out S: {s_ratios['logo_S']:.3f}")
        if "S_change" in s_ratios:
            print(
                f"  S Change: {s_ratios['S_change']:.3f} ({s_ratios['S_change_pct']:.1f}%)"
            )

        # Interpretation
        if "conventional_S" in s_ratios and "logo_S" in s_ratios:
            if s_ratios["conventional_S"] > 0.7 and s_ratios["logo_S"] < 0.3:
                print(f"  → STRONG author shortcut detected!")
            elif s_ratios["conventional_S"] > 0.5 and s_ratios["logo_S"] < 0.5:
                print(f"  → Moderate author shortcut detected")
            else:
                print(f"  → Weak or no author shortcut")

    # Print performance gaps
    if "performance_gaps" in effect_comparison:
        gaps = effect_comparison["performance_gaps"]
        print(f"\nPerformance Gaps:")
        print(
            f"  Conventional Direct vs Dummy: {gaps['conventional_direct_vs_dummy']:.3f}"
        )
        print(f"  LOAO Direct vs Dummy: {gaps['logo_direct_vs_dummy']:.3f}")
        print(
            f"  Gap Reduction: {gaps['gap_reduction']:.3f} ({gaps['gap_reduction_pct']:.1f}%)"
        )
    return


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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import paths
    from meta_enchrichment import MetadataEnricher
    return MetadataEnricher, mo, paths, pd


@app.cell
def _(paths, pd):
    df = pd.read_json(paths.data / "thermal_stability.json")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we would like to enrich the dataset with some bibliometric information from crossref.
    """)
    return


@app.cell
def _(MetadataEnricher):
    meta_enricher = MetadataEnricher()
    return (meta_enricher,)


@app.cell
def _(df, meta_enricher):
    df_enriched = meta_enricher.enrich_dataset(df, doi_column="doi")
    return (df_enriched,)


@app.cell
def _(df_enriched):
    df_enriched["authors_full_list"]
    return


@app.cell
def _(df_enriched, pd):
    expanded_rows = []

    for i, row in df_enriched.iterrows():
        for f, v in row["features"].items():
            row[f"feat_{f}"] = v
        expanded_rows.append(row)

    expanded_df = pd.DataFrame(expanded_rows)
    return (expanded_df,)


@app.cell
def _(expanded_df, paths):
    expanded_df.to_parquet(paths.output / "mof_thermal_stability.parquet")
    return


@app.cell
def _(paths, pd):
    df_solvent = pd.read_json(paths.data / "solvent_removal_stability.json")
    return (df_solvent,)


@app.cell
def _(df_solvent, meta_enricher):
    df_solvent_enriched = meta_enricher.enrich_dataset(
        df_solvent, doi_column="doi"
    )
    return (df_solvent_enriched,)


@app.cell
def _(df_solvent_enriched, paths, pd):
    expanded_rows_solvent = []

    for _i, solvent_row in df_solvent_enriched.iterrows():
        for f_solvent, v_solvent in solvent_row["features"].items():
            solvent_row[f"feat_{f_solvent}"] = v_solvent
        expanded_rows_solvent.append(solvent_row)

    expanded_df_solvent = pd.DataFrame(expanded_rows_solvent)
    expanded_df_solvent.to_parquet(paths.output / "mof_solvent_stability.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

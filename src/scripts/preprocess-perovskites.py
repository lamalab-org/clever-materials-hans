import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import paths
    import marimo as mo
    from descriptor_generation import DescriptorGenerator
    return DescriptorGenerator, mo, paths, pd


@app.cell
def _(paths, pd):
    df = pd.read_parquet(paths.data / "perovskite_solar_cell_database.parquet")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To prepare data for some analysis, we will compute composition features and compile the meta info into dedicated fields we can use for machine learning.
    """)
    return


@app.cell
def _(DescriptorGenerator, df):
    generator = DescriptorGenerator()
    df_enriched = generator.generate_composition_descriptors(
        df, formula_column="results.material.chemical_formula_reduced"
    )
    return (df_enriched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The meta info we should standardize at least a bit:
    - Authors into `authors_full_list`
    - `publication_year` column
    - `journal_name` column
    """)
    return


@app.cell
def _(df_enriched):
    df_enriched
    return


@app.function
def create_authors_full_lists(df):
    authors = []
    for _, row in df.iterrows():
        author = ""
        for i in range(36):
            col_name_first = f"data.ref.authors.{i}.first_name"
            col_name_last = f"data.ref.authors.{i}.last_name"
            first = row[col_name_first]
            last = row[col_name_last]
            if not last:
                break
            author += f"{first} {last}, "
        if len(author) <= 1:
            author = None
        if isinstance(author, str):
            author = author[:-2]
        authors.append(author)

    df["authors_full_list"] = authors


@app.cell
def _(df_enriched):
    create_authors_full_lists(df_enriched)
    return


@app.cell
def _(df_enriched):
    df_enriched["journal_name"] = df_enriched["data.ref.journal"]
    return


@app.cell
def _(df_enriched):
    df_enriched["publication_year"] = [
        int(df_enriched["data.ref.publication_date"].iloc[i].split("-")[0])
        if df_enriched["data.ref.publication_date"].iloc[i]
        else None
        for i in range(len(df_enriched))
    ]
    return


@app.cell
def _(df_enriched, paths):
    df_enriched.to_parquet(paths.output / "perovskite_processed.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

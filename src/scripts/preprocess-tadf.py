import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import paths
    from meta_enchrichment import MetadataEnricher
    from descriptor_generation import DescriptorGenerator
    return DescriptorGenerator, MetadataEnricher, mo, paths, pd


@app.cell
def _(paths, pd):
    df = pd.read_csv(
        paths.data / "tadf.csv"
    )  # https://figshare.com/articles/dataset/A_Database_of_Thermally_Activated_Delayed_Fluorescent_Molecules_Auto-generated_from_Scientific_Literature_with_ChemDataExtractor/24004182?file=43183206 (subsidary.csv)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df_emission = df[df["model_name"] == "EmissionWavelength"]
    return (df_emission,)


@app.cell
def _(df_emission):
    df_emission["raw_units"].unique()
    return


@app.cell
def _(DescriptorGenerator):
    descriptors = DescriptorGenerator()
    return (descriptors,)


@app.cell
def _(descriptors, df_emission):
    df_emission_enriched = descriptors.generate_molecular_descriptors(
        df_emission, smiles_column="compound.SMILES"
    )
    return (df_emission_enriched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, let's also add some meta information about the bibliometrics.
    """)
    return


@app.cell
def _(df_emission_enriched):
    df_emission_enriched
    return


@app.cell
def _(MetadataEnricher):
    meta_enricher = MetadataEnricher()
    return (meta_enricher,)


@app.cell
def _(df_emission_enriched):
    df_emission_enriched["doi"] = df_emission_enriched["doi"].apply(
        lambda x: x.replace("https://", "")
    )
    return


@app.cell
def _(df_emission_enriched, meta_enricher):
    df_emission_enriched_doi = meta_enricher.enrich_dataset(
        df_emission_enriched, doi_column="doi"
    )
    return (df_emission_enriched_doi,)


@app.cell
def _(df_emission_enriched_doi, paths):
    df_emission_enriched_doi.to_parquet(paths.output / "tadf_preprocess.parquet")
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

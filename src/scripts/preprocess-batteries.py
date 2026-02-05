import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import paths
    from collections import defaultdict
    from typing import List
    import ast
    from descriptor_generation import DescriptorGenerator
    from meta_enchrichment import MetadataEnricher
    return DescriptorGenerator, MetadataEnricher, ast, defaultdict, paths, pd


@app.cell
def _(paths, pd):
    df = pd.read_csv(paths.data / "battery_merged.csv")
    return (df,)


@app.cell
def _(df):
    df_capacity = df[df["Property"] == "Capacity"]
    return (df_capacity,)


@app.cell
def _(df_capacity):
    df_capacity_clean = df_capacity[df_capacity["Warning"].isna()]
    return (df_capacity_clean,)


@app.cell
def _(df_capacity_clean):
    df_capacity_clean
    return


@app.cell
def _(ast, defaultdict):
    def clean_composition(extracted_name: str) -> str:
        try:
            extracted_name = ast.literal_eval(extracted_name)
            composition = defaultdict(int)
            for d in extracted_name:
                for k, v in d.items():
                    composition[k] += int(float(v))

            formula = ""
            for k, v in composition.items():
                formula += f"{k}{v}"

            return formula
        except Exception:
            return None
    return (clean_composition,)


@app.cell
def _(clean_composition, df_capacity_clean):
    df_capacity_clean["composition"] = df_capacity_clean["Extracted_name"].apply(
        clean_composition
    )
    return


@app.cell
def _(DescriptorGenerator):
    generator = DescriptorGenerator()
    return (generator,)


@app.cell
def _(df_capacity_clean, generator):
    enriched_df = generator.generate_composition_descriptors(
        df_capacity_clean, formula_column="composition"
    )
    return (enriched_df,)


@app.cell
def _(MetadataEnricher):
    meta_enricher = MetadataEnricher()
    return (meta_enricher,)


@app.cell
def _(enriched_df, meta_enricher):
    df_enriched_enriched = meta_enricher.enrich_dataset(
        enriched_df, doi_column="DOI"
    )
    return (df_enriched_enriched,)


@app.cell
def _(df_enriched_enriched):
    df_enriched_enriched["authors_full_list"]
    return


@app.cell
def _(df_enriched_enriched, paths):
    df_enriched_enriched.to_parquet(paths.output / "battery_preprocessed.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

rule preprocess_tadf:
    input:
        "src/data/raw/tadf.csv",
    output:
        "src/tex/output/tadf_processed.parquet",
    script:
        "src/scripts/preprocess-tadf.py"

rule preprocess_perovskites:
    input:
        "src/data/raw/perovskite_solar_cell_database.parquet",
    output:
        "src/tex/output/perovskite_processed.parquet",
    script:
        "src/scripts/preprocess-perovskites.py"
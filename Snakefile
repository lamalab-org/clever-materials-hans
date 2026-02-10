rule preprocess_tadf:
    input:
        "src/data/raw/tadf.csv",
    output:
        "src/tex/output/tadf_processed.parquet",
    script:
        "src/scripts/preprocess-tadf.py"
rule preprocess_tadf:
    input:
        "src/data/tadf.csv",
        "src/scripts/utils.py",
        "src/scripts/descriptor_generation.py",
        "src/scripts/meta_enchrichment.py",
        "src/scripts/paths.py",
    output:
        "src/tex/output/tadf_processed.parquet",
    cache: True
    script:
        "src/scripts/preprocess-tadf.py"

rule preprocess_perovskites:
    input:
        "src/data/perovskite_solar_cell_database.parquet",
        "src/scripts/utils.py",
        "src/scripts/descriptor_generation.py",
        "src/scripts/meta_enchrichment.py",
        "src/scripts/paths.py",
    output:
        "src/tex/output/perovskite_processed.parquet",
    cache: True             
    script:
        "src/scripts/preprocess-perovskites.py"

rule preprocess_batteries:
    input:
        "src/data/battery_merged.csv",
        "src/scripts/utils.py",
        "src/scripts/descriptor_generation.py",
        "src/scripts/meta_enchrichment.py",
        "src/scripts/paths.py",
    output:
        "src/tex/output/battery_preprocessed.parquet",
    cache: True
    script:
        "src/scripts/preprocess-batteries.py"

rule preprocess_mof:
    input:
        "src/data/thermal_stability.json",
        "src/data/solvent_removal_stability.json",
        "src/scripts/utils.py",
        "src/scripts/descriptor_generation.py",
        "src/scripts/meta_enchrichment.py",
        "src/scripts/paths.py",
    output:
        "src/tex/output/mof_thermal_stability.parquet",
        "src/tex/output/mof_solvent_stability.parquet",
    script:
        "src/scripts/preprocess-mof.py"

rule analyze_mof_thermal_top10:
    input:
        "src/tex/output/mof_thermal_stability.parquet",
        "src/scripts/utils.py",
        "src/scripts/plotting_utils.py",
        "src/scripts/paths.py",
    output:
        "src/tex/figures/mof_thermal_top10_main_panel_four_columns.pdf",
        "src/tex/output/mof_thermal_top10_indirect_accuracy.txt",
        "src/tex/output/mof_thermal_top10_direct_accuracy.txt",
    script:
        "src/scripts/analyze-mof-thermal-top10.py"

rule analyze_mof_solvent:
    input:
        "src/tex/output/mof_solvent_stability.parquet",
        "src/scripts/utils.py",
        "src/scripts/plotting_utils.py",
        "src/scripts/paths.py",
    output:
        "src/tex/figures/mof_solvent_main_panel_four_columns.pdf",
        "src/tex/output/mof_solvent_indirect_accuracy.txt",
    script:
        "src/scripts/analyze-mof-solvent-stability.py"

rule analyze_perovskites_top10:
    input:
        "src/tex/output/perovskite_processed.parquet",
        "src/scripts/utils.py",
        "src/scripts/plotting_utils.py",
        "src/scripts/paths.py",
    output:
        "src/tex/figures/perovskite_top10_main_panel_four_columns.pdf",
        "src/tex/output/perovskite_top10_author_f1_micro.txt",
        "src/tex/output/perovskite_top10_indirect_accuracy.txt",
        "src/tex/output/perovskite_top10_direct_accuracy.txt",
    script:
        "src/scripts/analyze-perovskites-top10.py"

rule analyze_tadf:
    input:
        "src/tex/output/tadf_processed.parquet",
    output:
        "src/tex/figures/tadf_main_panel_meta_comparison.pdf",
        "src/tex/output/tadf_best_meta_accuracy.txt",
        "src/tex/output/tadf_best_dataset_size.txt",
        "src/tex/output/tadf_n_features.txt",
    script:
        "src/scripts/analyze-tadf.py"

rule analyze_batteries_top10:
    input:
        "src/tex/output/battery_preprocessed.parquet",
    output:
        "src/tex/figures/battery_main_panel_four_columns.pdf",
        "src/tex/output/battery_top10_author_f1_micro.txt",
        "src/tex/output/battery_dataset_size.txt",
        "src/tex/output/battery_n_features.txt",
    script:
        "src/scripts/analyze-batteries-top10.py"

rule analyze_perovskites:
    input:
        "src/tex/output/perovskite_processed.parquet",
    output:
        "src/tex/output/perovskite_dataset_size.txt",
        "src/tex/output/perovskite_n_features.txt",
    script:
        "src/scripts/analyze-perovskites.py"

rule analyze_mof_thermal:
    input:
        "src/tex/output/mof_thermal_stability.parquet",
    output:
        "src/tex/output/mof_thermal_dataset_size.txt",
        "src/tex/output/mof_thermal_n_features.txt",
    script:
        "src/scripts/analyze-mof-thermal-stability.py"

rule analyze_mof_solvent_dataset:
    input:
        "src/tex/output/mof_solvent_stability.parquet",
    output:
        "src/tex/output/mof_solvent_dataset_size.txt",
        "src/tex/output/mof_solvent_n_features.txt",
    script:
        "src/scripts/analyze-mof-solvent-stability.py"
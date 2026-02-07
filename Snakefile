# Custom Snakefile for Clever Materials Hans Analysis
# This supplements showyourwork's auto-generated rules

# Utility rules for development
rule clean_all:
    shell:
        """
        rm -rf src/tex/figures/*.pdf
        rm -rf src/tex/output/*.txt
        rm -rf src/tex/output/*.parquet
        """

rule clean_figures:
    shell: "rm -rf src/tex/figures/*.pdf"

rule clean_data:
    shell: "rm -rf src/tex/output/*.parquet"

rule clean_metrics:
    shell: "rm -rf src/tex/output/*.txt"

# Development rules for individual datasets
rule batteries_only:
    input:
        "src/tex/figures/battery_main_panel.pdf",
        "src/tex/figures/battery_top10_main_panel.pdf"

rule perovskites_only:
    input:
        "src/tex/figures/perovskite_top10_main_panel.pdf",
        "src/tex/figures/perovskite_jsc_main_panel.pdf", 
        "src/tex/figures/perovskite_voc_main_panel.pdf"

rule mof_only:
    input:
        "src/tex/figures/mof_thermal_main_panel.pdf",
        "src/tex/figures/mof_thermal_top10_main_panel.pdf",
        "src/tex/figures/mof_solvent_main_panel.pdf"

rule tadf_only:
    input: "src/tex/figures/tadf_main_panel.pdf"
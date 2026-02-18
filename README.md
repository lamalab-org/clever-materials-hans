# Clever Materials

Open-source scientific article built with the [showyourwork](https://github.com/showyourwork/showyourwork) workflow.

## Gist

Machine learning can accelerate materials discovery, but strong benchmark scores do not guarantee that models learn chemistry. This article tests an alternative hypothesis: property prediction may be driven by bibliographic confounding. Across five tasks (MOF thermal and solvent stability, perovskite efficiency, battery capacity, and TADF emission), models can predict author, journal, and year from standard descriptors above chance. When those predicted metadata (bibliographic fingerprints) are used as the only inputs, performance sometimes approaches conventional descriptor-based models. The results show that many datasets do not rule out non-chemical explanations of success and motivate routine falsification tests (group/time splits, metadata ablations), better dataset design, and clearer separation between predictive utility and evidence of chemical understanding.

## Build

```bash
showyourwork build
```

# Placental preeclampsia signature: code and derived results

Secondary re-analysis of two public GEO placental datasets:
- Discovery: GSE303463 (RNA-seq; DESeq2, design ~ batch + sex + gestational age + group)
- Validation: GSE75010 (Affymetrix array; limma adjusted for gestational age and sex)

## Contents
- validation_confounder_analysis.py: confounder-adjusted validation (AUC, DeLong test, Freedman-Lane permutation, fully adjusted per-gene model, non-linear gestational-age sensitivity analysis).
- signature_scores_by_sample.csv: signature score for each sample.
- fully_adjusted_model_364genes.csv: fully adjusted per-gene model results.
- Supplementary_data_300_replicated_genes.csv: the 300 replicated genes.
- Supplementary_data_PEonly_signature_191_genes.csv: signature derived from preeclampsia-only comparison.

## Data
Raw data are not redistributed. Download them from GEO using the accession numbers above.

## Author
Ahmed Danbous Obayes, Al-Qasim Green University, Iraq (ORCID 0000-0002-3546-4150)

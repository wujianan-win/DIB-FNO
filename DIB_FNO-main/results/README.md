# Reported result tables

These CSV files are the compact, publication-facing outputs retained after repository cleanup. They are not regenerated during installation.

- `Table01_prespecified_RMSE.csv`: absolute test RMSE by lead, channel, and model.
- `Table02_prespecified_paired_differences.csv`: paired DIB-FNO differences and pointwise intervals.
- `Table03_individual_seed_RMSE.csv`: per-seed absolute RMSE.
- `Table04_gate_seed_statistics.csv`: train-only soft-gate summaries.
- `Table05_training_config.csv`: model sizes, seeds, epochs, and checkpoint hashes.
- `Table06_gate_pairwise_similarity.csv`: cross-seed gate agreement.
- `Table07_block_length_sensitivity.csv`: 10-, 15-, and 30-day block sensitivity.
- `Supplement_Table_*.csv`: bias, case, input-gate, top-C, and training-history diagnostics.

The primary interval specification is 2,000 paired circular 15-day block replicates with joint date/seed resampling. Intervals are pointwise percentile 95% intervals without multiplicity correction.

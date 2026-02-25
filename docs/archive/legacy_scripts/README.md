# Legacy Scripts

Scripts moved here during the production cleanup. Each was either superseded by
the unified `taxembed` CLI or depended on the archived Facebook `hype/` library.

| Script | Reason for archiving |
|--------|---------------------|
| `train.py` | Imports from `hype/`, replaced by `train_small.py` |
| `prepare_data.py` | Duplicates root-level `prepare_taxonomy_data.py` |
| `remap_data.py` | Deprecated by `remap_edges.py` |
| `visualize.py` | Uses old `hype` framework |
| `visualize_embeddings.py` | Uses old `hype` framework |
| `monitor.py` | Unused utility |
| `evaluate.py` | Unused utility |
| `check_model.py` | Standalone debug utility |
| `check_dataset_composition.py` | Standalone debug utility |
| `analyze_hierarchy.py` | Superseded by `analyze_hierarchy_hyperbolic.py` |
| `sanity_check.py` | Superseded by `final_sanity_check.py` (no `main()`, inline code) |
| `remap_edges.py` | Orphaned utility, not referenced by CLI or tests |
| `cleanup_repo.sh` | References deleted files (`embed.py`, old scripts) |
| `regenerate_data.sh` | References deleted `remap_edges.py` and old workflow |

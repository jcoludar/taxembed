# TaxEmbed — one hyperbolic embedding for all named cellular life

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Embedding on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-taxembed--cellular-yellow)](https://huggingface.co/jcoludar/taxembed-cellular)

TaxEmbed embeds the NCBI Taxonomy of cellular life, all **1,102,163 taxon identifiers** under
"cellular organisms" (`new_taxdump` downloaded 2026-06-09), into a single **100-dimensional Poincaré
ball**. The released tensor is a lookup: 100 numbers per taxon, fixed-size and differentiable.

- **Embedding (441 MB, Apache-2.0):** https://huggingface.co/jcoludar/taxembed-cellular, with the
  taxid-to-row mapping, the parent-child edgelist, the training closure and a model card.
- **Paper:** Koludarov I, Rost B. *TaxEmbed: one hyperbolic embedding for all named cellular life*
  (2026, preprint to follow).
- **This repository:** the trainer that produced the release, the evaluation code, and the audit
  trail behind every number in the paper.

In the geometry, **radius is taxonomic depth** (set by a schedule at initialization and held there by
a radial nudge; the norm-depth correlation is +0.957 before the first gradient step and +0.957 after
training) and **direction is learned lineage** (same-depth taxa are ordered by lineage at 0.965 on a
radius-free criterion whose random expectation is 0). Half of a taxon's ten nearest embedded
neighbours are among its ten nearest by tree path (precision@10 0.546, chance 0.005). Against
TimeTree divergence times the angular distance correlates at +0.89 in vertebrates (NCBI path length
+0.79) and +0.30 in insects (path length +0.44): embedded distance is a taxonomic distance and should
not be read as a molecular clock.

The implementation began as a fork of the Poincaré-embeddings reference code (Nickel & Kiela, 2017;
open-sourced by Facebook Research in 2018) and has been entirely rewritten; no source code from the
original remains (see [NOTICE](NOTICE)).

---

## Use the released embedding

```python
import numpy as np, pandas as pd
from safetensors.numpy import load_file

emb = load_file("cellular_embedding.safetensors")["embedding"]           # (1102163, 100) float32
idx = pd.read_csv("taxid_to_index.tsv", sep="\t").set_index("taxid")["idx"]

def poincare_distance(u, v):
    sq = np.sum((u - v) ** 2)
    return np.arccosh(1 + 2 * sq / ((1 - u @ u) * (1 - v @ v)))

def angular_distance(u, v):
    """Lineage similarity lives in the direction; the radius is planted depth."""
    return np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))

h, m = emb[idx[9606]], emb[idx[10090]]                                     # Homo sapiens, Mus musculus
print(poincare_distance(h, m), angular_distance(h, m), np.linalg.norm(h))
```

The lineage structure that training determines is in the **direction** of each vector; the norm is
set by depth through a fixed schedule. Compare taxa by the angle between their directions, or by
Poincaré distance among taxa at the same depth. `edges_parent_child.tsv` (row indices) gives depths
and tree-path distances without the taxdump; `training_closure.npz` holds all 21,399,053
ancestor-descendant pairs the model was trained on. Names come from `names.dmp` of the same
`new_taxdump` release; 7.0 % of rows are superseded taxids kept beside their replacement, and 8.3 %
carry depositor placeholder binomials (details on the model card).

## Install

```bash
git clone https://github.com/jcoludar/taxembed.git
cd taxembed
uv sync                      # Python 3.11+, PyTorch; or: make install
uv run taxembed check        # imports, columnar I/O, model forward+backward
```

The `taxembed` CLI: `download` (NCBI taxdump into `data/`), `build <clade> [--clean]` (a clade's
transitive-closure dataset), `train <clade> -as <tag>`, `visualize <tag>`, `dim <clade>`, `check`.
Reference: [docs/CLI_COMMANDS.md](docs/CLI_COMMANDS.md), [docs/QUICKSTART.md](docs/QUICKSTART.md).

---

## Reproduce the training run

The released tensor was trained with one recipe: a Euclidean tangent parametrization of the ball, a
softmax objective over 300 sampled negatives, a four-stage depth curriculum, a radial nudge toward a
log depth schedule, and an effective batch of 2,048 (256 × 8) at 10^5 taxa and above. The Slurm
script that ran it is [`scripts/train_lrz_cellular_canonical.sh`](scripts/train_lrz_cellular_canonical.sh);
the same flags through the CLI:

```bash
uv run taxembed build 131567 --clean            # 2.6M raw nodes -> 1,102,163 clean, 21.4M pairs
uv run taxembed train \
    --file    data/taxopy/cellular_organisms_131567_clean/taxonomy_edges_cellular_organisms_131567_clean_transitive.npz \
    --mapping data/taxopy/cellular_organisms_131567_clean/taxonomy_edges_cellular_organisms_131567_clean.mapping.tsv \
    -as cellular_canonical --dim 100 --epochs 200 --seed 0 \
    --batch-size 256 --grad-accum-steps 8 --n-negatives 300 --lr 0.001 \
    --lr-schedule cosine_warmrestart --warm-restart-on-phase --lr-min-multiplier 0.01 \
    --curriculum --curriculum-phases auto --epoch-fraction 0.3 \
    --radial-nudge 0.05 --radial-schedule log --depth-scale-margin --margin-min 0.05 --margin-max 1.0 \
    --euclidean-param --loss softmax --early-stopping 999 --amp --gpu 0 --save-every 10
```

Wall-clock on one V100 (Slurm accounting): 25.5 h at 1,102,163 taxa, 20.5 h at 877,584 (Eukaryota),
10.5 to 11 h at 498,246 (Metazoa). Use the final checkpoint: angular structure keeps improving after
the loss plateaus. A reduced-budget configuration of the same objective (batch 512, 100 negatives,
lr 5×10^-3, constant schedule) collapsed at 498,246 taxa (lineage ordering 0.619 vs 0.973, three seeds
each); the job scripts for that contrast are `scripts/task9_lrz_recipe_contrast*.sh`.

Two things to know before training. `--seed` was added after the release, so the released tensor is
not seed-reproducible (a repeated CPU run of the current code reproduces its loss exactly). The
negative sampler draws from the descendant's depth without excluding the anchor's own descendants:
47.4 % of draws are false negatives on the cellular closure (`src/taxembed/eval/sampler_audit.py`);
the guarded sampler (`--exclude-descendant-negatives`, plan v2 Task 6) did not converge in three
seeded runs, so its effect on a converged model is open.

---

## Repository layout

```
taxembed/
├── train_hierarchical.py        # model, Poincaré distance, losses, negative sampling (trainer core)
├── train_small.py               # training orchestrator called by the CLI
├── build_transitive_closure.py  # closure builder
├── src/taxembed/
│   ├── cli/                     # the `taxembed` command
│   ├── builders/                # taxopy clade dataset builder (the --clean filter lives here)
│   ├── eval/                    # angular.py (S_angle), radial.py (init floor), sampler_audit.py,
│   │                            #   treedist.py, fidelity.py, bootstrap.py, linkpred.py, nulls.py, ...
│   ├── optim/                   # Riemannian Adam (not used by the release)
│   └── bridge/                  # ProtT5 -> taxonomy bridge (separate follow-up; not in the paper)
├── scripts/                     # Slurm job scripts (LRZ), scorers, figure-data extractors
├── helpers/                     # dated one-off analyses; each names the JSON it wrote
├── results/                     # every number in the paper has a JSON here
├── docs/                        # specs, plans, findings, the corrections ledger
├── tests/                       # pytest suite
├── data/, artifacts/            # gitignored: taxdumps, closures, checkpoints
└── release/                     # gitignored bundle mirrored on Hugging Face
```

---

## Evaluate

| What | Where |
|---|---|
| Lineage ordering *S*_angle (radius-free; random directions score 0) and its initialization null | `src/taxembed/eval/angular.py`; driver `scripts/score_recipe_checkpoints.py` |
| Norm-depth correlation and its initialization floor | `src/taxembed/eval/radial.py`, `scripts/radial_init_floor.py` |
| Tree-neighbour precision@k with taxon-level block bootstrap | `scripts/cophenetic_fidelity.py`; `src/taxembed/eval/{treedist,fidelity,bootstrap}.py` |
| Divergence-time comparison against TimeTree (pre-registered, frozen SHA256) | `helpers/_timetree_*.py`, `results/timetree_*.json` |
| Negative-sampler false-negative audit (closed form over the closure) | `src/taxembed/eval/sampler_audit.py` |
| Per-rank kNN purity and separation (descriptive; they inherit the radial schedule) | `scripts/analyze_hierarchy_hyperbolic.py`, `scripts/knn_purity_hyperbolic.py` |
| Superseded-taxid and placeholder census of the closure; per-domain counts; query cost | `helpers/_closure_taxid_drift_20260930.py`, `helpers/_owed_numbers_20260930.py` |

## Audit trail

- [`docs/MANUSCRIPT_CORRECTIONS_PENDING.md`](docs/MANUSCRIPT_CORRECTIONS_PENDING.md): the ledger of
  every correction the objective-integrity review forced (C1 to C14), each with its evidence file.
- [`docs/FINDING_protocols_that_are_vacuous_on_taxonomy_trees.md`](docs/FINDING_protocols_that_are_vacuous_on_taxonomy_trees.md):
  why held-out link prediction and RandomDAG controls cannot test generalisation for this model class.
- `docs/specs/`, `docs/plans/`: the pre-registrations and their amendments; `results/*_preregistration.json`
  carry the frozen hashes.

## Tests

```bash
uv run pytest            # 396 passed (2026-09-30)
```

The suite under `tests/` covers the trainer, the samplers and the evaluation modules. The bridge
suite under `src/taxembed/bridge/` is not collected by default.

---

## Contributing

Issues and pull requests: https://github.com/jcoludar/taxembed/issues. Open directions: a
Euclidean-distance arm in the trainer for a matched-geometry control at full scale (the small-scale
sweep is in the paper's Supplementary Table S3), low-dimensional releases (d = 8 to 16), and
inductive placement of taxa the taxonomy has not yet named.

## References

- Nickel M, Kiela D (2017). Poincaré embeddings for learning hierarchical representations. NeurIPS. https://arxiv.org/abs/1705.08039
- Nickel M, Kiela D (2018). Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. ICML. https://arxiv.org/abs/1806.03417
- Reference implementation (Facebook Research, 2018): https://github.com/facebookresearch/poincare-embeddings
- NCBI Taxonomy: https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/ and https://www.ncbi.nlm.nih.gov/taxonomy
- Kumar S et al. (2022). TimeTree 5. Mol Biol Evol 39:msac174.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). This project was originally forked from
`facebookresearch/poincare-embeddings` (CC BY-NC 4.0); it has since been fully reimplemented and contains none of
the original source, so the current code is released under Apache-2.0. The original method is credited to
Nickel & Kiela (2017).

## Citation

```
Koludarov I, Rost B. TaxEmbed: one hyperbolic embedding for all named cellular life. 2026. Preprint.
Code: https://github.com/jcoludar/taxembed
Embedding: https://huggingface.co/jcoludar/taxembed-cellular
```

Development history: [docs/JOURNEY.md](docs/JOURNEY.md).

*Last updated: 2026-09-30*

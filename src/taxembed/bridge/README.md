# Taxonomy Bridge (`taxembed.bridge`)

The bridge links **protein-language-model embeddings** (ProtT5) to a trained **Poincaré taxonomy
embedding**, in two complementary directions:

- **READ** — a linear ridge regressor maps each protein's ProtT5 representation into the *hyperbolic
  tangent space* of the taxonomy embedding. The predicted tangent vector is placed into the Poincaré
  ball (`exp0`), and the nearest taxonomy node is retrieved; its lineage is the per-rank taxonomic
  prediction. In short: *can protein embeddings be projected onto the tree of life?*
- **CLEAN** — LEACE (Least-squares Concept Erasure, from
  [`concept-erasure`](https://pypi.org/project/concept-erasure/)) removes the linear subspace that
  predicts taxonomy from the protein representations, and we measure how much *function* survives the
  erasure. A principal-angle overlap then measures whether the subspace READ predicts from is the same
  object CLEAN erases. In short: *can we erase the taxonomy signal without destroying function?*

## Install

```bash
pip install -e .            # installs taxembed + the bridge deps (incl. concept-erasure)
```

Optional external binary for identity-cluster leakage control (used by the CV splitters):

```bash
# macOS: brew install mmseqs2   |   conda: conda install -c bioconda mmseqs2
```

If `mmseqs` is not on `PATH`, the clustering step falls back to one-cluster-per-sequence (no crash).

## Data it needs, and where to get it

The bridge does **not** bundle the large model artifacts — only small git-tracked panels/resolutions
(`data/panels/`) and example result JSONs (`data/results/`) ship with the package. You supply:

### 1. The taxonomy embedding (target space)

- **Coordinates + `taxid_to_index.tsv`** — download from the Hugging Face release
  [`jcoludar/taxembed-cellular`](https://huggingface.co/jcoludar/taxembed-cellular). The coordinates
  are a **`.safetensors`** file whose tensor key is **`embedding`** (the loader also accepts a
  training `.pth` with key `embeddings`). The `taxid_to_index.tsv` maps each NCBI taxid to its row in
  the coordinate matrix.
- **Parent/depth `edgelist`** — **required companion file that the Hugging Face repo does *not*
  include.** Get it from the **Zenodo data deposit** for this project. The edgelist encodes the
  taxonomy tree (parent→child edges over the same integer indices as `taxid_to_index.tsv`) and is what
  `TaxonomyEmbedding` uses to compute node depths and exact tree distances. Without it, READ can place
  but cannot score against the true tree.

### 2. NCBI taxdump

A metazoa-subset `PROVENANCE.md` (+ SHA256s) ships under `data/panels/taxdump/`; the `.dmp` files
themselves are gitignored. Re-download `new_taxdump.tar.gz` from NCBI
(<https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/>) and extract it into that directory (or point
`TAXEMBED_TAXDUMP_DIR` elsewhere). The all-life leg needs the **full** new_taxdump — point
`TAXEMBED_FULL_TAXDUMP_DIR` at it.

### 3. Protein embeddings (per testbed)

- **PLA2 / ToxFam testbeds** — user-provided ProtT5 H5s + annotation tables (see the `PLA2_*` /
  `TOXFAM_*` entries in `config.py`); point them via the matching `TAXEMBED_*` variables.
- **SP-Metazoa** — the reviewed-Swiss-Prot `per-protein.h5` (~1.4 GB) downloads directly from UniProt
  (`config.SP_EMB_URL`); it is gitignored, downloaded once.
- **All-life** — the protein H5 is produced by **self-embedding** ProtT5 over the acquired sequences
  on a GPU (see below); it is not shipped.

## Pointing the code at your data

Every path is environment-parameterised (see `config.py`). The simplest setup puts everything under
one root:

```bash
export TAXEMBED_DATA_ROOT=/path/to/taxembed_data
# then drop the downloaded artifacts in, or override individual paths:
export TAXEMBED_CELLULAR_CKPT=$TAXEMBED_DATA_ROOT/cellular_canonical.safetensors
export TAXEMBED_CELLULAR_TAXMAP=$TAXEMBED_DATA_ROOT/cellular_taxid_to_index.tsv
export TAXEMBED_CELLULAR_EDGELIST=$TAXEMBED_DATA_ROOT/cellular.edgelist   # from Zenodo
export TAXEMBED_TAXDUMP_DIR=$TAXEMBED_DATA_ROOT/panels/taxdump
```

Unset variables fall back to sensible defaults under `TAXEMBED_DATA_ROOT` (default: `./data` next to
`config.py`).

## Running the batteries

```bash
# READ — place proteins into the taxonomy embedding and score per-rank retrieval
python -m taxembed.bridge.read_eval

# CLEAN — LEACE disentanglement matrix + READ/CLEAN coherence
python -m taxembed.bridge.clean_eval [--testbed pla2|pla2_mammals|multifamily]
```

Both write result JSONs (and CLEAN's figures via `python -m taxembed.bridge.make_sp_figures`).

## Reproducibility of the all-life leg

The all-life leg is **reproducible with compute**: sequences are acquired from the UniProt REST
`/stream` endpoint (sharded by length band under the 10M-result cap), and their per-protein embeddings
are produced by **self-embedding ProtT5 on a GPU** — there is no pre-baked all-life H5 to download. The
smaller SP-Metazoa leg is cheaper: its `per-protein.h5` (~1.4 GB) downloads directly from UniProt. The
Step-0 taxonomy gate and sizing gate (`step0_taxonomy_gate.py`, `sizing_gate.py`) are fail-loud
pre-flights that must pass before the GPU embedding job runs.

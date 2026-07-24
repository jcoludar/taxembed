"""Canonical paths + constants for the Taxonomy Bridge.

Every filesystem path is env-parameterised so the package carries no machine- or repo-specific
absolute paths. Set ``TAXEMBED_DATA_ROOT`` to point at wherever you keep the (largely user-provided)
data artifacts — checkpoints, taxdump, protein H5s. Individual artifacts can be repointed with their
own ``TAXEMBED_*`` variable; unset variables fall back to a sensible default under the data root.

Default data root is ``./data`` next to this file, which is also where the small git-tracked panels
(``data/panels``) and example results (``data/results``) ship.
"""
from __future__ import annotations

import os
from pathlib import Path


def _envpath(var: str, default):
    """Env override for a path: return ``Path(os.environ[var])`` if set, else ``default`` (which may
    be a ``Path`` or ``None``). Lets a GPU/compute run repoint large artifacts without editing code."""
    v = os.environ.get(var)
    if v:
        return Path(v)
    return default


_DATA_ROOT = Path(os.environ.get("TAXEMBED_DATA_ROOT", Path(__file__).resolve().parent / "data"))
PANELS = _DATA_ROOT / "panels"           # small git-tracked panels/resolutions/cluster-ids
RESULTS = _envpath("TAXEMBED_RESULTS", _DATA_ROOT / "results")

# --- taxonomy target space (metazoa testbed target) ---
CKPT = _envpath("TAXEMBED_CKPT", _DATA_ROOT / "metazoa_canonical.pth")
TAXMAP = _envpath("TAXEMBED_TAXMAP", _DATA_ROOT / "metazoa_taxid_to_index.tsv")
EDGELIST = _envpath("TAXEMBED_EDGELIST", _DATA_ROOT / "metazoa.edgelist")
TAXMANIFEST = _envpath("TAXEMBED_TAXMANIFEST", _DATA_ROOT / "metazoa_manifest.json")
ROOT_TAXID = 33208           # Metazoa (from manifest)

# --- taxdump (metazoa-subset copy shipped under panels/taxdump; .dmp files are gitignored) ---
TAXDUMP_DIR = _envpath("TAXEMBED_TAXDUMP_DIR", PANELS / "taxdump")

# --- testbeds (user-provided; ship under the data root or repoint via env) ---
PLA2_H5 = _envpath("TAXEMBED_PLA2_H5", _DATA_ROOT / "pla2_prott5.h5")
PLA2_CSV = _envpath("TAXEMBED_PLA2_CSV", _DATA_ROOT / "pla2_annotations.csv")
PLA2_FASTA = _envpath("TAXEMBED_PLA2_FASTA", _DATA_ROOT / "pla2_sequences.fasta")
TOXFAM_FASTA = _envpath("TAXEMBED_TOXFAM_FASTA", _DATA_ROOT / "toxfam_v2.fasta")
TOXFAM_LABELS = _envpath("TAXEMBED_TOXFAM_LABELS", _DATA_ROOT / "toxfam_v2_labels.csv")
TOXFAM_RESIDUE_H5 = _envpath("TAXEMBED_TOXFAM_RESIDUE_H5", _DATA_ROOT / "toxfam_residue_prott5.h5")
PEPTIDEMINER_H5 = _envpath("TAXEMBED_PEPTIDEMINER_H5", _DATA_ROOT / "peptideminer_prott5_meanpool.h5")

# --- frozen artifacts (git-tracked panels) ---
DATA = _envpath("TAXEMBED_PANELS", PANELS)
PLA2_RESOLUTION = DATA / "pla2_species_resolution.tsv"
MULTIFAMILY_RESOLUTION = DATA / "multifamily_species_resolution.tsv"

# --- constants ---
PCA_DIM = 64                 # reduce ProtT5 1024-d before ridge/LEACE (n<<d guard, spec §3)
RANKS = ["species", "genus", "family", "order", "class", "phylum"]
# Alias map (spec §3): annotation species name (KEY, verbatim from PLA2 `Species` column)
#   -> NCBI scientific name present in the metazoa embedding. Frozen in Task 5 (M1a).
# Spec anticipated 2; the annotations actually need 6 (verified 2026-06-17, all in-embedding).
# Each value's taxid is confirmed present in the 498k metazoa embedding (idx not None) and is a
# bona-fide synonym/spelling-fix of the same biological species — NOT a genus fallback.
ALIAS_MAP = {
    # Misspelling in annotation ("Deinakgistrodon" -> "Deinagkistrodon", k/g transposition).
    "Deinakgistrodon acutus": "Deinagkistrodon acutus",   # taxid 36307, idx 7031 (sharp-nosed pit viper)
    # Recent generic reclassifications (Nannopterum / Urile split from Phalacrocorax); NCBI scientific
    # name is still under Phalacrocorax, exact species epithet preserved.
    "Nannopterum auritus": "Phalacrocorax auritus",        # taxid 56069, idx 13533 (double-crested cormorant)
    "Urile pelagicus": "Phalacrocorax pelagicus",          # taxid 2723871, idx 402907 (pelagic cormorant)
    # Genus reclassification: Ophisaurus gracilis -> Dopasia gracilis (Asian glass lizard).
    "Ophisaurus gracilis": "Dopasia gracilis",             # taxid 182351, idx 48621
    # Trinomial subspecies in annotation -> species-rank synonym (exact subspecies epithet, same genus).
    "Ailurus fulgens styani": "Ailurus styani",            # taxid 424585, idx 107111 (Chinese red panda)
    "Apteryx australis mantelli": "Apteryx mantelli",      # taxid 2696672, idx 398652 (North Island brown kiwi)
}

# --- Stage-1 SP-Metazoa Pfam panel (spec 2026-06-17 v2) ---
SP_EMB_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/uniprot_sprot/per-protein.h5"
SP_METAZOA_H5 = _envpath("TAXEMBED_SP_METAZOA_H5", DATA / "per-protein.h5")   # downloaded once (gitignored)
SP_METAZOA_ANNOTATIONS = _envpath("TAXEMBED_SP_METAZOA_ANNOTATIONS", DATA / "sp_metazoa_annotations.tsv")  # REST snapshot (gitignored; release+SHA in manifest, NOT the filename — nothing joins on the filename)
SP_METAZOA_MANIFEST = DATA / "sp_metazoa_annotations_manifest.json"
SP_METAZOA_RESOLUTION = DATA / "sp_metazoa_resolution.tsv"
SP_METAZOA_PANEL = DATA / "sp_metazoa_panel.tsv"
SP_METAZOA_PANEL_FULL = DATA / "sp_metazoa_panel_full.tsv"  # zero-exclusion twin (B2 robustness)
SP_METAZOA_EC_PANEL = DATA / "sp_metazoa_ec_panel.tsv"
SP_METAZOA_EC_PANEL_FULL = DATA / "sp_metazoa_ec_panel_full.tsv"   # zero-exclusion EC twin (B2)
SP_METAZOA_EXCLUSIONS = DATA / "sp_metazoa_panel_exclusions.tsv"
SP_METAZOA_EC_EXCLUSIONS = DATA / "sp_metazoa_ec_panel_exclusions.tsv"
SP_CLUSTER_IDS = DATA / "sp_metazoa_cluster_ids.tsv"
SP_CLUSTER_IDS_MANIFEST = DATA / "sp_metazoa_cluster_ids_manifest.json"
SP_GATE_SCAN = RESULTS / "sp_metazoa_gate_scan.tsv"
SP_EC_GATE_SCAN = RESULTS / "sp_metazoa_ec_gate_scan.tsv"
SP_BUILD_LOG = RESULTS / "sp_metazoa_build_log.md"
SP_TAXONOMY_QUERY = "reviewed:true AND taxonomy_id:33208"
SP_REST_FIELDS = "accession,organism_id,organism_name,sequence,length,xref_pfam,ec,protein_name,keyword"
# pre-committed gate thresholds (spec §6.1) — DO NOT tune after the gate-scan
SP_PFAM_MEMBER_FLOOR = 50
SP_CROSS_MIN_ORDERS = 4
SP_CROSS_MIN_EFF_ORDERS = 3      # exp(H(order|family)) >= 3
SP_NESTING_MAX = 0.30            # both 1-H(tax|func) and 1-H(func|tax) must be < this
SP_NMI_MAX = 0.70                # NMI(Pfam,EC) must be < this for the contrast to be independent
SP_COVERAGE_MIN = 0.50           # Pfam domain coverage gate (reported-sensitivity in stage-1; see B4)
SP_JOIN_COVERAGE_MIN = 0.98      # >= this fraction of panel accessions must be in the h5
SP_MIN_CROSSED_FAMILIES = 15
SP_EC_LEVEL = 3                  # group EC at 3rd level (sub-subclass)
SP_KNN_K = 5                     # within-stratum kNN k (matches M1)

# Confound-erasure controls (C2b length, C2c composition): the headline "survives" a confound erasure iff
# within-stratum function purity barely moves (rel-drop < this) AND taxonomy stays recoverable above
# chance by at least this margin on the confound-erased reps. Matches the <~10% rel-drop posture the
# function-preservation verdict uses; the tax margin is a modest above-chance floor for the categorical probe.
SP_CONFOUND_PURITY_REL_DROP_MAX = 0.10
SP_CONFOUND_TAX_MARGIN_MIN = 0.05

# READ-side power-floor gates (D2a/D2b, spec §6.1/§6.3) — kept here for parity with the CLEAN gates above;
# pre-committed thresholds belong in config, not buried in a function default.
SP_READ_MIN_FOLD_N = 30          # leave-clade-out folds (and balanced folds) with n below this are dropped
SP_READ_CLAIMED_EFFECT = 0.05    # family-balanced variant "informative" iff accuracy CI half-width < this

# --- Stage-2 all-life target space (cellular_canonical; spec v3 §3) ---
# The cellular_canonical embedding covers all cellular life (three superkingdoms). It is large and
# user-provided: fetch the coordinates + mapping from the Hugging Face release and the parent/depth
# edgelist from the Zenodo deposit, then point these variables at them (see README).
CELLULAR_CKPT = _envpath("TAXEMBED_CELLULAR_CKPT", _DATA_ROOT / "cellular_canonical.safetensors")
CELLULAR_TAXMAP = _envpath("TAXEMBED_CELLULAR_TAXMAP", _DATA_ROOT / "cellular_taxid_to_index.tsv")
CELLULAR_EDGELIST = _envpath("TAXEMBED_CELLULAR_EDGELIST", _DATA_ROOT / "cellular.edgelist")
CELLULAR_ROOT_TAXID = 131567       # cellular organisms

# --- full new_taxdump pin + redirect audit (spec v3 §3/§4.3; resolved at Step-0, captured here) ---
TAXDUMP_DIR_FULL = _envpath("TAXEMBED_FULL_TAXDUMP_DIR", None)   # full new_taxdump dir for the all-life leg
DSS_BASE = os.environ.get("TAXEMBED_SCRATCH_BASE")   # allocation/scratch root for the os.statvfs pre-flight (#7)
TAXDUMP_RELEASE = None             # full new_taxdump release string, recorded by the Step-0 gate (§3)
TAXDUMP_SHA256 = None              # full new_taxdump SHA256, recorded by the Step-0 gate (§3)
TAXDUMP_REDIRECT_ABSENT_MAX = 0    # merged.dmp redirects to ids absent from the node set (§4.3 fail-loud bound)
# Step-0 full-new_taxdump floors (spec §3): a eukaryote/Metazoa-subset dump fails these. Conservative
# discriminating floors — the real full dump has millions of Bacteria / ~10^5 Archaea / ~10^6 Eukaryota.
SP_STEP0_SUPERKINGDOM_NODE_MIN = {"Bacteria": 10_000, "Archaea": 1_000, "Eukaryota": 10_000}

# --- Stage-2 all-life frozen artifacts (gitignored where large; spec v3 §2/§6) ---
ALL_LIFE_PANEL = DATA / "all_life_panel.tsv"
ALL_LIFE_PANEL_FULL = DATA / "all_life_panel_full.tsv"
ALL_LIFE_EC_PANEL = DATA / "all_life_ec_panel.tsv"
ALL_LIFE_EC_PANEL_FULL = DATA / "all_life_ec_panel_full.tsv"
ALL_LIFE_H5 = _envpath("TAXEMBED_ALL_LIFE_H5", DATA / "all_life_per-protein.h5")   # self-embedded (gitignored)

# --- per-family cap + total ceiling (committed NOW; spec v3 §5 — never tuned post-hoc) ---
SP_CAP_C = 1500                    # per-family member cap: cap = min(SP_CAP_C, members)
SP_TOTAL_EMBED_CEILING = int(os.environ.get("TAXEMBED_EMBED_CEILING", 250_000))  # hard ceiling; env-override for slice validation
SP_CAP_SEED = 0                    # project SEED convention

# --- headroom-aware verdict criteria, BOTH legs (spec v3 §7; replace the inline 0.10/0.05 literals) ---
ALL_LIFE_PURITY_REL_DROP_MAX = 0.10        # function-preserved rel-drop margin
ALL_LIFE_MIN_PURITY_HEADROOM = 0.05        # min purity_orig - function_chance for a scorable function leg
ALL_LIFE_MIN_TAX_HEADROOM = 0.05           # min tax_orig - tax_chance for a scorable taxonomy-erasable leg
ALL_LIFE_TAX_COLLAPSE_FRAC_MIN = 0.80      # (tax_orig - tax_erased)/(tax_orig - tax_chance) must exceed this

# --- grain-feasibility thresholds, promoted from clean_eval.py (spec v3 §4.2.5) ---
GRAIN_MIN_FRAC_STRATA = 0.50
GRAIN_MIN_COVERAGE = 0.80

# --- all-life acquisition (spec v3 §4); resolution floor is honestly lower than the h5-join gate ---
SP_RESOLUTION_RATE_MIN = 0.50      # taxid->node resolution floor (realistic all-life value, NOT 0.98)
SP2_METADATA_FIELDS = "accession,organism_id,xref_pfam,length,ec,reviewed"  # `reviewed` → §4.4 twin / §7 batch guard

# --- §4.2.1 span-denominator floor + #9 TrEMBL annotation-completeness relabel bound (committed; §9) ---
ALL_LIFE_SPAN_DENOM_MIN = 0.50     # min non-null-superkingdom fraction of admitted proteins (M-5/§4.2.1)
ALL_LIFE_TREMBL_RELABEL_MAX = 0.50  # a superkingdom losing > this fraction of rank relabels the claim (#9/§4.2.4)

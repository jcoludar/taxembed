"""Trim the locked Experiment-1 metazoa tag to just final + best checkpoints.

Result is locked + already analyzed (analysis_* dirs retained). All per-epoch /
milestone checkpoints remain on LRZ (recovery path), so local deletion is safe
and recoverable. Keeps: <tag>.pth (final), <tag>_best.pth (best), run.json,
analysis_*/ , knn_purity/. Deletes: every *_epoch*.pth (per-epoch + milestone).

Prints a manifest before deleting (auditable). One venv-python invocation.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG = ROOT / "artifacts" / "tags" / "metazoa_lower_lr_bigger_batch"
KEEP = {TAG / "metazoa_lower_lr_bigger_batch.pth",
        TAG / "metazoa_lower_lr_bigger_batch_best.pth"}

# Selection: any checkpoint whose name carries an "_epoch" tag (per-epoch or
# milestone). final/best do NOT contain "_epoch", so they are never selected.
candidates = sorted(p for p in TAG.glob("*_epoch*.pth"))

assert all(k.exists() for k in KEEP), f"KEEP set missing: {[k.name for k in KEEP if not k.exists()]}"
assert not (KEEP & set(candidates)), "SAFETY: a KEEP file matched the delete glob — aborting"

total = sum(p.stat().st_size for p in candidates)
print("KEEP (not touched):")
for k in sorted(KEEP):
    print(f"  {k.name}  ({k.stat().st_size/1e6:.0f} MB)")
print(f"\nDELETE ({len(candidates)} files, {total/1e9:.2f} GB):")
for p in candidates:
    print(f"  {p.name}")

freed = 0
for p in candidates:
    sz = p.stat().st_size
    p.unlink()
    freed += sz
print(f"\nDeleted {len(candidates)} checkpoints, reclaimed {freed/1e9:.2f} GB.")
print("Remaining in tag:")
for p in sorted(TAG.glob("*.pth")):
    print(f"  {p.name}  ({p.stat().st_size/1e6:.0f} MB)")

"""Single source of truth for NCBI taxdump download + extraction.

The NCBI new_taxdump bundle includes `nodes.dmp`, `names.dmp`, and `merged.dmp`,
which are everything the `taxopy.TaxDb` loader needs. We deliberately fetch only
those three files instead of unpacking the full archive (it ships ~30 dmp files
the rest of this project never reads).
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"

_REQUIRED_MEMBERS = ("nodes.dmp", "names.dmp", "merged.dmp")


def ensure_taxdump(
    data_dir: Path,
    *,
    force: bool = False,
    url: str = TAXDUMP_URL,
) -> Tuple[Path, Path, Optional[Path]]:
    """Download + extract the NCBI taxdump under ``data_dir`` if it's missing.

    Returns absolute paths to ``(nodes.dmp, names.dmp, merged.dmp)``. The merged
    path is ``None`` when the archive didn't ship a merged.dmp (rare).

    When ``force=True`` we re-download even if the dmp files already exist. When
    the required files are already present and ``force=False``, no network is
    touched — making this safe to call on a compute node with no outbound access
    as long as the files were staged in advance.
    """

    data_dir = Path(data_dir)
    nodes_file = data_dir / "nodes.dmp"
    names_file = data_dir / "names.dmp"
    merged_file = data_dir / "merged.dmp"

    if not force and nodes_file.exists() and names_file.exists():
        return nodes_file, names_file, merged_file if merged_file.exists() else None

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "new_taxdump.tar.gz"

    print(f"  Downloading NCBI taxdump from {url}")
    print(f"  → {archive_path}")
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as out_f:
        shutil.copyfileobj(response, out_f)

    print(f"  Extracting required members: {', '.join(_REQUIRED_MEMBERS)}")
    extracted: list[str] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member_name in _REQUIRED_MEMBERS:
            try:
                member = tar.getmember(member_name)
            except KeyError:
                continue
            tar.extract(member, path=data_dir)
            extracted.append(member_name)

    if "nodes.dmp" not in extracted or "names.dmp" not in extracted:
        raise RuntimeError(
            f"NCBI taxdump archive at {archive_path} did not include "
            "the expected nodes.dmp / names.dmp files."
        )

    print(f"  ✓ Taxdump ready at {data_dir}/")
    return nodes_file, names_file, merged_file if merged_file.exists() else None


def load_taxdb(data_dir: Path):
    """Load a ``taxopy.TaxDb`` from ``data_dir`` with the dmp files preserved on disk.

    Wraps ``ensure_taxdump`` so the call is safe on a fresh clone, and threads
    ``keep_files=True`` through to taxopy so the dmp files survive load — taxopy's
    default behavior is to delete them, which would force every subsequent CLI
    invocation to re-download from NCBI (a non-starter on offline LRZ compute nodes).
    """
    import taxopy  # local import keeps the module cheap to import when not needed

    nodes_path, names_path, merged_path = ensure_taxdump(data_dir)
    return taxopy.TaxDb(
        nodes_dmp=str(nodes_path),
        names_dmp=str(names_path),
        merged_dmp=str(merged_path) if merged_path is not None else None,
        keep_files=True,
    )


__all__ = ["TAXDUMP_URL", "ensure_taxdump", "load_taxdb"]

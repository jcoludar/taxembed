"""Mac submit-time provenance capture (spec v3 §9).

The GPU cluster src is an rsync'd directory with no .git, so `git rev-parse` at battery time returns
'unknown'. Capture the config git SHA on the Mac and bake it into the payload; every GPU cluster job asserts
it at start (see `assert_present`, wired into each sbatch in D2). `clean_eval._build_provenance`
already reads this file for `config_git_sha`.
"""
import json
import os
import subprocess


def _run(args):
    """Run a command, return stripped stdout or None. Degrades to None if the binary is absent
    (R1-5: a naive `.stdout` access crashes with FileNotFoundError before it is reached)."""
    try:
        return subprocess.run(args, capture_output=True, text=True).stdout.strip() or None
    except (FileNotFoundError, OSError):
        return None


def capture(repo_root, out_path):
    sha = _run(["git", "-C", repo_root, "rev-parse", "HEAD"]) or "unknown"
    mmseqs = _run(["mmseqs", "version"])
    with open(out_path, "w") as f:
        json.dump({"config_git_sha": sha,
                   "captured_at_note": "mac submit-time; GPU cluster has no .git (spec v3 §9)",
                   "mmseqs_version_expected": mmseqs}, f, indent=2)
    return sha


def assert_present(prov_path):
    if not os.path.exists(prov_path):
        raise SystemExit(f"[provenance] {prov_path} absent — capture it on the Mac before submit (spec v3 §9)")
    sha = json.load(open(prov_path)).get("config_git_sha")
    if not sha or sha == "unknown":
        raise SystemExit(f"[provenance] config_git_sha is {sha!r} — refuse to run without a real SHA")
    return sha


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(capture(repo_root=os.path.abspath(os.path.join(here, "..", "..")),
                  out_path=os.path.join(here, "config_provenance.json")))

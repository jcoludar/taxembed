"""Pre-GPU sizing + disk pre-flight (spec v3 §8/§11).

Self-contained POSIX `os.statvfs` — no vendor `dssusrinfo` dependency (no repo sbatch uses it).
The sizing gate is a HARD GO/NO-GO run BEFORE any GPU sbatch: it NO-GOs on N>ceiling, on a projected
per-shard wall-clock over budget, on the battery `--mem` (blocked-kNN working set) exceeding the J2
node RAM (M-4/§11 — the fourth §11 bullet, previously missing), or on the disk/inode pre-flight.
"""
import os


def preflight_disk(base, need_bytes, need_inodes):
    st = os.statvfs(base)
    free_bytes = st.f_bavail * st.f_frsize
    free_inodes = st.f_favail
    ok = free_bytes >= need_bytes and free_inodes >= need_inodes
    out = {"base": base, "free_bytes": int(free_bytes), "free_inodes": int(free_inodes),
           "need_bytes": int(need_bytes), "need_inodes": int(need_inodes), "ok": bool(ok)}
    if not ok:
        raise SystemExit(f"[preflight] DSS {base}: free={free_bytes}B/{free_inodes}i < "
                         f"need={need_bytes}B/{need_inodes}i (spec v3 §8) — abort before the DAG")
    return out


def blocked_knn_working_set(N, block=4096):
    """The blocked/streaming kNN working set (spec §7): block × N × 8 B (float64 row block), NOT the
    dense 3·N²·8 B Gram. Use this when no canary peak-RSS measurement is available."""
    return int(block) * int(N) * 8


def sizing_gate(N, length_hist, seq_per_s, n_shards, shard_time_h, dss_base,
                fasta_bytes, snapshot_bytes, battery_mem_bytes, j2_mem_bytes, margin=1.3):
    """GO/NO-GO over the four §11 lines. `battery_mem_bytes` is the realized battery working set —
    the §8.5 canary peak RSS, or `blocked_knn_working_set(N)` if unmeasured; `j2_mem_bytes` is the J2
    node RAM ceiling. Effective seq/s = N / Σ_bin(N_bin / seq_per_s_bin) (NOT the length-blind
    constant, §11)."""
    from . import config

    if N > config.SP_TOTAL_EMBED_CEILING:
        raise SystemExit(f"[sizing] NO-GO: N={N} > SP_TOTAL_EMBED_CEILING={config.SP_TOTAL_EMBED_CEILING}")

    denom = sum(length_hist[b] / seq_per_s[b] for b in length_hist if seq_per_s.get(b))
    eff = (N / denom) if denom else float("nan")
    gpu_hours_per_shard = (N / eff / n_shards) / 3600.0 if eff == eff and eff else float("inf")
    h5_bytes = N * 1024 * 2                                   # float16 (spec §6/§11); ignores gzip (conservative-high)
    need_bytes = int(h5_bytes + fasta_bytes + snapshot_bytes)
    disk = preflight_disk(dss_base, need_bytes, need_inodes=n_shards + 16)

    # M-4/§11 fourth bullet: battery --mem GO/NO-GO (blocked-kNN working set vs J2 node RAM).
    battery_ok = battery_mem_bytes * margin <= j2_mem_bytes
    time_ok = gpu_hours_per_shard * margin <= shard_time_h
    go = bool(battery_ok and time_ok)
    out = {"go": go, "N": int(N), "effective_seq_per_s": float(eff),
           "gpu_hours_per_shard": float(gpu_hours_per_shard), "h5_bytes": int(h5_bytes),
           "need_bytes": need_bytes, "disk": disk, "shard_time_h": shard_time_h,
           "battery_mem_bytes": int(battery_mem_bytes), "j2_mem_bytes": int(j2_mem_bytes),
           "battery_ok": bool(battery_ok), "time_ok": bool(time_ok)}
    if not battery_ok:
        raise SystemExit(f"[sizing] NO-GO: battery working set {battery_mem_bytes}B × {margin} > "
                         f"J2 --mem {j2_mem_bytes}B (spec v3 §11 blocked-kNN) — abort before any GPU sbatch")
    if not time_ok:
        raise SystemExit(f"[sizing] NO-GO: per-shard {gpu_hours_per_shard:.2f}h × {margin} > "
                         f"--time {shard_time_h}h (spec v3 §11)")
    return out

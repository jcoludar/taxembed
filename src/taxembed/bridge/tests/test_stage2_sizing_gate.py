import os

import pytest

from taxembed.bridge import sizing_gate


# --- A3: os.statvfs disk/inode pre-flight ---

def test_preflight_disk_ok_on_tmp(tmp_path):
    out = sizing_gate.preflight_disk(str(tmp_path), need_bytes=1, need_inodes=1)
    assert out["ok"] is True and out["free_bytes"] > 0


def test_preflight_disk_raises_when_insufficient(tmp_path):
    st = os.statvfs(str(tmp_path))
    too_much = st.f_bavail * st.f_frsize + (1 << 40)        # free + 1 TiB
    with pytest.raises(SystemExit):
        sizing_gate.preflight_disk(str(tmp_path), need_bytes=too_much, need_inodes=1)


# --- A4 + M-4: GO/NO-GO sizing gate including the battery --mem line ---

_OK = dict(n_shards=64, shard_time_h=8.0, dss_base="/tmp", fasta_bytes=1, snapshot_bytes=1,
           battery_mem_bytes=8 * 2 ** 30, j2_mem_bytes=64 * 2 ** 30)


def test_sizing_gate_go_under_all_ceilings():
    out = sizing_gate.sizing_gate(
        N=200_000, length_hist={"short": 120_000, "mid": 60_000, "long": 20_000},
        seq_per_s={"short": 300.0, "mid": 120.0, "long": 30.0}, **_OK)
    assert out["go"] is True
    # effective seq/s is the harmonic-style blend, strictly below the short-bin rate
    assert out["effective_seq_per_s"] < 300.0
    # h5 storage = N*1024*2 bytes (float16)
    assert out["h5_bytes"] == 200_000 * 1024 * 2
    assert out["battery_ok"] is True


def test_sizing_gate_nogo_over_ceiling():
    with pytest.raises(SystemExit):
        sizing_gate.sizing_gate(N=300_000, length_hist={"short": 300_000},
                                seq_per_s={"short": 300.0}, **_OK)


def test_sizing_gate_nogo_over_battery_mem():
    """M-4/§11: an N that fits GPU+disk but whose blocked-kNN working set exceeds the J2
    node RAM must NO-GO BEFORE any GPU sbatch, not OOM mid-J2."""
    with pytest.raises(SystemExit):
        sizing_gate.sizing_gate(
            N=200_000, length_hist={"short": 200_000}, seq_per_s={"short": 300.0},
            n_shards=64, shard_time_h=8.0, dss_base="/tmp", fasta_bytes=1, snapshot_bytes=1,
            battery_mem_bytes=200 * 2 ** 30, j2_mem_bytes=64 * 2 ** 30)   # 200 GB > 64 GB node


def test_sizing_gate_nogo_over_time():
    with pytest.raises(SystemExit):
        sizing_gate.sizing_gate(
            N=200_000, length_hist={"long": 200_000}, seq_per_s={"long": 30.0},
            n_shards=2, shard_time_h=0.1, dss_base="/tmp", fasta_bytes=1, snapshot_bytes=1,
            battery_mem_bytes=8 * 2 ** 30, j2_mem_bytes=64 * 2 ** 30)


def test_blocked_knn_working_set_is_block_times_N_times_8():
    assert sizing_gate.blocked_knn_working_set(250_000, block=4096) == 4096 * 250_000 * 8

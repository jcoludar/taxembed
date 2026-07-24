import pytest

from taxembed.bridge import resolve_all_life, config


class _Resolver:
    MAP = {9606: 11, 562: 12, 2157: None}                   # 2157 present-but-unresolved
    def idx_of_taxid(self, t):
        return self.MAP.get(int(t))


# --- B4: memoized resolver + resolution-rate GO/NO-GO ---

def test_memoized_resolve_unique_only():
    out = resolve_all_life.memoized_resolve([9606, 9606, 562, 9606], _Resolver())
    assert out == {9606: 11, 562: 12}


def test_resolution_preflight_go_above_floor():
    out = resolve_all_life.resolution_preflight([9606, 562, 9606], _Resolver(), floor=0.5)
    assert out["rate"] == 1.0 and out["go"] is True


def test_resolution_preflight_nogo_below_floor():
    # 1 of 3 unique resolves -> 0.33 < 0.5 -> NO-GO (floor is NOT lowered; decision #2)
    with pytest.raises(SystemExit):
        resolve_all_life.resolution_preflight([9606, 2157, 999], _Resolver(), floor=0.5)


# --- M-3: merged.dmp redirect audit + mapping ⊆ node-set subset audit (Mac-testable set arithmetic) ---

def test_redirect_audit_ok_within_bound():
    out = resolve_all_life.redirect_audit({100: 9606, 200: 562}, node_set={9606, 562, 11}, bound=0)
    assert out["n_absent_target"] == 0 and out["ok"] is True


def test_redirect_audit_raises_above_bound():
    with pytest.raises(SystemExit):  # 300->777 absent from node set, bound 0
        resolve_all_life.redirect_audit({100: 9606, 300: 777}, node_set={9606, 562}, bound=0)


def test_redirect_audit_reports_absent_under_bound():
    out = resolve_all_life.redirect_audit({300: 777}, node_set={9606}, bound=5)
    assert out["n_absent_target"] == 1 and out["ok"] is True and 777 in out["absent_targets"]


def test_subset_audit_passes_when_subset():
    out = resolve_all_life.subset_audit({9606, 562}, node_set={9606, 562, 2157})
    assert out["ok"] is True and out["n_not_in_node_set"] == 0


def test_subset_audit_raises_when_not_subset():
    with pytest.raises(SystemExit):
        resolve_all_life.subset_audit({9606, 99999}, node_set={9606, 562})


def test_taxdump_config_constants_exist():
    assert isinstance(config.TAXDUMP_REDIRECT_ABSENT_MAX, int)
    assert hasattr(config, "TAXDUMP_RELEASE") and hasattr(config, "TAXDUMP_SHA256")
    assert hasattr(config, "TAXDUMP_DIR_FULL")

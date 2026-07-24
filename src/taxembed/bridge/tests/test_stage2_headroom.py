from taxembed.bridge import clean_eval


def test_function_preserved_true_with_headroom_small_drop():
    # orig 0.9, chance 0.2 -> headroom 0.7 (>=0.05); after 0.88 -> rel-drop ~0.022 (<0.10) -> preserved
    assert clean_eval._function_preserved(0.90, 0.88, 0.20) is True


def test_function_preserved_false_big_drop():
    assert clean_eval._function_preserved(0.90, 0.60, 0.20) is False   # rel-drop 0.33


def test_function_preserved_not_scorable_when_no_headroom():
    # orig barely above chance -> headroom 0.02 (<0.05) -> NOT-SCORABLE (None), not a pass/fail
    assert clean_eval._function_preserved(0.22, 0.21, 0.20) is None


def test_taxonomy_erasable_true_full_collapse():
    # orig 0.8, chance 0.2 -> headroom 0.6; erased 0.22 -> collapse (0.8-0.22)/0.6=0.967 (>=0.80)
    # AND at-chance: 0.22 <= chance+ε (0.20+0.05=0.25). Spec v3 §7 KEEPS the tax_erased<=chance+ε floor.
    assert clean_eval._taxonomy_erasable(0.80, 0.22, 0.20) is True


def test_taxonomy_erasable_false_partial_collapse():
    # erased 0.55 -> collapse frac (0.8-0.55)/0.6 = 0.417 (<0.80) -> not erasable
    assert clean_eval._taxonomy_erasable(0.80, 0.55, 0.20) is False


def test_taxonomy_erasable_false_collapses_but_not_at_chance():
    # erased 0.28 -> collapse (0.8-0.28)/0.6=0.867 (>=0.80) BUT 0.28 > chance+ε (0.25) -> floor fails -> False
    # (this pins the spec v3 §7 "keep the tax_erased<=chance+ε floor" requirement explicitly)
    assert clean_eval._taxonomy_erasable(0.80, 0.28, 0.20) is False


def test_taxonomy_erasable_not_scorable_when_no_headroom():
    assert clean_eval._taxonomy_erasable(0.23, 0.21, 0.20) is None     # headroom 0.02 < 0.05

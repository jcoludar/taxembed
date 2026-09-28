#!/usr/bin/env python3
"""Write `p2_amendment_8_20260928` into results/p2_heldout_preregistration.json.

Same append-only, byte-preserving discipline as helpers/p2_write_amendment_6.py and
helpers/p2_write_amendment_7.py: refuses if the key exists, splices textually rather than
re-serialising, and asserts every prior byte survives.

WHY AN AMENDMENT BLOCK FOR A DECISION THAT CHANGES NO CODE. amendment_7 ended by naming an open
design decision and deliberately declining to take it: "whether the raw comparison should also
have to pass is a design decision for the USER, not one to take silently here". The USER has now
taken it, and the answer is NO. A reader who reaches amendment_7's disclosure must be able to find
that answer in the same chain, or the record dangles on its single most consequential open
question. Recording a REJECTION is also the only thing that forecloses the failure mode this
chain exists to prevent: a later session, having seen a disappointing number, "helpfully" adding
the raw gate and presenting it as a tightening rather than as post-hoc goalpost-moving. A decision
not to change the design is still a design decision, and it is only worth anything if it is
timestamped before the data.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
PREREG = _REPO / "results" / "p2_heldout_preregistration.json"
KEY = "p2_amendment_8_20260928"

BLOCK = {
    "status": (
        "WRITTEN 2026-09-28, recording a USER DESIGN RULING of the same date. Made BEFORE any P2 "
        "number was read, and that is verified rather than asserted: the 12-element training "
        "array 5812509 COMPLETED (exit 0:0, last element 2026-09-27T06:23:57), but the scoring "
        "job 5815567 (scripts/p2_lrz_score.sh) was still PENDING in the queue at the time of the "
        "ruling and its output directory artifacts/p2_scoring/ DID NOT EXIST on the cluster "
        "(checked by `ls`, which returned 'No such file or directory'). No scorer JSON, and "
        "therefore no P2 verdict, existed anywhere when this was decided. Additive only: every "
        "earlier block is left byte-identical."
    ),
    "the_question_being_answered": (
        "amendment_7's `generalises_no_longer_implies_out_ranking_the_control` disclosed that "
        "under amendment_6's ratio reading, GENERALISES does NOT imply the arm out-ranked its "
        "degree-matched control on the raw held-out metric: an arm whose control beats it ~2.25x "
        "on raw normalized_rank still reads GENERALISES, and so does an arm scoring exactly what "
        "its control scores. amendment_7 recorded `raw_mean_diff_vs_control` and "
        "`arm_beats_control_raw` in the reading and printed them, but GATED NOTHING on them, "
        "explicitly leaving the gating question to the USER. The candidate change -- 'amendment "
        "8' -- would have made the raw head-to-head an additional gate that GENERALISES must also "
        "pass."
    ),
    "the_ruling": {
        "verdict": (
            "REJECTED, 2026-09-28, before any P2 number was read. No gate is added. The verdict "
            "stays the amendment_6 ratio reading: the arm's improvement over its OWN tree's "
            "training-free degree prior, compared with the degree-matched control's improvement "
            "over ITS own prior. The raw head-to-head continues to be REPORTED beside it, exactly "
            "as the engine already does."
        ),
        "user_words_verbatim": (
            "If we are controlling how fast a runner runs, then making control run on a better "
            "tarmac is not fair."
        ),
        "the_reasoning": (
            "The degree-matched control's task is ~3.8x EASIER for a training-free ranker "
            "(degree-prior normalized_rank 0.1538 real vs 0.0404 degmatch, measured on these very "
            "production splits and independently re-derived 2026-09-26). A raw-score gate would "
            "therefore require the arm to beat a number its control gets for free by virtue of an "
            "easier task -- penalising the arm for the CONTROL'S tarmac, not for anything the "
            "model did or failed to do. That is the identical defect amendment_6 was written to "
            "remove; re-introducing it as an additional gate would reinstate it under a different "
            "name and, per helpers/p2_amendment6_reachability.py scenario R5, would again make "
            "GENERALISES structurally unreachable even for a checkpoint that saw every held-out "
            "edge."
        ),
    },
    "what_changes_in_the_code": (
        "NOTHING. This block adds no flag, retires none, and alters no threshold. The reading "
        "rule for the 5812509 array is UNCHANGED and remains "
        "`--amendment-1 --amendment-2 --amendment-4 --amendment-6`. Verified in the "
        "implementation rather than assumed: taxembed/eval/preregistration.py:936 computes "
        "`arm_beats_control_raw` and line 1063 places it in the reading; "
        "scripts/apply_preregistration.py:156-157 prints it and emits an explicit warning line "
        "when GENERALISES is returned while `arm_beats_control_raw` is False. It appears in no "
        "gate expression. Reported, not gated -- before this ruling and after it."
    ),
    "what_a_reader_of_the_verdict_must_still_do": (
        "The rejection does NOT retire amendment_7's disclosure; it makes it permanent. A "
        "GENERALISES verdict accompanied by `arm_beats_control_raw: false` is a real and "
        "publishable outcome under this design, and the manuscript sentence built on it must say "
        "which comparison was made -- the engine's own verdict prose already states 'its "
        "improvement on its OWN tree's training-free degree prior exceeds the control's "
        "improvement on ITS own tree's prior'. A sentence claiming the arm simply beat its "
        "control would be wrong, and the raw fields printed beside the verdict are what makes "
        "that detectable."
    ),
    "scope_limit": (
        "This is a ruling about P2's cross-tree gating only. It says nothing about the "
        "amendment_7 items left open and unfixed at that time (I-A's chance_mrr_mean k=1 "
        "inflation, I-B's all-below-all strictness, I-C's duplicate-blind "
        "assert_baselines_agree_across_seeds), and nothing about the separate manuscript-integrity "
        "question of what roll-set sizes the published Task 8/9 per_run_values were computed over."
    ),
    "verification": (
        "Three checks, each able to fail: (1) `ssh ai sacct -X -j 5812509` -- all 12 elements "
        "COMPLETED with ExitCode 0:0, so the ruling is not pre-empting a failed array; (2) `ls` "
        "on the cluster's artifacts/p2_scoring/ -- absent, so no scorer output existed; (3) grep "
        "for `arm_beats_control_raw` across src/ and scripts/ -- four live sites, all of them "
        "recording or printing, none in a gate expression. The ~3.8x difficulty figure is cited "
        "from amendment_6/amendment_7, where it was measured and re-derived; it is not re-measured "
        "here."
    ),
}


def main() -> int:
    original_text = PREREG.read_text()
    prereg = json.loads(original_text)

    if KEY in prereg:
        print(f"REFUSED: {KEY} already exists. An amendment is written once.")
        return 1

    before = {k: json.dumps(v, sort_keys=True) for k, v in prereg.items()}

    closing = original_text.rstrip()
    if not closing.endswith("}"):
        print("REFUSED: the file does not end with '}'.")
        return 1
    body = closing[:-1].rstrip()
    if not body.endswith("}"):
        print("REFUSED: cannot find the last top-level value to append after.")
        return 1
    rendered = json.dumps({KEY: BLOCK}, indent=2)
    inner = rendered[rendered.index("\n") + 1: rendered.rindex("\n")]
    PREREG.write_text(body + ",\n" + inner + "\n}\n")

    after_text = PREREG.read_text()
    after_doc = json.loads(after_text)
    after = {k: json.dumps(v, sort_keys=True) for k, v in after_doc.items()}
    changed = [k for k in before if before[k] != after.get(k)]
    added = [k for k in after if k not in before]
    prefix_preserved = after_text.startswith(body)

    print(f"added            : {added}")
    print(f"changed          : {changed or 'none'}")
    print(f"prior bytes kept : {prefix_preserved}")
    if changed or added != [KEY] or not prefix_preserved:
        print("FAILED: not purely additive. Restore with "
              "`git checkout -- results/p2_heldout_preregistration.json`.")
        return 1
    print(f"OK: {KEY} written, {len(after)} top-level keys, every prior byte identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

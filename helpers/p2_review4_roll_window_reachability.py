"""READ-ONLY PROBE (review 4, 2026-09-26).

C-C's hard raise (`p2_run_value`, preregistration.py:432) refuses any roll set that is not
exactly `P2_ROLL_WINDOW` long. Two questions:

  1. How many `${tag}_epoch*.pth` files does a run of N epochs ACTUALLY leave on disk? The
     trainer's queue is `deque(maxlen=5)` and it `os.remove`s `queue[0]` before appending
     (train_small.py:928-934). Replays that logic exactly, on real files in a temp directory.

  2. Can the 10-element roll set the whole of C-C is justified by actually arise from the
     resubmission scenario the code comments describe? The checkpoint PATHS are deterministic
     functions of the epoch number, so a second attempt writes -- and later deletes -- the same
     paths the first attempt left behind.

Writes only into a throwaway temp directory. Touches nothing in the repo.
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import deque
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from taxembed.eval.preregistration import P2_ROLL_WINDOW  # noqa: E402

TAG = "p2_vis00_s0"


def run_attempt(dirpath: Path, last_epoch: int, save_every: int = 10) -> None:
    """Replay train_small.py:907-942's checkpoint bookkeeping verbatim, with touch() for
    torch.save()."""
    queue: deque = deque(maxlen=5)                       # train_small.py:651
    for epoch in range(1, last_epoch + 1):
        ckpt = dirpath / f"{TAG}_epoch{epoch}.pth"       # :925
        ckpt.touch()                                     # :926 torch.save
        if len(queue) >= queue.maxlen:                   # :929
            old = queue[0]                               # :930
            if os.path.exists(old):                      # :931
                os.remove(old)                           # :932
        queue.append(ckpt)                               # :934
        if save_every > 0 and epoch % save_every == 0:   # :940
            (dirpath / f"{TAG}_milestone_epoch{epoch}.pth").touch()


def count_roll(dirpath: Path) -> list[int]:
    """The set `p2_lrz_score.sh` globs as the `_roll` group: ${tag}_epoch*.pth."""
    out = []
    for p in dirpath.glob(f"{TAG}_epoch*.pth"):
        out.append(int(p.name.split("_epoch")[1].removesuffix(".pth")))
    return sorted(out)


def main() -> int:
    print(f"P2_ROLL_WINDOW (the engine's hard requirement) = {P2_ROLL_WINDOW}")
    print()
    print("1. FRESH RUN of N epochs -- how many rolling checkpoints survive?")
    print(f"{'N epochs':>10} {'n_roll':>8}  epochs on disk           engine verdict")
    for n in (1, 2, 3, 4, 5, 6, 10, 20, 200):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            run_attempt(d, n)
            roll = count_roll(d)
            ok = "OK" if len(roll) == P2_ROLL_WINDOW else "RAISES"
            shown = roll if len(roll) <= 6 else f"[{roll[0]}..{roll[-1]}]"
            print(f"{n:>10} {len(roll):>8}  {str(shown):<24} {ok}")

    print()
    print("2. THE RESUBMISSION SCENARIO C-C is justified by")
    print("   (attempt 1 dies at epoch E1; attempt 2 starts from scratch and reaches E2)")
    print(f"{'E1':>6} {'E2':>6} {'n_roll':>8}  epochs on disk")
    for e1, e2 in ((120, 200), (195, 200), (196, 200), (199, 200), (200, 200),
                   (200, 150), (200, 120), (200, 3), (120, 3)):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            run_attempt(d, e1)          # attempt 1, killed at E1
            run_attempt(d, e2)          # attempt 2, NO clearing (pre-C-C behaviour)
            roll = count_roll(d)
            ep200 = (d / f"{TAG}_epoch200.pth").exists()
            ms200 = (d / f"{TAG}_milestone_epoch200.pth").exists()
            pre = ("passes" if (ep200 and ms200) else "BLOCKS")
            print(f"{e1:>6} {e2:>6} {len(roll):>8}  {str(roll):<46} "
                  f"old ep200 pre-flight: {pre}")

    print()
    print("3. Does `${tag}_epoch*.pth` also capture the MILESTONE files?")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        run_attempt(d, 200)
        allp = sorted(p.name for p in d.glob("*.pth"))
        print(f"   files present: {len(allp)}; matched by the roll glob: "
              f"{len(count_roll(d))}; milestone files: "
              f"{len([p for p in allp if '_milestone_' in p])}")
        print(f"   roll glob result: {count_roll(d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Wall-time budget of the full 4B run per callback cadence, for ticket 06.

    python .scratch/milestone-4-full-fine-tune/probes/cadence_budget.py CALLBACK_S_PER_50_DOCS ...

Every figure but the callback time is measured or fixed: 1,298 train examples after the one
over the sequence budget is dropped, effective batch 16, 3 epochs (246 steps), 25 s per step
(the probes' upper mean), and the free-tier budget of ticket 02: sessions of at most 3 h, a
checkpoint at least every 30 min. A checkpoint lands after the eval point's callback, so the
longest gap between checkpoints is one interval of steps plus one callback. The callback time
scales linearly with the dev subset.
"""

import math
import sys

sys.path.insert(0, "src")
from train import eval_interval, training_steps

N_TRAIN = 1298
EFFECTIVE_BATCH = 16
EPOCHS = 3
STEP_S = 25.0
OVERHEAD_S = 5 * 60  # model load, dataset, first-batch check, final push, per session
PUSH_S = 30  # one checkpoint push: adapter + 8-bit optimizer state
SESSION_S = 3 * 3600
CHECKPOINT_GAP_S = 30 * 60
EVAL_EVERY = (0.25, 0.5, 1.0)
DEV_DOCS = (50, 100, 200)


def main(callback_times: list[float]) -> None:
    steps = training_steps(N_TRAIN, effective_batch_size=EFFECTIVE_BATCH, epochs=EPOCHS)
    print(f"{steps} steps, {steps * STEP_S / 3600:.2f} h of pure training at {STEP_S:.0f} s/step\n")
    print(
        "| callback s/50 docs | eval_every | dev_docs | interval | eval points | wall h "
        "| sessions (≤ 3 h) | max checkpoint gap min | gap ≤ 30 min |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for per_50 in callback_times:
        for every in EVAL_EVERY:
            interval = eval_interval(steps, eval_every=every, epochs=EPOCHS)
            points = math.ceil(steps / interval)
            for docs in DEV_DOCS:
                callback = per_50 * docs / 50
                wall = steps * STEP_S + points * (callback + PUSH_S)
                sessions = math.ceil(wall / (SESSION_S - OVERHEAD_S))
                gap = interval * STEP_S + callback + PUSH_S
                print(
                    f"| {per_50:.0f} | {every} | {docs} | {interval} | {points} "
                    f"| {(wall + sessions * OVERHEAD_S) / 3600:.2f} | {sessions} "
                    f"| {gap / 60:.0f} | {'yes' if gap <= CHECKPOINT_GAP_S else 'NO'} |"
                )


if __name__ == "__main__":
    main([float(arg) for arg in sys.argv[1:]] or [150.0, 300.0, 600.0])

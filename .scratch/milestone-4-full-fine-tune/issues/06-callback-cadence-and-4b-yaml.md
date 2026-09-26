# 06 Callback cadence and the 4B YAML

Type: grilling
Status: resolved
Blocked by: 01, 02, 05, 11
HITL: yes

## Question

With the 4B's measured step and callback times (01) and a session budget
(02), write `configs/qwen3-4b-r16.yaml`'s values and settle the callback:

- How many eval points and how many dev documents per point fit the budget?
  Does checkpoint cadence stay tied to eval cadence (milestone-3 ticket 08),
  or does resume insurance want checkpoints more often than evaluation?
- `per_device_batch_size`, generation `batch_size`, `max_new_tokens`.
- Whether to add per-eval-point output logging (05).
- Confirm milestone-3's promise that milestone 4 "changes values only, no new
  keys" — or name the key that breaks it and why.

## Comments

### 2026-09-26: budget prep (no decisions)

`probes/cadence_budget.py CALLBACK_S_PER_50_DOCS ...` computes wall time,
sessions and the longest checkpoint gap per `eval_every` × `dev_docs`. It uses the
measured 25 s/step (ticket 01), 246 steps (1,298 kept examples, effective batch 16,
3 epochs), and ticket 02's free-tier budget: sessions ≤ 3 h, a checkpoint at least every
30 min. A checkpoint lands after its eval point's callback, so the longest gap is
one interval plus one callback. Pure training is **1.71 h**. Placeholder callback
times of 150 / 300 / 600 s per 50 documents show:

- **`eval_every: 1.0` never meets the 30-min gap** (37–75 min), whatever the
  callback costs. Keeping checkpoints tied to eval points rules it out.
- **`eval_every: 0.5` with 50 dev documents** (interval 41, **6** eval points,
  not the 7 the map's charting notes estimated) fits one ≤ 3 h session and a ≤ 28-min gap even at
  600 s per callback: 2.1 / 2.3 / 2.8 h.
- **`eval_every: 0.25`** (interval 20, 13 points) keeps the gap ≤ 19 min at 50
  documents but costs 0.35–1.3 h more wall time than 0.5.
- **200 dev documents per eval point** needs 2–4 sessions unless the callback
  is ≤ 150 s per 50. The final-evaluation ticket already scores all 200 once, after training.

To do when the fixed probes land: rerun with their measured callback times, plus a
worst case where every batch runs to 512 tokens (ticket 05).

## Answer

Resolved 2026-09-26 by grilling (all recommendations accepted). Inputs:
ticket 01 (step time), 11 (callback 243 s per 50 docs at generation batch 8,
≈ 1,050 s worst case), 02 (≤ 3 h sessions, a checkpoint at least every 30 min), 05.

1. **Cadence: `eval_every: 0.5`, `dev_docs: 50`**, with checkpoints tied to eval
   points as before. That is interval 41, **6 eval points**, the milestone-3 cadence on the
   same first-50 dev subset. Measured: ≈ 2.25 h, one session, ≤ 22-min
   checkpoint gap. Worst case (every batch loops to 512): 3.7 h, 2 sessions,
   a 35-min gap. That breach is accepted: it costs ≤ 5 min extra of lost work, and
   resume is the normal path. Rejected: 0.25 (+0.5 h measured for more curve points);
   200 docs per point (2–3 sessions; the final eval scores all 200 once);
   1.0 (a gap of 37 min or more whatever the callback costs).
2. **`per_device_batch_size: 2`**. Memory is identical at 4, and 2 is faster
   (22.4 vs 24.5 s/step).
3. **Generation `batch_size: 8`**. Measured 243 s vs 292 s at 4, with a 10.2 GiB
   whole-device peak. 16 is unmeasured at 4B and was rejected (an OOM at an eval point
   would kill the run). `max_new_tokens` stays 512 (ticket 05).
4. **Per-eval-point logging, always on, no new key.** The `predictions` table
   is logged at every eval point, and the separate end-of-run table goes (it
   duplicated the last eval point; ticket 08's cross-check reads that eval
   point's table). The callback also logs `dev/diagnostics/n_cut_off`.
5. **Names:** `configs/qwen3-4b-r16.yaml`, `name: qwen3-4b-r16`, adapter
   `fnl-es/qwen3-4b-muc4-lora-r16`, tags `[fine-tuned, qlora]`.
6. **"Values only, no new keys" holds.** The YAML differs from the smoke YAML
   in `name`, `tags`, `model`, `adapter`, `generation.batch_size`, and drops
   `train_docs`. `dev_docs: 50` stays, so the smoke YAML comment ("omits the
   subsets") is half wrong; the build fixes it. Resume (ticket 07) may still add a key or flag.
7. **Ticket 08's de-risk step is amended:** serve the 4B probe adapter
   (`fnl-es/qwen3-4b-muc4-lora-probe`, run `6svccyzx`, reference F1 15.8 on
   the first 50 dev docs), not the 0.6B smoke adapter. It tests the real run's shape.
8. **Probe leftovers:** keep the `probe` / `diagnostic` W&B runs (cited as
   evidence). Keep the Hub repo until step 7 has run; deleting it afterwards is the
   user's call. The build's first commit deletes `configs/qwen3-4b-probe-b{2,4}.yaml`.
   The `probes/` scripts stay with the map.

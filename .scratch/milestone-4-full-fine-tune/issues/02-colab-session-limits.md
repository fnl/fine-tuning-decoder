# 02 Free Colab T4 session limits

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

A full 4B run is at least an hour of training plus however long the
callbacks take. How long can it actually live on the free tier? Establish,
with sources and dates (official FAQ, recent first-hand reports):

1. Maximum session length on a free T4 runtime, and whether it is a hard cap
   or "up to".
2. Idle / inactivity timeout: what counts as activity (a running cell? browser
   tab open?), and whether a long-running cell with a closed tab survives.
3. Daily / rolling GPU quota: how it manifests (refused GPU, downgrade), how
   long until it resets, and whether one long session burns tomorrow's.
4. Background execution: is it free-tier at all today?
5. Anything that terminates sessions early (disk, RAM, "unusual activity").

Answer with a working budget: the wall time a single run should be designed
to fit in, and how often a resume should be expected.

## Answer

Resolved 2026-09-25 by a research subagent. Full findings with sources:
branch `research/colab-session-limits` (commit `ac2088d`),
`.scratch/milestone-4-full-fine-tune/research/02-colab-session-limits.md`.
Caveat: no first-hand free-tier duration reports from 2025–2026 were found;
duration figures come from 2021–2024 GitHub issues, some from paid users.

**Working budget:** design each free-T4 session for **≤ 3 h wall time**; push
a Hub checkpoint **at least every 30 min**; expect **one resume per 2–3 h** of
run; budget a possible GPU lockout of up to **24 h** after heavy use. Resume
is the normal path, not a contingency.

1. **Session length:** "at most 12 hours, depending on availability and your
   usage patterns" ([FAQ](https://research.google.com/colaboratory/faq.html),
   read 2026-09-25) — a ceiling, no floor. Reported terminations at 1–3 h
   (colabtools#4430, 2024-03-11; fast-stable-diffusion#2744, 2024-02, paid,
   anecdotal).
2. **Idle timeout:** unpublished; blogs claim ~90 min without a source. The FAQ
   implies idle timeouts apply only once code stops running, but free-tier
   coverage is unclear — assume a closed tab kills the session.
3. **Quota:** refused with "Cannot connect to GPU backend … usage limits"; no
   published reset. Reports: 12–24 h, occasionally 3–6-day lockouts
   (colabtools#1964, 2021; #3534, 2023). Heavy use likely lengthens the next
   wait (unverified).
4. **Background execution:** paid only (Pro+, Google AI Ultra per the
   [2026-09-22 announcement](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)).
   Not free-tier.
5. **Other early kills:** SSH / web-UI usage (may terminate without warning),
   RAM or disk OOM, idle GPU runtimes (allocated but unused), high demand.

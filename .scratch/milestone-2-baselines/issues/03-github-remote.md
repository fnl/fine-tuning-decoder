# 03 Create the public GitHub remote and push main

Type: task
Status: resolved
Blocked by: —
HITL: yes

## Question

The notebook clones the repo at a pinned ref, so the repo needs a hosted
remote. Decided: public `github.com/fnl/fine-tuning-decoder`, default branch
`main`. The user is logged in with `gh` as `fnl`.

Agent-driven steps (run only when the user says "go"):

0. **Fix `.gitignore` first**: `data/` also ignores `src/data/`, so
   `src/data/prepare.py` has never been committed (found in ticket 04,
   verified with `git check-ignore -v`). Change the pattern to `/data/`,
   `git add src/data`, commit.
1. `gh repo create fnl/fine-tuning-decoder --public --source . --remote origin --push`
   (or `git remote add origin ...` + `git push -u origin main` if the repo
   is created in the UI).
2. Confirm `.gitignore` excludes `data/raw`, `data/prepared`, caches, and
   nothing secret is tracked (`git ls-files | grep -i -E 'token|key'` → empty).
3. Decide with the user whether `docs/PROJECT_PLANNING.md` (currently
   untracked) is committed first.

Resolved when `git ls-remote origin main` returns the local HEAD. Record: the
repo URL and the pushed SHA.

## Answer

Resolved by the user 2026-09-20 (steps 1–2): `origin` =
https://github.com/fnl/fine-tuning-decoder.git, `main` pushed at
`9214c693b732adf81fa002d1e945879a0d085f37` (= local HEAD).
`.envrc` is git-ignored; no token/key files tracked.

Step 0 was **not** done: `.gitignore` line 14 `data/` still ignored
`src/data/prepare.py`, so the pushed tree lacks the codec that tickets 07/09
import (`render_target`, `canonical_events`, `SYSTEM_PROMPT`, `parse_target`).
Agent changed the pattern to `/data/` (working tree, uncommitted); pending:
`git add src/data .gitignore`, commit, push — and `docs/PROJECT_PLANNING.md`
(step 3) is still untracked, user's call whether it goes in the same commit.

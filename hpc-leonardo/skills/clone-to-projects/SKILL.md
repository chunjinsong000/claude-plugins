---
name: clone-to-projects
description: Use when the user wants to git-clone a repository on LEONARDO ("git clone X", "clone this repo", "下载 / 拉这个 repo"). Defaults to placing the clone under `$WORK/$USER/` per the team's storage layout. Use the user's explicit path if they specified one.
---

# Clone Repo to $WORK/$USER/

## Overview

Default destination for any `git clone` is **`$WORK/$USER/<repo_name>/`**.
This matches the team convention: **code lives in `projects/`, data lives in
`scratch/`** (see `../../references/storage-layout.md` § 2).

Override only when the user explicitly says where to put it.

## Steps

1. **Get the repo URL** from the user.
   - Accept `https://github.com/...`, `git@github.com:...`, `git://...`.
   - Reject: anything that doesn't look like a git URL.

2. **Derive the local directory name** from the URL's last path segment,
   stripping `.git`:
   - `https://github.com/Dao-AILab/flash-attention.git` → `flash-attention`
   - `https://github.com/openai/whisper` → `whisper`
   - `git@github.com:huggingface/peft.git` → `peft`

3. **Decide the destination**:
   - **Default**: `$WORK/$USER/<derived_name>/`
   - **Override**: if the user said something like "clone it to X" / "put
     it in scratch/Y" / "I want it under ~/work/Z", use their path.
     Don't argue unless they're picking somewhere actively bad
     (see "Forbidden destinations" below).
   - If `$WORK/$USER/<derived_name>/` already exists and is non-empty:
     STOP. Tell the user the path is taken and ask what to do (rename?
     pull instead? clone to a different name?). Do NOT overwrite.

4. **Confirm the node** (skill-internal sanity check):
   - Run `hostname` mentally / explicitly. If it matches `loginNN.leonardo.local`
     (any of `login01..login08`, reached via `leo`) → fine for clone (login nodes
     have direct outbound internet on LEONARDO).
   - If it matches `dmoverN` / `data.leonardo.cineca.it` → also fine for clone, but
     usually datamovers are for `rsync`/`scp`, not git work.
   - If on a Booster compute node (`lrdnXXXX`) or DCGP compute (`lrdcXXXX`) →
     compute nodes *generally* have direct internet on LEONARDO (unlike BSC),
     but verify with `curl -I https://github.com` before relying on it. If the
     verify fails, clone from `leo` first and access the repo via shared filesystem.

5. **Run the clone** into `$WORK/$USER/`:
   ```bash
   mkdir -p $WORK/$USER                                 # ensure project area exists
   cd $WORK/$USER
   git clone <url> <derived_name>
   ```

   For repos with submodules (e.g. `flash-attention` has `cutlass`):
   ```bash
   git clone --recursive <url> <derived_name>
   ```

   For repos where the user only wants a specific tag/version (cuts
   download size):
   ```bash
   git clone --depth 1 --branch <tag> --recursive <url> <derived_name>
   ```

6. **Verify**:
   - `cd $WORK/$USER/<derived_name> && git log -1 --oneline` (sanity-check
     it actually got commits).
   - `ls $WORK/$USER/<derived_name>/.git` (should exist).
   - If submodules expected: `ls $WORK/$USER/<derived_name>/<known-submod>`
     should be non-empty.

7. **Show the user**:
   - Final absolute path (`$WORK/$USER/<derived_name>/`)
     AND the convenient symlink (`$WORK/$USER/<derived_name>/`).
   - The latest commit hash + message.
   - **Optionally suggest** creating the per-project mirror in scratch:
     ```bash
     mkdir -p ~/scratch/<derived_name>
     ```
     Only suggest this if the project will obviously store data
     (training, datasets, models) — not for utility libraries or pure
     code repos.

## Forbidden destinations

Refuse to clone (or warn loudly) if the user picks one of these:

| Destination | Why bad |
|---|---|
| `$HOME/<name>/` directly (not via symlink) | $HOME is 50 GB; many repos blow past that |
| `$SCRATCH/` | scratch is for **data**, not code (per layout § 2) |
| `~/.local/`, `~/.config/`, etc. | these are for tool config, not project code |
| Anywhere outside the user's owned dirs | permissions |
| An existing non-empty directory | might overwrite work |

If the user really insists on one of these, do it but flag the policy
violation in the response so they know.

## Common Mistakes — DO NOT MAKE THESE

| Mistake | Right way |
|---|---|
| Cloning to `$HOME/<name>` (50 GB quota) | Use `$WORK/$USER/<name>/` |
| Cloning to `$SCRATCH/<name>` | `$SCRATCH` auto-purges after 40 days; code goes in `$WORK` |
| Cloning on a compute node when internet is gated | Verify with `curl -I` first; if blocked, clone from `leo` and access via shared filesystem |
| Using `git clone` (no submodules) for repos with submodules | Use `--recursive` if `.gitmodules` exists |
| Overwriting an existing dir without asking | Stop; ask if user wants to rename / pull / use new path |
| Auto-running `pip install` / `make` after clone | NEVER. Clone only. User decides next steps. |
| Cloning to a hardcoded `$WORK/<someone-else's-user>/...` path | Use `$WORK/$USER/` for portability (other team members clone the same code to their own `$USER` area) |

## What this skill does NOT do

- Run `pip install` / `make` / `setup.py` after cloning — user's call.
- Build wheels — see `../../references/storage-layout.md` § 7
  for the team's pre-compiled wheels location.
- Update an existing clone (`git pull`) — out of scope; just `cd` and
  `git pull` manually.
- Mirror to `~/scratch/<name>/` — only suggested, never auto-created
  unless user asks.

## Background References

- `../../references/storage-layout.md` — projects/ vs scratch/ convention,
  per-project mirroring, portability rule
- `../../references/hardware.md` § 1 — node classes (login / datamover /
  Booster compute / DCGP compute), where outbound internet is available

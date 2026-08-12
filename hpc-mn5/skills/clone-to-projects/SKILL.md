---
name: clone-to-projects
description: Use when the user wants to git-clone a repository on BSC MareNostrum5 ("git clone X", "clone this repo", "下载 / 拉这个 repo"). Defaults to placing the clone under `~/projects/` (= `/gpfs/projects/ehpc1003/${USER}/`) per the team's storage layout. Use the user's explicit path if they specified one.
---

# Clone Repo to ~/projects/

## Overview

Default destination for any `git clone` is **`~/projects/<repo_name>/`**
which resolves to `/gpfs/projects/ehpc1003/${USER}/<repo_name>/`. This
matches the team convention: **code lives in `projects/`, data lives in
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
   - **Default**: `~/projects/<derived_name>/`
   - **Override**: if the user said something like "clone it to X" / "put
     it in scratch/Y" / "I want it under ~/work/Z", use their path.
     Don't argue unless they're picking somewhere actively bad
     (see "Forbidden destinations" below).
   - If `~/projects/<derived_name>/` already exists and is non-empty:
     STOP. Tell the user the path is taken and ask what to do (rename?
     pull instead? clone to a different name?). Do NOT overwrite.

4. **Confirm the node** (skill-internal sanity check):
   - Run `hostname` mentally / explicitly. If it starts with `alogin` →
     fine for clone (login nodes have internet via proxy).
   - If it starts with `transfer` → also fine (better for very large
     repos with many files).
   - If on a compute node (`as*`, `gs*`) → compute nodes have **no
     internet**. Clone will fail. Tell the user to clone from an
     `alogin*` or `transfer*` node first, then come back.

5. **Run the clone**:
   ```bash
   mkdir -p ~/projects                                  # ensure symlink target exists
   cd ~/projects
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
   - `cd ~/projects/<derived_name> && git log -1 --oneline` (sanity-check
     it actually got commits).
   - `ls ~/projects/<derived_name>/.git` (should exist).
   - If submodules expected: `ls ~/projects/<derived_name>/<known-submod>`
     should be non-empty.

7. **Show the user**:
   - Final absolute path (`/gpfs/projects/ehpc1003/${USER}/<derived_name>/`)
     AND the convenient symlink (`~/projects/<derived_name>/`).
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
| `$HOME/<name>/` directly (not via symlink) | $HOME is 80 GB; many repos blow past that |
| `/gpfs/scratch/.../` | scratch is for **data**, not code (per layout § 2) |
| `~/.local/`, `~/.config/`, etc. | these are for tool config, not project code |
| Anywhere outside the user's owned dirs | permissions |
| An existing non-empty directory | might overwrite work |

If the user really insists on one of these, do it but flag the policy
violation in the response so they know.

## Common Mistakes — DO NOT MAKE THESE

| Mistake | Right way |
|---|---|
| Cloning to `$HOME/<name>` (not via `~/projects/`) | Use `~/projects/<name>/` (symlink → `/gpfs/projects/...`) |
| Cloning to `~/scratch/...` | scratch is for data; code goes in projects |
| Cloning on a compute node (no internet) | Clone from `alogin*` first, then access from compute |
| Using `git clone` (no submodules) for repos with submodules | Use `--recursive` if `.gitmodules` exists |
| Overwriting an existing dir without asking | Stop; ask if user wants to rename / pull / use new path |
| Auto-running `pip install` / `make` after clone | NEVER. Clone only. User decides next steps. |
| Cloning to a hardcoded `/gpfs/projects/ehpc1003/vlk370419/...` path | Use `~/projects/` symlink for portability (see storage-layout § 8) |

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
- `../../references/hardware.md` § 1 — node classes (login / transfer /
  compute), why compute can't `git clone`

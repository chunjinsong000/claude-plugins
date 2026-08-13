# LEONARDO — Storage Layout Convention

How we organize code, data, environments, and logs across `projects/`,
`scratch/`, and `$HOME`. Read this before placing any new file. Follow
this convention so that **other team members can clone your code and run
it with no path edits**.

**Related**: `slurm.md` (SLURM directive syntax for paths) ·
`lustre-resilience.md` (when Lustre IO hangs).

---

## 1. Account: `aifac_f02_378`

All shared storage lives under the `aifac_f02_378` allocation:

| Path | Quota | Backup | Lifecycle |
|---|---|---|---|
| `$WORK/` | 1 TB shared | yes | valid until **2027-03-25** |
| `$SCRATCH/`  | 50 TB shared | **NO BACKUP** | valid until **2027-03-25** |
| `/leonardo/home/userexternal/$USER/` |  50 GB | yes | personal, permanent |

`aifac_f02_378` here is the **storage account** — fixed until 2027-03-25 even
when the job-submission account rotates.

---

## 2. The big rule: code in `projects/`, everything else in `scratch/`

| Lives under `projects/` | Lives under `scratch/` |
|---|---|
| Git repos (source code, configs) | **Conda environments** |
| Small shared text files | **Datasets, video, pre-processed data** |
| `*.py`, `*.sh`, `*.yaml` | **Model weights, checkpoints** |
| `README.md`, `requirements.txt` | **SLURM logs (stdout/stderr)** |
| Notebooks (small ones) | **Build artifacts (wheels, cache)** |
| Plans / docs | Job outputs |

**Why**: `projects/` has 1 TB shared across the whole team — fills up if
you put data there. `scratch/` has 50× the space but no backup, so it's
right for things that can be regenerated (envs, downloads, derived data).

---

## 3. Per-project mirroring

For every project named `<name>`:

```
$WORK/$USER/<name>/    ← code (git repo)
$SCRATCH/<name>/     ← datasets, ckpts, outputs
```

The two directories share a name on purpose so you can locate a
project's data from its code path with a single `s/projects/scratch/`.

Examples currently:

| projects/ entry | scratch/ entry | Used for |
|---|---|---|
| `DiffSynth-valka/` | `DiffSynth-valka/` | code ↔ training data + ckpts |
| `DiffSynth1/` | `DiffSynth1/` | code ↔ data |
| `LiveAvatar/` | `LiveAvatar/` | code ↔ data |
| `MatAnyone/` | (none yet) | inference-only project |

---

## 4. Archiving old projects (`_archive/`)

When a project is no longer actively worked on, **move it to `_archive/`
in both projects/ and scratch/** — keep the per-project mirror.

```
$WORK/$USER/_archive/<old-project>/    ← old code
$SCRATCH/_archive/<old-project>/     ← old data
```

Why archive instead of delete:
- Old experiments may need to be reproduced or referenced.
- Code on `projects/` is **backed up** — `_archive/` keeps the backup
  history.
- `scratch/` has no backup — once data is gone, it's gone. If you need
  the historical data, leave it in `_archive/` rather than deleting.

When to archive:
- Project hasn't been touched in 3+ months and no foreseeable use.
- Project superseded by a newer fork (e.g. `DiffSynth/` → `DiffSynth-valka/`).
- Branch / experiment that didn't work, kept for reference only.

Move command (atomic, fast on Lustre — no copy):
```bash
mv $WORK/$USER/<name>  $WORK/$USER/_archive/
mv $SCRATCH/<name>   $SCRATCH/_archive/
```

When to truly delete (rare):
- Files that contain credentials / leaked tokens — purge, don't archive.
- Project size > 100 GB that you're sure will never be needed.
- Otherwise: archive and forget. Disk is cheap, regenerating data is not.

---

## 5. `$HOME` — only two symlinks

`$HOME` (`/leonardo/home/userexternal/$USER/`) has an  50 GB quota — keep it nearly
empty. The only project-related entries here are **two symlinks** that
short-cut into the Lustre volumes:

```
$HOME/projects → $WORK/$USER/
$HOME/scratch  → $SCRATCH/
```

So `$WORK/$USER/<name>/` resolves to the canonical project code dir, and
`~/scratch/<name>/` resolves to its data dir. **Maintain both symlinks.**

If they're missing, recreate them:

```bash
ln -s $WORK/$USER  ~/projects
ln -s $SCRATCH   ~/scratch
```

Convenience symlinks to specific projects (e.g.
`$HOME/DiffSynth-valka → $WORK/$USER/DiffSynth-valka`) are fine
to keep, but they are personal — don't bake them into shared code.

---

## 6. Special directories in `scratch/` (NOT per-project)

These are scratch dirs that don't follow the per-project mirror pattern.
They're shared utility / cache dirs that don't correspond to any one
project's code:

| Path | Purpose |
|---|---|
| `$SCRATCH/envs/` | Conda environments (one dir per env) |
| `$SCRATCH/wan_models/` | Pre-staged Wan model weights (~159 GB) — read by `MODELSCOPE_CACHE` |
| `$SCRATCH/slurm_logs/` | All SLURM stdout/stderr from your jobs (filename format: see `slurm.md` § 7; if writes here hang see `lustre-resilience.md`) |
| `$SCRATCH/cache/` | Persistent dev caches (pip, uv, ccache) — see `~/.bashrc` |
| `$SCRATCH/singularity/` | Singularity images (`*.sif`) and build cache (`.singularity/cache/` set via `$SINGULARITY_CACHEDIR` in `~/.bashrc`) |

---

## 7. Shared (cross-user) resources

These live under a non-user-specific scratch path so the whole team can
read them. Do not duplicate them in your own scratch:

| Path | What | Who writes |
|---|---|---|
| `$SCRATCH/shared/wheels/` | Pre-compiled Python wheels (e.g. flash-attn, deepspeed) | maintainer; readers `pip install /path/to/wheel.whl` |
| `$SCRATCH/livedealer/` | Shared dataset for live-dealer training | data team |

When building a new wheel that's expensive to compile (>5 min on A100),
copy the resulting `*.whl` into `$SCRATCH/shared/wheels/`
so teammates can `pip install` instead of rebuilding.

---

## 8. Portability rule: never hardcode paths

Code, configs, and SLURM scripts must be runnable by **any team member**
without path edits. This means:

### In shell / Python

| ❌ Don't | ✅ Do |
|---|---|
| `$WORK/xliu0006/X` | `$WORK/$USER/X` |
| `/leonardo/home/userexternal/xliu0006/scripts/...` | `$HOME/scripts/...` |
| `/leonardo_work/AIFAC_F02_378/xliu0006/<project>` | `$WORK/$USER/<project>` |

### In SLURM SBATCH directives

| ❌ Don't | ✅ Do |
|---|---|
| `--output=$SCRATCH/xliu0006/slurm_logs/%x_%j.out` | `--output=$SCRATCH/%u/slurm_logs/%x_%j.out` |
| `--chdir=$WORK/xliu0006/...` | `--chdir=$WORK/%u/...` |

`%u` is SLURM's expansion for `$USER`.

### In YAML / config files

```yaml
# ❌ Don't
data_path: $SCRATCH/xliu0006/<project>/data/
output_dir: /leonardo/home/userexternal/xliu0006/<project>/outputs/

# ✅ Do
data_path: ${oc.env:SCRATCH}/<project>/data/
output_dir: ${oc.env:WORK}/${oc.env:USER}/<project>/outputs/

# OR — if your YAML loader supports env var interpolation:
data_path: ${SCRATCH}/<project>/data/   # SCRATCH from shell
```

(The Hydra / OmegaConf syntax `${oc.env:VAR}` is the most common.)

### When to break this rule

The **only** legitimate hardcode is when the path is shared across users
and intentionally fixed:
- `$WORK/shared/wheels/` (team-wide pre-built wheels)
- `$WORK/shared/datasets/` (team-wide dataset root)
- `$WORK/shared/singularity/` (team-released SIF images)

In that case, hardcoding `AIFAC_F02_378/shared/...` is correct because it's a
team resource, not a personal one.

---

## 9. Translating existing code

If you find code with a hardcoded username (any team member's):

1. Replace `<username>` literal with `${USER}` (shell) / `os.environ["USER"]` (Python) /
   `${oc.env:USER}` (YAML).
2. Replace `/leonardo/home/userexternal/<username>/` with `$HOME` (shell) /
   `os.path.expanduser("~")` (Python).
3. Replace `/leonardo_work/AIFAC_F02_378/<username>/` with `$WORK/$USER/` for clarity.
4. Run a quick `grep -rnE "/leonardo/home/userexternal/[a-z]+[0-9]+" .` after the
   swap to confirm nothing is left behind.

Common offenders: SLURM scripts copied from past jobs, YAML configs
copied between projects, Jupyter notebooks where someone Ctrl+C'd a path.

---

## 10. Summary: where does `<thing>` live?

| Thing | Path |
|---|---|
| Source code (git repos) | `$WORK/$USER/<name>/` |
| Per-project data, ckpts | `$SCRATCH/<name>/` |
| Old / superseded code | `$WORK/$USER/_archive/<name>/` |
| Old / superseded data | `$SCRATCH/_archive/<name>/` |
| Conda envs | `$SCRATCH/envs/<envname>/` |
| Wan model weights | `$SCRATCH/wan_models/Wan-AI/...` |
| SLURM logs | `$SCRATCH/slurm_logs/` |
| Pip / uv cache | `$SCRATCH/cache/` |
| Singularity `*.sif` + build cache | `$SCRATCH/singularity/` |
| Pre-compiled wheels (team) | `$SCRATCH/shared/wheels/` |
| Personal config / dotfiles | `$HOME` |
| Personal helper scripts | `$HOME/scripts/` |
| Two convenience symlinks | `$HOME/projects`, `$HOME/scratch` |

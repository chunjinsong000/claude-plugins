# BSC MareNostrum5 — Storage Layout Convention

How we organize code, data, environments, and logs across `projects/`,
`scratch/`, and `$HOME`. Read this before placing any new file. Follow
this convention so that **other team members can clone your code and run
it with no path edits**.

**Related**: `slurm.md` (SLURM directive syntax for paths) ·
`gpfs-hang-resilience.md` (when GPFS IO hangs).

---

## 1. Account: `ehpc1003`

All shared storage lives under the `ehpc1003` allocation:

| Path | Quota | Backup | Lifecycle |
|---|---|---|---|
| `/gpfs/projects/ehpc1003/` | 1 TB shared | yes | see note below |
| `/gpfs/scratch/ehpc1003/`  | 50 TB shared | **NO BACKUP** | see note below |
| `/gpfs/home/<user>/` | 80 GB | yes | personal, permanent |

`ehpc1003` here is the **storage account** — fixed for the allocation's
lifetime even when the job-submission account rotates.

> Note: quotas and the validity date above were documented for the previous
> allocation (`ehpc679`, valid until 2027-03-25). Verify the `ehpc1003`
> figures with `bsc_quota` / `bsc_acct` and update this table.

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
/gpfs/projects/ehpc1003/${USER}/<name>/    ← code (git repo)
/gpfs/scratch/ehpc1003/${USER}/<name>/     ← datasets, ckpts, outputs
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
/gpfs/projects/ehpc1003/${USER}/_archive/<old-project>/    ← old code
/gpfs/scratch/ehpc1003/${USER}/_archive/<old-project>/     ← old data
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

Move command (atomic, fast on GPFS — no copy):
```bash
mv /gpfs/projects/ehpc1003/${USER}/<name>  /gpfs/projects/ehpc1003/${USER}/_archive/
mv /gpfs/scratch/ehpc1003/${USER}/<name>   /gpfs/scratch/ehpc1003/${USER}/_archive/
```

When to truly delete (rare):
- Files that contain credentials / leaked tokens — purge, don't archive.
- Project size > 100 GB that you're sure will never be needed.
- Otherwise: archive and forget. Disk is cheap, regenerating data is not.

---

## 5. `$HOME` — only two symlinks

`$HOME` (`/gpfs/home/<user>/`) has an 80 GB quota — keep it nearly
empty. The only project-related entries here are **two symlinks** that
short-cut into the GPFS volumes:

```
$HOME/projects → /gpfs/projects/ehpc1003/${USER}/
$HOME/scratch  → /gpfs/scratch/ehpc1003/${USER}/
```

So `~/projects/<name>/` resolves to the canonical project code dir, and
`~/scratch/<name>/` resolves to its data dir. **Maintain both symlinks.**

If they're missing, recreate them:

```bash
ln -s /gpfs/projects/ehpc1003/${USER}  ~/projects
ln -s /gpfs/scratch/ehpc1003/${USER}   ~/scratch
```

Convenience symlinks to specific projects (e.g.
`$HOME/DiffSynth-valka → /gpfs/projects/.../DiffSynth-valka`) are fine
to keep, but they are personal — don't bake them into shared code.

---

## 6. Special directories in `scratch/` (NOT per-project)

These are scratch dirs that don't follow the per-project mirror pattern.
They're shared utility / cache dirs that don't correspond to any one
project's code:

| Path | Purpose |
|---|---|
| `/gpfs/scratch/ehpc1003/${USER}/envs/` | Conda environments (one dir per env) |
| `/gpfs/scratch/ehpc1003/${USER}/wan_models/` | Pre-staged Wan model weights (~159 GB) — read by `MODELSCOPE_CACHE` |
| `/gpfs/scratch/ehpc1003/${USER}/slurm_logs/` | All SLURM stdout/stderr from your jobs (filename format: see `slurm.md` § 7; if writes here hang see `gpfs-hang-resilience.md`) |
| `/gpfs/scratch/ehpc1003/${USER}/cache/` | Persistent dev caches (pip, uv, ccache) — see `~/.bashrc` |
| `/gpfs/scratch/ehpc1003/${USER}/singularity/` | Singularity images (`*.sif`) and build cache (`.singularity/cache/` set via `$SINGULARITY_CACHEDIR` in `~/.bashrc`) |

---

## 7. Shared (cross-user) resources

These live under a non-user-specific scratch path so the whole team can
read them. Do not duplicate them in your own scratch:

| Path | What | Who writes |
|---|---|---|
| `/gpfs/scratch/ehpc1003/shared/wheels/` | Pre-compiled Python wheels (e.g. flash-attn, deepspeed) | maintainer; readers `pip install /path/to/wheel.whl` |
| `/gpfs/scratch/ehpc1003/livedealer/` | Shared dataset for live-dealer training | data team |

When building a new wheel that's expensive to compile (>5 min on H100),
copy the resulting `*.whl` into `/gpfs/scratch/ehpc1003/shared/wheels/`
so teammates can `pip install` instead of rebuilding.

---

## 8. Portability rule: never hardcode paths

Code, configs, and SLURM scripts must be runnable by **any team member**
without path edits. This means:

### In shell / Python

| ❌ Don't | ✅ Do |
|---|---|
| `/gpfs/projects/ehpc1003/vlk370419/X` | `/gpfs/projects/ehpc1003/${USER}/X` |
| `/home/vlk/vlk370419/scripts/...` | `$HOME/scripts/...` |
| `~/projects/DiffSynth-valka` | `${HOME}/projects/DiffSynth-valka` (or `$HOME/projects/...`) |

### In SLURM SBATCH directives

| ❌ Don't | ✅ Do |
|---|---|
| `--output=/gpfs/scratch/ehpc1003/vlk370419/slurm_logs/%x_%j.out` | `--output=/gpfs/scratch/ehpc1003/%u/slurm_logs/%x_%j.out` |
| `--chdir=/gpfs/projects/ehpc1003/vlk370419/...` | `--chdir=/gpfs/projects/ehpc1003/%u/...` |

`%u` is SLURM's expansion for `$USER`.

### In YAML / config files

```yaml
# ❌ Don't
data_path: /gpfs/scratch/ehpc1003/vlk370419/DiffSynth-valka/data/
output_dir: /home/vlk/vlk370419/DiffSynth-valka/outputs/

# ✅ Do
data_path: ${oc.env:HOME}/scratch/DiffSynth-valka/data/
output_dir: ${oc.env:HOME}/projects/DiffSynth-valka/outputs/

# OR — if your YAML loader supports env var interpolation:
data_path: ${SCRATCH_DIR}/DiffSynth-valka/data/   # SCRATCH_DIR set in shell
```

(The Hydra / OmegaConf syntax `${oc.env:VAR}` is the most common.)

### When to break this rule

The **only** legitimate hardcode is when the path is shared across users
and intentionally fixed:
- `/gpfs/scratch/ehpc1003/shared/wheels/` (team-wide)
- `/gpfs/scratch/ehpc1003/livedealer/` (team-wide dataset root)

In that case, hardcoding `ehpc1003/shared/...` is correct because it's a
team resource, not a personal one.

---

## 9. Translating existing code

If you find code with hardcoded `vlk370419` (or another username):

1. Replace with `${USER}` (shell) / `os.environ["USER"]` (Python) /
   `${oc.env:USER}` (YAML).
2. Replace `/home/vlk/vlk370419/` with `$HOME` (shell) / `os.path.expanduser("~")` (Python).
3. Replace `~/projects/<name>/` with `$HOME/projects/<name>/` for explicit shell expansion.
4. Run a quick `grep -rn "vlk370419\|/home/vlk/" .` after the swap to
   confirm nothing is left behind.

Common offenders: SLURM scripts copied from past jobs, YAML configs
copied between projects, Jupyter notebooks where someone Ctrl+C'd a path.

---

## 10. Summary: where does `<thing>` live?

| Thing | Path |
|---|---|
| Source code (git repos) | `/gpfs/projects/ehpc1003/${USER}/<name>/` |
| Per-project data, ckpts | `/gpfs/scratch/ehpc1003/${USER}/<name>/` |
| Old / superseded code | `/gpfs/projects/ehpc1003/${USER}/_archive/<name>/` |
| Old / superseded data | `/gpfs/scratch/ehpc1003/${USER}/_archive/<name>/` |
| Conda envs | `/gpfs/scratch/ehpc1003/${USER}/envs/<envname>/` |
| Wan model weights | `/gpfs/scratch/ehpc1003/${USER}/wan_models/Wan-AI/...` |
| SLURM logs | `/gpfs/scratch/ehpc1003/${USER}/slurm_logs/` |
| Pip / uv cache | `/gpfs/scratch/ehpc1003/${USER}/cache/` |
| Singularity `*.sif` + build cache | `/gpfs/scratch/ehpc1003/${USER}/singularity/` |
| Pre-compiled wheels (team) | `/gpfs/scratch/ehpc1003/shared/wheels/` |
| Personal config / dotfiles | `$HOME` |
| Personal helper scripts | `$HOME/scripts/` |
| Two convenience symlinks | `$HOME/projects`, `$HOME/scratch` |

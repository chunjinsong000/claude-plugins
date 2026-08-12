# claude-plugins

Personal Claude Code plugin marketplace. One repo, installed on every machine
I run Claude Code on (MacBook, Lambda GPU box, BSC clusters, LEONARDO).

## Layout philosophy — split by what varies

| What | Where | Why |
|---|---|---|
| Reusable skills (cluster workflows, habits) | **This repo** → `/plugin install` | One command per new box, updates via git |
| Project skills (launch commands, dataset prep, eval harness) | `.claude/skills/` committed **in the research repo** | Travels with the code you already clone to the GPU box |
| Machine facts (GPU count, conda env, scratch paths, scheduler) | `~/.claude/CLAUDE.md` **on that box**, uncommitted | Genuinely machine-specific; must not live in a shared skill (see `templates/`) |

Plugins are scoped per environment — a box only installs what applies to it:

| Plugin | Install on | Contents |
|---|---|---|
| `hpc-leonardo` | LEONARDO login nodes (and the Mac, if driving LEONARDO from there) | clone-to-projects, create-slurm-job-script, monitor-jobs |
| `hpc-mn5` *(future)* | BSC MareNostrum 5 | MN5 equivalents once the workflows stabilize there |
| `lambda` *(future)* | Lambda GPU box | bare-metal GPU workflows (no scheduler) |

## Setup on a new machine

```bash
# 1. Get the repo onto the box
git clone <REMOTE_URL> ~/claude-plugins
#    BSC note: if the login node has no outbound internet, rsync instead:
#    rsync -a ~/claude-plugins/ <bsc-host>:claude-plugins/

# 2. Register the marketplace and install what this box needs
claude plugin marketplace add ~/claude-plugins
claude plugin install hpc-leonardo@chunjin --scope user

# 3. Create the machine-facts file (pick the matching template)
cp ~/claude-plugins/templates/CLAUDE.md.<machine> ~/.claude/CLAUDE.md   # then edit
```

## Updating

- Edit skills here → commit → push.
- On other boxes: `git pull` in `~/claude-plugins` (or re-rsync), then
  `claude plugin marketplace update chunjin` (local-path marketplaces
  don't auto-refresh the way git-sourced ones do).

## Authoring conventions

- Each skill: `<plugin>/skills/<name>/SKILL.md` (+ optional `assets/`).
- Shared background docs live once per plugin in `<plugin>/references/`;
  SKILL.md files point at them via `../../references/<file>.md`.
- Cluster facts (QoS names, storage layout, hardware) belong in
  `references/`, not in SKILL.md — SKILL.md encodes procedure and judgment
  calls only.
- Machine-specific values (account codes, personal paths) belong in that
  machine's `~/.claude/CLAUDE.md`, never in this repo.

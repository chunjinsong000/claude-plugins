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
| `hpc-mn5` | BSC MareNostrum 5 login nodes | same three skills, MN5-flavored (GPFS, acc_* QoS, proxy-gated internet) |
| `lambda-gpu` | Lambda GPU boxes | single-node SLURM setup, wait analysis, lambda↔lambda / lambda→LEONARDO transfers |
| `diffsynth` | wherever Wan2.2-S2V (DiffSynth) training/eval runs | self-forcing, continue-training, eval-training, eval-self-forcing, benchmark-streaming, compare/concat eval results |
| `ltx` | boxes running LTX-2 experiments | LTX-2.3 LipDub dubbing |

Provenance: both hpc plugins derive from the team onboarding repos —
`valka-ai/bsc-onboarding` (imported @379d04c) and `valka-ai/LEONARDO-onboarding`
(content-identical to hpc-leonardo apart from two path fixes made here).
If the team repos' `scripts/skills/` change, re-sync manually and note the commit.
`lambda-gpu`, `diffsynth`, and `ltx` were imported from the now-retired
`chunjinsong000/claude_skills` repo (full history merged via `git subtree`,
final upstream commit 9512b62); that repo is archived — edit skills here only.

## Setup on a new machine

```bash
# 1. Get the repo onto the box
git clone https://github.com/chunjinsong000/claude-plugins.git ~/claude-plugins
#    BSC note: if the login node has no outbound internet, rsync instead:
#    rsync -a ~/claude-plugins/ <bsc-host>:claude-plugins/

# 2. Register the marketplace and install what this box needs
claude plugin marketplace add ~/claude-plugins
claude plugin install hpc-leonardo@chunjin --scope user

# 3. Create the machine-facts file (pick the matching template)
cp ~/claude-plugins/templates/CLAUDE.md.<machine> ~/.claude/CLAUDE.md   # then edit
```

## Updating

Plugin versions resolve from the git commit SHA (no `version` pins in the
manifests — do not add them, or pushes stop propagating until a manual bump).

- **Edit → commit → push** from the Mac.
- **Boxes with internet** that added the marketplace as
  `chunjinsong000/claude-plugins`: nothing to do — auto-refreshes at
  session start. Force mid-session with `/plugin marketplace update chunjin`.
- **BSC (no outbound internet)**: re-rsync `~/claude-plugins`, then
  `claude plugin marketplace update chunjin`.
- **Mac (local-path marketplace = the working copy)**: after committing,
  run `claude plugin update hpc-leonardo@chunjin` (or `/plugin marketplace
  update chunjin`) to refresh the installed cache.

## Authoring conventions

- Each skill: `<plugin>/skills/<name>/SKILL.md` (+ optional `assets/`).
- Shared background docs live once per plugin in `<plugin>/references/`;
  SKILL.md files point at them via `../../references/<file>.md`.
- Cluster facts (QoS names, storage layout, hardware) belong in
  `references/`, not in SKILL.md — SKILL.md encodes procedure and judgment
  calls only.
- Machine-specific values (account codes, personal paths) belong in that
  machine's `~/.claude/CLAUDE.md`, never in this repo.

---
name: transfer-lambda-to-leonardo
description: Use when the user wants to copy/transfer/upload/sync data or files FROM a Lambda cloud instance (or any non-LEONARDO Linux box that holds the data) TO LEONARDO ("copy X to leonardo", "transfer this dataset to $WORK", "upload these files to leonardo", "把数据传到 leonardo"). Covers the auth problem (no smallstep cert on Lambda) via SSH agent forwarding, dest-dir creation, and resumable rsync to the datamover.
---

# Transfer data: Lambda → LEONARDO

## Overview

The data lives on a **Lambda instance** (a plain Linux GPU box, e.g. user
`ubuntu`, a public IP like `209.20.157.219`). The destination is **LEONARDO**
under `$WORK` = `/leonardo_work/AIFAC_F02_378/<cineca-user>/...`.

Lambda is the **initiator**: it pushes to the LEONARDO **datamover**
`data.leonardo.cineca.it` (DTN) via `rsync`/`scp`/`sftp`. The login-node /
DTN-direction rules in `docs/data-transfer.md` still apply — push to the DTN,
use absolute paths, no env-var expansion on the remote side.

**The one hard problem:** a fresh Lambda box has **no LEONARDO auth** — no
`step` CLI, no smallstep cert, no ssh-agent. LEONARDO auth is a short-lived
(~12 h) smallstep **ECDSA certificate**, not a raw keypair, so a normal SSH key
won't get in. The fix is **SSH agent forwarding** from the user's Mac (where
the cert already lives in their agent after `step ssh login`). **Never copy a
private key onto Lambda.**

## Steps

1. **Inspect the source** — size and file count drive the strategy:
   ```bash
   ls -la <source_dir>/ | head
   find <source_dir>/ -maxdepth 1 -type f | wc -l
   du -sh <source_dir>/
   ```
   - **Few large files** → single-stream rsync to `data-leo` is fine.
   - **Many small files** (10k+) → transfer is **latency-bound**, not
     bandwidth-bound. Single stream is slow; consider the parallel
     `dmover1-4` split (Step 7).

2. **Check for an already-forwarded agent.** The Claude/agent shell runs
   separately from the user's interactive session, so it won't inherit
   `SSH_AUTH_SOCK`. Look for the forwarded socket on disk and test it:
   ```bash
   find /tmp -maxdepth 2 -type s -name 'agent.*' -user "$(whoami)"
   SSH_AUTH_SOCK=<socket> ssh-add -l        # want one ECDSA-CERT line
   ```
   If none is present, do Step 3.

3. **Have the user set up agent forwarding** (from their **Mac**):
   ```bash
   step ssh login '<their-email>' --provisioner cineca-hpc   # mint cert if stale
   ssh-add -l                                                # expect one ECDSA-CERT
   ssh -A ubuntu@<lambda-ip>                                 # -A forwards the agent
   echo $SSH_AUTH_SOCK                                       # paste this path back
   ```
   Then `export SSH_AUTH_SOCK=<that path>` in every transfer command below.
   The socket is owned by the same Linux user, so this shell can use it.

4. **Verify auth** to the datamover before moving data (uses the CINECA
   username — current team examples: `wxu00000`, `xliu0006`):
   ```bash
   export SSH_AUTH_SOCK=<socket>
   sftp -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -b /dev/null <cineca-user>@data.leonardo.cineca.it
   ```
   Clean exit = auth OK. `Permission denied (publickey,...)` = cert not in the
   forwarded agent → back to Step 3.

5. **Create the destination directory.** The remote rsync on the DTN is old
   (**3.1.3 — no `--mkpath`**), so the dest dir must exist first. The datamover
   has **no shell** and **no `mkdir -p`**, but `sftp mkdir` works for one level,
   or use a login node for nested paths:
   ```bash
   # one level, on the DTN:
   printf 'mkdir <abs_dest_dir>\n' | sftp -o BatchMode=yes \
     -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null <cineca-user>@data.leonardo.cineca.it
   # nested path: use a login node (has a real shell):
   ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     <cineca-user>@login.leonardo.cineca.it 'mkdir -p <abs_dest_dir>'
   ```

6. **Run the transfer** (resumable, in background, logged). Note the **trailing
   slash on the source** = copy *contents* into the dest dir. Do **not** pass
   `--mkpath` (remote is 3.1.3):
   ```bash
   export SSH_AUTH_SOCK=<socket>
   LOG=/tmp/<name>_rsync.log
   nohup rsync -ahP --stats --partial --partial-dir=.rsync-partial \
     -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
     <source_dir>/ \
     <cineca-user>@data.leonardo.cineca.it:<abs_dest_dir>/ \
     > "$LOG" 2>&1 &
   ```
   `-z` is intentionally omitted — over a fast link compression hurts.

7. **(Many-small-files speedup)** Split across the 4 named datamovers for ~4×
   throughput. Partition the file list into 4 buckets and run one rsync per
   host (`dmover1`–`dmover4`), then `wait`. Each uses absolute paths; auth is
   the same forwarded cert.

8. **Monitor** to completion:
   ```bash
   grep -oE 'to-chk=[0-9]+/[0-9]+' "$LOG" | tail -1      # local progress
   ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     <cineca-user>@login.leonardo.cineca.it \
     'cd <abs_dest_dir> && ls -1 | grep -c . && du -sh .'   # remote count + size
   ```
   A transient `du: cannot access ...jsonl.pXXXX` warning is harmless — that's
   rsync's in-flight temp file being renamed mid-listing.

9. **Verify done**: rsync exits 0, `--stats` block in the log shows
   sent/total; remote file count == source file count; remote `du -sh` matches
   source size (modulo filesystem block rounding).

## Resume after a drop

`--partial` + the incremental file list make this safe: **re-run the exact
same Step 6 command** and already-copied files are skipped. Auth happens once
at connection setup, so an established rsync survives the user closing their
`ssh -A` session — but keep it open if you might need to re-auth on a resume.

## Common Mistakes — DO NOT MAKE THESE

| Mistake | Right way |
|---|---|
| Copying the user's private key onto Lambda | Never. Use `ssh -A` agent forwarding — cert stays on the Mac |
| Assuming a normal SSH key authenticates to LEONARDO | It's a smallstep **ECDSA-CERT** from `step ssh login`; a raw key is rejected |
| Passing `--mkpath` | Remote rsync is 3.1.3 — unknown option, fails. Create the dir first (Step 5) |
| `mkdir -p` on the datamover | DTN has no shell. Use `sftp mkdir` (one level) or a **login node** for nested paths |
| `~`/`$WORK`/`$HOME` or globs in the remote path | DTN does **no** expansion — absolute paths only |
| `ssh data.leonardo.cineca.it` for a shell | DTN is transfer-only ("no interactive login access"). Use `sftp`/`scp`/`rsync`; for a shell use `login.leonardo.cineca.it` |
| Expecting this shell to inherit `SSH_AUTH_SOCK` | It runs separately from the user's session — find the socket on disk and `export` it |
| Single stream for 95k tiny files | Latency-bound; split across `dmover1-4` (Step 7) |
| Dropping the trailing slash on the source | `src/` copies *contents*; `src` copies the dir itself into dest — different result |

## What this skill does NOT do

- Run the transfer from inside a SLURM job — hostbased/forwarded auth is
  **disabled inside batch jobs** (`docs/data-transfer.md` §4). This skill is
  for Lambda-initiated (interactive) transfers only.
- LEONARDO → Lambda (the reverse) — initiate from LEONARDO via
  `sbatch -p lrd_all_serial`; see `docs/data-transfer.md`.
- Install `step` or mint certs — that's the user's Mac. This skill borrows the
  forwarded agent.

## Background References

- `references/storage-layout.md` — `$WORK` / `$SCRATCH` paths, where data vs
  code lives on LEONARDO
- `docs/data-transfer.md` in the `valka-ai/LEONARDO-onboarding` repo — the
  three transfer channels, DTN rules, parallel `dmover1-4` split, and the
  SLURM-batch auth caveat

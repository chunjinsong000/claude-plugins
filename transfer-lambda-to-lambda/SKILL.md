---
name: transfer-lambda-to-lambda
description: Use when the user wants to copy/transfer/move/sync data or files between two Lambda cloud instances (or any two plain Linux boxes reached with the same SSH keypair) — "copy X from this Lambda box to that one", "transfer the dataset to the new instance", "sync these files between the two Lambda machines", "把数据从一台 Lambda 传到另一台". Covers reaching the target from the source without copying a private key (SSH agent forwarding), dest-dir creation, and resumable rsync. Current example pair: source `ubuntu@150-136-211-158`, target `ubuntu@192-222-52-139`.
---

# Transfer data: Lambda → Lambda

## Overview

Both machines are **Lambda instances** — plain Linux GPU boxes, user `ubuntu`,
public IPs, authenticated with the **same Lambda SSH keypair** the user
configured at launch. Current example pair:

| Role | Host |
|---|---|
| **source** (holds the data, initiator) | `ubuntu@150.136.211.158` |
| **target** (receives the data) | `ubuntu@192.222.52.139` |

(The `150-136-211-158` form in the request is just the dashed IP — use dots in
`ssh`/`rsync`: `150.136.211.158`, `192.222.52.139`.)

The **source pushes** to the **target** via `rsync` over SSH. Unlike LEONARDO,
there is **no smallstep cert, no datamover, no DTN rules** — both ends run a
modern rsync and a real shell, so `--mkpath` and `mkdir -p` both work.

**The one real problem:** the source box has the *data* but not the *private
key*, so out of the box it **cannot SSH into the target** — only the public key
is in the target's `authorized_keys`. The fix is **SSH agent forwarding**: the
user SSHes into the source with `ssh -A`, and the forwarded agent (holding the
Lambda key on their Mac) authenticates the source→target hop. **Never copy the
private key onto either Lambda box.**

## Steps

1. **Inspect the source** — size and file count drive the strategy (run on the
   source box):
   ```bash
   ls -la <source_dir>/ | head
   find <source_dir>/ -type f | wc -l
   du -sh <source_dir>/
   ```
   - **Few large files** → single-stream rsync is fine.
   - **Many small files** (10k+) → transfer is **latency-bound**; run several
     rsyncs in parallel over disjoint subtrees (Step 6).

2. **Check for an already-forwarded agent.** The Claude/agent shell runs
   separately from the user's interactive session, so it won't inherit
   `SSH_AUTH_SOCK`. Look for the forwarded socket on disk and test it:
   ```bash
   find /tmp -maxdepth 2 -type s -name 'agent.*' -user "$(whoami)"
   SSH_AUTH_SOCK=<socket> ssh-add -l        # want the Lambda key listed
   ```
   If none is present, do Step 3.

3. **Have the user set up agent forwarding** (from their **Mac**, where the
   Lambda private key lives):
   ```bash
   ssh-add ~/.ssh/<lambda-key>        # if the key isn't in the agent yet
   ssh-add -l                         # confirm the Lambda key is listed
   ssh -A ubuntu@150.136.211.158      # -A forwards the agent to the source box
   echo $SSH_AUTH_SOCK                # paste this path back
   ```
   Then `export SSH_AUTH_SOCK=<that path>` in every transfer command below. The
   socket is owned by the same `ubuntu` user, so this shell can use it.

4. **Verify the source can reach the target** before moving data (run on the
   source box). The forwarded agent authenticates the hop:
   ```bash
   export SSH_AUTH_SOCK=<socket>
   ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
     ubuntu@192.222.52.139 'hostname && df -h <target_parent_dir>'
   ```
   Clean output = auth OK and there's room. `Permission denied (publickey)` =
   the key isn't in the forwarded agent → back to Step 3. `ssh -A` must have
   been used to reach the source, or nothing is forwarded.

5. **Create the destination directory** on the target (real shell, so this is
   trivial — or skip it and let rsync's `--mkpath` do it in Step 6):
   ```bash
   ssh -o StrictHostKeyChecking=accept-new ubuntu@192.222.52.139 \
     'mkdir -p <abs_target_dir>'
   ```

6. **Run the transfer** (resumable, backgrounded, logged) from the **source
   box**. Trailing slash on the source = copy *contents* into the dest:
   ```bash
   export SSH_AUTH_SOCK=<socket>
   LOG=/tmp/<name>_rsync.log
   nohup rsync -ahP --stats --mkpath --partial --partial-dir=.rsync-partial \
     -e 'ssh -o StrictHostKeyChecking=accept-new' \
     <source_dir>/ \
     ubuntu@192.222.52.139:<abs_target_dir>/ \
     > "$LOG" 2>&1 &
   ```
   - `-z` is intentionally omitted — over a fast link compression hurts. Add it
     only if the two boxes are in different regions on a slow link.
   - Both ends run modern rsync, so `--mkpath` is safe (unlike the LEONARDO
     3.1.3 datamover).

7. **(Many-small-files speedup)** Split the work across a few parallel rsyncs
   over disjoint top-level subdirs (or a partitioned file list via
   `--files-from`), then `wait`. 3–4 streams is usually enough to saturate the
   link; more just thrash. Each stream uses the same forwarded agent.

8. **Monitor** to completion:
   ```bash
   grep -oE 'to-chk=[0-9]+/[0-9]+' "$LOG" | tail -1      # source-side progress
   ssh ubuntu@192.222.52.139 \
     'cd <abs_target_dir> && find . -type f | wc -l && du -sh .'   # target count + size
   ```

9. **Verify done**: rsync exits 0, the `--stats` block in the log shows
   sent/total, target file count == source file count, and target `du -sh`
   matches source size (modulo filesystem block rounding).

## Resume after a drop

`--partial` + the incremental file list make this safe: **re-run the exact same
Step 6 command** and already-copied files are skipped. Auth happens once at
connection setup, so an established rsync survives the user closing their
`ssh -A` session — but keep it open if you might need to re-auth on a resume or
add parallel streams.

## Alternative: pull from the target instead of push from the source

If it's easier to drive from the target box, SSH into the **target** with
`ssh -A` and **pull** — swap source/dest and flip the direction. Auth is still
the forwarded agent; everything else is identical. Push-from-source is the
default because the initiator already has the data local for the inspection in
Step 1.

## Common Mistakes — DO NOT MAKE THESE

| Mistake | Right way |
|---|---|
| Copying the user's private key onto a Lambda box to bridge the two | Never. Use `ssh -A` agent forwarding — key stays on the Mac |
| `ssh`-ing into the source **without** `-A`, then hitting `Permission denied` on the target hop | The source has no private key; `-A` forwards the agent that does |
| Expecting this shell to inherit `SSH_AUTH_SOCK` | It runs separately from the user's session — find the socket on disk and `export` it |
| Using the dashed IP form (`150-136-211-158`) in `ssh`/`rsync` | That's just Lambda's hostname style; use dotted IPs `150.136.211.158` / `192.222.52.139` |
| Dropping the trailing slash on the source | `src/` copies *contents*; `src` copies the dir itself into dest — different result |
| Adding `-z` on a fast same-region link | Compression is CPU-bound and slows a fast link; omit unless cross-region/slow |
| Single stream for tens of thousands of tiny files | Latency-bound; run 3–4 parallel rsyncs over disjoint subtrees (Step 7) |
| First-connection `StrictHostKeyChecking` prompt hanging the background job | Use `-o StrictHostKeyChecking=accept-new` so the unattended run doesn't block |

## What this skill does NOT do

- Transfer to/from **LEONARDO** — that has a smallstep cert + datamover and its
  own rules; use `transfer-lambda-to-leonardo` instead.
- Provision or key the Lambda boxes — the SSH keypair and `authorized_keys` are
  set up at instance launch. This skill borrows the forwarded agent.
- Run inside a batch scheduler — this is an interactive, agent-forwarded push.

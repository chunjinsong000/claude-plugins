---
name: setup-slurm
description: Install and configure a single-node SLURM cluster on an Ubuntu machine, with Gmail-relayed email notifications for job BEGIN/END/FAIL. Detects CPUs, memory, and NVIDIA GPUs automatically and exposes GPUs as gres. Use when the user says "set up slurm on this machine", "install slurm", "configure slurm email notifications", "make slurm send emails on job start/fail/complete", or "set up a slurm node".
---

# Set up SLURM on a single machine (with email notifications)

Installs `slurm-wlm` + `munge`, writes a single-node `slurm.conf`, wires up Gmail
SMTP so SLURM can email you on job **BEGIN / END / FAIL**, starts the services, and
runs an end-to-end test (a real `sbatch` job that fires the notification emails).

Tested on **Ubuntu 22.04 / slurm-wlm 21.08 / cgroup v2 / NVIDIA A10**. Requires
passwordless `sudo` (or run as root).

## How to run it

You need a Gmail (or Google Workspace) address and a **Google App Password**
(16 characters — create at https://myaccount.google.com/apppasswords; normal account
passwords will not work for SMTP).

```bash
sudo bash .claude/skills/setup-slurm/setup_slurm.sh \
  --email you@example.com \
  --app-password "abcd efgh ijkl mnop"     # spaces are fine, keep the quotes
```

Common options (see `--help` for all):

| flag | default | meaning |
|---|---|---|
| `--email` | *(required)* | address that both sends and receives notifications |
| `--app-password` | *(required)* | Google App Password |
| `--smtp-host` / `--smtp-port` | `smtp.gmail.com` / `587` | relay host/port (change for a non-Gmail relay) |
| `--from` / `--smtp-user` | = `--email` | envelope-from / SMTP auth user |
| `--cluster` / `--partition` | `cluster` / `main` | names |
| `--no-test` | *(off)* | skip the test job + self-test email |

The script is **idempotent** — safe to re-run to change the email/config.

## Using it

Add to any `sbatch` script (or pass on the `srun`/`sbatch` command line):

```bash
#SBATCH --mail-type=BEGIN,END,FAIL     # or ALL, TIME_LIMIT_90, REQUEUE, ...
#SBATCH --mail-user=you@example.com
#SBATCH --gres=gpu:1                   # if the node has GPUs
```

## What it configures (and why)

- **Packages**: `slurm-wlm slurmd slurmctld munge msmtp msmtp-mta bsd-mailx`.
- **munge**: generates the key (if missing), enables the service, self-tests it.
- **Hardware**: parses `slurmd -C` for CPUs/sockets/cores/threads/memory (uses 99% of
  RealMemory as headroom so the node doesn't drop to `DOWN` on tiny accounting diffs).
- **GPUs**: any `/dev/nvidiaN` devices become `Gres=gpu:N` in `slurm.conf` plus a
  `gres.conf` mapping each to its device file; `GresTypes=gpu` is set.
- **Process tracking**: uses `proctrack/linuxproc` + `task/affinity` instead of
  cgroups — SLURM **21.08 does not reliably support cgroup v2**, which recent Ubuntu
  hosts use by default. Jobs and GPU allocation work; containment is slightly weaker
  than a cgroup setup.
- **Email**: `MailProg=/usr/bin/mail`. `mail` (bsd-mailx) pipes to
  `/usr/sbin/sendmail`, which `msmtp-mta` points at **msmtp**, which relays to Gmail.

## Gotchas baked into the script (learned the hard way)

1. **msmtp logfile path**: msmtp is **AppArmor-confined**. The profile
   (`/etc/apparmor.d/usr.bin.msmtp`) allows writing `/var/log/msmtp` but **not**
   `/var/log/msmtp.log`. Using the `.log` name makes msmtp fail with
   `cannot log ... Permission denied` even at mode 666. The config uses
   `logfile /var/log/msmtp`.
2. **msmtprc must be readable by the `slurm` user**: `slurmctld` runs `MailProg` as
   the `slurm` user, not root. So `/etc/msmtprc` is `chown root:slurm` / `chmod 640`.
   A root-only `600` file gives `no configuration file available` and no email.
3. **App Password, not account password**: Gmail SMTP rejects the normal password.
4. **The self-test runs as the `slurm` user** (`sudo -u slurm mail ...`) so it
   exercises the *actual* path SLURM uses, not root's.

## Verifying / troubleshooting

- `sinfo` — node should be `idle` (not `down`/`drain`). If drained:
  `sudo scontrol update NodeName=<host> State=RESUME`.
- `sudo tail -f /var/log/msmtp` — one `smtpstatus=250 ... exitcode=EX_OK` line per
  email SLURM sends. A successful job emits **2** (BEGIN + END); a failing job emits
  BEGIN + FAIL.
- `sudo tail /var/log/slurm/slurmctld.log` — controller log.
- Manual mail test on the real path:
  `echo hi | sudo -u slurm mail -s test you@example.com`
- Config files written: `/etc/slurm/slurm.conf`, `/etc/slurm/gres.conf`,
  `/etc/msmtprc`. Logs: `/var/log/slurm/`, `/var/log/msmtp`.

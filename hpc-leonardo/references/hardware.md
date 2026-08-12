# LEONARDO Booster — Hardware Reference

Background facts about the Booster (GPU) partition on LEONARDO. Read this
when you need the *why* behind a directive in the SLURM template.

**Related**: `slurm.md` (QoS, switches, CPU rule) · `storage-layout.md`
(paths) · `lustre-resilience.md` (Lustre quirks) · `docs/networking-nccl.md`
(canonical NCCL config and IB measurements — the single source of truth
for the data-plane settings).

---

## 1. Node Specification (Booster)

| Item | Value |
|---|---|
| Partition | `boost_usr_prod` (GPU); `lrd_all_serial` (file transfer, 4 h, no GPU budget) |
| GPU | NVIDIA A100 SXM4 (Ampere) |
| GPUs per node | **4** |
| Memory per GPU | **64 GB HBM2e** (~63.4 GB usable) |
| Compute capability | **8.0** (Ampere, sm_80) |
| SM count per GPU | 108 |
| NVLink | **NV4** (Gen3) — 200 GB/s aggregate per GPU pair |
| CPUs per node | **32 physical / 64 logical** (1× Intel Xeon Ice Lake 8358 @ 2.6 GHz) |
| NUMA | **Single socket — all 4 GPUs on NUMA node 0** (no cross-NUMA penalty) |
| RAM per node | **512 GiB DDR4** |
| Local disk | **None — diskless.** `$TMPDIR` = `/tmp` is 10 GB tmpfs (per-job cgroup); `/dev/shm` is 252 GB tmpfs (shared with training RAM) |
| InfiniBand | **4× Mellanox ConnectX-6 HDR-100** (100 Gb/s each, pure IB — no RoCE) |
| Aggregate inter-node BW | ~400 Gb/s = ~50 GB/s usable for NCCL across 4 HCAs |
| Compute hostnames | `lrdnXXXX` (4-digit pad, e.g. `lrdn0500`) |
| Total nodes | ~3 456 Booster (= 13 824 A100 GPUs cluster-wide) |
| CUDA driver | 12.0 base; for jobs needing CUDA > 12.0 use **CUDA Forward-Compat** (see `docs/sif-containers.md`) |

## 2. Node Specification (DCGP — CPU only, we have NO budget here)

| Item | Value |
|---|---|
| Partition | `dcgp_usr_prod` |
| CPUs per node | **112 physical** (2× Intel Sapphire Rapids 8480+, 56 cores each) |
| RAM per node | **512 GiB DDR5** |
| Local disk | up to **3 TB NVMe** per job (via `--gres=tmpfs:<N>`) |
| Compute hostnames | `lrdcXXXX` |
| Total nodes | ~1 536 |
| Use case | Pure CPU work — preprocessing, simulation. **`AIFAC_F02_378` has 0 budget here.** |

### CPU-per-GPU rule (CINECA convention, Booster only)
- 1 GPU → **8 logical CPUs**, 4 GPUs → **32 logical CPUs (full node)**
- Don't request `--cpus-per-task` higher than `(GPUs × 8)` — SLURM will reject or pack inefficiently
- Canonical full-node multi-GPU training: `--ntasks-per-node=4 --cpus-per-task=8`

### NUMA layout (all 4 GPUs on NUMA 0 — different from BSC!)

| GPU | CPU affinity | NUMA node |
|---|---|---|
| GPU0–GPU3 | 0–31 (+ HT siblings 32–63) | NUMA 0 (single socket) |

Booster is single-socket — no cross-NUMA penalty between GPUs. This simplifies
DataLoader pinning vs BSC's dual-socket layout. NCCL still uses NVLink for
intra-node GPU↔GPU; the 4× IB HCAs (CX-6 HDR-100) handle inter-node.

### Node usage policy — what to run where

| Node class | Hostnames | Use for | NEVER use for |
|---|---|---|---|
| **Login** | `loginNN.leonardo.local` (8 nodes via round-robin alias `leo`) | `sbatch` / `squeue` / `salloc`, light editing, light `git` | Heavy / long programs (>10 min CPU); large data transfers; `pip install` of big stacks |
| **Datamover (DTN)** | `dmover1..4.leonardo.cineca.it` (alias `data.leonardo.cineca.it` / `data-leo`) | **All large `rsync` / `scp` / `sftp`** to/from outside the cluster | Compute · GPU code · `python train.py` |
| **Booster compute** | `lrdnXXXX` (auto-assigned by SLURM via `boost_usr_prod`) | Training, inference, anything GPU | (no restriction within allocation) |
| **DCGP compute** | `lrdcXXXX` (auto-assigned by SLURM via `dcgp_usr_prod`) | CPU-only batch work | GPU code (no GPUs) |
| **Viz** | `viz*` | Small interactive visualization with 1× GPU | Multi-GPU training |

Operational rules:
- **Never run heavy programs on `leo` / login** — no GPU, admin will kill long-running CPU work
- **Never `rsync` / `scp` / `wget` / `rclone` GB-scale data on login** — use `data-leo` (DTN) for interactive transfers; for SLURM-batched transfers use `sbatch -p lrd_all_serial` (4 h walltime, no GPU budget burn, has internet)
- **Compute nodes (Booster + DCGP) generally have direct outbound internet** on LEONARDO (unlike BSC's no-internet-on-compute). Verify per-job: `srun curl -I https://huggingface.co`

---

## 3. Interactive GPU access

LEONARDO has no dedicated "interactive GPU login" host (unlike some clusters' `alogin3`). Use SLURM to allocate a Booster node with the debug QoS:

```bash
# Interactive shell on a full 4×A100 node (≤ 30 min, debug QoS):
salloc -A aifac_f02_378 -p boost_usr_prod --qos=boost_qos_dbg \
       -N 1 --gres=gpu:4 -t 00:30:00
srun --pty bash -l   # once allocation is granted
```

`boost_qos_dbg` schedules fast (30 min cap, 8 nodes max, max 2 concurrent jobs/user) — perfect for "give me a GPU shell now." Migrate to `--qos=normal` once you know what you're running.

---

## 4. Intra-Node Interconnect (NVLink NV4)

All 4 GPUs are fully meshed via NV4 NVLink (Gen3). Every GPU pair has 4 NVLink lanes.

| Metric | Value |
|---|---|
| Generation | NVLink 3 (Ampere) |
| Lanes per GPU pair | 4 |
| Single lane bandwidth | 50 GB/s |
| **GPU↔GPU peer bandwidth** | **~200 GB/s** (4 × 50) |

### Implication for the template
- ✅ Single-node training is bandwidth-fat (NVLink ≫ IB)
- ⚠️ Cross-node NCCL traffic over IB (~50 GB/s aggregate across 4 HCAs) is ~4× slower than NVLink — keep comms-heavy ops intra-node when possible
- Default NCCL on LEONARDO Booster: leave `NCCL_PXN_DISABLE=1` for the per-HCA parallelism win (see `docs/networking-nccl.md` for the 12 → 60 GB/s measurement)

---

## 5. Inter-Node Interconnect (InfiniBand HDR-100)

Each Booster node has **4× ConnectX-6 HDR-100** HCAs. Unlike BSC MN5 (which mixes IB and RoCE), LEONARDO Booster is **pure IB** — no RoCE.

| HCA name | Type | Rate | Used for |
|---|---|---|---|
| `mlx5_0` | InfiniBand HDR-100 | 100 Gb/s | NCCL data |
| `mlx5_1` | InfiniBand HDR-100 | 100 Gb/s | NCCL data |
| `mlx5_2` | InfiniBand HDR-100 | 100 Gb/s | NCCL data |
| `mlx5_3` | InfiniBand HDR-100 | 100 Gb/s | NCCL data |

Aggregate: **400 Gb/s = ~50 GB/s usable** for NCCL across 4 HCAs.

LEONARDO's HCA naming convention is uniform — `mlx5_0` through `mlx5_3` are
all IB on Booster, no Ethernet HCAs sharing the prefix. The IB LNET tag is
`@o2ib` (Lustre uses the same fabric).

(Viz nodes are different — they have Ethernet HCAs that share the `mlx5_*`
prefix, so on viz you must list `mlx5_0` explicitly, not use a wildcard.
See `docs/networking-nccl.md` § "Booster vs viz".)

---

## 6. NCCL config — see canonical doc

Full NCCL recipe (env vars, `NCCL_PXN_DISABLE=1` win, fabric measurements)
lives in `docs/networking-nccl.md`. Skill-side summary:

```bash
# Minimum effective NCCL config for LEONARDO Booster
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_PXN_DISABLE=1                    # CRITICAL: unlocks 60 GB/s vs default 12 GB/s
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3    # bootstrap candidates
export NCCL_DEBUG=WARN                        # or INFO for first multi-node run
```

⚠️ **`NCCL_PXN_DISABLE=1` is the key knob** — without it NCCL routes all
inter-node traffic through a single HCA, giving you 12 GB/s instead of the
~50 GB/s aggregate. This is the single biggest config bug to avoid on
LEONARDO Booster multi-node training.

For QoS-aware NCCL settings + `--switches=` topology hints, see
`slurm.md` § "Switches topology hint".

---

## 7. Key differences vs BSC MareNostrum 5

| | BSC MN5 (acc) | LEONARDO Booster |
|---|---|---|
| GPU | 4× H100 80GB SXM5 (sm_90) | 4× A100 64GB SXM4 (sm_80) |
| NVLink | NV6 (Gen4) | NV4 (Gen3) |
| CPU | 2× Sapphire Rapids 8480+ (112-core, 2 sockets) | 1× Ice Lake 8358 (32-core, 1 socket) |
| Local disk | NVMe scratch | **None — diskless** (10 GB tmpfs only) |
| Filesystem | GPFS | Lustre |
| NUMA | Dual-socket: GPU0/1 → NUMA 0, GPU2/3 → NUMA 1 | Single-socket: all 4 GPUs on NUMA 0 |
| IB fabric | 2× IB NDR-400 + 4× RoCE Ethernet | 4× IB HDR-100 (pure IB, no RoCE) |
| Compute internet | Proxy tunnel required | Direct outbound (verify per-job) |
| Login alias | `alogin1..2` / `glogin1..2` | `leo` (round-robin `login01..08`) |
| Compute hostnames | `as*` / `gs*` | `lrdnXXXX` / `lrdcXXXX` |

If porting code from BSC:
- **Diskless Booster** is the biggest gotcha — anything that wrote to `/scratch_local/` on BSC must redirect to `$SCRATCH` or `$FAST` on LEO
- **Single-NUMA** simplifies pinning — drop dual-socket NCCL_NSOCKS_PERTHREAD tweaks
- **No RoCE** simplifies NCCL config — no `NCCL_IB_GID_INDEX` knob needed
- **Direct compute internet** lets you `pip install` mid-job; no proxy tunneling

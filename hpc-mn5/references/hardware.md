# BSC MareNostrum5 ACC — Hardware Reference

Background facts about the ACC (accelerator) partition. Read this when you
need the *why* behind a directive in the SLURM template.

**Related**: `slurm.md` (QoS, switches, CPU rule) · `storage-layout.md`
(paths) · `gpfs-hang-resilience.md` (GPFS quirks).

---

## 1. Node Specification

| Item | Value |
|---|---|
| Partition | `acc` |
| GPU | NVIDIA H100 SXM5 |
| GPUs per node | **4** |
| Memory per GPU | **64 GB HBM3** (reported as 63.4 GB usable) |
| Compute capability | **9.0** (Hopper, sm_90a unlocks wgmma/TMA/clusters) |
| SM count per GPU | 132 |
| CPUs per node | **80 logical** (from cgroup; physical layout: 2 sockets × 40 cores) |
| RAM per node | ~512 GB |
| Local NVMe (`/scratch/tmp`) | 436 GB |
| Local RAM disk (`/dev/shm`) | 252 GB |
| Login nodes | `alogin1`, `alogin2` (no GPU) |
| Interactive node | `alogin3` (4 H100, `acc_interactive` only) |
| Total nodes | ~1120 (4480 GPUs cluster-wide) |

### CPU-per-GPU rule (BSC mandate)
- 1 GPU → 20 CPUs, 2 GPUs → 40 CPUs, **4 GPUs → 80 CPUs (full node)**
- This is enforced. Don't request `--cpus-per-task` higher or lower than
  the GPU count × 20.
- Multi-node submission stays at **80 per node**, not aggregated.

### NUMA layout
| GPU | CPU affinity | NUMA node |
|---|---|---|
| GPU0 | 0–39, 80–119 | NUMA 0 |
| GPU1 | 0–39, 80–119 | NUMA 0 |
| GPU2 | 40–79, 120–159 | NUMA 1 |
| GPU3 | 40–79, 120–159 | NUMA 1 |

Cross-NUMA traffic (e.g. GPU1↔GPU2) goes over the inter-socket link and
is slower than NUMA-local. NCCL routes through NVLink so this rarely
matters in practice, but DataLoader CPU pinning can be affected.

### Node usage policy — what to run where

BSC has four classes of nodes. Use the right one for the right task —
running heavy work on the wrong node will get processes killed by the
admin or hang the node for everyone else.

| Node class | Hostnames | Use for | NEVER use for |
|---|---|---|---|
| **ACC login** | `alogin1`, `alogin2` | `sbatch` / `squeue` / `salloc`, light editing, light `git` | Any heavy / long program; large data transfer; `pip install` of big stacks |
| **Transfer** | `transfer1.bsc.es` … `transfer4.bsc.es` | **All large `rsync` / `scp` / `sftp`** to or from BSC | Compute · GPU code · `python train.py` |
| **Interactive GPU** | `alogin3` (only via `acc_interactive` QoS) | Quick GPU exploration (`srun --pty bash`) | Long jobs (≤ 2h cap) · production training |
| **Compute** | `as*`, `gs*` (auto-assigned by SLURM) | Training, inference, anything heavy | (no restriction) |

Operational rules:
- **Never run any large program on `alogin1` / `alogin2`** (no GPU; admin
  will kill long-running processes).
- **Never `rsync` / `scp` large data on `alogin*`** — use `transferN.bsc.es`.
- **Compute nodes have no internet** — outbound only via login/transfer.
  Anything to download (HF model, pip wheel) must be staged from a
  login or transfer node first.

---

## 2. Intra-Node Interconnect (NVLink)

All 4 GPUs are **fully meshed** via NVLink — every pair has a direct link.

```
        GPU0 ←──NV6──→ GPU1
         ↑ ╲          ╱ ↑
         │   ╲      ╱   │
        NV6    NV6     NV6
         │   ╱      ╲   │
         ↓ ╱          ╲ ↓
        GPU2 ←──NV6──→ GPU3
```

| Metric | Value |
|---|---|
| Connection type | NVLink Gen4 (NV6 = 6 lanes) |
| Single lane bandwidth | 26.562 GB/s |
| Lanes per GPU pair | 6 |
| **GPU↔GPU bandwidth** | **~160 GB/s** (6 × 26.562) |

### Implication for the template
- ✅ Single-node training is bandwidth-fat. Use it for small models or
  validation runs.
- ⚠️ Cross-node (NCCL over IB) is ~1.6× slower than NVLink — keep
  comms-heavy ops intra-node when possible.

---

## 3. Inter-Node Interconnect (RoCE / InfiniBand)

Each ACC node has **4 RDMA-capable NICs** that NCCL uses for inter-node
traffic. Not all of them are pure InfiniBand — some are Ethernet running
**RoCE v2** (RDMA over Converged Ethernet).

### NIC layout per node

| HCA name | Type | Rate | Used for |
|---|---|---|---|
| `mlx5_0` | InfiniBand NDR | 400 Gb/s | NCCL data |
| `mlx5_1` | RoCE v2 (Ethernet) | 25 Gb/s | NCCL data |
| `mlx5_2` | Ethernet | 25 Gb/s | management / storage |
| `mlx5_3` | Ethernet | 25 Gb/s | management / storage |
| `mlx5_4` | RoCE v2 (Ethernet) | 25 Gb/s | NCCL data |
| `mlx5_5` | RoCE v2 (Ethernet) | 25 Gb/s | NCCL data |

The four NCCL-capable HCAs are: `mlx5_0`, `mlx5_1`, `mlx5_4`, `mlx5_5`.
That's the source of the template's:
```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
```

### Aggregate inter-node bandwidth
- IB NDR: 400 Gb/s
- RoCE v2 (×3 of 25 Gb/s): ~75 Gb/s
- **Total: ~475 Gb/s ≈ 60 GB/s** effective for NCCL

(Older docs claim 800 Gb/s aggregate. That assumes both ports of dual-rail
NDR 200×2 — current MN5-ACC has the single-port 400 Gb/s NDR, so the
correct number is ~475 Gb/s.)

---

## 4. **NCCL_SOCKET_IFNAME vs NCCL_IB_HCA — different layers**

This is the most misunderstood part of NCCL config. Read carefully.

```
                NCCL data flow on BSC ACC
   ┌──────────────────────────────────────────────────────────┐
   │  Step 1 — BOOTSTRAP / RENDEZVOUS (control plane)         │
   │  ────────────────────────────────                        │
   │  Uses: NCCL_SOCKET_IFNAME → kernel sockets (TCP)         │
   │  Needs: an IP address on the chosen interface            │
   │  Speed: a few KB; speed irrelevant                       │
   │                                                          │
   │  Step 2 — DATA TRANSFER (data plane)                     │
   │  ─────────────────────────                               │
   │  Uses: NCCL_IB_HCA → libibverbs (RDMA)                   │
   │  Needs: an HCA visible to the verbs API                  │
   │          NO IP ADDRESS required on the HCA itself        │
   │  Speed: full RDMA line-rate (400 Gb/s + 3×25 Gb/s)       │
   └──────────────────────────────────────────────────────────┘
```

### Two **different** kernel objects

| Variable | Lists what | Discoverable via |
|---|---|---|
| `NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3` | Network interfaces (have IP) | `ip -4 addr show` |
| `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5` | RDMA HCAs (verbs devices) | `ibstat -l` |

`ib0..ib3` are IPoIB interfaces (for the IB-attached NICs) and
IP-on-Ethernet (for the RoCE-attached NICs). They get IP addresses to
allow the bootstrap socket connection.

`mlx5_*` are the verbs devices — they identify by GID/LID, NOT by IP.

### **The important fact**
> An HCA in `NCCL_IB_HCA` does **not** need an IP address.
> RDMA verbs work via the device-level interface (GID/LID), not via the
> IP stack. NCCL data transfer flows through `mlx5_X` regardless of
> whether the matching `ibX` interface has an IP.

So if you see this:
```bash
$ ip -4 addr show ib3
ib3: <NO-CARRIER>  # no IP — that's fine!
$ ibstat mlx5_5
State: Active     # HCA is up — NCCL data still flows
```
…NCCL inter-node traffic over `mlx5_5` still works. The "no IP" only
prevents `ib3` from being usable for the bootstrap socket — but you have
3 other IPoIB interfaces (`ib0,1,2`) for bootstrap, so that's fine.

### What NCCL would do under each setting

| Config | Bootstrap | Data plane |
|---|---|---|
| `NCCL_SOCKET_IFNAME=ib0` only | uses `ib0`'s IP for TCP rendezvous | uses all `NCCL_IB_HCA` HCAs for RDMA |
| `NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3` | NCCL picks one with an IP | same as above |
| `NCCL_IB_HCA=mlx5_0` only | unaffected | only uses 400 Gb/s NDR HCA, leaves RoCE idle |
| `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5` | unaffected | rail-scheduled across all 4 RDMA NICs |

This is why the template lists **all four** for `NCCL_IB_HCA` and **all four**
for `NCCL_SOCKET_IFNAME` — bootstrap robustness × maximum data bandwidth.

---

## 5. NCCL Settings — what each one does

```bash
export NCCL_IB_DISABLE=0                         # use IB/RoCE, not TCP
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3        # bootstrap candidates
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5   # RDMA data-plane HCAs
export NCCL_NVLS_ENABLE=0                        # disable NVLink Sharp
export NCCL_NET_GDR_LEVEL=2                      # GPU Direct RDMA
export NCCL_IB_GID_INDEX=3                       # RoCE v2 GID
export NCCL_RAS_ENABLE=0                         # disable NCCL 2.20 RAS
```

| Setting | Purpose | Why this value |
|---|---|---|
| `NCCL_IB_DISABLE=0` | enable IB/RoCE backend | TCP fallback would be ~10× slower |
| `NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3` | rendezvous over IPoIB/IPoEth | all 4 IPoIB ifaces give bootstrap robustness |
| `NCCL_IB_HCA=mlx5_0,1,4,5` | data-plane RDMA across all 4 RDMA NICs | mlx5_2/3 are pure ethernet, exclude them |
| `NCCL_NVLS_ENABLE=0` | disable NVLink Sharp | flaky on BSC — Jon Navarro recommendation, ticket #428699 |
| `NCCL_NET_GDR_LEVEL=2` | GPU Direct RDMA over the NIC | GPU memory ↔ HCA without CPU bounce |
| `NCCL_IB_GID_INDEX=3` | RoCE v2 GID lookup | required for RoCE on this fabric |
| `NCCL_RAS_ENABLE=0` | disable RAS reporting | NCCL 2.20+ stability fix (BSC tickets) |

Source for these recommendations: BSC support engineer **Jon Navarro**,
ticket **#428699**, 2026-04-16: "problem-free and good performance on the
ACC fabric." Don't override unless a newer BSC-support recommendation
arrives.

---

## 6. NCCL AllReduce on multi-node training

```
  Step 1: Intra-Node Reduce (NVLink, ~160 GB/s)
  ─────────────────────────────────────────────
  GPU0..3 reduce → local rank 0 holds partial sum

  Step 2: Inter-Node AllReduce (IB+RoCE, ~60 GB/s)
  ────────────────────────────────────────────────
  Local rank 0 of each node exchanges over RDMA → all hold full sum

  Step 3: Intra-Node Broadcast (NVLink, ~160 GB/s)
  ────────────────────────────────────────────────
  Local rank 0 → 1, 2, 3 → every GPU has full sum
```

Bandwidth balance: each step happens at the bandwidth of its medium.
NVLink (160 GB/s) is faster than IB+RoCE aggregate (~60 GB/s), so the
inter-node step is the bottleneck for global all-reduce.

### Tuning hints
- **Gradient accumulation** reduces inter-node frequency → biggest single
  multi-node speedup.
- **ZeRO-3 with `bucket_size`** affects how much gets shuffled at once;
  smaller buckets = more round-trips but lower memory.
- **Rail topology aware** — use `--switches=...` to keep ranks within
  fewer leaf switches, reducing IB hop count. See `slurm.md` § 4.

---

## 7. Multi-node Topology Diagram (4 nodes example)

```
┌───────────────────────┐          ┌───────────────────────┐
│     Node 0 (Rank 0-3) │          │     Node 1 (Rank 4-7) │
│                       │          │                       │
│  GPU0 ←─NV6─→ GPU1    │          │  GPU4 ←─NV6─→ GPU5    │
│   ↑ ╲        ╱ ↑      │          │   ↑ ╲        ╱ ↑      │
│  NV6  NV6   NV6       │          │  NV6  NV6   NV6       │
│   ↓ ╱        ╲ ↓      │          │   ↓ ╱        ╲ ↓      │
│  GPU2 ←─NV6─→ GPU3    │          │  GPU6 ←─NV6─→ GPU7    │
│                       │          │                       │
└──┬──────────────┬─────┘          └─────┬──────────────┬──┘
   │              │                      │              │
   │ mlx5_0,1,4,5 │                      │ mlx5_0,1,4,5 │
   │              │                      │              │
┌──▼──────────────▼─────────────IB Switch────────────────┴──────────────┐
│                      InfiniBand + RoCE fabric                        │
└──┬──────────────┬─────────────────────┬──────────────┬───────────────┘
   │              │                     │              │
┌──▼──────────────▼─────┐          ┌────▼──────────────▼──┐
│     Node 2 (Rank 8-11)│          │   Node 3 (Rank 12-15)│
│   ... same as above   │          │   ... same as above  │
└───────────────────────┘          └──────────────────────┘
```

Source diagrams in `~/test/gpu_network_topology.md` (older but
geometrically still accurate).

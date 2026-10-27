# Fovea

Fovea is an out-of-core GNN training engine for link prediction with trainable node embeddings.
It separates read residency from update residency: neighbor reads are served from a global
INT8 read view of all embeddings, while full FP32 training state is kept only for a moving
window of partitions and a small always-resident set of high-degree nodes.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Execution](#execution)

## Introduction

This repository contains the training engine used in the paper (`src/ars`), its regression
tests (`tests`), the dataset conversion scripts (`tools/data`), the memory measurement tools
(`tools/measurement`), and the baseline launchers for DGL and MariusGNN (`baselines`).
Datasets, trained state, and result logs are not included.

## Installation

Run the commands below from the repository root unless noted otherwise.
The engine requires Linux and a CUDA-capable GPU; partition storage must support direct I/O.

1. Create a Python 3.10 environment and install the dependencies:
   ```bash
   pip3 install torch==2.3.1 numpy pandas
   pip3 install dgl==2.5.0   # baseline only
   ```
2. Convert the datasets into the partitioned layout and point `FOVEA_DATA` at the output
   directory:
   ```bash
   export FOVEA_DATA=/path/to/datasets
   python3 tools/data/convert_fb86m.py  $FOVEA_DATA/fb86m_def  1024
   python3 tools/data/convert_wkg90m.py $FOVEA_DATA/wkg90m_def 1024
   ```
   The scripts read the public Freebase86M and WikiKG90Mv2 releases from
   `$FOVEA_DATA/fb86m_raw` and `$FOVEA_DATA/wkg90m_raw`.

## Execution

1. **Navigate to the engine directory**:
   ```bash
   cd src
   ```

2. **Run the default configuration** (GPU read view and resident set, host window, Freebase86M):
   ```bash
   python3 -m ars.run --dataset fb86m --paged true --window_device cpu --edges_device disk \
     --cond C1n --replica int8 --replica_frac 1.0 --sched seq120 \
     --always_frac 0.0375 --always_sel deg --always_device cuda --state_device cuda \
     --nbr_dir split --fanout 20 --batch 4000 --nchunks 40 --eval_queries 100000 \
     --epochs 10 --seed 0 --counters false --state_root /path/to/state --out results/fb86m.json
   ```
   Use `--dataset wkg90m --sched seq130 --always_frac 0.0625` for WikiKG90Mv2.

3. **Variants**:
   - `--replica fp32` keeps the read view in FP32.
   - `--state_device cpu` moves the read view to the host; `--always_device cpu` moves the resident set.
   - `--always_frac 0` removes the resident set.
   - `--cond B --replica live --always_frac 0` restricts neighbor reads to the window (partition-buffer behavior).
   - `--counters true` records access counts for diagnostics; keep it off for timing runs.

4. **Output**:
   Per-epoch validation MRR and H@50, training and evaluation time, memory, and I/O are
   written to the JSON file given by `--out`.

## Tests

From the repository root, the read-view test runs without graph data:

```bash
python3 tests/test_replica_fp32.py
```

Most integration tests expect a prepared ogbl-collab dataset in
`$FOVEA_DATA/collab_buf`, which the large-graph conversion scripts do not create.

## Baselines

The DGL launcher is in `baselines/dgl`. The MariusGNN launcher requires a separate
Marius installation and a dataset-specific YAML configuration passed as its first
argument. The Marius source tree and configurations are not included in this package.

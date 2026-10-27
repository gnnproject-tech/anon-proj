import sys, os, numpy as np, pandas as pd, time

DATA = os.environ.get("FOVEA_DATA", "data")
SRC = f"{DATA}/fb86m_raw"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/fb86m_def"
P = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
N, R = 86054151, 14824
os.makedirs(f"{OUT}/edges", exist_ok=True)
os.makedirs(f"{OUT}/nodes", exist_ok=True)
t0 = time.time()


def load(name):
    parts = [
        c.to_numpy()
        for c in pd.read_csv(
            f"{SRC}/{name}.txt", sep="\t", header=None, dtype=np.int32, chunksize=20_000_000
        )
    ]
    a = np.concatenate(parts)
    print(f"  {name}: {len(a):,} rows  {time.time()-t0:.0f}s", flush=True)
    return a


tr, va, te = load("train"), load("valid"), load("test")
assert max(tr[:, :2].max(), va[:, :2].max(), te[:, :2].max()) < N and tr[:, 2].max() < R
perm = np.random.default_rng(0).permutation(N).astype(np.int32)
PSZ = -(-N // P)


def remap(a):
    out = np.empty_like(a)
    out[:, 0] = perm[a[:, 0]]
    out[:, 2] = perm[a[:, 1]]
    out[:, 1] = a[:, 2]
    return out


tr, va, te = remap(tr), remap(va), remap(te)
print(f"  remapping done {time.time()-t0:.0f}s", flush=True)
bid = (tr[:, 0] // PSZ).astype(np.int64) * P + (tr[:, 2] // PSZ)
order = np.argsort(bid, kind="stable")
tr = tr[order]
bid = bid[order]
sz = np.bincount(bid, minlength=P * P)
del order, bid
print(f"  bucket sort done {time.time()-t0:.0f}s", flush=True)
tr.astype(np.int32).tofile(f"{OUT}/edges/train_edges.bin")
va.astype(np.int32).tofile(f"{OUT}/edges/validation_edges.bin")
te.astype(np.int32).tofile(f"{OUT}/edges/test_edges.bin")
np.savetxt(f"{OUT}/edges/train_partition_offsets.txt", sz, fmt="%d")
sz.astype(np.int64).tofile(f"{OUT}/edges/train_partition_offsets.bin")
np.minimum(np.arange(N) // PSZ, P - 1).astype(np.int32).tofile(f"{OUT}/nodes/node_partitions.bin")
open(f"{OUT}/dataset.yaml", "w").write(
    f"dataset_dir: {OUT}/\nnum_edges: {len(tr)}\nnum_nodes: {N}\nnum_relations: {R}\nnum_train: {len(tr)}\nnum_valid: {len(va)}\nnum_test: {len(te)}\nnode_feature_dim: -1\nrel_feature_dim: -1\nnum_classes: -1\ninitialized: false\n"
)
assert sz.sum() == len(tr)
print(f"done: P={P} PSZ={PSZ} bucket total {sz.sum():,}  {time.time()-t0:.0f}s")
print("CONVERT DONE")

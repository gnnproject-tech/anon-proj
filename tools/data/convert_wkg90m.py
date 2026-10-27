import os, glob, sys, numpy as np, time

DATA = os.environ.get("FOVEA_DATA", "data")
SRC = f"{DATA}/wkg90m_raw"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/wkg90m_def"
P = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
N_EXP, R_EXP, E_EXP = 91_230_610, 1_387, 601_062_811
t0 = time.time()


def find(name):
    m = glob.glob(f"{SRC}/**/{name}", recursive=True)
    assert m, f"{name} not found; check ziplist.txt"
    return m[0]


tr = np.load(find("train_hrt.npy"), mmap_mode="r")
assert tr.ndim == 2 and tr.shape[1] == 3, tr.shape
assert tr.shape[0] == E_EXP, f"train edge count {tr.shape[0]:,} != expected {E_EXP:,}"
val_hr = np.load(find("val_hr.npy"))
val_t = np.load(find("val_t.npy")).ravel()
assert len(val_hr) == len(val_t) and val_hr.shape[1] == 2, (val_hr.shape, val_t.shape)

CH = 50_000_000
mh = mr = mt = 0
for i in range(0, len(tr), CH):
    c = np.asarray(tr[i : i + CH])
    mh = max(mh, int(c[:, 0].max()))
    mr = max(mr, int(c[:, 1].max()))
    mt = max(mt, int(c[:, 2].max()))
    print(f"  scan {i + len(c):,} / {len(tr):,}  {time.time()-t0:.0f}s", flush=True)
N = max(mh, mt, int(val_hr[:, 0].max()), int(val_t.max())) + 1
R = max(mr, int(val_hr[:, 1].max())) + 1
assert N == N_EXP and R == R_EXP, f"N={N} R={R} differ from the expected values; aborting"
PSZ = -(-N // P)

perm = np.random.default_rng(0).permutation(N).astype(np.int32)
out = np.empty((len(tr), 3), np.int32)
for i in range(0, len(tr), CH):
    c = np.asarray(tr[i : i + CH])
    out[i : i + len(c), 0] = perm[c[:, 0]]
    out[i : i + len(c), 1] = c[:, 1]
    out[i : i + len(c), 2] = perm[c[:, 2]]
print(f"  remapping done {time.time()-t0:.0f}s", flush=True)

bid = (out[:, 0] // PSZ).astype(np.int64) * P + (out[:, 2] // PSZ)
order = np.argsort(bid, kind="stable")
out = out[order]
bid = bid[order]
sz = np.bincount(bid, minlength=P * P)
del order, bid
print(f"  bucket sort done {time.time()-t0:.0f}s", flush=True)

os.makedirs(f"{OUT}/edges", exist_ok=True)
os.makedirs(f"{OUT}/nodes", exist_ok=True)
out.tofile(f"{OUT}/edges/train_edges.bin")
va = np.stack(
    [
        perm[val_hr[:, 0].astype(np.int64)],
        val_hr[:, 1].astype(np.int64),
        perm[val_t.astype(np.int64)],
    ],
    1,
).astype(np.int32)
va.tofile(f"{OUT}/edges/validation_edges.bin")
va.tofile(f"{OUT}/edges/test_edges.bin")
np.savetxt(f"{OUT}/edges/train_partition_offsets.txt", sz, fmt="%d")
sz.astype(np.int64).tofile(f"{OUT}/edges/train_partition_offsets.bin")
np.minimum(np.arange(N) // PSZ, P - 1).astype(np.int32).tofile(f"{OUT}/nodes/node_partitions.bin")
open(f"{OUT}/dataset.yaml", "w").write(
    f"dataset_dir: {OUT}/\nnum_edges: {len(out)}\nnum_nodes: {N}\nnum_relations: {R}\n"
    f"num_train: {len(out)}\nnum_valid: {len(va)}\nnum_test: {len(va)}\n"
    f"node_feature_dim: -1\nrel_feature_dim: -1\nnum_classes: -1\ninitialized: false\n"
)
assert sz.sum() == len(out)
print(
    f"done: N={N:,} R={R} E={len(out):,} valid={len(va):,} P={P} PSZ={PSZ}  {time.time()-t0:.0f}s"
)
print("CONVERT DONE")

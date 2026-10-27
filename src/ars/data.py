import math, os, time, numpy as np, torch

DATA = os.environ.get("FOVEA_DATA", "data")
PATHS = {
    "collab": (f"{DATA}/collab_buf", 64, 235868),
    "wiki": (f"{DATA}/wiki_fix", 256, 2500604),
    "fb86m": (f"{DATA}/fb86m_def", 1024, 86054151),
    "wkg90m": (f"{DATA}/wkg90m_def", 1024, 91230610),
}


class Data:
    def __init__(
        self, name, dev, edges_dev=None, nbr_dir="union", eval_queries=10000, eval_split="valid"
    ):
        D, self.P, self.N = PATHS[name]
        self.PSZ = math.ceil(self.N / self.P)
        e = np.fromfile(f"{D}/edges/train_edges.bin", dtype=np.int32).reshape(-1, 3)
        self.E = len(e)
        self.edge_file = f"{D}/edges/train_edges.bin"
        ed = edges_dev or dev
        self.edges_disk = ed == "disk"
        if self.edges_disk:
            self.SRC = self.REL = self.DST = None
        elif ed == "cpu" and dev != "cpu":
            from .storage.pinned import PinnedArena

            self.arena = PinnedArena()
            blk = self.arena.alloc((3, self.E), torch.int32)
            for i in range(3):
                blk[i].copy_(torch.from_numpy(np.ascontiguousarray(e[:, i])))
            self.SRC, self.REL, self.DST = blk[0], blk[1], blk[2]
        else:
            self.SRC, self.REL, self.DST = [torch.tensor(e[:, i], device=ed) for i in (0, 1, 2)]

        self.NREL = int(e[:, 1].max()) + 1
        try:
            import re as _re

            m = _re.search(r"num_relations:\s*(\d+)", open(f"{D}/dataset.yaml").read())
            if m:
                self.NREL = max(self.NREL, int(m.group(1)))
        except FileNotFoundError:
            pass
        sz = np.loadtxt(f"{D}/edges/train_partition_offsets.txt", dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(sz)])
        bid = (e[:, 0].astype(np.int64) // self.PSZ).clip(max=self.P - 1) * self.P + (
            e[:, 2].astype(np.int64) // self.PSZ
        ).clip(max=self.P - 1)
        assert (np.diff(bid) >= 0).all() and (
            np.bincount(bid, minlength=self.P * self.P) == sz
        ).all(), "bucket order and offsets disagree"

        cache = f"{D}/csr_cache"

        def build_or_load(tag, u, v):
            fi, fv = f"{cache}/indptr{tag}.bin", f"{cache}/indices{tag}.bin"
            if os.path.exists(fv):
                indptr = np.fromfile(fi, dtype=np.int64)
                vv = np.fromfile(fv, dtype=np.int64)
                assert len(indptr) == self.N + 1 and len(vv) == len(
                    u
                ), f"csr_cache{tag} does not match the dataset; delete it and rerun"
                return indptr, vv.astype(np.int32)
            t0 = time.time()
            o = torch.argsort(torch.from_numpy(u), stable=True).numpy()
            us, vv = u[o], v[o].astype(np.int64)
            del o
            indptr = np.concatenate([[0], np.cumsum(np.bincount(us, minlength=self.N))])
            del us
            if self.E > 50_000_000:
                os.makedirs(cache, exist_ok=True)
                indptr.tofile(fi)
                vv.tofile(fv)
                print(f"CSR{tag} built in {time.time()-t0:.0f}s -> cached", flush=True)
            return indptr, vv.astype(np.int32)

        self.nbr_dir = nbr_dir
        if nbr_dir == "split":
            ip_o, ix_o = build_or_load("_out", e[:, 0].astype(np.int64), e[:, 2])
            ip_i, ix_i = build_or_load("_in", e[:, 2].astype(np.int64), e[:, 0])
            self.csr = {
                "out": (torch.tensor(ip_o, device=dev), torch.tensor(ix_o, device=dev)),
                "in": (torch.tensor(ip_i, device=dev), torch.tensor(ix_i, device=dev)),
            }
            self.indptr = self.csr["out"][0] + self.csr["in"][0]
            self.indices = None
        else:
            u = np.concatenate([e[:, 0], e[:, 2]]).astype(np.int64)
            v = np.concatenate([e[:, 2], e[:, 0]])
            indptr, vv = build_or_load("", u, v)
            del u, v
            self.indptr = torch.tensor(indptr, device=dev)
            self.indices = torch.tensor(vv, device=dev)
            del vv
            self.csr = None
        self.deg = self.indptr[1:] - self.indptr[:-1]
        self.deg = self.deg.to(torch.int32)
        self.part = torch.tensor(
            np.minimum(np.arange(self.N) // self.PSZ, self.P - 1).astype(np.int32), device=dev
        )

        load = (
            lambda n: np.fromfile(f"{D}/edges/{n}.bin", dtype=np.int32)
            .reshape(-1, 3)
            .astype(np.int64)
        )
        va, te = load("validation_edges"), load("test_edges")
        rng = np.random.default_rng(0)
        src = te if eval_split == "test" else va
        self.q = torch.tensor(
            src[rng.choice(len(src), min(eval_queries, len(src)), replace=False)], device=dev
        )
        self.eval_split = eval_split
        self.q_test = torch.tensor(
            te[np.random.default_rng(0).choice(len(te), min(eval_queries, len(te)), replace=False)],
            device=dev,
        )
        self.full_eval = name in ("collab", "wiki")
        if self.full_eval:
            allk = np.concatenate([e[:, [0, 2]], va[:, [0, 2]], te[:, [0, 2]]])
            self.keyset = torch.tensor(np.unique(allk[:, 0] * self.N + allk[:, 1]), device=dev)
        self.eval_chunk = 500 if name == "collab" else 100
        self.hub_sampler = name != "collab"

    def window_edges(self, rng):
        parts = [
            np.fromfile(
                self.edge_file, dtype=np.int32, count=(hi - lo) * 3, offset=lo * 12
            ).reshape(-1, 3)
            for lo, hi in rng
        ]
        return torch.from_numpy(np.concatenate(parts) if len(parts) > 1 else parts[0]).to(
            self.part.device, non_blocking=False
        )

    @staticmethod
    def split(bt):
        return bt[:, 0].long(), bt[:, 1].long(), bt[:, 2].long()

    def edges(self, b):
        if self.SRC.device == b.device:
            return self.SRC[b].long(), self.REL[b].long(), self.DST[b].long()
        bc = b.to(self.SRC.device)
        return (
            self.SRC[bc].to(b.device, non_blocking=True).long(),
            self.REL[bc].to(b.device, non_blocking=True).long(),
            self.DST[bc].to(b.device, non_blocking=True).long(),
        )

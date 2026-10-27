from dataclasses import dataclass, field


@dataclass
class Config:
    dataset: str = "collab"
    cond: str = "C1n"
    sched: str | None = None
    mode: str = "GNN"
    epochs: int = 20
    seed: int = 0
    lr_emb: float = 0.01
    lr_dense: float = 0.01
    init: str = "ours"
    replica: str = "live"
    replica_frac: float = 1.0
    refresh_delta: float = 0.0
    nbr_grad: str = "none"
    gbuf: str = "fp32"
    gbuf_frac: float = 1.0
    gbuf_cap: float = 0.0
    redraw: bool = True
    device: str = "cuda"
    state_device: str = "cuda"
    out: str = "results/run.json"
    paged: bool = False
    direct_io: bool = True
    state_root: str = "state"
    gpu_cache_frac: float = 0.0
    eval_sampled: bool = False
    edges_device: str = ""
    profile: bool = False
    gbuf_q: bool = False
    neg_fanout: int = 0
    gbuf_acc: str = "fp16"
    init_scope: str = "part"
    nbr_dir: str = "union"
    negs_shared: bool = False
    gather: str = "unique"
    window_device: str = ""
    host_stale: int = 0
    eval_queries: int = 10000
    final_test: bool = False
    eval_split: str = "valid"
    max_windows: int = 0
    sampler: str = "exact"
    always_frac: float = 0.0
    always_sel: str = "deg"
    always_seed: int = 0
    always_device: str = "cpu"
    always_freeze_out: bool = False
    counters: bool = True
    layers: int = 1
    encoder: str = "sage"
    heads: int = 10
    slope: float = 0.2
    d: int = 50
    fanout: int = 10
    batch: int = 1000
    nchunks: int = 10
    negs: int = 500

    @property
    def windowed(self):
        return self.cond != "FULL"

    @property
    def r2(self):
        return self.cond in ("B", "C1g")

    @property
    def r3(self):
        return self.cond in ("B", "C1n")

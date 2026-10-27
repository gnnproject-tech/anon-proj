import argparse, json, math, os, re, time
import numpy as np, torch, torch.nn.functional as F
import dgl

DATA = os.environ.get("FOVEA_DATA", "data")
PATHS = {
    "collab": f"{DATA}/collab_buf",
    "fb86m": f"{DATA}/fb86m_def",
    "wkg90m": f"{DATA}/wkg90m_def",
}
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="fb86m")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--d", type=int, default=50)
ap.add_argument("--batch", type=int, default=4000)
ap.add_argument("--nchunks", type=int, default=40)
ap.add_argument("--negs", type=int, default=500)
ap.add_argument("--fanout", type=int, default=20)
ap.add_argument("--lr_emb", type=float, default=0.01)
ap.add_argument("--lr_dense", type=float, default=0.01)
ap.add_argument("--eval_queries", type=int, default=100000)
ap.add_argument("--final_test", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--emb_device", default="cpu")
ap.add_argument("--max_batches", type=int, default=0)
ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--out", default="results/dgl.json")
a = ap.parse_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = torch.device("cuda")
D = PATHS[a.dataset]
t_start = time.time()


def rss():
    return (
        int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) * 1024
    )


meta = open(f"{D}/dataset.yaml").read()
N = int(re.search(r"num_nodes:\s*(\d+)", meta).group(1))
NREL = int(re.search(r"num_relations:\s*(\d+)", meta).group(1))
load = lambda n: np.fromfile(f"{D}/edges/{n}.bin", dtype=np.int32).reshape(-1, 3)
e = load("train_edges")
E = len(e)
print(f"N={N:,} R={NREL} E={E:,}", flush=True)
va, te = load("validation_edges").astype(np.int64), load("test_edges").astype(np.int64)
q_val = torch.tensor(
    va[np.random.default_rng(0).choice(len(va), min(a.eval_queries, len(va)), replace=False)],
    device=dev,
)
q_test = torch.tensor(
    te[np.random.default_rng(0).choice(len(te), min(a.eval_queries, len(te)), replace=False)],
    device=dev,
)
src = torch.from_numpy(e[:, 0].copy())
dst = torch.from_numpy(e[:, 2].copy())
rel_np = e[:, 1].copy()
del e
e_src, e_dst = src, dst

g = dgl.heterograph(
    {("n", "out", "n"): (src, dst), ("n", "in", "n"): (dst, src)},
    num_nodes_dict={"n": N},
    idtype=torch.int64,
)
g = g.formats(["csc"])
g.create_formats_()
print(f"graph built {time.time()-t_start:.0f}s rss {rss()/2**30:.1f} GiB", flush=True)
K2 = a.fanout // 2
sampler = dgl.dataloading.NeighborSampler([{"out": K2, "in": K2}], replace=True)


emb_dev = torch.device(a.emb_device)
emb = torch.nn.Embedding(N, a.d, sparse=True)
with torch.no_grad():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(a.seed)
    for i in range(0, N, 1 << 22):
        emb.weight[i : i + (1 << 22)] = torch.randn((min(1 << 22, N - i), a.d), generator=gen) * 0.1
emb = emb.to(emb_dev)
opt_emb = torch.optim.Adagrad(emb.parameters(), lr=a.lr_emb, eps=1e-10)
R = torch.nn.Parameter(torch.ones(NREL, a.d, device=dev) + 0.1 * torch.randn(NREL, a.d, device=dev))
Ws = torch.nn.Linear(a.d, a.d, bias=True).to(dev)
Wn = torch.nn.Linear(a.d, a.d, bias=False).to(dev)
opt_dense = torch.optim.Adam(
    [R] + list(Ws.parameters()) + list(Wn.parameters()), lr=a.lr_dense, foreach=True
)
print(f"emb {2*N*a.d*4/2**30:.1f} GiB (emb+adagrad) rss {rss()/2**30:.1f} GiB", flush=True)

PROF = os.environ.get("PROF") == "1"
T = {}


def tick(k, t0):
    if PROF:
        torch.cuda.synchronize()
        T[k] = T.get(k, 0) + time.perf_counter() - t0


def encode(nodes):
    _t = time.perf_counter()
    seeds = nodes.cpu()
    inp, out, blocks = sampler.sample_blocks(g, seeds)
    if isinstance(inp, dict):
        inp = inp["n"]
    blk = blocks[0].to(dev)
    x_in = emb(inp.long().to(emb_dev)).to(dev)
    return agg_encode(blk, x_in)


@torch.no_grad()
def evaluate(q):
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    ranks = []
    for side in (1, 0):
        for i in range(0, len(q), 500):
            b = q[i : i + 500]
            s, r, t = b[:, 0], R[b[:, 1]], b[:, 2]
            anchor, true = (s, t) if side == 1 else (t, s)
            negs = torch.randint(N, (len(b), 1000), device=dev, generator=gen)
            nodes, inv = torch.unique(
                torch.cat([anchor, true, negs.flatten()]), return_inverse=True
            )
            hn = encode(nodes)
            ha, ht = hn[inv[: len(b)]], hn[inv[len(b) : 2 * len(b)]]
            hneg = hn[inv[2 * len(b) :]].view(len(b), 1000, -1)
            ts = (ha * r * ht).sum(-1, keepdim=True)
            sc = torch.bmm((ha * r)[:, None, :], hneg.transpose(1, 2)).squeeze(1)
            if not bool(torch.isfinite(ts).all() & torch.isfinite(sc).all()):
                raise RuntimeError("non-finite eval scores")
            ranks.append((sc > ts).sum(1) + 1)
    rk = torch.cat(ranks).float()
    return dict(
        h50=(rk <= 50).float().mean().item(), mrr=(1 / rk).mean().item(), mr=rk.mean().item()
    )


class LPSampler(dgl.dataloading.Sampler):
    def __init__(self, nbr, e_src, e_dst, rel, N, nch, negs):
        super().__init__()
        self.nbr, self.e_src, self.e_dst, self.rel, self.N, self.nch, self.negs = (
            nbr,
            e_src,
            e_dst,
            rel,
            N,
            nch,
            negs,
        )

    def sample(self, g, eids):
        s = self.e_src[eids].long()
        d = self.e_dst[eids].long()
        r = self.rel[eids].long()
        negs_t = torch.randint(self.N, (self.nch, self.negs))
        negs_h = torch.randint(self.N, (self.nch, self.negs))
        nodes, inv = torch.unique(
            torch.cat([s, d, negs_t.flatten(), negs_h.flatten()]), return_inverse=True
        )
        inp, out, blocks = self.nbr.sample_blocks(g, nodes)
        if isinstance(inp, dict):
            inp = inp["n"]
        return inp, blocks[0], inv, s, d, r


def agg_encode(blk, x_in):
    with blk.local_scope():
        blk.srcdata["h"] = x_in
        blk.update_all(dgl.function.copy_u("h", "m"), dgl.function.sum("m", "s"), etype="out")
        s_out = blk.dstdata["s"].clone()
        blk.update_all(dgl.function.copy_u("h", "m"), dgl.function.sum("m", "s"), etype="in")
        s_in = blk.dstdata["s"]
        deg = (
            (blk.in_degrees(etype="out") + blk.in_degrees(etype="in"))
            .clamp(min=1)
            .unsqueeze(1)
            .float()
        )
        agg = (s_out + s_in) / deg
    return F.relu(Ws(x_in[: blk.num_dst_nodes()]) + Wn(agg))


log = dict(args=vars(a), N=N, NREL=NREL, E=E, epochs=[])
nch, negs, Bsz = a.nchunks, a.negs, a.batch
chunk = math.ceil(Bsz / nch)
rel_t = torch.from_numpy(rel_np)
lp = LPSampler(sampler, e_src, e_dst, rel_t, N, nch, negs)
loader = dgl.dataloading.DataLoader(
    g,
    torch.arange(E),
    lp,
    batch_size=Bsz,
    shuffle=True,
    drop_last=False,
    num_workers=a.workers,
    use_prefetch_thread=False,
)
for ep in range(1, a.epochs + 1):
    t0 = time.time()
    tot = torch.zeros((), device=dev)
    nb = 0
    rss_max = 0
    torch.cuda.reset_peak_memory_stats()
    for inp, blk, inv, s, d_, r in loader:
        _t = time.perf_counter()
        blk = blk.to(dev)
        inv = inv.to(dev)
        s = s.to(dev)
        d_ = d_.to(dev)
        r = r.to(dev)
        B = len(s)
        tick("sample", _t)
        _t = time.perf_counter()
        x_in = emb(inp.to(emb_dev)).to(dev)
        tick("gather", _t)
        _t = time.perf_counter()
        h = agg_encode(blk, x_in)
        tick("fwd", _t)
        hs, hd = h[inv[:B]], h[inv[B : 2 * B]]
        nn_ = nch * negs
        hn_t = h[inv[2 * B : 2 * B + nn_]].view(nch, negs, a.d)
        hn_h = h[inv[2 * B + nn_ :]].view(nch, negs, a.d)
        rr = R[r]
        pos = (hs * rr * hd).sum(-1, keepdim=True)
        qs, qd = hs * rr, hd * rr
        pad = nch * chunk - B
        if pad:
            qs = F.pad(qs, (0, 0, 0, pad))
            qd = F.pad(qd, (0, 0, 0, pad))
        tail = torch.bmm(qs.view(nch, chunk, a.d), hn_t.transpose(1, 2)).reshape(-1, negs)[:B]
        head = torch.bmm(qd.view(nch, chunk, a.d), hn_h.transpose(1, 2)).reshape(-1, negs)[:B]
        tgt = torch.zeros(B, dtype=torch.long, device=dev)
        loss = F.cross_entropy(torch.cat([pos, tail], 1), tgt, reduction="sum") + F.cross_entropy(
            torch.cat([pos, head], 1), tgt, reduction="sum"
        )
        opt_dense.zero_grad(set_to_none=True)
        opt_emb.zero_grad(set_to_none=True)
        _t = time.perf_counter()
        loss.backward()
        tick("bwd", _t)
        _t = time.perf_counter()
        opt_emb.step()
        tick("emb_step", _t)
        _t = time.perf_counter()
        opt_dense.step()
        tick("dense_step", _t)
        if PROF and nb % 50 == 49:
            print(
                "  prof " + " ".join(f"{k} {v/(nb+1)*1000:.0f}ms" for k, v in T.items()), flush=True
            )
        tot += loss.detach()
        nb += 1
        if nb % 500 == 0:
            rss_max = max(rss_max, rss())
            print(
                f"  ep{ep} batch {nb}/{math.ceil(E/Bsz)} loss {float(tot)/nb:.1f} {time.time()-t0:.0f}s rss {rss_max/2**30:.1f} GiB",
                flush=True,
            )
        if a.max_batches and nb >= a.max_batches:
            break
    t_train = time.time() - t0
    rss_max = max(rss_max, rss())
    m = evaluate(q_val)
    t_eval = time.time() - t0 - t_train
    m.update(
        epoch=ep,
        loss=float(tot),
        sec_train=round(t_train, 1),
        sec_eval=round(t_eval, 1),
        batches=nb,
        vm_rss_max=rss_max,
        gpu_alloc_max_bytes=torch.cuda.max_memory_allocated(),
        gpu_reserved_max_bytes=torch.cuda.max_memory_reserved(),
    )
    if a.final_test and ep == a.epochs:
        mt = evaluate(q_test)
        m["test"] = {k: mt[k] for k in ("h50", "mrr", "mr")}
        print(f"  final test  H@50 {mt['h50']:.4f}  MRR {mt['mrr']:.4f}", flush=True)
    log["epochs"].append(m)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=1)
    print(
        f"e{ep:2d}  loss {float(tot):12.1f}  H@50 {m['h50']:.4f}  MRR {m['mrr']:.4f}  MR {m['mr']:9,.0f}  {t_train+t_eval:.1f}s (train {t_train:.1f}s + eval {t_eval:.1f}s)  rss {rss_max/2**30:.1f} GiB",
        flush=True,
    )

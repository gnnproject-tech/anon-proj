import os, sys, importlib.util, torch, torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, ROOT)
from ars.config import Config
from ars.data import Data
from ars.storage.state import State
from ars.graph.csr import Sampler
from ars.train.model import Model
from ars.train.prof import Prof


def load_baseline():
    spec = importlib.util.spec_from_file_location(
        "base_model", os.path.join(ROOT, "_baseline_pre23/ars/train/model.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Model


def build(ModelCls, cfg, data, dev, record=None):
    torch.manual_seed(0)
    st = State(cfg, data, dev)
    sampler = Sampler(data, cfg.fanout, False, dev, mode=cfg.sampler)
    if record is not None:
        orig = sampler.sample

        def rec(U, allowed, K=None):
            sel, cnt = orig(U, allowed, K)
            record.append((U.clone(), sel.clone(), cnt.clone()))
            return sel, cnt

        sampler.sample = rec
    model = ModelCls(cfg, data, st, sampler, dev)
    model.prof = Prof(False, dev)
    return st, model


def roots(data, dev, n=3000):
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    return torch.randint(data.N, (n,), device=dev, generator=g)


def lossf(h):
    return (h * torch.linspace(-1, 1, h.shape[1], device=h.device)).sum() + (h**2).sum() * 0.1


def grads_of(loss, params):
    for p in params:
        p.grad = None
    loss.backward()
    out = [p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
    return out


def mean_rows(x, sel, cnt):
    m = x[sel].mean(1)
    return torch.where((cnt > 0)[:, None], m, torch.zeros_like(m))


def test_regress(data, dev):
    cfg = Config(dataset="collab", cond="FULL", fanout=10, seed=0, device=dev)
    outs = []
    for M in (load_baseline(), Model):
        st, model = build(M, cfg, data, dev)
        U = roots(data, dev)
        torch.manual_seed(123)
        h = model.encode(U, None)
        loss = (h * torch.linspace(-1, 1, h.shape[1], device=dev)).sum() + (h**2).sum() * 0.1
        loss.backward()
        outs.append(
            (
                h.detach(),
                st.E.grad.detach().clone(),
                model.Ws.weight.grad.clone(),
                model.Wn.weight.grad.clone(),
                torch.get_rng_state().clone(),
            )
        )
    (h1, g1, ws1, wn1, r1), (h2, g2, ws2, wn2, r2) = outs
    exact = dev == "cpu"
    same = lambda a, b: torch.equal(a, b) if exact else bool((a - b).abs().max() < 1e-5)
    ok = same(h1, h2) and same(g1, g2) and same(ws1, ws2) and same(wn1, wn2) and torch.equal(r1, r2)
    print(
        f"A regression 1-layer SAGE ({'bitwise' if exact else 'atol 1e-5'}): output d {(h1-h2).abs().max():.1e}  E.grad d {(g1-g2).abs().max():.1e}  Ws/Wn.grad d {(ws1-ws2).abs().max():.1e}/{(wn1-wn2).abs().max():.1e}  RNG state equal {torch.equal(r1, r2)}  ->",
        "PASS" if ok else "FAIL",
    )
    return ok


def test_sage2(data, dev):
    cfg = Config(dataset="collab", cond="FULL", fanout=4, seed=0, device=dev, layers=2)
    rec = []
    st, model = build(Model, cfg, data, dev, rec)
    U = roots(data, dev, 400)
    torch.manual_seed(123)
    h = model.encode(U, None)
    assert len(rec) == 2, rec
    (U1, sel1, cnt1), (n1r, sel2, cnt2) = rec
    params = [
        st.E,
        model.Ws.weight,
        model.Ws.bias,
        model.Wn.weight,
        model.Ws2.weight,
        model.Ws2.bias,
        model.Wn2.weight,
    ]
    g_eng = grads_of(lossf(h), params)
    h = h.detach()
    if True:
        x = st.E
        sel1s = torch.where((cnt1 > 0)[:, None], sel1, U1[:, None])
        n1 = torch.unique(torch.cat([U1, sel1s.reshape(-1)]))
        assert torch.equal(n1, n1r), "2-hop roots = unique(roots + 1-hop)"
        h1 = F.relu(model.Ws(x[n1]) + model.Wn(mean_rows(x, sel2, cnt2)))
        pos = torch.full((data.N,), -1, dtype=torch.long, device=dev)
        pos[n1] = torch.arange(len(n1), device=dev)
        hn = h1[pos[sel1]].mean(1)
        hn = torch.where((cnt1 > 0)[:, None], hn, torch.zeros_like(hn))
        ref = F.relu(model.Ws2(h1[pos[U1]]) + model.Wn2(hn))
    g_ref = grads_of(lossf(ref), params)
    dg = max((a - b).abs().max().item() for a, b in zip(g_eng, g_ref))
    nz = int((g_eng[0].abs().sum(1) > 0).sum())
    nz2 = int((g_ref[0].abs().sum(1) > 0).sum())
    d = (h - ref.detach()).abs().max().item()
    ok = d < 1e-5 and dg < 1e-4 and nz == nz2
    print(
        f"B 2-layer SAGE matches reference: output max d {d:.1e}, gradients (E, Ws, Wn, Ws2, Wn2) max d {dg:.1e} (E nonzero rows {nz} vs {nz2}; {len(U)} roots, {len(n1)} 1-hop nodes, {sel2.numel()} 2-hop slots) ->",
        "PASS" if ok else "FAIL",
    )
    return ok


def test_gat(data, dev):
    cfg = Config(
        dataset="collab",
        cond="FULL",
        fanout=4,
        seed=0,
        device=dev,
        encoder="gat",
        heads=3,
        sampler="cand64",
    )
    rec = []
    st, model = build(Model, cfg, data, dev, rec)
    U = roots(data, dev, 300)
    allowed = torch.ones(data.N, dtype=torch.bool, device=dev)
    for r in U[:10].tolist():
        allowed[data.indices[data.indptr[r] : data.indptr[r + 1]].long()] = False
    torch.manual_seed(123)
    h = model.encode(U, allowed)
    ((U1, sel, cnt),) = rec
    H, dh, d = cfg.heads, cfg.d, cfg.d
    params = [st.E, model.W.weight, model.a_l, model.a_r, model.b]
    g_eng = grads_of(lossf(h), params)
    h = h.detach()
    if True:
        x = st.E
        W = model.W.weight

        ref = []
        for i in range(len(U1)):
            acc = torch.zeros(d, device=dev)
            nb = sel[i] if cnt[i] > 0 else sel[i][:0]
            ids = torch.cat([U1[i : i + 1], nb])
            for hh in range(H):
                Wh = W[hh * dh : (hh + 1) * dh]
                z = x[ids] @ Wh.T
                e = F.leaky_relu(
                    (model.a_l[hh] * z[0]).sum() + (z * model.a_r[hh]).sum(-1), cfg.slope
                )
                a = torch.softmax(e, 0)
                acc += (a[:, None] * z).sum(0)
            ref.append(F.relu(acc / H + model.b))
        ref = torch.stack(ref)
        g_ref = grads_of(lossf(ref), params)
        dg = max((a - b).abs().max().item() for a, b in zip(g_eng, g_ref))
        d1 = (h - ref.detach()).abs().max().item()

        with torch.no_grad():
            import dgl
            from dgl.nn import GATConv

            valid = cnt > 0
            ids = torch.unique(torch.cat([U1, sel[valid].reshape(-1)]))
            pos = torch.full((data.N,), -1, dtype=torch.long, device=dev)
            pos[ids] = torch.arange(len(ids), device=dev)
            rr = torch.arange(len(U1), device=dev).repeat_interleave(sel.shape[1])
            src = pos[sel.reshape(-1)][valid[rr]]
            dst = pos[U1[rr]][valid[rr]]
            g = dgl.graph(
                (torch.cat([src, pos[U1]]), torch.cat([dst, pos[U1]])), num_nodes=len(ids)
            ).to(dev)
            conv = GATConv(
                d, dh, H, negative_slope=cfg.slope, bias=False, allow_zero_in_degree=True
            ).to(dev)
            conv.fc.weight.copy_(W)
            conv.attn_l.copy_(model.a_r.view(1, H, dh))
            conv.attn_r.copy_(model.a_l.view(1, H, dh))
            out = conv(g, x[ids])
            ref2 = F.relu(out[pos[U1]].mean(1) + model.b)
            d2 = (h - ref2).abs().max().item()
    ok = d1 < 1e-5 and d2 < 1e-5 and dg < 1e-4
    print(
        f"C GAT: Marius loop reference output max d {d1:.1e}, gradients (E, W, a_l, a_r, b) max d {dg:.1e}; DGL GATConv (+self edges) output max d {d2:.1e} ({len(U1)} roots, {H} heads, {int((~valid).sum())} roots without neighbors) ->",
        "PASS" if ok else "FAIL",
    )
    return ok


def test_gradcheck():
    import types

    torch.manual_seed(0)
    m, K, u, d = 4, 3, 9, 6
    inv = torch.cat([torch.randperm(u)[:m], torch.randint(u, (m * K,))])
    valid = torch.tensor([True, True, False, True])
    feats = torch.randn(u, d, dtype=torch.float64, requires_grad=True)
    Ws = torch.nn.Linear(d, d).double()
    Wn = torch.nn.Linear(d, d, bias=False).double()
    ok = torch.autograd.gradcheck(
        lambda f: Model._sage(f, inv, valid, m, K, Ws, Wn), (feats,), eps=1e-6, atol=1e-5
    )
    cfg = Config(dataset="collab", cond="FULL", d=d, encoder="gat", heads=2, device="cpu")
    g = Model(cfg, types.SimpleNamespace(NREL=2), None, None, "cpu")
    g.W.double()
    g.a_l = torch.nn.Parameter(g.a_l.double())
    g.a_r = torch.nn.Parameter(g.a_r.double())
    g.b = torch.nn.Parameter(g.b.double())
    ok &= torch.autograd.gradcheck(
        lambda f: g._gat(f, inv, valid, m, K), (feats,), eps=1e-6, atol=1e-5
    )
    print("D gradcheck (float64, feats): _sage, _gat →", "PASS" if ok else "FAIL")
    return ok


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    data = Data("collab", dev)
    ok = test_regress(data, dev)
    ok &= test_sage2(data, dev)
    ok &= test_gat(data, dev)
    ok &= test_gradcheck()
    print("MODELS", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

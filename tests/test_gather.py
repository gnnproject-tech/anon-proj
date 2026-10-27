import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ars.config import Config
from ars.data import Data
from ars.storage.state import State
from ars.graph.csr import Sampler
from ars.train.model import Model
from ars.train.prof import Prof


def run(gather, data, dev):
    cfg = Config(dataset="collab", cond="FULL", gather=gather, fanout=10, seed=0)
    torch.manual_seed(0)
    st = State(cfg, data, dev)
    sampler = Sampler(data, cfg.fanout, True, dev)
    model = Model(cfg, data, st, sampler, dev)
    model.prof = Prof(False, dev)
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    U = torch.randint(data.N, (5000,), device=dev, generator=g)
    torch.manual_seed(123)
    h = model.encode(U, None)
    loss = (h * torch.linspace(-1, 1, h.shape[1], device=dev)).sum() + (h**2).sum() * 0.1
    loss.backward()
    return (
        h.detach(),
        st.E.grad.detach().clone(),
        model.Wn.weight.grad.detach().clone(),
        model.Ws.weight.grad.detach().clone(),
    )


def main():
    dev = "cuda"
    data = Data("collab", dev)
    h1, g1, wn1, ws1 = run("occ", data, dev)
    h2, g2, wn2, ws2 = run("unique", data, dev)
    dh = (h1 - h2).abs().max().item()
    dg = (g1 - g2).abs().max().item()
    dwn = (wn1 - wn2).abs().max().item()
    dws = (ws1 - ws2).abs().max().item()
    nz = int((g1.abs().sum(1) > 0).sum())
    nz2 = int((g2.abs().sum(1) > 0).sum())
    print(
        f"encode output max diff {dh:.2e} | E.grad max diff {dg:.2e} (nonzero rows {nz} vs {nz2}) | Wn.grad {dwn:.2e} Ws.grad {dws:.2e}"
    )
    ok = dh < 1e-5 and dg < 1e-5 and dwn < 1e-4 and dws < 1e-4 and nz == nz2
    print("GATHER", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

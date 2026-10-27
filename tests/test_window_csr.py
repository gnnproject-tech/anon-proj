import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ars.config import Config
from ars.data import Data
from ars.storage.state_paged import PagedState
from ars.graph.csr import Sampler


def main():
    dev = "cuda"
    cfg = Config(dataset="collab", cond="B", sched="seq", paged=True, fanout=10)
    data = Data("collab", dev)
    st = PagedState(cfg, data, dev, f"state/test_wcsr_{os.getpid()}", True)
    st.alloc(2)
    res = [3, 17]
    for p in res:
        st.load(p)
    sam = Sampler(data, cfg.fanout, True, dev, mode="exact")
    sam.set_window(st, res)
    allowed = st.resident_mask()
    roots = torch.nonzero(allowed).squeeze(1)
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    roots = roots[torch.randperm(len(roots), device=dev, generator=g)[:2000]]
    sel, cnt = sam.sample(roots, allowed)
    ok = True
    bad_nbr = bad_win = bad_cnt = 0
    ip, ix = data.indptr, data.indices
    for i in range(len(roots)):
        u = int(roots[i])
        nb = ix[ip[u] : ip[u + 1]].long()
        inwin = nb[allowed[nb]]
        n_in = int(len(inwin))
        if int(cnt[i]) != n_in:
            bad_cnt += 1
        if n_in == 0:
            continue
        s = sel[i]
        if not bool(torch.isin(s, nb).all()):
            bad_nbr += 1
        if not bool(allowed[s].all()):
            bad_win += 1
    print(f"2000 roots: not a real neighbor {bad_nbr}, outside the window {bad_win}, cnt mismatch {bad_cnt}")
    ok = bad_nbr == bad_win == bad_cnt == 0

    st.store.close()
    import shutil

    shutil.rmtree(f"state/test_wcsr_{os.getpid()}", ignore_errors=True)
    print("WINDOW_CSR", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

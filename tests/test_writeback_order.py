import os, sys, time, types, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ars.config import Config
from ars.data import Data
from ars.storage.state_paged import PagedState


def main():
    dev = "cuda"
    wd = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = Config(dataset="collab", cond="B", sched="seq", paged=True, epochs=1, window_device=wd)
    print("window_device =", wd or "cuda")
    data = Data("collab", dev)
    st = PagedState(cfg, data, dev, f"state/test_wb_{os.getpid()}", True)
    st.alloc(2)
    slow = st.store.store

    def delayed(p, e, o):
        time.sleep(1.0)
        slow(p, e, o)

    st.store.store = delayed
    if os.environ.get("WB_NOFIX"):
        st.wait_for_write = lambda parts: None
    ok = True
    for trial in range(3):
        for p in (0, 1):
            if int(st.slot_of[p]) < 0:
                st.load(p)
        s0 = int(st.slot_of[0])
        rows = slice(s0 * st.PSZ, (s0 + 1) * st.PSZ)
        marker = float(trial + 1)
        st.E[rows] += marker
        st.O[rows] += marker
        want_e, want_o = st.E[rows].clone(), st.O[rows].clone()
        out, inn = st.set_window(torch.tensor([1, 2], device=dev))
        for p in out:
            st.evict(p)
        st.flush_evicts()
        for p in inn:
            st.load(p)
        st.prefetch([0])
        out, inn = st.set_window(torch.tensor([0, 2], device=dev))
        for p in out:
            st.evict(p)
        st.flush_evicts()
        for p in inn:
            st.load(p)
        s0 = int(st.slot_of[0])
        got_e = st.E[s0 * st.PSZ : (s0 + 1) * st.PSZ]
        got_o = st.O[s0 * st.PSZ : (s0 + 1) * st.PSZ]
        same = torch.equal(got_e, want_e) and torch.equal(got_o, want_o)
        print(f"trial {trial}: reloaded E/O == latest {'OK' if same else 'FAIL (stale value read)'}")
        ok &= same

        out, inn = st.set_window(torch.tensor([0, 1], device=dev))
        for p in out:
            st.evict(p)
        st.flush_evicts()
        for p in inn:
            st.load(p)
        st.sync()

    def boom(p, e, o):
        raise IOError("disk full (test)")

    st.store.store = boom
    out, inn = st.set_window(torch.tensor([1, 2], device=dev))
    for p in out:
        st.evict(p)
    st.flush_evicts()
    try:
        st.sync()
        print("writer exception not propagated (silently ignored)")
        ok = False
    except RuntimeError as ex:
        print(f"writer exception propagated ({ex.__cause__})")
    st.store.close()
    import shutil

    shutil.rmtree(f"state/test_wb_{os.getpid()}", ignore_errors=True)
    print("WRITEBACK", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

import os, sys, types, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ars.storage.state_paged import PagedState

ok = True


def chk(cond, msg):
    global ok
    ok &= bool(cond)
    print(("✓ " if cond else "✗ ") + msg)


def stub(N=100, P=5, d=4, dev="cpu"):
    st = PagedState.__new__(PagedState)
    st.N, st.P, st.d, st.dev = N, P, d, dev
    st.PSZ = -(-N // P)
    st.data = types.SimpleNamespace(
        deg=torch.tensor([(i * 7) % 13 for i in range(N)], dtype=torch.int32),
        part=torch.arange(N) // st.PSZ,
    )
    st.slot_of = torch.full((P,), -1, dtype=torch.long)
    st.part_in_slot = [-1, -1, -1]
    st.kS = 0
    st.S_OFF = 3 * st.PSZ
    st.nslots = 3
    return st


st = stub()
g_before = torch.random.get_rng_state().clone()
st._select_always(0.1, "random", 7)
r1 = st.s_ids.clone()
chk(
    torch.equal(torch.random.get_rng_state(), g_before), "random selection leaves the global RNG state unchanged"
)
st2 = stub()
st2._select_always(0.1, "random", 7)
chk(torch.equal(st2.s_ids, r1), "random selection is reproducible from the seed")
st3 = stub()
st3._select_always(0.1, "deg", 0)
deg = st3.data.deg.long()
top = torch.argsort(-deg, stable=True)[:10]
chk(
    torch.equal(st3.s_ids, torch.sort(top).values) and st3.kS == 10,
    "deg selection: top-kS by degree, ties by ascending id, stored sorted",
)
chk(
    int((st3.srow >= 0).sum()) == 10
    and torch.equal(torch.nonzero(st3.srow >= 0).squeeze(1), st3.s_ids),
    "srow inverse mapping matches",
)

st3.slot_of[1] = 0
st3.part_in_slot = [1, -1, -1]
ids = torch.arange(st3.N)
r = st3.rows(ids)
inS = st3.srow >= 0
inWin = st3.data.part == 1
chk(torch.equal(r[inS], st3.S_OFF + st3.srow[inS].long()), "S node -> S region row (inside or outside the window)")
chk(torch.equal(r[inWin & ~inS], (ids - st3.PSZ)[inWin & ~inS]), "in-window non-S -> slot row")
chk(bool((r[~inWin & ~inS] == -1).all()), "out-of-window non-S -> -1")

lr = torch.cat([r[inWin & ~inS], r[inS]])
back = st3.global_of(lr)
chk(torch.equal(back, torch.cat([ids[inWin & ~inS], ids[inS]])), "global_of: inverse of slot rows and S rows")

st4 = stub()
st4.data.deg[-1] = 999
st4._select_always(0.01, "deg", 0)
chk(
    st4.kS == 1
    and int(st4.s_ids[0]) == st4.N - 1
    and int(st4.rows(torch.tensor([st4.N - 1]))[0]) == st4.S_OFF
    and int(st4.global_of(torch.tensor([st4.S_OFF]))[0]) == st4.N - 1,
    "boundary: kS=1, last id maps and inverts",
)
st5 = stub()
st5.kS = 0
chk(bool((st5.rows(torch.arange(st5.N)) == -1).all()), "S=0: legacy path (all non-resident -> -1)")
print("CPU PART", "PASS" if ok else "FAIL")


if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    SDEV = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    print("S device =", SDEV)
    from ars.config import Config
    from ars.data import Data
    from ars.storage.replica import Replica

    dev = "cuda"
    cfg = Config(
        dataset="collab",
        cond="C1n",
        sched="seq",
        paged=True,
        epochs=1,
        window_device="cpu",
        replica="int8",
        replica_frac=1.0,
        always_frac=0.05,
        host_stale=0,
    )
    data = Data("collab", dev)
    st = PagedState(cfg, data, dev, f"state/test_always_{os.getpid()}", True)
    st.alloc(2, 1, 1, always_frac=0.05, always_sel="deg", always_seed=0, always_device=SDEV)
    st.replica = Replica(data.N, cfg.d, "int8", 1.0, 0.0, data.deg, dev, home="cpu")
    st.init_replica(st.replica)
    kS, OFF = st.kS, st.S_OFF

    e = torch.empty(st.PSZ, st.d)
    o = torch.empty(st.PSZ, st.d)
    good = True
    for p in (0, st.P // 2, st.P - 1):
        lo, hi = p * st.PSZ, min((p + 1) * st.PSZ, st.N)
        st.store.load(p, e, o)
        a = int(torch.searchsorted(st.s_ids_cpu, lo))
        b = int(torch.searchsorted(st.s_ids_cpu, hi))
        if b > a:
            loc = st.s_ids_cpu[a:b] - lo
            good &= torch.equal(st.s_E()[a:b].cpu(), e[loc]) and torch.equal(
                st.s_O()[a:b].cpu(), o[loc]
            )
    chk(
        good and st.s_init_done,
        "init: S rows E/O == disk initial values (first, middle, last partition), done flag set once",
    )

    v = int(st.s_ids[0])
    pv = int(data.part[v])
    st.load(0 if pv != 0 else 1)
    st.begin_batch(None)
    idx = torch.tensor([v, v, v], device=dev)
    x = st.read(idx)
    loss = (x * torch.arange(1, 4, device=dev).view(3, 1)).sum()
    loss.backward()
    E0 = st.s_E()[0].cpu().clone()
    O0 = st.s_O()[0].cpu().clone()
    g_ref = torch.ones(st.d) * (1 + 2 + 3)
    O_ref = O0 + g_ref * g_ref
    E_ref = E0 - st.lr * g_ref / (O_ref.sqrt() + st.eps)
    st.step(None)
    st._join_updates()
    torch.cuda.synchronize()
    chk(
        torch.allclose(st.s_E()[0].cpu(), E_ref, atol=1e-6)
        and torch.allclose(st.s_O()[0].cpu(), O_ref, atol=1e-6),
        "one step: duplicate gradients summed, one Adagrad update per node (out-of-window S)",
    )

    st.load(pv) if int(st.slot_of[pv]) < 0 else None
    s_slot = int(st.slot_of[pv])
    slot_row = s_slot * st.PSZ + (v - pv * st.PSZ)
    st.E[slot_row] = 12345.0
    st.O[slot_row] = 777.0
    E1, O1 = st.s_E()[0].cpu().clone(), st.s_O()[0].cpu().clone()
    got = st.read(torch.tensor([v], device=dev))[0].cpu()
    chk(torch.allclose(got, E1, atol=1e-6), "ownership: reads come from the S region even inside the window (slot corruption ignored)")
    st.begin_batch(None)
    x = st.read(torch.tensor([v], device=dev))
    (x.sum()).backward()
    st.step(None)
    st._join_updates()
    O_ref2 = O1 + 1.0
    E_ref2 = E1 - st.lr * 1.0 / (O_ref2.sqrt() + st.eps)
    torch.cuda.synchronize()
    chk(
        torch.allclose(st.s_E()[0].cpu(), E_ref2, atol=1e-6)
        and torch.allclose(st.s_O()[0].cpu(), O_ref2, atol=1e-6)
        and bool((st.E[slot_row] == 12345.0).all()),
        "in-window S: S region updated once, slot row untouched",
    )
    E2, O2 = st.s_E()[0].cpu().clone(), st.s_O()[0].cpu().clone()
    out, inn = st.set_window(torch.tensor([p for p in range(st.P) if p != pv][:2], device=dev))
    for p in out:
        st.evict(p)
    st.flush_evicts()
    st.join_writer()
    for p in inn:
        st.load(p)
    st.prefetch([pv])
    out, inn = st.set_window(torch.tensor([pv, inn[0] if inn else 0], device=dev))
    for p in out:
        st.evict(p)
    st.flush_evicts()
    st.join_writer()
    for p in inn:
        st.load(p)
    chk(
        torch.equal(st.s_E()[0].cpu(), E2)
        and torch.equal(st.s_O()[0].cpu(), O2)
        and torch.allclose(st.read(torch.tensor([v], device=dev))[0].cpu(), E2, atol=1e-6),
        "ownership: S original and reads unchanged across evict -> writer -> reload",
    )

    chk(
        not bool(st.amask[st.s_ids][st.slot_of[data.part[st.s_ids].long()] < 0].any()),
        "amask: out-of-window S nodes are not in the resident mask (negative pool unchanged)",
    )

    Efull = st.full_E(host=True)
    chk(torch.equal(Efull[st.s_ids_cpu], st.s_E().cpu()), "eval merge: S rows of full_E == S region")
    print("GPU PART", "PASS" if ok else "FAIL")
    import shutil

    shutil.rmtree(f"state/test_always_{os.getpid()}", ignore_errors=True)
raise SystemExit(0 if ok else 1)

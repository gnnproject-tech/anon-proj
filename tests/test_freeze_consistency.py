import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ars.config import Config
from ars.data import Data
from ars.storage.state_paged import PagedState
from ars.storage.replica import Replica

ok = True


def chk(c, msg):
    global ok
    ok &= bool(c)
    print(("✓ " if c else "✗ ") + msg)


SDEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"
PART = sys.argv[2] if len(sys.argv) > 2 else "AB"
dev = "cuda"


def build(freeze):
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
        always_freeze_out=freeze,
    )
    data = Data("collab", dev)
    st = PagedState(cfg, data, dev, f"state/test_fc_{os.getpid()}_{int(freeze)}", True)
    st.alloc(2, 1, 1, always_frac=0.05, always_sel="deg", always_seed=0, always_device=SDEV)
    st.replica = Replica(data.N, cfg.d, "int8", 1.0, 0.0, data.deg, dev, home="cpu")
    st.init_replica(st.replica)
    return cfg, data, st


def quant(e):
    s = e.abs().amax(1).clamp(min=1e-8) / 127
    return torch.round(e / s[:, None]).to(torch.int8), s


for freeze in ((False, True) if "A" in PART else ()):
    cfg, data, st = build(freeze)

    st.load(0)
    st.load(1)
    st.wmask = st.resident_mask()
    sid = st.s_ids
    part = data.part[sid].long()
    v_in = int(sid[torch.nonzero(part <= 1)[0, 0]])
    v_out = int(sid[torch.nonzero(part > 1)[0, 0]])
    j_in = int(st.srow[v_in])
    j_out = int(st.srow[v_out])
    st.begin_batch(None)
    x = st.read(torch.tensor([v_in, v_out, v_in], device=dev))
    (x * 2.0).sum().backward()
    E_in0, O_in0 = st.s_E()[j_in].cpu().clone(), st.s_O()[j_in].cpu().clone()
    E_out0, O_out0 = st.s_E()[j_out].cpu().clone(), st.s_O()[j_out].cpu().clone()
    st.step(None)
    st._join_updates()
    torch.cuda.synchronize()
    in_changed = not torch.equal(st.s_E()[j_in].cpu(), E_in0) and not torch.equal(
        st.s_O()[j_in].cpu(), O_in0
    )
    out_same = torch.equal(st.s_E()[j_out].cpu(), E_out0) and torch.equal(
        st.s_O()[j_out].cpu(), O_out0
    )
    c = st.pop_counters()
    if freeze:
        chk(
            in_changed and out_same,
            f"freeze=on: in-window S updated, out-of-window S E and A unchanged (S={SDEV})",
        )
        chk(
            c["upd_S_in"] == 1 and c["upd_S_out"] == 0 and c["grad_S_out"] == 1,
            f"freeze=on counters: upd_S_in=1 upd_S_out=0 grad_S_out=1 <- {c['upd_S_in']},{c['upd_S_out']},{c['grad_S_out']}",
        )
    else:
        chk(in_changed and not out_same, f"freeze=off: both in-window and out-of-window S updated (S={SDEV})")
        chk(
            c["upd_S_in"] == 1 and c["upd_S_out"] == 1 and c["grad_S_out"] == 0,
            f"freeze=off counters: upd_S_in=1 upd_S_out=1 grad_S_out=0 <- {c['upd_S_in']},{c['upd_S_out']},{c['grad_S_out']}",
        )
    chk(
        c["uniq_rows"] == 3 and c["uniq_S"] == 3 and c["uniq_W"] == 0 and c["uniq_Q"] == 0,
        "read-row classification: all 3 rows in the S region",
    )
    st.sync()


if "B" not in PART:
    print("FREEZE PART", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
cfg, data, st = build(False)
rep = st.replica
PSZ = st.PSZ


def rows_of(p):
    lo, hi = p * PSZ, min((p + 1) * PSZ, data.N)
    ids = torch.arange(lo, hi, device=dev)
    return ids[st.srow[ids] < 0]


def q_equals_E(ids, E_rows):
    q, s = quant(E_rows)
    c = rep.rmap[ids].long().cpu()
    return torch.equal(rep.Q[c], q.cpu()) and torch.allclose(rep.S[c], s.cpu())


e = torch.empty(PSZ, st.d)
o = torch.empty(PSZ, st.d)
st.store.load(0, e, o)
ids0 = rows_of(0)
chk(q_equals_E(ids0, e[(ids0 - 0).cpu()]), "B1 right after init, Q == quantize(E) (partition 0, non-S)")

st.load(0)
st.load(1)
st.wmask = st.resident_mask()
ids_ns = rows_of(0)[:50]
s_in = st.s_ids[(data.part[st.s_ids] == 0)][:1]
st.begin_batch(None)
x = st.read(torch.cat([ids_ns, s_in]))
(x * 3.0).sum().backward()
st.step(None)
st._join_updates()
E_now = st.window_slice(0).cpu()
ids_all = rows_of(0)
qS_before = rep.Q[rep.rmap[s_in].long().cpu()].clone()
vals = st.window_slice(0)
keep = st.srow[torch.arange(0, len(E_now), device=dev)] < 0
rep.refresh_values(torch.arange(0, len(E_now), device=dev)[keep], vals[keep])
st.evict(0)
st.flush_evicts()
chk(
    q_equals_E(ids_all, E_now[ids_all.cpu()]),
    "B2 right after eviction, non-S Q == quantize(latest E) (including 50 updated rows)",
)
chk(torch.equal(rep.Q[rep.rmap[s_in].long().cpu()], qS_before), "B2 S rows are excluded from the eviction refresh of Q")

st.load(2)
st.wmask = st.resident_mask()
Q_snap = rep.Q[rep.rmap[ids_all].long().cpu()].clone()
ids2 = rows_of(2)[:50]
st.begin_batch(None)
x = st.read(ids2)
(x * 1.5).sum().backward()
st.step(None)
st._join_updates()
chk(
    torch.equal(rep.Q[rep.rmap[ids_all].long().cpu()], Q_snap),
    "B3 Q of out-of-window partitions unchanged while other windows train",
)
st.evict(2)
st.flush_evicts()
st.join_writer() if hasattr(st, "join_writer") else None
st.load(0)
chk(
    torch.equal(st.window_slice(0).cpu(), E_now),
    "B3 E of reloaded partition 0 == value written at eviction (disk round trip)",
)

v = int(s_in[0])
j = int(st.srow[v])
slot = int(st.slot_of[0])
row = slot * PSZ + v
st.E[row] = 9999.0
got = st.read(torch.tensor([v], device=dev))[0].detach().cpu()
chk(
    torch.allclose(got, st.s_E()[j].cpu()) and float(got[0]) != 9999.0,
    "B4 reads: S original is read even with a corrupted window slot copy",
)
st.sync()
full = st.full_E(host=True)
chk(
    torch.allclose(full[v], st.s_E()[j].cpu()),
    "B4 eval merge: S rows of full_E == S original (takes precedence over disk and slot copies)",
)
print("FREEZE/CONSISTENCY", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)

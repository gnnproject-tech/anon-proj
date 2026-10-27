import os, sys, torch

sys.path.insert(0, os.getcwd())
import ars.train.loop as loop, ars.run as run

mw = int(sys.argv[1]) if len(sys.argv) > 1 else 400
ok = True
got = {}


def chk(c, m):
    global ok
    ok &= bool(c)
    print(("✓ " if c else "✗ ") + m)


orig_run = loop.run


def run_w(cfg, data, state, model, sampler, sched, log):
    orig_run(cfg, data, state, model, sampler, sched, log)
    state.sync()
    torch.cuda.synchronize()
    got.update(state=state, data=data, log=log)


loop.run = run_w
run.loop.run = run_w
sys.argv = [
    "ars.run",
    "--dataset",
    "collab",
    "--cond",
    "C1n",
    "--sched",
    "seq",
    "--paged",
    "true",
    "--window_device",
    "cpu",
    "--host_stale",
    "0",
    "--replica",
    "int8",
    "--replica_frac",
    "1.0",
    "--nbr_grad",
    "none",
    "--always_frac",
    "0.05",
    "--always_sel",
    "deg",
    "--always_device",
    "cuda",
    "--state_device",
    "cuda",
    "--epochs",
    "2",
    "--max_windows",
    str(mw),
    "--eval_queries",
    "2000",
    "--seed",
    "0",
    "--out",
    f"results/test_epoch_boundary_{os.getpid()}.json",
]
run.main()
st, data, log = got["state"], got["data"], got["log"]
rep = st.replica
dev = st.dev
PSZ = st.PSZ


def quant(e):
    s = e.abs().amax(1).clamp(min=1e-8) / 127
    return torch.round(e / s[:, None]).to(torch.int8), s


bad = 0
lsb = 0
checked = 0
e = torch.empty(PSZ, st.d)
o = torch.empty(PSZ, st.d)
for p in range(st.P):
    if int(st.slot_of[p]) >= 0:
        continue
    lo, hi = p * PSZ, min((p + 1) * PSZ, data.N)
    st.store.load(p, e, o)
    ids = torch.arange(lo, hi, device=dev)
    ns = (st.srow[ids] < 0).cpu()
    idl = ids.cpu()[ns]
    q, s = quant(e[: hi - lo][ns].to(dev))
    c = rep.rmap[idl.to(dev)].long()
    Qc = rep.Q[c] if rep.Q.is_cuda else rep.Q[c.cpu()].to(dev)
    Sc = rep.S[c] if rep.S.is_cuda else rep.S[c.cpu()].to(dev)
    dq = (Qc.int() - q.int()).abs()
    bad += int((dq.amax(1) > 1).sum())
    lsb += int(((dq.amax(1) == 1)).sum())
    bad += int((~torch.isclose(Sc, s)).sum())
    checked += int(ns.sum())
chk(
    bad == 0,
    f"(1) after 2 epochs, {checked:,} out-of-window non-S nodes: Q == quantize(disk E) (mismatches beyond 1 LSB {bad}; rounding-boundary 1 LSB differences {lsb} rows)",
)

v = int(st.s_ids[0])
p = int(data.part[v])
lo = p * PSZ
st.store.load(p, e, o)
e[v - lo] = 4321.0
o[v - lo] = 1.0
st.store.store(p, e, o)
if int(st.slot_of[p]) >= 0:
    st.E[int(st.slot_of[p]) * PSZ + (v - lo)] = 8765.0
full = st.full_E(host=True)
chk(
    torch.allclose(full[v], st.s_E()[0].cpu()) and float(full[v][0]) not in (4321.0, 8765.0),
    "(2) S rows of full_E == S original even with corrupted disk and slot copies",
)

c1, c2 = log["epochs"][0]["counters"], log["epochs"][1]["counters"]
chk(
    all(
        c2[k] > 0 for k in ("nbr_occ", "nbr_occ_out", "uniq_rows", "upd_S_in", "upd_S_out", "upd_W")
    )
    and abs(c2["nbr_occ"] - c1["nbr_occ"]) < 0.2 * c1["nbr_occ"],
    f"(3) epoch-2 counters recorded with consistent magnitude (nbr_occ {c1['nbr_occ']:,} / {c2['nbr_occ']:,}, capture {c2['nbr_occ_out_S']/max(1,c2['nbr_occ_out']):.3f})",
)
chk(
    "init_io_read_bytes" in log and log["init_io_read_bytes"] > 0,
    f"initialization I/O recorded separately ({log.get('init_io_read_bytes',0)/1e6:.1f} MB)",
)
print("EPOCH-BOUNDARY", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)

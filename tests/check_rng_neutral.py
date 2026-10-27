import sys, os, torch

if sys.argv[1] == "--compare":

    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    ref = torch.load(sys.argv[4]) if len(sys.argv) > 4 else None
    ok = True

    def chk(c, m):
        global ok
        ok &= bool(c)
        print(("✓ " if c else "✗ ") + m)

    chk(
        len(a["batches"]) == len(b["batches"])
        and all(torch.equal(x[k], y[k]) for x, y in zip(a["batches"], b["batches"]) for k in x),
        f"{len(a['batches'])} batches: positive edges, negatives, and sampled neighbor ids identical",
    )
    chk(
        torch.equal(a["rng_cpu"], b["rng_cpu"]) and torch.equal(a["rng_cuda"], b["rng_cuda"]),
        "CPU and CUDA RNG states identical after the run",
    )
    dE = float((a["E"] - b["E"]).abs().max())
    dS = float((a["S_E"] - b["S_E"]).abs().max())
    dO = float((a["S_O"] - b["S_O"]).abs().max())
    dL = abs(a["loss"] - b["loss"])
    if ref is None:
        chk(
            dE <= 1e-4 and dS <= 1e-4 and dO <= 1e-4,
            f"embedding and S state match (atol 1e-4): max|dE| {dE:.2e} max|dS_E| {dS:.2e} max|dS_O| {dO:.2e}",
        )
        chk(
            dL <= 1e-5 * max(1.0, abs(a["loss"])),
            f"epoch loss matches ({a['loss']:.1f} vs {b['loss']:.1f})",
        )
    else:
        rE = float((a["E"] - ref["E"]).abs().max())
        rS = float((a["S_E"] - ref["S_E"]).abs().max())
        rO = float((a["S_O"] - ref["S_O"]).abs().max())
        rL = abs(a["loss"] - ref["loss"])
        chk(
            dE <= 2 * rE + 1e-6 and dS <= 2 * rS + 1e-6 and dO <= 2 * rO + 1e-6,
            f"numeric differences within repeat-run range: new max|dE| {dE:.2e} (repeat {rE:.2e}), |dS_E| {dS:.2e} ({rS:.2e}), |dS_O| {dO:.2e} ({rO:.2e})",
        )
        chk(
            dL <= 2 * rL + 1e-3,
            f"epoch loss difference within repeat-run range: {dL:.2f} (repeat {rL:.2f}); H@50 {a['h50']:.4f}/{b['h50']:.4f}/{ref['h50']:.4f}",
        )
    print("RNG-NEUTRAL", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
out = sys.argv[1]
mw = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NB = 40
sys.path.insert(0, os.getcwd())
import ars.graph.csr as csr, ars.train.step as step, ars.train.loop as loop, ars.run as run

rec = {"batches": []}
nb = [0]
orig_sample = csr.Sampler.sample


def sample(self, U, allowed, K=None):
    sel, cnt = orig_sample(self, U, allowed, K)
    if torch.is_grad_enabled() and nb[0] < NB:
        rec["batches"][-1]["sel"] = sel.detach().cpu().clone()
        rec["batches"][-1]["cnt"] = cnt.detach().cpu().clone()
    return sel, cnt


csr.Sampler.sample = sample
orig_tb = step.train_batch


def tb(cfg, data, state, model, src, rel, dst, nb_allowed, wmask, pool):
    if nb[0] < NB:
        rec["batches"].append(
            {"src": src.cpu().clone(), "dst": dst.cpu().clone(), "rel": rel.cpu().clone()}
        )
    r = orig_tb(cfg, data, state, model, src, rel, dst, nb_allowed, wmask, pool)
    nb[0] += 1
    return r


step.train_batch = tb
loop.train_batch = tb
orig_randint = torch.randint


def randint(*a, **k):
    r = orig_randint(*a, **k)
    if (
        torch.is_grad_enabled()
        and nb[0] < NB
        and rec["batches"]
        and r.dim() == 2
        and "negs" not in rec["batches"][-1]
        and r.is_cuda
    ):
        rec["batches"][-1]["negs"] = r.cpu().clone()
    elif torch.is_grad_enabled() and nb[0] < NB and rec["batches"] and r.dim() == 2 and r.is_cuda:
        rec["batches"][-1]["negs2"] = r.cpu().clone()
    return r


torch.randint = randint
orig_run = loop.run


def run_w(cfg, data, state, model, sampler, sched, log):
    orig_run(cfg, data, state, model, sampler, sched, log)
    state.sync()
    torch.cuda.synchronize()
    rec["E"] = state.full_E(host=True).clone()
    rec["S_E"] = state.s_E().cpu().clone()
    rec["S_O"] = state.s_O().cpu().clone()
    rec["rng_cpu"] = torch.random.get_rng_state().clone()
    rec["rng_cuda"] = torch.cuda.get_rng_state().clone()
    e = log["epochs"][-1]
    rec["loss"], rec["h50"], rec["mrr"] = e["loss"], e["h50"], e["mrr"]
    torch.save(rec, out)


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
    "1",
    "--max_windows",
    str(mw),
    "--eval_queries",
    "2000",
    "--seed",
    "0",
    "--out",
    out + ".json",
]
run.main()

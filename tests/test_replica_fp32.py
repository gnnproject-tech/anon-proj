import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ars.storage.replica import Replica

KEEP = []


def run(dev, home, frac):
    N, d = 5000, 16
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    deg = torch.randint(1, 100, (N,), generator=g).to(dev)
    reps = {m: Replica(N, d, m, frac, 0.0, deg, dev, home=home) for m in ("int8", "fp32")}
    KEEP.append(reps)
    ok = torch.equal(reps["int8"].has, reps["fp32"].has) and torch.equal(
        reps["int8"].rmap, reps["fp32"].rmap
    )
    ok &= reps["fp32"].bytes == 4 * reps["int8"].bytes
    truth = torch.zeros(N, d, device=dev)
    for step in range(5):
        idx = torch.randperm(N, generator=g)[:700].to(dev)
        vals = torch.randn(700, d, generator=g).to(dev) * (step + 1)
        for r in reps.values():
            r.refresh_values(idx, vals)
        truth[idx] = vals
    q = torch.randint(0, N, (1200,), generator=g).to(dev)
    cov = reps["fp32"].has[q]
    q = q[cov]
    out32 = reps["fp32"].read(q)
    out8 = reps["int8"].read(q)
    if dev == "cuda":
        torch.cuda.synchronize()
    ex = torch.equal(out32, truth[q])
    err8 = (out8 - truth[q]).abs().max().item()
    bound = (truth[q].abs().amax(1) / 254 + 1e-6).max().item()
    ok &= ex and bool(
        ((out8 - truth[q]).abs() <= truth[q].abs().amax(1, keepdim=True) / 254 + 1e-6).all()
    )
    print(
        f"  dev {dev} home {home} frac {frac}: fp32 bitwise {ex}, int8 max err {err8:.2e} (row bound {bound:.2e}), covered rows equal {torch.equal(reps['int8'].has, reps['fp32'].has)}, bytes 4x {reps['fp32'].bytes == 4 * reps['int8'].bytes} ->",
        "PASS" if ok else "FAIL",
    )
    return ok


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    ok = run(dev, None, 1.0) and run(dev, None, 0.3)
    if dev == "cuda":
        ok &= run(dev, "cpu", 1.0) and run(dev, "cpu", 0.3)
    print("REPLICA-FP32", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

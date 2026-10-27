import json, subprocess, sys, os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
PY = sys.executable
os.makedirs(f"{ROOT}/results", exist_ok=True)
CASES = {
    "B": "--cond B --sched seq",
    "deferred": "--cond C1n --sched seq --replica int8 --nbr_grad deferred --gbuf int8",
}


def run(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)


ep = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ok = True
for name, args in CASES.items():
    mo, po = f"{ROOT}/results/p1_mem_{name}.json", f"{ROOT}/results/p1_paged_{name}.json"
    r1 = (
        run(f"{PY} -m ars.run {args} --negs_shared true --epochs {ep} --out {mo}")
        if not os.path.exists(mo)
        else None
    )
    r2 = run(f"{PY} -m ars.run {args} --negs_shared true --paged true --epochs {ep} --out {po}")
    if (r1 and r1.returncode) or r2.returncode:
        print(name, "failed", (r1.stderr[-400:] if r1 else ""), r2.stderr[-900:])
        ok = False
        continue
    m = json.load(open(mo))["epochs"][-1]
    p = json.load(open(po))["epochs"][-1]
    dh, dm = abs(m["h50"] - p["h50"]), abs(m["mrr"] - p["mrr"])
    good = dh <= 0.02 and dm <= 0.01
    PSZ, d = 3686, 50
    part = PSZ * d * 4
    nwin = 2016
    print(
        f"{name:9s} mem {m['h50']:.4f}/{m['mrr']:.4f} | paged {p['h50']:.4f}/{p['mrr']:.4f} | Δ {dh:.4f}/{dm:.4f} {'✓' if good else '✗'}"
    )
    print(
        f"          I/O per epoch read {p['io_read_bytes']/1e9:.2f} GB ({p['io_reads']} partitions) write {p['io_write_bytes']/1e9:.2f} GB | expected (sequential, one swap per window): {(nwin*2*part)/1e9:.2f} GB | VmHWM {p['vm_hwm_bytes']/1e9:.2f} GB"
    )
    ok &= good
print("PAGED", "PASS" if ok else "FAIL")

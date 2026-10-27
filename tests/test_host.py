import json, subprocess, sys, os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
PY = sys.executable
os.makedirs(f"{ROOT}/results", exist_ok=True)
A = "--cond C1n --sched seq --replica int8 --nbr_grad deferred --gbuf int8 --paged true"


def run(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)


ep = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ref = f"{ROOT}/results/p1_paged_deferred.json"
out = f"{ROOT}/results/p2_host_deferred.json"
r = run(f"{PY} -m ars.run {A} --state_device cpu --epochs {ep} --out {out}")
if r.returncode:
    print("failed", r.stderr[-1200:])
    print("HOST FAIL")
    sys.exit(1)
a = json.load(open(ref))["epochs"][-1]
b = json.load(open(out))["epochs"][-1]
dh, dm = abs(a["h50"] - b["h50"]), abs(a["mrr"] - b["mrr"])
good = dh <= 0.02 and dm <= 0.01
print(
    f"deferred  device {a['h50']:.4f}/{a['mrr']:.4f} ({a['sec']}s/ep) | host {b['h50']:.4f}/{b['mrr']:.4f} ({b['sec']}s/ep) | Δ {dh:.4f}/{dm:.4f} {'✓' if good else '✗'}"
)
print(
    f"          per-epoch copy gather {b.get('rep_gather_bytes',0)/1e6:.0f} MB, buffer transfer {b.get('gbuf_xfer_bytes',0)/1e6:.0f} MB, disk read {b['io_read_bytes']/1e9:.2f} GB"
)
print("HOST", "PASS" if good else "FAIL")

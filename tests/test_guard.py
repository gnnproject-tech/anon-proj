import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import torch
from ars.storage.gradbuf import GradBuf

N, d = 64, 8
deg = torch.arange(N)


def mk(acc):
    return GradBuf(N, d, "int8", 1.0, 0.0, deg, "cpu", home="cpu", acc_dtype=acc)


ok = True
g = mk("fp16")
idx = torch.arange(N)
for _ in range(3):
    g.add(idx, torch.randn(N, d))
g.window_start()
print("normal fp16 passes amax_acc=%.2f" % g.win_stats[-1][3])
g = mk("fp16")
for _ in range(10):
    g.add(torch.tensor([3] * 32), torch.full((32, d), 300.0))
try:
    g.window_start()
    print("FAIL fp16 overflow not detected")
    ok = False
except RuntimeError as e:
    print("OK fp16 overflow detected:", str(e)[:120])
    ok &= "accumulator" in str(e)
g = mk("fp32")
for _ in range(10):
    g.add(torch.tensor([3] * 32), torch.full((32, d), 300.0))
g.window_start()
print("OK fp32 passes the same input amax_acc=%.0f" % g.win_stats[-1][3])
g = mk("fp16")
g.add(idx, torch.randn(N, d))
x = torch.randn(N, d)
x[5, 2] = float("nan")
g.add(idx, x)
try:
    g.window_start()
    print("FAIL NaN input not detected")
    ok = False
except RuntimeError as e:
    print("OK NaN input detected:", str(e)[:120])
    ok &= "input_grad" in str(e) and "first_bad_batch=1" in str(e)
g = GradBuf(N, d, "int8", 0.5, 0.0, deg, "cpu", home="cpu", acc_dtype="fp16")
x = torch.randn(N, d)
x[0, 0] = float("inf")
try:
    g.add(idx, x)
    g.window_start()
    print("FAIL inf in a row outside the buffer not detected")
    ok = False
except RuntimeError as e:
    print("OK inf in a row outside the buffer detected by the source check:", str(e)[:60])
g = mk("fp16")
g.add(idx, torch.randn(N, d))
g.window_start()
g.add(idx, torch.randn(N, d))
g.window_start()
g.window_start()
idx2, v = g.take(idx[:10])
print(
    "delay histogram sum after take",
    int(g.wait_hist_first.sum()),
    "first reset",
    int((g.first[:10] == -1).sum()),
)
print("reuse after reset passes, win_stats", len(g.win_stats))
print("GUARD", "PASS" if ok else "FAIL")

import json, sys


def load_mem(p):
    rows = []
    for l in open(p).read().splitlines()[1:]:
        t, el, rss, hwm, gpu = l.split("\t")
        rows.append((t[:8], float(el), int(rss) / 2**20, (int(gpu) if gpu.isdigit() else 0) / 1024))
    return rows


def phases(rows, in_eval, first_end, init_end=0, last_end=None):
    m = {"init": [0, 0], "train1": [0, 0], "eval": [0, 0], "train_post": [0, 0]}
    for t, el, rss, gpu in rows:
        if el < init_end:
            k = "init"
        elif last_end is not None and el > last_end:
            continue
        elif el < first_end:
            k = "train1"
        elif in_eval(t, el):
            k = "eval"
        else:
            k = "train_post"
        m[k][0] = max(m[k][0], rss)
        m[k][1] = max(m[k][1], gpu)
    return m


def ours(name):
    ep = json.load(open(f"results/{name}.json"))["epochs"]
    rows = load_mem(f"results/{name}.mem")
    total = rows[-1][1]
    init = max(0, total - sum(e["sec"] for e in ep))
    wins = []
    cur = init
    for e in ep:
        wins.append((cur + e["sec_train"] - 90, cur + e["sec"] + 60))
        cur += e["sec"]
    return phases(
        rows, lambda t, el: any(a <= el <= b for a, b in wins), wins[0][0], init, wins[-1][1]
    )


def marius(name):
    L = open(f"results/{name}.log").read().splitlines()
    ts = lambda l: l[10:18]
    wins = []
    start = None
    for l in L:
        if "Epoch Runtime" in l:
            start = ts(l)
        elif "Starting training epoch" in l and start:
            wins.append((start, ts(l)))
            start = None
        elif "Evaluation complete" in l and start:
            last = ts(l)
    if start:
        wins.append((start, last))
    rows = load_mem(f"results/{name}.mem")
    first = next(el for t, el, _, _ in rows if t >= wins[0][0])
    t0 = next((ts(l) for l in L if "Starting training epoch 1" in l), None)
    init = next((el for t, el, _, _ in rows if t0 and t >= t0), 0)
    last = next((el for t, el, _, _ in rows if t >= wins[-1][1]), None)
    return phases(rows, lambda t, el: any(s <= t <= e for s, e in wins), first, init, last)


if __name__ == "__main__":
    print(
        "| run | host: init | first-epoch train | eval peak | post-eval train | GPU: init | first-epoch train | eval peak | post-eval train |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for a in sys.argv[1:]:
        m = marius(a) if a.startswith("mgnn") else ours(a)
        print(
            f"| {a} | {m['init'][0]:.1f} | {m['train1'][0]:.1f} | {m['eval'][0]:.1f} | {m['train_post'][0]:.1f} | {m['init'][1]:.1f} | {m['train1'][1]:.1f} | {m['eval'][1]:.1f} | {m['train_post'][1]:.1f} |"
        )

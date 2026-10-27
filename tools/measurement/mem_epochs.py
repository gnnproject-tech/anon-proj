import json, sys

sys.path.insert(0, "tools")
from mem_phases import load_mem


def epochs(name):
    ep = json.load(open(f"results/{name}.json"))["epochs"]
    rows = load_mem(f"results/{name}.mem")
    total = rows[-1][1]
    cur = max(0, total - sum(e["sec"] for e in ep))
    out = []
    for e in ep:
        tr = (cur, cur + e["sec_train"] - 90)
        ev = (tr[1], cur + e["sec"] + 60)
        cur += e["sec"]
        t = [(r, g) for _, el, r, g in rows if tr[0] <= el < tr[1]]
        v = [(r, g) for _, el, r, g in rows if ev[0] <= el <= ev[1]]
        out.append(
            (
                max(r for r, _ in t) if t else 0,
                max(g for _, g in t) if t else 0,
                max(r for r, _ in v) if v else 0,
                e["h50"],
                e["mrr"],
                e["sec_train"] / 60,
            )
        )
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(
            f"\n**{name}** (host training / GPU training / host evaluation peak, GiB)\n\n| epoch | "
            + " | ".join(str(i + 1) for i in range(len(epochs(name))))
            + " |\n|---|"
            + "---|" * len(epochs(name))
        )
        E = epochs(name)
        for lab, f in (
            ("host train", lambda x: f"{x[0]:.1f}"),
            ("GPU train", lambda x: f"{x[1]:.1f}"),
            ("host eval peak", lambda x: f"{x[2]:.1f}"),
            ("MRR", lambda x: f"{x[4]:.3f}"),
            ("train min", lambda x: f"{x[5]:.0f}"),
        ):
            print(f"| {lab} | " + " | ".join(f(x) for x in E) + " |")

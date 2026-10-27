import sys, re, collections

pid = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
cur = None
rows = []
for ln in open(f"/proc/{pid}/smaps"):
    m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s*(.*)$", ln)
    if m:
        cur = [m.group(4).strip() or "[anon]", m.group(3), 0, 0]
        rows.append(cur)
        continue
    if cur and ln.startswith("Rss:"):
        cur[2] = int(ln.split()[1])
    if cur and ln.startswith("Locked:"):
        cur[3] = int(ln.split()[1])
tot = sum(r[2] for r in rows)
print(f"pid {pid}: Rss total {tot/2**20:.2f} GiB, {len(rows)} mappings")
for r in sorted(rows, key=lambda r: -r[2])[:n]:
    print(f"  {r[2]/2**20:7.2f} GiB  {r[1]}  {r[0][:70]}  locked {r[3]/2**20:.2f}")
g = collections.Counter()
for r in rows:
    g[
        (
            "anon"
            if r[0] == "[anon]"
            else ("nvidia" if "nvidia" in r[0] else ("heap" if "heap" in r[0] else "file/other"))
        )
    ] += r[2]
print("  groups:", {k: f"{v/2**20:.2f} GiB" for k, v in g.items()})

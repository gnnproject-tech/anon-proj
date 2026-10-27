import os, sys, time

pid0 = int(sys.argv[1])
out = sys.argv[2]
iv = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
CLK = os.sysconf("SC_CLK_TCK")


def tree(root):
    kids = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                st = f.read()
            pp = int(st[st.rindex(")") + 2 :].split()[1])
        except Exception:
            continue
        kids.setdefault(pp, []).append(int(p))
    res, q = [], [root]
    while q:
        x = q.pop()
        res.append(x)
        q += kids.get(x, [])
    return res


def read(pids):
    cpu = rb = wb = rss = 0
    for p in pids:
        try:
            with open(f"/proc/{p}/stat") as f:
                st = f.read()
            fs = st[st.rindex(")") + 2 :].split()
            cpu += int(fs[11]) + int(fs[12])
            rss += int(fs[21]) * 4096
            with open(f"/proc/{p}/io") as f:
                for ln in f:
                    if ln.startswith("read_bytes:"):
                        rb += int(ln.split()[1])
                    elif ln.startswith("write_bytes:"):
                        wb += int(ln.split()[1])
        except Exception:
            pass
    return cpu, rb, wb, rss


def disk():
    s = 0
    with open("/proc/diskstats") as f:
        for ln in f:
            fs = ln.split()
            if (
                fs[2].startswith(("loop", "ram"))
                or fs[2][-1].isdigit()
                and fs[2][:-1].rstrip("p") in ("nvme0n", "sd")
            ):
                continue
            s += int(fs[5]) * 512
    return s


with open(out, "w") as fo:
    fo.write("t\tnproc\tcpu_cores\tread_bytes\twrite_bytes\trss_bytes\tdisk_read_bytes\n")
    prev = None
    while os.path.exists(f"/proc/{pid0}"):
        pids = tree(pid0)
        cpu, rb, wb, rss = read(pids)
        dk = disk()
        now = time.time()
        if prev:
            dt = now - prev[0]
            cores = (cpu - prev[1]) / CLK / dt if dt > 0 else 0
            fo.write(f"{now:.0f}\t{len(pids)}\t{cores:.2f}\t{rb}\t{wb}\t{rss}\t{dk}\n")
            fo.flush()
        prev = (now, cpu)
        time.sleep(iv)

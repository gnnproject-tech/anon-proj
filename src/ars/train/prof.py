import time, torch
from contextlib import contextmanager


class Prof:
    def __init__(self, on, dev):
        self.on = bool(on)
        self.cuda = self.on and str(dev).startswith("cuda")
        self.t = {}
        self.n = {}

    def _now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    @contextmanager
    def __call__(self, name):
        if not self.on:
            yield
            return
        t0 = self._now()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + self._now() - t0
            self.n[name] = self.n.get(name, 0) + 1

    def pop(self):
        r = {k: round(v, 1) for k, v in self.t.items()}
        self.t = {}
        self.n = {}
        return r


class MemMon:
    def __init__(self, dev):
        import threading

        self.cuda = str(dev).startswith("cuda")
        self.phase = "init"
        self.rss = {}
        self.lock = threading.Lock()
        self.stop = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    @staticmethod
    def _vmrss():
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS"):
                return int(ln.split()[1]) * 1024
        return 0

    def _run(self):
        while not self.stop:
            r = self._vmrss()
            with self.lock:
                self.rss[self.phase] = max(self.rss.get(self.phase, 0), r)
            time.sleep(0.5)

    def begin(self, phase):
        done = self.end()
        with self.lock:
            self.phase = phase
            self.rss[phase] = self._vmrss()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return done

    def end(self):
        with self.lock:
            p = self.phase
            r = self.rss.get(p, self._vmrss())
        out = dict(phase=p, rss_max=r)
        if self.cuda:
            torch.cuda.synchronize()
            out["gpu_alloc_max"] = torch.cuda.max_memory_allocated()
            out["gpu_reserved_max"] = torch.cuda.max_memory_reserved()
        return out

import torch


class Replica:
    def __init__(self, N, d, mode, frac, delta, deg, dev, home=None):
        self.mode, self.delta, self.dev, self.N, self.d = mode, delta, dev, N, d
        self.view = mode in ("int8", "fp32")
        self.qdtype = torch.int8 if mode == "int8" else torch.float32
        self.esz = 1 if mode == "int8" else 4
        self.home = home or dev
        host = self.home == "cpu"
        self.has = torch.ones(N, dtype=torch.bool, device=dev)
        if frac < 1.0:
            k = int(N * frac)
            self.has[:] = False
            self.has[torch.topk(deg, k).indices] = True
        rows = torch.nonzero(self.has).squeeze(1)
        k = len(rows)
        self.k = k
        self.rmap = torch.full((N,), -1, dtype=torch.int32, device=dev)
        self.rmap[rows] = torch.arange(k, device=dev, dtype=torch.int32)
        from .pinned import pin_exact, PinnedArena

        self.Q = (
            pin_exact((k, d), self.qdtype)
            if host
            else torch.zeros(k, d, dtype=self.qdtype, device=dev)
        )
        self.S = pin_exact((k,)) if host else torch.ones(k, device=dev)
        if host:
            self.S.fill_(1.0)
            self.stage = PinnedArena()
            self.stage_n = 0
        self.snap = None
        self.last = None
        self.stat = [0, 0]
        self.cache_map = None
        self.bytes = k * d * self.esz

    def _quant(self, rows):
        if self.mode == "fp32":
            return rows.float(), torch.ones(rows.shape[0], device=rows.device)
        s = rows.abs().amax(1).clamp(min=1e-8) / 127
        return torch.round(rows / s[:, None]).to(torch.int8), s

    @torch.no_grad()
    def refresh(self, idx, E, force=False):
        if not self.view:
            return
        if self.delta > 0 and not force:
            if self.last is None:
                self.last = E.detach().clone()
            rows, last = E[idx], self.last[idx]
            keep = (rows - last).norm(dim=1) / (last.norm(dim=1) + 1e-8) > self.delta
            self.stat[0] += int(keep.sum())
            self.stat[1] += len(idx)
            idx = idx[keep]
            if len(idx) == 0:
                return
            self.last[idx] = E[idx]
        q, s = self._quant(E[idx])
        self._put(idx, q, s)

    def enable_gpu_cache(self, deg, frac):
        kc = min(int(self.N * frac), self.k)
        dm = deg.clone().float()
        dm[~self.has] = -1
        top = torch.topk(dm, kc).indices
        self.cache_map = torch.full((self.N,), -1, dtype=torch.int32, device=self.dev)
        self.cache_map[top] = torch.arange(kc, device=self.dev, dtype=torch.int32)
        self.cQ = torch.zeros(kc, self.d, dtype=self.qdtype, device=self.dev)
        self.cS = torch.ones(kc, device=self.dev)
        self.cache_rows = top
        self.cache_bytes = kc * self.d
        self.sync_cache_all()

    @torch.no_grad()
    def sync_cache_all(self):
        if self.cache_map is None:
            return
        c = self.rmap[self.cache_rows].long()
        ic = c.cpu() if self.home == "cpu" else c
        self.cQ.copy_(self.Q[ic].to(self.dev))
        self.cS.copy_(self.S[ic].to(self.dev))

    def _put(self, idx, q, s):
        c = self.rmap[idx].long()
        m = c >= 0
        if not bool(m.all()):
            idx, c, q, s = idx[m], c[m], q[m], s[m]
        if len(c) == 0:
            return
        if self.cache_map is not None:
            cr = self.cache_map[idx].long()
            mc = cr >= 0
            if bool(mc.any()):
                self.cQ[cr[mc]] = q[mc]
                self.cS[cr[mc]] = s[mc]
        if self.home == "cpu":
            c = c.cpu()
            q = q.cpu()
            s = s.cpu()
        self.Q[c] = q
        self.S[c] = s

    @torch.no_grad()
    def refresh_values(self, idx, vals, force=False):
        if not self.view:
            return
        if self.delta > 0 and not force:
            if self.last is None:
                self.last = torch.zeros(self.N, self.d, device=self.dev)
            last = self.last[idx]
            keep = (vals - last).norm(dim=1) / (last.norm(dim=1) + 1e-8) > self.delta
            self.stat[0] += int(keep.sum())
            self.stat[1] += len(idx)
            idx, vals = idx[keep], vals[keep]
            if len(idx) == 0:
                return
            self.last[idx] = vals
        q, s = self._quant(vals)
        self._put(idx, q, s)

    def snapshot(self, E):
        self.snap = E.detach().clone()

    def read(self, idx):
        if not self.view:
            return self.snap[idx]
        c = self.rmap[idx].clamp(min=0).long()
        if self.home == "cpu":
            if self.cache_map is not None:
                cr = self.cache_map[idx].long()
                hit = cr >= 0
                out = torch.empty(idx.shape + (self.d,), device=self.dev)
                if hit.any():
                    out[hit] = self._deq(self.cQ[cr[hit]], self.cS[cr[hit]])
                miss = ~hit
                if miss.any():
                    im = c[miss].cpu()
                    q = self.Q[im]
                    s = self.S[im]
                    self.gather_bytes = (
                        getattr(self, "gather_bytes", 0) + q.numel() * self.esz + s.numel() * 4
                    )
                    self.idx_bytes = getattr(self, "idx_bytes", 0) + im.numel() * 8
                    out[miss] = self._deq(
                        q.to(self.dev, non_blocking=True), s.to(self.dev, non_blocking=True)
                    )
                self.hits = getattr(self, "hits", 0) + int(hit.sum())
                self.reads = getattr(self, "reads", 0) + hit.numel()
                return out
            return self._host_read(c)
        return self._deq(self.Q[c], self.S[c])

    def _deq(self, q, s):
        return q if self.mode == "fp32" else q.float() * s.unsqueeze(-1)

    def _host_read(self, c):
        import threading

        n = c.numel()
        ev = getattr(self, "stage_ev", None)
        if ev is not None:
            ev.synchronize()
        if self.stage_n < n:
            cap = max(n, 2 * self.stage_n, 1 << 17)
            self.st_idx = self.stage.alloc((cap,), torch.int64)
            self.st_q = self.stage.alloc((cap, self.d), self.qdtype)
            self.st_s = self.stage.alloc((cap,))
            self.stage_n = cap
        ic = self.st_idx[:n]
        ic.copy_(c)
        self.idx_bytes = getattr(self, "idx_bytes", 0) + n * 8
        q, s = self.st_q[:n], self.st_s[:n]
        k = 4 if n >= 8192 else 1
        errs = []

        def part(i):
            try:
                sl = slice(i * n // k, (i + 1) * n // k)
                torch.index_select(self.Q, 0, ic[sl], out=q[sl])
                torch.index_select(self.S, 0, ic[sl], out=s[sl])
            except BaseException as ex:
                errs.append(ex)

        if k == 1:
            part(0)
        else:
            ths = [threading.Thread(target=part, args=(i,)) for i in range(k)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
        if errs:
            raise RuntimeError("host view gather thread failed") from errs[0]
        self.gather_bytes = getattr(self, "gather_bytes", 0) + q.numel() * self.esz + s.numel() * 4
        qd = q.to(self.dev, non_blocking=True)
        sd = s.to(self.dev, non_blocking=True)
        self.stage_ev = torch.cuda.Event()
        self.stage_ev.record()
        return self._deq(qd, sd)

    def pop_stat(self):
        r = self.stat[0] / max(1, self.stat[1]) if self.stat[1] else None
        self.stat = [0, 0]
        return r

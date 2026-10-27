import math, torch
from .partfile import PartitionStore


class PagedState:
    def __init__(self, cfg, data, dev, root, direct):
        self.cfg, self.data, self.dev = cfg, data, dev
        N, d, P, PSZ = data.N, cfg.d, data.P, data.PSZ
        self.N, self.d, self.P, self.PSZ = N, d, P, PSZ
        self.store = PartitionStore(root, P, PSZ, d, direct)

        def init_part(p):
            g = torch.Generator(device="cpu")
            g.manual_seed(cfg.seed * 1_000_003 + p)
            if cfg.init == "e_marius":
                lim = math.sqrt(6 / (N + d))
                return (torch.rand(PSZ, d, generator=g) * 2 - 1) * lim
            return torch.randn(PSZ, d, generator=g) * (0.03 if cfg.init == "mid" else 0.1)

        if cfg.init_scope == "global":
            g = torch.Generator(device="cpu")
            g.manual_seed(cfg.seed)
            if cfg.init == "e_marius":
                lim = math.sqrt(6 / (N + d))
                E0 = (torch.rand(N, d, generator=g) * 2 - 1) * lim
            else:
                E0 = torch.randn(N, d, generator=g) * (0.03 if cfg.init == "mid" else 0.1)
            self.store.init_from(E0, torch.zeros(N, d))
            del E0
        else:
            self.store.init_parts(init_part, N)

        self.c = None
        self.slot_of = torch.full((P,), -1, dtype=torch.long, device=dev)
        self.part_in_slot = []
        self.host_emb = None
        self.host_opt = None
        self.E = None
        self.O = None
        self.wmask = None
        self.replica = None
        self.gbuf = None
        self.pending = []
        self.lr, self.eps = cfg.lr_emb, 1e-10
        self.hwm = 0
        self._cnt_init()

    def _cnt_init(self):
        z = lambda: torch.zeros((), dtype=torch.long, device=self.dev)
        self.cnt = {
            k: z()
            for k in (
                "nbr_occ",
                "nbr_occ_out",
                "nbr_occ_out_S",
                "uniq_rows",
                "uniq_S",
                "uniq_W",
                "uniq_Q",
                "upd_S_in",
                "upd_S_out",
                "grad_S_out",
                "upd_W",
                "gath_W_rows",
                "gath_S_rows",
            )
        }
        self.cnt_bytes = {k: 0 for k in ("xfer_refresh", "xfer_eval")}

    def count_nbrs(self, sel, valid):
        if not self.cfg.counters or not hasattr(self, "amask") or not torch.is_grad_enabled():
            return
        flat = sel.reshape(-1)
        v = valid[:, None].expand_as(sel).reshape(-1)
        out = v & ~self.amask[flat]
        self.cnt["nbr_occ"] += v.sum()
        self.cnt["nbr_occ_out"] += out.sum()
        if self.kS:
            self.cnt["nbr_occ_out_S"] += (out & (self.srow[flat] >= 0)).sum()

    def pop_counters(self):
        r = {k: int(v) for k, v in self.cnt.items()}
        r.update(self.cnt_bytes)
        if self.host_window:
            r["xfer_W_in"] = r["gath_W_rows"] * self.d * 4
            r["xfer_W_idx"] = (r["gath_W_rows"] + r["gath_S_rows"]) * 8
            r["xfer_W_grad"] = r["upd_W"] * (self.d * 4 + 8)
            if self.kS and self.s_dev == "cpu":
                r["xfer_S_in"] = r["gath_S_rows"] * self.d * 4
                r["xfer_S_grad"] = (r["upd_S_in"] + r["upd_S_out"]) * (self.d * 4 + 8)
        if self.replica is not None and hasattr(self.replica, "gather_bytes"):
            r["xfer_Q_in"] = self.replica.gather_bytes
            r["xfer_Q_idx"] = getattr(self.replica, "idx_bytes", 0)
            self.replica.idx_bytes = 0
        for v in self.cnt.values():
            v.zero_()
        for k in self.cnt_bytes:
            self.cnt_bytes[k] = 0
        return r

    def alloc(
        self,
        c,
        max_out=None,
        max_in=None,
        always_frac=0.0,
        always_sel="deg",
        always_seed=0,
        always_device="cpu",
    ):
        self.c = c
        d, PSZ = self.d, self.PSZ
        self.ev_pool = []
        self.pf_pool = []
        self.wdev = "cpu" if self.cfg.window_device == "cpu" else self.dev
        self.host_window = self.wdev == "cpu"
        self.kS = 0
        self.S_OFF = None
        self.s_dev = always_device
        if always_frac > 0:
            assert self.host_window, "S requires a host write window"
            self._select_always(always_frac, always_sel, always_seed)
        if self.host_window:

            from .pinned import pin_exact, PinnedArena

            self.h_out = max_out if max_out is not None else max(1, c // 2)
            self.h_in = max_in if max_in is not None else max(1, c // 2)
            self.nslots = c + self.h_out + self.h_in
            self.S_OFF = self.nslots * PSZ
            tail = self.kS if (self.kS and self.s_dev == "cpu") else 0
            self.E = pin_exact((self.nslots * PSZ + tail, d))
            self.O = pin_exact((self.nslots * PSZ + tail, d))
            if self.kS and self.s_dev == "cuda":
                self.SE = torch.zeros(self.kS, d, device=self.dev)
                self.SO = torch.zeros(self.kS, d, device=self.dev)
            self.s_bytes = 2 * self.kS * d * 4
            self.s_init_done = self.kS == 0
            self.sstate = [0] * self.nslots
            self.arena = PinnedArena()
            self.gbuf_n = 1 << 17
            self.gbuf_h = self.arena.alloc((self.gbuf_n, d))
            self.gidx_h = self.arena.alloc((self.gbuf_n,), torch.int64)
            self.upd = []
            self.upd_error = None
            self.stale = int(self.cfg.host_stale)
        else:
            self.nslots = c
            self.E = torch.zeros(c * PSZ, d, device=self.wdev)
            self.O = torch.zeros(c * PSZ, d, device=self.wdev)
        self.leaf = None
        self.leaf_rows = None
        self.leaf_map = None
        self.rep_leaf = None
        self.rep_ids = None
        self.part_in_slot = [-1] * self.nslots
        self.amask = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        self.evicted = []
        self.wb_thread = None
        self.pf = {}
        self.pf_threads = {}
        self.pf_error = {}
        self.writing = set()
        self.wb_error = None

    @torch.no_grad()
    def _select_always(self, frac, sel, seed):
        N, dev = self.N, self.dev
        kS = int(N * frac)
        assert kS >= 1, f"always_frac {frac} yields an empty S (N={N})"
        if sel == "deg":
            ids = torch.argsort(-self.data.deg.long(), stable=True)[:kS]
        elif sel == "random":
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            ids = torch.randperm(N, generator=g)[:kS].to(dev)
        else:
            raise ValueError(sel)
        self.s_ids = torch.sort(ids).values.long()
        self.s_ids_cpu = self.s_ids.cpu()
        self.srow = torch.full((N,), -1, dtype=torch.int32, device=dev)
        self.srow[self.s_ids] = torch.arange(kS, dtype=torch.int32, device=dev)
        assert int((self.srow >= 0).sum()) == kS
        self.kS = kS
        self.s_sel = sel
        self.s_map_bytes = N * 4 + 2 * kS * 8
        self.s_reads_t = torch.zeros((), dtype=torch.long, device=dev)
        self.s_upd_t = torch.zeros((), dtype=torch.long, device=dev)

    def global_of(self, lr):
        pis = torch.tensor(self.part_in_slot, device=self.dev)
        if self.kS:
            isS = lr >= self.S_OFF
            lw = torch.where(isS, torch.zeros_like(lr), lr)
            base = pis[lw // self.PSZ] * self.PSZ + lw % self.PSZ
            return torch.where(
                isS, self.s_ids[(lr - self.S_OFF).clamp(min=0, max=self.kS - 1)], base
            )
        return pis[lr // self.PSZ] * self.PSZ + lr % self.PSZ

    def rows(self, idx):
        p = self.data.part[idx].long()
        s = self.slot_of[p]
        r = torch.where(s >= 0, s * self.PSZ + (idx - p * self.PSZ), torch.full_like(idx, -1))
        if self.kS:
            # S remains authoritative when a node also has a window slot.
            sr = self.srow[idx].long()
            r = torch.where(sr >= 0, self.S_OFF + sr, r)
        return r

    def _prange(self, p):
        return p * self.PSZ, min((p + 1) * self.PSZ, self.N)

    def _wait(self, key, t0):
        import time

        w = getattr(self, "waits", None)
        if w is None:
            w = self.waits = {}
        w[key] = w.get(key, 0.0) + (time.perf_counter() - t0)

    def join_writer(self):
        if self.wb_thread is not None:
            import time

            t0 = time.perf_counter()
            self.wb_thread.join()
            self._wait("writer_join", t0)
            self.wb_thread = None
            self.writing = set()
            if self.wb_error is not None:
                err, self.wb_error = self.wb_error, None
                raise RuntimeError("write-back thread failed") from err

    def wait_for_write(self, parts):
        if self.wb_thread is not None and (set(int(p) for p in parts) & self.writing):
            self.join_writer()

    def _free_slot(self):
        for s, st in enumerate(self.sstate):
            if st == 0:
                return s
        self.join_writer()
        for s, st in enumerate(self.sstate):
            if st == 0:
                return s
        raise RuntimeError("no free slot (max_out/max_in miscomputed)")

    @torch.no_grad()
    def evict(self, p):
        s = int(self.slot_of[p])
        assert s >= 0
        if self.host_window:
            self._join_updates()
            if not self.evicted:
                self.join_writer()
            self.sstate[s] = 2
            self.evicted.append((p, s))
        else:
            if not self.evicted:
                self.join_writer()
            a, b = s * self.PSZ, (s + 1) * self.PSZ
            e, o = (
                self.ev_pool.pop()
                if self.ev_pool
                else (
                    torch.empty(self.PSZ, self.d).pin_memory(),
                    torch.empty(self.PSZ, self.d).pin_memory(),
                )
            )
            e.copy_(self.E[a:b], non_blocking=True)
            o.copy_(self.O[a:b], non_blocking=True)
            self.evicted.append((p, e, o))
        self.slot_of[p] = -1
        self.part_in_slot[s] = -1
        lo, hi = self._prange(p)
        self.amask[lo:hi] = False

    def flush_evicts(self):
        if not self.evicted:
            return
        import threading

        items, self.evicted = self.evicted, []
        self.writing = set(int(it[0]) for it in items)
        self.wb_error = None
        if self.host_window:
            E, O, PSZ = self.E, self.O, self.PSZ

            def work():
                try:
                    for p, s in items:
                        self.store.store(p, E[s * PSZ : (s + 1) * PSZ], O[s * PSZ : (s + 1) * PSZ])
                        # Reuse the slot only after writeback completes.
                        self.sstate[s] = 0
                except BaseException as ex:
                    self.wb_error = ex

        else:
            torch.cuda.current_stream().synchronize()

            def work():
                try:
                    for p, e, o in items:
                        self.store.store(p, e, o)
                    self.ev_pool.extend((e, o) for _, e, o in items)
                except BaseException as ex:
                    self.wb_error = ex

        self.wb_thread = threading.Thread(target=work)
        self.wb_thread.start()

    def _pf_buf(self):
        if self.pf_pool:
            return self.pf_pool.pop()
        return (
            torch.empty(self.PSZ, self.d).pin_memory(),
            torch.empty(self.PSZ, self.d).pin_memory(),
        )

    def prefetch(self, parts):
        import threading

        parts = [p for p in parts if p not in self.pf]
        self.wait_for_write(parts)
        if self.host_window:
            E, O, PSZ = self.E, self.O, self.PSZ
            for p in parts:
                s = self._free_slot()
                self.sstate[s] = 3

                def rd(p=p, s=s):
                    try:
                        self.store.load(p, E[s * PSZ : (s + 1) * PSZ], O[s * PSZ : (s + 1) * PSZ])
                    except BaseException as ex:
                        self.pf_error[p] = ex

                th = threading.Thread(target=rd)
                th.start()
                self.pf[p] = s
                self.pf_threads[p] = th
            return
        torch.cuda.current_stream().synchronize()

        def rd(p, e, o):
            try:
                self.store.load(p, e, o)
            except BaseException as ex:
                self.pf_error[p] = ex

        for p in parts:
            e, o = self._pf_buf()
            th = threading.Thread(target=rd, args=(p, e, o))
            th.start()
            self.pf[p] = (e, o)
            self.pf_threads[p] = th

    @torch.no_grad()
    def load(self, p):
        if self.host_window:
            if p in self.pf:
                import time

                t0 = time.perf_counter()
                self.pf_threads.pop(p).join()
                self._wait("prefetch_join", t0)
                s = self.pf.pop(p)
                if p in self.pf_error:
                    raise RuntimeError(f"prefetch failed for p={p}") from self.pf_error.pop(p)
            else:
                self.wait_for_write([p])
                s = self._free_slot()
                self.store.load(
                    p,
                    self.E[s * self.PSZ : (s + 1) * self.PSZ],
                    self.O[s * self.PSZ : (s + 1) * self.PSZ],
                )
            self.sstate[s] = 1
        else:
            s = self.part_in_slot.index(-1)
            a, b = s * self.PSZ, (s + 1) * self.PSZ
            if p in self.pf:
                self.pf_threads.pop(p).join()
                e, o = self.pf.pop(p)
                if p in self.pf_error:
                    raise RuntimeError(f"prefetch failed for p={p}") from self.pf_error.pop(p)
            else:
                torch.cuda.current_stream().synchronize()
                self.wait_for_write([p])
                e, o = self._pf_buf()
                self.store.load(p, e, o)
            self.E[a:b].copy_(e, non_blocking=True)
            self.O[a:b].copy_(o, non_blocking=True)
            self.pf_pool.append((e, o))
        self.slot_of[p] = s
        self.part_in_slot[s] = p
        lo, hi = self._prange(p)
        self.amask[lo:hi] = True

    def resident_mask(self):
        return self.amask.clone()

    def sync(self):
        if getattr(self, "host_window", False):
            self._join_updates()
        self.flush_evicts()
        self.join_writer()

    def set_window(self, res):
        cur = set(x for x in self.part_in_slot if x >= 0)
        new = set(int(x) for x in res)
        out = cur - new
        inn = new - cur
        return sorted(out), sorted(inn)

    def s_E(self):
        return self.SE if self.s_dev == "cuda" else self.E[self.S_OFF : self.S_OFF + self.kS]

    def s_O(self):
        return self.SO if self.s_dev == "cuda" else self.O[self.S_OFF : self.S_OFF + self.kS]

    def _gather(self, rc):
        if not self.host_window:
            return self.E[rc]
        if self.kS and self.s_dev == "cuda":
            isS = rc >= self.S_OFF
            if bool(isS.any()):
                out = torch.empty(rc.numel(), self.d, device=self.dev)
                out[isS] = self.SE[rc[isS] - self.S_OFF]
                rest = ~isS
                if bool(rest.any()):
                    out[rest] = self._gather_host(rc[rest])
                return out
        return self._gather_host(rc)

    def _gather_host(self, rc):
        n = rc.numel()
        if self.cfg.counters:
            if self.kS and self.s_dev == "cpu":
                nS = (rc >= self.S_OFF).sum()
                self.cnt["gath_S_rows"] += nS
                self.cnt["gath_W_rows"] += n - nS
            else:
                self.cnt["gath_W_rows"] += n
        if self.stale == 0:
            self._join_updates()
        ev = getattr(self, "gather_ev", None)
        if ev is not None:
            ev.synchronize()
        if self.gbuf_n < n:
            self.gbuf_h = self.arena.alloc((max(n, 2 * self.gbuf_n), self.d))
            self.gbuf_n = self.gbuf_h.shape[0]
            self.gidx_h = self.arena.alloc((self.gbuf_n,), torch.int64)
        rcc = self.gidx_h[:n]
        rcc.copy_(rc)
        buf = self.gbuf_h[:n]
        torch.index_select(self.E, 0, rcc, out=buf)
        out = buf.to(self.dev, non_blocking=True)
        self.gather_ev = torch.cuda.Event()
        self.gather_ev.record()
        return out

    def _adagrad(self, u, acc, q=None):
        if not self.host_window:
            self.O[u] += (acc * acc) if q is None else q
            self.E[u] -= self.lr * acc / (self.O[u].sqrt() + self.eps)
            return
        if self.kS and self.s_dev == "cuda":
            isS = u >= self.S_OFF
            if bool(isS.any()):
                j = u[isS] - self.S_OFF
                a = acc[isS]
                qS = None if q is None else q[isS]
                j, a, qS = self._s_update_filter(j, a, qS)
                if j.numel():
                    o = self.SO[j] + ((a * a) if qS is None else qS)
                    self.SO[j] = o
                    self.SE[j] -= self.lr * a / (o.sqrt() + self.eps)
                rest = ~isS
                if not bool(rest.any()):
                    return
                u, acc = u[rest], acc[rest]
                q = None if q is None else q[rest]
        elif self.kS:
            isS = u >= self.S_OFF
            if bool(isS.any()):
                j, a, qS = self._s_update_filter(
                    u[isS] - self.S_OFF, acc[isS], None if q is None else q[isS]
                )
                u = torch.cat([u[~isS], j + self.S_OFF])
                acc = torch.cat([acc[~isS], a])
                q = None if q is None else torch.cat([q[~isS], qS])
                if not u.numel():
                    return
        if self.cfg.counters:
            self.cnt["upd_W"] += (u < self.S_OFF).sum() if self.kS else len(u)
        self._join_updates()
        n = len(u)
        slot = self._upd_slot(n)
        uc, ac = slot["u"][:n], slot["acc"][:n]
        uc.copy_(u, non_blocking=True)
        ac.copy_(acc, non_blocking=True)
        qc = None
        if q is not None:
            qc = slot["q"][:n]
            qc.copy_(q, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        import threading

        E, O, lr, eps = self.E, self.O, self.lr, self.eps
        k = 4 if n >= 8192 else 1
        errs = []

        def part(i):
            try:
                sl = slice(i * n // k, (i + 1) * n // k)
                uu, aa = uc[sl], ac[sl]
                o = torch.index_select(O, 0, uu)
                if qc is None:
                    o.addcmul_(aa, aa)
                else:
                    o += qc[sl]
                O.index_copy_(0, uu, o)
                E.index_add_(0, uu, -lr * aa / (o.sqrt() + eps))
            except BaseException as ex:
                errs.append(ex)

        def work():
            try:
                ev.synchronize()
                if k == 1:
                    part(0)
                else:
                    ths = [threading.Thread(target=part, args=(i,)) for i in range(k)]
                    for t in ths:
                        t.start()
                    for t in ths:
                        t.join()
                if errs:
                    raise errs[0]
            except BaseException as ex:
                self.upd_error = ex

        th = threading.Thread(target=work)
        th.start()
        self.upd.append(th)

    def _s_update_filter(self, j, a, qS):
        if not (self.cfg.counters or self.cfg.always_freeze_out):
            return j, a, qS
        in_win = self.amask[self.s_ids[j]]
        if self.cfg.counters:
            n_in = in_win.sum()
            self.cnt["upd_S_in"] += n_in
            if self.cfg.always_freeze_out:
                self.cnt["grad_S_out"] += j.numel() - n_in
            else:
                self.cnt["upd_S_out"] += j.numel() - n_in
        if self.cfg.always_freeze_out:
            j, a = j[in_win], a[in_win]
            qS = None if qS is None else qS[in_win]
        return j, a, qS

    def _upd_slot(self, n):
        sl = getattr(self, "upd_slot", None)
        if sl is None or sl["u"].shape[0] < n:
            cap = max(n, 2 * (sl["u"].shape[0] if sl else 1 << 16))
            sl = self.upd_slot = dict(
                u=self.arena.alloc((cap,), torch.int64),
                acc=self.arena.alloc((cap, self.d)),
                q=self.arena.alloc((cap, self.d)),
            )
        return sl

    def _join_updates(self):
        if not self.upd:
            return
        import time

        t0 = time.perf_counter()
        while self.upd:
            self.upd.pop(0).join()
        self._wait("update_join", t0)
        if getattr(self, "upd_error", None) is not None:
            err, self.upd_error = self.upd_error, None
            raise RuntimeError("host update thread failed") from err

    def window_slice(self, p):
        s = int(self.slot_of[p])
        lo, hi = self._prange(p)
        if self.host_window:
            self._join_updates()
        v = self.E[s * self.PSZ : s * self.PSZ + (hi - lo)]
        if self.host_window and self.cfg.counters:
            self.cnt_bytes["xfer_refresh"] += v.numel() * 4
        return v if not self.host_window else v.to(self.dev, non_blocking=True)

    def begin_batch(self, nodes):
        self.leaves = []

    def read(self, idx):
        if getattr(self, "eval_E", None) is not None:
            if not self.eval_E.is_cuda:
                self.cnt_bytes["xfer_eval"] += idx.numel() * self.d * 4
            return self.eval_E[idx] if self.eval_E.is_cuda else self.eval_E[idx.cpu()].to(self.dev)
        flat = idx.reshape(-1)
        r = self.rows(flat)
        resident = r >= 0
        if self.cfg.counters:
            self.cnt["uniq_rows"] += flat.numel()
            self.cnt["uniq_Q"] += (~resident).sum()
            if self.kS:
                isS = r >= self.S_OFF
                self.cnt["uniq_S"] += isS.sum()
                self.cnt["uniq_W"] += (resident & ~isS).sum()
            else:
                self.cnt["uniq_W"] += resident.sum()
        train = torch.is_grad_enabled() and getattr(self, "leaves", None) is not None
        if self.kS:
            assert self.s_init_done, "S region not initialized (read before init_replica)"
            self.s_reads_t += (r >= self.S_OFF).sum()
        if self.replica is None or self.cfg.nbr_grad == "full":
            rc = r.clamp(min=0)
            base = self._gather(rc)
            if train:
                base = base.detach().requires_grad_(True)
                self.leaves.append((rc, resident, base))
            return base.view(idx.shape + (self.d,))
        ri = torch.nonzero(resident).squeeze(1)
        ni = torch.nonzero(~resident).squeeze(1)
        out = torch.zeros(flat.numel(), self.d, device=self.dev)
        if ri.numel():
            base = self._gather(r[ri])
            if train:
                base = base.detach().requires_grad_(True)
                self.leaves.append((r[ri], None, base))
            out = out.index_put((ri,), base)
        if ni.numel():
            rep = self.replica.read(flat[ni])
            if self.cfg.nbr_grad == "deferred" and train:
                rep = rep.detach().requires_grad_(True)
                self.pending.append((flat[ni], rep))
            out = out.index_put((ni,), rep)
        return out.view(idx.shape + (self.d,))

    def visible(self, allowed):
        if self.replica is None or self.wmask is None:
            return allowed
        if getattr(self.replica, "has_all", None) is None:
            self.replica.has_all = bool(self.replica.has.all())
        if self.replica.has_all:
            return allowed
        vis = self.wmask | self.replica.has
        return vis if allowed is None else (allowed & vis)

    @torch.no_grad()
    def step(self, wmask):
        leaves = getattr(self, "leaves", None)
        if not leaves:
            return
        rows, grads = [], []
        for rc, resident, leaf in leaves:
            if leaf.grad is None:
                continue
            if resident is None:
                rows.append(rc.reshape(-1))
                grads.append(leaf.grad.reshape(-1, self.d))
                continue
            m = resident.reshape(-1)
            rows.append(rc.reshape(-1)[m])
            grads.append(leaf.grad.reshape(-1, self.d)[m])
        self.leaves = None
        if not rows:
            return
        r = torch.cat(rows)
        v = torch.cat(grads)
        # Combine all occurrences before the per-node Adagrad update.
        u, inv = torch.unique(r, return_inverse=True)
        acc = torch.zeros(len(u), self.d, device=self.dev)
        acc.index_add_(0, inv, v)
        if self.kS:
            self.s_upd_t += (u >= self.S_OFF).sum()
        self._adagrad(u, acc)

    def flush_pending(self):
        for idx, rep in self.pending:
            if rep.grad is not None:
                self.gbuf.add(idx.reshape(-1), rep.grad.reshape(-1, self.d).float())
        self.pending.clear()

    @torch.no_grad()
    def apply_deferred(self, idx):
        if self.gbuf is None:
            return
        sub, g = self.gbuf.take(idx)
        if len(sub) == 0:
            return
        r = self.rows(sub)
        m = r >= 0
        q = getattr(self.gbuf, "last_q", None)
        if bool(m.any()):
            self._adagrad(r[m], g[m], None if q is None else q[m])

    @torch.no_grad()
    def resident_rows_of(self, p):
        s = int(self.slot_of[p])
        return torch.arange(s * self.PSZ, (s + 1) * self.PSZ, device=self.dev)

    @torch.no_grad()
    def init_replica(self, replica):
        e = torch.empty(self.PSZ, self.d)
        o = torch.empty(self.PSZ, self.d)
        for p in range(self.P):
            lo, hi = p * self.PSZ, min((p + 1) * self.PSZ, self.N)
            self.store.load(p, e, o)
            replica.refresh_values(
                torch.arange(lo, hi, device=self.dev), e[: hi - lo].to(self.dev), force=True
            )
            if self.kS and not self.s_init_done:
                a = int(torch.searchsorted(self.s_ids_cpu, lo))
                b = int(torch.searchsorted(self.s_ids_cpu, hi))
                if b > a:
                    loc = self.s_ids_cpu[a:b] - lo
                    if self.s_dev == "cuda":
                        self.SE[a:b] = e[loc].to(self.dev)
                        self.SO[a:b] = o[loc].to(self.dev)
                    else:
                        self.E[self.S_OFF + a : self.S_OFF + b] = e[loc]
                        self.O[self.S_OFF + a : self.S_OFF + b] = o[loc]
        if self.kS:
            self.s_init_done = True
        self.init_io = dict(
            init_io_read_bytes=self.store.read_bytes, init_io_reads=self.store.reads
        )
        self.store.read_bytes = 0
        self.store.reads = 0

    def full_E(self, host=False):
        if getattr(self, "host_window", False):
            self._join_updates()
        E = self.store.load_all(self.N, "cpu" if host else self.dev)
        for s, p in enumerate(self.part_in_slot):
            if p >= 0:
                lo, hi = p * self.PSZ, min((p + 1) * self.PSZ, self.N)
                E[lo:hi] = self.E[s * self.PSZ : s * self.PSZ + (hi - lo)].to(E.device)
        if self.kS:
            ids = self.s_ids_cpu if E.device.type == "cpu" else self.s_ids
            E[ids] = self.s_E().to(E.device)
        return E

    def track_hwm(self):
        rss = None
        for ln in open("/proc/self/status"):
            if ln.startswith("VmHWM"):
                self.hwm = int(ln.split()[1]) * 1024
            elif ln.startswith("VmRSS"):
                rss = int(ln.split()[1]) * 1024
        if rss is not None:
            st = getattr(self, "rss_stat", None) or [0, 0, 0]
            st[0] = max(st[0], rss)
            st[1] += rss
            st[2] += 1
            self.rss_stat = st

    def pop_s_stat(self):
        if not self.kS:
            return {}
        r = dict(
            s_rows=self.kS,
            s_sel=self.s_sel,
            s_dev=self.s_dev,
            s_bytes=self.s_bytes,
            s_reads=int(self.s_reads_t),
            s_upd_rows=int(self.s_upd_t),
        )
        self.s_reads_t.zero_()
        self.s_upd_t.zero_()
        return r

    def pop_rss(self):
        st = getattr(self, "rss_stat", None)
        self.rss_stat = None
        r = dict(vm_rss_max=st[0], vm_rss_mean=st[1] / st[2]) if st else {}
        try:
            io = {l.split(":")[0]: int(l.split(":")[1]) for l in open("/proc/self/io")}
            prev = getattr(self, "_io_prev", None) or {"read_bytes": 0, "write_bytes": 0}
            r["proc_io_read_bytes"] = io["read_bytes"] - prev["read_bytes"]
            r["proc_io_write_bytes"] = io["write_bytes"] - prev["write_bytes"]
            self._io_prev = io
        except Exception:
            pass
        w = getattr(self, "waits", None)
        if w:
            r["waits_sec"] = {k: round(v, 2) for k, v in w.items()}
            self.waits = {}
        return r

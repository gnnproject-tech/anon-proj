import torch


class GradBuf:
    def __init__(self, N, d, dtype, frac, cap, deg, dev, home=None, acc_dtype="fp16", keep_q=False):
        self.N, self.d, self.dev = N, d, dev
        self.kind = dtype
        self.home = home or dev
        host = self.home == "cpu"
        gdt = {"fp16": torch.float16, "int8": torch.int8}.get(dtype, torch.float32)
        self.allow = torch.ones(N, dtype=torch.bool, device=dev)
        if frac < 1.0:
            self.allow[:] = False
            self.allow[torch.topk(deg, int(N * frac)).indices] = True
        self.cidx = torch.full((N,), -1, dtype=torch.int32, device=dev)
        arows = torch.nonzero(self.allow).squeeze(1)
        self.cidx[arows] = torch.arange(len(arows), device=dev, dtype=torch.int32)
        self.arows = arows
        k = len(arows)
        self.k = k
        from .pinned import pin_exact

        self.G = pin_exact((k, d), gdt) if host else torch.zeros(k, d, dtype=gdt, device=dev)
        self.S = pin_exact((k,)) if host else torch.zeros(k, device=dev)
        self.last = torch.full((k,), -1, dtype=torch.long)
        self.first = torch.full((k,), -1, dtype=torch.long)
        self.cap = int(N * cap) if cap > 0 else 0
        self.t = 0
        self.occ = []
        self.dropped = 0

        self.acc_dt = torch.float32 if acc_dtype == "fp32" else torch.float16
        self.acc = torch.zeros(k, d, dtype=self.acc_dt, device=dev)
        self.touched = torch.zeros(k, dtype=torch.bool, device=dev)
        self.gpu_acc_bytes = self.acc.numel() * self.acc.element_size()
        self.nz = torch.zeros(k, dtype=torch.bool, device=dev)
        self.keep_q = keep_q
        if keep_q:
            self.acc2 = torch.zeros(k, d, dtype=self.acc_dt, device=dev)
            self.G2 = (
                pin_exact((k, d), torch.uint8)
                if host
                else torch.zeros(k, d, dtype=torch.uint8, device=dev)
            )
            self.S2 = pin_exact((k,)) if host else torch.zeros(k, device=dev)
            self.gpu_acc_bytes *= 2
        self.bytes = k * d * {"fp16": 2, "int8": 1}.get(dtype, 4) * (2 if keep_q else 1)

        z = lambda: torch.zeros((), dtype=torch.bool, device=dev)
        self.bad_in, self.bad_acc, self.bad_q = z(), z(), z()
        self.bi = torch.zeros((), dtype=torch.long, device=dev)
        self.first_bad_in = torch.full((), -1, dtype=torch.long, device=dev)
        self.amax_in = torch.zeros((), device=dev)
        self.win_stats = []

        self.mass = torch.zeros(2, device=dev)
        self.rows_io = torch.zeros(2, dtype=torch.long, device=dev)
        self.wait_hist = torch.zeros(4096, dtype=torch.long)
        self.wait_hist_first = torch.zeros(4096, dtype=torch.long)

    @torch.no_grad()
    def add(self, idx, grad):
        c = self.cidx[idx].long()
        keep = c >= 0
        if getattr(self, "dropped_t", None) is None:
            self.dropped_t = torch.zeros((), dtype=torch.long, device=self.dev)
        self.dropped_t += (~keep).sum()
        bad = ~torch.isfinite(grad).all()
        self.first_bad_in = torch.where(bad & (self.first_bad_in < 0), self.bi, self.first_bad_in)
        self.bad_in |= bad
        self.amax_in = torch.maximum(self.amax_in, grad.abs().amax())
        self.bi += 1
        l1 = grad.abs().sum(1)
        kf = keep.float()
        self.mass += torch.stack([(l1 * kf).sum(), (l1 * (1 - kf)).sum()])
        self.rows_io += torch.stack([keep.sum(), (~keep).sum()])
        c = c.clamp(min=0)
        g = torch.where(keep.unsqueeze(-1), grad, torch.zeros_like(grad))
        self.acc.index_add_(0, c, g.to(self.acc_dt))
        self.touched[c] = True
        if self.keep_q:
            self.acc2.index_add_(0, c, (g * g).to(self.acc_dt))

    @torch.no_grad()
    def flush(self):
        tc = torch.nonzero(self.touched).squeeze(1)
        if len(tc) == 0:
            self._check(0, 0.0)
            return
        acc = self.acc[tc].float()
        self.bad_acc |= ~torch.isfinite(acc).all()
        amax_acc = acc.abs().amax()
        rc = tc.cpu() if self.home == "cpu" else tc
        if self.kind == "int8":
            cur = self.G[rc].to(self.dev).float() * self.S[rc].to(self.dev).unsqueeze(-1)
            tot = cur + acc
            s = tot.abs().amax(1).clamp(min=1e-12) / 127
            q = torch.round(tot / s[:, None]).to(torch.int8)
            self.bad_q |= ~(torch.isfinite(tot).all() & torch.isfinite(s).all())
            if self.home == "cpu":
                self.G[rc] = q.cpu()
                self.S[rc] = s.cpu()
            else:
                self.G[rc] = q
                self.S[rc] = s
        else:
            if self.home == "cpu":
                self.G[rc] += acc.to(self.G.dtype).cpu()
            else:
                self.G.index_add_(0, tc, acc.to(self.G.dtype))
        if self.keep_q:
            cur2 = self.G2[rc].to(self.dev).float() * self.S2[rc].to(self.dev).unsqueeze(-1)
            tot2 = cur2 + self.acc2[tc].float()
            s2 = tot2.amax(1).clamp(min=1e-12) / 255
            q2 = torch.round(tot2 / s2[:, None]).to(torch.uint8)
            if self.home == "cpu":
                self.G2[rc] = q2.cpu()
                self.S2[rc] = s2.cpu()
            else:
                self.G2[rc] = q2
                self.S2[rc] = s2
            self.acc2[tc] = 0
        self.xfer_bytes = getattr(self, "xfer_bytes", 0) + len(tc) * self.d * 2
        self.nz[tc] = True
        rh = rc.cpu() if rc.is_cuda else rc
        self.last[rh] = self.t
        f = self.first[rh]
        self.first[rh] = torch.where(f < 0, torch.full_like(f, self.t), f)
        self.acc[tc] = 0
        self.touched[tc] = False
        self._check(len(tc), amax_acc)

    def _check(self, ntouched, amax_acc):
        bi, bin_, bacc, bq = int(self.bi), bool(self.bad_in), bool(self.bad_acc), bool(self.bad_q)
        amax_in = float(self.amax_in)
        amax_acc = float(amax_acc)
        self.win_stats.append((self.t, ntouched, amax_in, amax_acc))
        if bin_ or bacc or bq:
            where = "input_grad" if bin_ else ("accumulator" if bacc else "quantize")
            raise RuntimeError(
                f"GBUF non-finite: stage={where} window={self.t} first_bad_batch={int(self.first_bad_in)} batches={bi} "
                f"amax_in={amax_in:.3e} amax_acc={amax_acc:.3e} acc_dtype={self.acc_dt} touched={ntouched}"
            )
        self.bi.zero_()
        self.amax_in.zero_()

    @torch.no_grad()
    def window_start(self):
        self.flush()
        self.t += 1
        n = int(self.nz.sum())
        if self.cap and n > self.cap:
            nz = torch.nonzero(self.nz).squeeze(1)
            last = self.last.to(self.dev)
            old = nz[torch.argsort(last[nz])[: n - self.cap]]
            oc = old.cpu() if self.home == "cpu" else old
            self.G[oc] = 0
            self.nz[old] = False
            self.dropped += len(old)
            n = self.cap
        self.occ.append(n)

    @torch.no_grad()
    def take(self, idx):
        c = self.cidx[idx].long()
        m = c >= 0
        self.last_q = None
        if not bool(m.any()):
            return idx[:0], torch.zeros(0, self.d, device=self.dev)
        idx, c = idx[m], c[m]
        m2 = self.nz[c]
        if not bool(m2.any()):
            return idx[:0], torch.zeros(0, self.d, device=self.dev)
        idx, c = idx[m2], c[m2]
        cc = c.cpu() if self.home == "cpu" else c
        ch = cc.cpu() if cc.is_cuda else cc
        w = (self.t - self.last[ch]).clamp(min=0, max=4095)
        self.wait_hist += torch.bincount(w, minlength=4096)
        wf = (self.t - self.first[ch]).clamp(min=0, max=4095)
        self.wait_hist_first += torch.bincount(wf, minlength=4096)
        self.first[ch] = -1
        v = self.G[cc].to(self.dev).float()
        if self.kind == "int8":
            v = v * self.S[cc].to(self.dev).unsqueeze(-1)
        self.G[cc] = 0
        self.nz[c] = False
        if self.keep_q:
            q = self.G2[cc].to(self.dev).float() * self.S2[cc].to(self.dev).unsqueeze(-1)
            self.G2[cc] = 0
            self.last_q = q
        return idx, v

    def pop_stat(self):
        if getattr(self, "dropped_t", None) is not None:
            self.dropped += int(self.dropped_t)
            self.dropped_t = None
        r = (
            dict(gbuf_rows_max=max(self.occ), gbuf_rows_mean=sum(self.occ) / len(self.occ))
            if self.occ
            else {}
        )
        if self.dropped:
            r["gbuf_dropped"] = self.dropped
        m = self.mass.tolist()
        n = self.rows_io.tolist()
        if n[0] + n[1]:
            r.update(gbuf_mass_in=m[0], gbuf_mass_out=m[1], gbuf_rows_in=n[0], gbuf_rows_out=n[1])
            self.mass.zero_()
            self.rows_io.zero_()
        for tag, h in (("wait", self.wait_hist), ("delay", self.wait_hist_first)):
            tot = int(h.sum())
            if not tot:
                continue
            cum = torch.cumsum(h, 0)
            q = lambda f: int(torch.searchsorted(cum, int(f * tot)))
            r.update(
                {
                    f"gbuf_{tag}_mean": float((h * torch.arange(len(h))).sum() / tot),
                    f"gbuf_{tag}_p50": q(0.5),
                    f"gbuf_{tag}_p90": q(0.9),
                    f"gbuf_{tag}_max": int(torch.nonzero(h).max()),
                    f"gbuf_{tag}_hist": h[:512].tolist(),
                }
            )
            h.zero_()
        if self.win_stats:
            r["gbuf_amax_acc"] = max(w[3] for w in self.win_stats)
            r["gbuf_amax_in"] = max(w[2] for w in self.win_stats)
            r["gbuf_amax_acc_window"] = max(self.win_stats, key=lambda w: w[3])[0]
            self.win_stats = []
        self.occ = []
        self.dropped = 0
        return r

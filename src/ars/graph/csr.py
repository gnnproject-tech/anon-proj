import torch


class Sampler:
    def __init__(self, data, K, hub, dev, mode="exact"):
        self.d, self.K, self.hub, self.dev, self.mode = data, K, hub, dev, mode
        self.split = getattr(data, "nbr_dir", "union") == "split"
        self.win = None
        if self.split and mode != "exact":
            assert hub, "split sampling is implemented only on the hub (candidate) path"

    def _hub(self, U, allowed, K, indptr, indices):
        st = indptr[U]
        dg = indptr[U + 1] - st
        m = len(U)
        M = 64
        cand = indices[
            (
                st[:, None] + (torch.rand(m, M, device=self.dev) * dg[:, None]).long().clamp(min=0)
            ).clamp(max=len(indices) - 1)
        ].long()
        valid = (dg > 0)[:, None].expand(-1, M).clone()
        if allowed is not None:
            valid &= allowed[cand]
        cnt = valid.sum(1)
        cs = valid.cumsum(1)
        t = (torch.rand(m, K, device=self.dev) * cnt[:, None]).long() + 1
        return cand.gather(1, torch.searchsorted(cs, t).clamp(max=M - 1)), cnt

    def sample(self, U, allowed, K=None):
        if K is not None and K != self.K:
            saved = self.K
            self.K = K
            try:
                return self.sample(U, allowed)
            finally:
                self.K = saved
        if self.mode == "exact" and (allowed is None or self.win is not None):
            return self._sample_exact(U, allowed)
        if self.split:
            K1 = self.K // 2
            K2 = self.K - K1
            so, co = self._hub(U, allowed, K1, *self.d.csr["out"])
            si, ci = self._hub(U, allowed, K2, *self.d.csr["in"])
            mo = (ci == 0) & (co > 0)
            mi = (co == 0) & (ci > 0)
            if bool(mo.any()):
                si[mo] = self._hub(U[mo], allowed, K2, *self.d.csr["out"])[0]
            if bool(mi.any()):
                so[mi] = self._hub(U[mi], allowed, K1, *self.d.csr["in"])[0]
            return torch.cat([so, si], 1), co + ci
        if self.hub:
            return self._hub(U, allowed, self.K, self.d.indptr, self.d.indices)
        st = self.d.indptr[U]
        dg = self.d.indptr[U + 1] - st
        m = len(U)
        maxd = int(dg.max().item()) if m else 1
        ar = torch.arange(maxd, device=self.dev)
        pos = (st[:, None] + ar[None, :]).clamp(max=len(self.d.indices) - 1)
        nb = self.d.indices[pos].long()
        valid = ar[None, :] < dg[:, None]
        if allowed is not None:
            valid = valid & allowed[nb]
        cnt = valid.sum(1)
        cs = valid.cumsum(1)
        t = (torch.rand(m, self.K, device=self.dev) * cnt[:, None]).long() + 1
        return nb.gather(1, torch.searchsorted(cs, t).clamp(max=maxd - 1)), cnt

    @staticmethod
    def _direct(U, K, indptr, indices):
        st = indptr[U]
        dg = indptr[U + 1] - st
        m = len(U)
        pos = (st[:, None] + (torch.rand(m, K, device=U.device) * dg[:, None]).long()).clamp(
            max=len(indices) - 1
        )
        return indices[pos].long(), dg

    @staticmethod
    def _csr(u, v, n):
        o = torch.argsort(u)
        us, vs = u[o], v[o]
        indptr = torch.zeros(n + 1, dtype=torch.long, device=u.device)
        indptr[1:] = torch.bincount(us, minlength=n).cumsum(0)
        return indptr, vs

    def set_window(self, state, res):
        data = self.d
        P = data.P
        cum = data.cum
        res = [int(x) for x in res]
        rng = [
            (int(cum[i * P + j]), int(cum[i * P + j + 1]))
            for i in res
            for j in res
            if cum[i * P + j + 1] > cum[i * P + j]
        ]
        if not rng:
            self.win = None
            return
        if getattr(data, "edges_disk", False):
            s, _, t = data.split(data.window_edges(rng))
        else:
            eidx = torch.cat([torch.arange(lo, hi, device=self.dev) for lo, hi in rng])
            s, _, t = data.edges(eidx)
        ls, lt = state.rows(s), state.rows(t)
        n = getattr(state, "nslots", state.c) * state.PSZ
        if self.split:
            self.win = (self._csr(ls, lt, n), self._csr(lt, ls, n))
        else:
            self.win = (self._csr(torch.cat([ls, lt]), torch.cat([lt, ls]), n),)
        self.win_rows = state.rows
        self.win_back = state

    def _sample_exact(self, U, allowed):
        if allowed is not None and getattr(self, "win", None) is not None:
            lu = self.win_rows(U)
            assert bool((lu >= 0).all()), "root of window-restricted sampling is outside the window"
            st = self.win_back
            back = lambda lr: st.global_of(lr)
            if self.split:
                K1 = self.K // 2
                K2 = self.K - K1
                so, co = self._direct(lu, K1, *self.win[0])
                si, ci = self._direct(lu, K2, *self.win[1])
                mo = (ci == 0) & (co > 0)
                mi = (co == 0) & (ci > 0)
                si = torch.where(mo[:, None], self._direct(lu, K2, *self.win[0])[0], si)
                so = torch.where(mi[:, None], self._direct(lu, K1, *self.win[1])[0], so)
                sel = torch.cat([so, si], 1)
                return back(sel), co + ci
            sel, cnt = self._direct(lu, self.K, *self.win[0])
            return back(sel), cnt
        if self.split:
            K1 = self.K // 2
            K2 = self.K - K1
            so, co = self._direct(U, K1, *self.d.csr["out"])
            si, ci = self._direct(U, K2, *self.d.csr["in"])
            mo = (ci == 0) & (co > 0)
            mi = (co == 0) & (ci > 0)
            si = torch.where(mo[:, None], self._direct(U, K2, *self.d.csr["out"])[0], si)
            so = torch.where(mi[:, None], self._direct(U, K1, *self.d.csr["in"])[0], so)
            return torch.cat([so, si], 1), co + ci
        return self._direct(U, self.K, self.d.indptr, self.d.indices)

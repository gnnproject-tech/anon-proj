import math, torch, torch.nn.functional as F


class Model:
    def __init__(self, cfg, data, state, sampler, dev):
        d, NREL = cfg.d, data.NREL
        self.cfg, self.state, self.sampler, self.dev = cfg, state, sampler, dev
        assert cfg.encoder in ("sage", "gat") and cfg.layers in (
            1,
            2,
        ), "encoder sage|gat, layers 1|2"
        if cfg.layers == 2 or cfg.encoder == "gat":
            assert cfg.gather == "unique" and cfg.mode == "GNN", "2-layer and GAT require the unique gather path"
        if cfg.encoder == "gat":
            assert cfg.layers == 1, "GAT supports one layer only"
        self.R = torch.nn.Parameter(
            torch.ones(NREL, d, device=dev) + 0.1 * torch.randn(NREL, d, device=dev)
        )
        dense = []
        if cfg.encoder == "sage":
            self.Ws = torch.nn.Linear(d, d, bias=True).to(dev)
            self.Wn = torch.nn.Linear(d, d, bias=False).to(dev)
            dense += list(self.Ws.parameters()) + list(self.Wn.parameters())
            if cfg.layers == 2:
                self.Ws2 = torch.nn.Linear(d, d, bias=True).to(dev)
                self.Wn2 = torch.nn.Linear(d, d, bias=False).to(dev)
                dense += list(self.Ws2.parameters()) + list(self.Wn2.parameters())
        else:
            H, dh = cfg.heads, d
            self.W = torch.nn.Linear(d, H * dh, bias=False).to(dev)
            with torch.no_grad():
                self.W.weight.normal_(0, math.sqrt(2 / (d + dh)))
            self.a_l = torch.nn.Parameter(torch.randn(H, dh, device=dev) * math.sqrt(2 / (dh + 1)))
            self.a_r = torch.nn.Parameter(torch.randn(H, dh, device=dev) * math.sqrt(2 / (dh + 1)))
            self.b = torch.nn.Parameter(torch.zeros(d, device=dev))
            dense += [self.W.weight, self.a_l, self.a_r, self.b]
        self.opt = torch.optim.Adam([self.R] + dense, lr=cfg.lr_dense, foreach=True)

    def _hop(self, U, allowed, K):
        sel, cnt = self.sampler.sample(U, allowed, K)
        valid = cnt > 0
        if hasattr(self.state, "count_nbrs"):
            self.state.count_nbrs(sel, valid)
        return torch.where(valid[:, None], sel, U[:, None]), valid

    @staticmethod
    def _sage(feats, inv, valid, m, K, Ws, Wn):
        x = torch.index_select(feats, 0, inv[:m])
        row = torch.arange(m, device=feats.device).repeat_interleave(K)
        col = inv[m:]
        keep = valid[row]
        # Coalescing preserves repeated samples by summing their weights.
        A = torch.sparse_coo_tensor(
            torch.stack([row, col]), keep.to(feats.dtype) / K, (m, feats.shape[0])
        ).coalesce()
        agg = torch.sparse.mm(A, feats)
        return F.relu(Ws(x) + Wn(agg))

    def _gat(self, feats, inv, valid, m, K):
        H, dh = self.cfg.heads, self.cfg.d
        u = feats.shape[0]
        Z = self.W(feats).view(u, H, dh)
        al = (Z * self.a_l).sum(-1)
        ar = (Z * self.a_r).sum(-1)
        root = inv[:m]
        col = torch.cat([root[:, None], inv[m:].view(m, K)], 1)
        e = F.leaky_relu(al[root][:, None, :] + ar[col], self.cfg.slope)
        ok = torch.cat(
            [torch.ones(m, 1, dtype=torch.bool, device=feats.device), valid[:, None].expand(m, K)],
            1,
        )
        alpha = torch.softmax(e.masked_fill(~ok[:, :, None], float("-inf")), 1)
        row = torch.arange(m, device=feats.device).repeat_interleave(K + 1)
        c = col.reshape(-1)
        outs = []
        for h in range(H):
            A = torch.sparse_coo_tensor(
                torch.stack([row, c]), alpha[:, :, h].reshape(-1), (m, u)
            ).coalesce()
            outs.append(torch.sparse.mm(A, Z[:, h, :]))
        return F.relu(torch.stack(outs, 0).mean(0) + self.b)

    def encode(self, U, allowed, K=None):
        allowed = self.state.visible(allowed)
        if self.cfg.mode == "EMB":
            return self.state.read(U)
        if self.cfg.gather == "occ":
            with self.prof("sample"):
                sel, cnt = self.sampler.sample(U, allowed, K)
            with self.prof("gather"):
                allr = self.state.read(torch.cat([U, sel.reshape(-1)]))
            with self.prof("fwd"):
                x = allr[: len(U)]
                m = allr[len(U) :].view(sel.shape[0], sel.shape[1], -1).mean(1)
                agg = torch.where((cnt > 0)[:, None], m, torch.zeros_like(m))
                return F.relu(self.Ws(x) + self.Wn(agg))

        if self.cfg.layers == 1:
            with self.prof("sample"):
                sel, valid = self._hop(U, allowed, K)
            with self.prof("gather"):
                m, K = sel.shape
                uid, inv = torch.unique(torch.cat([U, sel.reshape(-1)]), return_inverse=True)
                feats = self.state.read(uid)
            with self.prof("fwd"):
                if self.cfg.encoder == "gat":
                    return self._gat(feats, inv, valid, m, K)
                return self._sage(feats, inv, valid, m, K, self.Ws, self.Wn)

        with self.prof("sample"):
            sel1, valid1 = self._hop(U, allowed, K)
            m = len(U)
            K = sel1.shape[1]
            n1, inv1 = torch.unique(torch.cat([U, sel1.reshape(-1)]), return_inverse=True)
            sel2, valid2 = self._hop(n1, allowed, K)
        with self.prof("gather"):
            uid, inv2 = torch.unique(torch.cat([n1, sel2.reshape(-1)]), return_inverse=True)
            feats = self.state.read(uid)
        with self.prof("fwd"):
            h1 = self._sage(feats, inv2, valid2, len(n1), K, self.Ws, self.Wn)
            return self._sage(h1, inv1, valid1, m, K, self.Ws2, self.Wn2)

    @staticmethod
    def score(hs, r, hd):
        return (hs * r * hd).sum(-1)

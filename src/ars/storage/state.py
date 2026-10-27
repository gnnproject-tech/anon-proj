import math, torch


class State:
    def __init__(self, cfg, data, dev):
        N, d = data.N, cfg.d
        g = torch.Generator(device=dev)
        g.manual_seed(cfg.seed)
        if cfg.init == "e_marius":
            lim = math.sqrt(6 / (N + d))
            E = (torch.rand(N, d, device=dev, generator=g) * 2 - 1) * lim
        else:
            E = torch.randn(N, d, device=dev, generator=g) * (0.03 if cfg.init == "mid" else 0.1)
        self.E = torch.nn.Parameter(E)
        self.opt = torch.optim.Adagrad([self.E], lr=cfg.lr_emb)
        self.wmask = None
        self.replica = None
        self.gbuf = None
        self.pending = []
        self.cfg = cfg

    def read(self, idx):
        base = self.E[idx]
        if self.wmask is None or self.cfg.replica == "live":
            return base
        if self.cfg.nbr_grad == "full":
            return base
        rep = self.replica.read(idx)
        if self.cfg.nbr_grad == "deferred" and torch.is_grad_enabled():
            rep = rep.detach().requires_grad_(True)
            self.pending.append((idx, rep))
        return torch.where(self.wmask[idx].unsqueeze(-1), base, rep)

    def visible(self, allowed):
        if self.replica is None or self.wmask is None or bool(self.replica.has.all()):
            return allowed
        vis = self.wmask | self.replica.has
        return vis if allowed is None else (allowed & vis)

    def flush_pending(self):
        for idx, rep in self.pending:
            if rep.grad is not None:
                self.gbuf.add(idx.reshape(-1), rep.grad.reshape(-1, self.cfg.d).float())
        self.pending.clear()

    @torch.no_grad()
    def apply_deferred(self, idx):
        if self.gbuf is None:
            return
        sub, g = self.gbuf.take(idx)
        if len(sub) > 0:
            self.opt.zero_grad(set_to_none=True)
            self.E.grad = torch.zeros_like(self.E)
            self.E.grad[sub] = g
            self.opt.step()

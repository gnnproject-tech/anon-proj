import torch


@torch.no_grad()
def evaluate(data, state, model, dev):
    saved = state.wmask
    state.wmask = None
    paged = hasattr(state, "store")
    if paged:
        state.eval_E = state.full_E(host=not data.full_eval)
    N = data.N
    if not data.full_eval:
        g = torch.Generator(device=dev)
        g.manual_seed(0)
        ranks = []
        bad = torch.zeros((), dtype=torch.bool, device=dev)
        for side in (1, 0):
            for i in range(0, len(data.q), 500):
                b = data.q[i : i + 500]
                s, r, t = b[:, 0], model.R[b[:, 1]], b[:, 2]
                anchor, true = (s, t) if side == 1 else (t, s)
                negs = torch.randint(N, (len(b), 1000), device=dev, generator=g)
                hn = model.encode(torch.unique(torch.cat([anchor, true, negs.flatten()])), None)
                nodes, inv = torch.unique(
                    torch.cat([anchor, true, negs.flatten()]), return_inverse=True
                )
                ha, ht = hn[inv[: len(b)]], hn[inv[len(b) : 2 * len(b)]]
                hneg = hn[inv[2 * len(b) :]].view(len(b), 1000, -1)
                ts = (ha * r * ht).sum(-1, keepdim=True)
                sc = torch.bmm((ha * r)[:, None, :], hneg.transpose(1, 2)).squeeze(1)
                bad = bad | ~(torch.isfinite(ts).all() & torch.isfinite(sc).all())
                ranks.append((sc > ts).sum(1) + 1)
        if bool(bad):
            raise RuntimeError(
                "evaluation produced non-finite scores (NaN/inf); no metrics reported"
            )
        rk = torch.cat(ranks).float()
        state.wmask = saved
        if paged:
            state.eval_E = None
        return dict(
            h50=(rk <= 50).float().mean().item(),
            mrr=(1 / rk).mean().item(),
            mr=rk.mean().item(),
            h10=(rk <= 10).float().mean().item(),
        )
    H = torch.cat(
        [
            model.encode(torch.arange(i, min(i + 20000, N), device=dev), None)
            for i in range(0, N, 20000)
        ]
    )
    ranks = []
    for side in (1, 0):
        for i in range(0, len(data.q), data.eval_chunk):
            b = data.q[i : i + data.eval_chunk]
            s, r, t = b[:, 0], model.R[b[:, 1]], b[:, 2]
            anchor, true = (s, t) if side == 1 else (t, s)
            sc = (H[anchor] * r) @ H.T
            ts = sc.gather(1, true[:, None])
            if not bool(torch.isfinite(ts).all() & torch.isfinite(sc).all()):
                raise RuntimeError("evaluation non-finite scores (NaN/inf)")
            ar = torch.arange(N, device=dev)[None, :]
            cand = anchor[:, None] * N + ar if side == 1 else ar * N + anchor[:, None]
            mask = torch.isin(cand, data.keyset)
            mask.scatter_(1, true[:, None], False)
            ranks.append((sc.masked_fill(mask, float("-inf")) > ts).sum(1) + 1)
    rk = torch.cat(ranks).float()
    state.wmask = saved
    if paged:
        state.eval_E = None
    return dict(
        h50=(rk <= 50).float().mean().item(), mrr=(1 / rk).mean().item(), mr=rk.mean().item()
    )

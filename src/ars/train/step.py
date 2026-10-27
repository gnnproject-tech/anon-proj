import math, torch, torch.nn.functional as F


def train_batch(cfg, data, state, model, src, rel, dst, nb_allowed, wmask, pool):
    dev, d = model.dev, cfg.d
    Bsz = len(src)
    nch = cfg.nchunks
    chunk = math.ceil(Bsz / nch)
    prof = model.prof
    if hasattr(state, "begin_batch"):
        state.begin_batch(None)
    with prof("negs+unique"):
        draw = lambda: (
            pool[torch.randint(len(pool), (nch, cfg.negs), device=dev)]
            if pool is not None
            else torch.randint(data.N, (nch, cfg.negs), device=dev)
        )
        negs_t = draw()
        negs_h = negs_t if cfg.negs_shared else draw()
        nodes, inv = torch.unique(
            torch.cat([src, dst, negs_t.flatten(), negs_h.flatten()]), return_inverse=True
        )
    if cfg.neg_fanout and cfg.neg_fanout != cfg.fanout:
        is_pos = torch.isin(nodes, torch.cat([src, dst]))
        ip = torch.nonzero(is_pos).squeeze(1)
        ineg = torch.nonzero(~is_pos).squeeze(1)
        parts = [model.encode(nodes[ip], nb_allowed)]
        if len(ineg):
            parts.append(model.encode(nodes[ineg], nb_allowed, cfg.neg_fanout))
        back = torch.empty(len(nodes), dtype=torch.long, device=dev)
        back[torch.cat([ip, ineg])] = torch.arange(len(nodes), device=dev)
        h = torch.cat(parts)[back]
    else:
        h = model.encode(nodes, nb_allowed)
    with prof("score+loss"):
        hs, hd = torch.index_select(h, 0, inv[:Bsz]), torch.index_select(h, 0, inv[Bsz : 2 * Bsz])
        nn_ = nch * cfg.negs
        hn_t = torch.index_select(h, 0, inv[2 * Bsz : 2 * Bsz + nn_]).view(nch, cfg.negs, d)
        hn_h = torch.index_select(h, 0, inv[2 * Bsz + nn_ :]).view(nch, cfg.negs, d)
        r = torch.index_select(model.R, 0, rel)
        pos = model.score(hs, r, hd)[:, None]

        qs, qd = hs * r, hd * r
        pad = nch * chunk - Bsz
        if pad:
            qs = F.pad(qs, (0, 0, 0, pad))
            qd = F.pad(qd, (0, 0, 0, pad))
        tail = torch.bmm(qs.view(nch, chunk, d), hn_t.transpose(1, 2)).reshape(-1, cfg.negs)[:Bsz]
        head = torch.bmm(qd.view(nch, chunk, d), hn_h.transpose(1, 2)).reshape(-1, cfg.negs)[:Bsz]
        tgt = torch.zeros(Bsz, dtype=torch.long, device=dev)
        loss = F.cross_entropy(torch.cat([pos, tail], 1), tgt, reduction="sum") + F.cross_entropy(
            torch.cat([pos, head], 1), tgt, reduction="sum"
        )
    paged = hasattr(state, "store")
    if not paged:
        state.opt.zero_grad(set_to_none=True)
    model.opt.zero_grad(set_to_none=True)
    with prof("bwd"):
        loss.backward()
    with prof("gbuf"):
        if cfg.nbr_grad == "deferred":
            state.flush_pending()
    with prof("opt"):
        if paged:
            state.step(wmask)
        else:
            if wmask is not None and cfg.nbr_grad != "full":
                state.E.grad[~wmask] = 0
            state.opt.step()
        model.opt.step()
    return loss.detach()

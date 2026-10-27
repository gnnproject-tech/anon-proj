import time, json, numpy as np, torch
from ..metrics import evaluate
from .step import train_batch


def run(cfg, data, state, model, sampler, sched, log):
    dev = model.dev
    from .prof import MemMon

    mon = getattr(model, "memmon", None) or MemMon(dev)
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        tot = 0.0
        prev_phase = mon.begin("train")
        if ep == 1:
            log["mem_init"] = prev_phase
        if not cfg.windowed:
            perm = torch.randperm(data.E, device=dev)
            for i in range(0, len(perm), cfg.batch):
                b = perm[i : i + cfg.batch]
                s, r, d_ = data.edges(b)
                tot += train_batch(cfg, data, state, model, s, r, d_, None, None, None)
        else:
            perm = (
                np.arange(data.P) if (not cfg.redraw or ep == 1) else np.random.permutation(data.P)
            )
            windows = sched.plan(perm)
            if ep == 1:
                assert (
                    sum(hi - lo for _, rng in windows for lo, hi in rng) == data.E
                ), "windows do not cover all edges"
            paged = hasattr(state, "store")
            if cfg.replica == "epoch":
                state.replica.snapshot(state.full_E() if paged else state.E)
            if cfg.replica in ("int8", "fp32") and ep == 1:
                if paged:
                    _ti = time.time()
                    state.init_replica(state.replica)
                    log["sec_init_view"] = round(time.time() - _ti, 1)
                    if hasattr(state, "init_io"):
                        log.update(state.init_io)
                else:
                    state.replica.refresh(torch.arange(data.N, device=dev), state.E, force=True)
                state.replica.sync_cache_all()
            prev = None
            if cfg.max_windows:
                windows = windows[: cfg.max_windows]
            for wi, (res, rng) in enumerate(windows):
                if data.edges_disk:
                    wed = data.window_edges(rng)
                    eidx = None
                else:
                    eidx = sched.edge_index(rng)
                with model.prof("window"):
                    if paged:
                        out, inn = state.set_window(res)
                        for p in out:
                            lo, hi = p * data.PSZ, min((p + 1) * data.PSZ, data.N)
                            if cfg.replica in ("int8", "fp32"):
                                ids = torch.arange(lo, hi, device=dev)
                                vals = state.window_slice(p)
                                if getattr(state, "kS", 0):
                                    keep = state.srow[ids] < 0
                                    ids, vals = ids[keep], vals[keep]
                                if ids.numel():
                                    state.replica.refresh_values(ids, vals)
                            state.evict(p)
                        state.flush_evicts()
                        for p in inn:
                            state.load(p)
                        allowed = state.resident_mask()
                    else:
                        allowed = torch.isin(data.part, res.to(data.part.dtype))
                    if paged:
                        if wi + 1 < len(windows):
                            nxt = set(int(x) for x in windows[wi + 1][0]) - set(int(x) for x in res)
                            state.prefetch(sorted(nxt))
                        state.track_hwm()
                    elif prev is not None and cfg.replica in ("int8", "fp32"):
                        state.replica.refresh(torch.nonzero(prev).squeeze(1), state.E)
                    state.wmask = allowed
                    prev = allowed
                    if cfg.nbr_grad == "deferred":
                        state.gbuf.window_start()
                        state.apply_deferred(torch.nonzero(allowed).squeeze(1))
                    pool = torch.nonzero(allowed).squeeze(1) if cfg.r3 else None
                    nb_allowed = allowed if cfg.r2 else None
                    if cfg.r2 and paged and cfg.sampler == "exact":
                        sampler.set_window(state, res)
                nb_t = torch.zeros((), dtype=torch.long, device=dev)
                first_bad = torch.full((), -1, dtype=torch.long, device=dev)
                if data.edges_disk:
                    p = torch.randperm(len(wed), device=dev)
                    for i in range(0, len(p), cfg.batch):
                        s, r, d_ = data.split(wed[p[i : i + cfg.batch]])
                        l = train_batch(
                            cfg, data, state, model, s, r, d_, nb_allowed, allowed, pool
                        )
                        tot += l
                        first_bad = torch.where(
                            ~torch.isfinite(l) & (first_bad < 0), nb_t, first_bad
                        )
                        nb_t += 1
                    del wed
                else:
                    p = eidx[torch.randperm(len(eidx), device=dev)]
                    for i in range(0, len(p), cfg.batch):
                        b = p[i : i + cfg.batch]
                        s, r, d_ = data.edges(b)
                        l = train_batch(
                            cfg, data, state, model, s, r, d_, nb_allowed, allowed, pool
                        )
                        tot += l
                        first_bad = torch.where(
                            ~torch.isfinite(l) & (first_bad < 0), nb_t, first_bad
                        )
                        nb_t += 1
                fb = int(first_bad)
                if fb >= 0:
                    raise RuntimeError(
                        f"loss non-finite: epoch={ep} window={wi} first_bad_batch={fb} batches={int(nb_t)}"
                    )
        if hasattr(state, "sync"):
            state.sync()
        tot = float(tot)
        t_train = time.time() - t0
        mem_train = mon.begin("eval")
        with model.prof("eval"):
            if cfg.eval_split == "test" and ep < cfg.epochs:
                m = dict(h50=float("nan"), mrr=float("nan"), mr=float("nan"))
            else:
                m = evaluate(data, state, model, dev)
            if cfg.final_test and ep == cfg.epochs:
                qv = data.q
                data.q = data.q_test
                mt = evaluate(data, state, model, dev)
                data.q = qv
                m["test"] = {k: mt[k] for k in ("h50", "mrr", "mr")}
                print(
                    f"  final test  H@50 {mt['h50']:.4f}  MRR {mt['mrr']:.4f}  MR {mt['mr']:9,.0f}",
                    flush=True,
                )
        mem_eval = mon.end()
        m.update(
            epoch=ep,
            loss=tot,
            sec=round(time.time() - t0, 1),
            sec_train=round(t_train, 1),
            sec_eval=round(time.time() - t0 - t_train, 1),
        )
        if cfg.profile:
            pr = model.prof.pop()
            m["prof"] = pr
            print(
                "  prof "
                + "  ".join(f"{k} {v:.0f}s" for k, v in sorted(pr.items(), key=lambda kv: -kv[1])),
                flush=True,
            )
        if state.replica is not None and cfg.refresh_delta > 0:
            r = state.replica.pop_stat()
            m["refresh_frac"] = r if r is not None else 0.0
        if state.gbuf is not None:
            m.update(state.gbuf.pop_stat())
        if hasattr(state, "store"):
            m.update(state.store.pop_stat())
            m["vm_hwm_bytes"] = state.hwm
            m.update(state.pop_rss())
        if hasattr(state, "pop_s_stat"):
            m.update(state.pop_s_stat())
        if hasattr(state, "pop_counters"):
            m["counters"] = state.pop_counters()
        if str(dev).startswith("cuda"):
            m["gpu_alloc_max_bytes"] = torch.cuda.max_memory_allocated()
            m["gpu_reserved_max_bytes"] = torch.cuda.max_memory_reserved()
        m["mem"] = dict(train=mem_train, eval=mem_eval)
        if state.replica is not None and hasattr(state.replica, "gather_bytes"):
            m["rep_gather_bytes"] = state.replica.gather_bytes
            state.replica.gather_bytes = 0
        if state.replica is not None and hasattr(state.replica, "reads") and state.replica.reads:
            m["rep_cache_hit"] = state.replica.hits / state.replica.reads
            state.replica.hits = state.replica.reads = 0
        if state.gbuf is not None and hasattr(state.gbuf, "xfer_bytes"):
            m["gbuf_xfer_bytes"] = state.gbuf.xfer_bytes
            state.gbuf.xfer_bytes = 0
        log["epochs"].append(m)
        print(
            f"e{ep:2d}  loss {tot:12.1f}  H@50 {m['h50']:.4f}  MRR {m['mrr']:.4f}  MR {m['mr']:9,.0f}  {m['sec']}s (train {m['sec_train']}s + eval {m['sec_eval']}s)",
            flush=True,
        )
        json.dump(log, open(cfg.out, "w"), indent=1)

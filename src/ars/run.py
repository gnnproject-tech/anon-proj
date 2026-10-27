import argparse, json, os, sys, numpy as np, torch
from dataclasses import fields
from .config import Config
from .data import Data
from .graph.schedule import Schedule
from .graph.csr import Sampler
from .storage.state import State
from .storage.state_paged import PagedState
from .storage.replica import Replica
from .storage.gradbuf import GradBuf
from .train.model import Model
from .train import loop


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f in fields(Config):
        if f.type is bool:
            ap.add_argument(
                f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default
            )
        else:
            ap.add_argument(
                f"--{f.name}",
                type=(f.type if f.type in (int, float, str) else str),
                default=f.default,
            )
    a = ap.parse_args(argv)
    cfg = Config(**vars(a))
    assert (cfg.sched is not None) == cfg.windowed, "FULL takes no schedule; every other condition needs one"
    if cfg.always_frac > 0:
        assert (
            0 < cfg.always_frac <= 1
            and cfg.always_sel in ("deg", "random")
            and cfg.always_device in ("cpu", "cuda")
        ), "always_frac (0,1], always_sel deg|random, always_device cpu|cuda"
        assert (
            cfg.paged and cfg.window_device == "cpu" and cfg.host_stale == 0
        ), "S requires paged state, a host write window (window_device cpu), and host_stale 0"
        assert (
            cfg.replica in ("int8", "fp32") and cfg.replica_frac >= 1.0
        ), "S requires a full read view (int8|fp32)"
        assert (
            cfg.cond in ("C1n", "C1") and cfg.sampler == "exact" and cfg.nbr_grad == "none"
        ), "S: cond C1n|C1 (C1 = global negatives), exact sampler, gradient buffer off"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device
    from .train.prof import MemMon

    mon = MemMon(dev)
    data = Data(
        cfg.dataset, dev, cfg.edges_device or None, cfg.nbr_dir, cfg.eval_queries, cfg.eval_split
    )
    if cfg.eval_sampled:
        data.full_eval = False
    sched = Schedule(cfg.sched, data.P, data.cum, dev) if cfg.windowed else None
    if cfg.paged:
        assert cfg.windowed, "paged state needs a window schedule"
        state = PagedState(
            cfg, data, dev, f"{cfg.state_root}/{cfg.dataset}_{os.getpid()}", cfg.direct_io
        )
        state.alloc(
            sched.c,
            sched.max_out,
            sched.max_in,
            always_frac=cfg.always_frac,
            always_sel=cfg.always_sel,
            always_seed=cfg.always_seed,
            always_device=cfg.always_device,
        )
    else:
        state = State(cfg, data, dev)
    if cfg.windowed and cfg.replica != "live":
        state.replica = Replica(
            data.N,
            cfg.d,
            cfg.replica,
            cfg.replica_frac,
            cfg.refresh_delta,
            data.deg,
            dev,
            home=cfg.state_device,
        )
        if cfg.gpu_cache_frac > 0 and cfg.state_device == "cpu":
            state.replica.enable_gpu_cache(data.deg, cfg.gpu_cache_frac)
    if cfg.nbr_grad == "deferred":
        state.gbuf = GradBuf(
            data.N,
            cfg.d,
            cfg.gbuf,
            cfg.gbuf_frac,
            cfg.gbuf_cap,
            data.deg,
            dev,
            home=cfg.state_device,
            acc_dtype=cfg.gbuf_acc,
            keep_q=cfg.gbuf_q,
        )
    sampler = Sampler(data, cfg.fanout, data.hub_sampler, dev, mode=cfg.sampler)
    model = Model(cfg, data, state, sampler, dev)
    from .train.prof import Prof

    model.prof = Prof(cfg.profile, dev)
    model.memmon = mon
    mem = dict(
        buf_bytes=(8 * sched.c * data.PSZ * cfg.d) if (cfg.paged and sched) else None,
        rep_bytes=state.replica.bytes if state.replica else 0,
        gbuf_bytes=state.gbuf.bytes if state.gbuf else 0,
        s_rows=getattr(state, "kS", 0),
        s_bytes=getattr(state, "s_bytes", 0),
        s_map_bytes=getattr(state, "s_map_bytes", 0),
    )
    if sched:
        print(f"{len(sched.S)} windows, {sched.c} resident partitions per window ({sched.c/data.P*100:.1f}% resident)")
    log = dict(args=vars(a), mem=mem, epochs=[])
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    loop.run(cfg, data, state, model, sampler, sched, log)


if __name__ == "__main__":
    main()

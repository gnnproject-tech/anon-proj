import sys
import marius as m
import torch


def main():
    cfg = m.config.loadConfig(sys.argv[1], save=True)
    m.manager.marius_train(cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    st = {
        l.split(":")[0]: l.split()[1]
        for l in open("/proc/self/status")
        if l.startswith(("VmHWM", "VmRSS"))
    }
    print(
        f"MEMSTAT gpu_alloc_max_gb={torch.cuda.max_memory_allocated()/2**30:.2f} gpu_reserved_max_gb={torch.cuda.max_memory_reserved()/2**30:.2f} "
        f"vm_hwm_gb={int(st['VmHWM'])/2**20:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

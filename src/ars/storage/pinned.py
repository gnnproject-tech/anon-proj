import torch


def _pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


class PinnedArena:
    def __init__(self, align=256):
        self.chunks = []
        self.align = align
        self.requested = 0
        self.allocated = 0

    def _fit(self, nbytes):
        for i, (t, used) in enumerate(self.chunks):
            off = (used + self.align - 1) // self.align * self.align
            if off + nbytes <= t.numel():
                self.chunks[i] = (t, off + nbytes)
                return t, off
        size = _pow2(nbytes)
        t = torch.empty(size, dtype=torch.uint8)
        t = t.pin_memory() if torch.cuda.is_available() else t
        self.allocated += size
        self.chunks.append((t, nbytes))
        return t, 0

    def alloc(self, shape, dtype=torch.float32):
        shape = tuple(shape)
        n = 1
        for s in shape:
            n *= s
        nbytes = n * torch.empty(0, dtype=dtype).element_size()
        t, off = self._fit(nbytes)
        self.requested += nbytes
        return t[off : off + nbytes].view(dtype).view(shape)

    def stats(self):
        return dict(
            pinned_requested=self.requested,
            pinned_allocated=self.allocated,
            chunks=[c[0].numel() for c in self.chunks],
        )


def pin_exact(shape, dtype=torch.float32, zero=True):
    t = torch.zeros(shape, dtype=dtype) if zero else torch.empty(shape, dtype=dtype)
    if not torch.cuda.is_available():
        return t
    rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
    assert int(rc) == 0, f"cudaHostRegister failed with {rc}"
    return t

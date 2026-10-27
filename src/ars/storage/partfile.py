import os, mmap, numpy as np, torch

ALIGN = 4096


class PartitionStore:
    def __init__(self, root, P, PSZ, d, direct=True):
        self.root, self.P, self.PSZ, self.d, self.direct = root, P, PSZ, d, direct
        self.nbytes = PSZ * d * 4
        self.pad = (self.nbytes + ALIGN - 1) // ALIGN * ALIGN
        os.makedirs(f"{root}/emb", exist_ok=True)
        os.makedirs(f"{root}/opt", exist_ok=True)
        self.read_bytes = 0
        self.write_bytes = 0
        self.reads = 0
        self.writes = 0
        flags = os.O_RDWR | os.O_CREAT | (os.O_DIRECT if direct else 0)
        self.fd = {
            k: [os.open(f"{root}/{k}/p{i}.bin", flags, 0o644) for i in range(P)]
            for k in ("emb", "opt")
        }
        self._buf = mmap.mmap(-1, self.pad)
        self._np = np.frombuffer(self._buf, dtype=np.float32, count=PSZ * d).reshape(PSZ, d)

    def path(self, k, i):
        return f"{self.root}/{k}/p{i}.bin"

    def init_from(self, E, opt_state):
        N = E.shape[0]
        for i in range(self.P):
            lo, hi = i * self.PSZ, min((i + 1) * self.PSZ, N)
            for k, src in (("emb", E), ("opt", opt_state)):
                self._np[:] = 0
                self._np[: hi - lo] = src[lo:hi].numpy()
                os.pwrite(self.fd[k][i], self._buf, 0)
        for k in ("emb", "opt"):
            for i in range(self.P):
                os.fsync(self.fd[k][i])

    def init_parts(self, gen, N):
        zero = torch.zeros(self.PSZ, self.d)
        for i in range(self.P):
            lo, hi = i * self.PSZ, min((i + 1) * self.PSZ, N)
            E0 = gen(i)
            for k, src in (("emb", E0), ("opt", zero)):
                self._np[:] = 0
                self._np[: hi - lo] = src[: hi - lo].numpy()
                os.pwrite(self.fd[k][i], self._buf, 0)
        for k in ("emb", "opt"):
            for i in range(self.P):
                os.fsync(self.fd[k][i])

    def load(self, i, emb_out, opt_out):
        buf = mmap.mmap(-1, self.pad)
        arr = np.frombuffer(buf, dtype=np.float32, count=self.PSZ * self.d).reshape(
            self.PSZ, self.d
        )
        for k, out in (("emb", emb_out), ("opt", opt_out)):
            n = os.preadv(self.fd[k][i], [buf], 0)
            assert n == self.pad, (k, i, n)
            out.copy_(torch.from_numpy(arr))
            self.read_bytes += self.nbytes
            self.reads += 1
        del arr
        buf.close()

    def store(self, i, emb_in, opt_in):
        buf = mmap.mmap(-1, self.pad)
        arr = np.frombuffer(buf, dtype=np.float32, count=self.PSZ * self.d).reshape(
            self.PSZ, self.d
        )
        for k, src in (("emb", emb_in), ("opt", opt_in)):
            arr[:] = src.numpy()
            os.pwrite(self.fd[k][i], buf, 0)
            self.write_bytes += self.nbytes
            self.writes += 1
        del arr
        buf.close()

    def load_all(self, N, dev):
        E = torch.empty(self.P * self.PSZ, self.d)
        tmp = torch.empty(self.PSZ, self.d)
        tmpo = torch.empty(self.PSZ, self.d)
        rb = self.read_bytes
        for i in range(self.P):
            n = os.preadv(self.fd["emb"][i], [self._buf], 0)
            E[i * self.PSZ : (i + 1) * self.PSZ] = torch.from_numpy(self._np)
        self.eval_read_bytes = getattr(self, "eval_read_bytes", 0) + self.P * self.nbytes
        return E[:N] if dev == "cpu" else E[:N].to(dev)

    def pop_stat(self):
        r = dict(
            io_read_bytes=self.read_bytes,
            io_write_bytes=self.write_bytes,
            io_reads=self.reads,
            io_writes=self.writes,
        )
        self.read_bytes = self.write_bytes = self.reads = self.writes = 0
        return r

    def close(self):
        for k in self.fd:
            for f in self.fd[k]:
                os.close(f)

import numpy as np, torch


class Schedule:
    def __init__(self, spec, P, cum, dev):
        self.P, self.cum, self.dev = P, cum, dev
        self.S, self.B = [], []
        if spec.startswith("seq"):

            body = spec[3:]
            if ":" in body:
                c, m = (int(x) for x in body.split(":"))
                assert 1 <= m < c
                g = c - m
                groups = [list(range(i, min(i + g, P))) for i in range(0, P, g)]
                covered = set()

                def emit(res):
                    bk = [(i, j) for i in res for j in res if (i, j) not in covered]
                    if bk:
                        covered.update(bk)
                        self.S.append(list(res))
                        self.B.append(bk)

                for a in range(len(groups)):
                    emit(groups[a])
                    for b in range(a + 1, len(groups)):
                        for k in range(0, len(groups[b]), m):
                            emit(groups[a] + groups[b][k : k + m])
            else:
                c = int(body) if body else 2
                h = max(1, c // 2)
                blocks = [list(range(i, min(i + h, P))) for i in range(0, P, h)]
                nb = len(blocks)
                covered = set()

                pairs = [(0, 0)]
                for a in range(nb):
                    bs = list(range(a + 1, nb))
                    pairs += [(a, b) for b in (bs[::-1] if a % 2 else bs)]
                for a, b in pairs:
                    res = blocks[a] if a == b else blocks[a] + blocks[b]
                    bk = [(i, j) for i in res for j in res if (i, j) not in covered]
                    if not bk:
                        continue
                    covered.update(bk)
                    self.S.append(res)
                    self.B.append(bk)
            assert len(covered) == P * P, f"sequential schedule incomplete: {len(covered)}/{P*P}"
        else:
            for ln in open(spec):
                if ln[0] == "S":
                    self.S.append([int(x) for x in ln.split()[1:]])
                elif ln[0] == "B":
                    self.B.append(
                        [(int(x), int(y)) for x, y in (t.split(",") for t in ln.split()[1:])]
                    )
        self.c = max(len(s) for s in self.S)

        self.max_out = max(
            [len(set(self.S[i - 1]) - set(self.S[i])) for i in range(1, len(self.S))]
            + [len(self.S[0])]
        )
        self.max_in = max(
            [len(set(self.S[i]) - set(self.S[i - 1])) for i in range(1, len(self.S))] + [0]
        )
        if len(self.S) == 1:
            self.max_out = self.max_in = 0

    def plan(self, perm):
        out, P, cum = [], self.P, self.cum
        for s, b in zip(self.S, self.B):
            res = [int(perm[i]) for i in s]
            rng = [
                (int(cum[perm[i] * P + perm[j]]), int(cum[perm[i] * P + perm[j] + 1]))
                for (i, j) in b
                if cum[perm[i] * P + perm[j] + 1] > cum[perm[i] * P + perm[j]]
            ]
            if rng:
                out.append((torch.tensor(res, device=self.dev), rng))
        return out

    def edge_index(self, rng):
        return torch.cat([torch.arange(lo, hi, device=self.dev) for lo, hi in rng])

    def windows(self, perm):
        return [(res, self.edge_index(rng)) for res, rng in self.plan(perm)]

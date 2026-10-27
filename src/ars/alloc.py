from dataclasses import dataclass


@dataclass
class Plan:
    c: int
    k_r: float
    k_g: float
    buf_mb: float
    rep_mb: float
    gbuf_mb: float

    @property
    def total_mb(self):
        return self.buf_mb + self.rep_mb + self.gbuf_mb


def allocate(
    M_mb,
    N,
    d,
    P,
    PSZ,
    c_min=2,
    c_floor_frac=0.0,
    k_steps=(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01),
    occ=0.8,
):
    M = M_mb * 1e6
    part = 8 * PSZ * d
    row = N * d
    c_min = max(c_min, int(P * c_floor_frac))
    best = None
    for k_r in k_steps:
        for k_g in k_steps:
            if k_g > k_r:
                continue
            fixed = k_r * row + k_g * row * occ
            c = int((M - fixed) // part)
            if c < c_min:
                continue
            c = min(c, P)
            plan = Plan(c, k_r, k_g, c * part / 1e6, k_r * row / 1e6, k_g * row * occ / 1e6)

            key = (k_r, k_g, c)
            if best is None or key > best[0]:
                best = (key, plan)
    return best[1] if best else None


if __name__ == "__main__":
    import sys

    for name, N, P, PSZ, budgets in [
        ("collab", 235868, 64, 3686, (6, 12, 24, 47)),
        ("wikikg2", 2500604, 256, 9768, (125, 250, 470)),
    ]:
        print(name)
        for M in budgets:
            p = allocate(M, N, 50, P, PSZ, c_floor_frac=(0.06 if name == "collab" else 0.0))
            print(
                f"  M={M:4d} MB -> c={p.c:3d} ({p.c/P*100:4.1f}%)  copy {p.k_r*100:4.0f}%  buffer {p.k_g*100:4.0f}%   = {p.buf_mb:.1f} + {p.rep_mb:.1f} + {p.gbuf_mb:.1f} = {p.total_mb:.1f} MB"
                if p
                else f"  M={M} MB -> infeasible (c_min)"
            )

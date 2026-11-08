"""Hair flower + black hair ribbon (with physics tails), brooch, earrings."""
from __future__ import annotations

import math

import numpy as np

from .body import flower_part, ribbon_tube, sphere_part
from .builder import w_chain, w_single
from .geom import catmull_rom, merge_parts, normalize
from .physics import G_ACC, Physics

FLOWER_C = np.array([1.02, 18.95, -0.30])
FLOWER_N = normalize(np.array([0.80, 0.25, -0.55]))
RIBBON_KNOT = np.array([1.12, 18.55, -0.02])
RIBBON_BONE_Y = (18.35, 17.75, 17.15)
RIBBON_TAIL_Y = 16.55


class AccessoryBuilder:
    def __init__(self, mb):
        self.mb = mb
        self.sk = mb.sk

    def add_bones(self):
        sk = self.sk
        self.ribbon_chains = []
        for j, dx in enumerate((-0.02, 0.10)):
            pts = []
            for k, y in enumerate(RIBBON_BONE_Y + (RIBBON_TAIL_Y,)):
                t = k / 3
                pts.append(np.array([RIBBON_KNOT[0] + 0.10 + dx + 0.06 * t, y, RIBBON_KNOT[2] + 0.05 + 0.18 * j * t]))
            names = []
            parent = "頭"
            for k in range(len(RIBBON_BONE_Y)):
                nm = "リボン%d_%d" % (j + 1, k + 1)
                sk.add(nm, "ribbon%d_%d" % (j + 1, k + 1), pts[k], parent)
                names.append(nm)
                parent = nm
            for k in range(len(names) - 1):
                sk.connect(names[k], names[k + 1])
            sk.bones[sk[names[-1]]].tail = tuple(pts[-1] - pts[-2])
            self.ribbon_chains.append((names, np.array(pts)))

    def build(self):
        mb, sk = self.mb, self.sk
        nb = mb.nb
        head = sk["頭"]
        # big hair flower: two petal layers + centre
        fl = merge_parts([flower_part(FLOWER_C, FLOWER_N, 0.40, 14, layers=2),
                          flower_part(FLOWER_C + FLOWER_N * 0.03, FLOWER_N, 0.24, 10, layers=1)])
        fl.W = w_single(fl.n, nb, head)
        mb.add("flower", fl)
        c = sphere_part(FLOWER_C + FLOWER_N * 0.06, 0.11, 12, 8, squash=(1, 1, 1))
        c.pos = FLOWER_C + FLOWER_N * 0.05 + (c.pos - (FLOWER_C + FLOWER_N * 0.06)) * np.array([1, 1, 1])
        c.uv = 0.5 + (c.pos[:, [0, 1]] - (FLOWER_C + FLOWER_N * 0.05)[[0, 1]]) / 0.24
        c.W = w_single(c.n, nb, head)
        mb.add("flower_center", c)

        # ribbon bow (loops + knot) behind the flower
        parts = []
        k0 = RIBBON_KNOT
        a = normalize(np.array([0.15, 0.0, 1.0]))      # along the head side (front-back)
        up = np.array([0, 1.0, 0])
        out = normalize(np.cross(up, a))
        if out[0] < 0:
            out = -out
        for sgn, ang in ((1, 0.35), (-1, -0.35)):
            pts = np.array([[0, 0], [0.12, 0.10], [0.30, 0.16], [0.42, 0.06], [0.38, -0.08], [0.20, -0.10],
                            [0.06, -0.04]])
            path = catmull_rom(pts, 28, closed=True)
            dirv = normalize(a * math.cos(ang) + up * math.sin(ang)) * sgn
            perp = normalize(np.cross(out, dirv)) * sgn
            P = k0[None] + path[:, :1] * dirv + path[:, 1:2] * perp + (0.06 * (path[:, 0] / 0.42) ** 2)[:, None] * out
            parts.append(ribbon_tube(P, 0.14, 0.03, closed=True, up=out))
        parts.append(sphere_part(k0 + out * 0.03, 0.08, 10, 8, squash=(0.8, 1.0, 1.0)))
        g = merge_parts(parts)
        g.W = w_single(g.n, nb, head)
        mb.add("cloth_black_satin", g)
        # tails (weighted to physics chains)
        for names, pts in self.ribbon_chains:
            path = catmull_rom(np.vstack([k0 + out * 0.02, pts]), 30)
            w = np.linspace(0.13, 0.17, len(path))
            t = ribbon_tube(path, w, 0.025, up=out, vcut_end=0.10)
            keys = [(-(k0[1]), head), (-(pts[0][1] - 0.05), sk[names[0]])]
            for k in range(1, len(names)):
                keys.append((-(pts[k][1] + 0.12), sk[names[k - 1]]))
                keys.append((-(pts[k][1] - 0.12), sk[names[k]]))
            t.W = w_chain(-t.pos[:, 1], keys, nb)
            mb.add("cloth_black_satin", t)

        # brooch flower at the collar
        bc = np.array([0.0, 15.98, -0.50])
        bf = flower_part(bc, normalize(np.array([0, 0.25, -1.0])), 0.16, 12, layers=2)
        bf.W = w_single(bf.n, nb, sk["上半身2"])
        mb.add("flower", bf)
        bcc = sphere_part(bc + np.array([0, 0.01, -0.04]), 0.05, 10, 6)
        bcc.uv[:] = (0.5, 0.5)
        bcc.W = w_single(bcc.n, nb, sk["上半身2"])
        mb.add("flower_center", bcc)

        # cross earrings
        parts = []
        for side in (1, -1):
            top = np.array([0.92 * side, 17.05, -0.08])
            chain = [sphere_part(top + np.array([0, -0.05 * i, 0]), 0.016, 6, 4) for i in range(4)]
            cc = top + np.array([0, -0.32, 0])
            bar_v = _box(cc + np.array([0, 0.02, 0]), (0.018, 0.10, 0.012))
            bar_h = _box(cc + np.array([0, 0.05, 0]), (0.06, 0.016, 0.012))
            parts += chain + [bar_v, bar_h]
        g = merge_parts(parts)
        g.W = w_single(g.n, nb, head)
        mb.add("gold", g)

    def physics(self, ph: Physics):
        for j, (names, pts) in enumerate(self.ribbon_chains):
            prev = "頭"
            for k, nm in enumerate(names):
                ph.capsule(nm, "ribbon%d_%d" % (j + 1, k + 1), nm, pts[k], pts[k + 1], 0.06, group=G_ACC, mode=2,
                           mass=0.2, lin=0.8, ang=0.9, fric=0.3)
                ph.joint("J_" + nm, "J_ribbon%d_%d" % (j + 1, k + 1), prev, nm, pts[k], (15 + 5 * k, 8, 15 + 5 * k),
                         spring_rot=(10, 5, 10))
                prev = nm


def _box(c, h):
    from .geom import Part
    c = np.asarray(c, dtype=np.float64)
    hx, hy, hz = h
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]) + c
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    p = Part(v, f)
    from .geom import orient_outward
    orient_outward(p, lambda cc: np.tile(c, (len(cc), 1)))
    # orient_outward flips globally; verify per-face consistency by recomputing
    p.compute_normals(weld=False)
    return p

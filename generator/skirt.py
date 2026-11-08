"""Layered skirt: pleated white underskirt, black over-layer with orange
lining, white drape with orange lining, orange front tabs.  16 physics chains."""
from __future__ import annotations

import math

import numpy as np

from .body import torso_point, TORSO_ZC
from .builder import w_chain
from .geom import Part, grid_faces, normalize, orient_outward
from .physics import G_SKIRT, Physics, rot_y_to

Y_TOP = 12.45
Y_HEM = 6.45
N_CHAIN = 16
BONE_TAU = (0.05, 0.37, 0.69)
TAIL_TAU = 1.0


def _flare(theta):
    c = np.cos(theta)
    return 1.30 + 0.18 * np.clip(-c, 0, 1) + 0.05 * np.abs(np.sin(theta))


def _f(tau):
    tau = np.clip(tau, 0, None)
    return np.where(tau < 1, 1 - (1 - np.clip(tau, 0, 1)) ** 1.6, 1.0)


def skirt_point(theta, y, extra=0.0):
    theta = np.asarray(theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    top = torso_point(np.full(theta.shape, Y_TOP), theta, 0.06)
    u = top.copy()
    u[..., 1] = 0
    u[..., 2] -= TORSO_ZC
    r_top = np.linalg.norm(u, axis=-1)
    u = u / r_top[..., None]
    tau = (Y_TOP - y) / (Y_TOP - Y_HEM)
    r = r_top + _flare(theta) * _f(tau) + extra
    P = np.zeros(theta.shape + (3,))
    P[..., 0] = u[..., 0] * r
    P[..., 1] = y
    P[..., 2] = u[..., 2] * r + TORSO_ZC
    return P


def _tri(x):
    """triangle wave in [-1, 1] with period 1"""
    f = x - np.floor(x)
    return 4 * np.abs(f - 0.5) - 1


class SkirtBuilder:
    def __init__(self, mb):
        self.mb = mb
        self.sk = mb.sk
        self.nb = mb.nb

    # -- bones -----------------------------------------------------------
    def add_bones(self):
        sk = self.sk
        self.chains = []
        for i in range(N_CHAIN):
            th = 2 * math.pi * i / N_CHAIN
            names = []
            parent = "下半身"
            pts = []
            for k, tau in enumerate(BONE_TAU + (TAIL_TAU,)):
                y = Y_TOP - tau * (Y_TOP - Y_HEM)
                pts.append(skirt_point(np.array(th), np.array(y), 0.05 + 0.06 * tau))
            for k in range(len(BONE_TAU)):
                nm = "スカート%02d_%d" % (i, k + 1)
                sk.add(nm, "skirt%02d_%d" % (i, k + 1), pts[k], parent)
                names.append(nm)
                parent = nm
            for k in range(len(BONE_TAU) - 1):
                sk.connect(names[k], names[k + 1])
            sk.bones[sk[names[-1]]].tail = tuple(pts[-1] - pts[-2])
            self.chains.append((th, names, np.array(pts)))

    def weights(self, P):
        sk, nb = self.sk, self.mb.nb
        th = np.mod(np.arctan2(P[:, 0], -(P[:, 2] - TORSO_ZC)), 2 * np.pi)
        tau = (Y_TOP - P[:, 1]) / (Y_TOP - Y_HEM)
        a = th / (2 * np.pi / N_CHAIN)
        i0 = np.floor(a).astype(int) % N_CHAIN
        i1 = (i0 + 1) % N_CHAIN
        wa = a - np.floor(a)
        W = np.zeros((len(P), nb))
        cache = {}
        for i in range(N_CHAIN):
            names = self.chains[i][1]
            keys = [(-0.02, sk["下半身"]), (0.10, sk[names[0]]), (0.30, sk[names[0]]), (0.44, sk[names[1]]),
                    (0.62, sk[names[1]]), (0.76, sk[names[2]])]
            cache[i] = keys
        for i in range(N_CHAIN):
            m0 = i0 == i
            m1 = i1 == i
            m = m0 | m1
            if not m.any():
                continue
            Wi = w_chain(tau[m], cache[i], nb)
            f = np.where(m0[m], 1 - wa[m], 0) + np.where(m1[m], wa[m], 0)
            W[m] += Wi * f[:, None]
        return W

    # -- geometry ----------------------------------------------------------
    def _layer(self, th, y_hem, extra_fn, rows, mat_out, mat_in=None, pleats=0, wrap=False, lining_gap=0.014):
        """th: (C,) angles; y_hem: (C,) hem heights; extra_fn(tau)->radial offset."""
        C = len(th)
        t = np.linspace(0, 1, rows)
        Y = Y_TOP + (y_hem[None, :] - Y_TOP) * (t[:, None] ** 1.0)
        TH = np.tile(th, (rows, 1))
        tau = (Y_TOP - Y) / (Y_TOP - Y_HEM)
        ex = extra_fn(tau)
        if pleats:
            ex = ex + 0.075 * np.clip(tau, 0, 1) ** 0.75 * _tri(TH * pleats / (2 * np.pi))
        P = skirt_point(TH, Y, ex)
        uu = np.tile(np.linspace(0, 1, C), (rows, 1))
        uv = np.stack([uu, np.tile(t[:, None], (1, C))], -1)
        if wrap:
            P = np.concatenate([P, P[:, :1]], 1)
            uv = np.concatenate([uv, uv[:, :1]], 1)
            C += 1
        outer = Part(P.reshape(-1, 3), grid_faces(rows, C), uv.reshape(-1, 2))
        ref = lambda c: np.c_[np.zeros(len(c)), c[:, 1], np.full(len(c), TORSO_ZC)]
        orient_outward(outer, ref)
        outer.compute_normals()
        outer.W = self.weights(outer.pos)
        self.mb.add(mat_out, outer)
        if mat_in:
            P2 = skirt_point(TH, Y, ex - lining_gap)
            if wrap:
                P2 = np.concatenate([P2, P2[:, :1]], 1)
            inner = Part(P2.reshape(-1, 3), outer.faces[:, ::-1].copy(), uv.reshape(-1, 2))
            inner.compute_normals()
            inner.W = self.weights(inner.pos)
            self.mb.add(mat_in, inner)
        return outer

    def build(self):
        # 1. pleated white underskirt (full circle, double sided)
        C = 192
        th = np.linspace(0, 2 * np.pi, C, endpoint=False)
        hem = Y_HEM + 0.06 * np.cos(3 * th) - 0.05 * np.clip(-np.cos(th), 0, 1)
        self._layer(th, hem, lambda tau: 0.0 * tau, 20, "skirt_white", pleats=48, wrap=True)

        # 2. black layer, open at the front, handkerchief points, orange lining
        g = 0.72
        th = np.linspace(g, 2 * np.pi - g, 130)
        P = 5
        ph = (th - g) / (2 * np.pi - 2 * g) * P
        h = np.abs((ph - np.floor(ph)) - 0.5) * 2
        hem = 6.15 + 0.85 * h ** 1.4
        hem[0] = hem[-1] = 7.6
        self._layer(th, hem, lambda tau: 0.06 + 0.075 * np.clip(tau, 0, 1.2), 16, "skirt_black", "skirt_orange")

        # 3. white drape with orange lining (shorter, interleaved points)
        for g0, g1 in ((1.05, 2.45), (2 * np.pi - 2.45, 2 * np.pi - 1.05)):
            th = np.linspace(g0, g1, 60)
            P = 2
            ph = (th - g0) / (g1 - g0) * P + 0.5
            h = np.abs((ph - np.floor(ph)) - 0.5) * 2
            hem = 8.35 + 0.9 * h ** 1.3
            hem[0] = hem[-1] = 10.2
            self._layer(th, hem, lambda tau: 0.13 + 0.12 * np.clip(tau, 0, 1.2), 12, "skirt_white2",
                        "skirt_orange")

        # 4. orange tabs at the front sides
        for c0 in (0.36, -0.36):
            th = np.linspace(c0 - 0.08, c0 + 0.08, 7)
            u = (th - th[0]) / (th[-1] - th[0])
            hem = 8.2 + 0.7 * np.abs(u - 0.5) * 2
            self._layer(th, hem, lambda tau: 0.19 + 0.15 * np.clip(tau, 0, 1.2), 12, "skirt_orange_out",
                        "skirt_orange")

    # -- physics -----------------------------------------------------------
    def physics(self, ph: Physics):
        for i, (th, names, pts) in enumerate(self.chains):
            prev = "下半身"
            for k, nm in enumerate(names):
                p0, p1 = pts[k], pts[k + 1]
                mid = (p0 + p1) / 2
                L = np.linalg.norm(p1 - p0)
                r = np.linalg.norm([mid[0], mid[2] - TORSO_ZC])
                half_w = 2 * math.pi * r / N_CHAIN * 0.5 * 0.80
                rot = rot_y_to(p1 - p0)
                ph.body(nm, "skirt%02d_%d" % (i, k + 1), nm, 1, (0.03, L * 0.5 * 0.92, half_w), mid, rot,
                        group=G_SKIRT, mode=2, mass=0.5 - 0.1 * k, lin=0.8, ang=0.8, rest=0.0, fric=0.2)
                # joint frame aligned with the chain: local Z = tangential axis (in/out swing)
                lim = ((3, 5, 8)[k], 2, (5, 8, 12)[k])
                ph.joint("J_" + nm, "J_skirt%02d_%d" % (i, k + 1), prev, nm, p0, lim,
                         spring_rot=(80, 20, 80), rot=rot)
                prev = nm
        # horizontal joints between neighbouring chains keep the circumference
        for i in range(N_CHAIN):
            a_names, b_names = self.chains[i][1], self.chains[(i + 1) % N_CHAIN][1]
            a_pts, b_pts = self.chains[i][2], self.chains[(i + 1) % N_CHAIN][2]
            for k in range(len(a_names)):
                pa = (a_pts[k] + a_pts[k + 1]) / 2
                pb = (b_pts[k] + b_pts[k + 1]) / 2
                ph.joint("J_" + a_names[k] + "横", "J_skirt%02d_%d_h" % (i, k + 1), a_names[k], b_names[k],
                         (pa + pb) / 2, (10, 10, 10), spring_rot=(20, 20, 20), pos_lim=(0.08, 0.08, 0.08),
                         spring_pos=(50, 50, 50))

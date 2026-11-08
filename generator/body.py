"""Body, blouse, sleeves, hands, legs, socks and shoes."""
from __future__ import annotations

import math

import numpy as np

from .builder import w_chain, w_mix, w_single
from .geom import (Part, catmull_rom, frames_along, grid_faces, merge_parts, normalize, orient_outward,
                   pchip, sd_capsule, sd_ellipsoid, sd_round_box, smin, smax, smoothstep, surface_nets,
                   tube_part, vertex_normals)
from .skeleton import (ANKLE, ARM_DIR, ELBOW, FINGERS, HAND_X, HAND_Y, HAND_Z, KNEE, LEG, SHOULDER, TOE,
                       WRIST, finger_chain_local, hand_local_to_world, thumb_chain_local, ZEN)


def axis_inside(axis_pts):
    A = np.asarray(axis_pts)

    def fn(c):
        d = ((c[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        return A[np.argmin(d, axis=1)]
    return fn


# ---------------------------------------------------------------------------
# torso profile
# ---------------------------------------------------------------------------
#                y     rx     front  back
_TORSO = np.array([
    [10.95, 0.55, 0.32, 0.38],
    [11.05, 0.90, 0.46, 0.52],
    [11.25, 1.12, 0.58, 0.66],
    [11.55, 1.25, 0.63, 0.73],
    [11.90, 1.29, 0.64, 0.74],
    [12.40, 1.15, 0.60, 0.66],
    [12.90, 1.00, 0.57, 0.58],
    [13.20, 0.95, 0.56, 0.56],
    [13.60, 1.00, 0.60, 0.56],
    [14.10, 1.10, 0.70, 0.58],
    [14.60, 1.16, 0.74, 0.60],
    [15.00, 1.20, 0.70, 0.62],
    [15.30, 1.28, 0.60, 0.62],
    [15.50, 1.27, 0.50, 0.56],
    [15.65, 1.08, 0.45, 0.50],
    [15.80, 0.72, 0.38, 0.45],
    [15.92, 0.44, 0.31, 0.41],
    [16.02, 0.34, 0.26, 0.40],
])
_RX = pchip(_TORSO[:, 0], _TORSO[:, 1])
_ZF = pchip(_TORSO[:, 0], _TORSO[:, 2])
_ZB = pchip(_TORSO[:, 0], _TORSO[:, 3])
TORSO_ZC = 0.10
TORSO_POW = 2.3


def bust(y):
    return 0.13 * np.exp(-((y - 14.45) / 0.42) ** 2)


def torso_point(y, th, inflate=0.0):
    """theta = 0 at front centre, increasing toward character's left (+x)."""
    y = np.asarray(y, dtype=np.float64)
    th = np.asarray(th, dtype=np.float64)
    rx = _RX(y) + inflate
    zf = _ZF(y) + inflate
    zb = _ZB(y) + inflate
    s, c = np.sin(th), np.cos(th)
    e = 2.0 / TORSO_POW
    x = rx * np.sign(s) * np.abs(s) ** e
    zz = np.abs(c) ** e
    z = np.where(c > 0, -zf * zz, zb * zz)
    front = c > 0
    z = z - np.where(front, bust(y) * np.exp(-((np.abs(x) - 0.42) / 0.30) ** 2) * c, 0)
    return np.stack([x, y, z + TORSO_ZC], -1)


def torso_rings(ys, K, inflate=0.0):
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    Y, TH = np.meshgrid(ys, th, indexing="ij")
    return torso_point(Y, TH, inflate)


class BodyBuilder:
    def __init__(self, mb):
        self.mb = mb
        self.sk = mb.sk
        self.nb = mb.nb

    def B(self, name):
        return self.sk[name]

    # ------------------------------------------------------------------
    def torso_weights(self, P):
        sk, nb = self.sk, self.nb
        y = P[:, 1]
        W = w_chain(y, [(11.6, sk["下半身"]), (12.35, sk["下半身"]), (12.85, sk["上半身"]),
                        (13.45, sk["上半身"]), (14.25, sk["上半身2"]), (15.85, sk["上半身2"]),
                        (16.15, sk["首"])], nb)
        ax = np.abs(P[:, 0])
        w_sh = smoothstep(0.42, 1.05, ax) * smoothstep(14.5, 15.3, y)
        w_arm = smoothstep(1.0, 1.42, ax) * smoothstep(14.6, 15.2, y)
        for side, lr in ((1, "左"), (-1, "右")):
            m = (P[:, 0] * side) > 0
            Wsh = np.zeros_like(W)
            Wsh[:, sk[lr + "肩"]] = 1 - w_arm
            Wsh[:, sk[lr + "腕"]] = w_arm
            W[m] = w_mix(W[m], Wsh[m], (w_sh * 0.9)[m])
        # legs pull the bottom of the hips
        w_leg = smoothstep(11.75, 11.05, y) * smoothstep(0.0, 0.35, ax)
        for side, lr in ((1, "左"), (-1, "右")):
            m = (P[:, 0] * side) > 0
            Wl = np.zeros_like(W)
            Wl[:, sk[lr + "足D"]] = 1
            W[m] = w_mix(W[m], Wl[m], (w_leg * 0.8)[m])
        return W

    def torso(self):
        K = 56
        ys_top = np.r_[np.linspace(12.25, 15.65, 26), np.linspace(15.72, 16.02, 5)]
        rings = torso_rings(ys_top, K)
        blouse = tube_part(rings, v_coords=(ys_top - 12.0) / 4.0)
        orient_outward(blouse, lambda c: np.c_[np.zeros(len(c)), c[:, 1], np.full(len(c), TORSO_ZC)])
        blouse.compute_normals()
        blouse.W = self.torso_weights(blouse.pos)
        self.mb.add("cloth_white", blouse)

        ys_bot = np.r_[np.linspace(10.95, 12.3, 12)]
        rings = torso_rings(ys_bot, K)
        shorts = tube_part(rings, cap_start=True)
        orient_outward(shorts, lambda c: np.c_[np.zeros(len(c)), np.maximum(c[:, 1], 11.2), np.full(len(c), TORSO_ZC)])
        shorts.compute_normals()
        shorts.W = self.torso_weights(shorts.pos)
        self.mb.add("cloth_black", shorts)

    # ------------------------------------------------------------------
    def corset(self):
        K = 56
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        c = np.cos(th)
        y_bot = 12.32 - 0.16 * np.clip(c, 0, 1) ** 6
        y_top = 13.98 + 0.06 * (1 - np.clip(c, -1, 1)) * 0.5 - 0.05 * np.clip(c, 0, 1) ** 8
        R = 12
        t = np.linspace(0, 1, R)
        Y = y_bot[None, :] + (y_top - y_bot)[None, :] * t[:, None]
        TH = np.tile(th, (R, 1))
        P = torso_point(Y, TH, inflate=0.045)
        part = tube_part(P, v_coords=t)
        orient_outward(part, lambda cc: np.c_[np.zeros(len(cc)), cc[:, 1], np.full(len(cc), TORSO_ZC)])
        part.compute_normals()
        part.W = self.torso_weights(part.pos)
        self.mb.add("cloth_black_satin", part)
        # rim bands (slightly thicker)
        for yy, infl in ((y_bot + 0.03, 0.065), (y_top - 0.03, 0.065)):
            Y2 = np.stack([yy - 0.035, yy + 0.035])
            P2 = torso_point(Y2, np.tile(th, (2, 1)), inflate=infl)
            P2 = np.concatenate([torso_point(Y2[:1], th[None], 0.045), P2, torso_point(Y2[1:], th[None], 0.045)])
            band = tube_part(P2)
            orient_outward(band, lambda cc: np.c_[np.zeros(len(cc)), cc[:, 1], np.full(len(cc), TORSO_ZC)])
            band.compute_normals()
            band.W = self.torso_weights(band.pos)
            self.mb.add("cloth_black_satin", band)
        # gold buttons
        parts = []
        for x in (-0.20, 0.20):
            for yb in (12.55, 12.9, 13.25, 13.6):
                p = torso_point(np.array([yb]), np.array([math.asin(np.clip(x / 1.0, -1, 1))]), 0.06)[0]
                parts.append(sphere_part(p, 0.045, 8, 6, squash=(1, 1, 0.6)))
        # chain of beads hanging at the front right
        a = torso_point(np.array([12.30]), np.array([0.12]), 0.07)[0]
        b = torso_point(np.array([12.42]), np.array([1.05]), 0.07)[0]
        n = 22
        for i in range(n):
            t = i / (n - 1)
            p = a + (b - a) * t
            p[1] -= 0.32 * math.sin(math.pi * t)
            p[2] -= 0.06 * math.sin(math.pi * t)
            parts.append(sphere_part(p, 0.026, 6, 4))
        g = merge_parts(parts)
        g.W = self.torso_weights(g.pos)
        self.mb.add("gold", g)

    # ------------------------------------------------------------------
    def collar(self):
        sk, nb = self.sk, self.nb
        K = 40
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        cz = 0.12

        def ring(y, r, wav=0.0, nw=14, zoff=0.0):
            rr = r + wav * np.sin(nw * th)
            return np.stack([rr * np.sin(th), np.full(K, y), -rr * np.cos(th) * 0.95 + cz + zoff], -1)

        ys = [15.80, 15.95, 16.15, 16.32, 16.40]
        rs = [0.43, 0.37, 0.345, 0.345, 0.35]
        rings = np.stack([ring(y, r) for y, r in zip(ys, rs)])
        col = tube_part(rings)
        orient_outward(col, lambda c: np.c_[np.zeros(len(c)), c[:, 1], np.full(len(c), cz)])
        col.compute_normals()
        col.W = w_chain(col.pos[:, 1], [(15.85, sk["上半身2"]), (16.15, sk["首"])], nb)
        self.mb.add("cloth_white", col)
        # ruffle on top (double sided)
        rings = np.stack([ring(16.36, 0.345), ring(16.47, 0.40, 0.02), ring(16.55, 0.46, 0.045, 16)])
        fr = tube_part(rings)
        orient_outward(fr, lambda c: np.c_[np.zeros(len(c)), c[:, 1], np.full(len(c), cz)])
        fr.compute_normals()
        fr.W = w_single(fr.n, nb, sk["首"])
        self.mb.add("cloth_white_ds", fr)
        # base ruffle where the collar meets the blouse
        rings = np.stack([ring(15.86, 0.42), ring(15.80, 0.50, 0.025), ring(15.74, 0.58, 0.04, 18)])
        fr2 = tube_part(rings)
        orient_outward(fr2, lambda c: np.c_[np.zeros(len(c)), c[:, 1] + 0.3, np.full(len(c), cz)])
        fr2.compute_normals()
        fr2.W = w_single(fr2.n, nb, sk["上半身2"])
        self.mb.add("cloth_white_ds", fr2)

    # ------------------------------------------------------------------
    def jabot(self):
        """Cascading front ruffles."""
        nb, sk = self.nb, self.sk
        parts = []
        for k, (y0, y1, wmax) in enumerate(((15.62, 14.95, 0.20), (15.25, 14.45, 0.24), (14.85, 14.05, 0.26))):
            for side in (1, -1):
                R, C = 18, 6
                t = np.linspace(0, 1, R)
                u = np.linspace(0, 1, C)
                y = y0 + (y1 - y0) * t
                Y, U = np.meshgrid(y, u, indexing="ij")
                width = wmax * (0.35 + 0.65 * np.sin(np.pi * np.clip(t, 0, 1) * 0.9 + 0.15))[:, None]
                X = side * (0.01 + U * width)
                base = torso_point(Y, np.arcsin(np.clip(X / 1.1, -1, 1)), 0.05 + 0.015 * k)
                wave = 0.035 * np.sin(Y * 2 * np.pi / 0.13 + side) * U ** 1.2
                base[..., 2] -= 0.025 + wave + 0.03 * U
                parts.append(grid_part_local(base))
        g = merge_parts(parts)
        for p in parts:
            pass
        g.compute_normals(weld=False)
        g.W = self.torso_weights(g.pos)
        self.mb.add("cloth_white_ds", g)

    # ------------------------------------------------------------------
    def ascot(self):
        nb, sk = self.nb, self.sk
        parts = []
        c0 = np.array([0.0, 15.72, -0.52])
        # loops
        for side in (1, -1):
            pts = np.array([[0.0, 0.0], [0.16, 0.15], [0.42, 0.20], [0.56, 0.06], [0.50, -0.12], [0.28, -0.15],
                            [0.08, -0.05], [0.0, 0.0]])
            path = catmull_rom(pts[:-1], 30, closed=True)
            P = np.c_[path[:, 0] * side, path[:, 1], np.zeros(len(path))] + c0
            P[:, 2] += 0.16 * (np.abs(path[:, 0]) / 0.56) ** 1.5
            parts.append(ribbon_tube(P, 0.16, 0.03, closed=True, up=(0, 0, -1)))
        # tails
        for side in (1, -1):
            ctrl = np.array([[0.03 * side, 15.68, -0.56], [0.14 * side, 15.35, -0.64], [0.19 * side, 14.95, -0.74],
                             [0.22 * side, 14.50, -0.80]])
            path = catmull_rom(ctrl, 18)
            w = np.linspace(0.14, 0.26, len(path))
            parts.append(ribbon_tube(path, w, 0.02, up=(0, 0, -1), vcut_end=0.08))
        # knot
        parts.append(sphere_part(c0 + np.array([0, -0.01, -0.04]), 0.10, 10, 8, squash=(1.0, 0.9, 0.7)))
        g = merge_parts(parts)
        g.W = w_chain(g.pos[:, 1], [(15.3, sk["上半身2"]), (15.6, sk["上半身2"]), (15.9, sk["首"])], nb)
        g.W = w_mix(g.W, w_single(g.n, nb, sk["上半身2"]), 0.5)
        self.mb.add("cloth_black_satin", g)

    # ------------------------------------------------------------------
    def sleeves(self):
        sk, nb = self.sk, self.nb
        L = np.array([-0.35, 0.0, 0.5, 1.2, 2.0, 2.65, 3.0, 3.35, 4.2, 4.9, 5.35, 5.55])
        R = np.array([0.40, 0.47, 0.57, 0.59, 0.50, 0.37, 0.31, 0.37, 0.50, 0.44, 0.29, 0.25])
        rad = pchip(L, R)
        puff = pchip(L, np.array([0, 0.3, 1, 1, 0.8, 0.2, 0, 0.3, 1, 0.8, 0.1, 0]))
        d = np.r_[np.linspace(-0.35, 3.0, 24), np.linspace(3.06, 5.55, 18)]
        K = 28
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        for side, lr in ((1, "左"), (-1, "右")):
            S = np.array([side, 1, 1])
            ax = ARM_DIR * S
            up = np.array([math.sin(math.radians(37)) * side, math.cos(math.radians(37)), 0])
            bz = np.cross(ax, up)
            rings = []
            for di in d:
                c = SHOULDER * S + ax * di
                r = rad(di) * (1 + 0.05 * puff(di) * np.sin(9 * th + di * 2.0))
                # squash slightly front-back
                ring = c + (r * np.cos(th))[:, None] * up + (r * 0.95 * np.sin(th))[:, None] * bz
                rings.append(ring)
            rings = np.array(rings)
            part = tube_part(rings, v_coords=d / 5.55)
            axis = SHOULDER * S + ax * d[:, None]
            orient_outward(part, axis_inside(axis))
            part.compute_normals()
            part.W = self.arm_weights(part.pos, side)
            self.mb.add("cloth_white", part)
            # cuff frill (two layers)
            for k, (d0, d1, r0, r1, wv) in enumerate(((5.38, 5.95, 0.27, 0.52, 0.06), (5.30, 5.75, 0.30, 0.50, 0.05))):
                rr = []
                for t in np.linspace(0, 1, 5):
                    di = d0 + (d1 - d0) * t
                    c = SHOULDER * S + ax * di
                    r = r0 + (r1 - r0) * t ** 1.3 + wv * t * np.sin(15 * th + k)
                    rr.append(c + (r * np.cos(th))[:, None] * up + (r * np.sin(th))[:, None] * bz)
                fr = tube_part(np.array(rr))
                orient_outward(fr, axis_inside(SHOULDER * S + ax * np.linspace(d0, d1 + 2, 9)[:, None]))
                fr.compute_normals()
                fr.W = self.arm_weights(fr.pos, side)
                self.mb.add("cloth_white_ds", fr)
            # elbow bow (black band + loops)
            self.elbow_bow(side, rad, ax, up, bz)

    def elbow_bow(self, side, rad, ax, up, bz):
        S = np.array([side, 1, 1])
        parts = []
        K = 24
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        di = 3.0
        rings = []
        for dd, extra in ((-0.07, 0.0), (-0.05, 0.03), (0.05, 0.03), (0.07, 0.0)):
            c = SHOULDER * S + ax * (di + dd)
            r = rad(di + dd) + extra + 0.01
            rings.append(c + (r * np.cos(th))[:, None] * up + (r * np.sin(th))[:, None] * bz)
        band = tube_part(np.array(rings))
        orient_outward(band, axis_inside(SHOULDER * S + ax * np.linspace(di - 0.2, di + 0.2, 5)[:, None]))
        band.compute_normals()
        parts.append(band)
        # bow on the outer/back side of the elbow
        c = SHOULDER * S + ax * di + (-bz * 0.0 + up * 0.0)
        out_dir = normalize(up * 0.3 + bz * 0.95 * side * 0 + bz)  # toward the back of the arm
        out_dir = normalize(np.array([0.0, 0.0, 1.0]) * 0.8 + up * 0.6)
        knot = c + out_dir * (rad(di) + 0.05)
        loop_w = np.cross(out_dir, ax)
        for sgn in (1, -1):
            pts = np.array([[0, 0], [0.10, 0.07], [0.22, 0.08], [0.26, 0.0], [0.22, -0.07], [0.10, -0.06]])
            path = catmull_rom(pts, 24, closed=True)
            P = knot[None] + (path[:, 0] * sgn)[:, None] * ax + path[:, 1:2] * loop_w + \
                (0.05 * np.abs(path[:, 0]) / 0.26)[:, None] * out_dir
            parts.append(ribbon_tube(P, 0.075, 0.02, closed=True, up=out_dir))
        for sgn, ang in ((1, 0.45), (-1, -0.45)):
            dirv = normalize(-loop_w * 1.0 + ax * ang)
            path = knot[None] + np.linspace(0, 1, 8)[:, None] * dirv * 0.30 + \
                (np.linspace(0, 1, 8) ** 2 * 0.05)[:, None] * out_dir
            parts.append(ribbon_tube(path, np.linspace(0.06, 0.09, 8), 0.016, up=out_dir, vcut_end=0.04))
        parts.append(sphere_part(knot, 0.045, 8, 6))
        g = merge_parts(parts)
        g.W = self.arm_weights(g.pos, side)
        self.mb.add("cloth_black_satin", g)

    def arm_weights(self, P, side):
        sk, nb = self.sk, self.nb
        lr = "左" if side > 0 else "右"
        S = np.array([side, 1, 1])
        d = (P - SHOULDER * S) @ (ARM_DIR * S)
        keys = [(-0.40, sk[lr + "肩"]), (0.15, sk[lr + "腕"]), (0.75, sk[lr + "腕捩1"]),
                (1.5, sk[lr + "腕捩2"]), (2.25, sk[lr + "腕捩3"]), (2.82, sk[lr + "腕捩"]),
                (3.18, sk[lr + "ひじ"]), (3.0 + 0.65, sk[lr + "手捩1"]), (3.0 + 1.3, sk[lr + "手捩2"]),
                (3.0 + 1.95, sk[lr + "手捩3"]), (3.0 + 2.35, sk[lr + "手捩"]), (5.75, sk[lr + "手首"])]
        W = w_chain(d, keys, nb)
        # top of the sleeve head blends into the torso
        w_t = smoothstep(0.1, -0.4, d)
        Wt = self.torso_weights(P)
        return w_mix(W, Wt, w_t * 0.5)

    # ------------------------------------------------------------------
    def hands(self):
        for side in (1, -1):
            self._hand(side)

    def _hand(self, side):
        sk, nb = self.sk, self.nb
        lr = "左" if side > 0 else "右"
        chains = []
        for fname, knuckle, lengths in FINGERS:
            fc = finger_chain_local(knuckle, lengths)
            chains.append((fname, fc, (0.068 if fname != "小指" else 0.058)))
        tc = thumb_chain_local()

        def sdf(p):
            d = sd_round_box(p, np.array([0.42, -0.01, -0.005]), np.array([0.44, 0.105, 0.30]), 0.09)
            # taper palm toward the wrist
            d = smin(d, sd_capsule(p, np.array([-0.35, 0, 0]), np.array([0.15, 0, 0]), 0.18), 0.12)
            for fname, fc, r in chains:
                base = fc[0] - np.array([0.25, 0, 0])
                d = smin(d, sd_capsule(p, base, fc[0], r * 1.05), 0.05)
                for i in range(3):
                    rr = (r * (1 - 0.08 * i), r * (1 - 0.08 * (i + 1)))
                    d = smin(d, sd_capsule(p, fc[i], fc[i + 1], rr), 0.025)
            d = smin(d, sd_capsule(p, tc[0] + np.array([0.05, 0.02, 0.1]), tc[1], (0.13, 0.085)), 0.08)
            d = smin(d, sd_capsule(p, tc[1], tc[2], (0.08, 0.074)), 0.03)
            d = smin(d, sd_capsule(p, tc[2], tc[3], (0.074, 0.065)), 0.025)
            return d

        part = surface_nets(sdf, (-0.4, -0.3, -0.75), (2.0, 0.3, 0.5), 0.03)
        loc = part.pos.copy()
        # weights in local space
        W = np.zeros((part.n, nb))
        wrist_keys = [(-0.25, sk[lr + "手捩"]), (0.05, sk[lr + "手首"])]
        Wbase = w_chain(loc[:, 0], wrist_keys, nb)
        best = np.full(part.n, np.inf)
        assign = np.full(part.n, -1)
        params = np.zeros(part.n)
        fchains = [(lr + fname, fc) for fname, fc, _ in chains] + [(lr + "親指", tc)]
        for ci, (pref, fc) in enumerate(fchains):
            poly = np.vstack([fc[0] - (fc[1] - fc[0]) * 0.8, fc])
            dmin, sp = _polyline_dist(loc, poly)
            m = dmin < best
            best[m] = dmin[m]
            assign[m] = ci
            seglen = np.linalg.norm(fc[1] - fc[0]) * 0.8
            params[m] = (sp - seglen)[m]
        for ci, (pref, fc) in enumerate(fchains):
            m = assign == ci
            if not m.any():
                continue
            Ls = np.r_[0, np.cumsum(np.linalg.norm(np.diff(fc, axis=0), axis=1))]
            if pref.endswith("親指"):
                names = [pref + ZEN[0], pref + ZEN[1], pref + ZEN[2]]
                keys = [(-0.18, -1), (0.05, sk[names[0]]), (Ls[1] - 0.05, sk[names[0]]), (Ls[1] + 0.04, sk[names[1]]),
                        (Ls[2] - 0.03, sk[names[1]]), (Ls[2] + 0.03, sk[names[2]])]
            else:
                names = [pref + ZEN[1], pref + ZEN[2], pref + ZEN[3]]
                keys = [(-0.10, -1), (0.04, sk[names[0]]), (Ls[1] - 0.04, sk[names[0]]), (Ls[1] + 0.03, sk[names[1]]),
                        (Ls[2] - 0.03, sk[names[1]]), (Ls[2] + 0.03, sk[names[2]])]
            s = params[m]
            Wf = np.zeros((m.sum(), nb))
            ks = np.array([k[0] for k in keys])
            for r_i, sv in enumerate(s):
                i = np.clip(np.searchsorted(ks, sv, side="right") - 1, 0, len(ks) - 1)
                if i >= len(ks) - 1:
                    a, b, t = i, i, 0.0
                elif sv <= ks[0]:
                    a, b, t = 0, 0, 0.0
                else:
                    a, b = i, i + 1
                    t = (sv - ks[a]) / (ks[b] - ks[a])
                for idx, wt in ((keys[a][1], 1 - t), (keys[b][1], t)):
                    if idx < 0:
                        Wf[r_i] += Wbase[np.where(m)[0][r_i]] * wt
                    else:
                        Wf[r_i, idx] += wt
            W[m] = Wf
        W[assign < 0] = Wbase[assign < 0]
        # vertices far from any finger (palm back) -> wrist chain
        far = best > 0.16
        W[far] = Wbase[far]
        world = hand_local_to_world(loc, side)
        part.pos = world
        Rm = np.stack([HAND_X, HAND_Y, HAND_Z], 1)
        nrm = part.normals @ Rm.T
        if side < 0:
            nrm[:, 0] *= -1
            part.faces = part.faces[:, ::-1].copy()
        part.normals = normalize(nrm)
        part.uv = np.tile([0.5, 0.5], (part.n, 1))
        part.W = W
        self.mb.add("skin", part)

    # ------------------------------------------------------------------
    def legs(self):
        sk, nb = self.sk, self.nb
        #               y      rx     rz    zoff
        tab = np.array([[0.80, 0.23, 0.25, 0.00],
                        [1.20, 0.235, 0.255, 0.0],
                        [2.00, 0.27, 0.30, 0.02],
                        [3.20, 0.33, 0.37, 0.05],
                        [4.40, 0.39, 0.45, 0.08],
                        [5.40, 0.36, 0.40, 0.04],
                        [6.10, 0.35, 0.38, 0.0],
                        [7.20, 0.40, 0.43, 0.0],
                        [9.00, 0.48, 0.52, 0.02],
                        [10.6, 0.55, 0.59, 0.04],
                        [11.6, 0.58, 0.62, 0.05],
                        [12.1, 0.55, 0.60, 0.05]])
        frx = pchip(tab[:, 0], tab[:, 1])
        frz = pchip(tab[:, 0], tab[:, 2])
        fzo = pchip(tab[:, 0], tab[:, 3])
        K = 24
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        for side, lr in ((1, "左"), (-1, "右")):
            S = np.array([side, 1, 1])
            axis_ctrl = np.array([LEG + np.array([0, 0.5, 0.02]), LEG, KNEE, ANKLE, ANKLE - np.array([0, 0.3, 0])]) * S
            axis = catmull_rom(axis_ctrl, 60)

            def center(y):
                i = np.argsort(axis[:, 1])
                return np.stack([np.interp(y, axis[i, 1], axis[i, k]) for k in range(3)], -1)

            def rings_for(ys, extra=0.0):
                rr = []
                for y in ys:
                    c = center(y)
                    rx, rz = frx(y) + extra, frz(y) + extra
                    zo = fzo(y)
                    # inner thigh slightly flattened
                    x = rx * np.sin(th)
                    flat = 1 - 0.12 * smoothstep(9.0, 11.5, y) * (np.sin(th) * side < 0)
                    rr.append(np.stack([c[0] + x * flat,
                                        np.full(K, y), c[2] - rz * np.cos(th) + zo * (np.cos(th) < 0)], -1))
                return np.array(rr)

            ys = np.r_[np.linspace(3.85, 12.1, 34)]
            skin = tube_part(rings_for(ys), v_coords=ys / 12)
            orient_outward(skin, axis_inside(np.c_[center(ys)]))
            skin.compute_normals()
            skin.W = self.leg_weights(skin.pos, side)
            self.mb.add("skin", skin)
            ys = np.r_[np.linspace(0.8, 3.85, 14), 3.92, 3.98, 4.02]
            ex = np.where(ys > 3.88, 0.035, 0.012)
            rr = np.array([rings_for([y], e)[0] for y, e in zip(ys, ex)])
            sock = tube_part(rr, v_coords=(4.02 - ys) / 3.3)
            orient_outward(sock, axis_inside(np.c_[center(ys)]))
            sock.compute_normals()
            # u: 0.5 at front
            sock.uv[:, 0] = (sock.uv[:, 0] + 0.5) % 1.0001
            sock.W = self.leg_weights(sock.pos, side)
            self.mb.add("sock", sock)

    def leg_weights(self, P, side):
        sk, nb = self.sk, self.nb
        lr = "左" if side > 0 else "右"
        y = P[:, 1]
        return w_chain(y, [(0.95, sk[lr + "足首D"]), (1.35, sk[lr + "ひざD"]), (5.75, sk[lr + "ひざD"]),
                           (6.45, sk[lr + "足D"]), (11.2, sk[lr + "足D"]), (12.15, sk["下半身"])], nb)

    # ------------------------------------------------------------------
    def shoes(self):
        sk, nb = self.sk, self.nb
        for side, lr in ((1, "左"), (-1, "右")):
            S = np.array([side, 1, 1])
            heel_c = np.array([0.56, 0.0, 0.42])
            toe_c = np.array([0.62, 0.0, -1.10])

            def sdf(p):
                q = p * np.array([side, 1, 1])
                foot = sd_capsule(q, np.array([0.56, 0.42, 0.30]), np.array([0.60, 0.30, -0.55]), (0.27, 0.25))
                toe = sd_ellipsoid(q, np.array([0.61, 0.22, -0.78]), np.array([0.27, 0.20, 0.36]))
                d = smin(foot, toe, 0.15)
                ankle = sd_capsule(q, np.array([0.56, 0.45, 0.22]), np.array([0.56, 0.78, 0.18]), 0.29)
                d = smin(d, ankle, 0.12)
                # open top: cut at a sloped plane (lower at the instep)
                cut = q[:, 1] - (0.80 + 0.25 * np.clip(q[:, 2], -1, 0.4))
                d = smax(d, cut, 0.03)
                heel = sd_round_box(q, np.array([0.56, 0.18, 0.40]), np.array([0.16, 0.18, 0.15]), 0.04)
                d = smin(d, heel, 0.04)
                sole = sd_round_box(q, np.array([0.60, 0.06, -0.55]), np.array([0.26, 0.06, 0.48]), 0.05)
                d = smin(d, sole, 0.05)
                d = smax(d, -q[:, 1], 0.01)
                return d

            part = surface_nets(sdf, (-1.2 if side < 0 else 0.1, -0.1, -1.5), (-0.1 if side < 0 else 1.2, 1.2, 0.8), 0.05)
            part.uv = np.tile([0.5, 0.5], (part.n, 1))
            z = part.pos[:, 2]
            part.W = w_chain(-z, [(-0.1, sk[lr + "足首D"]), (0.45, sk[lr + "足首D"]), (0.75, sk[lr + "足先EX"])], nb)
            self.mb.add("shoe", part)
            # instep strap + bow with flower
            strap_c = np.array([0.58 * side, 0.66, -0.12])
            parts = []
            K = 20
            th = np.linspace(0, 2 * np.pi, K, endpoint=False)
            rings = []
            for dy, rr in ((-0.03, 0.30), (0.0, 0.315), (0.03, 0.30)):
                rings.append(np.stack([strap_c[0] + rr * np.sin(th), np.full(K, strap_c[1] + dy) - 0.12 * np.cos(th) ** 2 * 0,
                                       strap_c[2] - rr * 1.05 * np.cos(th)], -1))
            st = tube_part(np.array(rings))
            orient_outward(st, lambda c: np.tile(strap_c, (len(c), 1)))
            st.compute_normals()
            parts.append(st)
            knot = np.array([0.58 * side, 0.62, -0.47])
            for sgn in (1, -1):
                pts = np.array([[0, 0], [0.09, 0.06], [0.20, 0.07], [0.24, 0.0], [0.20, -0.06], [0.09, -0.05]])
                path = catmull_rom(pts, 20, closed=True)
                P = knot[None] + np.c_[path[:, 0] * sgn, path[:, 1] * 0.8 + 0.02, -path[:, 1] * 0.5]
                parts.append(ribbon_tube(P, 0.07, 0.018, closed=True, up=(0, 0.6, -0.8)))
            g = merge_parts(parts)
            g.W = w_single(g.n, nb, sk[lr + "足首D"])
            self.mb.add("cloth_black_satin", g)
            fl = flower_part(knot + np.array([0, 0.03, -0.05]), normalize(np.array([0, 0.5, -1.0])), 0.10, 10)
            fl_c = sphere_part(knot + np.array([0, 0.035, -0.075]), 0.035, 8, 6)
            fl.W = w_single(fl.n, nb, sk[lr + "足首D"])
            fl_c.W = w_single(fl_c.n, nb, sk[lr + "足首D"])
            fl_c.uv[:] = (0.5, 0.5)
            self.mb.add("flower", fl)
            self.mb.add("flower_center", fl_c)


# ---------------------------------------------------------------------------
# shared primitive builders
# ---------------------------------------------------------------------------

def _polyline_dist(P, poly):
    best = np.full(len(P), np.inf)
    sp = np.zeros(len(P))
    acc = 0.0
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        L = np.linalg.norm(ab)
        t = np.clip(((P - a) @ ab) / (L * L), 0, 1)
        q = a + t[:, None] * ab
        d = np.linalg.norm(P - q, axis=1)
        m = d < best
        best[m] = d[m]
        sp[m] = acc + t[m] * L
        acc += L
    return best, sp


def grid_part_local(P):
    R, C = P.shape[:2]
    uu, vv = np.meshgrid(np.linspace(0, 1, C), np.linspace(0, 1, R))
    return Part(P.reshape(-1, 3), grid_faces(R, C), np.stack([uu, vv], -1).reshape(-1, 2))


def sphere_part(c, r, nu=12, nv=8, squash=(1, 1, 1)):
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    ph = np.linspace(0, np.pi, nv)
    rings = []
    for p in ph[1:-1]:
        rings.append(np.stack([r * np.sin(p) * np.cos(th) * squash[0], np.full(nu, r * np.cos(p)) * squash[1],
                               r * np.sin(p) * np.sin(th) * squash[2]], -1) + c)
    part = tube_part(np.array(rings), cap_start=True, cap_end=True)
    # move caps to the poles
    part.pos[-2] = c + np.array([0, r * squash[1], 0])
    part.pos[-1] = c - np.array([0, r * squash[1], 0])
    orient_outward(part, lambda cc: np.tile(c, (len(cc), 1)))
    part.compute_normals()
    return part


def ribbon_tube(path, width, thick, closed=False, up=(0, 0, -1), vcut_end=0.0, K=8):
    """Flat ribbon (lens cross-section) along a path.  width may be an array."""
    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    width = np.broadcast_to(np.asarray(width, dtype=np.float64), (n,)).copy()
    T, N, Bn = frames_along(path if not closed else np.vstack([path[-1:], path, path[:1]]), up)
    if closed:
        T, N, Bn = T[1:-1], N[1:-1], Bn[1:-1]
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    rings = []
    for i in range(n):
        w = width[i] * 0.5
        u = np.cos(th) * w
        v = np.sin(th) * thick * 0.5
        ring = path[i] + u[:, None] * Bn[i] + v[:, None] * N[i]
        if vcut_end and i == n - 1:
            ring = ring + (np.abs(np.cos(th)) - 0.5)[:, None] * T[i] * vcut_end * -1
        rings.append(ring)
    rings = np.array(rings)
    if closed:
        R, Kk = rings.shape[:2]
        P = np.concatenate([rings, rings[:, :1]], 1)
        P = np.concatenate([P, P[:1]], 0)
        part = Part(P.reshape(-1, 3), grid_faces(R + 1, Kk + 1))
        uu, vv = np.meshgrid(np.linspace(0, 1, Kk + 1), np.linspace(0, 1, R + 1))
        part.uv = np.stack([uu, vv], -1).reshape(-1, 2)
    else:
        part = tube_part(rings, cap_start=True, cap_end=True)
    orient_outward(part, axis_inside(path))
    part.compute_normals()
    return part


def flower_part(center, normal, radius, n_petals=12, layers=2, seed=0):
    """Many pointed petals radiating from the centre (gerbera-like)."""
    normal = normalize(np.asarray(normal, dtype=np.float64))
    a = normalize(np.cross(normal, [0, 1, 0]) if abs(normal[1]) < 0.9 else np.cross(normal, [1, 0, 0]))
    b = np.cross(normal, a)
    parts = []
    for L in range(layers):
        k = n_petals
        off = (0.5 * L) * 2 * np.pi / k
        rad = radius * (1 - 0.18 * L)
        for i in range(k):
            ang = off + i * 2 * np.pi / k
            d = np.cos(ang) * a + np.sin(ang) * b
            t = np.linspace(0, 1, 6)
            path = center + np.outer(t * rad, d) + np.outer(0.25 * rad * (t ** 1.5) * (1 - 0.5 * L) - 0.06 * rad * L, normal)
            w = rad * 0.32 * np.sin(np.pi * np.clip(t * 0.92 + 0.08, 0, 1)) ** 0.8
            w[-1] = 0.002
            p = ribbon_tube(path, w, rad * 0.06, up=normal, K=6)
            parts.append(p)
    g = merge_parts(parts)
    return g

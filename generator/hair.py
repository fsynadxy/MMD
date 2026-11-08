"""Long wavy coral-orange hair: cap, bangs, side locks, two layers of back
strands, physics chains."""
from __future__ import annotations

import math

import numpy as np

from .body import TORSO_ZC, torso_point
from .builder import w_chain, w_single
from .geom import (Part, catmull_rom, frames_along, grid_faces, normalize, orient_outward, project_to_surface,
                   resample_polyline, sdf_gradient, smoothstep, surface_nets, tube_part, smax, sd_ellipsoid)
from .head import head_sdf, skull_sdf
from .physics import G_HAIR, Physics
from .skeleton import ARM_DIR, SHOULDER
from .skirt import skirt_point, Y_TOP as SK_TOP, Y_HEM as SK_HEM

HC = np.array([0.0, 18.35, 0.12])        # hair centre
BACK_CHAIN_PHI = (-66, -44, -22, 0, 22, 44, 66)
BACK_BONE_Y = (16.75, 15.35, 13.95, 12.55, 11.15)
BACK_TAIL_Y = 9.75
SIDE_BONE_Y = (17.35, 16.35, 15.35, 14.35)
SIDE_TAIL_Y = 13.3


# ---------------------------------------------------------------------------
# collision helpers
# ---------------------------------------------------------------------------

def _min_radius(y, theta):
    """Minimal horizontal distance from the body axis for hair at height y,
    in direction theta (torso convention: 0 front, pi back)."""
    y = np.asarray(y, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    yt = np.clip(y, 10.96, 16.0)
    tp = torso_point(yt, theta, 0.0)
    r = np.hypot(tp[..., 0], tp[..., 2] - TORSO_ZC) + 0.10
    r = np.where(y > 16.0, 0.0, r)
    # skirt
    ys = np.clip(y, SK_HEM, SK_TOP)
    sp = skirt_point(theta, ys, 0.30)
    rs = np.hypot(sp[..., 0], sp[..., 2] - TORSO_ZC)
    r = np.where((y < SK_TOP) & (y > SK_HEM - 1.0), np.maximum(r, rs), r)
    return r


def push_out(P, clear_head=0.10, body=True, iters=4, fixed=0):
    P = np.array(P, dtype=np.float64)
    for _ in range(iters):
        d = head_sdf(P)
        m = d < clear_head
        m[:fixed] = False
        if m.any():
            g = normalize(sdf_gradient(head_sdf, P[m]))
            P[m] += g * (clear_head - d[m])[:, None]
        if body:
            th = np.arctan2(P[:, 0], -(P[:, 2] - TORSO_ZC))
            r = np.hypot(P[:, 0], P[:, 2] - TORSO_ZC)
            rmin = _min_radius(P[:, 1], th)
            m = (r < rmin) & (P[:, 1] < 16.2)
            m[:fixed] = False
            if m.any():
                k = (rmin[m] / np.maximum(r[m], 1e-6))
                P[m, 0] *= k
                P[m, 2] = TORSO_ZC + (P[m, 2] - TORSO_ZC) * k
            # sleeves (upper arm capsules)
            for side in (1, -1):
                S = np.array([side, 1, 1])
                a = SHOULDER * S
                b = a + ARM_DIR * S * 2.6
                ab = b - a
                t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
                q = a + t[:, None] * ab
                dv = P - q
                dd = np.linalg.norm(dv, axis=1)
                rr = 0.66
                m = dd < rr
                m[:fixed] = False
                if m.any():
                    P[m] = q[m] + dv[m] / np.maximum(dd[m], 1e-6)[:, None] * rr
        # light smoothing to avoid kinks
        if len(P) > 4:
            Q = P.copy()
            Q[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
            Q[:max(fixed, 1)] = P[:max(fixed, 1)]
            P = Q
    return P


# ---------------------------------------------------------------------------
# strand geometry
# ---------------------------------------------------------------------------

def strand_part(path, width, thick, out_fn, K=10, curl=0.35, tip_taper=0.25, v0=0.0, v1=1.0):
    """Lens-shaped hair strand along `path`.

    width / thick: arrays (n,) ; out_fn(points)->outward directions."""
    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    T = np.zeros_like(path)
    T[1:-1] = path[2:] - path[:-2]
    T[0] = path[1] - path[0]
    T[-1] = path[-1] - path[-2]
    T = normalize(T)
    O = out_fn(path)
    N = normalize(O - T * np.sum(O * T, axis=1, keepdims=True))
    B = np.cross(T, N)
    phi = np.linspace(0, 2 * np.pi, K, endpoint=False)
    cu, sv = np.cos(phi), np.sin(phi)
    rings = []
    for i in range(n):
        w = width[i] * 0.5
        u = cu * w
        v = sv * thick[i] * 0.5 + (1 - cu ** 2) * thick[i] * 0.15 - curl * cu ** 2 * w * 0.35
        rings.append(path[i] + u[:, None] * B[i] + v[:, None] * N[i])
    rings = np.array(rings)
    part = tube_part(rings, v_coords=np.linspace(v0, v1, n))
    # orientation: cross-section ring direction (B -> N) with tangent T
    orient_outward(part, _nearest_on(path))
    part.compute_normals()
    return part


def _nearest_on(path):
    def fn(c):
        d = ((c[:, None, :] - path[None, :, :]) ** 2).sum(-1)
        return path[np.argmin(d, axis=1)]
    return fn


def out_from_axis(P):
    """Outward: from the head centre near the head, horizontally from the body axis below."""
    P = np.asarray(P)
    o_head = P - HC
    o_body = P - np.c_[np.zeros(len(P)), P[:, 1], np.full(len(P), TORSO_ZC)]
    t = smoothstep(17.6, 16.6, P[:, 1])[:, None]
    return normalize(o_head * (1 - t) + o_body * t)


def taper(n, w0, w1, tip=0.25, start=0.0, power=1.0):
    t = np.linspace(0, 1, n)
    w = w0 + (w1 - w0) * np.clip(t / max(1 - tip, 1e-6), 0, 1)
    tp = np.clip((t - (1 - tip)) / tip, 0, 1)
    w = w * (1 - tp ** 1.6) ** power
    if start > 0:
        w *= 0.55 + 0.45 * smoothstep(0, start, t)
    return np.maximum(w, 0.004)


class HairBuilder:
    def __init__(self, mb):
        self.mb = mb
        self.sk = mb.sk
        self.rng = np.random.default_rng(7)

    # ------------------------------------------------------------------
    def back_guide(self, phi_deg, tip_y, layer=0, n=48, jitter=0.0):
        phi = math.radians(phi_deg)
        dh = np.array([math.sin(phi), 0, math.cos(phi)])
        up = np.array([0, 1.0, 0])
        R0 = 1.30 + 0.08 * layer
        ctrl = []
        for e, rr in ((64, R0 - 0.08), (38, R0), (8, R0 + 0.04), (-28, R0 - 0.02)):
            er = math.radians(e)
            ctrl.append(HC + rr * (math.cos(er) * dh * np.array([1.0, 0, 1.08]) + math.sin(er) * up))
        theta = math.pi - phi
        ys = [16.5, 15.2, 13.8, 12.4, 11.0, tip_y]
        for i, y in enumerate(ys):
            th = theta * (1 - 0.10 * smoothstep(16, 11, y))
            rmin = float(_min_radius(np.array(y), np.array(th)))
            spread = 0.18 + 0.30 * float(smoothstep(15.5, 9.5, y)) + 0.12 * layer
            r = rmin + spread
            ctrl.append(np.array([math.sin(math.pi - th) * r, y, TORSO_ZC + math.cos(math.pi - th) * r]))
        ctrl = np.array(ctrl)
        path = catmull_rom(ctrl, n)
        path = resample_polyline(path, n)
        # waves (lower half)
        y = path[:, 1]
        a = 0.17 * smoothstep(14.5, 10.5, y)
        ph0 = phi * 3.1 + layer * 1.3 + jitter
        tang = np.array([math.cos(phi), 0, -math.sin(phi)])
        path = path + np.outer(a * np.sin(y / 2.1 * 2 * np.pi + ph0), tang)
        path = path + np.outer(0.5 * a * np.cos(y / 2.1 * 2 * np.pi + ph0), dh)
        return push_out(path, clear_head=0.16 + 0.07 * layer, fixed=0)

    def add_bones(self):
        sk = self.sk
        self.back_chains = []
        for ci, phi in enumerate(BACK_CHAIN_PHI):
            g = self.back_guide(phi, BACK_TAIL_Y - 0.2, layer=0, n=80)
            pts = []
            for y in BACK_BONE_Y + (BACK_TAIL_Y,):
                i = np.argmin(np.abs(g[:, 1] - y))
                pts.append(g[i] + normalize(g[i] - np.array([0, g[i][1], TORSO_ZC])) * 0.05)
            names = []
            parent = "頭"
            for k in range(len(BACK_BONE_Y)):
                nm = "後髪%d_%d" % (ci + 1, k + 1)
                sk.add(nm, "hair_back%d_%d" % (ci + 1, k + 1), pts[k], parent)
                names.append(nm)
                parent = nm
            for k in range(len(names) - 1):
                sk.connect(names[k], names[k + 1])
            sk.bones[sk[names[-1]]].tail = tuple(pts[-1] - pts[-2])
            self.back_chains.append((phi, names, np.array(pts)))
        self.side_chains = []
        for side, lr in ((1, "左"), (-1, "右")):
            g = self.side_guide(side, 0)
            pts = []
            for y in SIDE_BONE_Y + (SIDE_TAIL_Y,):
                i = np.argmin(np.abs(g[:, 1] - y))
                pts.append(g[i])
            names = []
            parent = "頭"
            for k in range(len(SIDE_BONE_Y)):
                nm = "%s横髪%d" % (lr, k + 1)
                sk.add(nm, "hair_side%s%d" % ("_L" if side > 0 else "_R", k + 1), pts[k], parent)
                names.append(nm)
                parent = nm
            for k in range(len(names) - 1):
                sk.connect(names[k], names[k + 1])
            sk.bones[sk[names[-1]]].tail = tuple(pts[-1] - pts[-2])
            self.side_chains.append((side, names, np.array(pts)))

    # ------------------------------------------------------------------
    def _chain_keys(self, names, ys, top_bone="頭"):
        sk = self.sk
        keys = [(-(ys[0] + 0.30), sk[top_bone])]
        for k, nm in enumerate(names):
            y0 = ys[k]
            y1 = ys[k + 1] if k + 1 < len(ys) else -1e3
            keys.append((-(y0 - 0.15), sk[nm]))
            if k + 1 < len(ys):
                keys.append((-(y1 + 0.25), sk[nm]))
        return keys

    def back_weights(self, P, phi_deg):
        nb = self.mb.nb
        phis = np.array(BACK_CHAIN_PHI, dtype=float)
        phi = np.clip(phi_deg, phis[0], phis[-1])
        i = int(np.clip(np.searchsorted(phis, phi) - 1, 0, len(phis) - 2))
        t = (phi - phis[i]) / (phis[i + 1] - phis[i])
        W = np.zeros((len(P), nb))
        for ci, f in ((i, 1 - t), (i + 1, t)):
            if f <= 1e-4:
                continue
            _, names, pts = self.back_chains[ci]
            keys = self._chain_keys(names, [p[1] for p in pts[:-1]])
            W += w_chain(-P[:, 1], keys, nb) * f
        return W

    def side_weights(self, P, side):
        nb = self.mb.nb
        _, names, pts = self.side_chains[0 if side > 0 else 1]
        keys = self._chain_keys(names, [p[1] for p in pts[:-1]])
        return w_chain(-P[:, 1], keys, nb)

    # ------------------------------------------------------------------
    def cap(self):
        nb = self.mb.nb

        def sdf(p):
            y = p[:, 1]
            th = 0.12 + 0.10 * smoothstep(18.0, 19.4, y) + 0.04 * smoothstep(18.3, 17.2, y)
            d = skull_sdf(p) - th
            face = sd_ellipsoid(p, np.array([0, 17.05, -1.2]), np.array([1.0, 1.32, 0.95]))
            d = smax(d, -face, 0.14)
            d = smax(d, 16.95 - y, 0.05)
            return d

        part = surface_nets(sdf, (-1.45, 16.6, -1.4), (1.45, 19.95, 1.65), 0.05)
        P = part.pos - HC
        az = np.arctan2(P[:, 0], P[:, 2])
        el = np.arcsin(np.clip(P[:, 1] / np.linalg.norm(P, axis=1), -1, 1))
        part.uv = np.c_[(az / (2 * np.pi) + 0.5) * 6.0, (1.0 - (el + 0.6) / 2.2)]
        part.W = w_single(part.n, nb, self.sk["頭"])
        self.mb.add("hair", part)

    # ------------------------------------------------------------------
    def bangs(self):
        nb = self.mb.nb
        head = self.sk["頭"]
        #   azimuth(deg from front, + = character's left), tip y, width, root elevation
        spec = [(-64, 17.45, 0.40, 50), (-50, 17.70, 0.36, 56), (-36, 17.92, 0.36, 60), (-23, 18.10, 0.34, 62),
                (-12, 17.86, 0.30, 63), (-2, 17.62, 0.20, 64), (8, 17.95, 0.30, 63), (19, 18.12, 0.34, 62),
                (31, 17.88, 0.36, 60), (45, 17.72, 0.36, 56), (58, 17.55, 0.40, 52), (70, 17.30, 0.38, 46)]
        parts = []
        for az, tip_y, w, el_r in spec:
            a = math.radians(az)
            dh = np.array([math.sin(a), 0, -math.cos(a)])
            up = np.array([0, 1.0, 0])
            ctrl = []
            for e, rr in ((el_r, 1.18), (el_r - 22, 1.24), (14, 1.20)):
                er = math.radians(e)
                ctrl.append(HC + rr * (math.cos(er) * dh * np.array([1.0, 0, 1.0]) + math.sin(er) * up))
            # down in front of the forehead / face
            mid_y = (18.45 + tip_y) / 2
            ctrl.append(np.array([math.sin(a * 1.08) * 1.02, mid_y, -0.95]))
            ctrl.append(np.array([math.sin(a * 1.15) * 1.04, tip_y, -0.90]))
            ctrl = np.array(ctrl)
            path = resample_polyline(catmull_rom(ctrl, 40), 26)
            path = push_out(path, clear_head=0.075, body=False)
            n = len(path)
            width = taper(n, w * 0.9, w, tip=0.45)
            thick = taper(n, 0.10, 0.07, tip=0.45)
            parts.append(strand_part(path, width, thick, lambda P: normalize(P - HC), K=8, curl=0.45))
        for p in parts:
            p.W = w_single(p.n, nb, head)
            self.mb.add("hair", p)

    # ------------------------------------------------------------------
    def side_guide(self, side, variant=0):
        S = np.array([side, 1, 1])
        if variant == 0:
            ctrl = np.array([[0.70, 19.15, -0.55], [0.98, 18.55, -0.55], [1.06, 17.70, -0.52], [1.06, 16.80, -0.58],
                             [1.06, 15.90, -0.66], [1.02, 15.00, -0.88], [0.98, 14.10, -0.98], [1.00, 13.25, -0.95],
                             [1.08, 12.85, -0.90]])
        else:
            ctrl = np.array([[0.85, 19.0, -0.30], [1.08, 18.40, -0.30], [1.14, 17.50, -0.25], [1.14, 16.60, -0.30],
                             [1.20, 15.80, -0.42], [1.22, 14.90, -0.58], [1.25, 14.10, -0.62], [1.32, 13.70, -0.55]])
        path = resample_polyline(catmull_rom(ctrl * S, 60), 44)
        y = path[:, 1]
        a = 0.08 * smoothstep(15.8, 13.5, y)
        path[:, 0] += side * a * np.sin(y / 1.3 * 2 * np.pi + variant)
        path[:, 2] += 0.5 * a * np.cos(y / 1.3 * 2 * np.pi + variant)
        return push_out(path, clear_head=0.11, iters=5)

    def side_locks(self):
        for side in (1, -1):
            for variant, (w0, w1, th) in enumerate(((0.36, 0.46, 0.13), (0.30, 0.40, 0.12))):
                path = self.side_guide(side, variant)
                n = len(path)
                width = taper(n, w0, w1, tip=0.30, start=0.15)
                thick = taper(n, th, th, tip=0.30)
                p = strand_part(path, width, thick, lambda P, s=side: normalize(
                    (P - np.c_[np.zeros(len(P)), P[:, 1], np.full(len(P), 0.1)]) * np.array([1, 0.2, 1])), K=10,
                    curl=0.4)
                p.W = self.side_weights(p.pos, side)
                hw = smoothstep(17.6, 18.3, p.pos[:, 1])[:, None]
                p.W = p.W * (1 - hw) + w_single(p.n, self.mb.nb, self.sk["頭"]) * hw
                self.mb.add("hair", p)

    # ------------------------------------------------------------------
    def back(self):
        rng = self.rng
        layers = [
            # layer, phis, tip base, width (top, bottom), thickness
            (0, np.linspace(-80, 80, 17), 9.55, (0.50, 0.70), 0.16),
            (1, np.linspace(-72, 72, 14), 10.2, (0.52, 0.74), 0.15),
            (2, np.linspace(-58, 58, 9), 11.4, (0.50, 0.66), 0.14),
        ]
        for layer, phis, tip_y, (w0, w1), th in layers:
            for phi in phis:
                tip = tip_y + rng.uniform(-0.45, 0.45) + 0.6 * (abs(phi) / 80) ** 2
                path = self.back_guide(float(phi), tip, layer=layer, n=44, jitter=rng.uniform(-0.6, 0.6))
                n = len(path)
                width = taper(n, w0, w1, tip=0.22, start=0.12)
                thick = taper(n, th, th * 1.1, tip=0.22)
                p = strand_part(path, width, thick, out_from_axis, K=8, curl=0.35)
                p.W = self.back_weights(p.pos, float(phi))
                self.mb.add("hair", p)

    # ------------------------------------------------------------------
    def physics(self, ph: Physics):
        for ci, (phi, names, pts) in enumerate(self.back_chains):
            prev = "頭"
            for k, nm in enumerate(names):
                r = 0.17 if k < 2 else 0.20
                ph.capsule(nm, "hair_back%d_%d" % (ci + 1, k + 1), nm, pts[k], pts[k + 1], r, group=G_HAIR, mode=2,
                           mass=0.7 - 0.08 * k, lin=0.9, ang=0.95, rest=0.0, fric=0.3)
                lim = (6 + 4 * k, 3, 6 + 4 * k)
                ph.joint("J_" + nm, "J_hair_back%d_%d" % (ci + 1, k + 1), prev, nm, pts[k], lim,
                         spring_rot=(60, 20, 60))
                prev = nm
        for side, names, pts in self.side_chains:
            prev = "頭"
            for k, nm in enumerate(names):
                ph.capsule(nm, "hair_side%d" % k, nm, pts[k], pts[k + 1], 0.13, group=G_HAIR, mode=2,
                           mass=0.4 - 0.05 * k, lin=0.9, ang=0.95, fric=0.3)
                lim = (8 + 4 * k, 4, 8 + 4 * k)
                ph.joint("J_" + nm, "J_hair_side%d" % k, prev, nm, pts[k], lim, spring_rot=(40, 20, 40))
                prev = nm

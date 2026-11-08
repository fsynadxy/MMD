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
SIDE_BONE_Y = (17.3, 16.3, 15.3, 14.3, 13.3)
SIDE_TAIL_Y = 12.2


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


def ramp_clear(low, top=0.07, y_top=19.25, y_low=18.55):
    """Clearance that is small at the crown (strand roots hide under the cap)
    and `low` further down."""
    return lambda y: top + (low - top) * smoothstep(y_top, y_low, y)


def push_out(P, clear_head=0.10, body=True, iters=4, fixed=0):
    P = np.array(P, dtype=np.float64)
    for _ in range(iters):
        d = head_sdf(P)
        ch = clear_head(P[:, 1]) if callable(clear_head) else np.full(len(P), clear_head)
        m = d < ch
        m[:fixed] = False
        if m.any():
            g = normalize(sdf_gradient(head_sdf, P[m]))
            P[m] += g * (ch[m] - d[m])[:, None]
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


def over_head(az_deg, elevations, offsets, front=True):
    """Points on the head surface (plus an outward offset) around the hair centre.
    Azimuth is measured from the front (front=True) or from the back, positive
    toward the character's left (+x)."""
    a = math.radians(az_deg)
    dh = np.array([math.sin(a), 0.0, -math.cos(a) if front else math.cos(a)])
    up = np.array([0.0, 1.0, 0.0])
    pts = []
    for e, off in zip(elevations, offsets):
        er = math.radians(e)
        d = normalize(math.cos(er) * dh + math.sin(er) * up)
        lo, hi = 0.2, 2.5
        for _ in range(40):                       # bisection on the head SDF
            mid = 0.5 * (lo + hi)
            if head_sdf((HC + d * mid)[None])[0] < 0:
                lo = mid
            else:
                hi = mid
        pts.append(HC + d * (0.5 * (lo + hi) + off))
    return pts


def add_curls(path, y_start, y_end, amp, pitch, phase, tang, radial, hook=0.0):
    """Helical S-curls below y_start (amplitude ramps in), optional hook at the tip."""
    P = path.copy()
    y = P[:, 1]
    ramp = smoothstep(y_start, y_end, y)
    ang = (y_start - y) / pitch * 2 * np.pi + phase
    if hook:
        t = np.linspace(0, 1, len(P))
        ang = ang + hook * np.pi * smoothstep(0.82, 1.0, t)
    off = (amp * ramp)[:, None] * (np.cos(ang)[:, None] * tang[None, :] + 0.8 * np.sin(ang)[:, None] * radial[None, :])
    return P + off


class HairBuilder:
    def __init__(self, mb):
        self.mb = mb
        self.sk = mb.sk
        self.rng = np.random.default_rng(11)

    # ------------------------------------------------------------------
    # guides
    # ------------------------------------------------------------------
    def back_guide(self, phi_deg, tip_y, layer=0, n=72, jitter=0.0, curls=True, e0=None):
        """Back hair.  phi: 0 = back centre.  layer 0 = longest / innermost."""
        phi = math.radians(phi_deg)
        R0 = 1.22 + 0.06 * layer
        e0 = (48, 60, 74)[layer] if e0 is None else e0
        ctrl = over_head(phi_deg, (e0, (e0 + 4) / 2, 4), (0.08, 0.15 + 0.05 * layer, 0.18 + 0.05 * layer),
                         front=False)
        # fall straight down from the widest point of the head (no tuck into the nape)
        ctrl.append(ctrl[-1] + np.array([0, -0.75, 0.02]))
        theta = math.pi - phi
        ys = [16.5, 15.2, 13.8, 12.4, 11.0, tip_y]
        ys = [y for y in ys if y > tip_y + 0.4] + [tip_y]
        for y in ys:
            th = theta * (1 - 0.08 * float(smoothstep(16, 11, y)))
            rmin = float(_min_radius(np.array(y), np.array(th)))
            spread = 0.20 + 0.42 * float(smoothstep(15.5, 9.5, y)) + 0.13 * layer
            r = rmin + spread
            # near the head the hair hangs straight down from the widest point of the head
            r = max(r, (R0 + 0.06) * float(smoothstep(14.2, 16.0, y)))
            ctrl.append(np.array([math.sin(th) * r, y, TORSO_ZC - math.cos(th) * r]))
        path = resample_polyline(catmull_rom(np.array(ctrl), n * 2), n)
        if curls:
            tang = np.array([math.cos(phi), 0, -math.sin(phi)])
            radial = np.array([math.sin(phi), 0, math.cos(phi)])
            amp = (0.36, 0.30, 0.15)[layer]
            ys0 = (14.6, 14.0, tip_y + 1.6)[layer]
            hand = 1.0 if (int(round(phi_deg)) // 7 + layer) % 2 == 0 else -1.0
            path = add_curls(path, ys0, ys0 - 2.0, amp, 1.7 - 0.12 * layer, jitter, tang * hand, radial,
                             hook=0.9)
        return push_out(path, clear_head=ramp_clear(0.12 + 0.05 * layer), fixed=0)

    def side_guide(self, side, variant=0, curls=True):
        S = np.array([side, 1, 1])
        if variant == 0:
            top = over_head(62 * side, (74, 52, 28), (0.07, 0.12, 0.15))
            ctrl = [p for p in top] + [np.array(q) * S for q in (
                [1.06, 17.65, -0.46], [1.08, 16.80, -0.52], [1.08, 15.85, -0.66], [1.06, 14.85, -0.90],
                [1.10, 13.85, -0.98], [1.18, 12.90, -0.92], [1.26, 12.15, -0.82])]
            y0 = 14.9
        else:
            top = over_head(80 * side, (70, 46, 20), (0.07, 0.13, 0.16))
            ctrl = [p for p in top] + [np.array(q) * S for q in (
                [1.18, 17.35, -0.22], [1.22, 16.50, -0.30], [1.26, 15.60, -0.42], [1.30, 14.60, -0.62],
                [1.36, 13.60, -0.66], [1.44, 12.75, -0.58])]
            y0 = 15.0
        path = resample_polyline(catmull_rom(np.array(ctrl), 160), 66)
        if curls:
            tang = np.array([1.0, 0, 0])
            radial = np.array([0, 0, -1.0])
            path = add_curls(path, y0, y0 - 1.6, 0.20, 1.35, 0.6 * variant + (0 if side > 0 else 1.7), tang, radial,
                             hook=1.0)
        return push_out(path, clear_head=ramp_clear(0.12), iters=5)

    # ------------------------------------------------------------------
    def add_bones(self):
        sk = self.sk
        self.back_chains = []
        for ci, phi in enumerate(BACK_CHAIN_PHI):
            g = self.back_guide(phi, BACK_TAIL_Y - 0.2, layer=0, n=90, curls=False)
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
            g = self.side_guide(side, 0, curls=False)
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
            keys.append((-(y0 - 0.15), sk[nm]))
            if k + 1 < len(ys):
                keys.append((-(ys[k + 1] + 0.25), sk[nm]))
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

    def _add(self, part, W):
        part.W = W
        self.mb.add("hair", part)

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def cap(self):
        nb = self.mb.nb

        def sdf(p):
            y = p[:, 1]
            th = 0.13 + 0.05 * smoothstep(18.0, 19.4, y) + 0.04 * smoothstep(18.3, 17.2, y)
            d = skull_sdf(p) - th
            face = sd_ellipsoid(p, np.array([0, 17.05, -1.2]), np.array([1.0, 1.32, 0.95]))
            d = smax(d, -face, 0.14)
            d = smax(d, 16.95 - y, 0.05)
            return d

        part = surface_nets(sdf, (-1.45, 16.6, -1.4), (1.45, 19.95, 1.65), 0.05)
        P = part.pos - (HC + np.array([0, 0, 0.12]))
        az = np.arctan2(P[:, 0], P[:, 2])
        el = np.arcsin(np.clip(P[:, 1] / np.linalg.norm(P, axis=1), -1, 1))
        # 7 texture repeats = 14 broad clumps radiating from the crown
        part.uv = np.c_[(az / (2 * np.pi) + 0.5) * 6.0, np.clip(0.75 - el / 2.4, 0, 1)]
        self._add(part, w_single(part.n, nb, self.sk["頭"]))

    def ahoge(self):
        nb = self.mb.nb
        for k, (dx, h, bend) in enumerate(((0.06, 0.30, 0.30), (-0.04, 0.22, -0.22))):
            ctrl = np.array([[dx * 0.5, 19.72, 0.10], [dx, 19.86 + h * 0.4, -0.05], [dx + bend * 0.3, 19.84 + h * 0.6, -0.30],
                             [dx + bend * 0.7, 19.76 + h * 0.35, -0.45]])
            path = resample_polyline(catmull_rom(ctrl, 40), 16)
            n = len(path)
            width = taper(n, 0.07, 0.05, tip=0.55)
            thick = taper(n, 0.035, 0.025, tip=0.55)
            p = strand_part(path, width, thick, lambda P: np.tile([0, 1.0, 0], (len(P), 1)), K=6, curl=0.1)
            self._add(p, w_single(p.n, nb, self.sk["頭"]))

    def bangs(self):
        nb = self.mb.nb
        #   azimuth from front, tip y, width, sideways sweep at the tip
        spec = [(-56, 17.35, 0.46, -0.10), (-43, 17.62, 0.50, -0.06), (-30, 17.98, 0.46, 0.02),
                (-17, 17.92, 0.44, 0.06), (-6, 17.52, 0.30, 0.04), (5, 17.96, 0.42, -0.04),
                (17, 18.02, 0.46, -0.05), (30, 17.80, 0.48, 0.03), (43, 17.66, 0.50, 0.06),
                (56, 17.40, 0.46, 0.10)]
        for az, tip_y, w, sweep in spec:
            top = over_head(az * 0.55, (80, 62), (0.05, 0.09))
            fore = over_head(az, (36, 16), (0.11, 0.11))
            a = math.radians(az)
            mid_y = 18.42 - 0.25 * abs(az) / 56
            ctrl = top + fore + [np.array([math.sin(a * 1.05) * 1.02 + sweep * 0.4, mid_y, -0.97]),
                                 np.array([math.sin(a * 1.10) * 1.05 + sweep, tip_y, -0.92])]
            path = resample_polyline(catmull_rom(np.array(ctrl), 80), 30)
            path = push_out(path, clear_head=ramp_clear(0.17, top=0.10, y_top=19.25, y_low=18.85), body=False)
            n = len(path)
            width = taper(n, w * 0.75, w, tip=0.42, power=1.2)
            thick = taper(n, 0.10, 0.08, tip=0.42)
            p = strand_part(path, width, thick, lambda P: normalize(P - HC), K=8, curl=0.45)
            self._add(p, w_single(p.n, nb, self.sk["頭"]))

    def side_locks(self):
        nb = self.mb.nb
        for side in (1, -1):
            for variant, (w0, w1, th) in enumerate(((0.34, 0.50, 0.15), (0.34, 0.52, 0.15))):
                path = self.side_guide(side, variant)
                n = len(path)
                width = taper(n, w0, w1, tip=0.16, start=0.10)
                thick = taper(n, th, th, tip=0.16)
                p = strand_part(path, width, thick, lambda P: normalize(
                    (P - np.c_[np.zeros(len(P)), P[:, 1], np.full(len(P), 0.1)]) * np.array([1, 0.2, 1])), K=8,
                    curl=0.4)
                W = self.side_weights(p.pos, side)
                hw = smoothstep(17.6, 18.3, p.pos[:, 1])[:, None]
                W = W * (1 - hw) + w_single(p.n, nb, self.sk["頭"]) * hw
                self._add(p, W)

    def back(self):
        rng = self.rng
        layers = [
            # layer, phis, tip y, width (top, bottom), thickness, start elevation
            (0, np.linspace(-86, 86, 12), 9.45, (0.46, 0.78), 0.17),
            (1, np.linspace(-78, 78, 10), 11.0, (0.48, 0.80), 0.16),
            (2, np.linspace(-66, 66, 8), 13.1, (0.52, 0.82), 0.15),
        ]
        for layer, phis, tip_y, (w0, w1), th in layers:
            for phi in phis:
                tip = tip_y + rng.uniform(-0.35, 0.35) + 0.7 * (abs(phi) / 86) ** 2
                path = self.back_guide(float(phi), tip, layer=layer, n=84, jitter=rng.uniform(0, 2 * np.pi))
                n = len(path)
                ys0 = (14.6, 14.0, tip + 1.6)[layer]
                ring = smoothstep(ys0, ys0 - 1.5, path[:, 1])
                width = taper(n, w0, w1, tip=0.14, start=0.12, power=0.8) * (1 - 0.32 * ring)
                thick = taper(n, th, th * 1.1, tip=0.14) * (1 + 0.35 * ring)
                p = strand_part(path, width, thick, out_from_axis, K=8, curl=0.35)
                self._add(p, self.back_weights(p.pos, float(phi)))

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

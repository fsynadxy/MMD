"""Facial features: eyes (sclera / iris / highlight / eyelid masks / lashes),
brows, mouth (masks / interior / teeth / line) and blush, plus their morphs."""
from __future__ import annotations

import numpy as np

from .builder import w_single
from .geom import Part, catmull_rom, grid_faces, normalize, ray_sdf, sdf_gradient
from .head import head_sdf
from .skeleton import IRIS_C

# offsets of the layers in front of the skin (along the surface normal)
OFF_SCLERA = 0.004
OFF_IRIS = 0.019
OFF_HL = 0.022
OFF_MASK = 0.030
OFF_LINE = 0.034
OFF_BROW = 0.012
OFF_BLUSH = 0.036
OFF_MOUTH_IN = 0.004
OFF_TEETH = 0.009

# palette texture swatches (u, v) for the 'lines' material
PAL = {
    "lash": (0.125, 0.125),
    "lash_soft": (0.375, 0.125),
    "brow": (0.625, 0.125),
    "mouth": (0.875, 0.125),
    "crease": (0.125, 0.375),
    "teeth": (0.375, 0.375),
}


def project(xy, offset=0.0):
    """Project 2D face-plane points (x, y) onto the head surface (front)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    o = np.c_[xy, np.full(len(xy), -3.0)]
    d = np.tile([0.0, 0.0, 1.0], (len(xy), 1))
    p = ray_sdf(head_sdf, o, d, t_max=6.0)
    if np.isnan(p).any():
        raise ValueError("projection missed the head")
    n = normalize(sdf_gradient(head_sdf, p))
    return p + n * offset, n


def face_uv(P):
    return np.c_[0.5 + P[:, 0] / 2.4, (19.7 - P[:, 1]) / 3.4]


def _strip_part(rows2d, offset, uv=None):
    """rows2d: (R, C, 2) -> projected grid Part; returns part, rest 2D array."""
    R, C = rows2d.shape[:2]
    flat = rows2d.reshape(-1, 2)
    P, N = project(flat, offset)
    part = Part(P, grid_faces(R, C), uv if uv is not None else np.zeros((len(P), 2)), N)
    return part


def _fix_orientation(part):
    """Make triangles face -z (towards the viewer)."""
    p, f = part.pos, part.faces
    n = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    if (n[:, 2] > 0).sum() > len(f) / 2:
        part.faces = f[:, ::-1].copy()
    return part


def _offsets_from_2d(rest2d, target2d, offset):
    """3D vertex offsets that move points along the surface from rest to target."""
    a, _ = project(rest2d.reshape(-1, 2), offset)
    b, _ = project(target2d.reshape(-1, 2), offset)
    return b - a


class FaceBuilder:
    def __init__(self, mb):
        self.mb = mb
        self.nb = mb.nb
        self.sk = mb.sk

    # ------------------------------------------------------------------
    def eyes(self):
        for side in (1, -1):
            self._eye(side)

    def _eye_curves(self, side, n=17):
        I = np.array([0.235, 17.70])
        O = np.array([0.705, 17.81])
        Up = np.array([I, [0.29, 17.855], [0.37, 17.915], [0.47, 17.94], [0.57, 17.925], [0.65, 17.885], O])
        Lo = np.array([I, [0.30, 17.615], [0.40, 17.575], [0.50, 17.565], [0.60, 17.595], [0.67, 17.68], O])
        U = catmull_rom(Up, n)
        L = catmull_rom(Lo, n)
        U[0], U[-1], L[0], L[-1] = I, O, I, O
        S = np.array([side, 1.0])
        return U * S, L * S, I * S, O * S

    def _eye(self, side):
        mb, nb = self.mb, self.nb
        lr = "左" if side > 0 else "右"
        head = self.sk["頭"]
        eyeb = self.sk[lr + "目"]
        n = 17
        U, L, I, O = self._eye_curves(side, n)
        s = np.linspace(0, 1, n)
        S = np.array([side, 1.0])
        # extended lid lines (shared beyond the corners)
        ext_in = np.array([I + np.array([-0.05 * side, -0.01]), I + np.array([-0.10 * side, -0.02])])[::-1]
        ext_out = np.array([O + np.array([0.06 * side, 0.02]), O + np.array([0.12 * side, 0.035])])
        Ue = np.vstack([ext_in, U, ext_out])
        Le = np.vstack([ext_in, L, ext_out])
        C = len(Ue)
        lid_cols = np.arange(2, 2 + n)

        # --- sclera (follows the eye opening with a margin, stays under the masks)
        Rs = 6
        bot = Le - np.array([0, 0.035])
        top = Ue + np.array([0, 0.045])
        g = np.stack([bot + (top - bot) * (r / (Rs - 1)) for r in range(Rs)])[::-1]
        y_top, y_bot = 17.99, 17.53
        uv = np.stack([np.zeros(g.shape[:2]), (y_top - g[..., 1]) / (y_top - y_bot)], -1).reshape(-1, 2)
        uv[:, 0] = 0.5
        uv[:, 1] = np.clip(uv[:, 1], 0.0, 1.0)
        scl = _fix_orientation(_strip_part(g, OFF_SCLERA, uv))
        scl.W = w_single(scl.n, nb, head)
        scl.edge[:] = 0
        mb.add("eye_white", scl)

        # --- iris -------------------------------------------------------
        cx, cy = IRIS_C[0] * side, IRIS_C[1]
        rx, ry = 0.128, 0.172
        gx, gy = np.meshgrid(np.linspace(cx - rx, cx + rx, 9), np.linspace(cy + ry, cy - ry, 11))
        g = np.stack([gx, gy], -1)
        u = (gx - (cx - rx)) / (2 * rx)
        uv = np.stack([u, (cy + ry - gy) / (2 * ry)], -1).reshape(-1, 2)
        iris = _fix_orientation(_strip_part(g, OFF_IRIS, uv))
        iris.W = w_single(iris.n, nb, eyeb)
        iris.edge[:] = 0
        mb.add("iris", iris)
        iris_rest = g.reshape(-1, 2)

        # --- highlights -------------------------------------------------
        hls = []
        for (hx, hy, hrx, hry) in ((0.040, 0.045, 0.036, 0.030), (-0.040, -0.050, 0.016, 0.013)):
            gx, gy = np.meshgrid(np.linspace(cx + hx - hrx, cx + hx + hrx, 3),
                                 np.linspace(cy + hy + hry, cy + hy - hry, 3))
            g = np.stack([gx, gy], -1)
            uv = np.stack([(gx - gx.min()) / (gx.max() - gx.min()), (gy.max() - gy) / (gy.max() - gy.min())],
                          -1).reshape(-1, 2)
            hp = _fix_orientation(_strip_part(g, OFF_HL, uv))
            hp.W = w_single(hp.n, nb, eyeb)
            hp.edge[:] = 0
            mb.add("highlight", hp)
            hls.append((hp, g.reshape(-1, 2), np.array([cx, cy])))

        # --- eyelid masks (skin overlay with an eye-shaped hole) ---------
        M = 5
        h_up, h_dn = 0.15, 0.13
        rows_up = np.stack([Ue + np.array([0, h_up * (m / (M - 1)) ** 1.0]) for m in range(M)])
        rows_dn = np.stack([Le - np.array([0, h_dn * (m / (M - 1))]) for m in range(M)])
        mask_up = _strip_part(rows_up, OFF_MASK)
        mask_up.uv = face_uv(mask_up.pos)
        mask_dn = _strip_part(rows_dn, OFF_MASK)
        mask_dn.uv = face_uv(mask_dn.pos)
        for mp in (mask_up, mask_dn):
            _fix_orientation(mp)
            mp.W = w_single(mp.n, nb, head)
            mp.edge[:] = 0
            mb.add("face_mask", mp)

        # --- upper lash -------------------------------------------------
        sl = np.r_[np.linspace(0, 1, n)]
        thick = 0.014 + 0.036 * np.clip(sl / 0.65, 0, 1) ** 0.8
        wing = np.array([O + np.array([0.045 * side, 0.016]), O + np.array([0.085 * side, 0.036]),
                         O + np.array([0.11 * side, 0.052])])
        wing_th = np.array([0.036, 0.022, 0.004])
        lash_mid = np.vstack([U, wing])
        lash_th = np.r_[thick, wing_th]
        lash_bot = lash_mid - np.array([0, 0.012])
        lash_bot[n:] = wing - np.array([0, 0.002])
        lash_top = lash_mid + np.c_[np.zeros(len(lash_mid)), lash_th]
        lash_rows = np.stack([lash_bot, lash_top])
        lash = _fix_orientation(_strip_part(lash_rows, OFF_LINE))
        lash.uv[:] = PAL["lash"]
        lash.W = w_single(lash.n, nb, head)
        lash.edge[:] = 0
        mb.add("lines", lash)

        # --- lower lash (outer half) -----------------------------------
        k0 = n // 2
        lo_mid = L[k0:]
        lth = 0.004 + 0.012 * np.linspace(0, 1, len(lo_mid)) ** 1.2
        lo_rows = np.stack([lo_mid + np.array([0, 0.004]), lo_mid - np.c_[np.zeros(len(lo_mid)), lth]])
        lol = _fix_orientation(_strip_part(lo_rows, OFF_LINE))
        lol.uv[:] = PAL["lash_soft"]
        lol.W = w_single(lol.n, nb, head)
        lol.edge[:] = 0
        mb.add("lines", lol)

        # --- crease line -------------------------------------------------
        ks = slice(5, n - 2)
        cr_mid = U[ks] + np.array([0, 0.065])
        cth = 0.008 * np.sin(np.linspace(0.15, np.pi - 0.15, len(cr_mid)))
        cr_rows = np.stack([cr_mid - np.c_[np.zeros(len(cr_mid)), cth * 0.5], cr_mid + np.c_[np.zeros(len(cr_mid)), cth]])
        crease = _fix_orientation(_strip_part(cr_rows, OFF_LINE))
        crease.uv[:] = PAL["crease"]
        crease.W = w_single(crease.n, nb, head)
        crease.edge[:] = 0
        mb.add("lines", crease)

        # ------------------------------------------------------------------
        # morph helpers
        def closed_line(kind):
            if kind == "blink":
                Cl = L + (U - L) * 0.30
                Cl[:, 1] -= 0.012 * np.sin(np.pi * s)
            elif kind == "smile":
                base = I + (O - I) * s[:, None]
                Cl = base.copy()
                Cl[:, 1] = base[:, 1] + 0.085 * np.sin(np.pi * s) ** 0.9 - 0.02
                Cl[:, 0] = (L[:, 0] + U[:, 0]) * 0.5
            elif kind == "calm":
                base = I + (O - I) * s[:, None]
                Cl = base.copy()
                Cl[:, 1] = base[:, 1] - 0.035 * np.sin(np.pi * s) - 0.01
                Cl[:, 0] = (L[:, 0] + U[:, 0]) * 0.5
            return Cl

        def lid_morph(name, en, up_t, dn_t, lash_close=True, wing_drop=0.0, panel=2):
            """up_t / dn_t: targets for the upper / lower lid lines (n, 2) or None."""
            # masks
            for mp, rows, tgt, sign in ((mask_up, rows_up, up_t, 1), (mask_dn, rows_dn, dn_t, -1)):
                if tgt is None:
                    continue
                delta = np.zeros((C, 2))
                delta[lid_cols] = tgt - rows[0][lid_cols]
                tg = rows.copy()
                for m in range(M):
                    w = (1 - m / (M - 1)) ** 1.2
                    tg[m] = rows[m] + delta * w
                off = _offsets_from_2d(rows, tg, OFF_MASK)
                mb.vmorph(name, en, panel, mp, np.arange(mp.n), off)
            # upper lash follows the upper lid
            if up_t is not None:
                tgt_mid = np.vstack([up_t, wing - np.array([0, wing_drop])])
                th = lash_th * (0.75 if lash_close else 1.0)
                tb = tgt_mid - np.array([0, 0.012])
                tb[n:] = tgt_mid[n:] - np.array([0, 0.002])
                tt = tgt_mid + np.c_[np.zeros(len(tgt_mid)), th]
                off = _offsets_from_2d(lash_rows, np.stack([tb, tt]), OFF_LINE)
                mb.vmorph(name, en, panel, lash, np.arange(lash.n), off)
                cr_t = up_t[ks] + np.array([0, 0.065 if not lash_close else 0.05])
                tr = np.stack([cr_t - np.c_[np.zeros(len(cr_t)), cth * 0.5], cr_t + np.c_[np.zeros(len(cr_t)), cth]])
                off = _offsets_from_2d(cr_rows, tr, OFF_LINE)
                mb.vmorph(name, en, panel, crease, np.arange(crease.n), off)
            if dn_t is not None:
                lm = dn_t[k0:]
                k_th = 0.25 if lash_close else 1.0
                tr = np.stack([lm + np.array([0, 0.002]), lm - np.c_[np.zeros(len(lm)), lth * k_th]])
                off = _offsets_from_2d(lo_rows, tr, OFF_LINE)
                mb.vmorph(name, en, panel, lol, np.arange(lol.n), off)

        both = side  # morph names: shared for both eyes unless wink
        Cb = closed_line("blink")
        Cs = closed_line("smile")
        Cc = closed_line("calm")
        lid_morph("まばたき", "blink", Cb, Cb, wing_drop=0.02)
        lid_morph("笑い", "smile", Cs, Cs, wing_drop=0.03)
        lid_morph("なごみ", "calm", Cc, Cc, wing_drop=0.025)
        if side > 0:
            lid_morph("ウィンク", "wink", Cs, Cs, wing_drop=0.03)
            lid_morph("ウィンク２", "wink2", Cb, Cb, wing_drop=0.02)
        else:
            lid_morph("ウィンク右", "wink_R", Cs, Cs, wing_drop=0.03)
            lid_morph("ｳｨﾝｸ２右", "wink2_R", Cb, Cb, wing_drop=0.02)
        jt = U + (L - U) * 0.40
        jt[:, 1] = np.minimum(jt[:, 1], (U[:, 1] + (L[:, 1] - U[:, 1]) * 0.40).mean() + 0.01)
        jt[0], jt[-1] = U[0], U[-1]
        lid_morph("じと目", "half-lid", jt, None, lash_close=False, wing_drop=0.03)
        at = U + (L - U) * (0.42 * (1 - s) ** 1.3 + 0.10)[:, None]
        at[0], at[-1] = U[0], U[-1]
        lid_morph("怒り目", "angry eyes", at, None, lash_close=False, wing_drop=0.0)
        wide_u = U + np.c_[np.zeros(n), 0.035 * np.sin(np.pi * s)]
        wide_l = L - np.c_[np.zeros(n), 0.02 * np.sin(np.pi * s)]
        lid_morph("びっくり", "surprised", wide_u, wide_l, lash_close=False, wing_drop=-0.01)
        # iris shrink for びっくり / 瞳小
        c2 = np.array([cx, cy])
        for name, en, k, pn in (("びっくり", "surprised", 0.78, 2), ("瞳小", "small pupil", 0.72, 2)):
            tg = c2 + (iris_rest - c2) * k
            mb.vmorph(name, en, pn, iris, np.arange(iris.n), _offsets_from_2d(iris_rest, tg, OFF_IRIS))
            for hp, hrest, _ in hls:
                tg = c2 + (hrest - c2) * k
                mb.vmorph(name, en, pn, hp, np.arange(hp.n), _offsets_from_2d(hrest, tg, OFF_HL))
        # store for brows / others
        self.__dict__.setdefault("eye_parts", []).append(dict(lash=lash, iris=iris))

    # ------------------------------------------------------------------
    def brows(self):
        mb, nb = self.mb, self.nb
        head = self.sk["頭"]
        n = 11
        s = np.linspace(0, 1, n)
        for side in (1, -1):
            S = np.array([side, 1.0])
            ctrl = np.array([[0.19, 18.105], [0.33, 18.165], [0.50, 18.185], [0.64, 18.16], [0.74, 18.11]])
            mid = catmull_rom(ctrl, n) * S
            th = 0.030 * (1 - 0.65 * s) * np.clip(s / 0.08 + 0.35, 0, 1)
            th = np.maximum(th, 0.006)
            rows = np.stack([mid - np.c_[np.zeros(n), th * 0.5], mid + np.c_[np.zeros(n), th * 0.5]])
            bp = _fix_orientation(_strip_part(rows, OFF_BROW))
            bp.uv[:] = PAL["brow"]
            bp.W = w_single(bp.n, nb, head)
            bp.edge[:] = 0
            self.mb.add("lines", bp)

            def bm(name, en, dy, dx=None):
                d = np.c_[np.zeros(n) if dx is None else dx * side, dy]
                tg = rows + d[None, :, :]
                mb.vmorph(name, en, 1, bp, np.arange(bp.n), _offsets_from_2d(rows, tg, OFF_BROW))

            bm("真面目", "serious", -0.03 * (1 - s) ** 1.2 - 0.01)
            bm("困る", "troubled", 0.075 * (1 - s) ** 1.4 - 0.025 * s)
            bm("にこり", "cheerful", 0.03 * np.sin(np.pi * s) + 0.015)
            bm("怒り", "angry", -0.075 * (1 - s) ** 1.3 + 0.03 * s, dx=-0.025 * (1 - s))
            bm("上", "brow up", np.full(n, 0.065))
            bm("下", "brow down", np.full(n, -0.04))

    # ------------------------------------------------------------------
    def mouth(self):
        mb, nb = self.mb, self.nb
        head = self.sk["頭"]
        n = 11
        s = np.linspace(0, 1, n)
        hw, my = 0.085, 17.05
        line = np.c_[np.linspace(-hw, hw, n), my - 0.008 * np.sin(np.pi * s)]
        ext_l = np.array([[-hw - 0.09, my + 0.01], [-hw - 0.045, my + 0.004]])
        ext_r = np.array([[hw + 0.045, my + 0.004], [hw + 0.09, my + 0.01]])
        Ue = np.vstack([ext_l, line, ext_r])
        C = len(Ue)
        cols = np.arange(2, 2 + n)
        M = 5
        rows_up = np.stack([Ue + np.array([0, 0.10 * m / (M - 1)]) for m in range(M)])
        rows_dn = np.stack([Ue - np.array([0, 0.24 * m / (M - 1)]) for m in range(M)])
        mu = _fix_orientation(_strip_part(rows_up, OFF_MASK))
        md = _fix_orientation(_strip_part(rows_dn, OFF_MASK))
        for mp in (mu, md):
            mp.uv = face_uv(mp.pos)
            mp.W = w_single(mp.n, nb, head)
            mp.edge[:] = 0
            mb.add("face_mask", mp)
        # interior
        gx, gy = np.meshgrid(np.linspace(-0.15, 0.15, 7), np.linspace(17.09, 16.86, 6))
        g = np.stack([gx, gy], -1)
        uv = np.stack([(gx + 0.15) / 0.30, (17.09 - gy) / 0.23], -1).reshape(-1, 2)
        inner = _fix_orientation(_strip_part(g, OFF_MOUTH_IN, uv))
        inner.W = w_single(inner.n, nb, head)
        inner.edge[:] = 0
        mb.add("mouth_in", inner)
        # teeth (upper)
        ts = slice(2, n - 2)
        tmid = line[ts]
        t_rows = np.stack([tmid + np.array([0, 0.006]), tmid - np.array([0, 0.024])])
        teeth = _fix_orientation(_strip_part(t_rows, OFF_TEETH))
        teeth.uv[:] = PAL["teeth"]
        teeth.W = w_single(teeth.n, nb, head)
        teeth.edge[:] = 0
        mb.add("lines", teeth)
        # line
        lt = 0.011 * np.sin(np.linspace(0.25, np.pi - 0.25, n)) ** 0.6
        l_rows = np.stack([line - np.c_[np.zeros(n), lt * 0.5], line + np.c_[np.zeros(n), lt * 0.5]])
        ml = _fix_orientation(_strip_part(l_rows, OFF_LINE))
        ml.uv[:] = PAL["mouth"]
        ml.W = w_single(ml.n, nb, head)
        ml.edge[:] = 0
        mb.add("lines", ml)

        def mm(name, en, up_d, dn_d, panel=3):
            """up_d / dn_d: (n,2) 2D offsets of upper / lower lip lines."""
            for mp, rows, d in ((mu, rows_up, up_d), (md, rows_dn, dn_d)):
                delta = np.zeros((C, 2))
                delta[cols] = d
                # extension columns follow the nearest corner (keeps the hole closed)
                delta[:2] = d[0]
                delta[-2:] = d[-1]
                tg = rows.copy()
                for m in range(M):
                    tg[m] = rows[m] + delta * (1 - m / (M - 1)) ** 1.2
                mb.vmorph(name, en, panel, mp, np.arange(mp.n), _offsets_from_2d(rows, tg, OFF_MASK))
            tg = l_rows + up_d[None]
            mb.vmorph(name, en, panel, ml, np.arange(ml.n), _offsets_from_2d(l_rows, tg, OFF_LINE))
            tg = t_rows + up_d[ts][None]
            mb.vmorph(name, en, panel, teeth, np.arange(teeth.n), _offsets_from_2d(t_rows, tg, OFF_TEETH))

        x = line[:, 0]
        sn = np.sin(np.pi * s)
        corner = (1 - sn) ** 1.5

        def lips(kx, open_dn, open_up, curve=0.8):
            dx = (kx - 1) * x
            up = np.c_[dx, open_up * sn]
            dn = np.c_[dx, -open_dn * sn ** curve]
            return up, dn

        mm("あ", "a", *lips(0.92, 0.15, 0.012))
        mm("い", "i", *lips(1.25, 0.032, 0.004))
        mm("う", "u", *lips(0.55, 0.05, 0.015, 0.6))
        mm("え", "e", *lips(1.12, 0.075, 0.008))
        mm("お", "o", *lips(0.70, 0.12, 0.02, 0.6))
        up = np.c_[0.08 * x, 0.03 * corner]
        mm("口角上げ", "mouth corner up", up, up)
        dn = np.c_[np.zeros(n), -0.025 * corner]
        mm("口角下げ", "mouth corner down", dn, dn)
        fr = np.c_[np.zeros(n), 0.022 * sn - 0.01 * corner]
        mm("∧", "frown", fr, fr)

    # ------------------------------------------------------------------
    def blush(self):
        mb, nb = self.mb, self.nb
        head = self.sk["頭"]
        for side in (1, -1):
            cx, cy = 0.50 * side, 17.43
            gx, gy = np.meshgrid(np.linspace(cx - 0.24, cx + 0.24, 7), np.linspace(cy + 0.11, cy - 0.11, 4))
            g = np.stack([gx, gy], -1)
            u = (gx - (cx - 0.24)) / 0.48
            uv = np.stack([u, (cy + 0.11 - gy) / 0.22], -1).reshape(-1, 2)
            bp = _fix_orientation(_strip_part(g, OFF_BLUSH, uv))
            bp.W = w_single(bp.n, nb, head)
            bp.edge[:] = 0
            mb.add("blush", bp)

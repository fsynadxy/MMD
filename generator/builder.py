"""Collects parts / materials / morphs and assembles a pmx.Model."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from . import pmx
from .geom import Part, merge_parts


# ---------------------------------------------------------------------------
# weights helpers (dense matrices, reduced to 4 influences at the end)
# ---------------------------------------------------------------------------

def w_single(n, nb, bone):
    W = np.zeros((n, nb))
    W[:, bone] = 1.0
    return W


def w_chain(s, keys, nb):
    """Piecewise-linear blend between bones placed at parameters s_k.

    keys: list of (s_k, bone_index) sorted by s_k.  Values outside clamp."""
    s = np.asarray(s, dtype=np.float64)
    W = np.zeros((len(s), nb))
    ks = np.array([k[0] for k in keys], dtype=np.float64)
    bs = [k[1] for k in keys]
    i = np.clip(np.searchsorted(ks, s, side="right") - 1, 0, len(ks) - 1)
    for r in range(len(s)):
        a = i[r]
        if a >= len(ks) - 1 or s[r] <= ks[0]:
            W[r, bs[a if s[r] > ks[0] else 0]] += 1.0
            continue
        t = (s[r] - ks[a]) / max(ks[a + 1] - ks[a], 1e-9)
        W[r, bs[a]] += 1 - t
        W[r, bs[a + 1]] += t
    return W


def w_mix(W1, W2, t):
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    return W1 * (1 - t) + W2 * t


def reduce_weights(W, k=4):
    n = len(W)
    idx = np.argsort(-W, axis=1)[:, :k]
    w = np.take_along_axis(W, idx, axis=1)
    w = np.maximum(w, 0)
    w[w < 0.005] = 0
    s = w.sum(1, keepdims=True)
    bad = s[:, 0] <= 0
    w = w / np.maximum(s, 1e-12)
    if bad.any():
        w[bad] = (1, 0, 0, 0)
    # quantise to avoid float noise, keep sum 1
    w = np.round(w, 4)
    w[:, 0] += 1 - w.sum(1)
    return idx.astype(np.int64), w


# ---------------------------------------------------------------------------
class MorphSpec:
    def __init__(self, name, name_en, panel, kind):
        self.name, self.name_en, self.panel, self.kind = name, name_en, panel, kind
        self.vertex = []    # (part, local_idx array, offsets (k,3))
        self.material = []  # (material key or None, MaterialMorphOffset-without-index)
        self.group = []     # (morph name, ratio)
        self.bone = []      # (bone index, t, q)


class ModelBuilder:
    def __init__(self, skeleton):
        self.sk = skeleton
        self.textures: list[str] = []
        self.mats: "OrderedDict[str, pmx.Material]" = OrderedDict()
        self.parts: "OrderedDict[str, list[Part]]" = OrderedDict()
        self.morphs: "OrderedDict[str, MorphSpec]" = OrderedDict()
        self.rigid_bodies: list[pmx.RigidBody] = []
        self.joints: list[pmx.Joint] = []
        self.frames: list = []

    @property
    def nb(self):
        return len(self.sk.bones)

    # -- textures / materials -----------------------------------------
    def tex(self, rel_path: str) -> int:
        if rel_path not in self.textures:
            self.textures.append(rel_path)
        return self.textures.index(rel_path)

    def material(self, key: str, mat: pmx.Material):
        self.mats[key] = mat
        self.parts.setdefault(key, [])

    def add(self, key: str, part: Part) -> Part:
        if key not in self.mats:
            raise KeyError("unknown material " + key)
        if part.W is None:
            raise ValueError("part %s has no weights" % part.name)
        if part.W.shape[1] != self.nb:
            raise ValueError("weight width mismatch for %s" % part.name)
        if part.normals is None:
            part.compute_normals()
        self.parts[key].append(part)
        return part

    def bone(self, name) -> int:
        return self.sk[name]

    # -- morphs ---------------------------------------------------------
    def morph(self, name, name_en, panel, kind) -> MorphSpec:
        if name not in self.morphs:
            self.morphs[name] = MorphSpec(name, name_en, panel, kind)
        return self.morphs[name]

    def vmorph(self, name, name_en, panel, part, local_idx, offsets):
        m = self.morph(name, name_en, panel, 1)
        local_idx = np.asarray(local_idx, dtype=np.int64).reshape(-1)
        offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
        if len(local_idx) != len(offsets):
            raise ValueError("morph %s: index/offset mismatch" % name)
        m.vertex.append((part, local_idx, offsets))
        return m

    # -- assembly -------------------------------------------------------
    def build(self, name, name_en, comment, comment_en) -> pmx.Model:
        pos, nrm, uv, edge, faces, Ws = [], [], [], [], [], []
        mats = []
        off = 0
        for key, mat in self.mats.items():
            plist = self.parts[key]
            if not plist:
                continue
            nidx = 0
            for p in plist:
                p.global_offset = off
                pos.append(p.pos)
                nrm.append(p.normals)
                uv.append(p.uv)
                edge.append(p.edge)
                Ws.append(p.W)
                faces.append(p.faces + off)
                nidx += 3 * len(p.faces)
                off += p.n
            mat.index_count = nidx
            mats.append((key, mat))
        P = np.vstack(pos)
        N = np.vstack(nrm)
        T = np.vstack(uv)
        E = np.concatenate(edge)
        Fc = np.vstack(faces)
        W = np.vstack(Ws)
        bi, bw = reduce_weights(W)

        model = pmx.Model(name=name, name_en=name_en, comment=comment, comment_en=comment_en)
        model.positions, model.normals, model.uvs = P, N, T
        model.bone_idx, model.bone_w, model.edge_scale = bi, bw, E
        model.faces = Fc
        model.textures = list(self.textures)
        model.materials = [m for _, m in mats]
        mat_index = {k: i for i, (k, _) in enumerate(mats)}
        model.bones = self.sk.bones

        morph_index = {nm: i for i, nm in enumerate(self.morphs)}
        out = []
        for nm, ms in self.morphs.items():
            mo = pmx.Morph(ms.name, ms.name_en, ms.panel, ms.kind)
            if ms.kind == 1:
                acc = {}
                for part, li, offs in ms.vertex:
                    if not hasattr(part, "global_offset"):
                        raise ValueError("morph %s references a part that was not added" % nm)
                    for i, d in zip(li, offs):
                        gi = int(part.global_offset + i)
                        acc[gi] = acc.get(gi, np.zeros(3)) + d
                mo.offsets = [(k, tuple(v)) for k, v in sorted(acc.items()) if np.abs(v).max() > 1e-7]
            elif ms.kind == 8:
                for key, o in ms.material:
                    mi = -1 if key is None else mat_index[key]
                    o = pmx.MaterialMorphOffset(mi, *o)
                    mo.offsets.append(o)
            elif ms.kind == 0:
                mo.offsets = [(morph_index[g], r) for g, r in ms.group]
            elif ms.kind == 2:
                mo.offsets = list(ms.bone)
            out.append(mo)
        model.morphs = out
        self.morph_index = morph_index
        self.mat_index = mat_index
        model.rigid_bodies = self.rigid_bodies
        model.joints = self.joints
        model.frames = self.frames
        return model

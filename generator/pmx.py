"""Minimal but complete PMX 2.0 writer / reader.

Coordinates follow MMD conventions: left-handed, Y up, the model faces -Z,
character's left side is +X.  A triangle (a, b, c) is front-facing when the
plain cross product (b-a) x (c-a) points outward.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------
# data classes
# --------------------------------------------------------------------------
@dataclass
class Bone:
    name: str
    name_en: str
    pos: Vec3
    parent: int = -1
    layer: int = 0
    tail: Union[int, Vec3] = (0.0, 0.0, 0.0)  # bone index (connected) or offset
    rotatable: bool = True
    movable: bool = False
    visible: bool = True
    operable: bool = True
    append_rotate: bool = False
    append_translate: bool = False
    append_local: bool = False
    append_parent: int = -1
    append_ratio: float = 0.0
    fixed_axis: Optional[Vec3] = None
    local_axis: Optional[Tuple[Vec3, Vec3]] = None  # (x axis, z axis)
    after_physics: bool = False
    ik_target: int = -1
    ik_loop: int = 0
    ik_limit: float = 0.0
    ik_links: List[Tuple[int, Optional[Tuple[Vec3, Vec3]]]] = field(default_factory=list)

    @property
    def is_ik(self) -> bool:
        return self.ik_target >= 0

    def flags(self) -> int:
        f = 0
        if isinstance(self.tail, int):
            f |= 0x0001
        if self.rotatable:
            f |= 0x0002
        if self.movable:
            f |= 0x0004
        if self.visible:
            f |= 0x0008
        if self.operable:
            f |= 0x0010
        if self.is_ik:
            f |= 0x0020
        if self.append_local:
            f |= 0x0080
        if self.append_rotate:
            f |= 0x0100
        if self.append_translate:
            f |= 0x0200
        if self.fixed_axis is not None:
            f |= 0x0400
        if self.local_axis is not None:
            f |= 0x0800
        if self.after_physics:
            f |= 0x1000
        return f


@dataclass
class Material:
    name: str
    name_en: str
    diffuse: Tuple[float, float, float, float] = (1, 1, 1, 1)
    specular: Vec3 = (0.0, 0.0, 0.0)
    spec_power: float = 5.0
    ambient: Vec3 = (0.5, 0.5, 0.5)
    double_sided: bool = False
    ground_shadow: bool = True
    cast_shadow: bool = True   # draw into self-shadow map
    receive_shadow: bool = True
    edge: bool = True
    edge_color: Tuple[float, float, float, float] = (0.25, 0.12, 0.1, 1.0)
    edge_size: float = 1.0
    texture: int = -1
    sphere: int = -1
    sphere_mode: int = 0       # 0 none 1 mul 2 add 3 sub-texture
    toon_shared: bool = False
    toon: int = -1             # texture index, or 0..9 when shared
    memo: str = ""
    index_count: int = 0

    def flags(self) -> int:
        f = 0
        if self.double_sided:
            f |= 0x01
        if self.ground_shadow:
            f |= 0x02
        if self.cast_shadow:
            f |= 0x04
        if self.receive_shadow:
            f |= 0x08
        if self.edge:
            f |= 0x10
        return f


@dataclass
class MaterialMorphOffset:
    material: int          # -1 = all materials
    op: int                # 0 multiply, 1 add
    diffuse: Tuple[float, float, float, float]
    specular: Vec3
    spec_power: float
    ambient: Vec3
    edge_color: Tuple[float, float, float, float]
    edge_size: float
    tex_tint: Tuple[float, float, float, float]
    sphere_tint: Tuple[float, float, float, float]
    toon_tint: Tuple[float, float, float, float]


@dataclass
class Morph:
    name: str
    name_en: str
    panel: int   # 1 eyebrow, 2 eye, 3 mouth, 4 other
    kind: int    # 0 group, 1 vertex, 2 bone, 3 uv, 8 material
    # vertex: list[(vidx, (dx,dy,dz))]; group: list[(morph_idx, ratio)]
    # bone: list[(bone_idx, (tx,ty,tz), (qx,qy,qz,qw))]; material: list[MaterialMorphOffset]
    offsets: list = field(default_factory=list)


@dataclass
class DisplayFrame:
    name: str
    name_en: str
    special: bool = False
    items: List[Tuple[int, int]] = field(default_factory=list)  # (0 bone / 1 morph, index)


@dataclass
class RigidBody:
    name: str
    name_en: str
    bone: int
    group: int            # 0..15
    mask: int             # collision mask (bit set = collide with that group)
    shape: int            # 0 sphere, 1 box, 2 capsule
    size: Vec3
    pos: Vec3
    rot: Vec3             # euler radians
    mass: float = 1.0
    lin_damp: float = 0.5
    ang_damp: float = 0.5
    restitution: float = 0.0
    friction: float = 0.5
    mode: int = 0         # 0 follow bone, 1 physics, 2 physics + bone position


@dataclass
class Joint:
    name: str
    name_en: str
    a: int
    b: int
    pos: Vec3
    rot: Vec3
    pos_min: Vec3 = (0, 0, 0)
    pos_max: Vec3 = (0, 0, 0)
    rot_min: Vec3 = (0, 0, 0)
    rot_max: Vec3 = (0, 0, 0)
    spring_pos: Vec3 = (0, 0, 0)
    spring_rot: Vec3 = (0, 0, 0)


@dataclass
class Model:
    name: str = ""
    name_en: str = ""
    comment: str = ""
    comment_en: str = ""
    positions: np.ndarray = None   # (N,3) float
    normals: np.ndarray = None     # (N,3)
    uvs: np.ndarray = None         # (N,2)
    bone_idx: np.ndarray = None    # (N,4) int
    bone_w: np.ndarray = None      # (N,4) float
    edge_scale: np.ndarray = None  # (N,)
    faces: np.ndarray = None       # (M,3) int
    textures: List[str] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    bones: List[Bone] = field(default_factory=list)
    morphs: List[Morph] = field(default_factory=list)
    frames: List[DisplayFrame] = field(default_factory=list)
    rigid_bodies: List[RigidBody] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
def _signed_size(n: int) -> int:
    if n < 128:
        return 1
    if n < 32768:
        return 2
    return 4


def _vertex_size(n: int) -> int:
    if n < 256:
        return 1
    if n < 65536:
        return 2
    return 4


_SIGNED_FMT = {1: "b", 2: "h", 4: "i"}
_VERTEX_FMT = {1: "B", 2: "H", 4: "i"}


class _Out:
    def __init__(self):
        self.parts: List[bytes] = []

    def raw(self, b: bytes):
        self.parts.append(b)

    def pack(self, fmt: str, *vals):
        self.parts.append(struct.pack("<" + fmt, *vals))

    def text(self, s: str):
        b = s.encode("utf-16-le")
        self.parts.append(struct.pack("<i", len(b)) + b)

    def bytes(self) -> bytes:
        return b"".join(self.parts)


def write_pmx(model: Model, path: str) -> None:
    n_v = len(model.positions)
    vsz = _vertex_size(n_v)
    tsz = _signed_size(len(model.textures))
    msz = _signed_size(len(model.materials))
    bsz = _signed_size(len(model.bones))
    mosz = _signed_size(len(model.morphs))
    rsz = _signed_size(len(model.rigid_bodies))
    vf, tf, mf, bf, mof, rf = (_VERTEX_FMT[vsz], _SIGNED_FMT[tsz], _SIGNED_FMT[msz],
                               _SIGNED_FMT[bsz], _SIGNED_FMT[mosz], _SIGNED_FMT[rsz])

    o = _Out()
    o.raw(b"PMX ")
    o.pack("f", 2.0)
    o.pack("B", 8)
    o.pack("8B", 0, 0, vsz, tsz, msz, bsz, mosz, rsz)  # encoding 0 = UTF-16LE, 0 additional UV
    o.text(model.name)
    o.text(model.name_en)
    o.text(model.comment)
    o.text(model.comment_en)

    # vertices --------------------------------------------------------------
    o.pack("i", n_v)
    P, N, T = model.positions, model.normals, model.uvs
    BI, BW, E = model.bone_idx, model.bone_w, model.edge_scale
    for i in range(n_v):
        o.pack("8f", P[i, 0], P[i, 1], P[i, 2], N[i, 0], N[i, 1], N[i, 2], T[i, 0], T[i, 1])
        w = BW[i]
        bi = BI[i]
        nz = [k for k in range(4) if w[k] > 1e-6]
        if len(nz) <= 1:
            k = nz[0] if nz else 0
            o.pack("B" + bf, 0, int(bi[k]))
        elif len(nz) == 2:
            a, b = nz
            s = w[a] + w[b]
            o.pack("B" + bf + bf + "f", 1, int(bi[a]), int(bi[b]), float(w[a] / s))
        else:
            s = float(w.sum())
            o.pack("B" + bf * 4 + "4f", 2, *[int(x) for x in bi], *[float(x / s) for x in w])
        o.pack("f", float(E[i]))

    # faces -----------------------------------------------------------------
    F = np.asarray(model.faces, dtype=np.int64).reshape(-1)
    o.pack("i", len(F))
    o.raw(np.asarray(F, dtype={1: "<u1", 2: "<u2", 4: "<i4"}[vsz]).tobytes())

    # textures --------------------------------------------------------------
    o.pack("i", len(model.textures))
    for t in model.textures:
        o.text(t)

    # materials -------------------------------------------------------------
    o.pack("i", len(model.materials))
    for m in model.materials:
        o.text(m.name)
        o.text(m.name_en)
        o.pack("4f", *m.diffuse)
        o.pack("3f", *m.specular)
        o.pack("f", m.spec_power)
        o.pack("3f", *m.ambient)
        o.pack("B", m.flags())
        o.pack("4f", *m.edge_color)
        o.pack("f", m.edge_size)
        o.pack(tf + tf, m.texture, m.sphere)
        o.pack("B", m.sphere_mode)
        o.pack("B", 1 if m.toon_shared else 0)
        if m.toon_shared:
            o.pack("B", m.toon)
        else:
            o.pack(tf, m.toon)
        o.text(m.memo)
        o.pack("i", m.index_count)

    # bones -----------------------------------------------------------------
    o.pack("i", len(model.bones))
    for b in model.bones:
        o.text(b.name)
        o.text(b.name_en)
        o.pack("3f", *b.pos)
        o.pack(bf, b.parent)
        o.pack("i", b.layer)
        o.pack("H", b.flags())
        if isinstance(b.tail, int):
            o.pack(bf, b.tail)
        else:
            o.pack("3f", *b.tail)
        if b.append_rotate or b.append_translate:
            o.pack(bf + "f", b.append_parent, b.append_ratio)
        if b.fixed_axis is not None:
            o.pack("3f", *b.fixed_axis)
        if b.local_axis is not None:
            o.pack("3f", *b.local_axis[0])
            o.pack("3f", *b.local_axis[1])
        if b.is_ik:
            o.pack(bf, b.ik_target)
            o.pack("i", b.ik_loop)
            o.pack("f", b.ik_limit)
            o.pack("i", len(b.ik_links))
            for idx, lim in b.ik_links:
                o.pack(bf, idx)
                if lim is None:
                    o.pack("B", 0)
                else:
                    o.pack("B", 1)
                    o.pack("3f", *lim[0])
                    o.pack("3f", *lim[1])

    # morphs ----------------------------------------------------------------
    o.pack("i", len(model.morphs))
    for mo in model.morphs:
        o.text(mo.name)
        o.text(mo.name_en)
        o.pack("BB", mo.panel, mo.kind)
        o.pack("i", len(mo.offsets))
        if mo.kind == 1:
            for vi, d in mo.offsets:
                o.pack(vf + "3f", int(vi), float(d[0]), float(d[1]), float(d[2]))
        elif mo.kind == 0:
            for mi, r in mo.offsets:
                o.pack(mof + "f", int(mi), float(r))
        elif mo.kind == 2:
            for bi_, t, q in mo.offsets:
                o.pack(bf + "3f4f", int(bi_), *t, *q)
        elif mo.kind == 3:
            for vi, d in mo.offsets:
                o.pack(vf + "4f", int(vi), *d)
        elif mo.kind == 8:
            for m in mo.offsets:
                o.pack(mf + "B", m.material, m.op)
                o.pack("4f3ff3f4ff4f4f4f", *m.diffuse, *m.specular, m.spec_power, *m.ambient,
                       *m.edge_color, m.edge_size, *m.tex_tint, *m.sphere_tint, *m.toon_tint)
        else:
            raise ValueError("unsupported morph kind %d" % mo.kind)

    # display frames --------------------------------------------------------
    o.pack("i", len(model.frames))
    for fr in model.frames:
        o.text(fr.name)
        o.text(fr.name_en)
        o.pack("B", 1 if fr.special else 0)
        o.pack("i", len(fr.items))
        for kind, idx in fr.items:
            o.pack("B", kind)
            o.pack(bf if kind == 0 else mof, idx)

    # rigid bodies ----------------------------------------------------------
    o.pack("i", len(model.rigid_bodies))
    for r in model.rigid_bodies:
        o.text(r.name)
        o.text(r.name_en)
        o.pack(bf, r.bone)
        o.pack("BH", r.group, r.mask & 0xFFFF)
        o.pack("B", r.shape)
        o.pack("3f", *r.size)
        o.pack("3f", *r.pos)
        o.pack("3f", *r.rot)
        o.pack("5f", r.mass, r.lin_damp, r.ang_damp, r.restitution, r.friction)
        o.pack("B", r.mode)

    # joints ----------------------------------------------------------------
    o.pack("i", len(model.joints))
    for j in model.joints:
        o.text(j.name)
        o.text(j.name_en)
        o.pack("B", 0)  # spring 6DOF
        o.pack(rf + rf, j.a, j.b)
        for v in (j.pos, j.rot, j.pos_min, j.pos_max, j.rot_min, j.rot_max, j.spring_pos, j.spring_rot):
            o.pack("3f", *v)

    with open(path, "wb") as fh:
        fh.write(o.bytes())


# --------------------------------------------------------------------------
# reader (used for validation / round trip)
# --------------------------------------------------------------------------
class _In:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0

    def unpack(self, fmt: str):
        s = struct.Struct("<" + fmt)
        v = s.unpack_from(self.d, self.p)
        self.p += s.size
        return v

    def one(self, fmt: str):
        return self.unpack(fmt)[0]

    def text(self, enc: str) -> str:
        n = self.one("i")
        s = self.d[self.p:self.p + n].decode(enc)
        self.p += n
        return s


def read_pmx(path: str) -> dict:
    with open(path, "rb") as fh:
        r = _In(fh.read())
    assert r.d[:4] == b"PMX ", "bad signature"
    r.p = 4
    ver = r.one("f")
    nglob = r.one("B")
    g = r.unpack("%dB" % nglob)
    enc = "utf-16-le" if g[0] == 0 else "utf-8"
    n_add_uv, vsz, tsz, msz, bsz, mosz, rsz = g[1:8]
    vf, tf, mf, bf, mof, rf = (_VERTEX_FMT[vsz], _SIGNED_FMT[tsz], _SIGNED_FMT[msz],
                               _SIGNED_FMT[bsz], _SIGNED_FMT[mosz], _SIGNED_FMT[rsz])
    out = {"version": ver, "globals": g}
    out["name"], out["name_en"], out["comment"], out["comment_en"] = (
        r.text(enc), r.text(enc), r.text(enc), r.text(enc))

    nv = r.one("i")
    verts = []
    for _ in range(nv):
        pos_n_uv = r.unpack("8f")
        if n_add_uv:
            r.unpack("%df" % (4 * n_add_uv))
        t = r.one("B")
        if t == 0:
            bones = (r.one(bf),)
            ws = (1.0,)
        elif t == 1:
            b1, b2, w = r.unpack(bf + bf + "f")
            bones, ws = (b1, b2), (w, 1 - w)
        elif t == 2:
            v = r.unpack(bf * 4 + "4f")
            bones, ws = v[:4], v[4:]
        elif t == 3:
            v = r.unpack(bf + bf + "f9f")
            bones, ws = v[:2], (v[2], 1 - v[2])
        else:
            raise ValueError("bad weight type %d" % t)
        edge = r.one("f")
        verts.append((pos_n_uv, t, bones, ws, edge))
    out["vertices"] = verts

    ni = r.one("i")
    fmt = {1: "<u1", 2: "<u2", 4: "<i4"}[vsz]
    idx = np.frombuffer(r.d, dtype=fmt, count=ni, offset=r.p)
    r.p += ni * vsz
    out["indices"] = idx

    out["textures"] = [r.text(enc) for _ in range(r.one("i"))]

    mats = []
    for _ in range(r.one("i")):
        m = {"name": r.text(enc), "name_en": r.text(enc)}
        m["diffuse"] = r.unpack("4f")
        m["specular"] = r.unpack("3f")
        m["spec_power"] = r.one("f")
        m["ambient"] = r.unpack("3f")
        m["flags"] = r.one("B")
        m["edge_color"] = r.unpack("4f")
        m["edge_size"] = r.one("f")
        m["texture"], m["sphere"] = r.unpack(tf + tf)
        m["sphere_mode"] = r.one("B")
        shared = r.one("B")
        m["toon_shared"] = shared
        m["toon"] = r.one("B") if shared else r.one(tf)
        m["memo"] = r.text(enc)
        m["index_count"] = r.one("i")
        mats.append(m)
    out["materials"] = mats

    bones = []
    for _ in range(r.one("i")):
        b = {"name": r.text(enc), "name_en": r.text(enc)}
        b["pos"] = r.unpack("3f")
        b["parent"] = r.one(bf)
        b["layer"] = r.one("i")
        fl = r.one("H")
        b["flags"] = fl
        b["tail"] = r.one(bf) if fl & 1 else r.unpack("3f")
        if fl & 0x300:
            b["append"] = r.unpack(bf + "f")
        if fl & 0x400:
            b["fixed_axis"] = r.unpack("3f")
        if fl & 0x800:
            b["local_axis"] = (r.unpack("3f"), r.unpack("3f"))
        if fl & 0x2000:
            b["ext_key"] = r.one("i")
        if fl & 0x20:
            ik = {"target": r.one(bf), "loop": r.one("i"), "limit": r.one("f"), "links": []}
            for _ in range(r.one("i")):
                li = r.one(bf)
                has = r.one("B")
                lim = (r.unpack("3f"), r.unpack("3f")) if has else None
                ik["links"].append((li, lim))
            b["ik"] = ik
        bones.append(b)
    out["bones"] = bones

    morphs = []
    for _ in range(r.one("i")):
        mo = {"name": r.text(enc), "name_en": r.text(enc)}
        mo["panel"], mo["kind"] = r.unpack("BB")
        n = r.one("i")
        offs = []
        for _ in range(n):
            k = mo["kind"]
            if k == 0:
                offs.append(r.unpack(mof + "f"))
            elif k == 1:
                offs.append(r.unpack(vf + "3f"))
            elif k == 2:
                offs.append(r.unpack(bf + "3f4f"))
            elif 3 <= k <= 7:
                offs.append(r.unpack(vf + "4f"))
            elif k == 8:
                offs.append(r.unpack(mf + "B" + "4f3ff3f4ff4f4f4f"))
            else:
                raise ValueError("unsupported morph kind %d" % k)
        mo["offsets"] = offs
        morphs.append(mo)
    out["morphs"] = morphs

    frames = []
    for _ in range(r.one("i")):
        fr = {"name": r.text(enc), "name_en": r.text(enc), "special": r.one("B"), "items": []}
        for _ in range(r.one("i")):
            k = r.one("B")
            fr["items"].append((k, r.one(bf if k == 0 else mof)))
        frames.append(fr)
    out["frames"] = frames

    rbs = []
    for _ in range(r.one("i")):
        rb = {"name": r.text(enc), "name_en": r.text(enc)}
        rb["bone"] = r.one(bf)
        rb["group"], rb["mask"] = r.unpack("BH")
        rb["shape"] = r.one("B")
        rb["size"] = r.unpack("3f")
        rb["pos"] = r.unpack("3f")
        rb["rot"] = r.unpack("3f")
        rb["mass"], rb["lin_damp"], rb["ang_damp"], rb["restitution"], rb["friction"] = r.unpack("5f")
        rb["mode"] = r.one("B")
        rbs.append(rb)
    out["rigid_bodies"] = rbs

    joints = []
    for _ in range(r.one("i")):
        j = {"name": r.text(enc), "name_en": r.text(enc), "type": r.one("B")}
        j["a"], j["b"] = r.unpack(rf + rf)
        for key in ("pos", "rot", "pos_min", "pos_max", "rot_min", "rot_max", "spring_pos", "spring_rot"):
            j[key] = r.unpack("3f")
        joints.append(j)
    out["joints"] = joints
    out["trailing_bytes"] = len(r.d) - r.p
    return out

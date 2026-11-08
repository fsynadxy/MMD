"""Rigid body / joint helpers.

Rigid body rotations are generated with the X angle fixed at 0, so the
result is identical whether a loader composes the Euler angles as YXZ (MMD)
or XYZ (three.js): R = Ry(a) * Rz(c)."""
from __future__ import annotations

import math

import numpy as np

from . import pmx
from .geom import normalize

G_BODY, G_HEAD, G_HAIR, G_SKIRT, G_ACC = 0, 1, 2, 3, 4
ALL = 0xFFFF


def mask_without(*groups):
    m = ALL
    for g in groups:
        m &= ~(1 << g)
    return m


MASK = {
    G_BODY: mask_without(G_BODY),
    G_HEAD: mask_without(G_HEAD, G_HAIR, G_ACC),
    G_HAIR: mask_without(G_HEAD, G_HAIR, G_SKIRT, G_ACC),
    G_SKIRT: mask_without(G_HEAD, G_HAIR, G_SKIRT, G_ACC),
    G_ACC: mask_without(G_HEAD, G_HAIR, G_SKIRT, G_ACC),
}


def rot_y_to(d):
    """Euler (0, a, c) such that Ry(a) Rz(c) maps +Y onto direction d."""
    d = normalize(np.asarray(d, dtype=np.float64))
    s = math.hypot(d[0], d[2])
    if s < 1e-8:
        return (0.0, 0.0, 0.0 if d[1] > 0 else math.pi)
    a = math.atan2(d[2], -d[0])
    c = math.atan2(s, d[1])
    return (0.0, a, c)


class Physics:
    def __init__(self, mb):
        self.mb = mb
        self.index = {}

    def body(self, name, en, bone, shape, size, pos, rot=(0, 0, 0), group=G_BODY, mode=0, mass=1.0,
             lin=0.5, ang=0.5, rest=0.0, fric=0.5):
        rb = pmx.RigidBody(name, en, self.mb.sk[bone] if isinstance(bone, str) else bone, group, MASK[group],
                           shape, tuple(float(x) for x in size), tuple(float(x) for x in pos),
                           tuple(float(x) for x in rot), mass, lin, ang, rest, fric, mode)
        self.mb.rigid_bodies.append(rb)
        self.index[name] = len(self.mb.rigid_bodies) - 1
        return self.index[name]

    def capsule(self, name, en, bone, p0, p1, r, **kw):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        return self.body(name, en, bone, 2, (r, max(L, 0.01), 0), (p0 + p1) / 2, rot_y_to(p1 - p0), **kw)

    def joint(self, name, en, a, b, pos, rot_lim, spring_rot=(0, 0, 0), pos_lim=(0, 0, 0), spring_pos=(0, 0, 0),
              rot=(0.0, 0.0, 0.0)):
        rl = np.radians(np.asarray(rot_lim, dtype=np.float64))
        pl = np.asarray(pos_lim, dtype=np.float64)
        j = pmx.Joint(name, en, self.index[a] if isinstance(a, str) else a,
                      self.index[b] if isinstance(b, str) else b, tuple(float(x) for x in pos),
                      tuple(float(x) for x in rot),
                      tuple(-pl), tuple(pl), tuple(-rl), tuple(rl), tuple(spring_pos), tuple(spring_rot))
        self.mb.joints.append(j)
        return j


def body_colliders(ph: Physics, sk):
    """Static colliders that follow the body (group 0) and head (group 1)."""
    ph.body("頭", "head", "頭", 0, (1.02, 0, 0), (0, 18.3, 0.12), group=G_HEAD)
    ph.capsule("首", "neck", "首", (0, 15.9, 0.12), (0, 16.6, 0.12), 0.26)
    ph.capsule("上半身2", "upper body2", "上半身2", (0, 14.0, 0.08), (0, 15.2, 0.08), 0.62)
    ph.capsule("上半身", "upper body", "上半身", (0, 12.7, 0.08), (0, 13.6, 0.08), 0.55)
    ph.capsule("下半身", "lower body", "下半身", (-0.5, 11.75, 0.08), (0.5, 11.75, 0.08), 0.60)
    for side, lr in ((1, "左"), (-1, "右")):
        S = np.array([side, 1, 1])
        from .skeleton import ANKLE, KNEE, LEG
        ph.capsule(lr + "足", "leg" + ("_L" if side > 0 else "_R"), lr + "足", LEG * S + np.array([0, -0.6, 0]),
                   KNEE * S + np.array([0, 0.3, 0]), 0.46)
        ph.capsule(lr + "ひざ", "knee" + ("_L" if side > 0 else "_R"), lr + "ひざ", KNEE * S + np.array([0, -0.2, 0]),
                   ANKLE * S + np.array([0, 0.6, 0]), 0.36)
        from .skeleton import ELBOW, SHOULDER, WRIST
        ph.capsule(lr + "腕", "arm" + ("_L" if side > 0 else "_R"), lr + "腕", SHOULDER * S + (ELBOW - SHOULDER) * S * 0.15,
                   ELBOW * S, 0.33)
        ph.capsule(lr + "ひじ", "elbow" + ("_L" if side > 0 else "_R"), lr + "ひじ", ELBOW * S, WRIST * S, 0.30)

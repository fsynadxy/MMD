"""Head / face geometry (SDF based) and facial feature decals."""
from __future__ import annotations

import numpy as np

from .geom import (sd_ellipsoid, sd_capsule, smin, smax, smoothstep, surface_nets, normalize,
                   ray_sdf, sdf_gradient, project_to_surface, pchip)

Y_C = 18.35       # start of the cranium cap
Y_TOP = 19.6
Y_BOT = 16.47     # bottom of the chin

#            y      a      front   back    z0
_TAB = np.array([
    [16.47, 0.00, -0.64, -0.64, -0.64],
    [16.52, 0.13, -0.71, -0.57, -0.62],
    [16.62, 0.28, -0.785, -0.44, -0.55],
    [16.75, 0.43, -0.835, -0.24, -0.45],
    [16.95, 0.63, -0.875, 0.12, -0.30],
    [17.20, 0.81, -0.915, 0.58, -0.12],
    [17.50, 0.925, -0.945, 0.95, 0.00],
    [17.75, 0.975, -0.965, 1.12, 0.05],
    [18.00, 1.00, -0.99, 1.20, 0.08],
    [18.35, 1.02, -1.01, 1.24, 0.10],
])
_A = pchip(_TAB[:, 0], _TAB[:, 1])
_F = pchip(_TAB[:, 0], _TAB[:, 2])
_B = pchip(_TAB[:, 0], _TAB[:, 3])
_Z0 = pchip(_TAB[:, 0], _TAB[:, 4])
FRONT_POW = 2.5


def _profiles(y):
    cap = np.sqrt(np.clip(1 - ((y - Y_C) / (Y_TOP - Y_C)) ** 2, 0, 1))
    up = y > Y_C
    yy = np.minimum(y, Y_C)
    a = _A(yy)
    fr = _F(yy)
    bk = _B(yy)
    z0 = _Z0(yy)
    a = np.where(up, a * cap, a)
    f = np.where(up, (z0 - fr) * cap ** 0.9, z0 - fr)
    b = np.where(up, (bk - z0) * cap ** 0.95, bk - z0)
    return a, z0, f, b


def skull_sdf(p):
    p = np.asarray(p, dtype=np.float64)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    lo, hi = Y_BOT + 1e-3, Y_TOP - 1e-3
    yc = np.clip(y, lo, hi)
    a, z0, f, b = _profiles(yc)
    zz = z - z0
    front = zz < 0
    d = np.where(front, f, b)
    a = np.maximum(a, 1e-3)
    d = np.maximum(d, 1e-3)
    pw = np.where(front, FRONT_POW, 2.0)
    r = (np.abs(x / a) ** pw + np.abs(zz / d) ** pw) ** (1.0 / pw)
    dist = (r - 1) * np.minimum(np.minimum(a, d), 0.6)
    return np.where((y < lo) | (y > hi), np.maximum(dist, 0) + np.abs(y - yc), dist)


def head_sdf(p):
    p = np.asarray(p, dtype=np.float64)
    d = skull_sdf(p)
    nose = sd_ellipsoid(p, np.array([0, 17.33, -0.905]), np.array([0.03, 0.08, 0.05]))
    d = smin(d, nose, 0.05)
    neck = sd_capsule(p, np.array([0, 15.2, 0.14]), np.array([0, 17.3, 0.2]), 0.28)
    d = smin(d, neck, 0.10)
    return d


def build_head_mesh(h=0.04):
    part = surface_nets(head_sdf, (-1.25, 15.3, -1.25), (1.25, 19.75, 1.4), h)
    return part

"""Geometry helpers: mesh parts, lofting, SDF + surface nets, normals."""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------

def v3(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def normalize(v, axis=-1, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def rot_axis(axis, ang):
    """Rotation matrix (column vectors) for a rotation of `ang` radians about axis."""
    a = normalize(axis)
    x, y, z = a
    c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def catmull_rom(points, n, closed=False):
    """Centripetal-ish uniform Catmull-Rom through control points -> n samples."""
    P = np.asarray(points, dtype=np.float64)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    segs = len(P) - 3
    out = []
    ts = np.linspace(0, segs, n)
    for t in ts:
        i = min(int(np.floor(t)), segs - 1)
        u = t - i
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        u2, u3 = u * u, u * u * u
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    return np.array(out)


def resample_polyline(P, n):
    P = np.asarray(P, dtype=np.float64)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, P[:, k]) for k in range(P.shape[1])], axis=1)


def arc_length(P):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


# ---------------------------------------------------------------------------
# mesh part
# ---------------------------------------------------------------------------
class Part:
    """A piece of geometry that belongs to one material."""

    def __init__(self, pos, faces, uv=None, normals=None, name=""):
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        n = len(self.pos)
        self.uv = np.zeros((n, 2)) if uv is None else np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        self.normals = normals
        self.W = None            # dense weights (n, n_bones)
        self.edge = np.ones(n)
        self.name = name

    @property
    def n(self):
        return len(self.pos)

    def copy(self):
        p = Part(self.pos.copy(), self.faces.copy(), self.uv.copy(),
                 None if self.normals is None else self.normals.copy(), self.name)
        p.W = None if self.W is None else self.W.copy()
        p.edge = self.edge.copy()
        return p

    def flip(self):
        self.faces = self.faces[:, ::-1].copy()
        if self.normals is not None:
            self.normals = -self.normals
        return self

    def compute_normals(self, weld=True, tol=1e-5):
        self.normals = vertex_normals(self.pos, self.faces, weld=weld, tol=tol)
        return self

    def mirrored_x(self):
        """Mirror across x = 0 (keeps faces front-facing)."""
        p = self.copy()
        p.pos[:, 0] *= -1
        if p.normals is not None:
            p.normals[:, 0] *= -1
        p.faces = p.faces[:, ::-1].copy()
        return p


def merge_parts(parts, name=""):
    parts = [p for p in parts if p is not None and p.n > 0]
    pos, faces, uv, nrm, W, edge = [], [], [], [], [], []
    off = 0
    have_n = all(p.normals is not None for p in parts)
    have_w = all(p.W is not None for p in parts)
    for p in parts:
        pos.append(p.pos)
        faces.append(p.faces + off)
        uv.append(p.uv)
        edge.append(p.edge)
        if have_n:
            nrm.append(p.normals)
        if have_w:
            W.append(p.W)
        off += p.n
    out = Part(np.vstack(pos), np.vstack(faces), np.vstack(uv), np.vstack(nrm) if have_n else None, name)
    if have_w:
        out.W = np.vstack(W)
    out.edge = np.concatenate(edge)
    return out


def vertex_normals(pos, faces, weld=True, tol=1e-5):
    pos = np.asarray(pos)
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    fn = np.cross(b - a, c - a)  # area weighted
    if weld:
        key = np.round(pos / tol).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        acc = np.zeros((inv.max() + 1, 3))
        for k in range(3):
            np.add.at(acc, inv[faces[:, k]], fn)
        n = acc[inv]
    else:
        n = np.zeros_like(pos)
        for k in range(3):
            np.add.at(n, faces[:, k], fn)
    bad = np.linalg.norm(n, axis=1) < 1e-14
    n[bad] = (0, 1, 0)
    return normalize(n)


# ---------------------------------------------------------------------------
# grid / loft
# ---------------------------------------------------------------------------

def grid_faces(rows, cols, wrap_cols=False, wrap_rows=False):
    """Quad grid faces for vertices laid out row-major (rows x cols).

    Triangle normal = d(col) x d(row).  Use Part.flip() if it points inwards.
    """
    faces = []
    cmax = cols if wrap_cols else cols - 1
    rmax = rows if wrap_rows else rows - 1
    for r in range(rmax):
        r1 = (r + 1) % rows
        for c in range(cmax):
            c1 = (c + 1) % cols
            a = r * cols + c
            b = r * cols + c1
            cc = r1 * cols + c1
            d = r1 * cols + c
            faces.append((a, b, cc))
            faces.append((a, cc, d))
    return np.array(faces, dtype=np.int64).reshape(-1, 3)


def grid_part(P, uv=None, name=""):
    """P: (rows, cols, 3) -> Part with quad grid; uv default (col/(cols-1), row/(rows-1))."""
    P = np.asarray(P, dtype=np.float64)
    R, C = P.shape[:2]
    if uv is None:
        uu, vv = np.meshgrid(np.linspace(0, 1, C), np.linspace(0, 1, R))
        uv = np.stack([uu, vv], axis=-1)
    return Part(P.reshape(-1, 3), grid_faces(R, C), np.asarray(uv).reshape(-1, 2), name=name)


def tube_part(rings, v_coords=None, cap_start=False, cap_end=False, name="", u_scale=1.0):
    """Closed rings (R, K, 3) -> tube with a duplicated seam column for UVs.

    Ring points should run counter-clockwise when looking along the tube
    direction from the start ring... orientation is fixed afterwards with
    orient_outward().
    """
    rings = np.asarray(rings, dtype=np.float64)
    R, K = rings.shape[:2]
    P = np.concatenate([rings, rings[:, :1]], axis=1)  # duplicate seam
    if v_coords is None:
        v_coords = np.linspace(0, 1, R)
    uu, vv = np.meshgrid(np.linspace(0, u_scale, K + 1), v_coords)
    part = Part(P.reshape(-1, 3), grid_faces(R, K + 1), np.stack([uu, vv], -1).reshape(-1, 2), name=name)
    extra_pos, extra_faces, extra_uv = [], [], []
    base = part.n
    if cap_start:
        c = rings[0].mean(axis=0)
        extra_pos.append(c)
        extra_uv.append((0.5, v_coords[0]))
        ci = base
        base += 1
        for k in range(K):
            extra_faces.append((ci, k + 1, k))
    if cap_end:
        c = rings[-1].mean(axis=0)
        extra_pos.append(c)
        extra_uv.append((0.5, v_coords[-1]))
        ci = base
        o = (R - 1) * (K + 1)
        for k in range(K):
            extra_faces.append((ci, o + k, o + k + 1))
    if extra_pos:
        part.pos = np.vstack([part.pos, np.array(extra_pos)])
        part.uv = np.vstack([part.uv, np.array(extra_uv)])
        part.faces = np.vstack([part.faces, np.array(extra_faces, dtype=np.int64)])
        part.edge = np.ones(part.n)
    return part


def orient_outward(part, inside_point_fn):
    """Flip all faces if the majority points toward the 'inside' reference.

    inside_point_fn(face_centers) -> reference interior points (F,3)."""
    p = part.pos
    f = part.faces
    a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    n = np.cross(b - a, c - a)
    cen = (a + b + c) / 3
    ref = inside_point_fn(cen)
    s = np.sum(n * (cen - ref), axis=1)
    if np.sum(np.sign(s) * np.linalg.norm(n, axis=1)) < 0:
        part.flip()
    return part


def ellipse_ring(center, ax_u, ax_v, ru, rv, K, phase=0.0, power=2.0, fn=None):
    """Ring in the plane spanned by ax_u, ax_v (unit).  fn(theta)->(du,dv) extra."""
    th = phase + np.linspace(0, 2 * np.pi, K, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    e = 2.0 / power
    cu = np.sign(c) * np.abs(c) ** e
    sv = np.sign(s) * np.abs(s) ** e
    U = ru * cu
    V = rv * sv
    if fn is not None:
        du, dv = fn(th)
        U = U + du
        V = V + dv
    return center[None, :] + U[:, None] * ax_u[None, :] + V[:, None] * ax_v[None, :]


def frames_along(curve, up_hint):
    """Parallel-transport frames along a polyline. Returns T, N, B (n,3).

    N starts as up_hint projected perpendicular to the first tangent."""
    C = np.asarray(curve)
    n = len(C)
    T = np.zeros_like(C)
    T[1:-1] = C[2:] - C[:-2]
    T[0] = C[1] - C[0]
    T[-1] = C[-1] - C[-2]
    T = normalize(T)
    N = np.zeros_like(C)
    up = np.asarray(up_hint, dtype=np.float64)
    n0 = up - T[0] * np.dot(up, T[0])
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross(T[0], [1, 0, 0])
    N[0] = normalize(n0)
    for i in range(1, n):
        v = N[i - 1] - T[i] * np.dot(N[i - 1], T[i])
        if np.linalg.norm(v) < 1e-9:
            v = N[i - 1]
        N[i] = normalize(v)
    B = np.cross(T, N)
    return T, N, B


# ---------------------------------------------------------------------------
# signed distance functions
# ---------------------------------------------------------------------------

def sd_ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_capsule(p, a, b, r):
    pa = p - a
    ba = np.asarray(b) - np.asarray(a)
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    rr = r if np.isscalar(r) else (r[0] + (r[1] - r[0]) * h)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - rr


def sd_round_box(p, c, half, r, R=None):
    """Rounded box in a frame R (3x3 columns = local axes)."""
    q = p - c
    if R is not None:
        q = q @ R
    d = np.abs(q) - (np.asarray(half) - r)
    return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(np.max(d, axis=1), 0) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return lerp(b, a, h) - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sdf_gradient(f, p, eps=1e-4):
    g = np.zeros_like(p)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        g[:, k] = (f(p + d) - f(p - d)) / (2 * eps)
    return g


def project_to_surface(f, p, iters=8, max_step=None):
    p = np.array(p, dtype=np.float64)
    for _ in range(iters):
        d = f(p)
        g = sdf_gradient(f, p)
        gg = np.maximum(np.sum(g * g, axis=1), 1e-12)
        step = (d / gg)[:, None] * g
        if max_step is not None:
            ln = np.linalg.norm(step, axis=1, keepdims=True)
            step *= np.minimum(1.0, max_step / np.maximum(ln, 1e-12))
        p -= step
    return p


def surface_nets(f, lo, hi, h, project=True):
    """Naive surface nets on SDF f over box [lo, hi] with spacing h."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    nx, ny, nz = (np.ceil((hi - lo) / h).astype(int) + 1)
    xs = lo[0] + h * np.arange(nx)
    ys = lo[1] + h * np.arange(ny)
    zs = lo[2] + h * np.arange(nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vals = np.empty(len(pts))
    CH = 400000
    for i in range(0, len(pts), CH):
        vals[i:i + CH] = f(pts[i:i + CH])
    F = vals.reshape(nx, ny, nz)
    inside = F < 0

    cx, cy, cz = nx - 1, ny - 1, nz - 1
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    cin = np.stack([inside[i:i + cx, j:j + cy, k:k + cz] for i, j, k in corners], 0)
    active = cin.any(0) & ~cin.all(0)
    idx = -np.ones((cx, cy, cz), dtype=np.int64)
    act = np.argwhere(active)
    idx[active] = np.arange(len(act))

    # vertex position: mass point of the edge crossings in each active cell
    acc = np.zeros((len(act), 3))
    cnt = np.zeros(len(act))
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]
    ai, aj, ak = act[:, 0], act[:, 1], act[:, 2]
    for (a, b) in edges:
        fa = F[ai + a[0], aj + a[1], ak + a[2]]
        fb = F[ai + b[0], aj + b[1], ak + b[2]]
        m = (fa < 0) != (fb < 0)
        t = np.where(m, fa / np.where(m, fa - fb, 1.0), 0)
        pa = np.stack([ai + a[0], aj + a[1], ak + a[2]], 1).astype(np.float64)
        pb = np.stack([ai + b[0], aj + b[1], ak + b[2]], 1).astype(np.float64)
        pc = pa + (pb - pa) * t[:, None]
        acc[m] += pc[m]
        cnt[m] += 1
    V = lo + h * (acc / np.maximum(cnt, 1)[:, None])

    quads = []
    # x edges
    e = inside[:-1, :, :] != inside[1:, :, :]
    e[:, 0, :] = False
    e[:, :, 0] = False
    for i, j, k in np.argwhere(e):
        q = [idx[i, j - 1, k - 1], idx[i, j, k - 1], idx[i, j, k], idx[i, j - 1, k]]
        quads.append(q if inside[i, j, k] else q[::-1])
    e = inside[:, :-1, :] != inside[:, 1:, :]
    e[0, :, :] = False
    e[:, :, 0] = False
    for i, j, k in np.argwhere(e):
        q = [idx[i - 1, j, k - 1], idx[i - 1, j, k], idx[i, j, k], idx[i, j, k - 1]]
        quads.append(q if inside[i, j, k] else q[::-1])
    e = inside[:, :, :-1] != inside[:, :, 1:]
    e[0, :, :] = False
    e[:, 0, :] = False
    for i, j, k in np.argwhere(e):
        q = [idx[i - 1, j - 1, k], idx[i, j - 1, k], idx[i, j, k], idx[i - 1, j, k]]
        quads.append(q if inside[i, j, k] else q[::-1])
    Q = np.array(quads, dtype=np.int64)
    if project:
        V = project_to_surface(f, V, iters=6, max_step=h)
    # split quads along the shorter diagonal
    d1 = np.linalg.norm(V[Q[:, 0]] - V[Q[:, 2]], axis=1)
    d2 = np.linalg.norm(V[Q[:, 1]] - V[Q[:, 3]], axis=1)
    use1 = d1 <= d2
    tri = np.concatenate([
        np.stack([Q[use1, 0], Q[use1, 1], Q[use1, 2]], 1),
        np.stack([Q[use1, 0], Q[use1, 2], Q[use1, 3]], 1),
        np.stack([Q[~use1, 0], Q[~use1, 1], Q[~use1, 3]], 1),
        np.stack([Q[~use1, 1], Q[~use1, 2], Q[~use1, 3]], 1)])
    # analytic-ish normals from the SDF gradient
    N = normalize(sdf_gradient(f, V, eps=h * 0.25))
    part = Part(V, tri, normals=N)
    return part


def laplacian_smooth(part, iters=2, lam=0.5, fixed=None):
    n = part.n
    f = part.faces
    nb = [[] for _ in range(n)]
    for a, b, c in f:
        nb[a] += [b, c]
        nb[b] += [a, c]
        nb[c] += [a, b]
    nb = [np.unique(x) for x in nb]
    P = part.pos.copy()
    for _ in range(iters):
        avg = np.array([P[x].mean(0) if len(x) else P[i] for i, x in enumerate(nb)])
        upd = P + lam * (avg - P)
        if fixed is not None:
            upd[fixed] = P[fixed]
        P = upd
    part.pos = P
    return part


def ray_sdf(f, origins, dirs, t_max=10.0, steps=200, scale=0.7):
    """Sphere-trace rays; returns hit points (nan where no hit)."""
    o = np.asarray(origins, dtype=np.float64)
    d = normalize(dirs)
    t = np.zeros(len(o))
    alive = np.ones(len(o), bool)
    for _ in range(steps):
        p = o + d * t[:, None]
        dist = f(p)
        hit = np.abs(dist) < 1e-5
        alive &= ~hit
        t = np.where(alive, t + np.maximum(dist * scale, 1e-4), t)
        alive &= t < t_max
        if not alive.any():
            break
    p = o + d * t[:, None]
    p = project_to_surface(f, p, iters=4)
    p[t >= t_max] = np.nan
    return p


def pchip(xs, ys):
    """Monotone cubic (Fritsch-Carlson) interpolator; xs ascending. Returns f(x)."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    h = np.diff(xs)
    dl = np.diff(ys) / h
    m = np.zeros_like(ys)
    m[0], m[-1] = dl[0], dl[-1]
    for i in range(1, len(xs) - 1):
        if dl[i - 1] * dl[i] <= 0:
            m[i] = 0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i])

    def f(x):
        x = np.asarray(x, dtype=np.float64)
        xc = np.clip(x, xs[0], xs[-1])
        i = np.clip(np.searchsorted(xs, xc) - 1, 0, len(xs) - 2)
        t = (xc - xs[i]) / h[i]
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i] +
                (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1])
    return f

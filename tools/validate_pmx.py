"""Structural validation of a PMX file (index ranges, names, weights, IK, morphs, physics).

    python tools/validate_pmx.py model/Touka/Touka.pmx
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generator.pmx import read_pmx  # noqa: E402

STANDARD = ["全ての親", "センター", "グルーブ", "腰", "上半身", "上半身2", "首", "頭", "両目", "左目", "右目",
            "下半身"]
for lr in ("左", "右"):
    STANDARD += [lr + n for n in ("肩P", "肩", "肩C", "腕", "腕捩", "腕捩1", "腕捩2", "腕捩3", "ひじ", "手捩", "手捩1",
                                  "手捩2", "手捩3", "手首", "親指０", "親指１", "親指２", "人指１", "人指２", "人指３",
                                  "中指１", "中指２", "中指３", "薬指１", "薬指２", "薬指３", "小指１", "小指２", "小指３",
                                  "足", "ひざ", "足首", "つま先", "足ＩＫ", "つま先ＩＫ", "足IK親", "足D", "ひざD", "足首D",
                                  "足先EX")]
    STANDARD.append("腰キャンセル" + lr)


def main(path):
    r = read_pmx(path)
    errors, warns = [], []
    nv = len(r["vertices"])
    idx = r["indices"]
    nb = len(r["bones"])
    nm = len(r["materials"])
    nmo = len(r["morphs"])
    nr = len(r["rigid_bodies"])
    nt = len(r["textures"])
    if r["version"] != 2.0:
        errors.append("version %s" % r["version"])
    if r["trailing_bytes"]:
        errors.append("%d trailing bytes" % r["trailing_bytes"])
    if len(idx) % 3:
        errors.append("index count not multiple of 3")
    if idx.max() >= nv:
        errors.append("face index out of range")
    if sum(m["index_count"] for m in r["materials"]) != len(idx):
        errors.append("material index counts do not sum to face indices")
    for m in r["materials"]:
        for k in ("texture", "sphere"):
            if not (-1 <= m[k] < nt):
                errors.append("material %s bad %s" % (m["name"], k))
        if not m["toon_shared"] and not (-1 <= m["toon"] < nt):
            errors.append("material %s bad toon" % m["name"])
    tex_dir = os.path.dirname(path)
    for t in r["textures"]:
        if not os.path.exists(os.path.join(tex_dir, t.replace("\\", "/"))):
            errors.append("missing texture file " + t)
    # vertices
    P = np.array([v[0][:3] for v in r["vertices"]])
    N = np.array([v[0][3:6] for v in r["vertices"]])
    if not np.isfinite(P).all() or not np.isfinite(N).all():
        errors.append("non-finite vertex data")
    nl = np.linalg.norm(N, axis=1)
    if (np.abs(nl - 1) > 1e-3).any():
        warns.append("%d normals not unit length" % (np.abs(nl - 1) > 1e-3).sum())
    for i, v in enumerate(r["vertices"]):
        bones, ws = v[2], v[3]
        if any(not (0 <= b < nb) for b, w in zip(bones, ws) if w > 0):
            errors.append("vertex %d bad bone index" % i)
            break
        if abs(sum(ws) - 1) > 1e-3:
            errors.append("vertex %d weights sum %f" % (i, sum(ws)))
            break
    # bones
    names = [b["name"] for b in r["bones"]]
    if len(set(names)) != len(names):
        errors.append("duplicate bone names")
    for i, b in enumerate(r["bones"]):
        if b["parent"] >= i:
            errors.append("bone %s parent after itself" % b["name"])
        if not (-1 <= b["parent"] < nb):
            errors.append("bone %s bad parent" % b["name"])
        if len(b["name"].encode("cp932", errors="replace")) > 15:
            warns.append("bone name > 15 bytes (VMD): " + b["name"])
        if "append" in b and not (0 <= b["append"][0] < nb):
            errors.append("bone %s bad append parent" % b["name"])
        if "ik" in b:
            ik = b["ik"]
            if not (0 <= ik["target"] < nb):
                errors.append("IK %s bad target" % b["name"])
            for li, lim in ik["links"]:
                if not (0 <= li < nb):
                    errors.append("IK %s bad link" % b["name"])
    for s in STANDARD:
        if s not in names:
            errors.append("missing standard bone " + s)
    # morphs
    mnames = [m["name"] for m in r["morphs"]]
    if len(set(mnames)) != len(mnames):
        errors.append("duplicate morph names")
    for m in r["morphs"]:
        if len(m["name"].encode("cp932", errors="replace")) > 15:
            warns.append("morph name > 15 bytes (VMD): " + m["name"])
        for o in m["offsets"]:
            k = m["kind"]
            if k == 1 and not (0 <= o[0] < nv):
                errors.append("morph %s bad vertex" % m["name"])
                break
            if k == 0 and not (0 <= o[0] < nmo):
                errors.append("morph %s bad group ref" % m["name"])
                break
            if k == 8 and not (-1 <= o[0] < nm):
                errors.append("morph %s bad material" % m["name"])
                break
    # frames
    for f in r["frames"]:
        for k, i in f["items"]:
            if (k == 0 and not (0 <= i < nb)) or (k == 1 and not (0 <= i < nmo)):
                errors.append("frame %s bad item" % f["name"])
    if not r["frames"] or r["frames"][0]["name"] != "Root" or r["frames"][1]["name"] != "表情":
        errors.append("first frames must be Root / 表情")
    # physics
    for rb in r["rigid_bodies"]:
        if not (-1 <= rb["bone"] < nb):
            errors.append("rigid body %s bad bone" % rb["name"])
        if not np.isfinite(rb["pos"] + rb["rot"] + rb["size"]).all():
            errors.append("rigid body %s non-finite" % rb["name"])
    for j in r["joints"]:
        if not (0 <= j["a"] < nr and 0 <= j["b"] < nr):
            errors.append("joint %s bad rigid body index" % j["name"])
    print("PMX %.1f  vertices %d  faces %d  materials %d  textures %d  bones %d  morphs %d  frames %d  rigid %d  joints %d"
          % (r["version"], nv, len(idx) // 3, nm, nt, nb, nmo, len(r["frames"]), nr, len(r["joints"])))
    for w in warns:
        print("WARN", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else "FAILED")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

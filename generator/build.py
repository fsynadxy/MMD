"""Build the character model and write PMX 2.0 + textures.

    python -m generator.build [output_dir]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from . import pmx, textures
from .builder import ModelBuilder, w_single
from .skeleton import build_standard_skeleton

MODEL_NAME = "橙花"
MODEL_NAME_EN = "Touka"
FILE_NAME = "Touka.pmx"
COMMENT = ("橙花 (Touka) — 参考三视图を元に手続き的に生成した PMX 2.0 モデル。\r\n"
           "ボーン: MMD 標準 / 準標準ボーン名 (全ての親, センター, グルーブ, 腰, 上半身2, 腕捩, 手捩, 足D, 足IK親 等)。\r\n"
           "表情モーフ, 物理演算 (髪・スカート・リボン) 付き。")
COMMENT_EN = ("Touka - procedurally generated PMX 2.0 model based on the reference sheets.\r\n"
              "Bones use MMD standard / semi-standard Japanese names. Includes facial morphs and physics.")

EDGE_SKIN = (0.45, 0.22, 0.18, 1.0)
EDGE_HAIR = (0.42, 0.14, 0.08, 1.0)
EDGE_CLOTH = (0.30, 0.26, 0.30, 1.0)
EDGE_DARK = (0.08, 0.06, 0.07, 1.0)


def define_materials(mb: ModelBuilder, tex: dict):
    T = lambda k: mb.tex("tex\\" + tex[k])
    M = pmx.Material

    def textured(name, en, t, toon, **kw):
        kw.setdefault("diffuse", (1, 1, 1, 1))
        kw.setdefault("ambient", (0.4, 0.4, 0.4))
        return M(name, en, texture=T(t), toon=T(toon), **kw)

    def plain(name, en, rgb, toon, **kw):
        r, g, b = rgb
        kw.setdefault("ambient", (0.4 * r, 0.4 * g, 0.4 * b))
        return M(name, en, diffuse=(r, g, b, 1.0), toon=T(toon), **kw)

    face_kw = dict(edge_color=EDGE_SKIN, edge_size=0.7)
    decal = dict(edge=False, ground_shadow=False, cast_shadow=False, receive_shadow=False)
    mb.material("face", textured("顔", "face", "face", "toon_skin", **face_kw))
    mb.material("eye_white", textured("白目", "eye white", "eye_white", "toon_flat", **decal))
    mb.material("iris", textured("瞳", "iris", "iris", "toon_flat", **decal))
    mb.material("highlight", textured("ハイライト", "highlight", "highlight", "toon_flat", **decal))
    mb.material("mouth_in", textured("口内", "mouth", "mouth", "toon_flat", **decal))
    mb.material("face_mask", textured("まぶた", "eyelid", "face", "toon_skin", edge=False, ground_shadow=False))
    mb.material("lines", textured("目線眉", "lines", "lines", "toon_flat", **decal))
    mb.material("blush", textured("頬", "blush", "blush", "toon_flat", diffuse=(1, 1, 1, 0.0), **decal))
    mb.material("skin", plain("肌", "skin", (1.0, 0.945, 0.918), "toon_skin", edge_color=EDGE_SKIN, edge_size=0.6))
    white = (0.975, 0.965, 0.955)
    mb.material("cloth_white", plain("ブラウス", "blouse", white, "toon_cloth", edge_color=EDGE_CLOTH, edge_size=0.6))
    mb.material("cloth_white_ds", plain("フリル", "frills", white, "toon_cloth", double_sided=True,
                                        edge_color=EDGE_CLOTH, edge_size=0.4))
    mb.material("cloth_black", plain("黒布", "black cloth", (0.17, 0.14, 0.16), "toon_dark", edge_color=EDGE_DARK,
                                     edge_size=0.6))
    mb.material("cloth_black_satin", plain("リボン黒", "black ribbon", (0.16, 0.13, 0.15), "toon_dark",
                                           specular=(0.25, 0.25, 0.28), spec_power=12.0, sphere=T("satin_s"),
                                           sphere_mode=2, edge_color=EDGE_DARK, edge_size=0.5))
    mb.material("gold", plain("金具", "gold", (0.86, 0.66, 0.30), "toon_dark", specular=(0.6, 0.5, 0.3),
                              spec_power=20.0, sphere=T("satin_s"), sphere_mode=2, edge_color=(0.35, 0.22, 0.08, 1),
                              edge_size=0.3))
    mb.material("sock", textured("靴下", "socks", "sock", "toon_cloth", edge_color=EDGE_CLOTH, edge_size=0.5))
    mb.material("shoe", plain("靴", "shoes", (0.12, 0.10, 0.11), "toon_dark", specular=(0.35, 0.35, 0.38),
                              spec_power=18.0, sphere=T("satin_s"), sphere_mode=2, edge_color=EDGE_DARK, edge_size=0.5))
    mb.material("skirt_white", plain("スカート白", "skirt white", white, "toon_cloth", double_sided=True,
                                     edge_color=EDGE_CLOTH, edge_size=0.5))
    mb.material("skirt_black", plain("スカート黒", "skirt black", (0.16, 0.13, 0.15), "toon_dark",
                                     sphere=T("satin_s"), sphere_mode=2, edge_color=EDGE_DARK, edge_size=0.5))
    mb.material("skirt_white2", plain("スカート白2", "skirt drape", white, "toon_cloth", edge_color=EDGE_CLOTH,
                                      edge_size=0.5))
    orange = (0.94, 0.52, 0.25)
    mb.material("skirt_orange", plain("スカート裏地", "skirt lining", orange, "toon_hair", edge=False))
    mb.material("skirt_orange_out", plain("スカート橙", "skirt orange", orange, "toon_hair",
                                          edge_color=(0.45, 0.18, 0.08, 1), edge_size=0.5))
    mb.material("hair", textured("髪", "hair", "hair", "toon_hair", sphere=T("hair_s"), sphere_mode=2,
                                 edge_color=EDGE_HAIR, edge_size=0.7))
    mb.material("flower", textured("花", "flower", "flower", "toon_hair", double_sided=True,
                                   edge_color=(0.45, 0.18, 0.08, 1), edge_size=0.3))
    mb.material("flower_center", textured("花芯", "flower centre", "flower_c", "toon_hair",
                                          edge_color=(0.40, 0.20, 0.10, 1), edge_size=0.3))


def _mat_morph(mb, name, en, key, op, diffuse, panel=4):
    one4, one3 = (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
    zero4, zero3 = (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    m = mb.morph(name, en, panel, 8)
    if op == 0:
        m.material.append((key, (0, diffuse, one3, 1.0, one3, one4, 1.0, one4, one4, one4)))
    else:
        m.material.append((key, (1, diffuse, zero3, 0.0, zero3, zero4, 0.0, zero4, zero4, zero4)))


def _group(mb, name, en, items, panel=4):
    m = mb.morph(name, en, panel, 0)
    m.group = list(items)


def define_morph_groups(mb):
    _mat_morph(mb, "照れ", "blush", "blush", 1, (0.0, 0.0, 0.0, 1.0))
    _mat_morph(mb, "ハイライト消し", "no highlight", "highlight", 0, (1.0, 1.0, 1.0, 0.0), panel=2)
    _group(mb, "微笑み", "smile (preset)", [("笑い", 0.35), ("にこり", 0.6), ("口角上げ", 0.7)])
    _group(mb, "恥ずかしい", "shy (preset)", [("まばたき", 1.0), ("困る", 0.55), ("照れ", 1.0), ("口角上げ", 0.25)])
    _group(mb, "怒る", "angry (preset)", [("怒り", 1.0), ("怒り目", 0.9), ("口角下げ", 0.6), ("∧", 0.4)])
    _group(mb, "眠い", "sleepy (preset)", [("なごみ", 0.85), ("下", 0.35), ("困る", 0.2)])
    _group(mb, "驚き", "surprised (preset)", [("びっくり", 1.0), ("上", 0.9), ("お", 0.35)])


def define_frames(mb, model):
    from .pmx import DisplayFrame
    sk = mb.sk
    B = lambda names: [(0, sk[n]) for n in names if n in sk.index]
    frames = [DisplayFrame("Root", "Root", True, B(["全ての親"]))]
    order = {1: 0, 2: 1, 3: 2, 4: 3}
    morphs = sorted(range(len(model.morphs)), key=lambda i: (order.get(model.morphs[i].panel, 9), i))
    frames.append(DisplayFrame("表情", "Exp", True, [(1, i) for i in morphs]))
    frames.append(DisplayFrame("センター", "center", False, B(["センター", "グルーブ"])))
    frames.append(DisplayFrame("ＩＫ", "IK", False, B(["左足IK親", "左足ＩＫ", "左つま先ＩＫ", "右足IK親", "右足ＩＫ", "右つま先ＩＫ"])))
    frames.append(DisplayFrame("体(上)", "upper body", False, B(["腰", "上半身", "上半身2", "首", "頭", "両目", "左目", "右目"])))
    arm = []
    for lr in ("左", "右"):
        arm += [lr + n for n in ("肩P", "肩", "腕", "腕捩", "ひじ", "手捩", "手首")]
    frames.append(DisplayFrame("腕", "arms", False, B(arm)))
    fing = []
    for lr in ("左", "右"):
        fing += [lr + "親指" + z for z in "０１２"]
        for f in ("人指", "中指", "薬指", "小指"):
            fing += [lr + f + z for z in "１２３"]
    frames.append(DisplayFrame("指", "fingers", False, B(fing)))
    frames.append(DisplayFrame("体(下)", "lower body", False, B(["下半身"])))
    legs = []
    for lr in ("左", "右"):
        legs += [lr + n for n in ("足", "ひざ", "足首")]
    for lr in ("左", "右"):
        legs += [lr + n for n in ("足D", "ひざD", "足首D", "足先EX")]
    frames.append(DisplayFrame("足", "legs", False, B(legs)))
    hair = [b.name for b in sk.bones if "髪" in b.name]
    frames.append(DisplayFrame("髪", "hair", False, B(hair)))
    skirt = [b.name for b in sk.bones if b.name.startswith("スカート")]
    frames.append(DisplayFrame("スカート", "skirt", False, B(skirt)))
    acc = [b.name for b in sk.bones if b.name.startswith("リボン")]
    frames.append(DisplayFrame("リボン", "ribbon", False, B(acc)))
    return frames


def build(out_dir: str, verbose=True):
    from .accessories import AccessoryBuilder
    from .body import BodyBuilder
    from .face import FaceBuilder
    from .hair import HairBuilder
    from .head import build_head_mesh
    from .physics import Physics, body_colliders
    from .skirt import SkirtBuilder
    from .builder import w_chain

    t0 = time.time()
    tex_dir = os.path.join(out_dir, "tex")
    tex = textures.build_all(tex_dir)
    sk = build_standard_skeleton()
    mb = ModelBuilder(sk)
    define_materials(mb, tex)

    # ---- bones that depend on geometry (hair / skirt / ribbon chains) ----
    hb = HairBuilder(mb)
    hb.add_bones()
    sb = SkirtBuilder(mb)
    sb.add_bones()
    ab = AccessoryBuilder(mb)
    ab.add_bones()

    # ---- head & face ----
    head = build_head_mesh()
    head.uv = np.c_[0.5 + head.pos[:, 0] / 2.4, (19.7 - head.pos[:, 1]) / 3.4]
    head.W = w_chain(head.pos[:, 1], [(16.0, sk["首"]), (16.75, sk["首"]), (17.05, sk["頭"])], mb.nb)
    mb.add("face", head)
    fb = FaceBuilder(mb)
    fb.eyes()
    fb.brows()
    fb.mouth()
    fb.blush()

    # ---- body & clothes ----
    bb = BodyBuilder(mb)
    bb.torso()
    bb.corset()
    bb.collar()
    bb.jabot()
    bb.ascot()
    bb.sleeves()
    bb.hands()
    bb.legs()
    bb.shoes()
    sb.build()

    # ---- hair & accessories ----
    hb.cap()
    hb.bangs()
    hb.side_locks()
    hb.back()
    ab.build()

    # ---- morph presets ----
    define_morph_groups(mb)

    # ---- physics ----
    ph = Physics(mb)
    body_colliders(ph, sk)
    hb.physics(ph)
    sb.physics(ph)
    ab.physics(ph)

    model = mb.build(MODEL_NAME, MODEL_NAME_EN, COMMENT, COMMENT_EN)
    model.frames = define_frames(mb, model)
    path = os.path.join(out_dir, FILE_NAME)
    pmx.write_pmx(model, path)
    if verbose:
        print("wrote %s: %d vertices, %d faces, %d materials, %d bones, %d morphs, %d rigid bodies, %d joints (%.1fs)"
              % (path, len(model.positions), len(model.faces), len(model.materials), len(model.bones),
                 len(model.morphs), len(model.rigid_bodies), len(model.joints), time.time() - t0))
    return path, model


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "model", "Touka")
    build(os.path.abspath(out))

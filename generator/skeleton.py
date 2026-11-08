"""MMD standard + semi-standard skeleton (準標準ボーン)."""
from __future__ import annotations

import math

import numpy as np

from .geom import normalize
from .pmx import Bone

# ---------------------------------------------------------------------------
# key proportions (MMD units, 1 unit ~= 7.9 cm, height ~155 cm)
# ---------------------------------------------------------------------------
HEAD_TOP = 19.6
CHIN_Y = 16.6
EYE_Y = 17.75
IRIS_C = (0.468, 17.745)
ARM_ANGLE = math.radians(37.0)      # A-pose angle below horizontal
SHOULDER = np.array([1.20, 15.55, 0.15])
UPPER_ARM = 3.0
FORE_ARM = 2.6
ARM_DIR = np.array([math.cos(ARM_ANGLE), -math.sin(ARM_ANGLE), 0.0])
ELBOW = SHOULDER + UPPER_ARM * ARM_DIR
WRIST = ELBOW + FORE_ARM * ARM_DIR
HAND_X = ARM_DIR                                   # along the fingers
HAND_Y = np.array([math.sin(ARM_ANGLE), math.cos(ARM_ANGLE), 0.0])  # back of hand
HAND_Z = np.cross(HAND_X, HAND_Y)                  # towards the back (+z)

LEG = np.array([0.62, 11.55, 0.02])
KNEE = np.array([0.60, 6.10, -0.08])
ANKLE = np.array([0.56, 1.00, 0.18])
TOE = np.array([0.60, 0.05, -1.05])
TOE_EX = np.array([0.59, 0.28, -0.60])

# finger layout in hand-local coordinates (x along hand, y back, z pinky side)
# name, knuckle (x, y, z), segment lengths
FINGERS = [
    ("人指", (0.84, 0.0, -0.235), (0.36, 0.24, 0.19)),
    ("中指", (0.87, 0.01, -0.075), (0.40, 0.27, 0.20)),
    ("薬指", (0.84, 0.0, 0.085), (0.36, 0.25, 0.19)),
    ("小指", (0.77, -0.02, 0.235), (0.29, 0.20, 0.16)),
]
FINGER_EN = {"人指": "index", "中指": "middle", "薬指": "ring", "小指": "little"}
FINGER_CURL = math.radians(7.0)       # relaxed curl per joint
THUMB_BASE = (0.12, -0.06, -0.17)
THUMB_LENGTHS = (0.30, 0.26, 0.22)
THUMB_DIR = (0.62, -0.22, -0.75)      # hand local

ZEN = "０１２３"


def hand_local_to_world(p, side=1):
    """Hand-local point -> world for side=+1 (left) / -1 (right)."""
    p = np.asarray(p, dtype=np.float64)
    w = WRIST + p[..., 0:1] * HAND_X + p[..., 1:2] * HAND_Y + p[..., 2:3] * HAND_Z
    if side < 0:
        w = w * np.array([-1, 1, 1])
    return w


def finger_chain_local(knuckle, lengths, curl=FINGER_CURL, base_dir=(1, 0, 0)):
    """Joint positions (len+1) of a finger in hand-local coordinates."""
    pts = [np.array(knuckle, dtype=np.float64)]
    d = normalize(np.array(base_dir, dtype=np.float64))
    for L in lengths:
        # curl about the local z axis (bends toward -y = palm side)
        c, s = math.cos(-curl), math.sin(-curl)
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
        pts.append(pts[-1] + d * L)
    return np.array(pts)


def thumb_chain_local():
    d = normalize(np.array(THUMB_DIR))
    pts = [np.array(THUMB_BASE, dtype=np.float64)]
    for L in THUMB_LENGTHS:
        pts.append(pts[-1] + d * L)
        # slight bend toward the fingers
        d = normalize(d + np.array([0.12, -0.05, 0.12]))
    return np.array(pts)


class Skeleton:
    def __init__(self):
        self.bones: list[Bone] = []
        self.index: dict[str, int] = {}

    # -- helpers -------------------------------------------------------
    def add(self, name, name_en, pos, parent=None, **kw) -> int:
        if name in self.index:
            raise ValueError("duplicate bone " + name)
        b = Bone(name=name, name_en=name_en, pos=tuple(float(x) for x in pos),
                 parent=self.index[parent] if isinstance(parent, str) else (-1 if parent is None else parent),
                 **kw)
        self.bones.append(b)
        self.index[name] = len(self.bones) - 1
        return self.index[name]

    def __getitem__(self, name) -> int:
        return self.index[name]

    def pos(self, name) -> np.ndarray:
        return np.array(self.bones[self.index[name]].pos)

    def connect(self, name, child):
        self.bones[self.index[name]].tail = self.index[child]

    def __len__(self):
        return len(self.bones)


def _local_axis(x_dir, side):
    x = normalize(np.asarray(x_dir, dtype=np.float64))
    if side < 0:
        x = -x
    z = np.array([0.0, 0.0, 1.0 if side > 0 else -1.0])
    z = normalize(z - x * np.dot(z, x))
    return (tuple(x), tuple(z))


def build_standard_skeleton() -> Skeleton:
    sk = Skeleton()
    A = sk.add
    A("全ての親", "master", (0, 0, 0), None, movable=True, tail=(0, 1.0, 0))
    A("センター", "center", (0, 9.0, 0), "全ての親", movable=True, tail=(0, -1.0, 0))
    A("グルーブ", "groove", (0, 9.4, 0), "センター", movable=True, tail=(0, -1.0, 0))
    A("腰", "waist", (0, 12.1, 0.30), "グルーブ", tail=(0, -0.8, 0))
    A("上半身", "upper body", (0, 12.6, 0.05), "腰")
    A("上半身2", "upper body2", (0, 13.9, 0.08), "上半身")
    A("首", "neck", (0, 15.95, 0.12), "上半身2")
    A("頭", "head", (0, 16.72, 0.08), "首", tail=(0, 1.3, 0))
    sk.connect("上半身", "上半身2")
    sk.connect("上半身2", "首")
    sk.connect("首", "頭")
    A("両目", "eyes", (0, 18.75, -1.45), "頭", tail=(0, 0, -0.45))
    for side, lr, en in ((1, "左", "_L"), (-1, "右", "_R")):
        A(lr + "目", "eye" + en, (IRIS_C[0] * side, IRIS_C[1], -0.45), "頭", tail=(0, 0, -0.4),
          append_rotate=True, append_parent=sk["両目"], append_ratio=1.0)

    # arms -------------------------------------------------------------------
    for side, lr, en in ((1, "左", "_L"), (-1, "右", "_R")):
        S = np.array([side, 1, 1])
        sh = np.array([0.28, 15.82, 0.13]) * S
        arm = SHOULDER * S
        elbow = ELBOW * S
        wrist = WRIST * S
        A(lr + "肩P", "shoulderP" + en, sh, "上半身2", tail=(0, 0, 0), visible=False)
        A(lr + "肩", "shoulder" + en, sh, lr + "肩P", local_axis=_local_axis(arm - sh, side))
        A(lr + "肩C", "shoulderC" + en, arm, lr + "肩", visible=False, operable=False,
          append_rotate=True, append_parent=sk[lr + "肩P"], append_ratio=-1.0)
        A(lr + "腕", "arm" + en, arm, lr + "肩C", local_axis=_local_axis(elbow - arm, side))
        sk.bones[sk[lr + "肩"]].tail = sk[lr + "腕"]
        twist = arm + (elbow - arm) * 0.6
        axis = tuple(normalize(elbow - arm))
        A(lr + "腕捩", "arm twist" + en, twist, lr + "腕", fixed_axis=axis,
          local_axis=_local_axis(elbow - arm, side))
        for k, r in ((1, 0.25), (2, 0.5), (3, 0.75)):
            A(lr + "腕捩%d" % k, "arm twist%d%s" % (k, en), arm + (elbow - arm) * r, lr + "腕",
              visible=False, operable=False, append_rotate=True,
              append_parent=sk[lr + "腕捩"], append_ratio=r)
        A(lr + "ひじ", "elbow" + en, elbow, lr + "腕捩", local_axis=_local_axis(wrist - elbow, side))
        sk.connect(lr + "腕", lr + "ひじ")
        sk.bones[sk[lr + "腕捩"]].tail = tuple(normalize(elbow - arm) * 0.5)
        htw = elbow + (wrist - elbow) * 0.6
        A(lr + "手捩", "wrist twist" + en, htw, lr + "ひじ", fixed_axis=tuple(normalize(wrist - elbow)),
          local_axis=_local_axis(wrist - elbow, side))
        for k, r in ((1, 0.25), (2, 0.5), (3, 0.75)):
            A(lr + "手捩%d" % k, "wrist twist%d%s" % (k, en), elbow + (wrist - elbow) * r, lr + "ひじ",
              visible=False, operable=False, append_rotate=True,
              append_parent=sk[lr + "手捩"], append_ratio=r)
        A(lr + "手首", "wrist" + en, wrist, lr + "手捩", local_axis=_local_axis(ARM_DIR * S, side))
        sk.connect(lr + "ひじ", lr + "手首")
        sk.bones[sk[lr + "手捩"]].tail = tuple(normalize(wrist - elbow) * 0.5)

        # thumb
        tc = hand_local_to_world(thumb_chain_local(), side)
        names = [lr + "親指" + ZEN[i] for i in range(3)]
        parent = lr + "手首"
        for i, nm in enumerate(names):
            A(nm, "thumb%d%s" % (i, en), tc[i], parent, local_axis=_local_axis(tc[i + 1] - tc[i], side))
            parent = nm
        for i in range(2):
            sk.connect(names[i], names[i + 1])
        sk.bones[sk[names[2]]].tail = tuple(tc[3] - tc[2])
        # fingers
        first_finger = None
        for fname, knuckle, lengths in FINGERS:
            fc = hand_local_to_world(finger_chain_local(knuckle, lengths), side)
            names = [lr + fname + ZEN[i] for i in (1, 2, 3)]
            parent = lr + "手首"
            for i, nm in enumerate(names):
                A(nm, "%s%d%s" % (FINGER_EN[fname], i + 1, en), fc[i], parent,
                  local_axis=_local_axis(fc[i + 1] - fc[i], side))
                parent = nm
            for i in range(2):
                sk.connect(names[i], names[i + 1])
            sk.bones[sk[names[2]]].tail = tuple(fc[3] - fc[2])
            if first_finger is None:
                first_finger = names[0]
        sk.bones[sk[lr + "手首"]].tail = tuple(ARM_DIR * S * 0.8)

    # lower body -------------------------------------------------------------
    A("下半身", "lower body", (0, 12.6, 0.05), "腰", tail=(0, -1.4, 0))
    for side, lr, en in ((1, "左", "_L"), (-1, "右", "_R")):
        S = np.array([side, 1, 1])
        A("腰キャンセル" + lr, "waist cancel" + en, LEG * S, "下半身", visible=False, operable=False,
          append_rotate=True, append_parent=sk["腰"], append_ratio=-1.0)
        A(lr + "足", "leg" + en, LEG * S, "腰キャンセル" + lr)
        A(lr + "ひざ", "knee" + en, KNEE * S, lr + "足")
        A(lr + "足首", "ankle" + en, ANKLE * S, lr + "ひざ")
        A(lr + "つま先", "toe" + en, TOE * S, lr + "足首", tail=(0, 0, 0), visible=False, operable=False)
        sk.connect(lr + "足", lr + "ひざ")
        sk.connect(lr + "ひざ", lr + "足首")
        sk.connect(lr + "足首", lr + "つま先")
    for side, lr, en in ((1, "左", "_L"), (-1, "右", "_R")):
        S = np.array([side, 1, 1])
        A(lr + "足IK親", "leg IKP" + en, np.array([ANKLE[0], 0, ANKLE[2]]) * S, "全ての親",
          movable=True, tail=(0, 0, 1.0))
        A(lr + "足ＩＫ", "leg IK" + en, ANKLE * S, lr + "足IK親", movable=True, tail=(0, 0, 1.0),
          ik_target=sk[lr + "足首"], ik_loop=40, ik_limit=2.0,
          ik_links=[(sk[lr + "ひざ"], ((-math.pi, 0.0, 0.0), (math.radians(-0.5), 0.0, 0.0))),
                    (sk[lr + "足"], None)])
        A(lr + "つま先ＩＫ", "toe IK" + en, TOE * S, lr + "足ＩＫ", movable=True, tail=(0, -1.0, 0),
          ik_target=sk[lr + "つま先"], ik_loop=3, ik_limit=4.0,
          ik_links=[(sk[lr + "足首"], None)])
    for side, lr, en in ((1, "左", "_L"), (-1, "右", "_R")):
        S = np.array([side, 1, 1])
        A(lr + "足D", "leg D" + en, LEG * S, "腰キャンセル" + lr, layer=1,
          append_rotate=True, append_parent=sk[lr + "足"], append_ratio=1.0)
        A(lr + "ひざD", "knee D" + en, KNEE * S, lr + "足D", layer=1,
          append_rotate=True, append_parent=sk[lr + "ひざ"], append_ratio=1.0)
        A(lr + "足首D", "ankle D" + en, ANKLE * S, lr + "ひざD", layer=1,
          append_rotate=True, append_parent=sk[lr + "足首"], append_ratio=1.0)
        A(lr + "足先EX", "toe EX" + en, TOE_EX * S, lr + "足首D", layer=1,
          tail=tuple((TOE - TOE_EX) * S * 0.8))
        sk.connect(lr + "足D", lr + "ひざD")
        sk.connect(lr + "ひざD", lr + "足首D")
        sk.connect(lr + "足首D", lr + "足先EX")
    return sk

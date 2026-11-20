"""Load one shot of the dataset as numpy arrays.

    from tools.load import load_shot, list_shots
    s = load_shot("data/d405/03")
    s["real_rgb"], s["sim_rgb"]            # (H, W, 3) uint8, RGB
    s["raw_m"], s["gt_m"], s["cdm_m"]      # (H, W) float32 metres, 0 = invalid (None if the file is absent)
    s["sdk_m"]                             # ZED Mini only: sim stereo pair -> ZED Isaac Sim extension -> ZED SDK depth
    s["K"]                                 # 3x3 intrinsics shared by all of the above
    s["meta"]                              # meta.json
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAMERAS = ("d435", "zed_mini", "d405")


def _depth(d, png=None, npy=None):
    if npy and (d / npy).exists():
        z = np.load(d / npy).astype(np.float32)
    elif png and (d / png).exists():
        z = cv2.imread(str(d / png), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    else:
        return None
    z[~np.isfinite(z) | (z < 0)] = 0
    return z


def _rgb(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None
    return None if im is None else im[:, :, ::-1].copy()


def load_shot(folder):
    d = Path(folder)
    if not d.is_absolute():
        d = ROOT / d
    meta = json.loads((d / "meta.json").read_text())
    return {"folder": d, "camera": meta["camera"], "n": meta["shot"], "meta": meta, "K": np.array(meta["K"], np.float64),
            "real_rgb": _rgb(d / "real_rgb.png"), "sim_rgb": _rgb(d / "sim_rgb.png"),
            "raw_m": _depth(d, png="real_depth_mm.png"),
            "gt_m": _depth(d, png="sim_depth_mm.png", npy="sim_depth_m.npy"),
            "cdm_m": _depth(d, png="cdm_depth_mm.png", npy="cdm_depth_m.npy"),
            "sdk_m": _depth(d, png="sim_sdk_depth_mm.png")}


def list_shots(camera):
    return sorted(p.parent for p in (ROOT / "data" / camera).glob("*/meta.json"))


if __name__ == "__main__":
    for cam in CAMERAS:
        for f in list_shots(cam):
            s = load_shot(f)
            ok = (s["raw_m"] > 0) & (s["gt_m"] > 0)
            e = (s["raw_m"] - s["gt_m"])[ok] * 1000
            print(f"{cam:9s} {s['n']:02d}  {s['raw_m'].shape[1]}x{s['raw_m'].shape[0]}  raw valid {np.mean(s['raw_m'] > 0) * 100:5.1f}%  "
                  f"raw - gt (no alignment): median {np.median(e):+7.1f} mm")

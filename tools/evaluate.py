"""Reproduce the error statistics in results/: align the real camera to the simulated one, then compare each
depth estimate with the simulation ground truth.

    python tools/evaluate.py d405                 # raw + CDM (+ sim_extension for zed_mini)
    python tools/evaluate.py d405 --mono mono/    # also monocular predictions saved as mono/<model>/<NN>.npy (metres)

Method
  * Ground-truth scene: fit the table plane in the simulated depth (RANSAC), table = within 4 mm of the plane,
    objects = 15-350 mm above it, both restricted to ground-truth depth <= near_m.
  * Pose: one rigid transform per camera (shared by its 10 shots) that maps the real raw-depth point cloud onto the
    ground-truth surface (projective point-to-plane ICP). The same transform is applied to CDM and monocular depth.
    Because the pose is estimated from raw depth, a constant bias of raw depth is absorbed by it.
  * "+scale": a monocular prediction multiplied by median(raw / prediction) over pixels with raw < 1.5 m.
  * fly_pct ("tail points"): around each object (+-12 px), points >= 15 mm above the table whose nearest
    ground-truth surface point is more than 20 mm away.
  * lean: for objects >= 90 mm tall, how much the side face leans along the viewing direction, minus the same
    quantity in the ground truth.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from load import ROOT, list_shots, load_shot

OBJ_MIN_H, OBJ_MAX_H = 0.015, 0.35
# near_m: ground-truth depth range that holds the table and objects. plane_range: depth range used to find the table plane.
CFG = {"d435": dict(near_m=1.2, plane_range=2.0), "zed_mini": dict(near_m=1.2, plane_range=1.2), "d405": dict(near_m=0.6, plane_range=0.6)}
SDK = "sim_extension"


def backproject(z, K):
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)


def project(P, K):
    return P[:, 0] / P[:, 2] * K[0, 0] + K[0, 2], P[:, 1] / P[:, 2] * K[1, 1] + K[1, 2]


def erode(m, r):
    return cv2.erode(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)


def dilate(m, r):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)


def plane_lsq(Q):
    """Least-squares plane with the normal facing the camera: n.p + d = height above the table."""
    c = Q.mean(0)
    n = np.linalg.eigh(np.cov((Q - c).T))[1][:, 0]
    d = -n @ c
    return (-n, -d) if d < 0 else (n, d)


def plane_ransac(P, iters=300, thr=0.004, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(P @ n - n @ a) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    return plane_lsq(P[best])


def plane_axes(n):
    f = np.array([0, 0, 1.0])
    ex = f - (f @ n) * n
    ex /= np.linalg.norm(ex)
    return ex, np.cross(ex, n)


def gt_scene(s, cfg):
    K, gt = s["K"], s["gt_m"]
    excl = s.get("excl", np.zeros(gt.shape, bool))
    P = backproject(gt, K)
    n, d = plane_ransac(P[(gt > 0) & (gt < cfg["plane_range"]) & ~excl])
    h = P @ n + d
    region = (gt > 0) & (gt <= cfg["near_m"]) & ~excl
    table = region & (np.abs(h) < 0.004)
    ex, ey = plane_axes(n)
    jump = np.zeros(gt.shape, bool)
    jump[:, 1:] |= np.abs(np.diff(gt, axis=1)) > 0.03
    jump[1:, :] |= np.abs(np.diff(gt, axis=0)) > 0.03
    return dict(P=P, n=n, d=d, h=h, region=region, table=table, objects=region & (h > OBJ_MIN_H) & (h < OBJ_MAX_H), jump=jump, ex=ex, ey=ey)


def components(mask, min_px=150):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [lab == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]


def _normals(Pimg):
    n = np.cross(np.roll(Pimg, -1, 1) - np.roll(Pimg, 1, 1), np.roll(Pimg, -1, 0) - np.roll(Pimg, 1, 0))
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def icp(pairs, K, iters=40):
    """pairs = [(real depth, ground-truth depth, region mask), ...] -> one (R, t) for all pairs."""
    data = []
    for src, dst, region in pairs:
        Pd = backproject(dst, K)
        m = region & (src > 0)
        m[::2, :] = False
        data.append((backproject(src, K)[m], Pd, _normals(Pd), dst.shape))
    R, t = np.eye(3), np.zeros(3)
    for it in range(iters):
        A_all, r_all = [], []
        for Ps, Pd, Nd, (H, W) in data:
            P = Ps @ R.T + t
            u, v = (np.round(c).astype(int) for c in project(P, K))
            ok = (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
            u, v, P = u[ok], v[ok], P[ok]
            q, n = Pd[v, u], Nd[v, u]
            r = np.einsum("ij,ij->i", P - q, n)
            good = (q[:, 2] > 0) & (np.abs(r) < (0.03 if it < 10 else 0.012))
            A_all.append(np.hstack([np.cross(P[good], n[good]), n[good]]))
            r_all.append(r[good])
        A, r = np.vstack(A_all), np.concatenate(r_all)
        x = np.linalg.lstsq(A, -r, rcond=None)[0]
        dR = _rodrigues(x[:3])
        R, t = dR @ R, dR @ t + x[3:]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return R, t, ang, float(np.median(np.abs(r)) * 1000)


def warp_depth(z, R, t, K):
    """Lift depth to 3D, move it by (R, t) and render it again with the same camera. Small holes get the neighbour median."""
    H, W = z.shape
    P = backproject(z, K)[z > 0] @ R.T + t
    u, v = (np.round(c).astype(int) for c in project(P, K))
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[:, 2] > 0)
    out = np.full(H * W, np.inf, np.float32)
    np.minimum.at(out, v[ok] * W + u[ok], P[ok, 2].astype(np.float32))
    out = out.reshape(H, W)
    out[~np.isfinite(out)] = 0
    hole = out == 0
    pad = np.pad(out, 1)
    nb = np.stack([pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy])
    cnt = (nb > 0).sum(0)
    nb = np.where(nb > 0, nb, np.nan)
    fill = hole & (cnt >= 5)
    out[fill] = np.nanmedian(nb[:, fill], axis=0)
    return out


def stat(e_mm):
    if e_mm.size == 0:
        return None
    a = np.abs(e_mm)
    return {"bias": round(float(np.median(e_mm)), 1), "abs": round(float(np.median(a)), 1), "p90": round(float(np.percentile(a, 90)), 1), "n": int(e_mm.size)}


def summ(x, extra=()):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    r = {"bias": round(float(np.median(x)), 1), "abs": round(float(np.median(np.abs(x))), 1), "n": int(x.size)}
    if "max" in extra:
        r["max"] = round(float(np.abs(x).max()), 1)
    return r


def lean_deg(P, h, mask, g, top):
    m = mask & (h > 0.2 * top) & (h < 0.85 * top)
    if m.sum() < 60:
        return None
    Q = P[m]
    A = np.stack([h[m], Q @ g["ey"], np.ones(m.sum())], 1)
    return float(np.degrees(np.arctan(np.linalg.lstsq(A, Q @ g["ex"], rcond=None)[0][0])))


def sources(s, mono_dir):
    out = {"raw": s["raw_m"]}
    if s["cdm_m"] is not None:
        out["cdm"] = s["cdm_m"]
    if s["sdk_m"] is not None:
        out[SDK] = s["sdk_m"]
    for d in sorted(p for p in mono_dir.iterdir() if p.is_dir()) if mono_dir else []:
        f = d / f"{s['n']:02d}.npy"
        if f.exists():
            z = np.load(f).astype(np.float32)
            z[~np.isfinite(z) | (z < 0)] = 0
            out[d.name] = z
            ok = (s["raw_m"] > 0) & (z > 0) & (s["raw_m"] < 1.5)
            out[d.name + "+scale"] = z * np.float32(np.median(s["raw_m"][ok] / z[ok]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camera", choices=list(CFG))
    ap.add_argument("--mono", default=None, help="folder with <model>/<NN>.npy monocular predictions (metres, same size and K as the dataset)")
    ap.add_argument("--out", default=None, help="write the statistics to this JSON file")
    a = ap.parse_args()
    cfg, cam = CFG[a.camera], a.camera
    S = [load_shot(f) for f in list_shots(cam)]
    K = S[0]["K"]
    zmax = np.inf
    if cam == "d405":
        # The wrist camera always sees its own gripper fingers at the same pixels (ground truth < 0.2 m in every shot): leave them out.
        fixed = np.all([(s["gt_m"] > 0) & (s["gt_m"] < 0.2) for s in S], axis=0)
        fixed = cv2.dilate(fixed.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        for s in S:
            s["excl"] = fixed
        zmax = cfg["near_m"] + 0.3      # raw D405 depth has a few pixels that jump to metres; they cannot be the table
    G = [gt_scene(s, cfg) for s in S]
    R, t, ang, res = icp([(s["raw_m"], s["gt_m"], g["region"]) for s, g in zip(S, G)], K)
    print(f"pose (real camera -> simulated camera): rotation {ang:.2f} deg, translation {np.round(t * 1000, 1)} mm, ICP residual {res:.1f} mm")
    acc, vs_sdk = {}, {"table": [], "object": []}
    I3, Z3 = np.eye(3), np.zeros(3)
    mono = Path(a.mono) if a.mono else None
    for s, g in zip(S, G):
        gt = s["gt_m"]
        tree = cKDTree(g["P"][g["region"]])
        objs = components(g["objects"])
        edge5, edge20 = dilate(g["jump"], 5), dilate(g["jump"], 20)
        faces = {"table": g["table"], "object": g["region"] & ~g["table"]}
        for name, z in sources(s, mono).items():
            A = acc.setdefault(name, {k: [] for k in ("b_table", "b_object", "a_table", "a_object", "flat", "lean", "top", "acc", "comp", "fly", "hole")})
            for fk, m in faces.items():     # before alignment (edges +-20 px excluded)
                ok = m & ~edge20 & (z > 0)
                A["b_" + fk].append((z - gt)[ok] * 1000)
            A["hole"].append(1 - ((z > 0) & g["region"]).sum() / g["region"].sum())
            tm = erode(g["table"], 20) & (z > 0)     # table flatness, independent of the pose
            if (tm & (z < zmax)).sum() > 0.5 * tm.sum():
                tm &= z < zmax
            Q = backproject(z, K)[tm]
            n_, d_ = plane_lsq(Q)
            r = Q @ n_ + d_
            A["flat"].append(1.4826 * np.median(np.abs(r - np.median(r))) * 1000)
            Rz, tz = (I3, Z3) if name == SDK else (R, t)     # sim_extension is already in the simulated camera frame
            w = z if name == SDK else warp_depth(z, R, t, K)
            if name == "raw" and s["sdk_m"] is not None:
                for fk, m in faces.items():
                    ok = m & ~edge5 & (w > 0) & (s["sdk_m"] > 0)
                    vs_sdk[fk].append((w - s["sdk_m"])[ok] * 1000)
            for fk, m in faces.items():     # after alignment (edges +-5 px excluded)
                ok = m & ~edge5 & (w > 0)
                A["a_" + fk].append((w - gt)[ok] * 1000)
            Pw = backproject(w, K)
            hw = Pw @ g["n"] + g["d"]
            Pall = backproject(z, K)[z > 0] @ Rz.T + tz
            u, v = (np.round(c).astype(int) for c in project(Pall, K))
            inimg = (u >= 0) & (u < gt.shape[1]) & (v >= 0) & (v < gt.shape[0])
            Pall, u, v = Pall[inimg], u[inimg], v[inimg]
            hall = Pall @ g["n"] + g["d"]
            for gm in objs:
                top_gt = float(np.percentile(g["h"][gm], 98))
                core = erode(gm, 4)
                if top_gt >= 0.09:
                    lg, le = lean_deg(g["P"], g["h"], core, g, top_gt), lean_deg(Pw, hw, core & (w > 0), g, top_gt)
                    if lg is not None and le is not None:
                        A["lean"].append(le - lg)
                c2 = erode(gm, 2) & (w > 0)
                if c2.sum() > 30:
                    A["top"].append((float(np.percentile(hw[c2], 98)) - top_gt) * 1000)
                sel = dilate(gm, 12)[v, u] & (hall > OBJ_MIN_H) & (Pall[:, 2] < cfg["near_m"] + 0.3)
                if sel.sum() > 30:
                    d_est = tree.query(Pall[sel])[0] * 1000
                    A["acc"].append(d_est)
                    A["fly"].append(float((d_est > 20).mean()))
                    A["comp"].append(cKDTree(Pall[sel]).query(g["P"][gm])[0] * 1000)
    out = {"_pose": {"R": R.tolist(), "t_m": t.tolist(), "rot_deg": round(ang, 2), "t_mm": np.round(t * 1000, 1).tolist(), "icp_residual_mm": round(res, 2)}}
    for name, A in acc.items():
        cat = lambda k: np.concatenate(A[k]) if A[k] else np.array([])  # noqa: E731
        ac, co = cat("acc"), cat("comp")
        out[name] = {"before": {"table": stat(cat("b_table")), "object": stat(cat("b_object"))},
                     "after": {"table": stat(cat("a_table")), "object": stat(cat("a_object"))},
                     "hole_pct": round(float(np.mean(A["hole"])) * 100, 1), "flat_mm": round(float(np.mean(A["flat"])), 1),
                     "lean": summ(A["lean"], ("max",)), "top": summ(A["top"]),
                     "obj3d": None if ac.size == 0 else {"acc_med": round(float(np.median(ac)), 1), "acc_p90": round(float(np.percentile(ac, 90)), 1),
                                                         "fly_pct": round(float(np.mean(A["fly"])) * 100, 1), "comp_med": round(float(np.median(co)), 1),
                                                         "comp_p90": round(float(np.percentile(co, 90)), 1), "n_obj": len(A["acc"])}}
    if vs_sdk["table"]:
        out["raw-" + SDK] = {k: stat(np.concatenate(v)) for k, v in vs_sdk.items()}
    g = lambda d, k: "    -" if d is None or d.get(k) is None else f"{d[k]:5.1f}"  # noqa: E731
    print(f"{'depth':26s} hole%  table |e| p90   object |e| p90   flat  lean max  fly%")
    for n, v in out.items():
        if "after" in v:
            print(f"{n:26s} {v['hole_pct']:5.1f}  {g(v['after']['table'], 'abs')} {g(v['after']['table'], 'p90')}   {g(v['after']['object'], 'abs')} {g(v['after']['object'], 'p90')}  "
                  f"{v['flat_mm']:5.1f}  {g(v['lean'], 'max')}    {g(v['obj3d'], 'fly_pct')}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()

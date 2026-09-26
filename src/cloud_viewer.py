"""Self-contained interactive viewer for a benchmark point-cloud snapshot.

write_html() turns one bench_clouds/*.npz into a single .html: the predicted
cloud unprojected with the GT cameras (predicted cameras when the snapshot
has none), colourable by RGB, by |pred-gt|/gt, by the model's confidence,
or by whether the pixel was a sparse-depth INPUT; a GT cloud to toggle
against it, placed beside it in the scene (offset along the camera's x
axis by the cloud's width, with its own frustums), or split-screen (GT
left, prediction right, one shared viewpoint); per-frame camera frustums; and a frame slider that either
accumulates frames 0..t (what the stream has seen) or shows frame t alone.
Per-frame metric AbsRel / delta1 are computed here and shown as the slider
moves, so a bad frame can be found by number and then looked at. When the
run's bench_results.json is found (next to bench_clouds/, or passed in),
the panel also shows this sequence's rows under all three protocols and
both TAEs, next to the dataset means at the same density -- quantitative
and qualitative in one page.

Everything is precomputed in numpy and embedded as base64 typed arrays; the
page needs only three.js from a CDN, so it opens through src/serve_glb.py
(any file in its directory is served) or straight from a local copy.
Points are subsampled by `every` on both axes (default 2 -> 1/4 of the
pixels): 32 frames of 518x392 at every=2 is ~1.6M points, ~25 MB of page.

    python cloud_viewer.py <run>/bench_clouds/*.npz          # -> <stem>.html next to each
    python serve_glb.py --glb-dir <run>/bench_clouds          # then open /<stem>.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from streamvggt.utils.geometry import unproject_depth_map_to_point_map

_THREE = "0.160.0"


def _c2w_to_w2c(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(pose)[:, :3, :4].astype(np.float32)


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    S = w2c.shape[0]
    m = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
    m[:, :3, :4] = w2c
    return np.linalg.inv(m)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _frame_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> list[dict]:
    out = []
    for i in range(pred.shape[0]):
        m = valid[i] & (gt[i] > 0) & np.isfinite(pred[i])
        if m.sum() == 0:
            out.append({"abs_rel": None, "delta1": None, "n": 0})
            continue
        r = pred[i][m] / gt[i][m]
        out.append(
            {
                "abs_rel": float(np.mean(np.abs(pred[i][m] - gt[i][m]) / gt[i][m])),
                "delta1": float(np.mean(np.maximum(r, 1.0 / r) < 1.25)),
                "n": int(m.sum()),
            }
        )
    return out


def load_metrics(
    results_path: Path, dataset: str, sequence: str, density: float
) -> dict | None:
    """This sequence's rows and the dataset aggregate at this density from a
    bench_results.json; None when the file does not exist."""
    if not results_path.is_file():
        return None
    with open(results_path) as f:
        r = json.load(f)
    dkey = f"d{density * 100:g}"

    def same(row: dict) -> bool:
        return (
            row["dataset"] == dataset
            and row["sequence"] == sequence
            and abs(row["density"] - density) < 1e-9
        )

    seq_rows = [row for row in r.get("rows", []) if same(row)]
    tae_rows = [row for row in r.get("tae_rows", []) if same(row)]
    mode = seq_rows[0]["mode"] if seq_rows else "stream"
    prefix = f"{dataset}/{mode}/{dkey}/"
    agg = {
        k[len(prefix) :]: v
        for k, v in r.get("aggregate", {}).items()
        if k.startswith(prefix)
    }
    return {
        "sequence": seq_rows[0] if seq_rows else None,
        "tae": tae_rows[0] if tae_rows else None,
        "dataset": agg,
        "source": str(results_path),
    }


def build_payload(
    npz_path: Path, every: int = 2, cameras: str = "auto", results: Path | None = None
) -> dict:
    """Everything the page needs, as JSON-able scalars plus base64 arrays."""
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    with np.load(npz_path) as d:
        rgb = d["rgb"]  # [S,H,W,3] uint8
        pred = d["pred_depth"].astype(np.float32)
        conf = d["pred_conf"].astype(np.float32)
        gt = d["gt_depth"].astype(np.float32)
        gt_valid = d["gt_valid"].astype(bool)
        sparse_mask = d["sparse_mask"].astype(bool)
        has_gt_cam = "K_gt" in d and "pose_gt" in d
        if cameras == "auto":
            cameras = "gt" if has_gt_cam else "pred"
        if cameras == "gt":
            if not has_gt_cam:
                raise ValueError(f"{npz_path.name} has no GT cameras")
            K = d["K_gt"].astype(np.float32)
            c2w = d["pose_gt"].astype(np.float32)
        elif cameras == "pred":
            K = d["K_pred"].astype(np.float32)
            c2w = _w2c_to_c2w(d["w2c_pred"].astype(np.float32))
        else:
            raise ValueError(f"cameras must be auto|gt|pred, got {cameras!r}")
        meta = {k: str(d[k]) for k in ("dataset", "sequence", "mode") if k in d}
        meta["density"] = float(d["density"]) if "density" in d else None
        frames = [str(f) for f in d["frames"]] if "frames" in d else []

    S, H, W = pred.shape
    w2c = _c2w_to_w2c(c2w)
    pred_pts = unproject_depth_map_to_point_map(pred[..., None], w2c, K)  # [S,H,W,3]
    gt_pts = unproject_depth_map_to_point_map(gt[..., None], w2c, K)
    stats = _frame_metrics(pred, gt, gt_valid)

    # per-pixel error as uint8 on [0, ERR_VMAX]; 255 = no GT
    ERR_VMAX = 0.5
    err = np.full((S, H, W), 255, np.uint8)
    ok = gt_valid & (gt > 0) & np.isfinite(pred)
    rel = np.abs(pred - gt) / np.where(gt > 0, gt, 1.0)
    err[ok] = np.clip(rel[ok] / ERR_VMAX * 254.0, 0, 254).astype(np.uint8)
    # confidence to uint8 on a robust range (expp1 head: floor 1, no ceiling)
    c_lo, c_hi = (
        np.percentile(conf[np.isfinite(conf)], [2, 98])
        if np.isfinite(conf).any()
        else (0.0, 1.0)
    )
    conf8 = np.clip((conf - c_lo) / max(c_hi - c_lo, 1e-6) * 255.0, 0, 255).astype(
        np.uint8
    )

    ys = np.arange(0, H, every)
    xs = np.arange(0, W, every)
    sub = np.ix_(ys, xs)
    per_frame = []
    for i in range(S):
        finite = np.isfinite(pred[i][sub]) & (pred[i][sub] > 0)
        p = pred_pts[i][sub][finite].astype(np.float32)
        g_ok = finite & gt_valid[i][sub] & (gt[i][sub] > 0)
        per_frame.append(
            {
                "n": int(finite.sum()),
                "pos": _b64(p),
                "rgb": _b64(rgb[i][sub][finite]),
                "err": _b64(err[i][sub][finite]),
                "conf": _b64(conf8[i][sub][finite]),
                "sparse": _b64(sparse_mask[i][sub][finite].astype(np.uint8)),
                "n_gt": int(g_ok.sum()),
                "gt_pos": _b64(gt_pts[i][sub][g_ok].astype(np.float32)),
                "gt_rgb": _b64(rgb[i][sub][g_ok]),
            }
        )
    span = (
        float(
            np.nanpercentile(
                np.linalg.norm(pred_pts[np.isfinite(pred_pts).all(-1)], axis=-1), 95
            )
        )
        if S
        else 1.0
    )
    if results is None:
        results = npz_path.parent.parent / "bench_results.json"
    metrics = (
        load_metrics(
            Path(results),
            meta.get("dataset", ""),
            meta.get("sequence", ""),
            meta["density"],
        )
        if meta.get("density") is not None
        else None
    )
    return {
        "metrics": metrics,
        "meta": meta,
        "frames": frames,
        "S": S,
        "H": H,
        "W": W,
        "every": every,
        "cameras": cameras,
        "err_vmax": ERR_VMAX,
        "conf_range": [float(c_lo), float(c_hi)],
        "span": span if np.isfinite(span) and span > 0 else 1.0,
        "c2w": c2w.reshape(S, 16).tolist(),
        "K": K.reshape(S, 9).tolist(),
        "stats": stats,
        "per_frame": per_frame,
    }


_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif;overflow:hidden}
 #ui{position:absolute;left:10px;top:10px;background:rgba(0,0,0,.65);padding:10px 12px;border-radius:6px;max-width:360px;line-height:1.5}
 #ui label{margin-right:10px;white-space:nowrap}
 #stats{position:absolute;right:10px;top:10px;background:rgba(0,0,0,.65);padding:10px 12px;border-radius:6px;font-family:ui-monospace,monospace;white-space:pre}
 #legend{position:absolute;left:10px;bottom:10px;background:rgba(0,0,0,.65);padding:6px 10px;border-radius:6px}
 .half{position:absolute;top:10px;transform:translateX(-50%);background:rgba(0,0,0,.65);padding:4px 10px;border-radius:6px;display:none;font-weight:bold}
 #metrics{position:absolute;right:10px;bottom:10px;background:rgba(0,0,0,.65);padding:8px 12px;border-radius:6px;font-family:ui-monospace,monospace;font-size:12px}
 #metrics table{border-collapse:collapse} #metrics td,#metrics th{padding:1px 8px;text-align:right} #metrics th{color:#9cf;font-weight:normal} #metrics td:first-child,#metrics th:first-child{text-align:left}
 input[type=range]{width:220px;vertical-align:middle}
 canvas{display:block}
</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@__THREE__/build/three.module.js","three/addons/":"https://unpkg.com/three@__THREE__/examples/jsm/"}}</script>
</head><body>
<div id="ui">
 <b>__TITLE__</b><br>
 <div id="meta"></div>
 colour:
 <label><input type="radio" name="mode" value="rgb" checked>RGB</label>
 <label><input type="radio" name="mode" value="err">|pred-gt|/gt</label>
 <label><input type="radio" name="mode" value="conf">confidence</label>
 <label><input type="radio" name="mode" value="sparse">sparse input</label><br>
 layout:
 <label><input type="radio" name="layout" value="twin" checked>GT cloud beside prediction</label>
 <label><input type="radio" name="layout" value="overlay">overlay</label>
 <label><input type="radio" name="layout" value="side">split screen</label><br>
 <label><input type="checkbox" id="showPred" checked>prediction</label>
 <label><input type="checkbox" id="showGt">GT cloud</label>
 <label><input type="checkbox" id="showCams" checked>cameras</label>
 <label><input type="checkbox" id="showSparse" checked>sparse-input points</label><br>
 frame <input type="range" id="frame" min="0" max="0" value="0"> <span id="frameLbl"></span>
 <label><input type="checkbox" id="accum" checked>accumulate 0..t</label><br>
 view: <button id="gotoShoulder">over the shoulder</button> <button id="gotoCam">through camera</button> <label><input type="checkbox" id="follow" checked>follow frame</label> <button id="fit">fit all</button><br>
 point size <input type="range" id="psize" min="1" max="30" value="8"> &nbsp; err vmax <span id="vmaxLbl"></span>
 <input type="range" id="vmax" min="1" max="100" value="50"><br>
 <span style="opacity:.7">drag: orbit &middot; wheel: zoom &middot; right-drag: pan &middot; keys: &larr; &rarr; frame, g GT overlay, a accumulate, t GT beside, s split screen, f fit all</span>
</div>
<div id="stats"></div>
<div id="halfL" class="half" style="left:25%">ground truth</div><div id="halfR" class="half" style="left:75%">prediction</div>
<div id="legend"></div>
<div id="metrics"></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const P = JSON.parse(document.getElementById('payload').textContent);
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const scene = new THREE.Scene();
// OpenCV (x right, y down, z forward) -> three.js (y up, z toward viewer)
const root = new THREE.Group(); root.scale.set(1, -1, -1); scene.add(root);
// GT lives in its own group so the "twin" layout can slide it sideways, in
// the scene, next to the prediction (same frames, same cameras)
const gtGroup = new THREE.Group(); root.add(gtGroup);
const renderer = new THREE.WebGLRenderer({antialias: false}); renderer.setPixelRatio(window.devicePixelRatio); renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.01, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
const turbo = t => { t = Math.min(Math.max(t, 0), 1); // Google turbo, polynomial fit
  const r = 0.13572138 + t*(4.61539260 + t*(-42.66032258 + t*(132.13108234 + t*(-152.94239396 + t*59.28637943))));
  const g = 0.09140261 + t*(2.19418839 + t*(4.84296658 + t*(-14.18503333 + t*(4.27729857 + t*2.82956604))));
  const b = 0.10667330 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))));
  return [r, g, b]; };
const frames = [], gtFrames = [], sparseFrames = [], camFrames = [], camFramesGt = [];
const mat = new THREE.PointsMaterial({size: 1, vertexColors: true, sizeAttenuation: false});
const matGt = new THREE.PointsMaterial({size: 1, vertexColors: true, sizeAttenuation: false, opacity: 0.6, transparent: true});
const matSp = new THREE.PointsMaterial({size: 1, color: 0xffff00, sizeAttenuation: false});
P.per_frame.forEach((f, i) => {
  const pos = dec(f.pos, Float32Array), rgb = dec(f.rgb, Uint8Array), err = dec(f.err, Uint8Array), conf = dec(f.conf, Uint8Array), sp = dec(f.sparse, Uint8Array);
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(f.n * 3), 3));
  const pts = new THREE.Points(geo, mat); pts.userData = {rgb, err, conf, sp}; root.add(pts); frames.push(pts);
  const gpos = dec(f.gt_pos, Float32Array), grgb = dec(f.gt_rgb, Uint8Array);
  const ggeo = new THREE.BufferGeometry(); ggeo.setAttribute('position', new THREE.BufferAttribute(gpos, 3));
  const gcol = new Float32Array(f.n_gt * 3); for (let k = 0; k < f.n_gt * 3; k++) gcol[k] = grgb[k] / 255;
  ggeo.setAttribute('color', new THREE.BufferAttribute(gcol, 3)); const gp = new THREE.Points(ggeo, matGt); gp.userData = {rgb: grgb}; gtGroup.add(gp); gtFrames.push(gp);
  let ns = 0; for (let k = 0; k < f.n; k++) ns += sp[k];
  const spos = new Float32Array(ns * 3); let j = 0; for (let k = 0; k < f.n; k++) if (sp[k]) { spos[j++] = pos[3*k]; spos[j++] = pos[3*k+1]; spos[j++] = pos[3*k+2]; }
  const sgeo = new THREE.BufferGeometry(); sgeo.setAttribute('position', new THREE.BufferAttribute(spos, 3)); const spts = new THREE.Points(sgeo, matSp); root.add(spts); sparseFrames.push(spts);
  // frustum from K and c2w (OpenCV): corners at depth d
  const K = P.K[i], M = P.c2w[i], d = P.span * 0.08;
  const cx = K[2], cy = K[5], fx = K[0], fy = K[4], W = P.W, H = P.H;
  const corner = (u, v) => new THREE.Vector3((u - cx) / fx * d, (v - cy) / fy * d, d).applyMatrix4(new THREE.Matrix4().fromArray(M).transpose());
  const o = new THREE.Vector3(M[3], M[7], M[11]);
  const c = [corner(0, 0), corner(W, 0), corner(W, H), corner(0, H)];
  const segs = []; c.forEach((p, k) => { segs.push(o, p, p, c[(k + 1) % 4]); });
  const lg = new THREE.BufferGeometry().setFromPoints(segs); const ln = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({color: 0x44aaff})); root.add(ln); camFrames.push(ln);
  const lnGt = ln.clone(); lnGt.visible = false; gtGroup.add(lnGt); camFramesGt.push(lnGt);
});
const $ = id => document.getElementById(id);
const frameEl = $('frame'); frameEl.max = P.S - 1; frameEl.value = P.S - 1;
$('meta').textContent = `${P.meta.dataset || ''} / ${P.meta.sequence || ''}  density ${P.meta.density != null ? (P.meta.density * 100).toFixed(0) + '%' : '?'}  cameras: ${P.cameras}  ${P.S} frames @ ${P.W}x${P.H} (every ${P.every}px)`;
let vmax = P.err_vmax;
function recolor() {
  const mode = document.querySelector('input[name=mode]:checked').value;
  frames.forEach(pts => {
    const col = pts.geometry.getAttribute('color').array, u = pts.userData, n = col.length / 3;
    for (let k = 0; k < n; k++) {
      let c;
      if (mode === 'rgb') c = [u.rgb[3*k] / 255, u.rgb[3*k+1] / 255, u.rgb[3*k+2] / 255];
      else if (mode === 'err') c = u.err[k] === 255 ? [0.35, 0.35, 0.35] : turbo(u.err[k] / 254 * P.err_vmax / vmax);
      else if (mode === 'conf') c = turbo(u.conf[k] / 255);
      else c = u.sp[k] ? [1, 1, 0] : [0.3, 0.3, 0.3];
      col[3*k] = c[0]; col[3*k+1] = c[1]; col[3*k+2] = c[2];
    }
    pts.geometry.getAttribute('color').needsUpdate = true;
  });
  const dim = layout() === 'overlay';
  gtFrames.forEach(gp => {
    const col = gp.geometry.getAttribute('color').array, rgb = gp.userData.rgb;
    for (let k = 0; k < col.length; k++) col[k] = dim ? rgb[k] / 255 * 0.5 + 0.25 : rgb[k] / 255;
    gp.geometry.getAttribute('color').needsUpdate = true;
  });
  $('legend').textContent = mode === 'err' ? `error: blue 0  ->  red ${vmax.toFixed(2)} (grey = no GT)` : mode === 'conf' ? `confidence: blue ${P.conf_range[0].toFixed(2)}  ->  red ${P.conf_range[1].toFixed(2)}` : mode === 'sparse' ? 'yellow = pixel given as sparse-depth input' : 'RGB';
}
const layout = () => document.querySelector('input[name=layout]:checked').value;
const twinOffset = (() => {
  const M = P.c2w[0], r = new THREE.Vector3(M[0], M[4], M[8]).normalize();  // camera x axis in OpenCV world
  let lo = Infinity, hi = -Infinity;
  frames.forEach(f => { const a = f.geometry.getAttribute('position').array; for (let k = 0; k < a.length; k += 3) { const d = a[k] * r.x + a[k+1] * r.y + a[k+2] * r.z; if (d < lo) lo = d; if (d > hi) hi = d; } });
  const w = (hi > lo) ? (hi - lo) : P.span;
  return r.multiplyScalar(w * 1.15);
})();
let onFrame = [];
function applyFrames() {
  const t = parseInt(frameEl.value, 10), acc = $('accum').checked;
  onFrame = []; for (let i = 0; i < P.S; i++) onFrame.push(acc ? i <= t : i === t);
  setVis('all');
  const s = P.stats[t]; const name = P.frames[t] ? P.frames[t].split('/').pop() : t;
  $('frameLbl').textContent = `${t}/${P.S - 1}`;
  $('stats').textContent = `frame ${t}  ${name}\nmetric AbsRel ${s.abs_rel != null ? s.abs_rel.toFixed(4) : 'n/a'}\ndelta1        ${s.delta1 != null ? s.delta1.toFixed(4) : 'n/a'}\nGT pixels     ${s.n}`;
}
// which objects a pass shows: 'all' (overlay, per the checkboxes), 'gt' (left half), 'pred' (right half)
function setVis(side) {
  const sp = $('showPred').checked, sg = $('showGt').checked, sc = $('showCams').checked, ss = $('showSparse').checked;
  const L = layout(), twin = L === 'twin';
  matGt.opacity = L === 'overlay' ? 0.6 : 1.0;
  gtGroup.position.copy(twin ? twinOffset : new THREE.Vector3());
  for (let i = 0; i < P.S; i++) {
    const on = onFrame[i];
    if (side === 'gt') { frames[i].visible = false; sparseFrames[i].visible = false; gtFrames[i].visible = on; camFrames[i].visible = on && sc; camFramesGt[i].visible = false; }
    else if (side === 'pred') { frames[i].visible = on; sparseFrames[i].visible = on && ss; gtFrames[i].visible = false; camFrames[i].visible = on && sc; camFramesGt[i].visible = false; }
    else if (twin) { frames[i].visible = on && sp; gtFrames[i].visible = on; sparseFrames[i].visible = on && ss; camFrames[i].visible = on && sc; camFramesGt[i].visible = on && sc; }
    else { frames[i].visible = on && sp; gtFrames[i].visible = on && sg; sparseFrames[i].visible = on && ss; camFrames[i].visible = on && sc; camFramesGt[i].visible = false; }
  }
}
function applyLayout() {
  recolor();
  const side = layout() === 'side';
  $('halfL').style.display = $('halfR').style.display = side ? 'block' : 'none';
  $('showGt').disabled = side || layout() === 'twin'; $('showPred').disabled = side;
  applyFrames();
}
function applySize() { const s = parseInt($('psize').value, 10) / 4; mat.size = s; matGt.size = s; matSp.size = s * 2.5; }
document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener('change', recolor));
['showPred', 'showGt', 'showCams', 'showSparse', 'accum'].forEach(id => $(id).addEventListener('change', applyFrames));
document.querySelectorAll('input[name=layout]').forEach(r => r.addEventListener('change', () => { applyLayout(); if ($('follow').checked) gotoCam(parseInt(frameEl.value, 10)); }));
frameEl.addEventListener('input', applyFrames); $('psize').addEventListener('input', applySize);
$('vmax').addEventListener('input', () => { vmax = parseInt($('vmax').value, 10) / 100; $('vmaxLbl').textContent = vmax.toFixed(2); recolor(); });
window.addEventListener('keydown', e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { frameEl.value = Math.min(P.S - 1, Math.max(0, +frameEl.value + (e.key === 'ArrowRight' ? 1 : -1))); applyFrames(); if ($('follow').checked) gotoCam(parseInt(frameEl.value, 10)); } else if (e.key === 'g') { $('showGt').checked = !$('showGt').checked; applyFrames(); } else if (e.key === 'f') { $('follow').checked = false; fitAll(); } else if (e.key === 's' || e.key === 't') { const want = e.key === 's' ? 'side' : 'twin'; const r = document.querySelector(`input[name=layout][value=${layout() === want ? 'overlay' : want}]`); r.checked = true; applyLayout(); if ($('follow').checked) gotoCam(parseInt(frameEl.value, 10)); } else if (e.key === 'a') { $('accum').checked = !$('accum').checked; applyFrames(); } });
window.addEventListener('resize', () => { renderer.setSize(innerWidth, innerHeight); });
// Put the viewer camera exactly where frame t's camera was, same orientation
// and field of view. c2w is OpenCV (x right, y down, z forward) in the
// OpenCV world; root flips y and z, so T_gl = F * c2w * F with F = diag(1,-1,-1).
const F = new THREE.Matrix4().makeScale(1, -1, -1);
// through: exactly frame t's camera (what the model saw). shoulder: in the
// scene next to that camera -- back, up and to the right of it in its own
// frame, looking at the point it looked at -- so the cloud reads as a
// scene, not an image, and depth errors show as geometry.
let viewMode = 'shoulder';  // default layout is twin (GT beside), set in the HTML
function gotoCam(t, mode) {
  viewMode = mode || viewMode;
  const Mcv = new THREE.Matrix4().fromArray(P.c2w[t]).transpose();
  const Mgl = new THREE.Matrix4().copy(F).multiply(Mcv).multiply(F);
  const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
  Mgl.decompose(pos, quat, scl);
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(quat), up = new THREE.Vector3(0, 1, 0).applyQuaternion(quat), right = new THREE.Vector3(1, 0, 0).applyQuaternion(quat);
  const d = Math.max(P.span * 0.5, 0.5);
  const target = pos.clone().addScaledVector(fwd, d);
  if (viewMode === 'through') {
    camera.position.copy(pos); camera.quaternion.copy(quat);
    const K = P.K[t]; camera.fov = 2 * Math.atan(P.H / (2 * K[4])) * 180 / Math.PI;
  } else {
    const twin = layout() === 'twin';
    const half = twin ? new THREE.Vector3().copy(twinOffset).multiplyScalar(0.5).applyMatrix4(new THREE.Matrix4().makeScale(1, -1, -1)) : new THREE.Vector3();
    target.add(half);
    camera.position.copy(pos).add(half).addScaledVector(fwd, -(twin ? 1.6 : 0.9) * d).addScaledVector(up, 0.55 * d).addScaledVector(right, 0.6 * d);
    camera.fov = 55;
  }
  camera.up.copy(up); camera.updateProjectionMatrix();
  controls.target.copy(target); controls.update();
}
function fitAll() {
  const box = new THREE.Box3();
  frames.forEach((f, i) => { if (onFrame[i]) { f.geometry.computeBoundingBox(); box.union(f.geometry.boundingBox.clone().applyMatrix4(root.matrix)); } });
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2;
  camera.up.set(0, 1, 0); camera.fov = 55; camera.updateProjectionMatrix();
  camera.position.copy(c).add(new THREE.Vector3(0, r * 0.5, r * 1.6)); controls.target.copy(c); controls.update();
}
$('gotoCam').addEventListener('click', () => gotoCam(parseInt(frameEl.value, 10), 'through'));
$('gotoShoulder').addEventListener('click', () => gotoCam(parseInt(frameEl.value, 10), 'shoulder'));
$('fit').addEventListener('click', () => { $('follow').checked = false; fitAll(); });
frameEl.addEventListener('input', () => { if ($('follow').checked) gotoCam(parseInt(frameEl.value, 10)); });
controls.addEventListener('start', () => { $('follow').checked = false; });
(function metricsPanel() {
  const M = P.metrics, el = $('metrics');
  if (!M || (!M.sequence && !M.tae)) { el.textContent = 'no bench_results.json next to this snapshot'; return; }
  const f = v => (v == null || Number.isNaN(v)) ? '-' : (+v).toFixed(4);
  const protos = ['published', 'sparse_aligned', 'metric'], mets = ['abs_rel', 'delta1', 'rmse'];
  let h = `<b>this sequence vs dataset mean</b> (${M.dataset.n_sequences ? M.dataset.n_sequences + ' seqs' : 'no aggregate'}, density ${(P.meta.density*100).toFixed(0)}%)<table><tr><th>protocol</th>`;
  mets.forEach(m => h += `<th>${m} seq</th><th>mean</th>`); h += '</tr>';
  protos.forEach(p => { h += `<tr><td>${p}</td>`; mets.forEach(m => h += `<td>${M.sequence ? f(M.sequence[p][m]) : '-'}</td><td style="opacity:.7">${f(M.dataset[p + '_' + m])}</td>`); h += '</tr>'; });
  if (M.tae || M.dataset.tae_vda != null) h += `<tr><td>tae_vda (x100)</td><td>${M.tae ? f(M.tae.tae_vda) : '-'}</td><td style="opacity:.7">${f(M.dataset.tae_vda)}</td><td colspan=4></td></tr><tr><td>tae_ours</td><td>${M.tae ? f(M.tae.tae_ours) : '-'}</td><td style="opacity:.7">${f(M.dataset.tae_ours)}</td><td colspan=4></td></tr>`;
  h += '</table>'; el.innerHTML = h;
})();
$('vmaxLbl').textContent = vmax.toFixed(2); applySize(); recolor(); applyFrames(); gotoCam(P.S - 1, 'shoulder');
renderer.setAnimationLoop(() => {
  controls.update();
  const w = innerWidth, h = innerHeight;
  if (layout() === 'side') {
    // one camera, two passes: GT on the left half, prediction on the right, same viewpoint
    camera.aspect = (w / 2) / h; camera.updateProjectionMatrix();
    renderer.setScissorTest(true);
    setVis('gt'); renderer.setViewport(0, 0, w / 2, h); renderer.setScissor(0, 0, w / 2, h); renderer.render(scene, camera);
    setVis('pred'); renderer.setViewport(w / 2, 0, w / 2, h); renderer.setScissor(w / 2, 0, w / 2, h); renderer.render(scene, camera);
    renderer.setScissorTest(false);
  } else {
    if (camera.aspect !== w / h) { camera.aspect = w / h; camera.updateProjectionMatrix(); }
    setVis('all'); renderer.setViewport(0, 0, w, h); renderer.render(scene, camera);
  }
});
</script></body></html>
"""


def write_html(
    npz_path: str | Path,
    out: str | Path | None = None,
    every: int = 2,
    cameras: str = "auto",
    results: str | Path | None = None,
) -> Path:
    npz_path = Path(npz_path)
    payload = build_payload(
        npz_path,
        every=every,
        cameras=cameras,
        results=None if results is None else Path(results),
    )
    out = Path(out) if out is not None else npz_path.with_suffix(".html")
    title = f"{payload['meta'].get('dataset', '')} {payload['meta'].get('sequence', '')} {npz_path.stem}"
    html = (
        _HTML.replace("__THREE__", _THREE)
        .replace("__TITLE__", title)
        .replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))
    )
    out.write_text(html)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("npz", nargs="+")
    ap.add_argument(
        "--every",
        type=int,
        default=2,
        help="pixel stride on both axes (1 = every pixel)",
    )
    ap.add_argument("--cameras", choices=["auto", "gt", "pred"], default="auto")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--results",
        default=None,
        help="bench_results.json (default: next to bench_clouds/)",
    )
    args = ap.parse_args()
    for p in args.npz:
        out = (
            None
            if args.out_dir is None
            else Path(args.out_dir) / (Path(p).stem + ".html")
        )
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
        print(f"{p} -> {write_html(p, out, args.every, args.cameras, args.results)}")


if __name__ == "__main__":
    main()

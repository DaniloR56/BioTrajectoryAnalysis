#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AntTrajectoryAnalysis_Biophysics.py

Trajectory analysis script for ant foraging videos.

Main focus:
  1. Spatial exploration and Monte Carlo area estimation
  2. Directional statistics and anisotropy analysis
  3. PCA and PCA-whitened coordinates
  4. Stochastic trajectory descriptors: turning angles, MSD, straightness
  5. Fractal/box-counting analysis of spatial exploration

The script intentionally does not make pi estimation central.  It is designed
for the revised paper framing: biological trajectories as experimentally
accessible stochastic samplers and anisotropic motion patterns.

Example:
  python AntTrajectoryAnalysis_Biophysics.py ground_bg_results/ant_positions.csv \
      --cm-per-pixel 0.02899367 --output ant_biophysics_analysis

If the CSV already contains x_cm and y_cm columns, they are used automatically.
Otherwise x/y pixel columns are converted if --cm-per-pixel is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from scipy.stats import circmean, rayleigh
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

SCRIPT_VERSION = "1.2.0"


# -----------------------------------------------------------------------------
# Data loading and coordinate handling
# -----------------------------------------------------------------------------

def load_positions(csv_path: Path, cm_per_pixel: Optional[float] = None) -> Tuple[pd.DataFrame, str, Optional[float]]:
    """
    Load tracking CSV and return columns x_analysis, y_analysis in a physical or pixel unit.

    Important coordinate convention
    -------------------------------
    OpenCV/image coordinates have the origin in the upper-left corner and y increases
    downward. For scientific plots we use a Cartesian-style convention with the
    origin at the lower-left and y increasing upward. Therefore, after loading and
    calibration, y is reflected as:

        y_analysis = ymax - y_analysis

    This changes only the display/orientation convention. Distances, areas,
    anisotropy ratios, MSD, straightness, and fractal dimensions are unchanged by
    this reflection. Only orientation angles change sign/convention.
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    if "x_cm" in cols and "y_cm" in cols:
        xcol, ycol = cols["x_cm"], cols["y_cm"]
        df = df.rename(columns={xcol: "x_analysis", ycol: "y_analysis"})
        unit = "cm"

        # infer cm_per_pixel only if possible
        if cm_per_pixel is None and "x" in cols:
            xpx = df[cols["x"]].to_numpy(dtype=float)
            xcm = df["x_analysis"].to_numpy(dtype=float)
            dxpx = np.nanmax(xpx) - np.nanmin(xpx)
            dxcm = np.nanmax(xcm) - np.nanmin(xcm)
            if dxpx > 0:
                cm_per_pixel = dxcm / dxpx

        # Convert image-style y coordinates to Cartesian-style plotting coordinates.
        ymax = df["y_analysis"].max()
        df["y_analysis"] = ymax - df["y_analysis"]

        return df, unit, cm_per_pixel

    # Fallback to x/y columns
    if "x" not in cols or "y" not in cols:
        raise ValueError("CSV must contain either x_cm/y_cm or x/y columns.")

    xcol, ycol = cols["x"], cols["y"]
    if cm_per_pixel is not None:
        df["x_analysis"] = df[xcol].astype(float) * cm_per_pixel
        df["y_analysis"] = df[ycol].astype(float) * cm_per_pixel
        unit = "cm"
    else:
        df["x_analysis"] = df[xcol].astype(float)
        df["y_analysis"] = df[ycol].astype(float)
        unit = "px"

    # Convert image-style y coordinates to Cartesian-style plotting coordinates.
    ymax = df["y_analysis"].max()
    df["y_analysis"] = ymax - df["y_analysis"]

    return df, unit, cm_per_pixel


def get_id_frame_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("track_id") or cols.get("object_id")
    frame_col = cols.get("frame") or cols.get("frame_number") or cols.get("t")
    return id_col, frame_col


def points_from_df(df: pd.DataFrame) -> np.ndarray:
    pts = df[["x_analysis", "y_analysis"]].to_numpy(dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


# -----------------------------------------------------------------------------
# Spatial exploration and area estimation
# -----------------------------------------------------------------------------

def convex_hull_analysis(points: np.ndarray) -> Dict:
    if len(points) < 3:
        raise ValueError("At least 3 points are required for convex hull analysis.")
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    area = float(hull.volume)      # In 2D, scipy calls area 'volume'
    perimeter = float(hull.area)   # In 2D, scipy calls perimeter 'area'
    circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else np.nan
    return {
        "hull": hull,
        "hull_points": hull_pts,
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
    }


def monte_carlo_area(points: np.ndarray, hull_points: np.ndarray, n_samples: int = 50000, seed: int = 12345) -> Dict:
    rng = np.random.default_rng(seed)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    box_area = (xmax - xmin) * (ymax - ymin)

    random_pts = np.column_stack([
        rng.uniform(xmin, xmax, n_samples),
        rng.uniform(ymin, ymax, n_samples),
    ])
    path = MplPath(hull_points)
    inside = path.contains_points(random_pts)
    fraction = inside.mean()
    area = box_area * fraction

    # Binomial standard error for area estimate
    se_fraction = math.sqrt(max(fraction * (1 - fraction), 0) / n_samples)
    se_area = box_area * se_fraction
    ci95 = (area - 1.96 * se_area, area + 1.96 * se_area)

    return {
        "area": float(area),
        "ci95": (float(ci95[0]), float(ci95[1])),
        "fraction_inside": float(fraction),
        "random_points": random_pts,
        "inside": inside,
        "box": (float(xmin), float(xmax), float(ymin), float(ymax)),
        "box_area": float(box_area),
    }


def occupancy_grid(points: np.ndarray, cell_size: float) -> Dict:
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    nx = max(1, int(math.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(math.ceil((ymax - ymin) / cell_size)))

    H, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])
    H = H.T  # rows are y bins for imshow
    visited = H > 0
    visited_area = visited.sum() * (cell_size ** 2)

    return {
        "H": H,
        "xedges": xedges,
        "yedges": yedges,
        "visited_area": float(visited_area),
        "visited_cells": int(visited.sum()),
        "cell_size": float(cell_size),
        "extent": (float(xmin), float(xmax), float(ymin), float(ymax)),
    }


# -----------------------------------------------------------------------------
# PCA and directional statistics
# -----------------------------------------------------------------------------

def pca_analysis(points: np.ndarray) -> Dict:
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    anisotropy = float(eigvals[0] / eigvals[1]) if eigvals[1] > 0 else np.inf
    explained = eigvals / eigvals.sum() if eigvals.sum() > 0 else np.array([np.nan, np.nan])
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Whiten coordinates: centered @ eigvecs @ diag(1/sqrt(eigvals))
    eps = 1e-12
    whitened = centered @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps))

    return {
        "centered": centered,
        "cov": cov,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "anisotropy": anisotropy,
        "explained": explained,
        "angle_deg": float(angle),
        "whitened": whitened,
    }


def displacement_vectors(df: pd.DataFrame, id_col: Optional[str], frame_col: Optional[str]) -> np.ndarray:
    vectors = []
    if id_col is not None:
        group_iter = df.groupby(id_col)
    else:
        group_iter = [(None, df)]

    for _, g in group_iter:
        if frame_col is not None:
            g = g.sort_values(frame_col)
        pts = g[["x_analysis", "y_analysis"]].to_numpy(dtype=float)
        if len(pts) >= 2:
            d = np.diff(pts, axis=0)
            norms = np.linalg.norm(d, axis=1)
            d = d[norms > 0]
            if len(d):
                vectors.append(d)

    if not vectors:
        return np.empty((0, 2))
    return np.vstack(vectors)


def directional_statistics(vectors: np.ndarray) -> Dict:
    if len(vectors) == 0:
        return {"angles": np.array([]), "mean_direction_deg": np.nan, "resultant_length": np.nan, "rayleigh_p": np.nan}

    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    C = np.cos(angles).sum()
    S = np.sin(angles).sum()
    R = math.sqrt(C**2 + S**2)
    n = len(angles)
    Rbar = R / n
    mean_dir = math.degrees(math.atan2(S, C))

    # Approximate Rayleigh test p-value for non-uniformity on the circle
    z = n * Rbar**2
    p = math.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2)) if n > 0 else np.nan
    p = min(max(p, 0.0), 1.0) if np.isfinite(p) else np.nan

    return {
        "angles": angles,
        "mean_direction_deg": float(mean_dir),
        "resultant_length": float(Rbar),
        "rayleigh_p": float(p),
    }


# -----------------------------------------------------------------------------
# Trajectory statistics
# -----------------------------------------------------------------------------

def turning_angles(df: pd.DataFrame, id_col: Optional[str], frame_col: Optional[str]) -> np.ndarray:
    all_phi = []
    group_iter = df.groupby(id_col) if id_col is not None else [(None, df)]
    for _, g in group_iter:
        if frame_col is not None:
            g = g.sort_values(frame_col)
        pts = g[["x_analysis", "y_analysis"]].to_numpy(dtype=float)
        if len(pts) < 3:
            continue
        v = np.diff(pts, axis=0)
        norms = np.linalg.norm(v, axis=1)
        valid = norms > 0
        v = v[valid]
        norms = norms[valid]
        if len(v) < 2:
            continue
        dot = np.sum(v[:-1] * v[1:], axis=1)
        denom = norms[:-1] * norms[1:]
        cosang = np.clip(dot / denom, -1.0, 1.0)
        phi = np.arccos(cosang)
        all_phi.append(phi)
    if not all_phi:
        return np.array([])
    return np.concatenate(all_phi)


def straightness_indices(df: pd.DataFrame, id_col: Optional[str], frame_col: Optional[str], min_points: int = 5) -> np.ndarray:
    vals = []
    group_iter = df.groupby(id_col) if id_col is not None else [(None, df)]
    for _, g in group_iter:
        if frame_col is not None:
            g = g.sort_values(frame_col)
        pts = g[["x_analysis", "y_analysis"]].to_numpy(dtype=float)
        if len(pts) < min_points:
            continue
        d = np.diff(pts, axis=0)
        path_len = np.linalg.norm(d, axis=1).sum()
        net = np.linalg.norm(pts[-1] - pts[0])
        if path_len > 0:
            vals.append(net / path_len)
    return np.array(vals)


def msd_analysis(df: pd.DataFrame, id_col: Optional[str], frame_col: Optional[str], max_lag: int = 100) -> Dict:
    lag_vals: List[int] = []
    msd_vals: List[float] = []
    counts: List[int] = []

    # gather per lag across tracks
    group_iter = df.groupby(id_col) if id_col is not None else [(None, df)]
    tracks = []
    for _, g in group_iter:
        if frame_col is not None:
            g = g.sort_values(frame_col)
        pts = g[["x_analysis", "y_analysis"]].to_numpy(dtype=float)
        if len(pts) > 2:
            tracks.append(pts)

    if not tracks:
        return {"lags": np.array([]), "msd": np.array([]), "counts": np.array([]), "beta": np.nan}

    max_possible = min(max_lag, max(len(t) for t in tracks) - 1)
    for lag in range(1, max_possible + 1):
        sq = []
        for pts in tracks:
            if len(pts) <= lag:
                continue
            disp = pts[lag:] - pts[:-lag]
            sq.append(np.sum(disp**2, axis=1))
        if sq:
            sq_all = np.concatenate(sq)
            lag_vals.append(lag)
            msd_vals.append(float(np.mean(sq_all)))
            counts.append(int(len(sq_all)))

    lags = np.array(lag_vals, dtype=float)
    msd = np.array(msd_vals, dtype=float)

    beta = np.nan
    valid = (lags > 0) & (msd > 0)
    if valid.sum() >= 5:
        # Fit log(MSD) = beta log(lag) + c over the first half of available lags to avoid long-lag noise
        idx = np.where(valid)[0]
        idx = idx[: max(5, len(idx)//2)]
        beta, _ = np.polyfit(np.log(lags[idx]), np.log(msd[idx]), 1)

    return {"lags": lags, "msd": msd, "counts": np.array(counts), "beta": float(beta)}


def fractal_box_counting(points: np.ndarray, n_scales: int = 12) -> Dict:
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    extent = max(xmax - xmin, ymax - ymin)
    if extent <= 0:
        return {"eps": np.array([]), "N": np.array([]), "D": np.nan}

    # Box sizes from large to small. Avoid too small boxes compared with point density.
    eps_values = np.logspace(np.log10(extent / 2), np.log10(extent / 80), n_scales)
    counts = []
    for eps in eps_values:
        ix = np.floor((points[:, 0] - xmin) / eps).astype(int)
        iy = np.floor((points[:, 1] - ymin) / eps).astype(int)
        boxes = set(zip(ix, iy))
        counts.append(len(boxes))
    counts = np.array(counts, dtype=float)

    valid = (eps_values > 0) & (counts > 0)
    D = np.nan
    if valid.sum() >= 4:
        x = np.log(1 / eps_values[valid])
        y = np.log(counts[valid])
        D, _ = np.polyfit(x, y, 1)
    return {"eps": eps_values, "N": counts, "D": float(D)}


# -----------------------------------------------------------------------------
# Spatial information geometry
# -----------------------------------------------------------------------------

def load_roi_from_config(config_path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[Path]]:
    """Load polygon ROI from the tracking config.json, if available."""
    if config_path is None:
        return None, None
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = json.loads(config_path.read_text())
    roi = config.get("roi")
    if not roi:
        return None, config_path
    roi = np.asarray(roi, dtype=float)
    if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3:
        raise ValueError("ROI in config.json must be a polygon with at least 3 [x,y] points.")
    return roi, config_path


def infer_frame_size(df: pd.DataFrame, roi: Optional[np.ndarray], frame_width: Optional[int],
                     frame_height: Optional[int]) -> Tuple[int, int, str]:
    """Resolve a camera-coordinate extent that contains both tracked data and ROI."""
    cols = {c.lower(): c for c in df.columns}
    if "x" not in cols or "y" not in cols:
        raise ValueError("Information-geometry analysis requires raw pixel columns x and y.")
    x = df[cols["x"]].to_numpy(dtype=float)
    y = df[cols["y"]].to_numpy(dtype=float)
    required_w = int(math.ceil(np.nanmax(x) + 1))
    required_h = int(math.ceil(np.nanmax(y) + 1))
    if roi is not None:
        required_w = max(required_w, int(math.ceil(np.nanmax(roi[:, 0]) + 1)))
        required_h = max(required_h, int(math.ceil(np.nanmax(roi[:, 1]) + 1)))

    notes = []
    if frame_width is None:
        frame_width = required_w; notes.append("width inferred")
    elif frame_width < required_w:
        notes.append(f"width enlarged {frame_width}->{required_w} to contain data/ROI")
        frame_width = required_w
    else:
        notes.append("explicit width accepted")

    if frame_height is None:
        frame_height = required_h; notes.append("height inferred")
    elif frame_height < required_h:
        notes.append(f"height enlarged {frame_height}->{required_h} to contain data/ROI")
        frame_height = required_h
    else:
        notes.append("explicit height accepted")

    return int(frame_width), int(frame_height), "; ".join(notes)


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Fisher--Rao distance for discrete probability vectors: 2 arccos(sum sqrt(p q))."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be 1D probability vectors with the same shape")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probability vectors cannot contain negative values")
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return np.nan
    p, q = p / sp, q / sq
    bc = float(np.sum(np.sqrt(p * q)))
    return float(2.0 * np.arccos(np.clip(bc, 0.0, 1.0)))


def bootstrap_fisher_rao_histograms(a: np.ndarray, b: np.ndarray, edges: np.ndarray,
                                    n_boot: int = 200,
                                    rng: Optional[np.random.Generator] = None) -> Dict:
    """Bootstrap uncertainty of Fisher--Rao distance between two angle samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2 or n_boot < 1:
        return {"observed": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if rng is None:
        rng = np.random.default_rng(12345)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    observed = fisher_rao_distance(ha.astype(float), hb.astype(float))
    vals = np.empty(n_boot)
    for k in range(n_boot):
        aa = rng.choice(a, len(a), replace=True)
        bb = rng.choice(b, len(b), replace=True)
        h1, _ = np.histogram(aa, bins=edges)
        h2, _ = np.histogram(bb, bins=edges)
        vals[k] = fisher_rao_distance(h1.astype(float), h2.astype(float))
    return {"observed": float(observed), "se": float(np.std(vals, ddof=1)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5))}


def local_turning_angle_grid(df: pd.DataFrame, id_col: Optional[str], frame_col: Optional[str],
                             frame_width: int, frame_height: int, nx: int = 20, ny: int = 12,
                             angle_bins: int = 18, min_samples: int = 30,
                             roi: Optional[np.ndarray] = None, pseudocount: float = 0.0,
                             bootstrap_samples: int = 0, random_seed: int = 12345) -> Dict:
    """
    Build local turning-angle distributions on a regular grid covering the full frame.

    Turning angles are assigned to the middle position of each three-point sequence.
    The analysis uses the raw image x/y coordinates, so it aligns directly with the
    tracking ROI.  Cells outside the ROI and cells with insufficient observations are
    represented by NaN in numerical maps, never by a physical zero.
    """
    if nx < 1 or ny < 1 or angle_bins < 2 or min_samples < 1:
        raise ValueError("nx, ny, angle_bins and min_samples must be positive (angle_bins >= 2).")
    cols = {c.lower(): c for c in df.columns}
    if "x" not in cols or "y" not in cols:
        raise ValueError("Local information geometry requires raw x and y pixel columns.")
    xcol, ycol = cols["x"], cols["y"]

    samples: List[List[List[float]]] = [[[] for _ in range(nx)] for _ in range(ny)]
    group_iter = df.groupby(id_col) if id_col is not None else [(None, df)]
    for _, g in group_iter:
        if frame_col is not None:
            g = g.sort_values(frame_col)
        pts = g[[xcol, ycol]].to_numpy(dtype=float)
        if len(pts) < 3:
            continue
        v1 = pts[1:-1] - pts[:-2]
        v2 = pts[2:] - pts[1:-1]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        valid = np.isfinite(pts[1:-1]).all(axis=1) & (n1 > 0) & (n2 > 0)
        mids = pts[1:-1][valid]
        dot = np.sum(v1[valid] * v2[valid], axis=1)
        phi = np.arccos(np.clip(dot / (n1[valid] * n2[valid]), -1.0, 1.0))
        for (x, y), a in zip(mids, phi):
            if not (0 <= x < frame_width and 0 <= y < frame_height):
                continue
            ix = min(nx - 1, int(x / frame_width * nx))
            iy = min(ny - 1, int(y / frame_height * ny))
            samples[iy][ix].append(float(a))

    counts = np.zeros((ny, nx), dtype=int)
    probs = np.full((ny, nx, angle_bins), np.nan, dtype=float)
    edges = np.linspace(0.0, np.pi, angle_bins + 1)

    # ROI validity is based on cell centres. Cells containing actual samples are
    # retained as ROI-valid too, which handles polygon boundaries conservatively.
    roi_valid = np.ones((ny, nx), dtype=bool)
    if roi is not None:
        path = MplPath(roi)
        xc = (np.arange(nx) + 0.5) * frame_width / nx
        yc = (np.arange(ny) + 0.5) * frame_height / ny
        centres = np.array([(x, y) for y in yc for x in xc])
        roi_valid = path.contains_points(centres).reshape(ny, nx)

    for iy in range(ny):
        for ix in range(nx):
            vals = np.asarray(samples[iy][ix], dtype=float)
            counts[iy, ix] = len(vals)
            if len(vals) > 0:
                roi_valid[iy, ix] = True
            if roi_valid[iy, ix] and len(vals) >= min_samples:
                h, _ = np.histogram(vals, bins=edges)
                h = h.astype(float) + float(pseudocount)
                if h.sum() > 0:
                    probs[iy, ix] = h / h.sum()

    valid_cells = np.isfinite(probs).all(axis=2)
    cell_indices = np.argwhere(valid_cells)  # [iy, ix]
    n_valid = len(cell_indices)
    D = np.full((n_valid, n_valid), np.nan, dtype=float)
    for a in range(n_valid):
        D[a, a] = 0.0
        ia, ja = cell_indices[a]
        for b in range(a + 1, n_valid):
            ib, jb = cell_indices[b]
            d = fisher_rao_distance(probs[ia, ja], probs[ib, jb])
            D[a, b] = D[b, a] = d

    # Mean Fisher--Rao distance to four spatial neighbours.
    gradient = np.full((ny, nx), np.nan, dtype=float)
    gradient_ci_low = np.full((ny, nx), np.nan, dtype=float)
    gradient_ci_high = np.full((ny, nx), np.nan, dtype=float)
    rng = np.random.default_rng(random_seed)
    for iy, ix in cell_indices:
        ds, lows, highs = [], [], []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < ny and 0 <= jx < nx and valid_cells[jy, jx]:
                ds.append(fisher_rao_distance(probs[iy, ix], probs[jy, jx]))
                if bootstrap_samples > 0:
                    boot = bootstrap_fisher_rao_histograms(
                        np.asarray(samples[iy][ix]), np.asarray(samples[jy][jx]),
                        edges, bootstrap_samples, rng)
                    lows.append(boot["ci_low"]); highs.append(boot["ci_high"])
        if ds:
            gradient[iy, ix] = float(np.mean(ds))
            if lows:
                gradient_ci_low[iy, ix] = float(np.mean(lows))
                gradient_ci_high[iy, ix] = float(np.mean(highs))

    return {
        "counts": counts, "probs": probs, "angle_edges": edges,
        "roi_valid": roi_valid, "valid_cells": valid_cells,
        "cell_indices": cell_indices, "distance_matrix": D,
        "gradient": gradient, "gradient_ci_low": gradient_ci_low, "gradient_ci_high": gradient_ci_high,
        "frame_width": frame_width, "frame_height": frame_height,
        "nx": nx, "ny": ny, "min_samples": min_samples, "angle_bins": angle_bins,
    }


def cluster_information_geometry(info: Dict, n_clusters: int = 4) -> np.ndarray:
    """Hierarchical clustering of valid cells using the Fisher--Rao distance matrix."""
    labels_map = np.full((info["ny"], info["nx"]), np.nan, dtype=float)
    idx = info["cell_indices"]
    D = info["distance_matrix"]
    n = len(idx)
    if n == 0:
        return labels_map
    if n == 1:
        labels = np.array([1])
    else:
        k = max(1, min(int(n_clusters), n))
        Z = linkage(squareform(D, checks=False), method="average")
        labels = fcluster(Z, t=k, criterion="maxclust")
    for (iy, ix), lab in zip(idx, labels):
        labels_map[iy, ix] = int(lab)
    return labels_map


def save_information_geometry_outputs(info: Dict, labels: np.ndarray, outdir: Path,
                                      roi: Optional[np.ndarray] = None) -> None:
    """Save numerical tables and spatial maps for the local information-geometry analysis."""
    extent = [0, info["frame_width"], info["frame_height"], 0]  # image coordinates

    # Sampling map
    fig, ax = plt.subplots(figsize=(10, 6))
    counts_plot = info["counts"].astype(float)
    counts_plot[~info["roi_valid"]] = np.nan
    im = ax.imshow(counts_plot, extent=extent, interpolation="nearest", aspect="equal")
    if roi is not None:
        rp = np.vstack([roi, roi[0]])
        ax.plot(rp[:, 0], rp[:, 1], linewidth=1.5)
    ax.set_title("Local Movement Sampling")
    ax.set_xlabel("X (pixels)"); ax.set_ylabel("Y (pixels)")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Turning-angle observations per cell")
    fig.tight_layout(); savefig(fig, outdir, "info_geometry_sampling_map")

    # Local information gradient
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(info["gradient"], extent=extent, interpolation="nearest", aspect="equal")
    if roi is not None:
        rp = np.vstack([roi, roi[0]])
        ax.plot(rp[:, 0], rp[:, 1], linewidth=1.5)
    ax.set_title("Local Fisher--Rao Heterogeneity")
    ax.set_xlabel("X (pixels)"); ax.set_ylabel("Y (pixels)")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Mean Fisher--Rao distance to valid neighbours")
    fig.tight_layout(); savefig(fig, outdir, "info_geometry_local_gradient")

    # Cluster/domain map
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(labels, extent=extent, interpolation="nearest", aspect="equal")
    if roi is not None:
        rp = np.vstack([roi, roi[0]])
        ax.plot(rp[:, 0], rp[:, 1], linewidth=1.5)
    ax.set_title("Information-Geometric Distributional Clusters")
    ax.set_xlabel("X (pixels)"); ax.set_ylabel("Y (pixels)")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Cluster label (categorical)")
    fig.tight_layout(); savefig(fig, outdir, "info_geometry_distributional_clusters")

    # Full Fisher--Rao matrix
    if len(info["cell_indices"]):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(info["distance_matrix"], origin="lower", interpolation="nearest", aspect="auto")
        ax.set_title("Fisher--Rao Distance Matrix Between Valid Spatial Cells")
        ax.set_xlabel("Valid-cell index"); ax.set_ylabel("Valid-cell index")
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Fisher--Rao distance")
        fig.tight_layout(); savefig(fig, outdir, "info_geometry_distance_matrix")

    # Machine-readable outputs
    rows = []
    for k, (iy, ix) in enumerate(info["cell_indices"]):
        row = {"cell_index": k, "grid_x": int(ix), "grid_y": int(iy),
               "n_turning_angles": int(info["counts"][iy, ix]),
               "cluster": int(labels[iy, ix]) if np.isfinite(labels[iy, ix]) else None,
               "local_gradient": float(info["gradient"][iy, ix]) if np.isfinite(info["gradient"][iy, ix]) else np.nan,
               "local_gradient_ci_low": float(info["gradient_ci_low"][iy, ix]) if np.isfinite(info["gradient_ci_low"][iy, ix]) else np.nan,
               "local_gradient_ci_high": float(info["gradient_ci_high"][iy, ix]) if np.isfinite(info["gradient_ci_high"][iy, ix]) else np.nan}
        for b, prob in enumerate(info["probs"][iy, ix]):
            row[f"p_angle_bin_{b:02d}"] = float(prob)
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "info_geometry_cells.csv", index=False)
    if len(info["cell_indices"]):
        names = [f"cell_{k}" for k in range(len(info["cell_indices"]))]
        pd.DataFrame(info["distance_matrix"], index=names, columns=names).to_csv(outdir / "fisher_rao_distance_matrix.csv")


# -----------------------------------------------------------------------------
# Temporal information geometry
# -----------------------------------------------------------------------------

def temporal_turning_angle_information_geometry(df: pd.DataFrame, id_col: Optional[str],
                                                frame_col: Optional[str], window_frames: int = 1800,
                                                step_frames: Optional[int] = None, angle_bins: int = 18,
                                                min_samples: int = 100, pseudocount: float = 0.0,
                                                fps: Optional[float] = None, bootstrap_samples: int = 0,
                                                random_seed: int = 12345) -> Dict:
    """Estimate turning-angle distributions in sliding time windows.

    Each turning angle is associated with the middle frame of a three-point sequence.
    Fisher--Rao distances are then calculated between all sufficiently sampled windows.
    Consecutive-window distance divided by the separation of the window centres provides
    an information-geometric speed.  Windows with insufficient observations remain NaN.
    """
    if frame_col is None:
        raise ValueError("Temporal information geometry requires a frame column.")
    if window_frames < 2 or angle_bins < 2 or min_samples < 1:
        raise ValueError("window_frames >= 2, angle_bins >= 2 and min_samples >= 1 are required.")
    if step_frames is None:
        step_frames = window_frames
    if step_frames < 1:
        raise ValueError("step_frames must be >= 1.")

    samples_frame, samples_angle = [], []
    group_iter = df.groupby(id_col) if id_col is not None else [(None, df)]
    for _, g in group_iter:
        g = g.sort_values(frame_col)
        pts = g[["x_analysis", "y_analysis"]].to_numpy(dtype=float)
        frames = g[frame_col].to_numpy(dtype=float)
        if len(pts) < 3:
            continue
        v1, v2 = pts[1:-1] - pts[:-2], pts[2:] - pts[1:-1]
        n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
        valid = (np.isfinite(pts[1:-1]).all(axis=1) & np.isfinite(frames[1:-1]) &
                 (n1 > 0) & (n2 > 0))
        if not np.any(valid):
            continue
        dot = np.sum(v1[valid] * v2[valid], axis=1)
        phi = np.arccos(np.clip(dot / (n1[valid] * n2[valid]), -1.0, 1.0))
        samples_frame.extend(frames[1:-1][valid].tolist())
        samples_angle.extend(phi.tolist())

    sf = np.asarray(samples_frame, dtype=float)
    sa = np.asarray(samples_angle, dtype=float)
    if len(sf) == 0:
        raise ValueError("No valid turning-angle observations for temporal information geometry.")

    first = int(math.floor(np.nanmin(sf)))
    last = int(math.ceil(np.nanmax(sf)))
    starts = np.arange(first, last + 1, int(step_frames), dtype=int)
    edges = np.linspace(0.0, np.pi, angle_bins + 1)
    probs, counts, centres, raw_window_samples = [], [], [], []
    for start in starts:
        stop = start + int(window_frames)
        vals = sa[(sf >= start) & (sf < stop)]
        counts.append(len(vals))
        centres.append(start + 0.5 * window_frames)
        raw_window_samples.append(np.asarray(vals, dtype=float))
        if len(vals) >= min_samples:
            h, _ = np.histogram(vals, bins=edges)
            h = h.astype(float) + float(pseudocount)
            probs.append(h / h.sum() if h.sum() > 0 else np.full(angle_bins, np.nan))
        else:
            probs.append(np.full(angle_bins, np.nan))
    probs = np.asarray(probs, dtype=float)
    counts = np.asarray(counts, dtype=int)
    centres = np.asarray(centres, dtype=float)
    valid = np.isfinite(probs).all(axis=1)
    valid_idx = np.flatnonzero(valid)

    D = np.full((len(valid_idx), len(valid_idx)), np.nan, dtype=float)
    for a, ia in enumerate(valid_idx):
        D[a, a] = 0.0
        for b in range(a + 1, len(valid_idx)):
            ib = valid_idx[b]
            D[a, b] = D[b, a] = fisher_rao_distance(probs[ia], probs[ib])

    consecutive_distance = np.full(len(starts), np.nan, dtype=float)
    consecutive_ci_low = np.full(len(starts), np.nan, dtype=float)
    consecutive_ci_high = np.full(len(starts), np.nan, dtype=float)
    ig_speed = np.full(len(starts), np.nan, dtype=float)
    rng = np.random.default_rng(random_seed)
    for i in range(1, len(starts)):
        if valid[i - 1] and valid[i]:
            d = fisher_rao_distance(probs[i - 1], probs[i])
            consecutive_distance[i] = d
            if bootstrap_samples > 0:
                boot = bootstrap_fisher_rao_histograms(
                    raw_window_samples[i - 1], raw_window_samples[i], edges,
                    bootstrap_samples, rng)
                consecutive_ci_low[i] = boot["ci_low"]
                consecutive_ci_high[i] = boot["ci_high"]
            dt_frames = centres[i] - centres[i - 1]
            dt = dt_frames / fps if fps and fps > 0 else dt_frames
            if dt > 0:
                ig_speed[i] = d / dt

    return {
        "starts": starts, "centres": centres, "counts": counts, "probs": probs,
        "valid": valid, "valid_indices": valid_idx, "distance_matrix": D,
        "consecutive_distance": consecutive_distance, "consecutive_ci_low": consecutive_ci_low,
        "consecutive_ci_high": consecutive_ci_high, "ig_speed": ig_speed,
        "angle_edges": edges, "window_frames": int(window_frames),
        "step_frames": int(step_frames), "angle_bins": int(angle_bins),
        "min_samples": int(min_samples), "fps": fps,
    }


def save_temporal_information_geometry_outputs(info: Dict, outdir: Path) -> None:
    """Save plots and CSV files for temporal information geometry."""
    fps = info.get("fps")
    x = info["centres"] / fps if fps and fps > 0 else info["centres"]
    xlabel = "Time (s)" if fps and fps > 0 else "Frame"

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, info["counts"], marker="o", markersize=3)
    ax.axhline(info["min_samples"], linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel); ax.set_ylabel("Turning-angle observations")
    ax.set_title("Temporal Information-Geometry Sampling")
    fig.tight_layout(); savefig(fig, outdir, "temporal_info_sampling")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, info["consecutive_distance"], marker="o", markersize=3)
    if np.isfinite(info["consecutive_ci_low"]).any():
        ax.fill_between(x, info["consecutive_ci_low"], info["consecutive_ci_high"], alpha=0.2)
    ax.set_xlabel(xlabel); ax.set_ylabel("Fisher--Rao distance")
    ax.set_title("Temporal Change in Movement Statistics")
    fig.tight_layout(); savefig(fig, outdir, "temporal_info_consecutive_distance")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, info["ig_speed"], marker="o", markersize=3)
    unit = "s$^{-1}$" if fps and fps > 0 else "frame$^{-1}$"
    ax.set_xlabel(xlabel); ax.set_ylabel(f"Information-geometric speed ({unit})")
    ax.set_title("Information-Geometric Speed")
    fig.tight_layout(); savefig(fig, outdir, "temporal_info_speed")

    if len(info["valid_indices"]):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(info["distance_matrix"], origin="lower", interpolation="nearest", aspect="auto")
        ax.set_xlabel("Valid time-window index"); ax.set_ylabel("Valid time-window index")
        ax.set_title("Fisher--Rao Distance Matrix Between Time Windows")
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Fisher--Rao distance")
        fig.tight_layout(); savefig(fig, outdir, "temporal_info_distance_matrix")

    rows = []
    for i, start in enumerate(info["starts"]):
        row = {"window_index": i, "start_frame": int(start),
               "centre_frame": float(info["centres"][i]),
               "n_turning_angles": int(info["counts"][i]),
               "valid": bool(info["valid"][i]),
               "fisher_rao_to_previous": float(info["consecutive_distance"][i]) if np.isfinite(info["consecutive_distance"][i]) else np.nan,
               "fisher_rao_ci_low": float(info["consecutive_ci_low"][i]) if np.isfinite(info["consecutive_ci_low"][i]) else np.nan,
               "fisher_rao_ci_high": float(info["consecutive_ci_high"][i]) if np.isfinite(info["consecutive_ci_high"][i]) else np.nan,
               "information_geometric_speed": float(info["ig_speed"][i]) if np.isfinite(info["ig_speed"][i]) else np.nan}
        if fps and fps > 0:
            row["centre_time_s"] = float(info["centres"][i] / fps)
        for b, prob in enumerate(info["probs"][i]):
            row[f"p_angle_bin_{b:02d}"] = float(prob) if np.isfinite(prob) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "temporal_info_windows.csv", index=False)
    if len(info["valid_indices"]):
        names = [f"window_{i}" for i in info["valid_indices"]]
        pd.DataFrame(info["distance_matrix"], index=names, columns=names).to_csv(
            outdir / "temporal_fisher_rao_distance_matrix.csv")

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_summary(points: np.ndarray, hull_info: Dict, grid: Dict, mc: Dict, pca: Dict, outdir: Path, unit: str) -> None:
    white_to_red = LinearSegmentedColormap.from_list("white_to_red", ["white", "orange", "red", "darkred"])
    xmin, xmax, ymin, ymax = grid["extent"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 1. Raw coordinates and convex hull
    ax1.scatter(points[:, 0], points[:, 1], s=0.2, alpha=0.25)
    hp = hull_info["hull_points"]
    hp_closed = np.vstack([hp, hp[0]])
    ax1.plot(hp_closed[:, 0], hp_closed[:, 1], linewidth=2)
    ax1.set_title(f"Trajectory Positions and Convex Hull\nArea = {hull_info['area']:.2f} {unit}$^2$")
    ax1.set_xlabel(f"X ({unit})")
    ax1.set_ylabel(f"Y ({unit})")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_aspect("equal", adjustable="box")

    # 2. Occupancy grid with zero white
    H = grid["H"]
    im = ax2.imshow(H, origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="auto", cmap=white_to_red)
    ax2.set_title(f"Grid-Based Foraging Intensity\nCell size = {grid['cell_size']:.2f} {unit}")
    ax2.set_xlabel(f"X ({unit})")
    ax2.set_ylabel(f"Y ({unit})")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("Visits per cell")

    # 3. PCA eigenvalues
    ax3.bar(["PC1", "PC2"], pca["eigvals"])
    ax3.set_title(f"PCA Anisotropy = {pca['anisotropy']:.2f}\nOrientation = {pca['angle_deg']:.1f}$^\\circ$")
    ax3.set_ylabel(f"Variance ({unit}$^2$)")

    # 4. PCA-whitened coordinates
    w = pca["whitened"]
    ax4.scatter(w[:, 0], w[:, 1], s=0.2, alpha=0.25)
    ax4.set_title("PCA-Whitened Coordinates")
    ax4.set_xlabel("Whitened PC1")
    ax4.set_ylabel("Whitened PC2")
    ax4.set_aspect("equal", adjustable="box")

    fig.suptitle("Spatial Exploration and Anisotropy Analysis", fontsize=14)
    fig.tight_layout()
    savefig(fig, outdir, "spatial_exploration_anisotropy")


def plot_monte_carlo_area(points: np.ndarray, hull_info: Dict, mc: Dict, outdir: Path, unit: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    random_pts = mc["random_points"]
    inside = mc["inside"]
    ax.scatter(random_pts[~inside, 0], random_pts[~inside, 1], s=0.5, alpha=0.25, label="Outside hull")
    ax.scatter(random_pts[inside, 0], random_pts[inside, 1], s=0.5, alpha=0.25, label="Inside hull")
    hp = hull_info["hull_points"]
    hp_closed = np.vstack([hp, hp[0]])
    ax.plot(hp_closed[:, 0], hp_closed[:, 1], linewidth=2, label="Convex hull")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_title(f"Monte Carlo Area Estimation\nA = {mc['area']:.2f} {unit}$^2$ (95% CI {mc['ci95'][0]:.2f}--{mc['ci95'][1]:.2f})")
    ax.legend(markerscale=6)
    fig.tight_layout()
    savefig(fig, outdir, "monte_carlo_area")


def plot_directional_and_trajectory_stats(directional: Dict, phi: np.ndarray, msd: Dict, fractal: Dict, outdir: Path, unit: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    angles_deg = np.degrees(directional["angles"])
    if len(angles_deg):
        ax1.hist(angles_deg, bins=72)
    ax1.set_title(f"Displacement Direction Distribution\nMean = {directional['mean_direction_deg']:.1f}$^\\circ$, Rayleigh p = {directional['rayleigh_p']:.2e}")
    ax1.set_xlabel("Direction angle (degrees)")
    ax1.set_ylabel("Frequency")

    if len(phi):
        ax2.hist(np.degrees(phi), bins=60)
    ax2.set_title("Turning-Angle Distribution")
    ax2.set_xlabel("Turning angle (degrees)")
    ax2.set_ylabel("Frequency")

    if len(msd["lags"]):
        ax3.loglog(msd["lags"], msd["msd"], marker="o", linestyle="-")
    ax3.set_title(f"Mean Squared Displacement\nMSD exponent $\\beta$ = {msd['beta']:.2f}")
    ax3.set_xlabel("Lag (frames)")
    ax3.set_ylabel(f"MSD ({unit}$^2$)")

    if len(fractal["eps"]):
        ax4.loglog(1 / fractal["eps"], fractal["N"], marker="o", linestyle="-")
    ax4.set_title(f"Box-Counting Analysis\nFractal dimension D = {fractal['D']:.2f}")
    ax4.set_xlabel(f"1 / box size (1/{unit})")
    ax4.set_ylabel("Occupied boxes")

    fig.suptitle("Directional and Stochastic Trajectory Statistics", fontsize=14)
    fig.tight_layout()
    savefig(fig, outdir, "trajectory_statistics")


def plot_density(points: np.ndarray, outdir: Path, unit: str, bins: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    H, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
    H = H.T
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "lightblue", "blue", "darkblue"])
    im = ax.imshow(H, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto", cmap=cmap)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_title("Spatial Occupancy Density")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Counts")
    fig.tight_layout()
    savefig(fig, outdir, "density_heatmap")



# -----------------------------------------------------------------------------
# Uncertainty estimates / bootstrap statistics
# -----------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = 1000,
                 ci: float = 95.0, seed: int = 12345) -> Dict:
    """
    Non-parametric bootstrap confidence interval for a 1D sample.

    This is useful for quantities already computed per track, such as the
    straightness index. It returns the observed statistic, bootstrap standard
    error, and percentile confidence interval.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"value": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}

    rng = np.random.default_rng(seed)
    obs = float(statistic(values))
    boots = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = statistic(sample)

    alpha = (100.0 - ci) / 2.0
    return {
        "value": obs,
        "se": float(np.std(boots, ddof=1)),
        "ci_low": float(np.percentile(boots, alpha)),
        "ci_high": float(np.percentile(boots, 100.0 - alpha)),
        "n": int(n),
    }


def linear_fit_slope_ci(x: np.ndarray, y: np.ndarray, ci: float = 95.0) -> Dict:
    """
    Ordinary least-squares slope and approximate confidence interval.

    Used for log-log MSD slopes and box-counting fractal dimensions. The
    confidence interval is based on the standard error of the regression slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"slope": np.nan, "intercept": np.nan, "se": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n": int(len(x))}

    slope, intercept = np.polyfit(x, y, 1)
    yfit = slope * x + intercept
    residuals = y - yfit
    dof = len(x) - 2
    s_err = math.sqrt(np.sum(residuals**2) / dof)
    sxx = np.sum((x - np.mean(x))**2)
    slope_se = s_err / math.sqrt(sxx) if sxx > 0 else np.nan

    # For n >= 30, 1.96 is effectively the 95% t critical value. For the
    # present fits n is usually moderate; this approximation is adequate for
    # reporting descriptive uncertainties in an educational-methods paper.
    z = 1.96 if abs(ci - 95.0) < 1e-9 else 1.96
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "se": float(slope_se),
        "ci_low": float(slope - z * slope_se),
        "ci_high": float(slope + z * slope_se),
        "n": int(len(x)),
    }


def estimate_descriptor_uncertainties(msd: Dict, fractal: Dict, straight: np.ndarray,
                                      n_boot: int = 1000) -> Dict:
    """
    Collect uncertainty estimates for the descriptors most likely to be shown
    in the manuscript table.
    """
    out = {}

    # MSD exponent uncertainty from the same log-log range used in msd_analysis
    lags = np.asarray(msd.get("lags", []), dtype=float)
    vals = np.asarray(msd.get("msd", []), dtype=float)
    valid = (lags > 0) & (vals > 0) & np.isfinite(vals)
    if valid.sum() >= 5:
        idx = np.where(valid)[0]
        idx = idx[: max(5, len(idx)//2)]
        out["msd_beta"] = linear_fit_slope_ci(np.log(lags[idx]), np.log(vals[idx]))
    else:
        out["msd_beta"] = {"slope": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}

    # Fractal dimension uncertainty from regression of log N(eps) vs log(1/eps)
    eps = np.asarray(fractal.get("eps", []), dtype=float)
    N = np.asarray(fractal.get("N", []), dtype=float)
    valid = (eps > 0) & (N > 0) & np.isfinite(N)
    if valid.sum() >= 4:
        out["fractal_D"] = linear_fit_slope_ci(np.log(1.0 / eps[valid]), np.log(N[valid]))
    else:
        out["fractal_D"] = {"slope": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}

    # Straightness uncertainty from track-to-track variability
    out["mean_straightness"] = bootstrap_ci(straight, statistic=np.mean, n_boot=n_boot)
    return out


def fmt_pm(value: float, se: float, ndigits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    if not np.isfinite(se):
        return f"{value:.{ndigits}f}"
    return f"{value:.{ndigits}f} ± {se:.{ndigits}f}"

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def write_report(outdir: Path, csv_path: Path, unit: str, cm_per_pixel: Optional[float], n_points: int,
                 hull: Dict, grid: Dict, mc: Dict, pca: Dict, directional: Dict,
                 phi: np.ndarray, msd: Dict, straight: np.ndarray, fractal: Dict, uncertainty: Optional[Dict] = None) -> None:
    report = []
    report.append("="*80)
    report.append("ANT TRAJECTORY ANALYSIS REPORT")
    report.append("Biophysics/statistical-physics framing")
    report.append("="*80)
    report.append(f"Script version: {SCRIPT_VERSION}")
    report.append(f"Input file: {csv_path}")
    report.append(f"Coordinate unit: {unit}")
    if cm_per_pixel is not None:
        report.append(f"cm_per_pixel: {cm_per_pixel:.8g}")
    report.append(f"Total positions: {n_points:,}")
    report.append("")

    report.append("1. SPATIAL EXPLORATION AND MONTE CARLO AREA")
    report.append("-"*80)
    report.append(f"Convex hull area: {hull['area']:.4f} {unit}^2")
    report.append(f"Convex hull perimeter: {hull['perimeter']:.4f} {unit}")
    report.append(f"Circularity index: {hull['circularity']:.4f} (1 = perfect circle)")
    report.append(f"Grid visited area: {grid['visited_area']:.4f} {unit}^2")
    report.append(f"Grid cell size: {grid['cell_size']:.4f} {unit}")
    report.append(f"Monte Carlo area: {mc['area']:.4f} {unit}^2")
    report.append(f"Monte Carlo 95% CI: [{mc['ci95'][0]:.4f}, {mc['ci95'][1]:.4f}] {unit}^2")
    report.append(f"Fraction of random points inside hull: {mc['fraction_inside']:.4f}")
    report.append("")

    report.append("2. PCA ANISOTROPY")
    report.append("-"*80)
    report.append(f"Eigenvalue 1: {pca['eigvals'][0]:.6g} {unit}^2")
    report.append(f"Eigenvalue 2: {pca['eigvals'][1]:.6g} {unit}^2")
    report.append(f"Anisotropy ratio lambda1/lambda2: {pca['anisotropy']:.4f}")
    report.append(f"Principal orientation: {pca['angle_deg']:.2f} degrees")
    report.append(f"Explained variance PC1: {100*pca['explained'][0]:.2f}%")
    report.append(f"Explained variance PC2: {100*pca['explained'][1]:.2f}%")
    report.append("")

    report.append("3. DIRECTIONAL STATISTICS")
    report.append("-"*80)
    report.append(f"Number of displacement vectors: {len(directional['angles']):,}")
    report.append(f"Mean direction: {directional['mean_direction_deg']:.2f} degrees")
    report.append(f"Mean resultant length: {directional['resultant_length']:.4f}")
    report.append(f"Rayleigh p-value: {directional['rayleigh_p']:.4e}")
    report.append("")

    report.append("4. STOCHASTIC TRAJECTORY DESCRIPTORS")
    report.append("-"*80)
    report.append(f"Turning angles computed: {len(phi):,}")
    if len(phi):
        report.append(f"Mean turning angle: {np.degrees(np.mean(phi)):.2f} degrees")
        report.append(f"Median turning angle: {np.degrees(np.median(phi)):.2f} degrees")
    report.append(f"MSD exponent beta: {msd['beta']:.4f}")
    if uncertainty is not None:
        b = uncertainty.get("msd_beta", {})
        report.append(f"MSD beta SE: {b.get('se', np.nan):.4f}")
        report.append(f"MSD beta 95% CI: [{b.get('ci_low', np.nan):.4f}, {b.get('ci_high', np.nan):.4f}]")
    report.append(f"Straightness tracks analysed: {len(straight):,}")
    if len(straight):
        report.append(f"Mean straightness: {np.mean(straight):.4f}")
        report.append(f"Median straightness: {np.median(straight):.4f}")
        if uncertainty is not None:
            st = uncertainty.get("mean_straightness", {})
            report.append(f"Mean straightness SE: {st.get('se', np.nan):.4f}")
            report.append(f"Mean straightness 95% CI: [{st.get('ci_low', np.nan):.4f}, {st.get('ci_high', np.nan):.4f}]")
    report.append(f"Fractal dimension D: {fractal['D']:.4f}")
    if uncertainty is not None:
        fd = uncertainty.get("fractal_D", {})
        report.append(f"Fractal dimension SE: {fd.get('se', np.nan):.4f}")
        report.append(f"Fractal dimension 95% CI: [{fd.get('ci_low', np.nan):.4f}, {fd.get('ci_high', np.nan):.4f}]")
    report.append("")

    report.append("INTERPRETATION")
    report.append("-"*80)
    report.append("The analysis treats ant trajectories as experimentally acquired stochastic")
    report.append("samples of space. Monte Carlo sampling estimates the explored irregular")
    report.append("domain, while PCA and directional statistics quantify anisotropy and")
    report.append("preferred directions of motion. These quantities provide a compact")
    report.append("framework for comparing biological motion with ideal isotropic random walks.")
    report.append("="*80)

    txt = "\n".join(report)
    (outdir / "analysis_report.txt").write_text(txt)
    print(txt)

    # JSON output
    data = {
        "script_version": SCRIPT_VERSION,
        "input_file": str(csv_path),
        "unit": unit,
        "cm_per_pixel": cm_per_pixel,
        "n_positions": int(n_points),
        "convex_hull_area": hull["area"],
        "convex_hull_perimeter": hull["perimeter"],
        "circularity": hull["circularity"],
        "grid_visited_area": grid["visited_area"],
        "grid_cell_size": grid["cell_size"],
        "monte_carlo_area": mc["area"],
        "monte_carlo_ci95": mc["ci95"],
        "pca_eigenvalues": pca["eigvals"].tolist(),
        "anisotropy_ratio": pca["anisotropy"],
        "pca_angle_deg": pca["angle_deg"],
        "pca_explained_variance": pca["explained"].tolist(),
        "mean_direction_deg": directional["mean_direction_deg"],
        "mean_resultant_length": directional["resultant_length"],
        "rayleigh_p": directional["rayleigh_p"],
        "mean_turning_angle_deg": float(np.degrees(np.mean(phi))) if len(phi) else None,
        "msd_beta": msd["beta"],
        "mean_straightness": float(np.mean(straight)) if len(straight) else None,
        "fractal_dimension": fractal["D"],
        "uncertainty": uncertainty,
    }
    (outdir / "analysis_results.json").write_text(json.dumps(data, indent=2))

    # Compact one-row table useful for combining multiple datasets in the paper
    if uncertainty is not None:
        beta_u = uncertainty.get("msd_beta", {})
        D_u = uncertainty.get("fractal_D", {})
        S_u = uncertainty.get("mean_straightness", {})
    else:
        beta_u, D_u, S_u = {}, {}, {}

    summary = pd.DataFrame([{
        "dataset": csv_path.stem,
        "n_positions": n_points,
        f"convex_hull_area_{unit}2": hull["area"],
        "circularity_index": hull["circularity"],
        f"monte_carlo_area_{unit}2": mc["area"],
        f"monte_carlo_area_ci95_low_{unit}2": mc["ci95"][0],
        f"monte_carlo_area_ci95_high_{unit}2": mc["ci95"][1],
        "anisotropy_ratio": pca["anisotropy"],
        "msd_beta": msd["beta"],
        "msd_beta_se": beta_u.get("se", np.nan),
        "msd_beta_ci95_low": beta_u.get("ci_low", np.nan),
        "msd_beta_ci95_high": beta_u.get("ci_high", np.nan),
        "mean_straightness": float(np.mean(straight)) if len(straight) else np.nan,
        "mean_straightness_se": S_u.get("se", np.nan),
        "mean_straightness_ci95_low": S_u.get("ci_low", np.nan),
        "mean_straightness_ci95_high": S_u.get("ci_high", np.nan),
        "fractal_dimension_D": fractal["D"],
        "fractal_dimension_se": D_u.get("se", np.nan),
        "fractal_dimension_ci95_low": D_u.get("ci_low", np.nan),
        "fractal_dimension_ci95_high": D_u.get("ci_high", np.nan),
    }])
    summary.to_csv(outdir / "summary_for_paper.csv", index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Biophysics-oriented ant trajectory analysis")
    parser.add_argument("csv", type=Path, help="CSV file containing tracking positions")
    parser.add_argument("--output", "-o", type=Path, default=Path("ant_trajectory_biophysics_analysis"), help="Output directory")
    parser.add_argument("--cm-per-pixel", type=float, default=None, help="Optional calibration factor if CSV is in pixels")
    parser.add_argument("--mc-samples", type=int, default=50000, help="Number of Monte Carlo samples for area estimation")
    parser.add_argument("--grid-cell", type=float, default=None, help="Grid cell size in analysis units. Default: 2%% of max spatial extent")
    parser.add_argument("--max-msd-lag", type=int, default=100, help="Maximum lag in frames for MSD")
    parser.add_argument("--density-bins", type=int, default=200, help="Bins for density heatmap")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for confidence intervals")
    parser.add_argument("--info-geometry", action="store_true", help="Enable local spatial information-geometry analysis")
    parser.add_argument("--config", type=Path, default=None, help="Tracking config.json containing the polygon ROI")
    parser.add_argument("--frame-width", type=int, default=None, help="Full camera-frame width in pixels (recommended)")
    parser.add_argument("--frame-height", type=int, default=None, help="Full camera-frame height in pixels (recommended)")
    parser.add_argument("--info-grid-x", type=int, default=20, help="Information-geometry grid columns")
    parser.add_argument("--info-grid-y", type=int, default=12, help="Information-geometry grid rows")
    parser.add_argument("--info-angle-bins", type=int, default=18, help="Bins over 0..180 degrees for local turning-angle distributions")
    parser.add_argument("--info-min-samples", type=int, default=30, help="Minimum turning-angle observations required per spatial cell")
    parser.add_argument("--info-clusters", type=int, default=4, help="Number of hierarchical information-geometric distributional clusters")
    parser.add_argument("--info-pseudocount", type=float, default=0.0, help="Optional histogram pseudocount per angle bin (default 0)")
    parser.add_argument("--info-bootstrap", type=int, default=200, help="Bootstrap replicates for spatial neighbour Fisher--Rao CIs; 0 disables")
    parser.add_argument("--info-seed", type=int, default=12345, help="Random seed for information-geometry bootstrap")
    parser.add_argument("--temporal-info-geometry", action="store_true", help="Enable time-dependent information-geometry analysis")
    parser.add_argument("--temporal-window-frames", type=int, default=1800, help="Temporal IG window length in frames")
    parser.add_argument("--temporal-step-frames", type=int, default=None, help="Temporal IG step in frames; default equals window length")
    parser.add_argument("--temporal-min-samples", type=int, default=100, help="Minimum turning-angle observations per temporal window")
    parser.add_argument("--temporal-bootstrap", type=int, default=200, help="Bootstrap replicates for temporal consecutive-distance CIs; 0 disables")
    parser.add_argument("--fps", type=float, default=None, help="Video frame rate; if supplied temporal axes and IG speed are expressed in seconds")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ANT TRAJECTORY ANALYSIS")
    print("Spatial exploration | Monte Carlo area | PCA anisotropy | stochastic descriptors")
    print("="*80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Loading: {args.csv}")

    df, unit, cm_per_pixel = load_positions(args.csv, args.cm_per_pixel)
    points = points_from_df(df)
    id_col, frame_col = get_id_frame_columns(df)

    print(f"Loaded {len(points):,} positions")
    print(f"Coordinate unit: {unit}")
    if cm_per_pixel is not None:
        print(f"cm_per_pixel: {cm_per_pixel:.8g}")
    print(f"ID column: {id_col}; frame column: {frame_col}")

    # Spatial analyses
    hull = convex_hull_analysis(points)
    mc = monte_carlo_area(points, hull["hull_points"], n_samples=args.mc_samples)

    extent_x = points[:, 0].max() - points[:, 0].min()
    extent_y = points[:, 1].max() - points[:, 1].min()
    default_cell = max(extent_x, extent_y) * 0.02
    cell_size = args.grid_cell if args.grid_cell is not None else default_cell
    grid = occupancy_grid(points, cell_size=cell_size)

    # PCA and directional statistics
    pca = pca_analysis(points)
    vectors = displacement_vectors(df, id_col, frame_col)
    directional = directional_statistics(vectors)

    # Trajectory descriptors
    phi = turning_angles(df, id_col, frame_col)
    msd = msd_analysis(df, id_col, frame_col, max_lag=args.max_msd_lag)
    straight = straightness_indices(df, id_col, frame_col)
    fractal = fractal_box_counting(points)

    # Uncertainty estimates for manuscript tables
    uncertainty = estimate_descriptor_uncertainties(msd, fractal, straight, n_boot=args.bootstrap)

    # Plots
    print("Generating figures...")
    plot_spatial_summary(points, hull, grid, mc, pca, args.output, unit)
    plot_monte_carlo_area(points, hull, mc, args.output, unit)
    plot_directional_and_trajectory_stats(directional, phi, msd, fractal, args.output, unit)
    plot_density(points, args.output, unit, bins=args.density_bins)

    # Optional spatial information geometry (raw camera pixel coordinates)
    if args.info_geometry:
        print("Running local spatial information-geometry analysis...")
        roi, config_used = load_roi_from_config(args.config)
        fw, fh, frame_source = infer_frame_size(df, roi, args.frame_width, args.frame_height)
        print(f"Information-geometry frame: {fw} x {fh} px ({frame_source})")
        if roi is None:
            print("WARNING: no ROI polygon supplied; the full camera frame is treated as accessible.")
        info = local_turning_angle_grid(
            df, id_col, frame_col, fw, fh,
            nx=args.info_grid_x, ny=args.info_grid_y,
            angle_bins=args.info_angle_bins, min_samples=args.info_min_samples,
            roi=roi, pseudocount=args.info_pseudocount,
            bootstrap_samples=args.info_bootstrap, random_seed=args.info_seed)
        labels = cluster_information_geometry(info, n_clusters=args.info_clusters)
        save_information_geometry_outputs(info, labels, args.output, roi=roi)
        info_summary = {
            "frame_width_px": fw, "frame_height_px": fh, "frame_size_source": frame_source,
            "config": str(config_used) if config_used else None,
            "grid_nx": args.info_grid_x, "grid_ny": args.info_grid_y,
            "angle_bins": args.info_angle_bins, "min_samples": args.info_min_samples,
            "n_valid_cells": int(info["valid_cells"].sum()),
            "n_roi_cells": int(info["roi_valid"].sum()),
            "n_clusters_requested": args.info_clusters,
            "fisher_rao_convention": "2*arccos(sum(sqrt(p*q)))",
        }
        (args.output / "info_geometry_summary.json").write_text(json.dumps(info_summary, indent=2))
        print(f"Information geometry: {info_summary['n_valid_cells']} sufficiently sampled cells")

    # Optional temporal information geometry
    if args.temporal_info_geometry:
        print("Running temporal information-geometry analysis...")
        temporal_info = temporal_turning_angle_information_geometry(
            df, id_col, frame_col, window_frames=args.temporal_window_frames,
            step_frames=args.temporal_step_frames, angle_bins=args.info_angle_bins,
            min_samples=args.temporal_min_samples, pseudocount=args.info_pseudocount,
            fps=args.fps, bootstrap_samples=args.temporal_bootstrap,
            random_seed=args.info_seed)
        save_temporal_information_geometry_outputs(temporal_info, args.output)
        temporal_summary = {
            "window_frames": temporal_info["window_frames"],
            "step_frames": temporal_info["step_frames"],
            "fps": args.fps,
            "angle_bins": args.info_angle_bins,
            "min_samples": args.temporal_min_samples,
            "n_windows": int(len(temporal_info["starts"])),
            "n_valid_windows": int(temporal_info["valid"].sum()),
            "fisher_rao_convention": "2*arccos(sum(sqrt(p*q)))",
        }
        (args.output / "temporal_info_geometry_summary.json").write_text(json.dumps(temporal_summary, indent=2))
        print(f"Temporal information geometry: {temporal_summary['n_valid_windows']} sufficiently sampled windows")

    # Report
    write_report(args.output, args.csv, unit, cm_per_pixel, len(points), hull, grid, mc, pca, directional, phi, msd, straight, fractal, uncertainty)
    print(f"\nAll results saved to: {args.output}")


if __name__ == "__main__":
    main()

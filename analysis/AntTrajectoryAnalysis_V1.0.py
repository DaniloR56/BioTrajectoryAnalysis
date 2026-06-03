#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# AntTrajectoryAnalysis_V1.0.py
# =============================================================================
#
# Statistical Analysis of Biological Trajectories
#
# This program analyses trajectory files produced by AntTracking_V1.0.py.
# It computes geometric, statistical-physics, and trajectory-based
# descriptors of biological motion, with particular application to ant
# foraging trajectories recorded in natural environments.
#
# Developed as part of the project:
#
# "Quantitative Analysis of Biological Trajectories Using Computer Vision
# and Statistical Physics: An Application to Ant Motion"
#
# Main Features
# -------------
# - Import tracked positions from CSV files
# - Convert pixel coordinates to physical units
# - Estimate explored area using convex hulls
# - Estimate area using Monte Carlo sampling
# - Generate spatial occupancy maps
# - Quantify anisotropy using Principal Component Analysis (PCA)
# - Compute directional statistics
# - Compute Mean Squared Displacement (MSD)
# - Compute straightness indices
# - Estimate fractal dimension using box counting
# - Generate publication-quality figures
# - Export analysis reports and summary tables
#
# Input
# -----
# - ant_tracks.csv or ant_positions.csv
# - Optional calibration factor in cm/pixel
#
# Output
# ------
# - analysis_report.txt
# - analysis_results.json
# - spatial_exploration_anisotropy.pdf
# - density_heatmap.pdf
# - trajectory_statistics.pdf
# - monte_carlo_area.pdf
# - publication-ready comparison figures
#
# Dependencies
# ------------
# Python >= 3.10
# NumPy
# Pandas
# SciPy
# Matplotlib
# scikit-learn
#
# Author
# ------
# Danilo Roccatano
# School of Mathematics and Physics
# University of Lincoln
# Lincoln, United Kingdom
#
# Email:
# danilo.roccatano@gmail.com
#
# Version: 1.0
# Created: 2026-05-27
# Last Modified: 2026-06-03
#
# License
# -------
# MIT License
#
# GitHub Repository:
# https://github.com/<USERNAME>/<REPOSITORY>
#
# Citation
# --------
# If you use this software in published work, please cite:
#
# Roccatano, D.
# Quantitative Analysis of Biological Trajectories Using Computer Vision
# and Statistical Physics: An Application to Ant Motion.
#
# =============================================================================

"""
AntTrajectoryAnalysis_V1.0

This script performs statistical analysis of biological trajectories
extracted from video recordings. It was developed for the analysis of ant
motion, but the methods are general and may be applied to other organisms
exhibiting approximately planar motion, including protozoa, rotifers,
nematodes, and other small animals.

The script computes complementary descriptors of trajectory organization:

    - Explored area
    - Convex hull area
    - Monte Carlo area estimation
    - Spatial occupancy maps
    - PCA anisotropy
    - Directional statistics
    - Mean Squared Displacement
    - Straightness index
    - Fractal dimension

Example
-------
python AntTrajectoryAnalysis_V1.0.py ant_tracks.csv --output analysis_results

python AntTrajectoryAnalysis_V1.0.py ant_tracks.csv \
       --cm-per-pixel 0.02 \
       --output ant_trajectory_biophysics_analysis

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

SCRIPT_VERSION = "biophysics_reframed_2026_05_27"


# -----------------------------------------------------------------------------
# Data loading and coordinate handling
# -----------------------------------------------------------------------------

def load_positions(csv_path: Path, cm_per_pixel: Optional[float] = None) -> Tuple[pd.DataFrame, str, Optional[float]]:
    """Load tracking CSV and return columns x_analysis, y_analysis in a physical or pixel unit."""
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
# Reporting
# -----------------------------------------------------------------------------

def write_report(outdir: Path, csv_path: Path, unit: str, cm_per_pixel: Optional[float], n_points: int,
                 hull: Dict, grid: Dict, mc: Dict, pca: Dict, directional: Dict,
                 phi: np.ndarray, msd: Dict, straight: np.ndarray, fractal: Dict) -> None:
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
    report.append(f"Straightness tracks analysed: {len(straight):,}")
    if len(straight):
        report.append(f"Mean straightness: {np.mean(straight):.4f}")
        report.append(f"Median straightness: {np.median(straight):.4f}")
    report.append(f"Fractal dimension D: {fractal['D']:.4f}")
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
    }
    (outdir / "analysis_results.json").write_text(json.dumps(data, indent=2))


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

    # Plots
    print("Generating figures...")
    plot_spatial_summary(points, hull, grid, mc, pca, args.output, unit)
    plot_monte_carlo_area(points, hull, mc, args.output, unit)
    plot_directional_and_trajectory_stats(directional, phi, msd, fractal, args.output, unit)
    plot_density(points, args.output, unit, bins=args.density_bins)

    # Report
    write_report(args.output, args.csv, unit, cm_per_pixel, len(points), hull, grid, mc, pca, directional, phi, msd, straight, fractal)
    print(f"\nAll results saved to: {args.output}")


if __name__ == "__main__":
    main()

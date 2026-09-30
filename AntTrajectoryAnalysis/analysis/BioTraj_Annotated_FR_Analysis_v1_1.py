#!/usr/bin/env python3
"""
BioTraj Annotated Fisher-Rao Analysis
=====================================

Generic annotation-aware analysis for BioTraj spatial information geometry.

Combines:
  * BioTraj info_geometry_cells.csv
  * BioTraj scene-annotation JSON (format BioTraj_scene_annotations)
  * optional trajectory CSV
  * optional first-frame image

Supported annotations
---------------------
polygon:
    surface, biological, physical_constraint, artifact,
    observation_boundary, other

polyline:
    surface_transition, physical_constraint, observation_boundary, other

circle:
    biological, artifact, physical_constraint, other

Annotations with:
    "exclude_from_tracking_analysis": true
are used as quality-control (QC) exclusion regions.

For polygons, cells are assigned by their centre. For exclusion polygons/circles,
the default QC rule is conservative: a cell is excluded if its centre lies
inside the annotation OR the annotation intersects the cell rectangle.

Outputs
-------
annotated_FR_cells.csv
surface_summary_raw.csv
surface_summary_qc.csv
transition_summary_raw.csv
transition_summary_qc.csv
transition_distance_binned_raw.csv
transition_distance_binned_qc.csv
annotation_summary.csv
analysis_metadata.json
annotated_FR_overlay.{png,pdf}                 [if --image supplied]
surface_FR_boxplot_raw.{png,pdf}
surface_FR_boxplot_qc.{png,pdf}
FR_vs_transition_distance_raw.{png,pdf}
FR_vs_transition_distance_qc.{png,pdf}

Important
---------
This script is descriptive. It deliberately does NOT calculate p-values by
treating spatial cells as independent biological replicates.

Example (WALL)
--------------
python3 BioTraj_Annotated_FR_Analysis.py \
    --annotations FirstFrame_scene_annotations_corrected.json \
    --info-cells BioTraj_V1_2_revision2/info_geometry_cells.csv \
    --tracks ant_tracks.csv \
    --image FirstFrame.jpg \
    --grid-x 20 \
    --grid-y 12 \
    --output WALL_annotated_FR

The script assumes annotation and trajectory coordinates refer to the same
original image-pixel coordinate system.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle, Rectangle


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def point_segment_distance(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def distance_to_polyline(x, y, pts, closed=False):
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.nan
    if len(pts) == 1:
        return math.hypot(x - pts[0, 0], y - pts[0, 1])
    pairs = [(i, i + 1) for i in range(len(pts) - 1)]
    if closed:
        pairs.append((len(pts) - 1, 0))
    return min(
        point_segment_distance(x, y, *pts[i], *pts[j])
        for i, j in pairs
    )


def point_in_polygon(x, y, pts):
    pts = np.asarray(pts, dtype=float)
    return bool(MplPath(pts).contains_point((x, y), radius=1e-9))


def point_in_circle(x, y, center, radius):
    cx, cy = center
    return math.hypot(x - cx, y - cy) <= radius


def distance_to_circle_edge(x, y, center, radius):
    cx, cy = center
    return abs(math.hypot(x - cx, y - cy) - radius)


def orientation(a, b, c, eps=1e-9):
    v = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(v) < eps:
        return 0
    return 1 if v > 0 else 2


def on_segment(a, b, c, eps=1e-9):
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
    )


def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def rect_corners(cx, cy, cw, ch):
    x0, x1 = cx - cw / 2, cx + cw / 2
    y0, y1 = cy - ch / 2, cy + ch / 2
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def polygon_intersects_rect(poly, cx, cy, cw, ch):
    """Conservative polygon/cell-rectangle overlap test without Shapely."""
    poly = np.asarray(poly, dtype=float)
    rect = rect_corners(cx, cy, cw, ch)

    # Any rectangle corner in polygon?
    path = MplPath(poly)
    if any(path.contains_point(p, radius=1e-9) for p in rect):
        return True

    # Any polygon vertex in rectangle?
    x0, x1 = cx - cw / 2, cx + cw / 2
    y0, y1 = cy - ch / 2, cy + ch / 2
    if any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in poly):
        return True

    # Edge intersections.
    rect_edges = list(zip(rect, rect[1:] + rect[:1]))
    poly_list = [tuple(p) for p in poly]
    poly_edges = list(zip(poly_list, poly_list[1:] + poly_list[:1]))
    for a, b in rect_edges:
        for c, d in poly_edges:
            if segments_intersect(a, b, c, d):
                return True
    return False


def circle_intersects_rect(center, radius, cx, cy, cw, ch):
    """Circle/axis-aligned rectangle overlap."""
    ccx, ccy = center
    x0, x1 = cx - cw / 2, cx + cw / 2
    y0, y1 = cy - ch / 2, cy + ch / 2
    qx = min(max(ccx, x0), x1)
    qy = min(max(ccy, y0), y1)
    return (ccx - qx) ** 2 + (ccy - qy) ** 2 <= radius ** 2


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def safe_name(s):
    s = str(s).strip().lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() or ch in "_-" else "_")
    return "".join(out).strip("_") or "unnamed"


def load_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "BioTraj_scene_annotations":
        raise ValueError(
            "Annotation JSON is not format 'BioTraj_scene_annotations'. "
            "Use the generic scene annotator JSON."
        )
    if "annotations" not in data:
        raise ValueError("Annotation JSON has no 'annotations' array.")
    return data


def get_circle_geometry(ann):
    # Support several reasonable JSON encodings.
    if "center" in ann:
        center = ann["center"]
    elif "center_px" in ann:
        center = ann["center_px"]
    elif "points" in ann and len(ann["points"]) >= 1:
        center = ann["points"][0]
    else:
        raise ValueError(f"Circle annotation {ann.get('id')} has no centre.")

    if "radius" in ann:
        radius = float(ann["radius"])
    elif "radius_px" in ann:
        radius = float(ann["radius_px"])
    elif "points" in ann and len(ann["points"]) >= 2:
        p = ann["points"][1]
        radius = math.hypot(float(p[0]) - float(center[0]),
                            float(p[1]) - float(center[1]))
    else:
        raise ValueError(f"Circle annotation {ann.get('id')} has no radius.")
    return np.asarray(center, dtype=float), radius


def infer_grid(cells, tracks=None, ann=None, grid_x=None, grid_y=None):
    """Determine the FULL BioTraj grid, not merely the populated grid rows.

    Sparse info_geometry_cells.csv files contain only cells that pass the
    minimum-sampling criterion. Therefore max(grid_y)+1 can underestimate the
    requested grid size. Explicit --grid-x/--grid-y values take precedence.
    """
    if not {"grid_x", "grid_y"}.issubset(cells.columns):
        raise ValueError("info_geometry_cells.csv must contain grid_x and grid_y.")

    observed_gx = int(cells["grid_x"].max()) + 1
    observed_gy = int(cells["grid_y"].max()) + 1

    gx = int(grid_x) if grid_x is not None else observed_gx
    gy = int(grid_y) if grid_y is not None else observed_gy

    if gx < observed_gx or gy < observed_gy:
        raise ValueError(
            f"Requested grid {gx}x{gy} is smaller than populated cell indices "
            f"requiring at least {observed_gx}x{observed_gy}."
        )

    if tracks is not None and {"x", "y"}.issubset(tracks.columns):
        fw = int(np.floor(tracks["x"].max())) + 1
        fh = int(np.floor(tracks["y"].max())) + 1
        source = "trajectory coordinate extrema (BioTraj revision2 convention)"
    elif ann is not None and "image_width" in ann and "image_height" in ann:
        fw = int(ann["image_width"])
        fh = int(ann["image_height"])
        source = "annotation image dimensions"
    else:
        raise ValueError(
            "Cannot infer image/grid geometry: provide --tracks or annotation "
            "image_width/image_height."
        )

    cw = fw / gx
    ch = fh / gy
    return gx, gy, fw, fh, cw, ch, source, observed_gx, observed_gy


def descriptive_summary(values):
    v = pd.Series(values).dropna()
    if len(v) == 0:
        return dict(
            n_cells=0, mean_FR_gradient=np.nan, median_FR_gradient=np.nan,
            sd=np.nan, q25=np.nan, q75=np.nan, minimum=np.nan, maximum=np.nan
        )
    return dict(
        n_cells=int(len(v)),
        mean_FR_gradient=float(v.mean()),
        median_FR_gradient=float(v.median()),
        sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
        q25=float(v.quantile(.25)),
        q75=float(v.quantile(.75)),
        minimum=float(v.min()),
        maximum=float(v.max()),
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ann = load_annotations(args.annotations)
    cells = pd.read_csv(args.info_cells)

    if "local_gradient" not in cells.columns:
        raise ValueError("info_geometry_cells.csv has no 'local_gradient' column.")

    tracks = pd.read_csv(args.tracks) if args.tracks else None

    gx, gy, fw, fh, cw, ch, geom_source, observed_gx, observed_gy = infer_grid(
        cells, tracks, ann, args.grid_x, args.grid_y
    )

    cells = cells.copy()
    cells["x_center"] = (cells["grid_x"].astype(float) + 0.5) * cw
    cells["y_center"] = (cells["grid_y"].astype(float) + 0.5) * ch

    # Annotation/image dimensions must be compatible with trajectory coordinates.
    ann_w = ann.get("image_width")
    ann_h = ann.get("image_height")
    if ann_w is not None and ann_h is not None:
        if fw > float(ann_w) + 2 or fh > float(ann_h) + 2:
            raise ValueError(
                f"BioTraj inferred domain {fw}x{fh} exceeds annotation image "
                f"{ann_w}x{ann_h}; refusing silent coordinate rescale."
            )

    if args.image:
        img = plt.imread(args.image)
        ih, iw = img.shape[:2]
        if ann_w is not None and ann_h is not None and (iw, ih) != (ann_w, ann_h):
            raise ValueError(
                f"Image is {iw}x{ih}, annotation is {ann_w}x{ann_h}; "
                "refusing silent rescale."
            )
    else:
        img = None

    centers = cells[["x_center", "y_center"]].to_numpy(float)

    # Per-cell annotation membership/distance columns.
    surface_annotations = []
    transition_annotations = []
    exclusion_annotations = []
    annotation_rows = []

    for a in ann["annotations"]:
        aid = a.get("id", "")
        label = safe_name(a.get("label", f"annotation_{aid}"))
        category = a.get("category", "other")
        shape = a.get("shape", "")
        exclude = bool(a.get("exclude_from_tracking_analysis", False))

        row = dict(
            annotation_id=aid, label=label, category=category,
            shape=shape, exclude_from_tracking_analysis=exclude
        )

        if shape == "polygon":
            pts = np.asarray(a.get("points", []), dtype=float)
            if len(pts) < 3:
                raise ValueError(f"Polygon annotation {aid}/{label} has <3 points.")
            path = MplPath(pts)
            inside = path.contains_points(centers, radius=1e-9)
            cells[f"inside__{label}"] = inside
            cells[f"distance__{label}_boundary_px"] = [
                distance_to_polyline(x, y, pts, closed=True) for x, y in centers
            ]
            row["n_cell_centres_inside"] = int(inside.sum())

            if category == "surface":
                surface_annotations.append((label, a, inside))
            if exclude:
                exclusion_annotations.append((label, a))

        elif shape == "polyline":
            pts = np.asarray(a.get("points", []), dtype=float)
            if len(pts) < 2:
                raise ValueError(f"Polyline annotation {aid}/{label} has <2 points.")
            dist = np.asarray([
                distance_to_polyline(x, y, pts, closed=False) for x, y in centers
            ])
            cells[f"distance__{label}_px"] = dist
            row["minimum_cell_distance_px"] = float(np.nanmin(dist))
            if category == "surface_transition":
                transition_annotations.append((label, a, dist))
            if exclude:
                exclusion_annotations.append((label, a))

        elif shape == "circle":
            center, radius = get_circle_geometry(a)
            inside = np.asarray([
                point_in_circle(x, y, center, radius) for x, y in centers
            ])
            cells[f"inside__{label}"] = inside
            cells[f"distance__{label}_edge_px"] = [
                distance_to_circle_edge(x, y, center, radius) for x, y in centers
            ]
            row["n_cell_centres_inside"] = int(inside.sum())
            row["radius_px"] = float(radius)
            if exclude:
                exclusion_annotations.append((label, a))

        else:
            raise ValueError(
                f"Unsupported shape '{shape}' for annotation {aid}/{label}. "
                "Supported: polygon, polyline, circle."
            )

        annotation_rows.append(row)

    # Surface assignment. A centre may belong to >1 polygon; report ambiguity.
    surface_labels = [x[0] for x in surface_annotations]
    if surface_annotations:
        membership = np.column_stack([x[2] for x in surface_annotations])
        counts = membership.sum(axis=1)
        assigned = []
        memberships_text = []
        for i in range(len(cells)):
            labs = [surface_labels[j] for j in np.flatnonzero(membership[i])]
            memberships_text.append(";".join(labs))
            if len(labs) == 0:
                assigned.append("unassigned")
            elif len(labs) == 1:
                assigned.append(labs[0])
            else:
                assigned.append("ambiguous")
        cells["surface_memberships"] = memberships_text
        cells["surface"] = assigned
        cells["surface_membership_count"] = counts
    else:
        cells["surface_memberships"] = ""
        cells["surface"] = "unassigned"
        cells["surface_membership_count"] = 0

    # Nearest transition.
    if transition_annotations:
        dmat = np.column_stack([x[2] for x in transition_annotations])
        j = np.nanargmin(dmat, axis=1)
        cells["nearest_transition"] = [
            transition_annotations[k][0] for k in j
        ]
        cells["distance_to_nearest_transition_px"] = dmat[np.arange(len(cells)), j]
    else:
        cells["nearest_transition"] = ""
        cells["distance_to_nearest_transition_px"] = np.nan

    # QC exclusions.
    cells["qc_excluded"] = False
    cells["qc_exclusion_reasons"] = ""

    reasons = [[] for _ in range(len(cells))]
    for label, a in exclusion_annotations:
        shape = a["shape"]
        hit = np.zeros(len(cells), dtype=bool)

        if shape == "polygon":
            pts = np.asarray(a["points"], dtype=float)
            for i, (x, y) in enumerate(centers):
                hit[i] = polygon_intersects_rect(pts, x, y, cw, ch)

        elif shape == "circle":
            center, radius = get_circle_geometry(a)
            for i, (x, y) in enumerate(centers):
                hit[i] = circle_intersects_rect(center, radius, x, y, cw, ch)

        elif shape == "polyline":
            pts = np.asarray(a["points"], dtype=float)
            # Conservative one-half-cell diagonal proximity rule.
            cutoff = 0.5 * math.hypot(cw, ch)
            hit = np.asarray([
                distance_to_polyline(x, y, pts, closed=False) <= cutoff
                for x, y in centers
            ])

        cells[f"qc_excluded_by__{label}"] = hit
        for i in np.flatnonzero(hit):
            reasons[i].append(label)

    if exclusion_annotations:
        cells["qc_excluded"] = np.asarray([len(r) > 0 for r in reasons])
        cells["qc_exclusion_reasons"] = [";".join(r) for r in reasons]

    cells["qc_retained"] = ~cells["qc_excluded"]

    # Save cell-level table before summaries.
    cells.to_csv(out / "annotated_FR_cells.csv", index=False)
    pd.DataFrame(annotation_rows).to_csv(out / "annotation_summary.csv", index=False)

    # Surface summaries.
    def make_surface_summary(df, suffix):
        rows = []
        for label in surface_labels:
            mask = df["surface"] == label
            row = {"surface": label}
            row.update(descriptive_summary(df.loc[mask, "local_gradient"]))
            rows.append(row)
        # Explicitly preserve unassigned/ambiguous information.
        for label in ("unassigned", "ambiguous"):
            if (df["surface"] == label).any():
                row = {"surface": label}
                row.update(descriptive_summary(
                    df.loc[df["surface"] == label, "local_gradient"]
                ))
                rows.append(row)
        tab = pd.DataFrame(rows)
        tab.to_csv(out / f"surface_summary_{suffix}.csv", index=False)
        return tab

    raw_surface = make_surface_summary(cells, "raw")
    qc = cells.loc[cells["qc_retained"]].copy()
    qc_surface = make_surface_summary(qc, "qc")

    # Transition summaries and binned distances.
    band = max(cw, ch)

    def transition_outputs(df, suffix):
        srows = []
        brows = []
        for label, a, _dist_original in transition_annotations:
            col = f"distance__{label}_px"
            if col not in df:
                continue
            dist = df[col]
            for multiplier, band_name in [
                (1.0, "within_1_cell"),
                (2.0, "within_2_cells"),
            ]:
                mask = dist <= multiplier * band
                row = {
                    "transition": label,
                    "band": band_name,
                    "distance_cutoff_px": multiplier * band,
                }
                row.update(descriptive_summary(df.loc[mask, "local_gradient"]))
                srows.append(row)

            # One-cell-width bins, only over occupied range.
            finite = df[[col, "local_gradient"]].dropna()
            if len(finite):
                maxd = float(finite[col].max())
                edges = np.arange(0.0, maxd + band, band)
                if len(edges) < 2:
                    edges = np.array([0.0, band])
                z = finite.copy()
                z["distance_bin"] = pd.cut(
                    z[col], bins=edges, include_lowest=True
                )
                for b, g in z.groupby("distance_bin", observed=True):
                    v = g["local_gradient"].dropna()
                    if len(v):
                        brows.append({
                            "transition": label,
                            "distance_mid_px": float((b.left + b.right) / 2),
                            "n_cells": int(len(v)),
                            "mean_FR_gradient": float(v.mean()),
                            "median_FR_gradient": float(v.median()),
                            "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                            "q25": float(v.quantile(.25)),
                            "q75": float(v.quantile(.75)),
                        })

        st = pd.DataFrame(srows)
        bt = pd.DataFrame(brows)
        st.to_csv(out / f"transition_summary_{suffix}.csv", index=False)
        bt.to_csv(out / f"transition_distance_binned_{suffix}.csv", index=False)
        return st, bt

    raw_transition, raw_bins = transition_outputs(cells, "raw")
    qc_transition, qc_bins = transition_outputs(qc, "qc")

    # Metadata.
    metadata = {
        "annotations_file": str(Path(args.annotations)),
        "info_cells_file": str(Path(args.info_cells)),
        "tracks_file": str(Path(args.tracks)) if args.tracks else None,
        "image_file": str(Path(args.image)) if args.image else None,
        "grid_x": gx,
        "grid_y": gy,
        "observed_populated_grid_extent_x": observed_gx,
        "observed_populated_grid_extent_y": observed_gy,
        "inferred_ig_frame_width_px": fw,
        "inferred_ig_frame_height_px": fh,
        "grid_cell_width_px": cw,
        "grid_cell_height_px": ch,
        "one_cell_analysis_band_px": band,
        "geometry_source": geom_source,
        "n_info_geometry_cells": int(len(cells)),
        "n_cells_with_defined_local_gradient": int(cells["local_gradient"].notna().sum()),
        "n_qc_excluded": int(cells["qc_excluded"].sum()),
        "n_qc_retained": int(cells["qc_retained"].sum()),
        "surface_labels": surface_labels,
        "transition_labels": [x[0] for x in transition_annotations],
        "exclusion_labels": [x[0] for x in exclusion_annotations],
        "note": (
            "Descriptive spatial-cell analysis only. No inferential p-values are "
            "computed because information-geometry cells are spatially dependent "
            "and are not independent biological replicates."
        ),
    }
    (out / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------

    # Overlay.
    if img is not None:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(img)

        if tracks is not None and {"x", "y"}.issubset(tracks.columns):
            step = max(1, len(tracks) // 50000)
            ax.scatter(
                tracks["x"].iloc[::step], tracks["y"].iloc[::step],
                s=0.4, alpha=0.10
            )

        # Annotation geometry.
        for a in ann["annotations"]:
            label = safe_name(a.get("label", f"annotation_{a.get('id','')}"))
            shape = a["shape"]
            exclude = bool(a.get("exclude_from_tracking_analysis", False))
            lw = 2.2 if a.get("category") == "surface_transition" else 1.5
            ls = "--" if exclude else "-"

            if shape == "polygon":
                pts = np.asarray(a["points"], float)
                ax.add_patch(MplPolygon(
                    pts, closed=True, fill=False, lw=lw, ls=ls
                ))
                c = pts.mean(axis=0)
                ax.text(c[0], c[1], label, fontsize=8,
                        bbox=dict(facecolor="white", alpha=.65, edgecolor="none"))

            elif shape == "polyline":
                pts = np.asarray(a["points"], float)
                ax.plot(pts[:, 0], pts[:, 1], lw=lw, ls=ls)
                c = pts.mean(axis=0)
                ax.text(c[0], c[1], label, fontsize=8,
                        bbox=dict(facecolor="white", alpha=.65, edgecolor="none"))

            elif shape == "circle":
                center, radius = get_circle_geometry(a)
                ax.add_patch(MplCircle(
                    center, radius, fill=False, lw=lw, ls=ls
                ))
                ax.text(center[0], center[1], label, fontsize=8,
                        bbox=dict(facecolor="white", alpha=.65, edgecolor="none"))

        # Plot each FR cell as a rectangle in its actual spatial extent.
        valid = cells["local_gradient"].notna()
        vals = cells.loc[valid, "local_gradient"]
        norm = plt.Normalize(vals.min(), vals.max()) if len(vals) else None
        cmap = plt.get_cmap()

        for _, r in cells.loc[valid].iterrows():
            face = cmap(norm(r["local_gradient"])) if norm is not None else "none"
            rect = Rectangle(
                (r["x_center"] - cw / 2, r["y_center"] - ch / 2),
                cw, ch, facecolor=face, edgecolor="none", alpha=.48
            )
            ax.add_patch(rect)

        if len(vals):
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Local Fisher–Rao gradient")

        # Mark QC-excluded cells.
        ex = cells["qc_excluded"]
        if ex.any():
            ax.scatter(
                cells.loc[ex, "x_center"], cells.loc[ex, "y_center"],
                marker="x", s=35, label="QC excluded"
            )
            ax.legend(loc="upper right")

        ax.set_xlim(0, ann_w if ann_w is not None else fw)
        ax.set_ylim(ann_h if ann_h is not None else fh, 0)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_title("BioTraj annotated local Fisher–Rao analysis")
        fig.tight_layout()
        fig.savefig(out / "annotated_FR_overlay.png", dpi=220)
        fig.savefig(out / "annotated_FR_overlay.pdf")
        plt.close(fig)

    def surface_boxplot(df, suffix):
        labels = [s for s in surface_labels if (df["surface"] == s).any()]
        if not labels:
            return
        arrays = [
            df.loc[df["surface"] == s, "local_gradient"].dropna().to_numpy()
            for s in labels
        ]
        arrays_labels = [(a, l) for a, l in zip(arrays, labels) if len(a)]
        if not arrays_labels:
            return
        arrays, labels = zip(*arrays_labels)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(arrays, tick_labels=labels, showfliers=True)
        ax.set_ylabel("Local Fisher–Rao gradient")
        ax.set_xlabel("Annotated surface")
        ax.set_title(f"FR gradient by annotated surface ({suffix})")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out / f"surface_FR_boxplot_{suffix}.png", dpi=220)
        fig.savefig(out / f"surface_FR_boxplot_{suffix}.pdf")
        plt.close(fig)

    surface_boxplot(cells, "raw")
    surface_boxplot(qc, "qc")

    def transition_plot(bins, suffix):
        if bins.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for label in bins["transition"].unique():
            q = bins[bins["transition"] == label].copy()
            # Plot all bins but distinguish low-n values by not connecting
            # through missing bins. CSV retains exact n for manuscript filtering.
            ax.plot(
                q["distance_mid_px"], q["mean_FR_gradient"],
                marker="o", label=label
            )
        ax.set_xlabel("Distance from annotated surface transition (px)")
        ax.set_ylabel("Mean local Fisher–Rao gradient")
        ax.set_title(f"FR gradient versus physical-transition distance ({suffix})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"FR_vs_transition_distance_{suffix}.png", dpi=220)
        fig.savefig(out / f"FR_vs_transition_distance_{suffix}.pdf")
        plt.close(fig)

    transition_plot(raw_bins, "raw")
    transition_plot(qc_bins, "qc")

    # Console report.
    print("BioTraj Annotated Fisher-Rao Analysis")
    print("-------------------------------------")
    print(f"IG grid (full requested): {gx} x {gy}")
    print(f"Populated CSV index extent: {observed_gx} x {observed_gy}")
    print(f"IG inferred frame: {fw} x {fh} px")
    print(f"IG cell size: {cw:.2f} x {ch:.2f} px")
    print(f"Input IG cells: {len(cells)}")
    print(f"Defined local gradients: {cells['local_gradient'].notna().sum()}")
    print(f"Surfaces: {', '.join(surface_labels) if surface_labels else 'none'}")
    print(
        "Transitions: "
        + (", ".join(x[0] for x in transition_annotations)
           if transition_annotations else "none")
    )
    print(
        "QC exclusions: "
        + (", ".join(x[0] for x in exclusion_annotations)
           if exclusion_annotations else "none")
    )
    print(f"QC retained cells: {cells['qc_retained'].sum()} / {len(cells)}")

    if not raw_surface.empty:
        print("\nSURFACE SUMMARY (RAW)")
        print(raw_surface.to_string(index=False))
    if not qc_surface.empty and exclusion_annotations:
        print("\nSURFACE SUMMARY (QC)")
        print(qc_surface.to_string(index=False))

    if not raw_transition.empty:
        print("\nTRANSITION SUMMARY (RAW)")
        print(raw_transition.to_string(index=False))

    print(f"\nSaved to: {out}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Generic scene-annotation analysis for BioTraj local Fisher-Rao maps."
    )
    p.add_argument(
        "--annotations", required=True,
        help="BioTraj scene-annotation JSON."
    )
    p.add_argument(
        "--info-cells", required=True,
        help="BioTraj info_geometry_cells.csv."
    )
    p.add_argument(
        "--tracks", default=None,
        help="Optional ant_tracks.csv; recommended to reproduce BioTraj's inferred frame."
    )
    p.add_argument(
        "--image", default=None,
        help="Optional first-frame image for registration/FR overlay."
    )
    p.add_argument(
        "--grid-x", type=int, default=None,
        help=("Full BioTraj spatial grid size in x. Strongly recommended when "
              "info_geometry_cells.csv is sparse; e.g. 20.")
    )
    p.add_argument(
        "--grid-y", type=int, default=None,
        help=("Full BioTraj spatial grid size in y. Strongly recommended when "
              "info_geometry_cells.csv is sparse; e.g. 12.")
    )
    p.add_argument(
        "--output", default="BioTraj_annotated_FR_analysis",
        help="Output directory."
    )
    return p


if __name__ == "__main__":
    analyse(build_parser().parse_args())

#!/usr/bin/env python3
"""
BioTraj Synthetic Trajectory Generator v1.0
===========================================

Ground-truth simulator for validation of BioTraj spatial and temporal
information-geometry analyses.

Outputs
-------
ant_tracks.csv
    Compatible with BioTraj's trajectory-analysis input.
config.json
    ROI polygon and simulation metadata.
ground_truth.json
    Exact imposed scenario and parameters.
trajectory_preview.png
    Preview of simulated trajectories and imposed spatial boundary.
expected_validation.txt
    Human-readable statement of what BioTraj should recover.

Scenarios
---------
1. Homogeneous stationary:
   Same movement law everywhere and at all times.
2. Spatial two-domain:
   Movement law changes at a known vertical spatial boundary.
3. Temporal two-state:
   Movement law changes at a known frame.
4. Spatiotemporal:
   Spatial domains plus a known temporal change.
5. Structured arena:
   A biologically motivated localization benchmark containing a circular
   nest/entrance zone, a directional corridor, a wall/surface-transition
   boundary, and ordinary observation-frame exits. The semantic geometry is
   saved as ground truth but is not supplied to BioTraj.

Movement model
--------------
theta(t+1) = theta(t) + dtheta
dtheta ~ von Mises(0, kappa)
step length ~ truncated Normal(mean_step, step_sd), constrained positive

Larger kappa -> stronger directional persistence.
Reflecting boundaries keep trajectories inside the rectangular ROI.

This program is deliberately simple: its purpose is controlled validation,
not realistic modelling of ant behaviour.
"""

import csv
import json
import math
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

VERSION = "BioTraj_SyntheticTrajectoryGenerator_v1.2"


def ask_int(prompt, default, minimum=1):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return int(default)
        try:
            v = int(s)
            if v < minimum:
                raise ValueError
            return v
        except ValueError:
            print(f"  Please enter an integer >= {minimum}.")


def ask_float(prompt, default, minimum=None, maximum=None):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return float(default)
        try:
            v = float(s)
            if minimum is not None and v < minimum:
                raise ValueError
            if maximum is not None and v > maximum:
                raise ValueError
            return v
        except ValueError:
            limits = []
            if minimum is not None:
                limits.append(f">= {minimum}")
            if maximum is not None:
                limits.append(f"<= {maximum}")
            print("  Please enter a valid number " + " and ".join(limits) + ".")


def ask_yes_no(prompt, default=True):
    suffix = "Y/n" if default else "y/N"
    s = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not s:
        return default
    return s.startswith("y")


def choose_scenario():
    print("\nGROUND-TRUTH SCENARIO")
    print("  1 - Homogeneous stationary (null test)")
    print("  2 - Spatial two-domain")
    print("  3 - Temporal two-state")
    print("  4 - Spatiotemporal")
    print("  5 - Structured arena (nest + corridor + wall + frame exits)")
    while True:
        s = input("Select scenario [1]: ").strip() or "1"
        if s in {"1", "2", "3", "4", "5"}:
            return int(s)
        print("  Choose 1, 2, 3, 4, or 5.")


def positive_normal(rng, mean, sd, n=1):
    if sd <= 0:
        return np.full(n, max(mean, 1e-9))
    x = rng.normal(mean, sd, n)
    bad = x <= 0
    while np.any(bad):
        x[bad] = rng.normal(mean, sd, np.sum(bad))
        bad = x <= 0
    return x


def reflect_coordinate(pos, heading_component, lo, hi):
    """Reflect one coordinate and corresponding velocity component."""
    while pos < lo or pos > hi:
        if pos < lo:
            pos = lo + (lo - pos)
            heading_component *= -1
        elif pos > hi:
            pos = hi - (pos - hi)
            heading_component *= -1
    return pos, heading_component


def move_one(x, y, theta, step, width, height):
    vx = step * math.cos(theta)
    vy = step * math.sin(theta)
    xn = x + vx
    yn = y + vy

    xn, vx = reflect_coordinate(xn, vx, 0.0, width - 1.0)
    yn, vy = reflect_coordinate(yn, vy, 0.0, height - 1.0)
    theta_new = math.atan2(vy, vx)
    return xn, yn, theta_new


def state_parameters(scenario, frame, x, temporal_transition, spatial_boundary,
                     kappa_a, kappa_b, kappa_c):
    """
    Return (state_name, kappa).

    Scenario 4 uses four logically distinguishable combinations:
      before transition: left=A, right=B
      after transition : left=C, right=B
    This deliberately creates both spatial and temporal structure while
    retaining one reference state (B) across the transition.
    """
    if scenario == 1:
        return "A_stationary", kappa_a

    if scenario == 2:
        if x < spatial_boundary:
            return "A_left", kappa_a
        return "B_right", kappa_b

    if scenario == 3:
        if frame < temporal_transition:
            return "A_before", kappa_a
        return "B_after", kappa_b

    # Scenario 4
    if frame < temporal_transition:
        if x < spatial_boundary:
            return "A_left_before", kappa_a
        return "B_right_reference", kappa_b
    else:
        if x < spatial_boundary:
            return "C_left_after", kappa_c
        return "B_right_reference", kappa_b



def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def structured_region(x, y, params):
    """Return the independently known semantic region for scenario 5."""
    dx = x - params["nest_x"]
    dy = y - params["nest_y"]
    r = math.hypot(dx, dy)
    if r <= params["nest_radius_px"]:
        return "nest_zone"
    if abs(y - params["corridor_y"]) <= 0.5 * params["corridor_width_px"]:
        return "corridor"
    if x >= params["wall_x"]:
        return "wall_top"
    return "open"


def structured_step(rng, x, y, theta, params):
    """
    One movement update for the structured-arena benchmark.

    The biological/geometrical structures are intentionally simple:
      * nest zone: headings are biased radially toward or away from the hole;
      * corridor: headings align with the corridor axis (0 or pi);
      * wall top: the same underlying motion is observed through an anisotropic
        2-D projection, mimicking a change of physical plane;
      * frame boundary: trajectories may leave the field and terminate, with
        NO special behavioural law near the image edge (negative control).
    """
    region = structured_region(x, y, params)
    step = positive_normal(
        rng, params["mean_step_px"], params["step_sd_px"], 1
    )[0]

    if region == "nest_zone":
        radial = math.atan2(params["nest_y"] - y, params["nest_x"] - x)
        # Fixed per-step mixture of entering and emerging movement.
        target = radial if rng.random() < 0.5 else wrap_angle(radial + math.pi)
        theta = target + rng.vonmises(0.0, params["nest_kappa"])
    elif region == "corridor":
        target = 0.0 if math.cos(theta) >= 0 else math.pi
        theta = target + rng.vonmises(0.0, params["corridor_kappa"])
    else:
        theta = theta + rng.vonmises(0.0, params["open_kappa"])

    vx = step * math.cos(theta)
    vy = step * math.sin(theta)

    # On the wall-top side, compress one observed component. This represents
    # a projection/surface-orientation transition rather than a behavioural
    # state change.
    if region == "wall_top":
        vy *= params["wall_projection_factor"]

    return x + vx, y + vy, math.atan2(vy, vx), region


def simulate_structured(params):
    """Generate tracklets in a structured arena; tracks end at frame exits."""
    rng = np.random.default_rng(params["random_seed"])
    width, height = params["frame_width"], params["frame_height"]
    rows, state_counts = [], {}
    next_track_id = 1

    # Re-seed a new tracklet after an exit. This preserves approximately the
    # requested amount of data while keeping exits explicit and never bridging
    # a trajectory across the observation boundary.
    active = []
    for _ in range(params["n_agents"]):
        active.append([
            next_track_id,
            rng.uniform(0.08 * width, 0.92 * width),
            rng.uniform(0.08 * height, 0.92 * height),
            rng.uniform(-math.pi, math.pi),
        ])
        next_track_id += 1

    exit_events = []
    for frame in range(params["n_frames"]):
        for a in range(params["n_agents"]):
            track_id, x, y, theta = active[a]
            region = structured_region(x, y, params)
            state_counts[region] = state_counts.get(region, 0) + 1
            rows.append((track_id, frame, x, y, region, np.nan))

            if frame == params["n_frames"] - 1:
                continue

            xn, yn, theta_n, _ = structured_step(rng, x, y, theta, params)
            if xn < 0 or xn >= width or yn < 0 or yn >= height:
                if xn < 0:
                    edge = "left"
                elif xn >= width:
                    edge = "right"
                elif yn < 0:
                    edge = "top"
                else:
                    edge = "bottom"
                exit_events.append({
                    "track_id": int(track_id), "frame": int(frame + 1),
                    "edge": edge, "x_attempted": float(xn),
                    "y_attempted": float(yn)
                })

                # New independent tracklet enters at a random interior location.
                active[a] = [
                    next_track_id,
                    rng.uniform(0.08 * width, 0.92 * width),
                    rng.uniform(0.08 * height, 0.92 * height),
                    rng.uniform(-math.pi, math.pi),
                ]
                next_track_id += 1
            else:
                active[a] = [track_id, xn, yn, theta_n]

    return rows, state_counts, exit_events

def simulate(params):
    if params["scenario"] == 5:
        rows, state_counts, exit_events = simulate_structured(params)
        params["_exit_events"] = exit_events
        return rows, state_counts

    rng = np.random.default_rng(params["random_seed"])
    n_agents = params["n_agents"]
    n_frames = params["n_frames"]
    width = params["frame_width"]
    height = params["frame_height"]

    x = rng.uniform(0.05 * width, 0.95 * width, n_agents)
    y = rng.uniform(0.05 * height, 0.95 * height, n_agents)
    theta = rng.uniform(-math.pi, math.pi, n_agents)

    rows = []
    state_counts = {}

    for frame in range(n_frames):
        for a in range(n_agents):
            state, kappa = state_parameters(
                params["scenario"], frame, x[a],
                params["temporal_transition_frame"],
                params["spatial_boundary_x"],
                params["kappa_a"], params["kappa_b"], params["kappa_c"]
            )
            state_counts[state] = state_counts.get(state, 0) + 1

            # Save current position in BioTraj-compatible form.
            rows.append((a + 1, frame, x[a], y[a], state, kappa))

            # Update for next frame.
            if frame < n_frames - 1:
                dtheta = rng.vonmises(0.0, kappa)
                theta[a] += dtheta
                step = positive_normal(
                    rng, params["mean_step_px"], params["step_sd_px"], 1
                )[0]
                x[a], y[a], theta[a] = move_one(
                    x[a], y[a], theta[a], step, width, height
                )

    return rows, state_counts


def write_outputs(outdir, rows, params, state_counts):
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "ant_tracks.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        # First four columns are the essential BioTraj-compatible fields.
        # Extra ground-truth columns are ignored by BioTraj but useful for validation.
        w.writerow(["track_id", "frame", "x", "y", "true_state", "true_kappa"])
        w.writerows(rows)

    roi = [
        [0, 0],
        [params["frame_width"] - 1, 0],
        [params["frame_width"] - 1, params["frame_height"] - 1],
        [0, params["frame_height"] - 1],
    ]
    config = {
        "roi": roi,
        "frame_width": params["frame_width"],
        "frame_height": params["frame_height"],
        "fps": params["fps"],
        "synthetic": True,
        "generator_version": VERSION,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    truth = dict(params)
    truth["generator_version"] = VERSION
    truth["total_positions"] = len(rows)
    truth["state_counts"] = state_counts
    (outdir / "ground_truth.json").write_text(json.dumps(truth, indent=2))

    if params["scenario"] == 5:
        semantic = {
            "coordinate_system": "video pixels; origin at top-left",
            "nest": {
                "type": "circle",
                "centre": [params["nest_x"], params["nest_y"]],
                "radius_px": params["nest_radius_px"],
                "interpretation": "synthetic nest/entrance transition zone",
            },
            "corridor": {
                "type": "horizontal_strip",
                "centre_y": params["corridor_y"],
                "width_px": params["corridor_width_px"],
                "interpretation": "directionally constrained movement corridor",
            },
            "wall": {
                "type": "vertical_surface_transition",
                "x": params["wall_x"],
                "projection_factor": params["wall_projection_factor"],
                "interpretation": "change of observed 2-D projection across a surface edge",
            },
            "frame_boundary": {
                "type": "observation_boundary",
                "bounds": [0, 0, params["frame_width"], params["frame_height"]],
                "interpretation": "trajectory exits only; no imposed behavioural change",
            },
        }
        (outdir / "semantic_ground_truth.json").write_text(
            json.dumps(semantic, indent=2)
        )
        (outdir / "exit_events.json").write_text(
            json.dumps(params.get("_exit_events", []), indent=2)
        )

    scenario_names = {
        1: "Homogeneous stationary",
        2: "Spatial two-domain",
        3: "Temporal two-state",
        4: "Spatiotemporal",
        5: "Structured arena",
    }

    expected = [
        f"BioTraj synthetic validation dataset",
        f"Generator: {VERSION}",
        f"Scenario: {scenario_names[params['scenario']]}",
        f"Positions: {len(rows):,}",
        f"Agents: {params['n_agents']}",
        f"Frames: {params['n_frames']}",
        f"FPS: {params['fps']}",
        f"Frame: {params['frame_width']} x {params['frame_height']} px",
        "",
    ]
    if params["scenario"] == 1:
        expected += [
            "EXPECTED RESULT:",
            "Spatial and temporal Fisher--Rao structure should be weak and",
            "consistent with finite-sampling fluctuations. No persistent",
            "spatial boundary or temporal block is imposed.",
        ]
    elif params["scenario"] == 2:
        expected += [
            "EXPECTED RESULT:",
            f"A spatial transition is imposed at x = {params['spatial_boundary_x']:.1f} px.",
            "The spatial Fisher--Rao map/matrix should show structure associated",
            "with this boundary. No temporal transition is imposed.",
        ]
    elif params["scenario"] == 3:
        expected += [
            "EXPECTED RESULT:",
            f"A temporal transition is imposed at frame {params['temporal_transition_frame']}",
            f"({params['temporal_transition_frame']/params['fps']:.2f} s).",
            "The temporal Fisher--Rao matrix should show two temporal blocks.",
            "No spatial behavioural boundary is imposed.",
        ]
    elif params["scenario"] == 4:
        expected += [
            "EXPECTED RESULT:",
            f"Spatial boundary: x = {params['spatial_boundary_x']:.1f} px.",
            f"Temporal transition: frame {params['temporal_transition_frame']}",
            f"({params['temporal_transition_frame']/params['fps']:.2f} s).",
            "Both spatial and temporal information-geometry outputs should show",
            "structured departures from the homogeneous null case.",
        ]
    else:
        expected += [
            "EXPECTED RESULT:",
            "Structured-arena localization benchmark.",
            f"Nest centre/radius: ({params['nest_x']:.1f}, {params['nest_y']:.1f}) / "
            f"{params['nest_radius_px']:.1f} px.",
            f"Corridor: y = {params['corridor_y']:.1f} px, width "
            f"{params['corridor_width_px']:.1f} px.",
            f"Wall/surface transition: x = {params['wall_x']:.1f} px.",
            "The local Fisher--Rao field should localize movement-statistical",
            "transitions associated with the nest, corridor, and wall geometry.",
            "Frame exits have no imposed behavioural change and therefore serve",
            "as an observation-boundary negative control.",
        ]
    (outdir / "expected_validation.txt").write_text("\n".join(expected) + "\n")

    # Preview: thin a little for very large simulations, but preserve all agents.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    by_agent = {}
    for track_id, frame, x, y, state, kappa in rows:
        by_agent.setdefault(track_id, [[], []])
        by_agent[track_id][0].append(x)
        by_agent[track_id][1].append(y)

    stride = max(1, params["n_frames"] // 2500)
    for track_id, (xs, ys) in by_agent.items():
        ax.plot(xs[::stride], ys[::stride], lw=0.55, alpha=0.55)

    if params["scenario"] in (2, 4):
        ax.axvline(params["spatial_boundary_x"], ls="--", lw=2,
                   label="Ground-truth spatial boundary")
        ax.legend(loc="best")
    elif params["scenario"] == 5:
        from matplotlib.patches import Circle, Rectangle
        ax.add_patch(Circle(
            (params["nest_x"], params["nest_y"]),
            params["nest_radius_px"], fill=False, lw=2, ls="--",
            label="Nest/entrance zone"
        ))
        ax.add_patch(Rectangle(
            (0, params["corridor_y"] - 0.5 * params["corridor_width_px"]),
            params["frame_width"], params["corridor_width_px"],
            fill=False, lw=1.5, ls=":", label="Corridor"
        ))
        ax.axvline(params["wall_x"], lw=2, ls="-.",
                   label="Wall/surface boundary")
        ax.legend(loc="best", fontsize=8)

    ax.set_xlim(0, params["frame_width"])
    ax.set_ylim(params["frame_height"], 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"Synthetic trajectories — {scenario_names[params['scenario']]}")
    fig.tight_layout()
    fig.savefig(outdir / "trajectory_preview.png", dpi=180)
    plt.close(fig)

    return csv_path


def interactive_main():
    print("=" * 78)
    print("BIOTRAJ SYNTHETIC TRAJECTORY GENERATOR")
    print("Ground-truth validation of spatial and temporal information geometry")
    print("=" * 78)
    print(f"Version: {VERSION}")

    scenario = choose_scenario()

    print("\nSAMPLING")
    n_agents = ask_int("Number of simulated agents", 20)
    fps = ask_float("Frame rate (fps)", 60.0, minimum=0.001)

    if ask_yes_no("Match an experimental target number of positions?", True):
        target = ask_int("Target total positions", 242318)
        n_frames = max(3, int(round(target / n_agents)))
        print(f"  -> Using {n_frames:,} frames gives approximately "
              f"{n_frames*n_agents:,} positions.")
    else:
        n_frames = ask_int("Number of frames", 12000, minimum=3)

    print("\nARENA")
    width = ask_int("Frame width (px)", 1905)
    height = ask_int("Frame height (px)", 726)

    print("\nMOVEMENT")
    mean_step = ask_float("Mean step per frame (px)", 2.5, minimum=0.001)
    step_sd = ask_float("Step-length SD (px)", 0.5, minimum=0.0)
    print("Turning persistence uses a von Mises distribution.")
    print("  kappa ~ 0: random turning; larger kappa: more persistent motion.")
    kappa_a = ask_float("State A kappa", 0.5, minimum=0.0)
    kappa_b = ask_float("State B kappa", 5.0, minimum=0.0)
    kappa_c = ask_float("State C kappa (spatiotemporal case)", 12.0, minimum=0.0)

    spatial_boundary = 0.5 * width
    if scenario in (2, 4):
        spatial_boundary = ask_float(
            "Spatial boundary x (px)", 0.5 * width, minimum=1, maximum=width - 2
        )

    temporal_transition = n_frames // 2
    if scenario in (3, 4):
        temporal_transition = ask_int(
            "Temporal transition frame", n_frames // 2, minimum=1
        )
        temporal_transition = min(temporal_transition, n_frames - 1)

    seed = ask_int("Random seed", 12345, minimum=0)
    default_out = {
        1: "synthetic_stationary",
        2: "synthetic_spatial",
        3: "synthetic_temporal",
        4: "synthetic_spatiotemporal",
        5: "synthetic_structured_arena",
    }[scenario]
    outname = input(f"Output directory [{default_out}]: ").strip() or default_out

    params = {
        "scenario": scenario,
        "n_agents": n_agents,
        "n_frames": n_frames,
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "mean_step_px": mean_step,
        "step_sd_px": step_sd,
        "kappa_a": kappa_a,
        "kappa_b": kappa_b,
        "kappa_c": kappa_c,
        "spatial_boundary_x": spatial_boundary,
        "temporal_transition_frame": temporal_transition,
        "random_seed": seed,
        # Structured-arena defaults; ignored by scenarios 1--4.
        "nest_x": 0.23 * width,
        "nest_y": 0.30 * height,
        "nest_radius_px": 0.055 * min(width, height),
        "nest_kappa": 8.0,
        "corridor_y": 0.68 * height,
        "corridor_width_px": 0.12 * height,
        "corridor_kappa": 10.0,
        "wall_x": 0.72 * width,
        "wall_projection_factor": 0.35,
        "open_kappa": kappa_a,
    }

    print("\nSimulating...")
    rows, state_counts = simulate(params)
    csv_path = write_outputs(Path(outname), rows, params, state_counts)

    print("\n" + "=" * 78)
    print("SIMULATION COMPLETE")
    print("=" * 78)
    print(f"Scenario: {scenario}")
    print(f"Agents: {n_agents:,}")
    print(f"Frames: {n_frames:,}")
    print(f"Positions: {len(rows):,}")
    print(f"Output: {Path(outname).resolve()}")
    print("\nFiles:")
    print("  ant_tracks.csv             -> BioTraj input")
    print("  config.json                -> rectangular ROI / metadata")
    print("  ground_truth.json          -> exact imposed parameters")
    print("  expected_validation.txt    -> expected BioTraj result")
    print("  trajectory_preview.png     -> simulation preview")
    print("\nSuggested BioTraj command:")
    print(
        f"python AntTrajectoryAnalysis_V1.2_info_geometry_revision2.py "
        f"{csv_path} --config {Path(outname)/'config.json'} "
        f"--info-geometry --temporal-info-geometry --fps {fps:g} "
        f"--output {outname}_analysis"
    )



SCENARIO_DIRS = {
    1: "synthetic_stationary",
    2: "synthetic_spatial",
    3: "synthetic_temporal",
    4: "synthetic_spatiotemporal",
    5: "synthetic_structured_arena",
}

SCENARIO_NAMES = {
    1: "Homogeneous stationary",
    2: "Spatial two-domain",
    3: "Temporal two-state",
    4: "Spatiotemporal",
    5: "Structured arena",
}


def default_batch_params(scenario, seed=12345):
    """
    Reproduce the defaults used by the interactive v1.0 generator.

    With 20 agents and target ~242318 positions, n_frames=12116,
    giving 242320 positions. The default temporal transition is the
    midpoint (frame 6058 = 100.97 s at 60 fps) unless overridden.
    """
    n_agents = 20
    target_positions = 242318
    n_frames = max(3, int(round(target_positions / n_agents)))
    width = 1905
    height = 726
    fps = 60.0

    return {
        "scenario": scenario,
        "n_agents": n_agents,
        "n_frames": n_frames,
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "mean_step_px": 2.5,
        "step_sd_px": 0.5,
        "kappa_a": 0.5,
        "kappa_b": 5.0,
        "kappa_c": 12.0,
        "spatial_boundary_x": 0.5 * width,
        "temporal_transition_frame": n_frames // 2,
        "random_seed": int(seed),
        "nest_x": 0.23 * width,
        "nest_y": 0.30 * height,
        "nest_radius_px": 0.055 * min(width, height),
        "nest_kappa": 8.0,
        "corridor_y": 0.68 * height,
        "corridor_width_px": 0.12 * height,
        "corridor_kappa": 10.0,
        "wall_x": 0.72 * width,
        "wall_projection_factor": 0.35,
        "open_kappa": 0.5,
    }


def run_biotraj_analysis(outdir, analysis_script, python_executable=sys.executable):
    """Run the requested BioTraj analysis command inside one scenario folder."""
    outdir = Path(outdir)
    analysis_script = Path(analysis_script).expanduser().resolve()

    cmd = [
        python_executable,
        str(analysis_script),
        "ant_tracks.csv",
        "--config", "config.json",
        "--info-geometry",
        "--info-grid-x", "20",
        "--info-grid-y", "12",
        "--info-min-samples", "30",
        "--info-clusters", "4",
        "--info-bootstrap", "200",
        "--temporal-info-geometry",
        "--fps", "60",
        "--temporal-window-frames", "900",
        "--temporal-step-frames", "900",
        "--temporal-min-samples", "100",
        "--temporal-bootstrap", "200",
        "--output", "BioTraj_V1_2_revision2",
    ]

    log_path = outdir / "BioTraj_analysis.log"
    print("  Running BioTraj analysis...")
    print("  " + " ".join(cmd))
    with log_path.open("w") as log:
        result = subprocess.run(
            cmd,
            cwd=outdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode == 0:
        print(f"  Analysis complete -> {outdir/'BioTraj_V1_2_revision2'}")
    else:
        print(f"  WARNING: analysis failed (exit code {result.returncode}).")
        print(f"  See {log_path}")
    return result.returncode


def batch_main(args):
    base = Path(args.base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    analysis_script = Path(args.analysis_script).expanduser()
    if not analysis_script.is_absolute():
        analysis_script = (Path.cwd() / analysis_script).resolve()

    if not analysis_script.exists() and not args.generate_only:
        raise FileNotFoundError(
            f"BioTraj analysis script not found: {analysis_script}\n"
            "Use --analysis-script to give its path, or --generate-only."
        )

    summary_rows = []

    for scenario in (1, 2, 3, 4, 5):
        outdir = base / SCENARIO_DIRS[scenario]
        params = default_batch_params(scenario, seed=args.seed)

        if args.temporal_transition_frame is not None:
            params["temporal_transition_frame"] = min(
                max(1, args.temporal_transition_frame),
                params["n_frames"] - 1,
            )
        if args.spatial_boundary_x is not None:
            params["spatial_boundary_x"] = float(args.spatial_boundary_x)

        print("\n" + "=" * 78)
        print(f"SCENARIO {scenario}: {SCENARIO_NAMES[scenario]}")
        print("=" * 78)
        print(f"Generating -> {outdir}")

        rows, state_counts = simulate(params)
        write_outputs(outdir, rows, params, state_counts)

        rc = None
        if not args.generate_only:
            rc = run_biotraj_analysis(
                outdir,
                analysis_script,
                python_executable=args.python,
            )

        summary_rows.append({
            "scenario": scenario,
            "scenario_name": SCENARIO_NAMES[scenario],
            "directory": str(outdir),
            "random_seed": params["random_seed"],
            "n_agents": params["n_agents"],
            "n_frames": params["n_frames"],
            "total_positions": len(rows),
            "spatial_boundary_x_px": (
                params["spatial_boundary_x"] if scenario in (2, 4) else ""
            ),
            "temporal_transition_frame": (
                params["temporal_transition_frame"] if scenario in (3, 4) else ""
            ),
            "temporal_transition_s": (
                params["temporal_transition_frame"] / params["fps"]
                if scenario in (3, 4) else ""
            ),
            "analysis_return_code": "" if rc is None else rc,
        })

    summary_path = base / "validation_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 78)
    print("FIVE-SCENARIO VALIDATION COMPLETE")
    print("=" * 78)
    print(f"Summary: {summary_path}")
    for row in summary_rows:
        status = "generated only" if args.generate_only else (
            "OK" if row["analysis_return_code"] == 0 else "FAILED"
        )
        print(f"  {row['scenario_name']:<24} {status}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate BioTraj synthetic validation data. With --all, generate "
            "all five scenarios and optionally run BioTraj automatically."
        )
    )
    p.add_argument(
        "--all", action="store_true",
        help="Generate all four validation scenarios non-interactively."
    )
    p.add_argument(
        "--base-dir", default=".",
        help="Parent directory for the five synthetic_* folders (default: current directory)."
    )
    p.add_argument(
        "--analysis-script",
        default="../../BioTrajectoryAnalysis/AntTrajectoryAnalysis/analysis/"
                "AntTrajectoryAnalysis_V1.2_info_geometry_revision2.py",
        help="Path to the BioTraj analysis script."
    )
    p.add_argument(
        "--python", default=sys.executable,
        help="Python executable used to launch BioTraj (default: current Python)."
    )
    p.add_argument(
        "--seed", type=int, default=12345,
        help="Random seed used for all five scenarios (default: 12345)."
    )
    p.add_argument(
        "--generate-only", action="store_true",
        help="Generate datasets/previews but do not launch BioTraj."
    )
    p.add_argument(
        "--spatial-boundary-x", type=float, default=None,
        help="Override the default spatial boundary (default: half the frame width)."
    )
    p.add_argument(
        "--temporal-transition-frame", type=int, default=None,
        help="Override the default temporal transition frame (default: half the run)."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        batch_main(args)
    else:
        interactive_main()

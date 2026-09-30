# AntTrajectoryAnalysis / BioTraj v1.2

A Python toolkit for extracting, analysing, and interpreting biological
trajectories from video recordings.

BioTraj was developed as part of the BioTrajectoryAnalysis project and
was originally designed for quantitative analysis of ant foraging
trajectories. The current release combines computer-vision tracking,
conventional statistical-physics descriptors, spatial and temporal
information geometry, scene annotation, and annotation-aware Fisher--Rao
analysis.

## Features

### Tracking

-   Automatic trajectory extraction using OpenCV
-   Interactive ROI selection
-   Multi-object tracking
-   Background subtraction and frame differencing
-   Export of trajectories in CSV format
-   Calibration from pixels to physical units

### Conventional trajectory analysis

-   Convex-hull and explored-area estimation
-   Monte Carlo area calculation
-   Occupancy maps
-   Principal Component Analysis (PCA) and anisotropy
-   Mean Squared Displacement (MSD)
-   Straightness analysis
-   Fractal-dimension estimation
-   Summary tables and figures

### Information geometry

-   Turning-angle probability distributions
-   Spatial Fisher--Rao distance analysis
-   Local Fisher--Rao gradient maps
-   Fisher--Rao distance matrices
-   Distributional clustering
-   Temporal Fisher--Rao analysis
-   Bootstrap uncertainty estimation

### Scene annotation and annotation-aware analysis

-   Interactive annotation of regions, surfaces, transitions, biological
    features, physical constraints, and artifact/exclusion regions
-   JSON storage of annotations in image coordinates
-   Mapping of information-geometric cells onto independently defined
    scene features
-   Regional and transition-based Fisher--Rao summaries
-   Distance-from-transition diagnostics and quality-control outputs

### Synthetic validation

BioTraj v1.2 includes a synthetic trajectory generator for controlled
validation of spatial and temporal information-geometric analyses. It
provides homogeneous, spatial-transition, temporal-transition, combined
spatiotemporal, and structured-arena scenarios.

## Main Scripts

-   `tracking/AntTracking.py` -- video-based trajectory extraction and
    tracking
-   `analysis/AntTrajectoryAnalysis.py` -- conventional trajectory
    analysis and spatial/temporal information geometry
-   `analysis/BioTraj_Scene_Annotator.py` -- interactive annotation of a
    reference video frame
-   `analysis/BioTraj_Annotated_FR_Analysis_v1_1.py` -- annotation-aware
    analysis of spatial Fisher--Rao results
-   `validation/BioTraj_SyntheticTrajectoryGenerator_v1_2.py` --
    controlled synthetic trajectory generation for validation

See `analysis/README.md` and the user manual in `docs/` for detailed
usage.

## Example Workflow

    Video recording
          |
          v
    tracking/AntTracking.py
          |
          v
    ant_tracks.csv
          |
          v
    analysis/AntTrajectoryAnalysis.py
          |
          +--> conventional trajectory descriptors
          +--> spatial Fisher--Rao analysis
          +--> temporal Fisher--Rao analysis

    Reference frame
          |
          v
    analysis/BioTraj_Scene_Annotator.py
          |
          v
    scene_annotations.json
          |
          +-------------------------------+
                                          |
    Spatial information-geometry results |
          |                               |
          +-------------------------------+
                                          v
                     BioTraj_Annotated_FR_Analysis_v1_1.py
                                          |
                                          v
                             annotation-aware results

## Example Dataset

The `examples/WallAnts/` directory provides a compact example for
testing the BioTraj workflow. It contains a sample video, tracking
outputs, BioTraj analysis outputs, a reference frame, scene annotations,
and annotation-aware Fisher--Rao results.

The included sample video and associated files are provided as a
reproducible software example and do not constitute the complete
experimental dataset used for all results reported in the associated
study. Complete experimental trajectory datasets and original
full-resolution recordings are available from the author upon reasonable
request.

## Applications

BioTraj was developed for ant trajectory analysis but can be adapted to
other systems in which trajectories can be represented in approximately
two-dimensional coordinates, including protozoa, rotifers, nematodes,
slugs, and other small organisms.

The interpretation of trajectory statistics and information-geometric
patterns remains system-dependent. Fisher--Rao heterogeneity identifies
differences between empirical movement distributions but does not by
itself establish a behavioural state or biological mechanism.

## Documentation

The BioTraj v1.2 user manual is provided in the `docs/` directory in PDF
and LaTeX form. It describes installation, input data, conventional
trajectory descriptors, spatial and temporal information geometry,
finite-sampling controls, synthetic validation, scene annotation, and
annotation-aware analysis.

## Associated Publication

Roccatano, D.

*BioTraj: Information-Geometric Mapping of Spatial and Temporal
Heterogeneity in Biological Trajectories.*

Manuscript prepared for submission.

## Part of the BioTrajectoryAnalysis Project

This module is one component of the broader BioTrajectoryAnalysis
framework for extracting and analysing biological trajectories from
video recordings.

## Disclaimer

This software is provided for educational and research purposes.
Although every effort has been made to verify the correctness of the
algorithms, users are responsible for validating the results obtained
for their specific applications. Biological tracking data may be
affected by image quality, occlusions, segmentation errors, projection
effects, finite sampling, and other experimental factors.

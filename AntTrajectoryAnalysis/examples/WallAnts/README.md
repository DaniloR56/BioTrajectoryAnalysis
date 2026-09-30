# WallAnts Example Dataset

This directory provides a compact worked example for the BioTraj
tracking, trajectory-analysis, and annotation-aware Fisher--Rao
workflow.

`SampleVideo.mov` is a 37-second excerpt from the longer AntWall
experimental recording. It is provided as a manageable real-data example
for testing the software and inspecting the generated outputs. The
example is not intended to replace the complete AntWall experimental
dataset used for scientific analysis; the shorter recording provides
substantially less spatial and temporal sampling.

## 1. Contents

The supplied example is organised as follows:

    WallAnts/
    ├── README.md
    ├── SampleVideo.mov
    ├── config.json
    ├── FirstFrame.jpg
    ├── FirstFrame_scene_annotations.json
    ├── FirstFrame_scene_annotated.png
    │
    ├── tracking_results/
    │   ├── ant_positions.csv
    │   ├── ant_tracks.csv
    │   ├── config.json
    │   ├── summary.json
    │   ├── track_stats.csv
    │   └── BioTraj_V1_2_revision2_batch.log
    │
    ├── BioTraj_V1_2_revision2/
    │   ├── analysis_report.txt
    │   ├── analysis_results.json
    │   ├── summary_for_paper.csv
    │   ├── density_heatmap.*
    │   ├── monte_carlo_area.*
    │   ├── spatial_exploration_anisotropy.*
    │   ├── trajectory_statistics.*
    │   ├── fisher_rao_distance_matrix.csv
    │   ├── info_geometry_cells.csv
    │   ├── info_geometry_summary.json
    │   ├── info_geometry_sampling_map.*
    │   ├── info_geometry_local_gradient.*
    │   ├── info_geometry_distance_matrix.*
    │   ├── info_geometry_behavioural_domains.*
    │   ├── temporal_fisher_rao_distance_matrix.csv
    │   ├── temporal_info_windows.csv
    │   ├── temporal_info_geometry_summary.json
    │   ├── temporal_info_sampling.*
    │   ├── temporal_info_speed.*
    │   ├── temporal_info_distance_matrix.*
    │   └── temporal_info_consecutive_distance.*
    │
    └── WALL_annotated_FR_v1_1/
        ├── analysis_metadata.json
        ├── annotated_FR_cells.csv
        ├── annotated_FR_overlay.*
        ├── annotation_summary.csv
        ├── FR_vs_transition_distance_qc.*
        ├── FR_vs_transition_distance_raw.*
        ├── surface_FR_boxplot_qc.*
        ├── surface_FR_boxplot_raw.*
        ├── surface_summary_qc.csv
        ├── surface_summary_raw.csv
        ├── transition_distance_binned_qc.csv
        ├── transition_distance_binned_raw.csv
        ├── transition_summary_qc.csv
        └── transition_summary_raw.csv

Here `*` indicates that both PNG and PDF graphical versions are supplied
where available.

The three result directories correspond to three distinct stages of the
workflow:

-   `tracking_results/` -- output of `AntTracking.py`;
-   `BioTraj_V1_2_revision2/` -- output of `AntTrajectoryAnalysis.py`;
-   `WALL_annotated_FR_v1_1/` -- output of the annotation-aware
    Fisher--Rao analysis.

## 2. Workflow

The complete example illustrates the following sequence:

    SampleVideo.mov
          |
          v
    AntTracking.py
          |
          v
    tracking_results/ant_tracks.csv
          |
          v
    AntTrajectoryAnalysis.py
          |
          v
    BioTraj_V1_2_revision2/
          |
          +-------------------------------+
                                          |
    FirstFrame.jpg                        |
          |                               |
          v                               |
    BioTraj_Scene_Annotator.py            |
          |                               |
          v                               |
    FirstFrame_scene_annotations.json     |
          |                               |
          +-------------------------------+
                          |
                          v
             BioTraj_Annotated_FR_Analysis_v1_1.py
                          |
                          v
                 WALL_annotated_FR_v1_1/

Users interested only in trajectory analysis can bypass video tracking
and use the supplied `tracking_results/ant_tracks.csv` directly. Users
interested only in annotation-aware analysis can use the supplied scene
annotation and spatial information-geometry outputs.

## 3. Tracking

The tracking program is located at:

    ../../tracking/AntTracking.py

The supplied `config.json` contains the settings associated with the
example, including the region of interest and spatial calibration.

A reference tracking run is already supplied in `tracking_results/`. Its
principal files are:

-   `ant_positions.csv` -- detected ant positions;
-   `ant_tracks.csv` -- reconstructed trajectories and the principal
    input to BioTraj analysis;
-   `track_stats.csv` -- reconstructed-track statistics;
-   `summary.json` -- machine-readable tracking summary;
-   `config.json` -- configuration associated with the generated
    results.

The short example video is intended primarily as a software test and
tutorial dataset.

## 4. Trajectory analysis

The trajectory-analysis program is located at:

    ../../analysis/AntTrajectoryAnalysis.py

It generates conventional trajectory descriptors together with spatial
and temporal information-geometric analyses.

From `examples/WallAnts/`, the supplied trajectory data can be analysed
with:

    python3 ../../analysis/AntTrajectoryAnalysis.py \
        tracking_results/ant_tracks.csv \
        --output ./BioTraj_V1_2_revision2 \
        --config tracking_results/config.json \
        --info-geometry \
        --info-grid-x 20 \
        --info-grid-y 12 \
        --info-angle-bins 18 \
        --info-min-samples 30 \
        --info-clusters 4 \
        --info-pseudocount 0 \
        --info-bootstrap 200 \
        --info-seed 12345 \
        --temporal-info-geometry \
        --temporal-window-frames 900 \
        --temporal-step-frames 900 \
        --temporal-min-samples 100 \
        --temporal-bootstrap 200 \
        --fps 60

The spatial information-geometry analysis therefore uses a 20 x 12 grid,
18 turning-angle bins, a minimum of 30 samples per eligible spatial
cell, four distributional clusters, and 200 bootstrap replicates.

The temporal analysis uses 900-frame non-overlapping windows with a
minimum of 100 samples per window and 200 bootstrap replicates. At 60
frames per second, 900 frames correspond to 15 seconds.

The reference results are supplied in:

    BioTraj_V1_2_revision2/

These include conventional descriptors, spatial sampling and Fisher--Rao
outputs, and temporal Fisher--Rao outputs.

## 5. Scene annotation

The example also demonstrates the optional annotation workflow.

The reference image is:

    FirstFrame.jpg

The scene-annotation program is:

    ../../analysis/BioTraj_Scene_Annotator.py

It can be launched from the WallAnts directory with:

    python3 ../../analysis/BioTraj_Scene_Annotator.py FirstFrame.jpg

The supplied annotation files are:

    FirstFrame_scene_annotations.json
    FirstFrame_scene_annotated.png

Annotations are defined independently of the Fisher--Rao map and stored
in original-image coordinates. They can represent scene regions,
surfaces, transitions, biological features, physical constraints,
boundaries, and artifact/exclusion regions as appropriate.

## 6. Annotation-aware Fisher--Rao analysis

The annotation-aware program is:

    ../../analysis/BioTraj_Annotated_FR_Analysis_v1_1.py

It combines the independently defined scene annotations with the spatial
information-geometric cell results.

From `examples/WallAnts/`, the reference analysis can be run as:

    python3 ../../analysis/BioTraj_Annotated_FR_Analysis_v1_1.py \
        --annotations FirstFrame_scene_annotations.json \
        --info-cells BioTraj_V1_2_revision2/info_geometry_cells.csv \
        --tracks tracking_results/ant_tracks.csv \
        --image FirstFrame.jpg \
        --grid-x 20 \
        --grid-y 12 \
        --output WALL_annotated_FR_v1_1

The resulting directory contains:

-   an annotated Fisher--Rao overlay;
-   cell-level annotation assignments;
-   annotation summaries;
-   raw and quality-controlled surface summaries;
-   raw and quality-controlled transition summaries;
-   binned distance-from-transition data;
-   distance-from-transition diagnostic figures;
-   analysis metadata.

The annotation-aware analysis is descriptive. Spatial cells within a
single recording are not independent biological replicates, and
Fisher--Rao heterogeneity alone does not establish a behavioural state
or biological mechanism.

## 7. Important limitation of the 37-second example

The supplied video contains only 37 seconds of the original AntWall
recording and therefore substantially fewer trajectory observations than
the complete experimental dataset.

Consequently, fewer spatial cells or temporal windows may satisfy the
sampling requirements, and the information-geometric patterns can differ
from those obtained from the complete recording. The supplied example
should therefore be regarded as a demonstration and reproducibility test
of the software workflow rather than as a reproduction of all scientific
results from the complete AntWall experiment.

Scientific conclusions concerning the complete experiment should be
based on the complete experimental trajectories and the corresponding
analysis parameters.

## 8. Interpretation and quality control

Before interpreting a new dataset, users should verify the ROI, spatial
calibration where required, video frame rate, tracking quality,
trajectory reconstruction, spatial sampling, temporal sampling, and
analysis parameters.

Fisher--Rao distances compare empirical movement distributions.
Physically nearby regions can have different distributions, while
spatially separated regions can have similar distributions. The
information-geometric results should therefore be interpreted together
with conventional quantities such as occupancy, speed, explored area,
trajectory geometry, and anisotropy, and with knowledge of the
experimental conditions.

## 9. Reproducibility

Small numerical differences can arise from Python and numerical-library
versions and from stochastic procedures such as Monte Carlo estimation,
bootstrapping, and clustering. The supplied outputs should therefore be
regarded as reference results rather than requiring bit-for-bit
reproduction.

Detailed installation, methodology, parameter definitions, and
interpretation are provided in:

    ../../docs/BioTraj_User_Manual_v1_2.pdf

The required Python packages are listed in:

    ../../requirements.txt

## 10. Example data versus complete experimental data

The data in this directory are intentionally compact. The sample video
and associated trajectories do not constitute the complete experimental
dataset used for all results reported in the associated study.

The complete experimental trajectory datasets and original
full-resolution video recordings are available from the corresponding
author upon reasonable request.

## Citation and license

For citation information and licensing terms, consult the
repository-level `CITATION.cff` and `LICENSE` files.

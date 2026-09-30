# BioTraj Tracking

This directory contains the video-tracking component of BioTraj.

## Script

`AntTracking.py`

The program extracts positions and reconstructs trajectories of multiple
moving organisms from video recordings using OpenCV-based image processing
and tracking.

It was developed primarily for ant recordings but can be applied to other
small moving organisms when the imaging geometry and segmentation conditions
are appropriate.

## Main functions

The tracking workflow provides:

- video loading and processing;
- interactive region-of-interest (ROI) definition;
- background subtraction and frame differencing;
- detection of moving objects;
- multi-object trajectory reconstruction;
- spatial calibration;
- export of positions, trajectories, tracking statistics, configuration,
  and summary information.

## Running the tracker

A complete worked example is provided in:

`../examples/WallAnts/`

From that directory, the supplied example can be processed using:

    python3 ../../tracking/AntTracking.py \
        --config config.json \
        SampleVideo.mov

The precise command-line options and interactive controls are described in
the BioTraj User Manual.

## Output

A typical tracking run creates a `tracking_results/` directory containing:

    tracking_results/
    ├── ant_positions.csv
    ├── ant_tracks.csv
    ├── config.json
    ├── summary.json
    └── track_stats.csv

### ant_positions.csv

Contains the detected object positions obtained during video processing.

### ant_tracks.csv

Contains the reconstructed trajectories. This is the principal input to:

`../analysis/AntTrajectoryAnalysis.py`

### track_stats.csv

Contains statistics associated with the reconstructed tracks.

### summary.json

Contains a machine-readable summary of the tracking run.

### config.json

Records the configuration associated with the tracking results and helps
document the provenance of the subsequent analysis.

## Downstream analysis

The reconstructed trajectories can be analysed with:

`../analysis/AntTrajectoryAnalysis.py`

BioTraj provides conventional trajectory descriptors together with spatial
and temporal information-geometric analyses.

Scene annotation and annotation-aware Fisher--Rao analyses are provided by
the companion programs in:

`../analysis/`

## Quality control

Tracking results should be visually inspected before quantitative analysis.
In particular, users should verify the ROI, spatial calibration, frame rate,
object detection, trajectory continuity, and possible effects of occlusion
or segmentation errors.

## Documentation

Detailed installation, configuration, tracking, calibration, and analysis
instructions are provided in:

`../docs/BioTraj_User_Manual_v1_2.pdf`

A compact real-data example is provided in:

`../examples/WallAnts/`

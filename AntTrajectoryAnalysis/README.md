# AntTrajectoryAnalysis

A Python toolkit for extracting and analysing ant trajectories from video recordings.

The software was developed as part of the BioTrajectoryAnalysis project and was originally designed for the quantitative analysis of ant foraging trajectories. It demonstrates how computer vision and statistical-physics descriptors can be combined to study biological motion using inexpensive imaging equipment and open-source software.

## Features

### Tracking

- Automatic trajectory extraction using OpenCV
- Interactive ROI selection
- Multi-object tracking
- Background subtraction and frame differencing
- Export of trajectories in CSV format
- Calibration from pixels to physical units

### Analysis

- Convex hull and explored area estimation
- Monte Carlo area calculation
- Occupancy maps
- Principal Component Analysis (PCA)
- Mean Squared Displacement (MSD)
- Straightness analysis
- Fractal dimension estimation
- Publication-quality figures

## Main Scripts

- AntTracking_V1.0.py – trajectory extraction and tracking
- AntTrajectoryAnalysis_V1.0.py – statistical analysis and figure generation

## Example Workflow

text Video Recording         ↓ AntTracking_V1.0.py         ↓ ant_tracks.csv         ↓ AntTrajectoryAnalysis_V1.0.py         ↓ Statistics, Figures and Reports 

## Applications

The software was developed for ant trajectory analysis but can be adapted to other biological systems exhibiting approximately planar motion, including:

- Protozoa
- Rotifers
- Nematodes
- Slugs
- Other small organisms

## Associated Publication

Roccatano, D.

Quantitative Analysis of Biological Trajectories Using Computer Vision and Statistical Physics: An Application to Ant Motion.

(under preparation)

## Part of the BioTrajectoryAnalysis Project

This module is one component of the broader BioTrajectoryAnalysis framework for extracting and analysing biological trajectories from video recordings.


## Disclaimer

This software is provided for educational and research purposes.
Although every effort has been made to verify the correctness of the
algorithms, users are responsible for validating the results obtained
for their specific applications. Biological tracking data may be
affected by image quality, occlusions, segmentation errors, and other
experimental factors.

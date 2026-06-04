# BioTrajectoryAnalysis

BioTrajectoryAnalysis is an open-source collection of Python tools for the extraction, visualization, and quantitative analysis of biological trajectories from video recordings.

The project combines computer vision, automated tracking, and statistical-physics descriptors to study the motion of organisms across a wide range of spatial and temporal scales. The software was initially developed for the analysis of ant foraging trajectories, but the framework is designed to be applicable to many other biological systems.

## Current Modules

### AntTrajectoryAnalysis

Tools for tracking and analysing ant motion from video recordings.

Features include:

- Automatic object detection and tracking
- Region-of-interest (ROI) selection
- Trajectory reconstruction
- Convex-hull area estimation
- Monte Carlo area calculation
- Occupancy maps
- Principal Component Analysis (PCA)
- Mean Squared Displacement (MSD)
- Straightness analysis
- Fractal dimension estimation
- Publication-quality figures

This module was developed for the study:

> Quantitative Analysis of Biological Trajectories Using Computer Vision and Statistical Physics: An Application to Ant Motion

## Planned Modules

### ProtistTrajectoryAnalysis

Trajectory extraction and statistical analysis of microorganisms such as:

- Euglena
- Paramecium
- Rotifers
- Other protists

Planned analyses include:

- Swimming speed distributions
- Mean Squared Displacement
- Directional persistence
- Rotational diffusion
- Occupancy maps
- Population statistics

### SlugTrajectoryAnalysis

Tools for analysing slug locomotion and exploratory behaviour.

Planned features include:

- Trail reconstruction
- Velocity analysis
- Turning-angle statistics
- Space-use characterization
- Environmental preference mapping

## Project Goals

The aim of BioTrajectoryAnalysis is to provide a flexible and educational framework for applying concepts from:

- Computer Vision
- Statistical Physics
- Random Walk Theory
- Movement Ecology
- Active Matter
- Data Science

to real biological trajectory datasets.

## Requirements

- Python 3.10+
- NumPy
- Pandas
- SciPy
- Matplotlib
- OpenCV
- scikit-learn

## License

MIT License

## Author

Danilo Roccatano

School of Mathematics and Physics

University of Lincoln, United Kingdom

## Citation

If you use this software in published work, please cite the associated publication and provide a link to this repository.

## Future Development

The repository will continue to expand with additional modules dedicated to the analysis of microorganisms, invertebrates, and other biological systems exhibiting two-dimensional or quasi-two-dimensional motion.



# BioTraj Development Roadmap

BioTraj v1.2 provides a complete workflow for two-dimensional
trajectory analysis using conventional movement descriptors and
information-geometric comparison of empirical movement distributions.

The following items represent possible directions for future
development. They are not part of the current v1.2 implementation
and do not constitute a fixed development schedule.

## Possible future extensions

### 1. Three-dimensional trajectories
Extend the current 2D framework to accept externally reconstructed
(x, y, z, t) trajectories, for example from stereoscopic imaging,
depth-resolved microscopy, or other 3D tracking systems.

### 2. Additional movement observables
Generalize the information-geometric analysis beyond unsigned turning
angle to distributions of quantities such as:

- signed turning angle / turning handedness;
- step length and speed;
- turning frequency;
- run length between successive turns;
- orientation;
- joint distributions of multiple movement variables.

### 3. Collective-motion descriptors
For multi-object trajectories, investigate distributions based on:

- inter-individual distance;
- local alignment;
- neighbourhood structure;
- other measures of collective organization.

### 4. Adaptive spatial analysis
Explore alternatives to the current regular spatial grid, including
adaptive partitions based on local sampling density or scene geometry.

### 5. Finite-sampling calibration
Develop additional methods for quantifying uncertainty in empirical
Fisher--Rao distances, including explicit finite-sample calibration
and hierarchical or spatial block-resampling approaches.

### 6. Complex and non-planar environments
Investigate integration with image rectification, calibrated scene
geometry, and externally reconstructed 3D surfaces to reduce
projection effects in complex recording environments.

### 7. Additional distributional measures
Provide optional comparison with complementary distances and
divergences such as Hellinger, Jensen--Shannon, and Wasserstein
measures.

### 8. Trajectory-state and change-point methods
Explore optional interfaces to state-based or change-point approaches
for comparison with the state-free Fisher--Rao representation used
by BioTraj.

## Suggestions and contributions

Suggestions, feature requests, and contributions are welcome through
the GitHub issue tracker. The roadmap may evolve as BioTraj is applied
to new trajectory-analysis problems.

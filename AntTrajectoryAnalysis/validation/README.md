# BioTraj Synthetic Validation

This directory contains the synthetic trajectory generator used to test
and validate the spatial and temporal information-geometric components
of BioTraj.

## Script

`BioTraj_SyntheticTrajectoryGenerator_v1_2.py`

The generator produces controlled trajectory datasets for which the
underlying spatial and/or temporal structure is known in advance.

## Validation scenarios

Five scenarios are available:

1. homogeneous stationary motion;
2. spatial two-domain motion;
3. temporal two-state motion;
4. combined spatiotemporal motion;
5. structured-arena motion.

These datasets are intended for software validation and methodological
testing. They are not intended as mechanistic models of ant behaviour.

## Generate all validation datasets

From the `validation/` directory:

    python3 BioTraj_SyntheticTrajectoryGenerator_v1_2.py \
        --all \
        --base-dir synthetic_validation \
        --seed 12345 \
        --generate-only

Each generated scenario includes an `ant_tracks.csv` file compatible
with `AntTrajectoryAnalysis.py`, together with configuration,
ground-truth information, and preview outputs.

## Reference configuration

The validation datasets used for BioTraj v1.2 were generated using a
fixed random seed (`12345`). The reference synthetic system contains
20 agents and 12,116 frames in a 1905 x 726 pixel arena.

For the complete analysis parameters and validation procedure, see:

`../docs/BioTraj_User_Manual_v1_2.pdf`

## Reproducibility

The synthetic datasets can be regenerated rather than stored in the
software distribution. Using the fixed seed and documented parameters
provides a reproducible input for testing the BioTraj analysis
pipeline.

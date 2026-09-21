WallAnts Example Dataset

This directory provides a compact worked example for the BioTraj tracking and trajectory-analysis workflow.

SampleVideo.mov is a 37-second excerpt from the longer AntWall experimental recording. The excerpt is provided as a manageable real-data example for testing the tracking software, inspecting the generated trajectory files, and becoming familiar with the BioTraj analysis workflow.

The example is not intended to replace the complete AntWall experimental dataset used for scientific analysis. In particular, the shorter recording provides substantially less spatial and temporal sampling than the complete experiment.

⸻

1. Purpose of the example

The WallAnts example allows users to:

1. run AntTracking.py on a short real experimental video;
2. examine the use of a region of interest (ROI) and tracking configuration;
3. inspect the generated position and trajectory files;
4. use the reconstructed trajectories as input to AntTrajectoryAnalysis.py;
5. reproduce conventional trajectory-analysis outputs;
6. explore the spatial information-geometry analysis;
7. explore the temporal information-geometry analysis;
8. inspect the numerical, graphical, and machine-readable output formats produced by BioTraj.

Because the supplied recording is only 37 seconds long, this example should primarily be regarded as a software demonstration and reproducibility test of the workflow rather than as a reproduction of the complete AntWall experiment.

⸻

2. Workflow

The BioTraj workflow consists of two principal computational stages: video tracking followed by trajectory analysis.

37-s AntWall video excerpt
          |
          v
    AntTracking.py
          |
          v
   tracking_results/
          |
          +---- ant_tracks.csv
                    |
                    v
      AntTrajectoryAnalysis.py
                    |
                    v
          analysis_results/
             |            |
             v            v
      Conventional     Information-
        metrics         geometry
                         analysis

Users interested only in trajectory analysis can bypass the video-tracking stage and use the supplied ant_tracks.csv directly.

⸻

3. Recommended directory structure

The example can be organised as:

WallAnts/
├── README.md
├── SampleVideo.mov
├── config.json
│
├── tracking_results/
│   ├── ant_positions.csv
│   ├── ant_tracks.csv
│   ├── config.json
│   ├── summary.json
│   └── track_stats.csv
│
└── analysis_results/
    ├── analysis_report.txt
    ├── analysis_results.json
    ├── summary_for_paper.csv
    │
    ├── density_heatmap.png
    ├── monte_carlo_area.png
    ├── spatial_exploration_anisotropy.png
    ├── trajectory_statistics.png
    │
    ├── fisher_rao_distance_matrix.csv
    ├── info_geometry_cells.csv
    ├── info_geometry_summary.json
    ├── info_geometry_sampling_map.png
    ├── info_geometry_local_gradient.png
    ├── info_geometry_distance_matrix.png
    ├── info_geometry_behavioural_domains.png
    │
    ├── temporal_fisher_rao_distance_matrix.csv
    ├── temporal_info_windows.csv
    ├── temporal_info_geometry_summary.json
    ├── temporal_info_sampling.png
    ├── temporal_info_speed.png
    ├── temporal_info_distance_matrix.png
    └── temporal_info_consecutive_distance.png

The analysis program may additionally generate PDF versions of the graphical outputs. PNG files are convenient for viewing directly on GitHub, whereas PDF files are useful for publication-quality figures.

⸻

4. Example video

SampleVideo.mov contains a 37-second excerpt of the AntWall recording.

The purpose of supplying a short excerpt rather than the complete experimental video is to provide a compact example that can be downloaded and processed quickly.

It can be used to test:

* video loading;
* ROI definition;
* ant detection;
* trajectory reconstruction;
* tracking output;
* downstream trajectory analysis.

The example video is therefore primarily a software test and tutorial dataset.

⸻

5. Region of interest

The tracking workflow supports definition of a region of interest (ROI).

The ROI restricts tracking and subsequent spatial analysis to the experimentally relevant part of the image. This is particularly important when the complete camera frame contains regions outside the accessible experimental arena.

The supplied configuration corresponds to the ROI used for this example.

When analysing a new recording, users should visually verify the ROI before running the complete analysis.

Area-dependent quantities should only be interpreted after checking that the ROI and spatial calibration correctly represent the experimental geometry.

⸻

6. Tracking configuration

config.json contains the configuration associated with the example tracking run.

The tracking program is located at:

../../tracking/AntTracking.py

The configuration records the relevant tracking settings so that the supplied example can be reproduced and the origin of the generated trajectory data can be documented.

Detailed information on configuration parameters, calibration, ROI selection, tracking controls, and program operation is provided in the BioTraj User Manual.

⸻

7. Tracking results

The directory

tracking_results/

contains the reference output from the tracking stage.

ant_positions.csv

Contains the detected ant positions obtained during video processing.

These data represent the detections from which trajectories are reconstructed.

ant_tracks.csv

Contains the reconstructed ant trajectories.

This is the principal trajectory file used by:

AntTrajectoryAnalysis.py

and therefore provides the connection between the tracking and analysis stages of BioTraj.

track_stats.csv

Contains statistics associated with the reconstructed tracks.

summary.json

Contains a machine-readable summary of the tracking run.

config.json

Records the tracking configuration associated with the generated results.

Keeping the configuration together with the trajectory files helps document the provenance of the analysis.

⸻

8. Trajectory analysis

The trajectory-analysis program is located at:

../../analysis/AntTrajectoryAnalysis.py

The program can analyse the trajectories produced by AntTracking.py and generate conventional trajectory descriptors together with spatial and temporal information-geometric analyses.

The principal input for this stage is:

tracking_results/ant_tracks.csv

The analysis produces numerical results, machine-readable files, and graphical summaries.

⸻

9. Conventional trajectory analysis

BioTraj calculates several conventional descriptors of spatial exploration and reconstructed trajectories.

Representative outputs include the following.

Density heat map

density_heatmap.png

Shows the spatial distribution of recorded positions and can reveal preferentially occupied regions of the experimental arena.

Monte Carlo explored area

monte_carlo_area.png

Summarises the Monte Carlo estimate of the spatial area explored by the tracked animals.

Spatial exploration and anisotropy

spatial_exploration_anisotropy.png

Characterises the geometry and directional structure of spatial exploration.

Trajectory statistics

trajectory_statistics.png

Provides graphical summaries of statistical properties derived from the reconstructed trajectories.

⸻

10. Numerical analysis outputs

Several files provide numerical or machine-readable summaries of the analysis.

analysis_report.txt

Provides a human-readable summary of the analysis and principal results.

analysis_results.json

Contains analysis results in machine-readable JSON format.

This format is useful for automated post-processing or integration into other analysis workflows.

summary_for_paper.csv

Provides a compact tabular summary of principal quantitative results.

This file can be useful when comparing multiple experimental datasets or collecting results for subsequent statistical analysis.

⸻

11. Spatial information geometry

BioTraj includes an information-geometric analysis designed to compare local movement statistics across the experimental arena.

The accessible region is spatially partitioned, and movement distributions are estimated for sufficiently sampled spatial regions.

Information-geometric distances can then be used to quantify differences between these local statistical distributions.

This approach complements conventional measures based only on position, occupancy, speed, or explored area.

info_geometry_cells.csv

Contains numerical information associated with the spatial cells included in the analysis.

Only cells satisfying the required sampling criteria are included in the statistical comparison.

fisher_rao_distance_matrix.csv

Contains pairwise information-geometric distances between sampled spatial regions.

The matrix quantifies statistical differences between the local movement distributions.

info_geometry_sampling_map.png

Shows the spatial sampling available for the information-geometric analysis.

This figure is an important quality-control output because it distinguishes adequately sampled regions from poorly sampled or unsampled areas.

info_geometry_local_gradient.png

Maps local variation in movement statistics across the sampled arena.

Regions with relatively large local statistical changes can therefore be identified spatially.

info_geometry_distance_matrix.png

Provides a graphical representation of the pairwise information-geometric distance matrix.

info_geometry_behavioural_domains.png

Shows spatial regions grouped according to similarities in their local movement statistics.

These regions are data-driven statistical domains. They should not automatically be interpreted as distinct biological behavioural states without additional experimental evidence.

info_geometry_summary.json

Provides a machine-readable summary of the spatial information-geometry analysis.

⸻

12. Temporal information geometry

BioTraj can also examine changes in movement statistics through time.

The trajectory data are divided into temporal windows, and movement distributions are estimated independently for sufficiently sampled windows.

Information-geometric distances between these distributions can then be used to identify temporal changes in movement statistics.

temporal_info_windows.csv

Contains numerical information associated with the temporal windows used in the analysis.

temporal_fisher_rao_distance_matrix.csv

Contains pairwise information-geometric distances between temporal windows.

temporal_info_sampling.png

Summarises temporal sampling and indicates the windows available for analysis.

temporal_info_speed.png

Shows changes in movement-speed statistics through time.

temporal_info_distance_matrix.png

Provides a graphical representation of the information-geometric distances between temporal windows.

temporal_info_consecutive_distance.png

Shows the information-geometric distance between consecutive temporal windows.

Large values indicate periods in which the estimated movement statistics change more strongly between successive windows.

temporal_info_geometry_summary.json

Provides a machine-readable summary of the temporal information-geometry analysis.

⸻

13. Important limitation of the 37-second example

The spatial and temporal information-geometry analyses depend on adequate statistical sampling.

The supplied example contains only 37 seconds of the original AntWall recording. It therefore contains substantially fewer trajectory observations than the complete experimental dataset.

Consequently:

* fewer spatial cells may satisfy the minimum sampling requirement;
* spatial information-geometric maps may be less complete;
* fewer temporal windows may be available;
* some temporal analyses may have limited statistical support;
* distance matrices may differ substantially from those obtained from the complete experiment;
* statistical domains obtained from the short sample should not be interpreted as reproducing those from the full AntWall dataset.

The information-geometric results generated from this example are therefore intended primarily to demonstrate the analysis procedure, software behaviour, and output formats.

Scientific conclusions concerning the complete AntWall experiment should be based on the complete experimental trajectories and the corresponding analysis parameters.

⸻

14. Reproducing the complete example

The WallAnts example can be reproduced directly from the command line. The commands below assume that the current working directory is:

examples/WallAnts/

There are two ways to use the supplied example: the complete tracking-and-analysis workflow, or an analysis-only workflow using the supplied trajectories.

A. Complete tracking and analysis workflow

Step 1: Run the tracking

Run AntTracking.py on the supplied 37-second video using the provided tracking configuration:

python3 ../../tracking/AntTracking.py \
    --config config.json \
    SampleVideo.mov

The configuration specifies the tracking parameters, region of interest (ROI), and spatial calibration used for the example.

The tracking stage generates:

tracking_results/
├── ant_positions.csv
├── ant_tracks.csv
├── config.json
├── summary.json
└── track_stats.csv

The principal output for subsequent analysis is:

tracking_results/ant_tracks.csv

Step 2: Run the trajectory analysis

Run the conventional trajectory analysis together with the spatial and temporal information-geometry analyses:

python3 ../../analysis/AntTrajectoryAnalysis.py \
    tracking_results/ant_tracks.csv \
    --output ./analysis_results \
    --config tracking_results/config.json \
    --info-geometry \
    --info-grid-x 20 \
    --info-grid-y 12 \
    --info-min-samples 30 \
    --info-clusters 4 \
    --info-bootstrap 200 \
    --temporal-info-geometry \
    --temporal-window-frames 900 \
    --temporal-step-frames 900 \
    --temporal-min-samples 100 \
    --temporal-bootstrap 200

For this example, the spatial information-geometry analysis therefore uses:

Parameter	Value
Spatial grid	20 x 12
Minimum samples per cell	30
Distributional clusters	4
Bootstrap replicates	200

The temporal information-geometry analysis uses:

Parameter	Value
Window length	900 frames
Window step	900 frames
Minimum samples per window	100
Bootstrap replicates	200

At a recording rate of 60 frames per second, a 900-frame temporal window corresponds to 15 seconds. The 900-frame step therefore gives non-overlapping 15-second analysis windows.

The generated results are written to:

analysis_results/

and can be compared with the reference outputs supplied with this example.

B. Analysis-only workflow

It is not necessary to repeat the video-tracking stage in order to test the trajectory-analysis software.

The repository already contains the reference trajectory file:

tracking_results/ant_tracks.csv

The complete analysis can therefore be run directly using:

python3 ../../analysis/AntTrajectoryAnalysis.py \
    tracking_results/ant_tracks.csv \
    --output ./analysis_results \
    --config tracking_results/config.json \
    --info-geometry \
    --info-grid-x 20 \
    --info-grid-y 12 \
    --info-min-samples 30 \
    --info-clusters 4 \
    --info-bootstrap 200 \
    --temporal-info-geometry \
    --temporal-window-frames 900 \
    --temporal-step-frames 900 \
    --temporal-min-samples 100 \
    --temporal-bootstrap 200

This analysis-only workflow is useful for testing the conventional trajectory and information-geometry analyses without repeating the video-processing stage.

Important note

SampleVideo.mov contains only a 37-second excerpt of the complete AntWall recording. The commands above reproduce the example workflow and its reference outputs, not the complete experimental AntWall analysis.

Because the short example contains substantially less spatial and temporal sampling than the complete dataset, the resulting information-geometric analysis should be interpreted as a demonstration of the software workflow rather than as a reproduction of the scientific results obtained from the full experiment.

⸻

15. Interpreting the information-geometric results

The information-geometric analysis compares statistical distributions of movement.

It therefore measures something different from simple physical separation in space or time.

Two spatial regions that are physically separated may have similar movement distributions and consequently a relatively small information-geometric distance.

Conversely, nearby spatial regions may exhibit substantially different movement statistics.

The same principle applies to temporal windows: two periods separated in time may exhibit similar movement statistics, whereas consecutive periods can sometimes differ substantially.

Information geometry therefore provides a complementary description to conventional quantities such as:

* occupancy;
* speed;
* explored area;
* trajectory geometry;
* directional anisotropy.

The information-geometric results should be interpreted together with these conventional measures and with knowledge of the experimental conditions.

⸻

16. Quality control

Before interpreting results obtained from a new dataset, users should check:

1. that the ROI correctly represents the experimental region;
2. that spatial calibration is appropriate where physical units are required;
3. that the video frame rate is correct;
4. that detected positions correspond to actual animals;
5. that reconstructed trajectories do not contain obvious tracking artefacts;
6. that spatial cells used for information geometry have adequate sampling;
7. that temporal windows satisfy the required sampling criteria;
8. that analysis parameters are appropriate for the duration and density of the dataset.

Visual inspection of trajectories and sampling maps remains an important part of the analysis workflow.

⸻

17. Expected numerical variation

Small numerical differences between systems or repeated analyses may occur because of differences in:

* Python versions;
* numerical-library versions;
* operating systems;
* stochastic sampling;
* Monte Carlo calculations;
* bootstrap procedures;
* clustering procedures.

The supplied results should therefore be regarded as reference outputs rather than requiring bit-for-bit reproduction.

Large discrepancies should prompt verification of the configuration, ROI, calibration, input trajectories, sampling thresholds, and analysis parameters.

⸻

18. Software requirements

The required Python packages are listed in the repository-level:

requirements.txt

Use of a dedicated Python virtual environment is recommended.

For example:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

On Windows, activation of the virtual environment will use the corresponding Windows command.

Detailed installation and operating instructions are provided in:

docs/BioTraj_User_Manual_v1_2.pdf

⸻

19. Using BioTraj with new experimental data

A typical workflow for a new dataset is:

1. record the experimental video;
2. define the appropriate ROI;
3. determine the spatial calibration if physical units are required;
4. configure the tracking parameters;
5. run AntTracking.py;
6. visually inspect detections and reconstructed trajectories;
7. examine the tracking statistics;
8. run AntTrajectoryAnalysis.py;
9. inspect conventional trajectory statistics;
10. inspect spatial sampling before interpreting spatial information geometry;
11. inspect temporal sampling before interpreting temporal information geometry;
12. compare quantitative results among experimental conditions.

For long recordings, users should ensure that the selected spatial and temporal sampling parameters are appropriate for the amount and density of available trajectory data.

⸻

20. Example data versus full experimental data

The data in this directory are intentionally compact.

The 37-second AntWall excerpt is included to make it practical for a new user to download the repository and test the software without requiring the complete experimental video.

It should therefore be distinguished from the complete experimental dataset used for quantitative scientific comparisons.

Where full experimental datasets are archived separately, users should refer to the associated publication and data repository for the complete data and analysis parameters.

⸻

Citation

If you use BioTraj, AntTracking.py, AntTrajectoryAnalysis.py, or the supplied example dataset in research or teaching, please cite the software using the information provided in the repository-level:

CITATION.cff

If the software accompanies a scientific publication, please also cite the associated publication as indicated in the main repository documentation.

⸻

License

The software, example data, and associated documentation are distributed under the terms specified in the repository-level:

LICENSE

Please consult that file before redistribution or reuse.

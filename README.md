# IJDE reproducibility package

This repository contains the audited aggregate outputs, source-data mappings
and Python scripts supporting the International Journal of Digital Earth
manuscript, *A vertically harmonised Digital Earth framework for
subsidence-sensitive coastal inundation screening*.

The current audited package is archived as release `v1.1.1`.
Release `v1.1.0` was generated from the same source commit as `v1.0.0` and is
therefore superseded; it should not be used to reproduce the IJDE analysis.

## Persistent record

- Current `v1.1.1` version DOI: https://doi.org/10.5281/zenodo.21965956
- Zenodo concept DOI for all versions: https://doi.org/10.5281/zenodo.21862032
- Previous `v1.0.0` version DOI: https://doi.org/10.5281/zenodo.21862033
- Superseded `v1.1.0` version DOI: https://doi.org/10.5281/zenodo.21965633

Cite 10.5281/zenodo.21965956 when referring to the exact IJDE analysis package. Use the concept DOI only when referring to the evolving record as a whole.

## Contents

- common-domain and administrative-coverage audits;
- stabilised, decaying-trend and continued-trend scenario summaries;
- terrain, vertical-datum and connectivity sensitivity summaries;
- six-site historical-comparison metadata and pixel-distance summary;
- source-data mappings for manuscript figures and tables;
- Python scripts used for the revised analysis and selected figures;
- editable draw.io and SVG sources for the Digital Earth framework.

Run a script with `--help` to inspect its required arguments. Large local input
rasters are intentionally not included, so reproducing raster-level outputs
requires the source products described in the manuscript and user-supplied
paths.

## Availability boundary

Large derived SBAS-InSAR time-series rasters, deformation rasters, spatial
masks, observation-level six-site series and third-party source datasets are
not redistributed. Their aggregate outputs and provenance are documented in
this repository. Original Sentinel, Copernicus, AR6, HKO, IBTrACS, GSHHG,
WorldCover, GADM and SSP products remain subject to their providers' terms.

No synthetic values are included. Machine-readable files were copied or
reformatted without numerical alteration from the audited manuscript build
inputs. The six-site figure, matching table and distance summary use one
consistent benchmark-pixel pairing.

## Licence

Python code is released under the MIT License. Author-generated aggregate
tables, metadata and framework graphics are released under CC BY 4.0, subject
to the exclusions in `DATA_LICENSE.md`.

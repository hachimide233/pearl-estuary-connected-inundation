# Pearl River Estuary coastal-inundation reproducibility package

This repository provides audited aggregate outputs, provenance metadata and
Python workflows for a multi-source Earth-observation analysis of land
subsidence and connected coastal-inundation exposure in the Pearl River
Estuary.

Release `v1.2.0` is journal-neutral and supersedes the journal-specific
wording used in `v1.1.1`. Scientific values are unchanged except for the
documented correction to the six-site benchmark-to-pixel distance summary.
Published earlier releases remain available as part of the version history.

## Persistent record

- Zenodo concept DOI for all versions: https://doi.org/10.5281/zenodo.21862032
- Previous `v1.1.1` version DOI: https://doi.org/10.5281/zenodo.21965956
- Previous `v1.0.0` version DOI: https://doi.org/10.5281/zenodo.21862033

Use the concept DOI to cite the evolving record. After Zenodo archives this
release, use the new version-specific DOI when an exact-file citation is
required. The version DOI is minted by Zenodo and is therefore not predicted
or hard-coded here.

## Contents

- common-domain and administrative-coverage audits;
- stabilised, decaying-trend and continued-trend scenario summaries;
- terrain, vertical-datum and connectivity sensitivity summaries;
- six-site historical-comparison metadata and pixel-distance statistics;
- source-to-output mappings for disclosed figures and tables;
- Python scripts for the common-domain analysis and selected visual outputs;
- editable draw.io and SVG sources for the analytical framework.

Run any Python script with `--help` to inspect its required arguments. Large
local rasters are intentionally excluded, so raster-level reproduction
requires the source products cited by the accompanying study and user-supplied
paths.

## Interpretation boundaries

The 2050 scenarios are the principal assessment horizon. Results for 2100 are
retained as exploratory, high-uncertainty extensions and should not be given
the same evidential weight as the 2050 results.

The six-site comparison uses fixed benchmark-pixel pairs with separations of
8.30-27.97 m (mean 21.26 m; median 23.93 m). Levelling observations from
2013-2015 and InSAR observations from 2017-2025 do not overlap in time.
Accordingly, correlation, RMSE and fitted curves are historical-trend
consistency diagnostics, not independent or contemporaneous accuracy metrics.

## Availability boundary

Large derived SBAS-InSAR time-series rasters, deformation rasters, future
subsidence rasters, spatial masks, observation-level six-site series and
third-party source datasets are not redistributed. Their aggregate outputs
and provenance are documented here. Original Sentinel, Copernicus, AR6, HKO,
IBTrACS, GSHHG, WorldCover, GADM and SSP products remain subject to their
providers' terms.

No synthetic scientific values are included. Machine-readable files were
copied or reformatted without numerical alteration from audited analysis
inputs, apart from explicitly documented factual corrections.

## Licence

Python code is released under the MIT License. Author-generated aggregate
tables, metadata and framework graphics are released under CC BY 4.0, subject
to the exclusions in `DATA_LICENSE.md`.

# Pearl Estuary connected-inundation analysis

Code and machine-readable aggregate tables supporting the manuscript:

> Land subsidence and vertical-datum uncertainty shape estimates of future connected coastal inundation in the Pearl River Estuary

Target journal: *Ocean & Coastal Management*.

## Scope

This repository contains scientific analysis and quality-control scripts for:

- IBTrACS storm selection and Hong Kong Observatory tide preparation;
- IPCC AR6 no-VLM sea-level extraction at Shek Pik;
- Shek Pik Chart Datum, HKPD and EGM2008 harmonisation;
- OLS and Huber continued-trend subsidence sensitivity analysis;
- eight-neighbour connected-inundation screening;
- area-weighted SSP population exposure;
- historical-event sensitivity analysis; and
- levelling/InSAR nearest-pixel quality control.

It also contains small aggregate CSV tables reported in or supporting the manuscript.

## Important reproducibility boundary

The processed SBAS-InSAR time series, deformation rasters, future-subsidence rasters,
subsidence-adjusted inundation masks and extracted point time series are intentionally
not distributed. HDF5, GeoTIFF and NPZ products are excluded.

Consequently, this is a partial reproducibility package. Public-source preparation and
table inspection are reproducible from the documented sources. Analyses that depend on
the omitted InSAR products require private inputs and cannot be reproduced end to end
from this repository alone.

Pure plotting scripts and manuscript-formatting utilities are excluded because they do
not implement scientific analysis.

## Repository layout

- scripts/data_preparation: public-source acquisition and datum preparation
- scripts/analysis: trend, inundation, population and event calculations
- scripts/validation: levelling and pixel-provenance quality control
- data/derived_tables: aggregate machine-readable results
- data/DATA_SOURCES.csv: source, version, identifier and redistribution status
- docs: reproducibility, availability and release guidance
- config/paths.example.json: example local input layout

The ignored directories external_data, private_inputs and outputs are created locally as
needed and are not versioned.

## Environment

Python 3.10 or later is recommended. Create the Conda environment with:

    conda env create -f environment.yml
    conda activate pearl-estuary-inundation

Or use a virtual environment:

    python -m venv .venv
    python -m pip install -r requirements.txt

## Running the workflow

Each script provides command-line options, for example:

    python scripts/data_preparation/filter_ibtracs_prd.py --help
    python scripts/data_preparation/extract_ar6_shekpik_verified.py --help
    python scripts/analysis/run_connected_inundation_shek_pik_egm2008.py --help
    python scripts/validation/leveling_nearest_pixel_qc.py --help

See docs/REPRODUCIBILITY.md for execution order, required inputs and output mapping.

## Data provenance

All third-party datasets are listed in data/DATA_SOURCES.csv. This repository does not
redistribute third-party rasters or archives. Obtain them from the original providers
and comply with the provider's licence and terms.

## Citation

After the first public release is archived by Zenodo, cite the version DOI assigned to
that release. GitHub can also display citation metadata from CITATION.cff.

Recommended publication route:

1. Publish this repository on GitHub.
2. Connect the public repository to Zenodo.
3. Create GitHub Release v1.0.0.
4. Use the resulting Zenodo version DOI in the manuscript.

Detailed Chinese instructions are in docs/GITHUB_ZENODO_STEPS_ZH.md.

## Licence

Original source code is released under the MIT License. Original aggregate tables are
made available under CC BY 4.0, subject to the exclusions and third-party rights stated
in DATA_LICENSE.md.

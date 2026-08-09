# Reproducibility guide

## Reproducibility level

This release supports three levels of inspection:

1. Public-source preparation: scripts can download or process the cited public
   AR6, HKO, IBTrACS and GSHHG sources.
2. Result inspection: all included aggregate CSV tables can be inspected
   without the omitted rasters.
3. Conditional rerun: inundation, future-subsidence and levelling analyses can
   be rerun only when the user supplies the private inputs listed below.

The repository does not provide full end-to-end reproduction of the
SBAS-InSAR processing.

## Local directory convention

- external_data: third-party files obtained from original providers.
- private_inputs: processed InSAR and local raster inputs not distributed.
- outputs: newly generated results.
- data/derived_tables: versioned aggregate tables shipped with the release.

The expected layout can be customised with command-line arguments. See
config/paths.example.json.

## Recommended execution order

### 1. Select historical storms from IBTrACS

Place the IBTrACS v04r01 global CSV under external_data/ibtracs or pass an
explicit path.

    python scripts/data_preparation/filter_ibtracs_prd.py

Main output: outputs/ibtracs_filter.

### 2. Download shoreline and HKO water-level inputs

    python scripts/data_preparation/download_coastal_storm_datum_inputs.py

This step uses the filtered closest-event table. The raw downloads are written
under external_data/coastal_hazard and are ignored by Git.

The smaller add_hko_astronomical_tide_to_ibtracs.py script is an optional
event-table preparation route when only predicted astronomical tide is needed.

### 3. Extract AR6 no-VLM projections

Download the official AR6 regional no-VLM NetCDF files and location list from
Zenodo release 10.5281/zenodo.6382554, then run:

    python scripts/data_preparation/extract_ar6_shekpik_verified.py

Main output: outputs/vertical_datum.

### 4. Build the Shek Pik station-datum offset

Supply official Hong Kong geodetic-control files and an EGM2008 grid:

    python scripts/data_preparation/build_shekpik_hkpd_egm2008_offset.py

### 5. Harmonise reference periods

Supply PSMSL monthly records, HKO predicted tides, the local mean-dynamic-
topography raster and datum-calibration provenance:

    python scripts/data_preparation/align_shek_pik_mdt_ar6_reference_period.py

The local h0 raster is not distributed.

### 6. Validate and extrapolate InSAR trends

Private input required:

- private_inputs/insar_settlement_timeseries_tc06_20170312_20251231.npz

    python scripts/analysis/robust_linear_validate_predict.py

The NPZ and generated subsidence GeoTIFFs are not distributed. Aggregate
validation and future-statistics tables are included under
data/derived_tables/trend_validation.

### 7. Run connected-inundation screening

Private inputs required:

- aligned Copernicus DEM;
- 2050 and 2100 OLS future-subsidence rasters; and
- optional Natural Earth comparison mask.

    python scripts/analysis/run_connected_inundation_shek_pik_egm2008.py --help

The main model uses eight-neighbour ocean connectivity. The output raster masks
are not distributed; aggregate area tables are included.

### 8. Calculate SSP population exposure

Required inputs:

- Wang et al. SSP2 and SSP5 population archives from Figshare; and
- private connected-inundation masks from step 7.

    python scripts/analysis/calculate_ssp_population_exposure.py --help

The main exposure estimate uses average resampling to calculate the flooded
fraction of each population cell. Nearest-neighbour counting is retained as a
sensitivity calculation.

### 9. Run historical-event sensitivity

Required inputs include the private DEM, land mask, future-subsidence rasters
and a locally obtained 2025 population raster. The archived manuscript result
used the recovered 11-event HKO summary at
external_data/coastal_hazard/processed/nested_archive_recovery/
storm_event_water_level_summary_recovered.csv.

    python scripts/analysis/run_historical_event_sensitivity.py --help

The model applies observed Shek Pik total water levels plus AR6 no-VLM
increments. It does not add the storm surge a second time.

### 10. Run levelling/InSAR validation

Private inputs required:

- original eight-point levelling observations;
- ERA5-corrected vertical-displacement HDF5 time series;
- optional coherence GeoTIFF; and
- for provenance auditing, the 249 vertical-displacement GeoTIFFs.

    python scripts/validation/leveling_nearest_pixel_qc.py --help
    python scripts/validation/audit_leveling_pixel_provenance.py --help

The HDF5, GeoTIFF and point time-series inputs are not distributed. Only the
selected-point statistics and distance summaries are included.

## Scientific scripts and manuscript outputs

- robust_linear_validate_predict.py supports the OLS/Huber validation and
  future-subsidence sensitivity tables.
- run_connected_inundation_shek_pik_egm2008.py supports the main flood-area,
  vertical-datum and shoreline-sensitivity tables.
- calculate_ssp_population_exposure.py supports the projected exposure tables.
- run_historical_event_sensitivity.py supports the historical-event
  sensitivity table.
- leveling_nearest_pixel_qc.py supports the levelling comparison table and
  pixel-distance statistics.

## Excluded figure generation

Figure-only scripts, ArcGIS styling files and document rendering scripts are
not part of this release. The analysis tables required to recreate scientific
plots are included where they do not expose the omitted core InSAR products.

## Integrity

MANIFEST.sha256 records checksums for all release files except the manifest
itself. Recompute it after any change and before creating a new release.

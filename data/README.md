# Data contents

## What is included

The derived_tables directory contains 20 small CSV files. These are aggregate
scenario inputs, summary statistics and quality-control outputs. They do not
contain raster cell arrays or full InSAR point time series.

### vertical_datum

- sea_level_scenarios_shekpik_ar6_novlm_verified.csv: verified AR6 no-VLM
  increments at Shek Pik.
- sea_level_scenarios_shek_pik_absolute_egm2008.csv: absolute EGM2008 water
  levels after station-datum and reference-period harmonisation.
- shek_pik_psmsl_annual_means.csv: annual PSMSL station statistics.
- station_datum_offsets_shek_pik_verified.csv: Chart Datum to EGM2008 offset
  and provenance summary.

### connected_inundation

- scenario_inputs_shek_pik_egm2008_completed.csv: model scenario inputs.
- flood_area_summary_shek_pik_egm2008_main.csv: main connected-inundation area
  results.
- flood_area_datum_sensitivity_shek_pik_egm2008.csv: datum uncertainty
  sensitivity.
- flood_area_shoreline_sensitivity_shek_pik_egm2008.csv: shoreline-mask
  sensitivity.
- shoreline_mask_comparison.csv: aggregate mask-area comparison.

### population_exposure

- population_exposure_shek_pik_egm2008_long.csv: long-format exposure results.
- population_exposure_shek_pik_egm2008_summary.csv: scenario summary.

### historical_events

- all_11_events_shek_pik_egm2008_scenarios.csv: 44 combinations from 11
  historical events, two scenarios and two target years.
- absolute_event_inundation_population_summary_shek_pik_egm2008.csv: area and
  population results for all event combinations.
- absolute_event_scenarios_manuscript_table_shek_pik_egm2008.csv: compact
  manuscript table for representative events.

### trend_validation

- validation_summary.csv: OLS and Huber validation metrics.
- future_prediction_stats.csv: aggregate future-subsidence scenario
  statistics.
- signed_vlm_main_scenario_sensitivity.csv: positive-only versus signed-VLM
  sensitivity.

### leveling_validation

- best6_quality_selected_stats.csv: statistics for the six retained levelling
  locations.
- best6_renumbering.csv: final point-number mapping.
- pixel_distance_summary.csv: minimum, maximum, mean and median pixel distance.

## Units and missing values

Units are encoded in column names where practical, including suffixes such as
_m, _mm, _km2, _mm_yr and _million. Blank CSV cells represent unavailable or
not-applicable values unless a script-specific validation rule states
otherwise.

## Path sanitisation

Local drive paths in the original calculation outputs were replaced by file
basenames or explicit not-distributed labels. No numerical result was changed.

## What is excluded

- HDF5 and NPZ time-series products;
- GeoTIFF deformation, DEM, population, shoreline and inundation rasters;
- future-subsidence rasters;
- subsidence-derived spatial masks;
- candidate-pixel and extracted point time-series tables; and
- third-party source archives.

See DATA_SOURCES.csv for source providers, versions and access routes.

from __future__ import annotations

import argparse
import calendar
import csv
import io
import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "external_data" / "coastal_hazard"
DEFAULT_EVENTS_CSV = (
    REPO_ROOT
    / "outputs"
    / "ibtracs_filter"
    / "ibtracs_prd_storm_summary_2017_2025_500km_closest11.csv"
)

GSHHG_URLS = [
    "https://github.com/GenericMappingTools/gshhg-gmt/releases/download/2.3.7/gshhg-shp-2.3.7.zip",
    "https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/gshhg-shp-2.3.7.zip",
    "https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip",
]
HKO_PREDICTION_API = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
HKO_OBSERVED_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/tide/ALL_en.csv"
ARCHIVE_LIST_API = "https://app.data.gov.hk/v1/historical-archive/list-file-versions"
ARCHIVE_GET_API = "https://app.data.gov.hk/v1/historical-archive/get-file"

OFFICIAL_URLS = {
    "hko_tide_notes": "https://www.hko.gov.hk/en/tide/enotes.htm",
    "hko_open_data": "https://www.hko.gov.hk/en/abouthko/opendata_intro.htm",
    "hko_api_documentation": "https://data.weather.gov.hk/weatherAPI/doc/HKO_Open_Data_API_Documentation.pdf",
    "data_gov_hk_archive_api": "https://data.gov.hk/en/help/api-spec",
    "copernicus_dem_collection": "https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM",
    "psmsl_macau_station_269": "https://psmsl.org/data/obtaining/stations/269.php",
    "psmsl_macau_datum_269": "https://psmsl.org/data/obtaining/rlr.diagrams/269.php",
    "psmsl_shek_pik_station_1902": "https://psmsl.org/data/obtaining/stations/1902.php",
    "psmsl_shek_pik_datum_1902": "https://psmsl.org/data/obtaining/rlr.diagrams/1902.php",
}


@dataclass
class ManifestRow:
    item: str
    status: str
    path: str
    url: str
    bytes: int = 0
    note: str = ""


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "PRD-coastal-hazard-reproducibility/1.0"})
    return s


def stream_download(
    s: requests.Session,
    url: str,
    target: Path,
    force: bool = False,
    timeout: int = 120,
) -> ManifestRow:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not force:
        return ManifestRow(target.stem, "exists", str(target), url, target.stat().st_size)

    part = target.with_suffix(target.suffix + ".part")
    existing = part.stat().st_size if part.exists() and not force else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    try:
        with s.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as response:
            response.raise_for_status()
            if existing and response.status_code != 206:
                existing = 0
                mode = "wb"
            with part.open(mode) as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        part.replace(target)
        return ManifestRow(target.stem, "downloaded", str(target), url, target.stat().st_size)
    except Exception as exc:
        return ManifestRow(target.stem, "failed", str(target), url, 0, repr(exc))


def download_first_available(
    s: requests.Session,
    urls: list[str],
    target: Path,
    force: bool,
) -> ManifestRow:
    if target.exists() and not force:
        try:
            with zipfile.ZipFile(target) as archive:
                if archive.testzip() is None and any(name.lower().endswith("gshhs_f_l1.shp") for name in archive.namelist()):
                    return ManifestRow(
                        "gshhg_shapefiles_2_3_7", "exists_validated", str(target), urls[0], target.stat().st_size
                    )
        except (zipfile.BadZipFile, OSError):
            return ManifestRow(
                "gshhg_shapefiles_2_3_7",
                "incomplete",
                str(target),
                urls[0],
                target.stat().st_size,
                "Existing ZIP is incomplete. Resume it with curl -C - before running shoreline processing.",
            )
    failures = []
    for url in urls:
        row = stream_download(s, url, target, force=force)
        if row.status in {"downloaded", "exists"}:
            row.item = "gshhg_shapefiles_2_3_7"
            return row
        failures.append(f"{url}: {row.note}")
    return ManifestRow("gshhg_shapefiles_2_3_7", "failed", str(target), urls[-1], note=" | ".join(failures))


def fetch_small(
    s: requests.Session,
    url: str,
    target: Path,
    force: bool,
    params: dict | None = None,
) -> ManifestRow:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not force:
        return ManifestRow(target.stem, "exists", str(target), url, target.stat().st_size)
    try:
        response = s.get(url, params=params, timeout=90, allow_redirects=True)
        response.raise_for_status()
        target.write_bytes(response.content)
        return ManifestRow(target.stem, "downloaded", str(target), response.url, len(response.content))
    except Exception as exc:
        return ManifestRow(target.stem, "failed", str(target), url, 0, repr(exc))


def load_events(events_csv: Path) -> pd.DataFrame:
    events = pd.read_csv(events_csv)
    events["event_time_utc"] = pd.to_datetime(events["time_at_min_dist"], utc=True)
    events["event_time_hkt"] = events["event_time_utc"].dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
    return events


def required_months(events: pd.DataFrame, window_hours: int) -> list[str]:
    months: set[str] = set()
    delta = pd.Timedelta(hours=window_hours)
    for t in events["event_time_hkt"]:
        start = t - delta
        end = t + delta
        cursor = pd.Timestamp(start.year, start.month, 1)
        last = pd.Timestamp(end.year, end.month, 1)
        while cursor <= last:
            months.add(cursor.strftime("%Y%m"))
            cursor = cursor + pd.offsets.MonthBegin(1)
    return sorted(months)


def download_hko_predictions(
    s: requests.Session,
    events: pd.DataFrame,
    out: Path,
    force: bool,
) -> list[ManifestRow]:
    rows = []
    for year in sorted(events["event_time_hkt"].dt.year.unique()):
        target = out / f"SPW_HHOT_{year}.csv"
        row = fetch_small(
            s,
            HKO_PREDICTION_API,
            target,
            force,
            params={"dataType": "HHOT", "station": "SPW", "year": int(year), "rformat": "csv"},
        )
        row.item = f"hko_spw_predicted_tide_{year}"
        rows.append(row)
    return rows


def month_bounds(month: str) -> tuple[str, str]:
    year = int(month[:4])
    mon = int(month[4:])
    return f"{year:04d}{mon:02d}01", f"{year:04d}{mon:02d}{calendar.monthrange(year, mon)[1]:02d}"


def download_hko_observed_archives(
    s: requests.Session,
    months: list[str],
    out: Path,
    force: bool,
) -> list[ManifestRow]:
    rows = []
    for month in months:
        start, end = month_bounds(month)
        metadata_path = out / f"SPW_observed_archive_{month}_index.json"
        try:
            response = s.get(
                ARCHIVE_LIST_API,
                params={"url": HKO_OBSERVED_URL, "start": start, "end": end},
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            metadata_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            monthly = [item for item in data.get("data-files", []) if item.get("period") == "M"]
            if not monthly:
                rows.append(
                    ManifestRow(
                        f"hko_observed_archive_{month}",
                        "unavailable",
                        str(metadata_path),
                        response.url,
                        note="No monthly archive is available for this period.",
                    )
                )
                continue
            item = monthly[0]
            timestamp = str(item["timestamp"])
            target = out / str(item["filename"])
            row = stream_download(
                s,
                ARCHIVE_GET_API + f"?url={requests.utils.quote(HKO_OBSERVED_URL, safe='')}&time={timestamp}",
                target,
                force=force,
                timeout=180,
            )
            row.item = f"hko_observed_archive_{month}"
            row.note = f"Monthly archive, {item.get('resource_file_count', '')} snapshots. " + row.note
            rows.append(row)
        except Exception as exc:
            rows.append(
                ManifestRow(
                    f"hko_observed_archive_{month}",
                    "failed",
                    str(metadata_path),
                    ARCHIVE_LIST_API,
                    note=repr(exc),
                )
            )
    return rows


def parse_hhot_year(path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    hour_columns = [f"{i:02d}" for i in range(1, 25)]
    long = frame.melt(id_vars=["MM", "DD"], value_vars=hour_columns, var_name="hour", value_name="predicted_m")
    long["predicted_m"] = pd.to_numeric(long["predicted_m"], errors="coerce")
    long["datetime_hkt"] = pd.to_datetime(
        {
            "year": year,
            "month": long["MM"].astype(int),
            "day": long["DD"].astype(int),
            "hour": long["hour"].astype(int) - 1,
        },
        errors="coerce",
    )
    return long.dropna(subset=["datetime_hkt", "predicted_m"])[["datetime_hkt", "predicted_m"]].sort_values("datetime_hkt")


def parse_observed_archives(archives: list[Path], events: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    windows = [
        (row.SID, row.event_time_hkt - pd.Timedelta(hours=window_hours), row.event_time_hkt + pd.Timedelta(hours=window_hours))
        for row in events.itertuples()
    ]
    rows = []
    timestamp_re = re.compile(r"(\d{8}-\d{4})-ALL_en\.csv$", re.IGNORECASE)
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as zf:
                for name in zf.namelist():
                    match = timestamp_re.search(name.replace("\\", "/"))
                    if not match:
                        continue
                    snapshot = pd.to_datetime(match.group(1), format="%Y%m%d-%H%M", errors="coerce")
                    if pd.isna(snapshot) or not any(start <= snapshot <= end for _, start, end in windows):
                        continue
                    raw = zf.read(name).decode("utf-8-sig", errors="replace")
                    for record in csv.DictReader(io.StringIO(raw)):
                        if (record.get("Tide Station") or "").strip().lower() != "shek pik":
                            continue
                        height = pd.to_numeric(record.get("Height(m)"), errors="coerce")
                        measured_time = pd.to_datetime(
                            f"{record.get('Date', '')} {record.get('Time', '')}", errors="coerce"
                        )
                        if pd.notna(height) and pd.notna(measured_time):
                            rows.append(
                                {
                                    "datetime_hkt": measured_time,
                                    "observed_total_level_m_cd": float(height),
                                    "archive_snapshot_hkt": snapshot,
                                    "archive": archive.name,
                                }
                            )
        except zipfile.BadZipFile:
            continue
    if not rows:
        return pd.DataFrame(columns=["datetime_hkt", "observed_total_level_m_cd", "archive_snapshot_hkt", "archive"])
    result = pd.DataFrame(rows).sort_values(["datetime_hkt", "archive_snapshot_hkt"])
    return result.drop_duplicates("datetime_hkt", keep="last").reset_index(drop=True)


def interpolate_predictions(observed: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    if observed.empty:
        return observed.assign(predicted_tide_m_cd=np.nan, surge_residual_m=np.nan)
    series = predicted.drop_duplicates("datetime_hkt").set_index("datetime_hkt")["predicted_m"].sort_index()
    target = pd.DatetimeIndex(observed["datetime_hkt"])
    expanded = series.reindex(series.index.union(target)).sort_index().interpolate(method="time")
    out = observed.copy()
    out["predicted_tide_m_cd"] = expanded.reindex(target).to_numpy()
    out["surge_residual_m"] = out["observed_total_level_m_cd"] - out["predicted_tide_m_cd"]
    return out


def summarize_events(events: pd.DataFrame, observed: pd.DataFrame, predicted: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    combined = interpolate_predictions(observed, predicted)
    summaries = []
    for event in events.itertuples():
        start = event.event_time_hkt - pd.Timedelta(hours=window_hours)
        end = event.event_time_hkt + pd.Timedelta(hours=window_hours)
        sub = combined[(combined["datetime_hkt"] >= start) & (combined["datetime_hkt"] <= end)].copy()
        record = event._asdict()
        record["window_start_hkt"] = start
        record["window_end_hkt"] = end
        record["observed_points"] = len(sub)
        record["water_level_datum"] = "HKO Chart Datum"
        if sub.empty:
            record["status"] = "no_historical_observed_archive"
        else:
            peak = sub.loc[sub["observed_total_level_m_cd"].idxmax()]
            surge_peak = sub.loc[sub["surge_residual_m"].idxmax()]
            record.update(
                {
                    "status": "observed_and_predicted_available",
                    "max_observed_total_level_m_cd": peak["observed_total_level_m_cd"],
                    "max_observed_time_hkt": peak["datetime_hkt"],
                    "predicted_tide_at_observed_peak_m_cd": peak["predicted_tide_m_cd"],
                    "surge_residual_at_observed_peak_m": peak["surge_residual_m"],
                    "max_surge_residual_m": surge_peak["surge_residual_m"],
                    "max_surge_residual_time_hkt": surge_peak["datetime_hkt"],
                    "observed_level_at_max_residual_m_cd": surge_peak["observed_total_level_m_cd"],
                }
            )
        summaries.append(record)
    return pd.DataFrame(summaries)


def cache_official_sources(s: requests.Session, out: Path, force: bool) -> list[ManifestRow]:
    rows = []
    for name, url in OFFICIAL_URLS.items():
        suffix = ".pdf" if url.lower().endswith(".pdf") else ".html"
        row = fetch_small(s, url, out / f"{name}{suffix}", force)
        row.item = name
        rows.append(row)
    return rows


def download_psmsl_diagram_assets(s: requests.Session, official_dir: Path, force: bool) -> list[ManifestRow]:
    rows = []
    for station_id, label in [(269, "macau"), (1902, "shek_pik")]:
        page = official_dir / f"psmsl_{label}_datum_{station_id}.html"
        if not page.exists():
            continue
        text = page.read_text("utf-8", errors="replace")
        for href in re.findall(r"(?:src|href)=[\"']([^\"']+)[\"']", text, flags=re.IGNORECASE):
            if not re.search(r"\.(?:png|gif|jpe?g|pdf)(?:\?|$)", href, flags=re.IGNORECASE):
                continue
            url = urljoin(f"https://psmsl.org/data/obtaining/rlr.diagrams/{station_id}.php", href)
            suffix = Path(href.split("?")[0]).suffix or ".bin"
            target = official_dir / f"psmsl_{label}_{station_id}_datum_asset{suffix}"
            row = fetch_small(s, url, target, force)
            row.item = f"psmsl_{label}_{station_id}_datum_asset"
            rows.append(row)
            break
    return rows


def write_vertical_datum_chain(out: Path) -> None:
    rows = [
        {
            "station": "SHEK_PIK",
            "source_water_datum": "HKO Chart Datum",
            "intermediate_datum": "Hong Kong Principal Datum (HKPD)",
            "offset_source_to_intermediate_m": -0.146,
            "offset_intermediate_to_egm2008_m": "",
            "total_offset_to_egm2008_m": "",
            "formula": "H_HKPD = H_CD - 0.146; H_EGM2008 = H_HKPD + offset_HKPD_to_EGM2008",
            "status": "CD_to_HKPD_official; HKPD_to_EGM2008_unresolved",
            "source": "HKO Tide Notes: Chart Datum is 0.146 m below HKPD",
        },
        {
            "station": "MACAU",
            "source_water_datum": "PSMSL RLR / local tide datum",
            "intermediate_datum": "BM II / Macao Height Datum",
            "offset_source_to_intermediate_m": "",
            "offset_intermediate_to_egm2008_m": "",
            "total_offset_to_egm2008_m": "",
            "formula": "H_EGM2008 = H_reported + verified station-to-local offset + verified local-to-EGM2008 offset",
            "status": "requires_PSMLS_diagram_and_DSCC_benchmark_verification",
            "source": "PSMSL station/datum diagram; DSCC benchmark information required",
        },
    ]
    pd.DataFrame(rows).to_csv(out / "station_vertical_datum_chain.csv", index=False, encoding="utf-8-sig")


def write_manifest(rows: list[ManifestRow], out: Path) -> None:
    pd.DataFrame([asdict(row) for row in rows]).to_csv(out / "download_manifest.csv", index=False, encoding="utf-8-sig")
    (out / "download_manifest.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_CSV)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window-hours", type=int, default=48)
    parser.add_argument("--skip-gshhg", action="store_true", help="Skip the large GSHHG shoreline package.")
    parser.add_argument("--skip-hko-observed", action="store_true", help="Skip monthly historical observed-tide archives.")
    parser.add_argument("--skip-official", action="store_true", help="Skip official datum/source pages.")
    args = parser.parse_args()

    output = args.output
    raw_coast = output / "raw" / "gshhg"
    predicted_dir = output / "raw" / "hko_predicted_tide"
    observed_dir = output / "raw" / "hko_observed_archive"
    official_dir = output / "raw" / "official_datum_sources"
    processed_dir = output / "processed"
    for folder in [output, raw_coast, predicted_dir, observed_dir, official_dir, processed_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    if not args.events_csv.is_file():
        raise FileNotFoundError(
            f"Filtered IBTrACS event table not found: {args.events_csv}. "
            "Run filter_ibtracs_prd.py first or pass --events-csv."
        )
    events = load_events(args.events_csv)
    months = required_months(events, args.window_hours)
    s = session()
    manifest: list[ManifestRow] = []

    if not args.skip_gshhg:
        print("Downloading GSHHG shoreline package...")
        manifest.append(
            download_first_available(s, GSHHG_URLS, raw_coast / "gshhg-shp-2.3.7.zip", args.force)
        )
    else:
        manifest.append(ManifestRow("gshhg_shapefiles_2_3_7", "skipped", str(raw_coast), GSHHG_URLS[0]))

    print("Downloading HKO predicted tides...")
    manifest.extend(download_hko_predictions(s, events, predicted_dir, args.force))

    if not args.skip_hko_observed:
        print("Downloading HKO historical observed-tide archives...")
        manifest.extend(download_hko_observed_archives(s, months, observed_dir, args.force))
    else:
        manifest.append(ManifestRow("hko_observed_archives", "skipped", str(observed_dir), ARCHIVE_LIST_API))

    if not args.skip_official:
        print("Downloading official vertical-datum sources...")
        manifest.extend(cache_official_sources(s, official_dir, args.force))
        manifest.extend(download_psmsl_diagram_assets(s, official_dir, args.force))

    predicted_frames = []
    for path in sorted(predicted_dir.glob("SPW_HHOT_*.csv")):
        year_match = re.search(r"(\d{4})", path.stem)
        if year_match:
            predicted_frames.append(parse_hhot_year(path, int(year_match.group(1))))
    predicted = pd.concat(predicted_frames, ignore_index=True).sort_values("datetime_hkt")
    predicted.to_csv(processed_dir / "hko_spw_predicted_tide_long.csv", index=False, encoding="utf-8-sig")

    archives = sorted(observed_dir.glob("*.zip"))
    observed = parse_observed_archives(archives, events, args.window_hours)
    combined = interpolate_predictions(observed, predicted)
    combined.to_csv(processed_dir / "hko_spw_observed_predicted_surges.csv", index=False, encoding="utf-8-sig")
    summary = summarize_events(events, observed, predicted, args.window_hours)
    summary.to_csv(processed_dir / "storm_event_water_level_summary.csv", index=False, encoding="utf-8-sig")
    write_vertical_datum_chain(processed_dir)
    write_manifest(manifest, output)

    status_counts = pd.Series([row.status for row in manifest]).value_counts().to_dict()
    print("Output:", output)
    print("Status:", status_counts)
    print("Observed tide points:", len(observed))
    print("Events with observed data:", int((summary["status"] == "observed_and_predicted_available").sum()))
    print("Event summary:", processed_dir / "storm_event_water_level_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

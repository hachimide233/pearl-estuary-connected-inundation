import csv
import argparse
import math
from pathlib import Path

# ====== 1. Input/output paths ======
repo_root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(
    description="Filter IBTrACS v04r01 tracks around the Pearl River Delta."
)
parser.add_argument(
    "--input",
    type=Path,
    default=repo_root / "external_data" / "ibtracs" / "ibtracs.ALL.list.v04r01.csv",
)
parser.add_argument("--out-dir", type=Path, default=repo_root / "outputs" / "ibtracs_filter")
parser.add_argument("--center-lon", type=float, default=113.55)
parser.add_argument("--center-lat", type=float, default=22.20)
parser.add_argument("--radius-km", type=float, default=500.0)
parser.add_argument("--year-min", type=int, default=2017)
parser.add_argument("--year-max", type=int, default=2025)
parser.add_argument("--closest-count", type=int, default=11)
parser.add_argument("--lon-min", type=float, default=112.0529938)
parser.add_argument("--lon-max", type=float, default=114.7303744)
parser.add_argument("--lat-min", type=float, default=21.2276408)
parser.add_argument("--lat-max", type=float, default=22.7856837)
args = parser.parse_args()

in_file = args.input
out_dir = args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

period_tag = f"{args.year_min}_{args.year_max}"
radius_tag = f"{args.radius_km:g}km"
out_tracks_500 = out_dir / f"ibtracs_prd_track_points_{period_tag}_{radius_tag}.csv"
out_summary_500 = out_dir / f"ibtracs_prd_storm_summary_{period_tag}_{radius_tag}.csv"
out_summary_closest11 = out_dir / (
    f"ibtracs_prd_storm_summary_{period_tag}_{radius_tag}_closest{args.closest_count}.csv"
)
out_tracks_bbox = out_dir / f"ibtracs_prd_track_points_inside_insar_bbox_{period_tag}.csv"
out_readme = out_dir / "README_ibtracs_prd_filter.txt"

# ====== 2. Study area settings ======
# Pearl River Delta / Hengqin-Macau approximate center
center_lon = args.center_lon
center_lat = args.center_lat
radius_km = args.radius_km

year_min = args.year_min
year_max = args.year_max

# InSAR bounding box from your MintPy GeoTIFF extent
bbox = {
    "lon_min": args.lon_min,
    "lon_max": args.lon_max,
    "lat_min": args.lat_min,
    "lat_max": args.lat_max,
}

# ====== 3. Helper functions ======
def to_float(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def to_int(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def norm_lon(lon):
    if lon is None:
        return None
    if lon > 180:
        return lon - 360
    return lon


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def in_bbox(lon, lat):
    return (
        bbox["lon_min"] <= lon <= bbox["lon_max"]
        and bbox["lat_min"] <= lat <= bbox["lat_max"]
    )


def best_value(row, fields):
    for field in fields:
        value = to_float(row.get(field))
        if value is not None:
            return value
    return None


# Around the western North Pacific/South China Sea, HKO/CMA/TOKYO often have useful values.
wind_fields = ["WMO_WIND", "HKO_WIND", "CMA_WIND", "TOKYO_WIND", "USA_WIND"]
pres_fields = ["WMO_PRES", "HKO_PRES", "CMA_PRES", "TOKYO_PRES", "USA_PRES"]

track_rows_500 = []
track_rows_bbox = []
summary = {}

print("reading:", in_file)
if not in_file.exists():
    raise FileNotFoundError(in_file)

with in_file.open("r", encoding="utf-8", errors="ignore", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # IBTrACS second row is a unit row: SEASON=Year, LAT=degrees_north, etc.
        sid = str(row.get("SID", "")).strip()
        season_raw = str(row.get("SEASON", "")).strip()
        iso_time = str(row.get("ISO_TIME", "")).strip()

        if sid == "" or sid.upper() == "SID" or season_raw.lower() == "year":
            continue

        year = to_int(season_raw)
        if year is None or not (year_min <= year <= year_max):
            continue

        lat = to_float(row.get("LAT"))
        lon = norm_lon(to_float(row.get("LON")))
        if lat is None or lon is None:
            continue

        dist = haversine_km(center_lon, center_lat, lon, lat)
        inside_bbox = in_bbox(lon, lat)

        # Keep storms that are either near the PRD center or directly inside the InSAR bbox.
        if dist > radius_km and not inside_bbox:
            continue

        name = str(row.get("NAME", "")).strip()
        basin = str(row.get("BASIN", "")).strip()
        subbasin = str(row.get("SUBBASIN", "")).strip()
        nature = str(row.get("NATURE", "")).strip()
        track_type = str(row.get("TRACK_TYPE", "")).strip()
        dist2land = to_float(row.get("DIST2LAND"))
        landfall = to_float(row.get("LANDFALL"))
        wind = best_value(row, wind_fields)
        pres = best_value(row, pres_fields)
        storm_speed = to_float(row.get("STORM_SPEED"))
        storm_dir = to_float(row.get("STORM_DIR"))

        out_row = {
            "SID": sid,
            "SEASON": year,
            "NAME": name,
            "ISO_TIME": iso_time,
            "BASIN": basin,
            "SUBBASIN": subbasin,
            "NATURE": nature,
            "TRACK_TYPE": track_type,
            "LAT": round(lat, 4),
            "LON": round(lon, 4),
            "DIST_KM_TO_PRD_CENTER": round(dist, 3),
            "IN_INSAR_BBOX": int(inside_bbox),
            "DIST2LAND_KM": "" if dist2land is None else dist2land,
            "LANDFALL_KM": "" if landfall is None else landfall,
            "BEST_WIND_KT": "" if wind is None else wind,
            "BEST_PRES_MB": "" if pres is None else pres,
            "STORM_SPEED_KT": "" if storm_speed is None else storm_speed,
            "STORM_DIR_DEG": "" if storm_dir is None else storm_dir,
        }

        if dist <= radius_km:
            track_rows_500.append(out_row)
        if inside_bbox:
            track_rows_bbox.append(out_row)

        if sid not in summary:
            summary[sid] = {
                "SID": sid,
                "SEASON": year,
                "NAME": name,
                "BASIN": basin,
                "SUBBASIN": subbasin,
                "min_dist_km": dist,
                "time_at_min_dist": iso_time,
                "lat_at_min_dist": lat,
                "lon_at_min_dist": lon,
                "max_wind_kt": wind,
                "time_at_max_wind": iso_time if wind is not None else "",
                "min_pres_mb": pres,
                "time_at_min_pres": iso_time if pres is not None else "",
                "track_points_within_500km": 0,
                "points_inside_insar_bbox": 0,
            }

        s = summary[sid]
        if dist <= radius_km:
            s["track_points_within_500km"] += 1
        if inside_bbox:
            s["points_inside_insar_bbox"] += 1

        if dist < s["min_dist_km"]:
            s["min_dist_km"] = dist
            s["time_at_min_dist"] = iso_time
            s["lat_at_min_dist"] = lat
            s["lon_at_min_dist"] = lon

        if wind is not None and (s["max_wind_kt"] is None or wind > s["max_wind_kt"]):
            s["max_wind_kt"] = wind
            s["time_at_max_wind"] = iso_time

        if pres is not None and (s["min_pres_mb"] is None or pres < s["min_pres_mb"]):
            s["min_pres_mb"] = pres
            s["time_at_min_pres"] = iso_time


track_fields = [
    "SID", "SEASON", "NAME", "ISO_TIME", "BASIN", "SUBBASIN", "NATURE", "TRACK_TYPE",
    "LAT", "LON", "DIST_KM_TO_PRD_CENTER", "IN_INSAR_BBOX", "DIST2LAND_KM", "LANDFALL_KM",
    "BEST_WIND_KT", "BEST_PRES_MB", "STORM_SPEED_KT", "STORM_DIR_DEG",
]

with out_tracks_500.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=track_fields)
    writer.writeheader()
    writer.writerows(track_rows_500)

with out_tracks_bbox.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=track_fields)
    writer.writeheader()
    writer.writerows(track_rows_bbox)

summary_rows = list(summary.values())
summary_rows.sort(key=lambda x: (x["SEASON"], x["min_dist_km"]))

summary_fields = [
    "SID", "SEASON", "NAME", "BASIN", "SUBBASIN",
    "min_dist_km", "time_at_min_dist", "lat_at_min_dist", "lon_at_min_dist",
    "max_wind_kt", "time_at_max_wind", "min_pres_mb", "time_at_min_pres",
    "track_points_within_500km", "points_inside_insar_bbox",
]

with out_summary_500.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    for item in summary_rows:
        item = dict(item)
        item["min_dist_km"] = round(item["min_dist_km"], 3)
        item["lat_at_min_dist"] = round(item["lat_at_min_dist"], 4)
        item["lon_at_min_dist"] = round(item["lon_at_min_dist"], 4)
        if item["max_wind_kt"] is None:
            item["max_wind_kt"] = ""
        if item["min_pres_mb"] is None:
            item["min_pres_mb"] = ""
        writer.writerow(item)

closest_rows = sorted(summary.values(), key=lambda x: x["min_dist_km"])[: args.closest_count]
with out_summary_closest11.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    for item in closest_rows:
        item = dict(item)
        item["min_dist_km"] = round(item["min_dist_km"], 3)
        item["lat_at_min_dist"] = round(item["lat_at_min_dist"], 4)
        item["lon_at_min_dist"] = round(item["lon_at_min_dist"], 4)
        if item["max_wind_kt"] is None:
            item["max_wind_kt"] = ""
        if item["min_pres_mb"] is None:
            item["min_pres_mb"] = ""
        writer.writerow(item)

out_readme.write_text(
    "IBTrACS PRD filter outputs\n\n"
    f"Input: {in_file}\n"
    f"Period: {year_min}-{year_max}\n"
    f"Study center: lon={center_lon}, lat={center_lat}\n"
    f"Search radius: {radius_km:g} km\n"
    f"InSAR bbox: lon {bbox['lon_min']}-{bbox['lon_max']}, "
    f"lat {bbox['lat_min']}-{bbox['lat_max']}\n\n"
    "Files:\n"
    f"1. {out_tracks_500.name}\n"
    "   All storm track points within 500 km of the PRD/Hengqin-Macau center. Use for path plotting.\n\n"
    f"2. {out_summary_500.name}\n"
    "   One row per storm. Use min_dist_km, max_wind_kt, min_pres_mb to select representative events.\n\n"
    f"3. {out_summary_closest11.name}\n"
    f"   The {args.closest_count} closest storms by min_dist_km.\n\n"
    f"4. {out_tracks_bbox.name}\n"
    "   Track points whose storm centers entered the exact InSAR bounding box. This is stricter than 500 km.\n\n"
    "Note: IBTrACS does not directly provide storm surge height. It helps select historical typhoon cases.\n"
    "For inundation modeling, combine selected storms with tide-gauge surge levels or scenario water levels.\n",
    encoding="utf-8",
)

print("track points within 500 km:", len(track_rows_500))
print("track points inside InSAR bbox:", len(track_rows_bbox))
print("storms selected:", len(summary_rows))
print("wrote:", out_tracks_500)
print("wrote:", out_summary_500)
print("wrote:", out_summary_closest11)
print("wrote:", out_tracks_bbox)
print("wrote:", out_readme)

print("\nClosest 30 storms by min_dist_km:")
for item in sorted(summary.values(), key=lambda x: x["min_dist_km"])[:30]:
    print(
        item["SEASON"], item["NAME"], item["SID"],
        "min_dist_km=", round(item["min_dist_km"], 1),
        "time=", item["time_at_min_dist"],
        "max_wind_kt=", item["max_wind_kt"],
        "min_pres_mb=", item["min_pres_mb"],
        "bbox_points=", item["points_inside_insar_bbox"],
    )

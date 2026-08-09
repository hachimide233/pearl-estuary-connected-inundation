from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = (
    REPO_ROOT
    / "outputs"
    / "ibtracs_filter"
    / "ibtracs_prd_storm_summary_2017_2025_500km_closest11.csv"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "outputs"
    / "ibtracs_filter"
    / "ibtracs_prd_storm_summary_2017_2025_500km_closest11_hko_astronomical_tide.csv"
)


def hko_hhot_url(year: int, station: str) -> str:
    return (
        "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
        f"?dataType=HHOT&rformat=csv&station={station}&year={year}"
    )


def load_hko_hhot_year(year: int, station: str, hour_mode: str) -> pd.DataFrame:
    """Load HKO hourly astronomical tide predictions and return long-form rows.

    HKO HHOT CSV uses columns MM, DD, 01, ..., 24.
    The default mapping used here is zero_based: 01 -> 00:00 and 24 -> 23:00.
    Use --hour-mode one_based if you want 01 -> 01:00 and 24 -> next-day 00:00.
    """
    df = pd.read_csv(hko_hhot_url(year, station))
    required = {"MM", "DD"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"HKO CSV for {station} {year} is missing columns: {sorted(missing)}")

    hour_cols = [f"{i:02d}" for i in range(1, 25)]
    missing_hours = [col for col in hour_cols if col not in df.columns]
    if missing_hours:
        raise ValueError(f"HKO CSV for {station} {year} is missing hour columns: {missing_hours}")

    long = df.melt(id_vars=["MM", "DD"], value_vars=hour_cols, var_name="hko_hour_col", value_name="tide_m")
    long["tide_m"] = pd.to_numeric(long["tide_m"], errors="coerce")
    long = long.dropna(subset=["tide_m"])

    col_num = long["hko_hour_col"].astype(int)
    if hour_mode == "zero_based":
        hour = col_num - 1
        day_offset = 0
    elif hour_mode == "one_based":
        hour = col_num % 24
        day_offset = (col_num == 24).astype(int)
    else:
        raise ValueError(f"Unsupported hour_mode: {hour_mode}")

    long["datetime_hkt"] = pd.to_datetime(
        {
            "year": year,
            "month": long["MM"].astype(int),
            "day": long["DD"].astype(int),
            "hour": hour,
        },
        errors="coerce",
    )
    if hour_mode == "one_based":
        long["datetime_hkt"] = long["datetime_hkt"] + pd.to_timedelta(day_offset, unit="D")

    long = long.dropna(subset=["datetime_hkt"]).sort_values("datetime_hkt")
    return long[["datetime_hkt", "tide_m", "hko_hour_col"]].reset_index(drop=True)


def nearest_tide(tide: pd.DataFrame, event_time_hkt: pd.Timestamp) -> tuple[float, pd.Timestamp, float, str]:
    deltas = (tide["datetime_hkt"] - event_time_hkt).abs()
    idx = deltas.idxmin()
    matched_time = tide.loc[idx, "datetime_hkt"]
    diff_hours = (matched_time - event_time_hkt).total_seconds() / 3600.0
    return float(tide.loc[idx, "tide_m"]), matched_time, diff_hours, str(tide.loc[idx, "hko_hour_col"])


def interpolated_tide(tide: pd.DataFrame, event_time_hkt: pd.Timestamp) -> float:
    series = tide.set_index("datetime_hkt")["tide_m"].sort_index()
    if event_time_hkt < series.index.min() or event_time_hkt > series.index.max():
        return float("nan")
    expanded = series.reindex(series.index.union(pd.DatetimeIndex([event_time_hkt]))).sort_index()
    return float(expanded.interpolate(method="time").loc[event_time_hkt])


def add_tides(events_path: Path, out_path: Path, station: str, hour_mode: str) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    if "time_at_min_dist" not in events.columns:
        raise ValueError(f"{events_path} does not contain time_at_min_dist")

    tide_cache: dict[int, pd.DataFrame] = {}
    out_rows = []
    for _, row in events.iterrows():
        event_utc = pd.to_datetime(row["time_at_min_dist"], utc=True)
        event_hkt = event_utc.tz_convert("Asia/Hong_Kong").tz_localize(None)
        year = int(event_hkt.year)

        if year not in tide_cache:
            tide_cache[year] = load_hko_hhot_year(year, station, hour_mode)

        tide = tide_cache[year]
        nearest_m, matched_hkt, diff_hours, hko_col = nearest_tide(tide, event_hkt)
        interp_m = interpolated_tide(tide, event_hkt)

        new_row = row.to_dict()
        new_row.update(
            {
                "event_time_utc": event_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "event_time_hkt": event_hkt.strftime("%Y-%m-%d %H:%M:%S"),
                "hko_station": station,
                "hko_hour_mode": hour_mode,
                "hko_matched_time_hkt": matched_hkt.strftime("%Y-%m-%d %H:%M:%S"),
                "hko_hour_col": hko_col,
                "hko_time_diff_hours": round(diff_hours, 3),
                "astronomical_tide_nearest_m": round(nearest_m, 3),
                "astronomical_tide_interp_m": round(interp_m, 3) if pd.notna(interp_m) else "",
            }
        )
        out_rows.append(new_row)

    out = pd.DataFrame(out_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add HKO predicted astronomical tide levels to selected IBTrACS PRD storm events."
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--station", default="SPW", help="HKO tide station code. Default: SPW = Shek Pik.")
    parser.add_argument(
        "--hour-mode",
        choices=["zero_based", "one_based"],
        default="zero_based",
        help="Map HKO HHOT columns. zero_based: 01=00:00, 24=23:00. one_based: 01=01:00, 24=next-day 00:00.",
    )
    args = parser.parse_args()

    out = add_tides(args.events, args.out, args.station, args.hour_mode)
    print(f"wrote: {args.out}")
    cols = [
        "SEASON",
        "NAME",
        "time_at_min_dist",
        "event_time_hkt",
        "hko_matched_time_hkt",
        "astronomical_tide_nearest_m",
        "astronomical_tide_interp_m",
    ]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()

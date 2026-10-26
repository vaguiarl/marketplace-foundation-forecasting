"""Prepare an auditable, small Uber rider-fare panel from official NYC TLC files.

Uses only January outcomes for route selection. Never labels the observed fare
as a surge multiplier, a no-surge fare, or a binding advance price quote.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
MONTHS = ("2026-01", "2026-02", "2026-03")
SOURCE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
DICTIONARY = "https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
COLUMNS = ["hvfhs_license_num", "pickup_datetime", "dropoff_datetime",
           "PULocationID", "DOLocationID", "base_passenger_fare", "trip_miles",
           "trip_time", "shared_request_flag", "shared_match_flag", "access_a_ride_flag"]
MIN_COUNT = 10
AIRPORTS = {1, 132, 138}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download(url, path):
    """curl supplies retries, range resume, TLS validation and HTTP failure checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--retry-delay", "2",
                        "--connect-timeout", "30", "--continue-at", "-", "--silent", "--show-error",
                        "--output", str(partial), url], check=True)
        partial.replace(path)
    if path.suffix == ".parquet":
        pq.ParquetFile(path)  # Refuse corrupt/incomplete cached downloads.
    print(f"Ready {path.name}: {path.stat().st_size / 1e6:.1f} MB", flush=True)
    return {"url": url, "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def clean_batch(frame, month, audit):
    """Sequential exclusion counts are disjoint and sum with retained to input."""
    audit["rows_read"] += len(frame)

    def keep(mask, label):
        nonlocal frame
        mask = mask.fillna(False)
        audit[label] += int((~mask).sum())
        frame = frame.loc[mask].copy()

    keep(frame.hvfhs_license_num.eq("HV0003"), "exclude_other_or_missing_provider")
    start = pd.Timestamp(month)
    stop = start + pd.offsets.MonthBegin()
    keep(frame.pickup_datetime.ge(start) & frame.pickup_datetime.lt(stop)
         & frame.dropoff_datetime.notna(), "exclude_missing_or_out_of_month_datetime")
    # TLC timestamps have no offset. Reject non-existent/ambiguous source clock
    # times; do not invent a repeated or missing local hour. Forecast grid is UTC.
    for column in ("pickup_datetime", "dropoff_datetime"):
        frame[column] = frame[column].dt.tz_localize("America/New_York", nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
    keep(frame.pickup_datetime.notna() & frame.dropoff_datetime.notna(), "exclude_invalid_dst_clock_time")
    duration = (frame.dropoff_datetime - frame.pickup_datetime).dt.total_seconds()
    keep(duration.gt(0) & duration.le(10800)
         & frame.trip_time.gt(0) & frame.trip_time.le(10800), "exclude_invalid_or_over_3h_duration")
    keep(frame.base_passenger_fare.gt(0) & frame.base_passenger_fare.le(500), "exclude_fare_outside_0_500")
    keep(frame.trip_miles.ge(0.1) & frame.trip_miles.le(100), "exclude_miles_outside_0_1_100")
    keep(frame.shared_request_flag.eq("N") & frame.shared_match_flag.eq("N"), "exclude_shared_or_unknown_flags")
    keep(frame.access_a_ride_flag.eq("N"), "exclude_access_a_ride_or_unknown")
    keep(frame.PULocationID.between(1, 263) & frame.DOLocationID.between(1, 263), "exclude_unknown_zone")
    keep(~frame.PULocationID.isin(AIRPORTS) & ~frame.DOLocationID.isin(AIRPORTS), "exclude_airport")
    keep(frame.PULocationID.ne(frame.DOLocationID), "exclude_same_zone")
    audit["rows_retained"] += len(frame)
    frame["timestamp"] = frame.pickup_datetime.dt.floor("h")
    return frame


def batches(path, month, audit):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250_000, columns=COLUMNS,
                                                 use_threads=False):
        yield clean_batch(batch.to_pandas(), month, audit)


def select_routes(january_path, zones, count=6):
    audit = Counter()
    parts = []
    for frame in batches(january_path, "2026-01", audit):
        parts.append(frame.groupby(["PULocationID", "DOLocationID", "timestamp"],
                                   observed=True).size().rename("count").reset_index())
    hourly = pd.concat(parts, ignore_index=True).groupby(
        ["PULocationID", "DOLocationID", "timestamp"], observed=True)["count"].sum().reset_index()
    stats = hourly.groupby(["PULocationID", "DOLocationID"], observed=True).agg(
        train_trips=("count", "sum"), train_observed_hours=("count", "size"),
        train_valid_hours=("count", lambda x: int(x.ge(MIN_COUNT).sum())),
        train_15plus_hours=("count", lambda x: int(x.ge(15).sum()))).reset_index()
    stats["train_coverage"] = stats.train_valid_hours / 744
    stats["train_mean_hourly_count"] = stats.train_trips / 744
    borough = zones.set_index("LocationID").Borough
    manhattan = stats.PULocationID.map(borough).eq("Manhattan") & stats.DOLocationID.map(borough).eq("Manhattan")
    preferred = stats.loc[manhattan & stats.train_coverage.ge(.60) & stats.train_mean_hourly_count.ge(15)]
    if len(preferred) >= count:
        pool, rule = preferred, "Manhattan-to-Manhattan, >=60% hours with >=10 rides, >=15 rides/hour January mean"
    else:
        pool = stats.loc[stats.train_coverage.ge(.60) & stats.train_mean_hourly_count.ge(15)]
        rule = "All nonairport cross-zone NYC routes; Manhattan pool had fewer than six qualifying routes; >=60% hours with >=10 rides and >=15 rides/hour January mean"
        if len(pool) < count:
            pool = stats
            rule += "; fewer than six passed coverage/volume screen, fallback rank all by valid hours then volume"
    selected = pool.sort_values(["train_valid_hours", "train_trips", "PULocationID", "DOLocationID"],
                               ascending=[False, False, True, True]).head(count).copy()
    selected["series_id"] = selected.PULocationID.astype(str) + "_" + selected.DOLocationID.astype(str)
    labels = zones.set_index("LocationID").Zone
    selected["origin"] = selected.PULocationID.map(labels)
    selected["destination"] = selected.DOLocationID.map(labels)
    return selected, rule, dict(audit)


def aggregate_selected(frames, selected):
    """True trip-level medians, not medians of batch medians."""
    trip = pd.concat(frames, ignore_index=True)
    ag = trip.groupby(["PULocationID", "DOLocationID", "timestamp"], observed=True).agg(
        fare=("base_passenger_fare", "median"), count=("base_passenger_fare", "size"),
        median_miles=("trip_miles", "median"), median_duration_min=("duration_min", "median")).reset_index()
    ag["series_id"] = ag.PULocationID.astype(str) + "_" + ag.DOLocationID.astype(str)
    grid = pd.MultiIndex.from_product([selected.series_id,
        pd.date_range("2026-01-01 05:00", "2026-04-01 04:00", freq="h", inclusive="left")],
        names=["series_id", "timestamp"])
    panel = ag.set_index(["series_id", "timestamp"])[["fare", "count", "median_miles", "median_duration_min"]].reindex(grid)
    panel["count"] = panel["count"].fillna(0).astype("int32")
    panel.loc[panel["count"] < MIN_COUNT, "fare"] = np.nan
    panel = panel.reset_index()
    local = panel.timestamp.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    panel["local_hour"] = local.dt.hour.astype("int8")
    panel["local_dayofweek"] = local.dt.dayofweek.astype("int8")
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=6)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    t0 = time.monotonic()
    raw = ROOT / "data/raw"
    urls = [(f"https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{m}.parquet",
             raw / f"fhvhv_tripdata_{m}.parquet") for m in MONTHS]
    urls += [(ZONE_URL, raw / "taxi_zone_lookup.csv")]
    with ThreadPoolExecutor(max_workers=3) as executor:
        sources = list(executor.map(lambda x: download(*x), urls))
    if args.download_only:
        return
    zones = pd.read_csv(raw / "taxi_zone_lookup.csv")
    selected, rule, selection_audit = select_routes(raw / "fhvhv_tripdata_2026-01.parquet", zones, args.routes)
    print("Selected January-only routes:", selected.to_dict("records"), flush=True)
    # Publish the selected route manifest before scanning held-out months.
    selected.to_json(ROOT / "data/routes.json", orient="records", indent=2)
    wanted = set(zip(selected.PULocationID, selected.DOLocationID))
    pieces, audits = [], {}
    for month in MONTHS:
        audit = Counter()
        for frame in batches(raw / f"fhvhv_tripdata_{month}.parquet", month, audit):
            keep = pd.MultiIndex.from_frame(frame[["PULocationID", "DOLocationID"]]).isin(wanted)
            small = frame.loc[keep, ["PULocationID", "DOLocationID", "timestamp", "base_passenger_fare", "trip_miles"]].copy()
            small["duration_min"] = (frame.loc[keep, "dropoff_datetime"] - frame.loc[keep, "pickup_datetime"]).dt.total_seconds() / 60
            pieces.append(small)
            audit["selected_route_trips"] += len(small)
        audits[month] = dict(audit)
        print(f"Processed {month}: {audit['selected_route_trips']:,} selected trips", flush=True)
    panel = aggregate_selected(pieces, selected)
    path = ROOT / "data/panel.parquet"
    panel.to_parquet(path, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_landing_page": SOURCE, "official_data_dictionary": DICTIONARY,
        "sources": sources, "panel_sha256": sha256(path), "panel_rows": len(panel),
        "route_selection_period": "2026-01-01 through 2026-01-31 NYC local time only",
        "route_selection_rule": rule,
        "route_ranking": "descending January hours with at least 10 selected rides, then total rides; zone IDs break ties",
        "routes": selected.to_dict("records"), "selection_pass_audit": selection_audit,
        "monthly_disjoint_exclusion_audits": audits,
        "minimum_rides_per_target_hour": MIN_COUNT,
        "target": "Median realized base_passenger_fare in USD on completed, nonshared, nonairport, cross-zone Uber trips with specified validity filters",
        "time_convention": "UTC timezone-naive pickup hour, complete 2159-hour grid Jan1 05:00Z through Apr1 04:00Z (end exclusive). Source NYC local timestamps localized with ambiguous/nonexistent=NaT, invalid records dropped, then converted to UTC. Local_hour and local_dayofweek derive America/New_York calendar and respect DST.",
        "availability": "Retrospective pseudo-online experiment: TLC public monthly releases are delayed approximately two months. To mimic an internal completed-trip stream, callers must embargo each hour until its end plus three hours, because retained rides can last up to three hours. No real-time public feed is claimed.",
        "limitations": ["No surge multiplier or no-surge counterfactual", "No Uber One membership, discounts or product-tier identifier", "Only completed trips; no quotes, declined requests, conversion or unconstrained demand", "Composition within OD can change; this is observed rider fare, not causal surge", "TLC does not guarantee accuracy or completeness", "No driver compensation fields used", "No deduplication attempted: absent trip IDs and identical values may be legitimate separate trips"],
        "elapsed_seconds": time.monotonic()-t0,
    }
    (ROOT / "data/metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(panel):,} panel rows; {panel.fare.notna().sum():,} observed target hours in {time.monotonic()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

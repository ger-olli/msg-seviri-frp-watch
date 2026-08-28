import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urljoin

import h5py
import numpy as np
import requests
from bs4 import BeautifulSoup
from shapely.geometry import Point, Polygon

POLYGON = Polygon([
    (21.30252, 44.83812),
    (21.21291, 44.79014),
    (20.99648, 44.89789),
    (21.10188, 44.96886),
])

USERNAME = os.environ.get("LSASAF_USERNAME")
PASSWORD = os.environ.get("LSASAF_PASSWORD")
if not USERNAME or not PASSWORD:
    print("ERROR: LSASAF_USERNAME/LSASAF_PASSWORD not set", file=sys.stderr)
    sys.exit(2)

BASE = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5"
STATUS_PATH = Path("data/msg_status.json")
CURSOR_PATH = Path("data/msg_cursor.json")
SEEN_PATH = Path("data/msg_seen.json")
EVENTS_PATH = Path("data/msg_events.jsonl")
DOWNLOAD_DIR = Path("data/msg_downloads")

PATTERN = re.compile(r"^HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_MSG-Disk_(\d{12})$")

session = requests.Session()
session.auth = (USERNAME, PASSWORD)
session.headers.update({"User-Agent": "msg-seviri-frp-watch/1.0"})


def get(url, timeout=60):
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def listproducts_for_day(day):
    url = f"{BASE}/{day:%Y/%m/%d}/"
    soup = BeautifulSoup(get(url).text, "html.parser")
    found = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        name = href.rstrip("/").split("/")[-1]
        m = PATTERN.match(name)
        if m:
            found.append((m.group(1), name, urljoin(url, href)))
    return sorted(found)


def all_recent_products():
    now = datetime.now(timezone.utc)
    rows = []
    errors = []
    for delta in (2, 1, 0):
        day = now - timedelta(days=delta)
        try:
            rows.extend(listproducts_for_day(day))
        except Exception as exc:
            errors.append({"day": day.strftime("%Y-%m-%d"), "error": str(exc)})
    dedup = {name: (stamp, name, url) for stamp, name, url in rows}
    return sorted(dedup.values()), errors


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [decode_attr(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def physical_values(ds):
    raw = np.asarray(ds[...])
    attrs = ds.attrs
    missing = attrs.get("MISSING_VALUE", None)
    scale = float(attrs.get("SCALING_FACTOR", 1.0))
    offset = float(attrs.get("OFFSET", 0.0))
    arr = raw.astype(float)
    if missing is not None:
        arr[raw == np.asarray(missing).reshape(-1)[0]] = np.nan
    if scale == 0:
        raise RuntimeError(f"Invalid SCALING_FACTOR=0 for {ds.name}")
    # LSA SAF HDF5 convention used by these integer-packed fields:
    # physical value = raw / SCALING_FACTOR + OFFSET.
    return arr / scale + offset


def parse_measurement_time(h5, filename):
    for attr in ("IMAGE_ACQUISITION_TIME", "SENSING_START_TIME"):
        value = decode_attr(h5.attrs.get(attr, ""))
        if isinstance(value, str) and re.fullmatch(r"\d{14}", value):
            return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat()
    m = PATTERN.match(filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc).isoformat()
    return None


def extract_hotspots(path, filename):
    with h5py.File(path, "r") as h5:
        required = ["LATITUDE", "LONGITUDE", "FRP", "FRP_UNCERTAINTY", "FIRE_CONFIDENCE", "PIXEL_SIZE"]
        missing = [name for name in required if name not in h5]
        if missing:
            raise RuntimeError("Missing required datasets: " + ", ".join(missing))

        lat = physical_values(h5["LATITUDE"])
        lon = physical_values(h5["LONGITUDE"])
        frp = physical_values(h5["FRP"])
        unc = physical_values(h5["FRP_UNCERTAINTY"])
        conf = physical_values(h5["FIRE_CONFIDENCE"])
        pix = physical_values(h5["PIXEL_SIZE"])

        n = len(frp)
        if not all(len(x) == n for x in (lat, lon, unc, conf, pix)):
            raise RuntimeError("LSA-502 dataset lengths do not match")

        measurement_time = parse_measurement_time(h5, filename)
        satellite_attr = decode_attr(h5.attrs.get("SATELLITE", []))
        instrument_attr = decode_attr(h5.attrs.get("INSTRUMENT_ID", []))
        overall_quality = decode_attr(h5.attrs.get("OVERALL_QUALITY_FLAG", None))
        nominal_product_time = decode_attr(h5.attrs.get("NOMINAL_PRODUCT_TIME", None))

        hits = []
        for i in range(n):
            if not (np.isfinite(lat[i]) and np.isfinite(lon[i]) and np.isfinite(frp[i])):
                continue
            point = Point(float(lon[i]), float(lat[i]))
            if not (POLYGON.contains(point) or POLYGON.touches(point)):
                continue
            hits.append({
                "source": "LSA SAF MSG/SEVIRI FRP-PIXEL",
                "product": "LSA-502",
                "satellite": satellite_attr,
                "instrument": instrument_attr,
                "measurement_time_utc": measurement_time,
                "latitude": float(lat[i]),
                "longitude": float(lon[i]),
                "frp_mw": float(frp[i]),
                "frp_uncertainty_mw": None if not np.isfinite(unc[i]) else float(unc[i]),
                "fire_confidence_percent": None if not np.isfinite(conf[i]) else float(conf[i]),
                "pixel_size_km": None if not np.isfinite(pix[i]) else float(pix[i]),
                "overall_quality_flag": overall_quality,
                "nominal_product_time": nominal_product_time,
                "product_file": filename,
            })

        metadata = {
            "record_count_full_msg_disk": n,
            "measurement_time_utc": measurement_time,
            "satellite": satellite_attr,
            "instrument": instrument_attr,
            "overall_quality_flag": overall_quality,
            "nominal_product_time": nominal_product_time,
            "dataset_mapping": {
                "latitude_dataset": "LATITUDE",
                "longitude_dataset": "LONGITUDE",
                "frp_dataset": "FRP",
                "frp_uncertainty_dataset": "FRP_UNCERTAINTY",
                "confidence_dataset": "FIRE_CONFIDENCE",
                "pixel_size_dataset": "PIXEL_SIZE",
            },
        }
        return hits, metadata


def main():
    checked = datetime.now(timezone.utc).isoformat()
    cursor = load_json(CURSOR_PATH, {"last_processed_file": None})
    seen = set(load_json(SEEN_PATH, []))

    status = {
        "checked_at_utc": checked,
        "source": "LSA SAF MSG/SEVIRI FRP-PIXEL",
        "product": "LSA-502",
        "polygon": list(POLYGON.exterior.coords),
        "processed_files": [],
        "new_hotspots": [],
        "new_hotspot_count": 0,
        "errors": [],
    }

    try:
        products, listing_errors = all_recent_products()
        if listing_errors:
            status["listing_warnings"] = listing_errors
        if not products:
            raise RuntimeError("No LSA-502 ListProduct found in the last 3 UTC days")

        status["files_available"] = len(products)
        last = cursor.get("last_processed_file")
        status["last_processed_file_before_run"] = last

        if last:
            pending = [(stamp, fn, url) for stamp, fn, url in products if fn > last]
        else:
            pending = products[-1:]

        # Limit one run to a bounded catch-up while preserving order.
        pending = pending[:12]
        status["pending_file_count"] = len(pending)
        last_success = last
        new_hits = []

        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for stamp, filename, url in pending:
            row = {"product_file": filename, "product_url": url}
            try:
                r = get(url, timeout=180)
                path = DOWNLOAD_DIR / filename
                path.write_bytes(r.content)
                hits, metadata = extract_hotspots(path, filename)
                row.update(metadata)
                row["download_bytes"] = len(r.content)
                row["inside_polygon"] = len(hits)

                for hit in hits:
                    key = "|".join([
                        filename,
                        f"{hit['latitude']:.5f}",
                        f"{hit['longitude']:.5f}",
                        f"{hit['frp_mw']:.3f}",
                    ])
                    hit["_key"] = key
                    if key not in seen:
                        seen.add(key)
                        new_hits.append(hit)

                last_success = filename
                save_json(CURSOR_PATH, {"last_processed_file": filename})
                try:
                    path.unlink()
                except OSError:
                    pass
            except Exception as exc:
                row["error"] = str(exc)
                status["errors"].append({"product_file": filename, "error": str(exc)})
                status["processed_files"].append(row)
                break
            status["processed_files"].append(row)

        status["last_processed_file_after_run"] = last_success
        status["new_hotspots"] = new_hits
        status["new_hotspot_count"] = len(new_hits)

        if new_hits:
            EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EVENTS_PATH.open("a", encoding="utf-8") as f:
                for hit in new_hits:
                    f.write(json.dumps({"detected_at_utc": checked, **hit}, ensure_ascii=False) + "\n")

    except Exception as exc:
        status["errors"].append({"general": str(exc)})

    save_json(STATUS_PATH, status)
    save_json(SEEN_PATH, sorted(seen))
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

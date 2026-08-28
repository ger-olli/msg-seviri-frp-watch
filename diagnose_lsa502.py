import os
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urljoin

import h5py
import requests
from bs4 import BeautifulSoup

BASE = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5/"
USER = os.environ["LSASAF_USERNAME"]
PASSWORD = os.environ["LSASAF_PASSWORD"]

session = requests.Session()
session.auth = (USER, PASSWORD)
session.headers.update({"User-Agent": "msg-seviri-frp-watch/diagnostic"})


def get(url):
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r


def links(url):
    soup = BeautifulSoup(get(url).text, "html.parser")
    return [a.get("href") for a in soup.find_all("a") if a.get("href")]


def find_latest_listproduct():
    # We deliberately discover the live directory layout instead of assuming it.
    # Search recent date-like subdirectories recursively, with a small depth limit.
    candidates = []
    seen = set()

    def walk(url, depth):
        if depth > 5 or url in seen:
            return
        seen.add(url)
        try:
            hrefs = links(url)
        except Exception as exc:
            print(f"Cannot list {url}: {exc}", file=sys.stderr)
            return
        for href in hrefs:
            if href in ("../", "./") or href.startswith("?"):
                continue
            full = urljoin(url, href)
            name = href.rstrip("/").split("/")[-1]
            if href.endswith("/"):
                # Follow only plausible archive/date directories.
                if re.fullmatch(r"(?:20\d{2}|\d{2}|\d{4}|\d{6}|\d{8})", name) or depth == 0:
                    walk(full, depth + 1)
            else:
                low = name.lower()
                if ("listproduct" in low or "list_product" in low) and (low.endswith(".h5") or "hdf5" in low):
                    candidates.append((name, full))

    walk(BASE, 0)
    if not candidates:
        raise RuntimeError("No LSA-502 ListProduct file discovered from the live server path")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def describe_hdf5(path):
    print("\n=== HDF5 STRUCTURE ===")
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET {name} shape={obj.shape} dtype={obj.dtype}")
                for k, v in obj.attrs.items():
                    print(f"  ATTR {k}={v}")
            elif isinstance(obj, h5py.Group):
                print(f"GROUP {name or '/'}")
        f.visititems(visitor)
        print("\n=== ROOT ATTRIBUTES ===")
        for k, v in f.attrs.items():
            print(f"ATTR {k}={v}")


name, url = find_latest_listproduct()
print("Checked UTC:", datetime.now(timezone.utc).isoformat())
print("Discovered file:", name)
print("Source URL:", url)

r = get(url)
local = "/tmp/lsa502_latest.h5"
with open(local, "wb") as fh:
    fh.write(r.content)
print("Downloaded bytes:", len(r.content))

describe_hdf5(local)

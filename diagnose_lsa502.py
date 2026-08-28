import os
import re
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin

import h5py
import requests
from bs4 import BeautifulSoup

BASE = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5"
USER = os.environ["LSASAF_USERNAME"]
PASSWORD = os.environ["LSASAF_PASSWORD"]

session = requests.Session()
session.auth = (USER, PASSWORD)
session.headers.update({"User-Agent": "msg-seviri-frp-watch/diagnostic"})


def get(url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r


def listproducts_for_day(day):
    url = f"{BASE}/{day:%Y/%m/%d}/"
    soup = BeautifulSoup(get(url).text, "html.parser")
    found = []
    pattern = re.compile(r"^HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_MSG-Disk_(\d{12})$")
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        name = href.rstrip("/").split("/")[-1]
        m = pattern.match(name)
        if m:
            found.append((m.group(1), name, urljoin(url, href)))
    return sorted(found)


def find_latest_listproduct():
    # Query only today's UTC directory. If it is not yet available, try yesterday.
    now = datetime.now(timezone.utc)
    for delta in (0, 1):
        day = now - timedelta(days=delta)
        try:
            found = listproducts_for_day(day)
        except requests.RequestException as exc:
            print(f"Cannot list {day:%Y-%m-%d}: {exc}")
            continue
        if found:
            _, name, url = found[-1]
            return name, url
    raise RuntimeError("No LSA-502 ListProduct found for today or yesterday")


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

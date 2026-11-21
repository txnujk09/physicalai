"""Download the real Danish National Biobank ULT freezer data (Huang et al., Sci Data 2023,
doi:10.1038/s41597-023-02808-6) freezer by freezer from the authors' DTU GitLab, reduce each
1-minute file to hourly signals, and delete the raw file.

Output: freezer/data/real_hourly/<id>.parquet  (+ fault_summary_R1.xlsx)
Hourly columns: chamber, suction1, ihx, evap, sump2, ontime (1st-stage compressor on-time per
cycle, minutes, from State bit 0), door_open_min (minutes door open per hour, State bit 4), n (samples).
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

API = "https://lab.compute.dtu.dk/api/v4/projects/taohu%2Fult-freezers-labelled-dataset-sci-data/repository"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COLS = {"RTD": "chamber", "1st Suc.": "suction1", "H.E.": "ihx", "Evap. Out": "evap", "2nd Sump": "sump2"}


def curl(path, dest):
    url = f"{API}/files/{path.replace('/', '%2F')}/raw?ref=main"
    subprocess.run(["curl", "-sS", "--retry", "3", "-m", "900", "-o", dest, url], check=True)


def ontime_hourly(t, state):
    on = (state & 1).astype(np.int8)
    d = np.diff(on, prepend=on[0])
    starts, ends = t[d == 1], t[d == -1]
    if len(starts) == 0 or len(ends) == 0:
        return None
    j = np.searchsorted(ends, starts, side="right")
    ok = j < len(ends)
    dur = (ends[j[ok]] - starts[ok]) / np.timedelta64(1, "m")
    s = pd.Series(dur, index=pd.DatetimeIndex(ends[j[ok]]))
    s = s[(s > 1) & (s < 600)]
    return s.resample("1h").mean()


def process(fid):
    dst = os.path.join(OUT, "real_hourly", f"{fid}.parquet")
    if os.path.exists(dst):
        return fid, "cached"
    raw = os.path.join(OUT, f"raw_{fid}.parquet")
    try:
        curl(f"data/dataset/Revco/{fid}_temp.parquet", raw)
        df = pd.read_parquet(raw, columns=["Datetime", *COLS, "State"])
        df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
        t = df["Datetime"].values
        H = df.set_index("Datetime")[list(COLS)].rename(columns=COLS).apply(pd.to_numeric, errors="coerce")
        n = H["chamber"].resample("1h").count().rename("n")
        H = H.resample("1h").mean()
        st = df["State"].fillna(0).astype(int).values
        ot = ontime_hourly(t, st)
        H["ontime"] = ot.reindex(H.index) if ot is not None else np.nan
        door = pd.Series(((st >> 4) & 1).astype(float), index=pd.DatetimeIndex(t)).resample("1h").sum()
        H["door_open_min"] = door.reindex(H.index)
        H["n"] = n
        H.to_parquet(dst)
        return fid, f"{len(df):,} rows -> {len(H):,} h, {H.index.min().date()}..{H.index.max().date()}"
    except Exception as e:  # keep going on one bad file
        return fid, f"ERROR {e}"
    finally:
        if os.path.exists(raw):
            os.remove(raw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--ids", nargs="*", default=None)
    args = p.parse_args()
    os.makedirs(os.path.join(OUT, "real_hourly"), exist_ok=True)
    fs = os.path.join(OUT, "fault_summary_R1.xlsx")
    if not os.path.exists(fs):
        curl("data/dataset/fault_summary_R1.xlsx", fs)
    import json
    tree = subprocess.run(["curl", "-sS", f"{API}/tree?path=data/dataset/Revco&per_page=100"],
                          capture_output=True, text=True, check=True).stdout
    ids = args.ids or sorted({f["name"].split("_")[0] for f in json.loads(tree) if f["name"].endswith("_temp.parquet")})
    print(f"{len(ids)} Revco freezers", flush=True)
    with Pool(args.workers) as pool:
        for fid, msg in pool.imap_unordered(process, ids):
            print(fid, msg, flush=True)

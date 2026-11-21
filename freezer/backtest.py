"""Backtest freezer failure-warning methods on REAL Danish biobank ULT freezers.

Methods (none is tuned on the real data):
  sim_model : the simulator-trained MLP from the freezer team's repo (model.json), same
              features and alarm rule (p(healthy) < THR for CONSEC checks every 5 h).
  z_rule    : max |z| of any signal's 30-h mean vs baseline > K_Z for CONSEC checks
              (K_Z = 6.05, the team's threshold calibrated on simulated healthy freezers).
  minus70   : conventional alarm, hourly chamber mean above -70 C.

Honest scoring: every method runs on identical 47-day windows.
  fault windows   : [service date - 45 d, service date + 2 d] for each labelled fault
  healthy windows : 47-day tiles of the same freezers, >= 60 d before / 14 d after any label
The question is whether a method alarms (before the service date) in fault windows
MORE OFTEN than it alarms in healthy windows. A method that alarms everywhere "detects"
every fault and is useless.
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SIGS = ["chamber", "ontime", "suction1", "ihx", "evap", "sump2", "door"]
K_Z = 6.05
PRE_D, POST_D = 45, 2
TARGET = {"Refrigerant leakage": "leak", "1st stage compressor": "comp1", "1st stage compressor (No fix)": "comp1",
          "2nd stage compressor": "comp2", "Door tightness": "door", "Bad insulation": "door"}


def load_model(path):
    M = json.load(open(path))
    for k in ("cls", "rul"):
        n = M[k]
        n["W1m"] = np.array(n["W1"]).reshape(n["nin"], n["nh"])
        n["W2m"] = np.array(n["W2"]).reshape(n["nh"], n["nout"])
    M["mu"], M["sd"] = np.array(M["mu"]), np.array(M["sd"])
    return M


def mlp(net, X):
    return np.tanh(X @ net["W1m"] + np.array(net["b1"])) @ net["W2m"] + np.array(net["b2"])


def sl(x):
    return np.sign(x) * np.log1p(np.abs(x))


def window_features(seg, M):
    """Vectorised mirror of features() in the team's index.html / real_freezer_backtest.py."""
    BN, W, CE = M["BASE_N"], M["WIN"], M["CHECK_EVERY"]
    Y = seg[SIGS].interpolate(limit_direction="both").values  # (T, 7)
    base = seg.iloc[:BN]
    mu = base[SIGS].mean().values
    sd = np.maximum(base[SIGS].std().values, [0.5 if s == "ontime" else 0.05 for s in SIGS])
    idx = np.arange(BN + W, len(seg), CE)
    win = np.stack([Y[i - W:i] for i in idx])  # (C, W, 7)
    xs = np.arange(W) - (W - 1) / 2
    m = win.mean(1)
    slope = (win * xs[None, :, None]).sum(1) / (xs ** 2).sum()
    sdw = win.std(1, ddof=1)
    F = np.stack([sl((m - mu) / sd), sl(slope * W / sd), sl(sdw / sd - 1)], axis=2)  # (C, 7, 3)
    dead = ~np.isfinite(mu) | seg[SIGS].isna().all().values
    F[:, dead, :] = 0
    F = np.nan_to_num(F.reshape(len(idx), -1))
    z = np.abs((m - mu) / sd)
    z[:, dead] = 0
    return idx, F, np.nan_to_num(z).max(1)


def first_streak(flags, consec):
    s = 0
    for k, f in enumerate(flags):
        s = s + 1 if f else 0
        if s >= consec:
            return k
    return None


def run_window(seg, M):
    idx, F, zmax = window_features(seg, M)
    X = np.clip((F - M["mu"]) / M["sd"], -5, 5)
    o = mlp(M["cls"], X)
    p = np.exp(o - o.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    out = {}
    k = first_streak(p[:, 0] < M["THR"], M["CONSEC"])
    if k is None:
        out["sim_model"] = None
    else:
        votes = p[max(0, k - M["CONSEC"] + 1):k + 1, 1:].sum(0)
        out["sim_model"] = (seg.index[idx[k]], M["classes"][1 + int(np.argmax(votes))])
    k = first_streak(zmax > K_Z, M["CONSEC"])
    out["z_rule"] = None if k is None else (seg.index[idx[k]], None)
    warm = seg["chamber"].iloc[M["BASE_N"]:] > -70
    out["minus70"] = (warm.idxmax(), None) if warm.any() else None
    return out


def usable(seg, M):
    if len(seg) < M["BASE_N"] + M["WIN"] + 50:
        return False
    c = seg["chamber"]
    return c.notna().mean() > 0.8 and c.iloc[:M["BASE_N"]].median() < -70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "model_sim.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_real_backtest.json"))
    args = ap.parse_args()
    M = load_model(args.model)
    faults = pd.read_excel(os.path.join(DATA, "fault_summary_R1.xlsx"), sheet_name=0)
    faults["date"] = pd.to_datetime(faults["registered_date"]) + pd.Timedelta(hours=12)
    files = sorted(glob.glob(os.path.join(DATA, "real_hourly", "*.parquet")))
    methods = ["sim_model", "z_rule", "minus70"]
    fault_rows, healthy_rows = [], []
    for f in files:
        fid = int(os.path.basename(f).split(".")[0])
        H = pd.read_parquet(f)
        H["door"] = np.nan  # door pull-down time is not in the real data (model input held flat, as in the team's script)
        H = H.asfreq("1h")
        lab = faults[faults.freezer_id == fid]
        for _, r in lab.iterrows():
            d = r["date"]
            seg = H.loc[d - pd.Timedelta(days=PRE_D):d + pd.Timedelta(days=POST_D)]
            row = {"freezer": fid, "date": str(d.date()), "category": r["Fault categories"],
                   "target": TARGET.get(r["Fault categories"]), "usable": bool(usable(seg, M))}
            if row["usable"]:
                res = run_window(seg, M)
                for m in methods:
                    a = res[m]
                    row[m] = None if a is None else {"days_before": round((d - a[0]).total_seconds() / 86400, 1),
                                                     "named": a[1]}
            fault_rows.append(row)
        # healthy tiles
        t = H.index.min()
        bad = [(d - pd.Timedelta(days=60), d + pd.Timedelta(days=14)) for d in lab["date"]]
        L = pd.Timedelta(days=PRE_D + POST_D)
        while t + L <= H.index.max():
            if not any(t <= b1 and t + L >= b0 for b0, b1 in bad):
                seg = H.loc[t:t + L]
                if usable(seg, M):
                    res = run_window(seg, M)
                    healthy_rows.append({"freezer": fid, "start": str(t.date()),
                                         **{m: res[m] is not None for m in methods}})
            t += L
        print(f"{fid}: {len(lab)} labels, healthy windows so far {len(healthy_rows)}", flush=True)

    fr = pd.DataFrame(fault_rows)
    hr = pd.DataFrame(healthy_rows)
    tgt = fr[fr.usable & fr.target.notna()]
    summary = {"n_freezers": len(files), "n_labelled_events": len(fr), "n_target_usable": len(tgt),
               "n_healthy_windows": len(hr), "methods": {}}
    print(f"\n{len(files)} freezers | {len(fr)} labelled events, {len(tgt)} usable target faults "
          f"(leak / compressor / door-insulation) | {len(hr)} healthy 47-day windows\n")
    print(f"{'method':10s} {'alarm before service date':>26s} {'false alarm (healthy)':>22s} {'median lead (d)':>16s}")
    for m in methods:
        hit = tgt[m].apply(lambda a: a is not None and a["days_before"] >= 0)
        leads = [a["days_before"] for a in tgt[m] if a is not None and a["days_before"] >= 0]
        fa = hr[m].mean() if len(hr) else float("nan")
        summary["methods"][m] = {"hit_rate": float(hit.mean()), "hits": int(hit.sum()), "false_alarm_rate": float(fa),
                                 "median_lead_days": float(np.median(leads)) if leads else None}
        print(f"{m:10s} {hit.mean():20.0%} ({hit.sum()}/{len(tgt)}) {fa:15.0%} ({int(hr[m].sum())}/{len(hr)}) "
              f"{(np.median(leads) if leads else float('nan')):12.1f}")
    print("\nper fault (target categories):")
    for _, r in tgt.iterrows():
        cell = lambda a: "-" if a is None else f"{a['days_before']:+.0f}d" + (f" {a['named']}" if a.get("named") else "")
        print(f"  {r.freezer} {r.date} {r.target:5s} sim_model {cell(r.sim_model):18s} z_rule {cell(r.z_rule):8s} -70C {cell(r.minus70)}")
    json.dump({"summary": summary, "faults": fault_rows, "healthy": healthy_rows}, open(args.out, "w"), indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

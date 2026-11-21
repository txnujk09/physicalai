"""Improvement: train the fault detector on SIMULATED faults + REAL healthy operation, test on
REAL labelled faults, with freezer-level cross-validation. No real fault label is ever used
for training or for choosing a threshold.

  positives : simulated freezers, checks within 10 days before failure (team's simulator data)
  negatives : simulated healthy checks + real healthy checks from the TRAINING freezers
  threshold : set so 10% of the TRAINING freezers' healthy 47-day windows would alarm
  test      : held-out freezers' real labelled faults (alarm before the service date?) and
              their healthy windows (false alarm?)

Every method (team's sim-only model, z-rule, -70 C alarm, hybrid) is scored the same way:
a window score = max over checks of the score's 6-check rolling minimum (the alarm rule), and
a threshold set on training freezers' healthy windows at the same 10% target.
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SIM_COLS = {"chamber_c": "chamber", "compressor_on_min": "ontime", "suction1_c": "suction1",
            "interstage_hx_c": "ihx", "evap_out_c": "evap", "sump2_c": "sump2"}
METHODS = ["minus70", "z_rule", "sim_model", "hybrid"]


def roll_min(x, k):
    if len(x) < k:
        return np.array([-np.inf])
    return np.lib.stride_tricks.sliding_window_view(x, k).min(1)


def sim_checks(M, horizon_h=240):
    d = pd.read_csv(os.path.join(DATA, "ult_freezer_sim.csv.gz"))
    d = d[d.brand == "revco"]
    X, y = [], []
    for fid, g in d.groupby("freezer_id"):
        seg = g.rename(columns=SIM_COLS).set_index(pd.to_datetime(g.hour, unit="h"))
        seg["door"] = np.nan  # not available on the real freezers, so hidden here too
        idx, F, _ = bt.window_features(seg, M)
        act = g.fault_active.values[idx] == 1
        htf = pd.to_numeric(g.hours_to_failure, errors="coerce").values[idx]
        pos = act & (htf <= horizon_h)
        neg = ~act
        X += [F[pos], F[neg]]
        y += [np.ones(pos.sum()), np.zeros(neg.sum())]
    return np.concatenate(X), np.concatenate(y)


def real_windows(M):
    faults = pd.read_excel(os.path.join(DATA, "fault_summary_R1.xlsx"), sheet_name=0)
    faults["date"] = pd.to_datetime(faults["registered_date"]) + pd.Timedelta(hours=12)
    W = []
    for f in sorted(glob.glob(os.path.join(DATA, "real_hourly", "*.parquet"))):
        fid = int(os.path.basename(f).split(".")[0])
        H = pd.read_parquet(f).asfreq("1h")
        H["door"] = np.nan
        lab = faults[faults.freezer_id == fid]

        def add(seg, kind, label=None, cat=None):
            if not bt.usable(seg, M):
                return
            idx, F, z = bt.window_features(seg, M)
            t = seg.index[idx]
            keep = np.ones(len(idx), bool) if label is None else (t <= label)
            ch = seg["chamber"].iloc[M["BASE_N"]:]
            if label is not None:
                ch = ch[ch.index <= label]
            tt = t[keep]
            W.append({"freezer": fid, "kind": kind, "category": cat, "target": bt.TARGET.get(cat),
                      "date": None if label is None else str(label.date()),
                      "F": F[keep], "z": z[keep], "chamber_max": float(np.nanmax(ch.values)) if len(ch) else -99,
                      "t": tt, "ch": ch, "label": label})

        for _, r in lab.iterrows():
            d = r["date"]
            add(H.loc[d - pd.Timedelta(days=bt.PRE_D):d + pd.Timedelta(days=bt.POST_D)], "fault", d, r["Fault categories"])
        bad = [(d - pd.Timedelta(days=60), d + pd.Timedelta(days=14)) for d in lab["date"]]
        L, t = pd.Timedelta(days=bt.PRE_D + bt.POST_D), H.index.min()
        while t + L <= H.index.max():
            if not any(t <= b1 and t + L >= b0 for b0, b1 in bad):
                add(H.loc[t:t + L], "healthy")
            t += L
    return W


def sim_model_p(M, F):
    X = np.clip((F - M["mu"]) / M["sd"], -5, 5)
    o = bt.mlp(M["cls"], X)
    p = np.exp(o - o.max(1, keepdims=True))
    return 1 - p[:, 0] / p.sum(1)


def score_series(w, method, M, clf=None):
    """(times, alarm-rule score) for one window."""
    C = M["CONSEC"]
    if method == "minus70":
        return w["ch"].index, w["ch"].values
    if len(w["F"]) < C:
        return w["t"][:0], np.array([])
    if method == "z_rule":
        s = w["z"]
    elif method == "sim_model":
        s = sim_model_p(M, w["F"])
    else:
        s = clf.predict_proba(w["F"])[:, 1]
    return w["t"][C - 1:], roll_min(s, C)


def window_score(w, method, M, clf=None):
    C = M["CONSEC"]
    if method == "minus70":
        return w["chamber_max"]
    if len(w["F"]) == 0:
        return -np.inf
    if method == "z_rule":
        s = w["z"]
    elif method == "sim_model":
        s = sim_model_p(M, w["F"])
    else:
        s = clf.predict_proba(w["F"])[:, 1]
    return float(roll_min(s, C).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--target-fa", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    M = bt.load_model(os.path.join(HERE, "model_sim.json"))
    Xs, ys = sim_checks(M)
    print(f"sim checks: {int(ys.sum())} fault (<=10 d to failure), {int((1 - ys).sum())} healthy")
    W = real_windows(M)
    fids = sorted({w["freezer"] for w in W})
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(fids)
    fold_of = {f: i % args.folds for i, f in enumerate(perm)}
    for w in W:
        w["fold"] = fold_of[w["freezer"]]
        w["scores"] = {}
    for k in range(args.folds):
        tr = [w for w in W if w["fold"] != k and w["kind"] == "healthy"]
        Xr = np.concatenate([w["F"] for w in tr])
        sub = rng.choice(len(Xr), size=min(len(Xr), 150_000), replace=False)
        X = np.concatenate([Xs, Xr[sub]])
        y = np.concatenate([ys, np.zeros(len(sub))])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                             random_state=args.seed).fit(X, y)
        thr = {}
        for m in METHODS:
            tr_scores = np.array([window_score(w, m, M, clf) for w in tr])
            thr[m] = float(np.quantile(tr_scores, 1 - args.target_fa))
        for w in W:
            if w["fold"] == k:
                for m in METHODS:
                    s = window_score(w, m, M, clf)
                    w["scores"][m] = s
                    w.setdefault("alarm", {})[m] = bool(s > thr[m])
                    if w["kind"] == "fault" and s > thr[m]:
                        ts, v = score_series(w, m, M, clf)
                        first = ts[int(np.argmax(np.nan_to_num(v, nan=-np.inf) > thr[m]))]
                        w.setdefault("days_before", {})[m] = round((w["label"] - first).total_seconds() / 86400, 1)
        print(f"fold {k}: {len(tr)} training healthy windows, thresholds "
              + ", ".join(f"{m}={v:.3g}" for m, v in thr.items()), flush=True)

    tgt = [w for w in W if w["kind"] == "fault" and w["target"]]
    hl = [w for w in W if w["kind"] == "healthy"]
    print(f"\n{len(fids)} real freezers, {len(tgt)} usable labelled target faults, {len(hl)} healthy windows; "
          f"thresholds set on training folds' healthy windows (target {args.target_fa:.0%} false alarm)\n")
    print(f"{'method':10s} {'hit before service':>20s} {'false alarm':>14s} {'AUC':>6s}  (95% bootstrap CI)")
    out = {}
    B = 2000
    for m in METHODS:
        hits = np.array([w["alarm"][m] for w in tgt])
        fas = np.array([w["alarm"][m] for w in hl])
        sp = np.array([w["scores"][m] for w in tgt])
        sn = np.array([w["scores"][m] for w in hl])
        auc = roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn])
        boots = []
        for _ in range(B):
            a, b = rng.choice(sp, len(sp)), rng.choice(sn, len(sn))
            boots.append(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out[m] = {"hit_rate": float(hits.mean()), "hits": int(hits.sum()), "n_faults": len(hits),
                  "false_alarm_rate": float(fas.mean()), "n_healthy": len(fas), "auc": float(auc), "auc_ci95": [float(lo), float(hi)]}
        print(f"{m:10s} {hits.mean():12.0%} ({hits.sum():2d}/{len(hits)}) {fas.mean():9.1%} {auc:8.2f}  ({lo:.2f}-{hi:.2f})")
    print("\nwarning time vs the conventional -70 C alarm, on faults where both alarm (days, + = earlier):")
    for m in METHODS[1:]:
        both = [w for w in tgt if w["alarm"][m] and w["alarm"]["minus70"]]
        diff = [w["days_before"][m] - w["days_before"]["minus70"] for w in both]
        only = sum(1 for w in tgt if w["alarm"][m] and not w["alarm"]["minus70"])
        out[m]["earlier_than_minus70_days"] = diff
        out[m]["alarms_where_minus70_silent"] = only
        if diff:
            print(f"  {m:10s} n={len(diff):2d}  median {np.median(diff):+.1f} d  (range {min(diff):+.1f} to {max(diff):+.1f}); "
                  f"caught {only} fault(s) the -70 C alarm missed")
    print("\nper fault:")
    for w in tgt:
        print(f"  {w['freezer']} {w['date']} {w['target']:5s} " + "  ".join(
            f"{m}:{(str(w['days_before'][m]) + 'd') if w['alarm'][m] else '-':>6s}" for m in METHODS))
    by = {}
    for w in tgt:
        by.setdefault(w["target"], []).append(w)
    print("\nhit rate by fault type: " + "; ".join(
        f"{t} (n={len(v)}): " + ", ".join(f"{m} {np.mean([x['alarm'][m] for x in v]):.0%}" for m in METHODS) for t, v in by.items()))
    json.dump({"settings": vars(args), "results": out,
               "faults": [{k: w.get(k) for k in ("freezer", "date", "category", "target", "fold", "scores", "alarm", "days_before")} for w in tgt]},
              open(os.path.join(HERE, "results_hybrid.json"), "w"), indent=1)
    print("wrote freezer/results_hybrid.json")


if __name__ == "__main__":
    main()

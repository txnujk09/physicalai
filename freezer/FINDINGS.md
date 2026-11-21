# Freezer predictive maintenance: does the simulator-trained model work on real freezers?

We took the freezer team's simulator-trained failure model ([bernardopetochi/null_hackathon_predictive_maintenance](https://github.com/bernardopetochi/null_hackathon_predictive_maintenance), `model.json`) and tested it, for the first time, on the real labelled dataset it was designed for: Huang et al., *Scientific Data* 2023, doi:10.1038/s41597-023-02808-6. That dataset covers 38 Revco −80 °C freezers at the Danish National Biobank, 10 years at 1-minute resolution, with faults taken from service reports.

## Data

- `fetch_real.py` downloads each freezer from the authors' DTU GitLab, reduces it to hourly signals (chamber, 1st-stage suction, interstage HX, evaporator out, 2nd-stage sump, compressor on-time from the state bit) and deletes the raw file. All 38 freezers (~5 GB raw) come down to 78 MB.
- 46 labelled service events. 19 are the fault types the model predicts (refrigerant leak, 1st/2nd-stage compressor, door/insulation) and have usable data around the service date.
- 2,500 healthy 47-day windows (at least 60 days before and 14 days after any service event) to measure false alarms.
- Labels are service-report dates (12:00 on the registered day). The true onset is earlier and unknown.

## Bug found in the team's backtest script

`real_freezer_backtest.py` maps sensors by keyword (`suction`, `interstage`, `evaporator outlet`), but the real columns are `1st Suc.`, `H.E.` and `Evap. Out`. Only the chamber, sump and on-time signals would have been mapped, and the model would have run with the other inputs held at zero. Our `backtest.py` maps the columns explicitly.

## Results

### At the model's own alarm threshold (`backtest.py`)

| method | alarm before service date | false alarm in healthy 47-day windows | median lead |
|---|---|---|---|
| simulator-trained model | 68% (13/19) | 24% (609/2500) | 5.4 d |
| z-score rule (team's sim-calibrated threshold) | 63% (12/19) | 28% (703/2500) | 17.8 d |
| conventional −70 °C chamber alarm | 58% (11/19) | 14% (338/2500) | 6.4 d |

On simulated freezers the team reports 100% caught, ~8.5 days median lead and 0.4% false alarms. On real freezers a 24% false-alarm rate per 47-day window means roughly two false alarms per freezer per year. When it does alarm before a real fault, it names the right fault type 5 times out of 13.

### At a matched 10% false-alarm rate (`hybrid.py`, 5-fold cross-validation by freezer)

Thresholds are set on the training freezers' healthy windows only; scores are on held-out freezers. No real fault label is used to fit anything.

| method | caught before service date | false alarm | AUC (95% bootstrap CI) |
|---|---|---|---|
| conventional −70 °C alarm | 53% (10/19) | 10.4% | 0.80 (0.68-0.91) |
| z-score rule | 47% (9/19) | 10.1% | 0.73 (0.58-0.86) |
| simulator-trained model | 53% (10/19) | 10.1% | 0.74 (0.59-0.88) |
| sim faults + real healthy data (our attempt) | 0% (0/19) | 11.6% | 0.59 (0.48-0.70) |

On the 8 faults where both the sim model and the −70 °C alarm fire, the sim model is a median 0.6 days *later* (range: 1.1 days later to 11.9 days earlier). It caught 2 faults the −70 °C alarm missed.

### Our improvement attempt failed, and why that matters

We trained a classifier on simulated fault data plus real healthy operation, so it would learn real-world noise (door openings, defrosts, setpoint changes) instead of simulated noise. It did worse than everything else. The classifier learned to separate simulator data from real data rather than faulty from healthy. In other words, the simulator's fault signatures do not look like real faults. We did not iterate on it, because with 19 real faults any further tuning would be tuning on the test set.

## Bottom line

- On real freezers, the simulator-trained model is roughly as good as the alarm every −80 °C freezer already has, not better. The headline sim numbers (100% caught, ~8 days' warning) do not transfer.
- The real dataset is small (19 usable target faults), so every number above has wide error bars. A difference of one or two faults moves a rate by 5-10 points.

## What would actually improve it

1. **Calibrate the simulator's fault signatures on real data.** Measure how each signal moves in the weeks before real faults, and set the simulator's fault effects from that. Validate leave-one-freezer-out so no freezer is used for both calibration and testing.
2. **Learn "normal" from the real fleet, not the simulator.** 2,500 healthy windows across 38 freezers is plenty to model real day-to-day variation for anomaly detection.
3. **Use the signals the real data has and the simulator ignores:** mains voltage, 2nd-stage compressor state bit, door events, backup-system injection.
4. **Score the thing a customer buys:** days of warning earlier than the existing −70 °C alarm, at a fixed false-alarm budget per freezer-year. Hit rate in a pre-fault window on its own is misleading.

## Reproduce

```bash
python freezer/fetch_real.py      # ~10 min, downloads + reduces 38 freezers
python freezer/backtest.py        # native-threshold backtest -> freezer/results_real_backtest.json
python freezer/hybrid.py          # matched-false-alarm comparison -> freezer/results_hybrid.json
```

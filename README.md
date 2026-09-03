# LSTM-ABS: personal research notes on learning a brake controller from logged telemetry

This repository is one of my personal research projects into machine learning for
vehicle control inside the BeamNG.drive simulator. It is a working notebook, not a
product: the code, models, and numbers here document an investigation as it happened,
including the dead ends. Nothing in it is intended for, or suitable for, deployment on
a real vehicle.

**The question:** can an anti-lock brake controller be learned purely from telemetry
that already exists on disk? The data is 25 GB of 2 kHz episodes (2,950 simulated
stops) plus 11 GB of 2 kHz logs from my own classical PID-based ABS (2,237 stops). No
simulator in the training loop, no new data collected for this project.

**The answer so far (2026-09-02):** yes, to within about 5% of the teacher on a
60 mph straight-line stop in the simulator, with known weaknesses that are listed
below rather than hidden.

## Scope and limits

- Simulation only (BeamNG.drive 0.39), one car (ETK 800), straight-line stops on dry
  asphalt. No cornering, wet or low-grip claims are made.
- The learned controller imitates logged controllers; it does not discover new
  behaviour. Its ceiling is the best controller in the data.
- Every model here still lets wheels lock below about 10 m/s. That is the open
  problem, not a footnote.
- Results are from a small number of runs per configuration (2 to 6). They are
  consistent run to run, but they are not a statistical study.
- Training seed matters more than most design choices: the same recipe across seeds
  spans about 1.02 to 1.15 g. Single-seed comparisons in the tables below are
  indicative only; the grader band-width sweep and a 5-seed ensemble both landed
  inside that spread.

## Results, 60 mph straight-line stop, standard 2 kHz brake metric

| controller | avg g (mean of n runs) | distance | yaw during stop |
|---|---|---|---|
| DynamicABS (hand-written PID, the teacher) | 1.207 (n=4) | 30.3 m | 1.0 deg |
| **LSTM `band_g_mix_lock_sy`** | **1.143 (n=2)** | **32.0 m** | 1.8 deg |
| LSTM `band_g_mix_b1o1_s2` | 1.147 (n=2) | 31.9 m | 2.0 deg |
| LSTM `band_g_mix` | 1.130 (n=8) | 32.4 m | 3.1 deg |
| LSTM `baseline_dyn` | 1.123 (n=7) | 32.6 m | 4.7 deg |
| locked wheels (control run) | 1.002 (n=1) | 36.6 m | 3.5 deg |

The top three LSTM entries are a statistical tie: seed-to-seed spread on an unchanged
recipe is about 0.06 g, larger than the gap between them.

Full per-run data and every training run's validation error: `results/LSTM_ABS_RESULTS.xlsx`
(generated from the experiment folders); corpus statistics behind the findings:
`results/LSTM_ABS_RESULTS_2026-09-02.xlsx`.

## How it works

- **Inputs (17, what a real ABS ECU could have):** four wheel speeds, IMU x/y/z,
  yaw rate, steering input and angle, front wheel angle, brake and throttle pedal,
  rpm, gear. Vehicle speed, slip, friction and contact data are never inputs; they are
  used offline only to grade episodes.
- **Outputs (4):** per-wheel brake torque as a fraction of that car's maximum.
- **Model:** LSTM(17 -> 64 -> 64) + linear head, 55k parameters, 150 ms context,
  runs at 200 Hz. Exported to a Lua weight module and executed inside the vehicle's
  Lua VM (`mod/lstm_abs/`), parity-checked against PyTorch to 1e-6.
- **Training:** behaviour cloning with per-tick weights ("lenses"). Each lens is a
  different way of deciding which logged actions deserve to be copied.

## What the investigation found, in order

1. **A control test before any tuning.** Hardcoding the output to 1.0 produced a
   1.00 g locked-wheel stop, proving the actuation path. Every earlier bad result was
   the model.
2. **One controller's logs poisoned the corpus.** Ticks that look like brake onset
   (pedal down, no decel, no slip) averaged 7% torque, almost all from an early
   controller that sat on released brakes. The net learned to release at onset, which
   is self-fulfilling in closed loop. Diagnosed with an offline probe that reproduces
   the in-game numbers exactly (`onset_probe.py`).
3. **Forward-window decel grading cannot learn to release.** Releasing lowers the
   next 200 ms of decel even when it is the right move, so every net trained with that
   grade locks its wheels.
4. **Cloning one good controller beats cloning many.** Adding 116 stock-ABS runs to
   1,391 DynamicABS runs with uniform weights flipped the onset back into the release
   trap.
5. **A speed-band grader fixes the mix.** Grade every stop per 5 mph band, rank each
   band against every other stop of the same car, weight ticks by rank. On a pool of
   three controllers it picks the best one per band and beats plain DynamicABS
   cloning. On a single controller it only removes data and hurts.
6. **Tick weights that describe the action beat ones that describe the state.**
   Up-weighting ticks whose measured yaw matched the steering command suppressed wheel
   lock (one variant completed a whole stop with peak slip 0.47 and no locked wheel).
   Up-weighting ticks sitting in the "good" 0.08-0.23 slip band did the opposite: those
   are precisely the moments when the logged controller was holding steady and changing
   torque the least (mean |change| 0.111 per tick, versus 0.127-0.134 outside the band),
   so the net learned to hold rather than modulate.
7. **Remaining gap to the teacher:** most nets still let wheels lock below ~10 m/s,
   where the logs contain almost no fully-locked states to learn from. Candidate next
   steps are narrower grading bands, longer context, an ensemble, or relabeling the
   logs with the teacher's own control law.

## Layout

```
*.py                 cloners, trainer (lenses + band grader), exporter, offline probes, test runners
experiments/<name>/  one folder per trained model: model.pt, meta.json (args, data filter, val error), runs.csv (in-game stops)
mod/lstm_abs/        the game mod: one controller, one jbeam part; weight modules are generated here by the exporter (not in git)
configs/LSTMABS.pc   the single vehicle config; the active model is chosen at runtime, not by config
recorder/            the 2 kHz vehicle-Lua recorder that produced the corpus
results/             generated workbook (build_results_xlsx.py), historical run CSV, raw logs and traces
docs/                investigation log with the full history of findings
recorded_data_*      junctions to the data folders (not in git)
```

Workflow: train (writes `experiments/<name>/`), `python deploy_mod.py --lenses a,b --set-lens a`
(exports the Lua weights and copies the mod into the game), then
`python control_test_drive.py a,b 60 2 mytag` (switches models in the running game through
`controller.setLens`, appends each stop to that experiment's `runs.csv`).
`python build_results_xlsx.py` regenerates the workbook from the experiment folders.

## Reproducing

```
python clone_corpus_clean.py            # 99-col 2 kHz masters -> 29-col parquet
python clone_telemetry_dynabs.py        # DynamicABS telemetry logs -> same schema
python train_lenses_corpus.py --roots <corpus>,<tel> --cache <npz> \
    --lenses band_g --variants dyn_abs,stock_tel,no_abs,no_safety --models etk800 \
    --min-peak-g 0.9 --sensor-src gy2 --zero-cols sensor_z,roll_angle,pitch_angle,gear --tag _mix
python deploy_mod.py --lenses band_g_mix --set-lens band_g_mix
python control_test_drive.py band_g_mix 60 2 mytest
```

`BAND_MPH` and `BAND_OVERLAP` at the top of the trainer set the grader's band width
and how many overlapping windows cover each speed.

## Related

The classical controller used as the teacher (DynamicABS), the brake-test tooling that
produced the metric, and the reinforcement-learning side of the same question live in
separate repositories.

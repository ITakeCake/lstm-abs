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

### What the model sees and does

<!-- OBS:START -->

| # | channel | unit | source in the vehicle | what it is |
|---|---|---|---|---|
| 1 | `ws_fr` | m/s | `wheels.wheelRotators[2].wheelSpeed` | front-right wheel speed |
| 2 | `ws_fl` | m/s | `wheels.wheelRotators[3].wheelSpeed` | front-left wheel speed |
| 3 | `ws_rr` | m/s | `wheels.wheelRotators[0].wheelSpeed` | rear-right wheel speed |
| 4 | `ws_rl` | m/s | `wheels.wheelRotators[1].wheelSpeed` | rear-left wheel speed |
| 5 | `sensor_x` | m/s2 | `ffiSensors.sensorX or gx2` | lateral acceleration |
| 6 | `sensor_y` | m/s2 | `ffiSensors.sensorY or gy2` | longitudinal acceleration (positive under braking) |
| 7 | `sensor_z` | m/s2 | `ffiSensors.sensorZ or gz2` | vertical acceleration (zeroed for telemetry-trained nets) |
| 8 | `yaw_rate` | rad/s | `obj:getYawAngularVelocity()` | body yaw rate |
| 9 | `steer_input` | -1..1 | `electrics.values.steering_input` | driver steering input |
| 10 | `steer_angle` | rad | `electrics.values.steering` | steering column angle |
| 11 | `front_wheel_angle` | rad | `obj:nodeVecPlanarCosRightForward` | actual road-wheel angle from node geometry |
| 12 | `brake_input` | 0..1 | `input.brake` | driver brake pedal (pinned to 1.0 while engaged) |
| 13 | `throttle_input` | 0..1 | `input.throttle` | driver throttle |
| 14 | `roll_angle` | rad | `obj:getRollPitchYaw()` | body roll (zeroed for telemetry-trained nets) |
| 15 | `pitch_angle` | rad | `obj:getRollPitchYaw()` | body pitch (zeroed for telemetry-trained nets) |
| 16 | `rpm` | rev/min | `electrics.values.rpm` | engine speed |
| 17 | `gear` | index | `electrics.values.gear_A` | selected gear (zeroed for telemetry-trained nets) |

**Outputs (4):**

| channel | range | meaning |
|---|---|---|
| `brake_torque_fr/fl/rr/rl` | 0..1 | per-wheel brake torque as a fraction of that car's measured maximum |

**Grade-only columns.** Legal for computing training weights offline, never model inputs:

| column | unit | used for |
|---|---|---|
| `speed_velo / Airspeed` | m/s | true vehicle speed; drives every decel and slip grade |
| `wheelbase` | m | bicycle-model expected yaw |
| `damage` | - | episode rejection |
| `grip_pattern` | label | never used: measured to be fictional |

<!-- OBS:END -->

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

## Every experiment

One folder per trained model under `experiments/<name>/` holding `model.pt`, `meta.json`
(the exact training arguments, data filter and validation error) and `runs.csv` (every
in-game stop it made). This table is generated from those folders by
`build_readme_tables.py`; all figures are 60 mph straight-line stops on the standard
2 kHz metric. Seed-to-seed spread on an unchanged recipe is about 0.06 g, so differences
smaller than that are noise.

<!-- EXPERIMENTS:START -->

| experiment | training data and weighting | runs | avg g | dist (m) | yaw (deg) |
|---|---|---|---|---|---|
| **DynamicABS (reference)** | hand-written PID, not learned | 4 | **1.207** | 30.3 | 1.0 |
| `band_g_mix_b1o1_s2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, seed 2 | 2 | 1.147 | 31.9 | 2.0 |
| `band_g_mix_lock_sy` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, slipband (0.08, 0.23) x1.5, yaw dz 0.015, seed 0 | 2 | 1.143 | 32.0 | 1.8 |
| `band_g_mix_lock_sy_s1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, slipband (0.08, 0.23) x1.5, yaw dz 0.015, seed 1 | 2 | 1.138 | 32.2 | 2.9 |
| `band_g_mix` | dyn_abs,stock_tel,no_abs,no_safety, etk800, peak>=0.9g, band_g, bands 5mph x1, seed 0 | 8 | 1.130 | 32.4 | 3.1 |
| `band_g_mix_keep_s0` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, slipkeep (0.05, 0.25), seed 0 | 2 | 1.126 | 32.5 | 3.5 |
| `baseline_dyn` | telemetry dyn_abs, etk800, peak>=0.9g, baseline, gy2 sensors | 7 | 1.123 | 32.6 | 4.7 |
| `band_gy_mix` | dyn_abs,stock_tel,no_abs,no_safety, etk800, peak>=0.9g, band_gy, bands 5mph x1, seed 0 | 2 | 1.122 | 32.6 | 1.0 |
| `band_g_mix_b1_sy_s2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, lock>0.5 x0.1, slipband (0.08, 0.23) x1.5, yaw dz 0.015, seed 2 | 2 | 1.120 | 32.7 | 2.5 |
| `band_g_mix_b1_sy_s0` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, lock>0.5 x0.1, slipband (0.08, 0.23) x1.5, yaw dz 0.015, seed 0 | 2 | 1.110 | 33.0 | 2.6 |
| `band_g_mix_lock` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, seed 0 | 4 | 1.109 | 33.0 | 3.5 |
| `band_g_mix_lock_s2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, seed 2 | 2 | 1.107 | 33.1 |  |
| `band_g_mix_lock_y` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, yaw dz 0.015, seed 0 | 2 | 1.106 | 33.1 | 1.7 |
| `band_g_mix_keep_s2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, slipkeep (0.05, 0.25), seed 2 | 2 | 1.105 | 33.1 | 3.7 |
| `band_g_mix_b1_y_s0` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, lock>0.5 x0.1, yaw dz 0.015, seed 0 | 2 | 1.102 | 33.2 | 1.5 |
| `band_g_mix_b1o1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, seed 0 | 2 | 1.102 | 33.2 | 1.5 |
| `band_g_mix_b1_y_s1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, lock>0.5 x0.1, yaw dz 0.015, seed 1 | 2 | 1.099 | 33.3 |  |
| `band_g_mix_lock_s1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, seed 1 | 2 | 1.098 | 33.3 | 3.2 |
| `band_g_mix_s2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, seed 2 | 2 | 1.092 | 33.5 | 4.9 |
| `band_g_mix_b2p5o2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 2.5mph x2, seed 0 | 2 | 1.088 | 33.6 | 1.6 |
| `band_g_mix_keepy_s0` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, yaw dz 0.015, slipkeep (0.05, 0.25), seed 0 | 2 | 1.088 | 33.6 | 3.6 |
| `band_g_mix_b1_sy_s1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, lock>0.5 x0.1, slipband (0.08, 0.23) x1.5, yaw dz 0.015, seed 1 | 2 | 1.087 | 33.7 |  |
| `baseline_dyn_lock` | dyn_abs, etk800, lens baseline, lock>0.5 x0.1, seed 0 | 2 | 1.079 | 33.9 | 4.4 |
| `band_g_mix_lock_sy_s2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, slipband (0.08, 0.23) x1.5, yaw dz 0.015, seed 2 | 2 | 1.074 | 34.1 |  |
| `band_g_mix_lock_s` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, slipband (0.08, 0.23) x1.5, seed 0 | 4 | 1.074 | 34.1 | 3.2 |
| `band_g_mix_b1o1_s1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, seed 1 | 2 | 1.070 | 34.2 | 2.1 |
| `baseline_stock800` | corpus stock_abs+stock_full, etk800, peak>=0.9g, baseline | 3 | 1.067 | 34.3 | 2.4 |
| `ens_band_g_mix5` | ensemble: 5 seeds of band_g_mix averaged at runtime in Lua | 2 | 1.064 | 34.4 | 2.2 |
| `band_g_dyn` | telemetry dyn_abs, etk800, peak>=0.9g, band_g, bands 5mph x1 | 2 | 1.064 | 34.4 | 4.2 |
| `band_g_mix_b7p5o3` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 7.5mph x3, seed 0 | 2 | 1.061 | 34.5 | 1.4 |
| `band_g_mix_b10o1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 10.0mph x1, seed 0 | 2 | 1.061 | 34.5 | 2.4 |
| `band_g_mix_b5o2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x2, seed 0 | 2 | 1.058 | 34.6 | 4.1 |
| `band_g_mix_s3` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, seed 3 | 2 | 1.055 | 34.7 | 2.2 |
| `band_gy_dyn` | telemetry dyn_abs, etk800, peak>=0.9g, band_gy, bands 5mph x1 | 2 | 1.051 | 34.9 | 2.6 |
| `band_g_mix_s1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, seed 1 | 2 | 1.043 | 35.1 | 2.9 |
| `band_g_mix_keep_s1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 5.0mph x1, lock>0.5 x0.1, slipkeep (0.05, 0.25), seed 1 | 2 | 1.039 | 35.3 |  |
| `band_g_mix_b1_y_s2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, lock>0.5 x0.1, yaw dz 0.015, seed 2 | 2 | 1.035 | 35.4 | 3.0 |
| `avg_g_window_hg` | corpus minus abs_1f, peak>=0.9g, avg_g_window | 2 | 1.030 | 35.5 | 2.5 |
| `curation` | S1-S5+SAC corpus, top-quartile episodes, curation | 3 | 1.029 | 35.6 | 0.8 |
| `avg_g_window_stock` | corpus stock_abs+stock_full, 4 cars, peak>=0.9g, avg_g_window | 2 | 1.029 | 35.6 |  |
| `band_g_mix_b1o1_s3` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 1.0mph x1, seed 3 | 4 | 1.023 | 35.8 | 3.2 |
| `baseline_hg` | corpus minus abs_1f, peak>=0.9g, baseline | 2 | 1.016 | 36.0 | 3.0 |
| `baseline FORCE_CMD=1.0` | control run: output hardcoded to full torque, no model | 1 | 1.002 | 36.6 | 3.5 |
| `baseline_stocknorpm` | corpus stock_abs+stock_full, 4 cars, peak>=0.9g, baseline, rpm+gear zeroed | 1 | 0.987 | 37.1 | 1.0 |
| `baseline_stock` | corpus stock_abs+stock_full, 4 cars, peak>=0.9g, baseline | 3 | 0.894 | 41.0 | 3.6 |
| `curation_hg` | corpus minus abs_1f, peak>=0.9g, curation | 2 | 0.887 | 41.3 |  |
| `baseline_dynstocknoabs2` | dyn_abs,stock_tel,no_abs,no_safety, etk800, peak>=0.9g, baseline | 2 | 0.823 | 44.5 | 1.9 |
| `baseline_stock16` | corpus stock_abs+stock_full, 4 cars, peak>=0.9g, baseline, 16 epochs | 1 | 0.747 | 49.0 |  |
| `band_g_mix_b20o1` | dyn_abs,stock_tel,no_abs,no_safety, etk800, bands 20.0mph x1, seed 0 | 2 | 0.723 | 62.7 | 2.0 |
| `baseline` | S1-S5+SAC corpus, all 5 variants, baseline (uniform) | 3 | 0.411 | 122.8 | 2.1 |
| `baseline_dynnoimu` | telemetry dyn_abs, etk800, peak>=0.9g, baseline, IMU zeroed | 2 | 0.367 | 99.7 | 4.6 |
| `baseline_dynstock2` | dyn_abs + telemetry stock, etk800, peak>=0.9g, baseline | 2 | 0.249 | 147.1 | 2.8 |
| `peak_g` | S1-S5+SAC corpus, all 5 variants, peak_g | 2 | 0.230 | 159.1 |  |
| `avg_g_window` | S1-S5+SAC corpus, all 5 variants, avg_g_window | 3 | 0.228 | 160.4 | 3.2 |
| `yaw_intent` | S1-S5+SAC corpus, all 5 variants, yaw_intent | 2 | 0.226 | 162.2 |  |
| `peak_g_norpm` | S1-S5+SAC corpus, all 5 variants, peak_g, rpm+gear zeroed | 2 | 0.225 | 163.1 |  |
| `baseline_dynstock` | dyn_abs + corpus stock, etk800, baseline, IMU zeroed (invalid) | 2 | 0.224 | 163.4 | 3.0 |
| `baseline_dynstocknoabs` | dyn_abs + stock + no-ABS, etk800, baseline, IMU zeroed (invalid) | 2 | 0.198 | 184.7 | 3.8 |

Trained but not run in the car: `band_g_mix_s4`, `band_g_mix_s5`, `baseline_norpm`, `combined`.

<!-- EXPERIMENTS:END -->

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
